import numpy as np
import scipy.sparse.linalg as splinalg
from scipy import interpolate
from tqdm import tqdm
import cmasher as cmr
import matplotlib.pyplot as plt
import os, shutil, time
from joblib import Parallel, delayed
from PIL import Image

#------------------------------------------------
os.chdir('/home/alexa/Home/My_Projects/PDE_solvers/CFD')
#------------------------------------------------

#------------------------------------------------
#  One of the most intriguing problems in computer graphics is the
#  simulation of fluid-like behavior. A good fluid solver is of great
#  importance in many different areas. In the special effects industry
#  there is a high demand to convincingly mimic the appearance and
#  behavior of fluids such as smoke, water and fire. Paint programs
#  can also benefit from fluid solvers to emulate traditional techniques
#  such as watercolor and oil paint. Texture synthesis is another possible 
#  application. 


#------------------------------------------------
#  w0(x) --> w1(x) --> w2(x) --> w3(x) --> w4(x)
#    add force   advect    diffuse    project
#------------------------------------------------

H  = 7
N  = 130
dt = 0.1
T  = 7
# t = np.arange(0, T, dt)
KINEMATIC_VISCOSITY = 0.00001

#------------------------------------------------
x,y = np.linspace(0, H, N), np.linspace(0, H, N)
X,Y = np.meshgrid(x, y,
                  indexing='ij')
COORDINATES = np.stack((X,Y), axis=-1)
dx = H / (N - 1)

w0 = np.zeros(X.shape + (2,)) # initial state (Nx, Ny, 2)

#------------------------------------------------

def force(t, X, Y, a):
    # Shape: (Nx, Ny, 2)
    force_field = np.zeros(X.shape + (2,))
    decay = 1
    # circle_pathx, circle_pathy = H/4 * np.cos(t/10) + H/2, -H/4 * np.sin(t/10) + H/2
    # region = ((X > H/2-0.1) & (X < H/2+0.1) &
    #           (Y > H/2-0.1) & (Y < H/2+0.1))
    # b = 0.3
    # region = ((X > circle_pathx-b) & (X < circle_pathx+b) &
    #           (Y > circle_pathy-b) & (Y < circle_pathy+b))

    # r = np.sqrt((X - circle_pathx)**2 + (Y - circle_pathy)**2)
    # region = r < 0.3
    # force_field[region, 0] = decay *  a*np.cos(t)
    # force_field[region, 1] = decay * -a*np.sin(t)

    region = ((X > 1) & (X < 2) &
              (Y > H/2-0.3) & (Y < H/2+0.3))
    
    force_field[region, 0] = -3*np.sin(Y[region])
    force_field[region, 1] = 3*np.cos(X[region])

    # region = ((X > H-0.7) & (X < H) &
    #           (Y > H/2-0.3) & (Y < H/2+0.3))
    
    # force_field[region, 0] = -9
    # force_field[region, 1] = -3 

    # region = ((X > H/2-0.3) & (X < H/2+0.3) &
    #           (Y > 0) & (Y < 0.7))
    
    # force_field[region, 0] = -3
    # force_field[region, 1] = 9

    # region = ((X > H/2-0.3) & (X < H/2+0.3) &
    #           (Y > H-0.7) & (Y < H))
    
    # force_field[region, 0] = 3
    # force_field[region, 1] = -9

    return force_field # (Nx, Ny, 2)

def advect(w1, dt):
    coordinates = COORDINATES
    # RK4
    def interpolation(vf1):
        x_d = interpolate.interpn(points=(x, y), values=w1[..., 0], xi=vf1, 
                                  bounds_error=False, fill_value=None)
        y_d = interpolate.interpn(points=(x, y), values=w1[..., 1], xi=vf1, 
                                  bounds_error=False, fill_value=None)
        return np.stack([x_d, y_d], axis=-1)
    
    k1 = -w1
    k2 = -interpolation(coordinates + dt/2 * k1)
    k3 = -interpolation(coordinates + dt/2 * k2)
    k4 = -interpolation(coordinates + dt * k3)
    
    displacement = dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    backtraced_positions = np.clip(coordinates + displacement, 0, H)
    w2 = interpolation(backtraced_positions)
    
    return w2 # (Nx, Ny, 2)

def Laplacian(vec_field, dx):
    vec_fx, vec_fy = vec_field[..., 0], vec_field[..., 1]
    lap_x, lap_y = np.zeros_like(vec_fx), np.zeros_like(vec_fy)

    lap_x[1:-1, 1:-1] = ( vec_fx[0:-2, 1:-1] +
                          vec_fx[1:-1, 0:-2] +
                          vec_fx[2:  , 1:-1] +
                          vec_fx[1:-1, 2:  ] -
                          4*vec_fx[1:-1, 1:-1])/(dx**2)
                          
    lap_y[1:-1, 1:-1] = ( vec_fy[0:-2, 1:-1] +
                          vec_fy[1:-1, 0:-2] +
                          vec_fy[2:  , 1:-1] +
                          vec_fy[1:-1, 2:  ] -
                          4*vec_fy[1:-1, 1:-1])/(dx**2)
    
    return np.stack([lap_x, lap_y], axis=-1) # Laplacian (Nx, Ny, 2)

def Gradient(p, dx):
    partial_x, partial_y = np.zeros_like(p), np.zeros_like(p)
    partial_x[1:-1, 1:-1] = (p[2:  , 1:-1] -
                             p[0:-2, 1:-1])/(2*dx)
    partial_y[1:-1, 1:-1] = (p[1:-1, 2:  ] -
                             p[1:-1, 0:-2])/(2*dx)

    return np.stack([partial_x, partial_y], axis=-1) # Gradient matrix (Nx, Ny, 2)

def Divergence(field, dx):
    div = np.zeros_like(X)
    field_x, field_y = field[..., 0], field[..., 1]
    div[1:-1, 1:-1] = ((field_x[2:  , 1:-1] -
                        field_x[0:-2, 1:-1])/(2*dx)
                        +
                        (field_y[1:-1, 2:  ] -
                         field_y[1:-1, 0:-2])/(2*dx))
    return div

def Curl(field, dx):
    curl = np.zeros_like(X)
    field_x, field_y = field[..., 0], field[..., 1]
    curl[1:-1, 1:-1] = ((field_y[2: , 1:-1] -
                        field_y[0:-2, 1:-1])/(2*dx)
                        -
                        (field_x[1:-1, 2: ] -
                        field_x[1:-1, 0:-2])/(2*dx))
    return curl

def diffuse(w2, dt, dx):
    def operator(field_flat):
        field = field_flat.reshape((N, N, 2))
        diff = field - KINEMATIC_VISCOSITY*dt*Laplacian(field, dx)
        return diff.flatten()
    
    matrix = splinalg.LinearOperator(shape=(N*N*2, N*N*2),
                                     matvec=operator)
    
    w3, info = splinalg.cg(A=matrix,
                           b=w2.flatten())

    return w3.reshape((N, N, 2)) # Diffused field (Nx, Ny, 2)

def project(w3):
    def operator(scalar_field_flat):
        scalar_field = scalar_field_flat.reshape((N, N))
        lap = np.zeros_like(X)
        lap[1:-1, 1:-1] = (scalar_field[0:-2, 1:-1] +
                           scalar_field[1:-1, 0:-2] +
                           scalar_field[2:  , 1:-1] +
                           scalar_field[1:-1, 2:  ] -
                           4*scalar_field[1:-1, 1:-1])/(dx**2)
        return lap.flatten()
    
    matrix = splinalg.LinearOperator(shape=(N*N, N*N),
                                     matvec=operator)
    
    p, info = splinalg.cg(A=matrix,
                           b=Divergence(w3, dx).flatten())
    p = p.reshape((N,N))
    w4 = w3 - Gradient(p, dx)

    return w4

def render_fluid_frames(solution, H, N, dx, dpi=200, num_workers=13, framerate=30):

    """
    solution: np.ndarray of shape (T, N, N, 2)  velocity field at each timestep
    H: domain size
    N: grid resolution
    dx: grid spacing
    """
    start_time = time.time()

    output_dir = "frames_fluid"
    os.makedirs(output_dir, exist_ok=True)

    def render_frame(frame):
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(13,13))
        ax.set_aspect("equal")
        ax.set_axis_off()

        w = solution[frame]  # (N,N,2)
        curl = Curl(w, dx)

        # background: curl heatmap
        ax.contourf(X, Y, curl, cmap=cmr.redshift,
                    vmin=-7, vmax=7, levels=300)
       
        magnitude = np.linalg.norm(w, axis=-1)
    
        u = np.divide(w[..., 0], 
                    magnitude, 
                    out=np.zeros_like(w[..., 0]), 
                    where=magnitude != 0)
        v = np.divide(w[..., 1], 
                    magnitude, 
                    out=np.zeros_like(w[..., 1]), 
                    where=magnitude != 0)
        
        step = 2
        ax.quiver(X[::step, ::step], Y[::step, ::step],
                  u[::step, ::step], v[::step, ::step],
                  magnitude[::step, ::step], scale=70, cmap="viridis")
        
        plt.savefig(f"{output_dir}/frame_{frame:04d}.png",
                    bbox_inches="tight", pad_inches=0, dpi=dpi)
        plt.close(fig)

    print("Rendering frames in parallel...")
    Parallel(n_jobs=num_workers)(
        delayed(render_frame)(t) for t in tqdm(range(solution.shape[0]), desc="Rendering")
    )

    # Check resolution
    sample_frame = Image.open(f"{output_dir}/frame_0000.png")
    width, height = sample_frame.size
    even_width  = width  if width  % 2 == 0 else width+1
    even_height = height if height % 2 == 0 else height+1
    print(f"Sample frame resolution: {width}x{height} → using {even_width}x{even_height}")

    video_filename = "fluid_simulation.mp4"
    print("Stitching frames into video...")

    ffmpeg_cmd = f"""
    ffmpeg -loglevel error -y -framerate {framerate} -i {output_dir}/frame_%04d.png \
    -vf "scale={even_width}:{even_height}:flags=lanczos" \
    -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p {video_filename}
    """
    os.system(ffmpeg_cmd)

    print("Deleting frames folder...")
    shutil.rmtree(output_dir)

    print(f"Video saved to {video_filename}")
    print(f"Total render time: {time.time()-start_time:.2f}s")

nsteps = int(T/dt)
solution = np.zeros((nsteps, N, N, 2))

w = w0.copy()
for i in tqdm(range(nsteps), desc="Simulating"):
    t = i * dt
    w1 = w + dt * force(t, X, Y, 9)
    w2 = advect(w1, dt)
    w3 = diffuse(w2, dt, dx)
    w4 = project(w3)
    w = w4.copy()
    solution[i] = w

render_fluid_frames(solution, H, N, dx, dpi=200, num_workers=15, framerate=30)