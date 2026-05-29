import numpy as np
import scipy.sparse.linalg as splinalg
from scipy import interpolate
from tqdm import tqdm
import cmasher as cmr
import matplotlib.pyplot as plt
import os, shutil, time
from joblib import Parallel, delayed
from PIL import Image

#------------------------------------------------------------------------------------------------

"""

 One of the most intriguing problems in computer graphics is the
 simulation of fluid-like behavior. A good fluid solver is of great
 importance in many different areas. In the special effects industry
 there is a high demand to convincingly mimic the appearance and
 behavior of fluids such as smoke, water and fire. Paint programs
 can also benefit from fluid solvers to emulate traditional techniques
 such as watercolor and oil paint. Texture synthesis is another possible 
 application. 

+----------------------------------------------------+
|  w0(x) --> w1(x) --> w2(x) --> w3(x) --> w4(x)     |
|    add force   advect    diffuse    project        |
+----------------------------------------------------+
"""

#================================================================================================
# EXTERNAL FORCE (user-defined)
#================================================================================================

def force(t, X, Y, a):

    force_field = np.zeros(X.shape + (2,))

    region = ((X > 7-0.7) & (X < 7) &
              (Y > 7/2-0.3) & (Y < 7/2+0.3))

    force_field[region, 0] = -12
    force_field[region, 1] = 3

    return force_field


#================================================================================================
# FLUID SOLVER
#================================================================================================

class FluidSolver:

    def __init__(self,
                 H=7,
                 N=230,
                 dt=0.1,
                 T=10,
                 KINEMATIC_VISCOSITY=0.001,
                 force_function=None,
                 obstacle_mask=None):

        self.H  = H
        self.N  = N
        self.dt = dt
        self.T  = T

        self.KINEMATIC_VISCOSITY = KINEMATIC_VISCOSITY
        self.force_function = force_function

        #------------------------------------------------

        self.x, self.y = np.linspace(0, H, N), np.linspace(0, H, N)

        self.X, self.Y = np.meshgrid(self.x, self.y,
                                      indexing='ij')

        self.COORDINATES = np.stack((self.X, self.Y), axis=-1)

        self.dx = H / (N - 1)

        self.w0 = np.zeros(self.X.shape + (2,))

        self.obstacle_mask = obstacle_mask

    #================================================================================================
    # ADVECTION
    #================================================================================================

    def advect(self, w1):

        coordinates = self.COORDINATES
        dt = self.dt

        def interpolation(vf1):

            x_d = interpolate.interpn(points=(self.x, self.y),
                                      values=w1[..., 0],
                                      xi=vf1,
                                      bounds_error=False,
                                      fill_value=None)

            y_d = interpolate.interpn(points=(self.x, self.y),
                                      values=w1[..., 1],
                                      xi=vf1,
                                      bounds_error=False,
                                      fill_value=None)

            return np.stack([x_d, y_d], axis=-1)

        k1 = -w1
        k2 = -interpolation(coordinates + dt/2 * k1)
        k3 = -interpolation(coordinates + dt/2 * k2)
        k4 = -interpolation(coordinates + dt * k3)

        displacement = dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        backtraced = np.clip(coordinates + displacement, 0, self.H)

        return interpolation(backtraced)

    #================================================================================================
    # DIFFERENTIAL OPERATORS
    #================================================================================================

    def Laplacian(self, vec_field):

        fx, fy = vec_field[..., 0], vec_field[..., 1]

        lap_x = np.zeros_like(fx)
        lap_y = np.zeros_like(fy)

        lap_x[1:-1, 1:-1] = (fx[:-2, 1:-1] +
                             fx[1:-1, :-2] +
                             fx[2:, 1:-1] +
                             fx[1:-1, 2:] -
                             4*fx[1:-1, 1:-1])/(self.dx**2)

        lap_y[1:-1, 1:-1] = (fy[:-2, 1:-1] +
                             fy[1:-1, :-2] +
                             fy[2:, 1:-1] +
                             fy[1:-1, 2:] -
                             4*fy[1:-1, 1:-1])/(self.dx**2)

        return np.stack([lap_x, lap_y], axis=-1)

    #------------------------------------------------------------------------------------------------

    def Gradient(self, p):

        px = np.zeros_like(p)
        py = np.zeros_like(p)

        px[1:-1, 1:-1] = (p[2:, 1:-1] - p[:-2, 1:-1])/(2*self.dx)
        py[1:-1, 1:-1] = (p[1:-1, 2:] - p[1:-1, :-2])/(2*self.dx)

        return np.stack([px, py], axis=-1)

    #------------------------------------------------------------------------------------------------

    def Divergence(self, field):

        div = np.zeros_like(self.X)

        fx, fy = field[..., 0], field[..., 1]

        div[1:-1, 1:-1] = ((fx[2:, 1:-1] - fx[:-2, 1:-1])/(2*self.dx) +
                           (fy[1:-1, 2:] - fy[1:-1, :-2])/(2*self.dx))

        return div

    #------------------------------------------------------------------------------------------------

    def Curl(self, field):

        curl = np.zeros_like(self.X)

        fx, fy = field[..., 0], field[..., 1]

        curl[1:-1, 1:-1] = ((fy[2:, 1:-1] - fy[:-2, 1:-1])/(2*self.dx) -
                            (fx[1:-1, 2:] - fx[1:-1, :-2])/(2*self.dx))

        return curl

    #================================================================================================
    # DIFFUSE + PROJECT
    #================================================================================================

    def diffuse(self, w2):

        def operator(flat):

            field = flat.reshape((self.N, self.N, 2))

            return (field -
                    self.KINEMATIC_VISCOSITY * self.dt * self.Laplacian(field)
                   ).flatten()

        A = splinalg.LinearOperator(
            shape=(self.N*self.N*2, self.N*self.N*2),
            matvec=operator
        )

        w3, _ = splinalg.cg(A, w2.flatten())

        return w3.reshape((self.N, self.N, 2))

    #------------------------------------------------------------------------------------------------

    def project(self, w3):

        def operator(flat):

            p = flat.reshape((self.N, self.N))

            lap = np.zeros_like(self.X)

            lap[1:-1, 1:-1] = (p[:-2, 1:-1] +
                               p[1:-1, :-2] +
                               p[2:, 1:-1] +
                               p[1:-1, 2:] -
                               4*p[1:-1, 1:-1])/(self.dx**2)

            return lap.flatten()

        A = splinalg.LinearOperator(
            shape=(self.N*self.N, self.N*self.N),
            matvec=operator
        )

        p, _ = splinalg.cg(A, self.Divergence(w3).flatten())

        p = p.reshape((self.N, self.N))

        return w3 - self.Gradient(p)  

    #================================================================================================
    # OBSTACLE APPLICATION
    #================================================================================================

    def apply_obstacle(self, field):

        if self.obstacle_mask is None:
            return field

        field = field.copy()
        field[self.obstacle_mask, :] = 0
        return field

    #================================================================================================
    # SIMULATION
    #================================================================================================

    def solve(self):

        nsteps = int(self.T / self.dt)

        solution = np.zeros((nsteps, self.N, self.N, 2))

        w = self.w0.copy()

        for i in tqdm(range(nsteps), desc="Simulating"):

            t = i * self.dt

            F = self.force_function(t, self.X, self.Y, 9)

            w1 = w + self.dt * F
            w1 = self.apply_obstacle(w1)

            w2 = self.advect(w1)
            w2 = self.apply_obstacle(w2)  

            w3 = self.diffuse(w2)
            w3 = self.apply_obstacle(w3)

            w4 = self.project(w3)
            w4 = self.apply_obstacle(w4)
  
            w = w4.copy()
            solution[i] = w

        return solution

    #================================================================================================
    # RENDER
    #================================================================================================

    def render_fluid_frames(self,  
                            solution,
                            dpi=200,
                            num_workers=13,
                            framerate=30,
                            video_filename="Fluid_Simulation.mp4"):

        start_time = time.time()

        output_dir = "frames_fluid"
        os.makedirs(output_dir, exist_ok=True)

        def render_frame(frame):

            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(13, 13))

            ax.set_aspect("equal")
            ax.set_axis_off()

            w = solution[frame]
            curl = self.Curl(w)

            ax.contourf(self.X, self.Y, curl,
                        cmap=cmr.redshift,
                        vmin=-7, vmax=7, levels=300)

            magnitude = np.linalg.norm(w, axis=-1)

            u = np.divide(w[..., 0], magnitude,
                          out=np.zeros_like(w[..., 0]),
                          where=magnitude != 0)

            v = np.divide(w[..., 1], magnitude,
                          out=np.zeros_like(w[..., 1]),
                          where=magnitude != 0)

            step = 2

            ax.quiver(self.X[::step, ::step],
                      self.Y[::step, ::step],
                      u[::step, ::step],
                      v[::step, ::step],
                      magnitude[::step, ::step],
                      scale=70,
                      cmap="viridis")

            plt.savefig(f"{output_dir}/frame_{frame:04d}.png",
                        bbox_inches="tight",
                        pad_inches=0,
                        dpi=dpi)

            plt.close(fig)

        Parallel(n_jobs=num_workers)(
            delayed(render_frame)(t)
            for t in tqdm(range(solution.shape[0]), desc="Rendering")
        )

        sample = Image.open(f"{output_dir}/frame_0000.png")
        w, h = sample.size

        w += (w % 2)
        h += (h % 2)

        ffmpeg_cmd = f"""
        ffmpeg -loglevel error -y -framerate {framerate} -i {output_dir}/frame_%04d.png \
        -vf "scale={w}:{h}:flags=lanczos" \
        -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p {video_filename}
        """

        os.system(ffmpeg_cmd)

        shutil.rmtree(output_dir)

        print(f"Video saved to {video_filename}")
        print(f"Total render time: {time.time()-start_time:.2f}s")


#================================================================================================
# MAIN
#================================================================================================

if __name__ == "__main__":

    obstacle_mask = ((np.linspace(0,7,230)[:,None] - 3.5)**2 +
                     (np.linspace(0,7,230)[None,:] - 4.5)**2) < 0.75**2

    solver = FluidSolver(
        H=7,
        N=230,
        dt=0.1,
        T=10,
        force_function=force,
        obstacle_mask=obstacle_mask
    )

    solution = solver.solve()

    solver.render_fluid_frames(solution,
                                dpi=200,
                                num_workers=15,
                                framerate=30)