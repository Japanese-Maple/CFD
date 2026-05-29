import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def sketch_obstacle(X, Y, mask, H,
                    save_path:str="obstacle_boundary.jpeg"):
    """
    Generates a high-contrast geometric sketch of the fluid domain 
    and the embedded solid obstacle boundary.
    """

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6, 6))

    contour = ax.contourf(X, Y, mask.astype(float), levels=[-0.5, 0.5, 1.5], 
                          colors=['#0F2027', '#E63946'])
    
    ax.contour(X, Y, mask, colors=['#FFFFFF'], linewidths=1.5, levels=[0.5])
    
    ax.set_xlim(0, H)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    
    ax.set_title("Fluid Domain & Obstacle Geometry", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("X", fontsize=10, color='#888888')
    ax.set_ylabel("Y", fontsize=10, color='#888888')
    
    ax.grid(True, linestyle=':', alpha=0.4, color='#444444')
    
    fluid_patch = mpatches.Patch(color='#0F2027', label='Active Fluid Workspace')
    solid_patch = mpatches.Patch(color='#E63946', label='Solid Obstacle (DBC)')
    ax.legend(handles=[fluid_patch, solid_patch], loc='upper right', framealpha=0.8)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()