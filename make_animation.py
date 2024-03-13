import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5agg")
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def make_animation(coords, stokes=0.1):
    x, y, z = coords
    num_bubbles = x.shape[0]
    num_frames = x.shape[1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # initialize empty scatter plot
    scatter = ax.scatter([], [], [], c='r', alpha=0.6)

    def update(frame):
        ax.clear()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'frame {frame}')

        scatter = ax.scatter(x[:, frame], y[:, frame], z[:, frame], c='b', s=0.1, alpha=0.6)
        
        return scatter,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)

    # Set perspective projection
    ax.set_proj_type('persp')  # Perspective projection

    plt.show()


