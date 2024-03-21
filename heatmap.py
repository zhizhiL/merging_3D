import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os

# Directory where your .npy files are stored
data_directory = './noInflux_random_sims/temp'
# Updated to only include files that end with '.npy'
file_names = sorted([f for f in os.listdir(data_directory) if f.endswith('.npy')])

# Function to load data
def load_data(frame_number):
    file_path = os.path.join(data_directory, file_names[frame_number])
    return np.load(file_path)

# Set up the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialization function: plot the background of each frame
def init():
    ax.clear()
    ax.set_xlim([xmin, xmax])  # Set these based on your data
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])
    return fig,

# Animation update function
def update(frame_number):
    ax.clear()  # Clear old frames
    data = load_data(frame_number)
    ax.scatter(data[:,0], data[:,1], data[:,2])
    return fig,

# Number of frames (adjust according to your number of files)
num_frames = len(file_names)

# Optionally, specify the limits of your plot
xmin, xmax = -10, 10
ymin, ymax = -10, 10
zmin, zmax = -10, 10

# Create animation
ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False)

# Save the animation
ani.save('particle_animation.mp4', fps=10)  # Adjust FPS as needed

plt.show()
