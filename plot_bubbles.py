# %matplotlib tk
import matplotlib
import matplotlib.pyplot as plt
import numpy as np



def plot_bubbles(updated_states: tuple, R0):

    # Plot grid lines
    x_out = updated_states[0]
    y_out = updated_states[1]
    z_out = updated_states[2]

    # Create data for a sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))



    for time in np.arange(0, 1501, 150):

        inside = np.sqrt(x_out[:, time]**2 + y_out[:, time]**2 + z_out[:, time]**2) <= R0
        outside = np.sqrt(x_out[:, time]**2 + y_out[:, time]**2 + z_out[:, time]**2) > R0

        # inside_particles = out[inside, :, time]
        # check = np.sqrt(out[inside, 0, time]**2 + out[inside, 1, time]**2 + out[inside, 2, time]**2)
        # avg_loc = np.mean(inside_particles, axis=0)[:3]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x_out[inside, time], y_out[inside, time], z_out[inside, time], s=0.1, c='r', marker='.', label='Inside',    alpha=0.6)
        ax.scatter(x_out[outside, time], y_out[outside, time], z_out[outside, time], s=0.1, c='b', marker='.', label='Outside', alpha=0.6)


        ax.plot_surface(x, y, z, color='y', alpha=0.3, label='Sphere')
        # Plot the y-z plane
        x_plane = np.zeros_like(y)  # Constant x values
        ax.plot_surface(x_plane, y, z, color='r', alpha=0.3, label='y-z plane')


        # ax.text(-4, 0, 2, 'aggregation at \n ({:.2f}, {:.2f}, {:.2f})'.format(avg_loc[0], avg_loc[1], avg_loc[2]))
        ax.legend()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([3,3,3])
        # plt.savefig('3Daggregation_{}.png'.format(time), dpi=300)
        plt.show()
        # plt.close()
        
