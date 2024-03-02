import numpy as np
import scipy as sp
from multiprocessing import Pool
import matplotlib.pyplot as plt

# Solve the IVP RK4 -- for inertial particles

# define the parameters here
a, alpha, R, Fr, gravity = 1, 1, 0.3, 5, True



def solve_ivp_active_3d(args):


        q0, t_span = args

        # gravity = True
        positive_time = True

        def active_tracer_traj(t, Z) :
                # state-space vector 
                # Z[0] : xp, Z[1] : yp, Z[2] : zp
                # Z[3] : vx, Z[4] : vy, Z[5] : vz
                # compute flow field at particle position (xp, yp, zp)

                # for dimensional reason, the following propertes must be 1
                a, alpha = 1, 1

                xp = Z[0]
                yp = Z[1]
                zp = Z[2]
                dist = np.sqrt(xp**2 + yp**2+ zp**2)
                r_planar = np.sqrt(yp**2 + zp**2)

                if dist <= a:
                        Ux = alpha * (a**2 - dist**2 - r_planar**2) / 5
                        Uy = alpha * xp * yp / 5
                        Uz = xp * zp / 5

                        dUxdt = 0
                        dUydt = 0
                        dUzdt = 0
                                                   
                        dUxdx = -2 * alpha * xp / 5
                        dUxdy = -4 * alpha * yp / 5
                        dUxdz = -4 * alpha * zp / 5

                        dUydx = alpha * yp / 5
                        dUydy = alpha * xp / 5
                        dUydz = 0

                        dUzdx = zp / 5
                        dUzdy = 0
                        dUzdz = xp / 5


                else :
                        Ux = (-alpha*(a**5)/15) * (2/(a**3) + (r_planar**2 - 2*(xp**2))/(dist**5))
                        Uy = alpha* a**5 * yp * xp/ ( 5*(dist ** 5))
                        Uz = alpha* a**5 * zp * xp/ ( 5*(dist ** 5))

                        dUxdt = 0
                        dUydt = 0
                        dUzdt = 0

                        dUxdx = (-alpha*(a**5)/15) * (6 * (xp ** 3) - 9 * (xp * (r_planar ** 2)))/(dist**7)
                        dUxdy = (-alpha*(a**5)/15) * yp * (-3 * (r_planar ** 2) + 12 * (xp ** 2))/(dist**7)
                        dUxdz = (-alpha*(a**5)/15) * zp * (-3 * (r_planar ** 2) + 12 * (xp ** 2))/(dist**7)

                        dUydx = (alpha * (a**5)/5) * yp * ((r_planar ** 2) - 4 * ( xp ** 2)) / (dist**7)
                        dUydy = (alpha * (a**5)/5) * ((xp ** 3) - 4 * (xp * (yp ** 2)) + xp * (zp**2)) / (dist**7)
                        dUydz = - ( xp * yp * zp) / (dist**7)

                        dUzdx = (alpha * (a**5)/5) * zp * ((r_planar ** 2) - 4 * ( xp ** 2)) / (dist**7)
                        dUzdy = - ( xp * yp * zp) / (dist**7)
                        dUzdz = (alpha * (a**5)/5) * ((xp ** 3) - 4 * (xp * (zp ** 2)) + xp * (yp**2)) / (dist**7)



                # define derivatives
                dxpdt = Z[3]  # vx
                dypdt = Z[4]  # vy
                dzpdt = Z[5]  # vz
                ddxpdtt = R*(Ux - dxpdt)/St + (3*R/2) * (dUxdt + Ux*dUxdx + Uy*dUxdy + Uz*dUxdz)
                ddypdtt = R*(Uy - dypdt)/St + (3*R/2) * (dUydt + Ux*dUydx + Uy*dUydy + Uz*dUydz) 
                ddzpdtt = R*(Uz - dzpdt)/St + (3*R/2) * (dUzdt + Ux*dUzdx + Uy*dUzdy + Uz*dUzdz) - gravity * (1-3*R/2) / (Fr**2)
                 

                if positive_time:
                        return [dxpdt, dypdt, dzpdt, ddxpdtt, ddypdtt, ddzpdtt]
                else:
                        return [-dxpdt, -dypdt, -dzpdt, -ddxpdtt, -ddypdtt, -ddzpdtt]


        x0, y0, z0, vx0, vy0, vz0, St = q0
        sol = sp.integrate.solve_ivp(active_tracer_traj, [t_span[0], t_span[-1]], [x0, y0, z0, vx0, vy0, vz0], t_eval=t_span, vectorized=True)
        xpt, ypt, zpt, vxt, vyt, vzt = sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], sol.y[3][-1], sol.y[4][-1], sol.y[5][-1]
        #   t_eval = sol.t

        # return xpt, ypt, zpt, vxt, vyt, vzt
        return sol.y[0], sol.y[1], sol.y[2], sol.y[3], sol.y[4], sol.y[5]


def advect_bubbles(bubbles_df_to_advect, t0, tf, plot_path=False,  this_ax=None, color=None):

        '''
        this_ax: axis to  on
        N_cycle: number of cycles to plot to represent color
        
        '''

    
        initial_states = bubbles_df_to_advect[:, 1:8]
        t_span = np.linspace(t0,tf,500)

        n_proc = 12
        with Pool(n_proc) as pool:
                args = list(zip(initial_states, [t_span]*len(initial_states)))
                res = pool.map( solve_ivp_active_3d, args )


        res_array = np.stack(res, axis=0)  # shape (n_bubbles, 4, len(t_span))


        # plt.figure()
        if plot_path == True:
                # plt.sca(ax=this_ax)
                this_ax.scatter(res_array[:, 0, :].T, res_array[:, 1, :].T, res_array[:, 2, :].T,  s=0.01, c=color, linewidths=0)
                plt.axis('equal')
                this_ax.set_xlim(-2, 2)
                this_ax.set_ylim(-2, 2)
                this_ax.set_xlabel('x')
                this_ax.set_ylabel('y')
                this_ax.set_zlabel('z')

        plt.show()

        return res_array[:,:,-1]