from numbalsoda import lsoda_sig, lsoda
import numba as nb
import numpy as np

a, alpha, R, gravity, Fr = 1, 1, 2, True, 5

@nb.cfunc(lsoda_sig)
def rhs(t, q, dq, p):
    '''
    q: state variable
    q[0] : x , q[1] : y , q[2] : z
    q[3] : vx , q[4] : vy , q[5] : vz
    dq: derivative of q
    p: parameters
    '''
    
    St = p[0]
    xp, yp, zp = q[0], q[1], q[2]
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

    else:

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


    # define derivatives on the right hand side
    dq[0] = q[3]
    dq[1] = q[4]
    dq[2] = q[5]
    dq[3] = R*(Ux - q[3])/St + (3*R/2) * (dUxdt + Ux*dUxdx + Uy*dUxdy + Uz*dUxdz)
    dq[4] = R*(Uy - q[4])/St + (3*R/2) * (dUydt + Ux*dUydx + Uy*dUydy + Uz*dUydz) 
    dq[5] = R*(Uz - q[5])/St + (3*R/2) * (dUzdt + Ux*dUzdx + Uy*dUzdy + Uz*dUzdz) - gravity * (1-3*R/2) / (Fr**2)


funcptr = rhs.address
# t_eval = np.linspace(0.0,20.0,201)

@nb.njit(parallel=True)
def main(initial_states, t0, tf):

    n = initial_states.shape[0]
    t_eval = np.linspace(t0, tf, 501)

    q1 = np.empty((n,len(t_eval)), np.float64)
    q2 = np.empty((n,len(t_eval)), np.float64)
    q3 = np.empty((n,len(t_eval)), np.float64)
    q4 = np.empty((n,len(t_eval)), np.float64)
    q5 = np.empty((n,len(t_eval)), np.float64)
    q6 = np.empty((n,len(t_eval)), np.float64)

    for i in nb.prange(n):
        q0 = np.zeros((6,), np.float64)

        q0[0], q0[1], q0[2], q0[3], q0[4], q0[5] = initial_states[i, 0], initial_states[i, 1], initial_states[i, 2], \
            initial_states[i, 3], initial_states[i, 4], initial_states[i, 5]

        data = np.array(initial_states[i, 6]) #initial_states[i, 6]
        usol, _ = lsoda(funcptr, q0, t_eval, data = data)
        q1[i] = usol[:,0]
        q2[i] = usol[:,1]
        q3[i] = usol[:,2]
        q4[i] = usol[:,3]
        q5[i] = usol[:,4]
        q6[i] = usol[:,5]
  
    
    return q1[:,-1], q2[:,-1], q3[:,-1], q4[:,-1], q5[:,-1], q6[:,-1]
    # return q1, q2, q3, q4, q5, q6


# def rhs_res(t, q, dq, p):
#     '''
#     q: state variable
#     q[0] : x , q[1] : y , q[2] : z
#     q[3] : vx , q[4] : vy , q[5] : vz
#     dq: derivative of q
#     p: parameters
#     '''
    
#     St = p[0]
#     xp, yp, zp = q[0], q[1], q[2]
#     dist = np.sqrt(xp**2 + yp**2+ zp**2)
#     r_planar = np.sqrt(yp**2 + zp**2)

#     Ux = (-alpha*(a**5)/15) * (2/(a**3) + (r_planar**2 - 2*(xp**2))/(dist**5))
#     Uy = alpha* a**5 * yp * xp/ ( 5*(dist ** 5))
#     Uz = alpha* a**5 * zp * xp/ ( 5*(dist ** 5))

#     dUxdt = 0
#     dUydt = 0
#     dUzdt = 0

#     dUxdx = (-alpha*(a**5)/15) * (6 * (xp ** 3) - 9 * (xp * (r_planar ** 2)))/(dist**7)
#     dUxdy = (-alpha*(a**5)/15) * yp * (-3 * (r_planar ** 2) + 12 * (xp ** 2))/(dist**7)
#     dUxdz = (-alpha*(a**5)/15) * zp * (-3 * (r_planar ** 2) + 12 * (xp ** 2))/(dist**7)

#     dUydx = (alpha * (a**5)/5) * yp * ((r_planar ** 2) - 4 * ( xp ** 2)) / (dist**7)
#     dUydy = (alpha * (a**5)/5) * ((xp ** 3) - 4 * (xp * (yp ** 2)) + xp * (zp**2)) / (dist**7)
#     dUydz = - ( xp * yp * zp) / (dist**7)

#     dUzdx = (alpha * (a**5)/5) * zp * ((r_planar ** 2) - 4 * ( xp ** 2)) / (dist**7)
#     dUzdy = - ( xp * yp * zp) / (dist**7)
#     dUzdz = (alpha * (a**5)/5) * ((xp ** 3) - 4 * (xp * (zp ** 2)) + xp * (yp**2)) / (dist**7)


#     # define derivatives on the right hand side
#     dq[0] = q[3]
#     dq[1] = q[4]
#     dq[2] = q[5]
#     dq[3] = R*(Ux - q[3])/St + (3*R/2) * (dUxdt + Ux*dUxdx + Uy*dUxdy + Uz*dUxdz)
#     dq[4] = R*(Uy - q[4])/St + (3*R/2) * (dUydt + Ux*dUydx + Uy*dUydy + Uz*dUydz) 
#     dq[5] = R*(Uz - q[5])/St + (3*R/2) * (dUzdt + Ux*dUzdx + Uy*dUzdy + Uz*dUzdz) 

# funcptr_res = rhs_res.address
# # t_eval = np.linspace(0.0,20.0,201)

# @nb.njit(parallel=True)
# def main_res(initial_states, t0, tf):

#     n = initial_states.shape[0]
#     t_eval = np.linspace(t0, tf, 501)

#     q1 = np.empty((n,len(t_eval)), np.float64)
#     q2 = np.empty((n,len(t_eval)), np.float64)
#     q3 = np.empty((n,len(t_eval)), np.float64)
#     q4 = np.empty((n,len(t_eval)), np.float64)
#     q5 = np.empty((n,len(t_eval)), np.float64)
#     q6 = np.empty((n,len(t_eval)), np.float64)

#     for i in nb.prange(n):
#         q0 = np.zeros((6,), np.float64)

#         q0[0], q0[1], q0[2], q0[3], q0[4], q0[5] = initial_states[i, 0], initial_states[i, 1], initial_states[i, 2], \
#             initial_states[i, 3], initial_states[i, 4], initial_states[i, 5]

#         data = np.array(initial_states[i, 6]) #initial_states[i, 6]
#         usol, _ = lsoda(funcptr_res, q0, t_eval, data = data)
#         q1[i] = usol[:,0]
#         q2[i] = usol[:,1]
#         q3[i] = usol[:,2]
#         q4[i] = usol[:,3]
#         q5[i] = usol[:,4]
#         q6[i] = usol[:,5]
  
    
#     # return q1[:,-1], q2[:,-1], q3[:,-1], q4[:,-1], q5[:,-1], q6[:,-1]
#     return q1, q2, q3, q4, q5, q6