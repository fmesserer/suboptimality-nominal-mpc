import numpy as np
from scipy.linalg import solve_discrete_are
import casadi as ca
from solver_treeMPC import treeOCP

def set_up_problem(noise_dist:float ='symmetric') -> treeOCP:

    x = ca.SX.sym("x")
    u = ca.SX.sym("u")
    w = ca.SX.sym("w")
    sig = ca.SX.sym("sig")
    ode = lambda x, u, _:  x +  x**3 + u

    N = 10
    T = 3
    f_discr = rk4_step(ode, x, u, 0, T/N)  + sig * w

    gamma = 1
    eps_soft_constr = 1e-2
    smooth_soft_constr = lambda y: 10 * (-y +  ca.sqrt(eps_soft_constr**2 + y**2))

    x_lb = -.1
    Q = 5
    R = 1
    lk_x = Q * x**2
    lk_x += smooth_soft_constr(x - x_lb)
    lx_u = R * u**2
    lk = lk_x + lx_u
    
    # get terminal cost via discrete time algebraic riccati equation
    # linearized system at origin
    A_func = ca.Function('A_func', [x, u, w, sig], [ca.jacobian(f_discr, x)])
    B_func = ca.Function('B_func', [x, u, w, sig], [ca.jacobian(f_discr, u)])
    A = A_func(0, 0, 0, 0).full()
    B = B_func(0, 0, 0, 0).full()
    QN = solve_discrete_are(A, B, Q, R)
    lN = QN * x**2
    lN += smooth_soft_constr(x - x_lb)

    if noise_dist == 'no_noise':
        Wval  = [0]
        Wdist = [1]
    elif noise_dist == 'symmetric':
        Wval =  [-1, 1]
        Wdist = [.5, .5]
    elif noise_dist == 'asymmetric':
        Wval = [-1 / np.sqrt(2), 2 / np.sqrt(2) ]
        Wdist = [2/3, 1/3]
    else:
        raise ValueError(f"noise_dist = {noise_dist} not implemented")
    p = ca.veccat(*[sig])

    return treeOCP(x, u, p, w, f_discr, lk, lN, N, Wval, Wdist, gamma)


def rk4_step(ode, x, u, w, dt):
    k1       = ode(x,             u, w)
    k2       = ode(x + dt/2 * k1, u, w)
    k3       = ode(x + dt/2 * k2, u, w)
    k4       = ode(x + dt * k3,   u, w)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)