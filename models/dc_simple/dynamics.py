import numpy as np


def transition(X_t, U_t, U_k, params=None):
    X_tp1 = params.R*(X_t - U) + params.y*U_k
    return fun_val


def EV_G(X_t, U, params):
    N = len(U)
    M = N
    G = np.empty(M, float)
    
    G = X_t - U
    return G


def utility(U_t, U_k, params=None):
    import pdb; pdb.set_trace()
    util = np.log(U_t) - U_k
    
    return util


def control_bounds(params):
    N = 1
    # Vector of lower and upper bounds
    U_L = np.zeros(N,float)
    U_U = np.zeros(N,float)
    U_L[0] = params.c_bar
    U_U[0] = params.c_up
    return U_L, U_U


def constraint_bounds(params):
    M = 1
    # Vector of lower and upper bounds
    G_L = np.zeros(M,float)
    G_U = np.zeros(M,float)

    # Set bounds for the constraints
    G_L[0] = 0
    G_U[0] = 1e10

    return G_L, G_U
