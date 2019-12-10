#=======================================================================
#
#     ipopt_wrapper.py : an interface to IPOPT and PYIPOPT
#
#     Simon Scheidegger, 06/17
#
#=======================================================================

from . import model
import numpy as np

#=======================================================================
#   Objective Function to start VFI (in our case, the value function)


def EV_F(X, k_init, n_agents, params):
    # Extract Variables
    cons = X[0:n_agents]
    lab = X[n_agents:2 * n_agents]
    inv = X[2 * n_agents:3 * n_agents]

    knext = (1 - params.delta) * k_init + inv

    # Compute Value Function
    VT_sum = model.utility(cons, lab,
                           params) + params.beta * V_INFINITY(knext, params)

    return VT_sum


# V infinity
def V_INFINITY(k=[], params=None):
    e = np.ones(len(k))
    c = model.output_f(k, e, params)
    v_infinity = model.utility(c, e, params) / (1 - params.beta)
    return v_infinity


#=======================================================================
#   Objective Function during VFI (note - we need to interpolate on an "old" GPR)


def EV_F_ITER(X, k_init, n_agents, gp_old, params):

    # Extract Variables
    cons = X[0:n_agents]
    lab = X[n_agents:2 * n_agents]
    inv = X[2 * n_agents:3 * n_agents]

    knext = (1 - delta) * k_init + inv

    #transform to comp. domain of the model
    knext_cube = model.box_to_cube(knext)

    # initialize correct data format for training point
    s = (1, n_agents)
    Xtest = np.zeros(s)
    Xtest[0, :] = knext_cube

    # interpolate the function, and get the point-wise std.
    V_old, sigma_test = gp_old.predict(Xtest, return_std=True)

    VT_sum = utility(cons, lab) + beta * V_old

    return VT_sum


#=======================================================================
#   Computation of gradient (first order finite difference) of initial objective function


def EV_GRAD_F(X, k_init, n_agents, params):

    N = len(X)
    GRAD = np.zeros(N, float)  # Initial Gradient of Objective Function
    h = 1e-4

    for ixN in range(N):
        xAdj = np.copy(X)

        if (xAdj[ixN] - h >= 0):
            xAdj[ixN] = X[ixN] + h
            fx2 = EV_F(xAdj, k_init, n_agents, params)

            xAdj[ixN] = X[ixN] - h
            fx1 = EV_F(xAdj, k_init, n_agents, params)

            GRAD[ixN] = (fx2 - fx1) / (2.0 * h)

        else:
            xAdj[ixN] = X[ixN] + h
            fx2 = EV_F(xAdj, k_init, n_agents, params)

            xAdj[ixN] = X[ixN]
            fx1 = EV_F(xAdj, k_init, n_agents, params)
            GRAD[ixN] = (fx2 - fx1) / h

    return GRAD


#=======================================================================
#   Computation of gradient (first order finite difference) of the objective function


def EV_GRAD_F_ITER(X, k_init, n_agents, gp_old):

    N = len(X)
    GRAD = np.zeros(N, float)  # Initial Gradient of Objective Function
    h = 1e-4

    for ixN in range(N):
        xAdj = np.copy(X)

        if (xAdj[ixN] - h >= 0):
            xAdj[ixN] = X[ixN] + h
            fx2 = EV_F_ITER(xAdj, k_init, n_agents, gp_old)

            xAdj[ixN] = X[ixN] - h
            fx1 = EV_F_ITER(xAdj, k_init, n_agents, gp_old)

            GRAD[ixN] = (fx2 - fx1) / (2.0 * h)

        else:
            xAdj[ixN] = X[ixN] + h
            fx2 = EV_F_ITER(xAdj, k_init, n_agents, gp_old)

            xAdj[ixN] = X[ixN]
            fx1 = EV_F_ITER(xAdj, k_init, n_agents, gp_old)
            GRAD[ixN] = (fx2 - fx1) / h

    return GRAD


#======================================================================
#   Equality constraints for the first time step of the model


def EV_G(X, k_init, n_agents, params):
    N = len(X)
    M = 3 * n_agents + 1  # number of constraints
    G = np.empty(M, float)

    # Extract Variables
    cons = X[:n_agents]
    lab = X[n_agents:2 * n_agents]
    inv = X[2 * n_agents:3 * n_agents]

    # first n_agents equality constraints
    for i in range(n_agents):
        G[i] = cons[i]
        G[i + n_agents] = lab[i]
        G[i + 2 * n_agents] = inv[i]

    f_prod = model.output_f(k_init, lab, params)
    Gamma_adjust = 0.5 * params.zeta * k_init * (
        (inv / k_init - params.delta)**2.0)
    sectors_sum = cons + inv - params.delta * k_init - (f_prod - Gamma_adjust)
    G[3 * n_agents] = np.sum(sectors_sum)

    return G


#======================================================================
#   Equality constraints during the VFI of the model


def EV_G_ITER(X, k_init, n_agents):
    N = len(X)
    M = 3 * n_agents + 1  # number of constraints
    G = np.empty(M, float)

    # Extract Variables
    cons = X[:n_agents]
    lab = X[n_agents:2 * n_agents]
    inv = X[2 * n_agents:3 * n_agents]

    # first n_agents equality constraints
    for i in range(n_agents):
        G[i] = cons[i]
        G[i + n_agents] = lab[i]
        G[i + 2 * n_agents] = inv[i]

    f_prod = output_f(k_init, lab)
    Gamma_adjust = 0.5 * zeta * k_init * ((inv / k_init - delta)**2.0)
    sectors_sum = cons + inv - delta * k_init - (f_prod - Gamma_adjust)
    G[3 * n_agents] = np.sum(sectors_sum)

    return G


#======================================================================
#   Computation (finite difference) of Jacobian of equality constraints
#   for first time step


def EV_JAC_G(X, k_init, n_agents, params):
    N = len(X)
    M = 3 * n_agents + 1
    NZ = M * N
    A = np.empty(NZ, float)

    # Finite Differences
    h = 1e-4
    gx1 = EV_G(X, k_init, n_agents, params)

    for ixM in range(M):
        for ixN in range(N):
            xAdj = np.copy(X)
            xAdj[ixN] = xAdj[ixN] + h
            gx2 = EV_G(xAdj, k_init, n_agents, params)
            A[ixN + ixM * N] = (gx2[ixM] - gx1[ixM]) / h
    return A


#======================================================================
#   Computation (finite difference) of Jacobian of equality constraints
#   during iteration


def EV_JAC_G_ITER(X, k_init, n_agents):
    N = len(X)
    M = 3 * n_agents + 1
    NZ = M * N
    A = np.empty(NZ, float)

    # Finite Differences
    h = 1e-4
    gx1 = EV_G_ITER(X, k_init, n_agents)

    for ixM in range(M):
        for ixN in range(N):
            xAdj = np.copy(X)
            xAdj[ixN] = xAdj[ixN] + h
            gx2 = EV_G_ITER(xAdj, k_init, n_agents)
            A[ixN + ixM * N] = (gx2[ixM] - gx1[ixM]) / h
    return A


#======================================================================


def sparsity_jac_g(N, M):
    NZ = M * N
    ACON = np.empty(NZ, int)
    AVAR = np.empty(NZ, int)
    for ixM in range(M):
        for ixN in range(N):
            ACON[ixN + (ixM) * N] = ixM
            AVAR[ixN + (ixM) * N] = ixN

    return (ACON, AVAR)


def sparsity_hess(N):
    NZ = (N**2 - N) // 2 + N
    A1 = np.empty(NZ, int)
    A2 = np.empty(NZ, int)
    idx = 0
    for ixI in range(N):
        for ixJ in range(ixI + 1):
            A1[idx] = ixI
            A2[idx] = ixJ
            idx += 1

    return (A1, A2)