import numpy as np


def output_f(kap=[], lab=[], params=None):
    fun_val = params.big_A * (kap**params.psi) * (lab**(1.0 - params.psi))
    return fun_val


def EV_G(X_t, U, params):
    N = len(U)
    M = 3 * params.n_agents + 1  # number of constraints
    G = np.empty(M, float)

    # Extract Variables
    cons = U[:params.n_agents]
    lab = U[params.n_agents:2 * params.n_agents]
    inv = U[2 * params.n_agents:3 * params.n_agents]

    # first params.n_agents equality constraints
    for i in range(params.n_agents):
        G[i] = cons[i]
        G[i + params.n_agents] = lab[i]
        G[i + 2 * params.n_agents] = inv[i]

    f_prod = output_f(X_t, lab, params)
    Gamma_adjust = 0.5 * params.zeta * X_t * ((inv / X_t - params.delta)**2.0)
    sectors_sum = cons + inv - params.delta * X_t - (f_prod - Gamma_adjust)
    G[3 * params.n_agents] = np.sum(sectors_sum)

    return G


def EV_JAC_G(X_t, U, params):
    N = len(U)
    M = 3 * params.n_agents + 1
    NZ = M * N
    A = np.empty(NZ, float)

    # Finite Differences
    h = 1e-4
    gu1 = EV_G(X_t, U, params)

    for iuM in range(M):
        for iuN in range(N):
            uAdj = np.copy(U)
            uAdj[iuN] = uAdj[iuN] + h
            gu2 = EV_G(X_t, uAdj, params)
            A[iuN + iuM * N] = (gu2[iuM] - gu1[iuM]) / h
    return A


def sparsity_jac_g(N, M):
    NZ = M * N
    ACON = np.empty(NZ, int)
    AVAR = np.empty(NZ, int)
    for iuM in range(M):
        for iuN in range(N):
            ACON[iuN + (iuM) * N] = iuM
            AVAR[iuN + (iuM) * N] = iuN

    return (ACON, AVAR)


def utility(cons=[], lab=[], params=None):
    sum_util = 0.0
    n = len(cons)
    for i in range(n):
        nom1 = (cons[i] / params.big_A)**(1.0 - params.gamma) - 1.0
        den1 = 1.0 - params.gamma

        nom2 = (1.0 - params.psi) * ((lab[i]**(1.0 + params.eta)) - 1.0)
        den2 = 1.0 + params.eta

        sum_util += (nom1 / den1 - nom2 / den2)

    util = sum_util

    return util


def control_bounds(params):
    n_agents = params.n_agents
    N = 3 * n_agents
    # Vector of variables -> solution of non-linear equation system
    U = np.empty(N)
    # Vector of lower and upper bounds
    U_L = np.empty(N)
    U_U = np.empty(N)

    # get coords of an individual grid points
    U_L[:n_agents] = params.c_bar
    U_U[:n_agents] = params.c_up

    U_L[n_agents:2 * n_agents] = params.l_bar
    U_U[n_agents:2 * n_agents] = params.l_up

    U_L[2 * n_agents:3 * n_agents] = params.inv_bar
    U_U[2 * n_agents:3 * n_agents] = params.inv_up

    return U_L, U_U


def constraint_bounds(params):
    n_agents = params.n_agents
    M = 3 * n_agents + 1  # number of constraints

    # Vector of lower and upper bounds
    G_L = np.empty(M)
    G_U = np.empty(M)

    # Set bounds for the constraints
    G_L[:n_agents] = params.c_bar
    G_U[:n_agents] = params.c_up

    G_L[n_agents:2 * n_agents] = params.l_bar
    G_U[n_agents:2 * n_agents] = params.l_up

    G_L[2 * n_agents:3 * n_agents] = params.inv_bar
    G_U[2 * n_agents:3 * n_agents] = params.inv_up

    G_L[3 * n_agents] = 0.0  # both values set to 0 for equality contraints
    G_U[3 * n_agents] = 0.0

    return G_L, G_U
