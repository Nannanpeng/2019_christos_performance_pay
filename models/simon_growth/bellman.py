from collections import namedtuple
import numpy as np

from . import dynamics
from . import utils


#=======================================================================
#   Objective Function to start VFI (in our case, the value function)
def EV_F(X_t, U, params, *args):
    # Extract Variables
    cons = U[0:params.n_agents]
    lab = U[params.n_agents:2 * params.n_agents]
    inv = U[2 * params.n_agents:3 * params.n_agents]

    X_tp1 = (1 - params.delta) * X_t + inv

    # Compute Value Function
    VT_sum = dynamics.utility(
        cons, lab, params) + params.beta * V_INFINITY(X_tp1, params)

    return VT_sum


# V infinity
def V_INFINITY(X=[], params=None):
    e = np.ones(len(X))
    c = dynamics.output_f(X, e, params)
    v_infinity = dynamics.utility(c, e, params) / (1 - params.beta)
    return v_infinity


#=======================================================================
#   Objective Function during VFI (note - we need to interpolate on an "old" GPR)


def EV_F_ITER(X_t, U, params, V_tp1, *args):
    # Extract Variables
    cons = U[0:params.n_agents]
    lab = U[params.n_agents:2 * params.n_agents]
    inv = U[2 * params.n_agents:3 * params.n_agents]

    X_tp1 = (1 - params.delta) * X_t + inv

    #transform to comp. domain of the model
    X_tp1_cube = utils.box_to_cube(X_tp1, params)

    # initialize correct data format for training point
    s = (1, params.n_agents)
    Xtest = np.zeros(s)
    Xtest[0, :] = X_tp1_cube

    # interpolate the function, and get the point-wise std.
    V_old, sigma_test = V_tp1.predict(Xtest, return_std=True)

    VT_sum = dynamics.utility(cons, lab, params) + params.beta * V_old

    return VT_sum


#=======================================================================
#   Computation of gradient (first order finite difference) of initial objective function

def EV_GRAD_F(X_t, U, params, *args):
    return _EV_GRAD_F_IMPL(X_t,U,params,EV_F)

def EV_GRAD_F_ITER(X_t, U, params, V_tp1, *args):
    return _EV_GRAD_F_IMPL(X_t,U,params,EV_F_ITER,V_tp1)

def _EV_GRAD_F_IMPL(X_t, U, params, _EV_F, V_tp1 = None):
    N = len(U)
    GRAD = np.zeros(N, float)  # Initial Gradient of Objective Function
    h = 1e-4

    for iuN in range(N):
        uAdj = np.copy(U)

        if (uAdj[iuN] - h >= 0):
            uAdj[iuN] = U[iuN] + h
            fu2 = _EV_F(X_t, uAdj, params, V_tp1)

            uAdj[iuN] = U[iuN] - h
            fu1 = _EV_F(X_t, uAdj, params, V_tp1)

            GRAD[iuN] = (fu2 - fu1) / (2.0 * h)

        else:
            uAdj[iuN] = U[iuN] + h
            fu2 = _EV_F(X_t, uAdj, params, V_tp1)

            uAdj[iuN] = U[iuN]
            fu1 = _EV_F(X_t, uAdj, params, V_tp1)
            GRAD[iuN] = (fu2 - fu1) / h

    return GRAD

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
