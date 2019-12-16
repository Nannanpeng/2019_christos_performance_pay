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
    # V_old, sigma_test = V_tp1(Xtest, return_std=True)
    V_old = V_tp1(Xtest)

    VT_sum = dynamics.utility(cons, lab, params) + params.beta * V_old

    return VT_sum

