from collections import namedtuple
import numpy as np

from . import dynamics
from . import utils


def state_value(X_t, U, params, *args):

    # Compute Value Function
    VT_sum = dynamics.utility(
        cons, lab, params) + params.beta * V_INFINITY(X_tp1, params)

    return VT_sum


# Terminal value function
def V_T(X=[], params=None):
    return 0.0


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
