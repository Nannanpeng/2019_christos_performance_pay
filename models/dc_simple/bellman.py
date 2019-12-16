from collections import namedtuple
import numpy as np

from . import dynamics
from . import utils


def state_action_value(X_t, U_t, U_k, params, V_tp1, *args):
    X_tp1 = dynamics.transition(X_t,U_t,U_k,params)

    # Compute Value Function
    VT_sum = dynamics.utility(U_t,U_k) + params.beta * V_tp1(X_tp1, params, maximum=True)

    return VT_sum

# Terminal value function
def V_T(X=[], params=None, maximum=False):
    return 0.0
