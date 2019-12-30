from collections import namedtuple
import numpy as np
from scipy.special import logsumexp

from . import dynamics
from . import utils


def state_action_value(X_t, U_t, U_k, params, V_tp1, *args):
    X_tp1 = dynamics.transition(X_t,U_t,U_k,params)

    # Compute Value Function
    # VT_sum = dynamics.utility(U_t,U_k) + params.beta * V_tp1(X_tp1.reshape(-1,1), maximum=True)[0]
    # Retiree cannot resume work
    W_tp1 = 0.0
    if U_k == 0:
        W_tp1 =  V_tp1(X_tp1.reshape(-1,1),k=0)[0]
    else:
        V = V_tp1(X_tp1.reshape(-1,1)).reshape(-1,) # need 1D array
        # W_tp1 = params.sigma_e * np.log(sum(np.exp(V / params.sigma_e)))
        W_tp1 = params.sigma_e * logsumexp(V / params.sigma_e)

    VT_sum = dynamics.utility(U_t,U_k) + params.beta * W_tp1

    if utils.test_inf_nan(VT_sum):
        import pdb; pdb.set_trace()

    return VT_sum

# Terminal value function
def V_T(X=[], k = None, params=None, maximum=False):
    if maximum or k is not None:
        return np.array([0.0])
    return np.array([0.0, 0.0])
