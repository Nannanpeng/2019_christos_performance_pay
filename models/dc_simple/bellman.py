from collections import namedtuple
import numpy as np
from scipy.special import logsumexp
import torch

from . import dynamics
from . import utils


def state_action_value(X_t, U_t, U_k, params, V_tp1, *args):
    # assert isinstance(X_t,torch.Tensor), "X_t was not tensor..."
    X_tp1 = dynamics.transition(X_t,U_t,U_k,params)

    # Compute Value Function
    # Retiree cannot resume work
    W_tp1 = torch.tensor(0.0)
    if U_k == 0:
        W_tp1 =  V_tp1(X_tp1.view(-1,1),k=0)[0]
    else:
        V = V_tp1(X_tp1.view(-1,1)).view(-1,) # need 1D array
        W_tp1 = params.sigma_e * torch.logsumexp(V / params.sigma_e)

    VT_sum = dynamics.utility(U_t,U_k) + params.beta * W_tp1
    if utils.test_inf_nan(VT_sum.detach().numpy()):
        import pdb; pdb.set_trace()

    return VT_sum

# Terminal value function
def V_T(X=[], k = None, params=None, maximum=False):
    if maximum or k is not None:
        return torch.tensor([0.0])
    return torch.tensor([0.0, 0.0])
