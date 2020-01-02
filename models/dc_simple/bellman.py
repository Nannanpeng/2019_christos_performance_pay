from collections import namedtuple
import numpy as np
import torch

from . import dynamics
from . import utils


def state_action_value(X_t, U_t, U_k, params, V_tp1, *args):
    # assert isinstance(X_t,torch.Tensor), "X_t was not tensor..."
    print('starting bellman')
    X_tp1 = dynamics.transition(X_t,U_t,U_k,params)

    # Compute Value Function
    # Retiree cannot resume work
    W_tp1 = torch.tensor(0.0)
    if U_k == 0:
        print('about to eval vtp1 1')
        print(X_tp1)
        W_tp1 =  V_tp1(X_tp1,k=0)[0]
        print(W_tp1)
    else:
        print('about to eval vtp1')
        V = V_tp1(X_tp1)
        W_tp1 = params.sigma_e * torch.logsumexp(V / params.sigma_e,1) # sum across rows
    print('finishing eval bellman')
    VT_sum = dynamics.utility(U_t,U_k) + params.beta * W_tp1
    if utils.test_inf_nan(VT_sum.detach().numpy()):
        import pdb; pdb.set_trace()
    print('returning from bellman')
    return VT_sum

# Terminal value function
def V_T(X=[], k = None, params=None, maximum=False):
    if maximum or k is not None:
        return torch.tensor([[0.0]])
    return torch.tensor([[0.0, 0.0]])
