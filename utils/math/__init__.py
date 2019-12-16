# Utility functions for optimization, math
import itertools
import functools
from operator import mul
import numpy as np
import copy
from scipy.optimize import fminbound

from .differentiate import jacobian_fd, dense_jacobian, derivative_fd, dense_hessian

# Maximize function V on interval [a,b]
def maximum(V, a, b):
    return float(V(fminbound(lambda x: -V(x), a, b)))

# Return Maximizer of function V on interval [a,b]
def maximizer(V, a, b):
    return float(fminbound(lambda x: -V(x), a, b))

def list_prod(lst):
    return functools.reduce(mul,lst)

def ndlist(init, *args):
    dp = init
    for x in reversed(args):
        dp = [copy.deepcopy(dp) for _ in range(x)]
    return dp

def lspaces_to_grid(*lps):
    grd = itertools.product(*lps)
    grd = [list(p) for p in grd]
    grd = np.array(grd)
    return grd

def tuple_to_idx(tp,shape):
    idx = 0
    step = 1
    for dim,val in zip(reversed(shape),reversed(tp)):
        idx = idx + step*val
        step = step * dim
    return idx

def idx_to_tuple(idx,shape):
    lst = []
    ndim = len(shape)
    for i in range(ndim):
        step = list_prod(shape[i+1:ndim]) if i < (ndim-1) else 1
        val = idx // step
        idx = idx - val*step
        lst.append(val)

    return tuple(lst)
