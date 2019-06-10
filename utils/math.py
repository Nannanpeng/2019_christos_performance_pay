# Utility functions for optimization, math
import itertools
import functools
from operator import mul
import numpy as np
from scipy.optimize import fminbound

# Maximize function V on interval [a,b]
def maximum(V, a, b):
    return float(V(fminbound(lambda x: -V(x), a, b)))

# Return Maximizer of function V on interval [a,b]
def maximizer(V, a, b):
    return float(fminbound(lambda x: -V(x), a, b))

def list_prod(lst):
    return functools.reduce(mul,lst)


def lspaces_to_grid(*lps):
    grd = itertools.product(*lps)
    grd = [list(p) for p in grd]
    grd = np.array(grd)
    return grd

def tuple_to_idx(tp,shape):
    idx = 0
    step = 1
    for dim,val in zip(reversed(shape),reversed(tp)):
        print(dim,val)
        idx = idx + step*val
        step = step * dim
        print('step %d' % step)
    return idx

def idx_to_tuple(idx,shape):
    lst = []
    for i in reversed(range(len(shape))):
        step = list_prod(shape[0:i]) if i > 0 else 1
        val = idx // step
        idx = idx - val*step
        print(val,step)
        lst.append(val)

    return tuple(lst)
