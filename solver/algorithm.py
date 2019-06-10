import logging
logger = logging.getLogger(__name__)
import numpy as np
import math
import functools
import itertools
from operator import mul

from .models import State
from .stats import MultiDiscretizationLI


def lspaces_to_grid(*lps):
    grd = itertools.product(*lps)
    grd = [list(p) for p in grd]
    grd = np.array(grd)
    return grd


class VFA_LI(object):
    def __init__(self,T,discrete_shape,grid):
        self._discrete_shape = discrete_shape
        self._discrete_ndim = len(discrete_shape)
        self._vfarray = [None]*functools.reduce(mul,discrete_shape)


    def __call__(self,state: State):
        pass

    def _discrete_to_idx(self,discrete_state):
        pass



def solve(model):
    # Stores fitted value functions
    vfs = [None]*model.T

    # Define grid over continuous portion of state space
    z_grid = np.linspace(-10,10,3) # on a log scale, variance of noise is 0.06. If the process starts at 0, this range seems OK
    h_grid = np.linspace(0,30,3) # human capital, growth is upper bounded by 1 each time period.
    a_grid = np.exp(np.linspace(0,math.log(1.2e6),3))

    b = grid_from_linspaces(z_grid,h_grid,a_grid)
    print(b)

    # values for terminal state
    values = np.array([model.reward(State(model.T,(0,0,0),(0.0,0.0,a)),None,None) for a in a_grid])



    print(values)
