import logging
logger = logging.getLogger(__name__)
import numpy as np
import math
from scipy.interpolate import LinearNDInterpolator


from .models import State
from utils import math as um



class VFA_LI(object):
    def __init__(self,T,discrete_shape,grid):
        self._discrete_shape = discrete_shape
        self._discrete_ndim = len(discrete_shape)
        # Use T+1 because we may also estimate value function at time 0..
        self._vfarray = [[None]*um.list_prod(discrete_shape)]*(T+1)
        self._grid = grid

    def __call__(self,state: State):
        t = state.t
        ds = state.discrete
        dc = state.continuous
        idx = um.tuple_to_idx(ds,self._discrete_shape)
        vf = self._vfarray[t][idx]
        if vf is None:
            raise RuntimeError('Value function not estimated for time t=%d at discrete state %s' % (t,str(ds)))

        return vf(state.continuous)

    def add_values(self,t,ds,vals):
        idx = um.tuple_to_idx(ds,self._discrete_shape)
        if self._grid.shape[0] != len(vals):
            raise RuntimeError("vals should have same length as number of grid points")
        vf = self._vfarray[t][idx]
        if vf is not None:
            raise RuntimeError("already added values at this idx and time...")
        self._vfarray[t][idx] = LinearNDInterpolator(self._grid,vals)



def solve(model):
    # Stores fitted value functions
    vfs = [None]*model.T

    # Define grid over continuous portion of state space
    z_grid = np.linspace(-10,10,3) # on a log scale, variance of noise is 0.06. If the process starts at 0, this range seems OK
    h_grid = np.linspace(0,30,3) # human capital, growth is upper bounded by 1 each time period.
    a_grid = np.exp(np.linspace(0,math.log(1.2e6),3))

    b = um.grid_from_linspaces(z_grid,h_grid,a_grid)
    print(b)

    # values for terminal state
    values = np.array([model.reward(State(model.T,(0,0,0),(0.0,0.0,a)),None,None) for a in a_grid])



    print(values)
