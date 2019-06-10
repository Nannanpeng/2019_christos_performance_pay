import logging
logger = logging.getLogger(__name__)
import numpy as np
import math
from typing import Tuple
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from models import State
from utils import math as um


class Multi_LI_VFA(object):
    """
    Implements value function approximation. Can represent state values.
    """
    def __init__(self,T,discrete_shape,grid):
        self._discrete_shape = discrete_shape
        self._discrete_ndim = len(discrete_shape)
        # Use T+1 because we may also estimate value function at time 0..
        self._vfarray = um.ndlist(None,T+1,um.list_prod(discrete_shape))
        self._vfarray_extrap = um.ndlist(None,T+1,um.list_prod(discrete_shape))
        self._grid = grid

    def __call__(self,state: State):
        t = state.t
        ds = state.discrete
        dc = state.continuous
        idx = um.tuple_to_idx(ds,self._discrete_shape)
        vf = self._vfarray[t][idx]
        vfe = self._vfarray_extrap[t][idx]
        if vf is None:
            raise RuntimeError('Value function not estimated for time t=%d at discrete state %s' % (t,str(ds)))

        val = vf(state.continuous)
        # Use nearest neighbor for extrapolation if outside Conv(grid)
        val = val if not math.isnan(val) else vfe(state.continuous)
        return val

    def add_values(self,t,ds,vals):
        idx = um.tuple_to_idx(ds,self._discrete_shape)
        if self._grid.shape[0] != len(vals):
            raise RuntimeError("vals should have same length as number of grid points")
        vf = self._vfarray[t][idx]
        if vf is not None:
            raise RuntimeError("already added values at this idx and time...")
        # uses rescaling because dims may have very different scales
        self._vfarray[t][idx] = LinearNDInterpolator(self._grid,vals,rescale=True)
        self._vfarray_extrap[t][idx] = NearestNDInterpolator(self._grid,vals,rescale=True)


class Multi_LI_QFA(object):
    """
    Implements value function approximation. Represent state-action values (for a discrete set of actions).
    """
    def __init__(self,T,discrete_shape,K,grid):
        self._discrete_shape = discrete_shape
        self._discrete_ndim = len(discrete_shape)
        self._num_actions = K
        # Use T+1 because we may also estimate value function at time 0..
        self._vfarray = um.ndlist(None,T+1,K,um.list_prod(discrete_shape))
        self._vfarray_extrap = um.ndlist(None,T+1,K,um.list_prod(discrete_shape))
        self._grid = grid

    def __call__(self,state: State,k: int):
        t = state.t
        ds = state.discrete
        dc = state.continuous
        idx = um.tuple_to_idx(ds,self._discrete_shape)
        vf = self._vfarray[t][k][idx]
        vfe = self._vfarray_extrap[t][k][idx]
        if vf is None:
            raise RuntimeError('Value function not estimated for time t=%d at discrete state %s' % (t,str(ds)))

        val = vf(state.continuous)
        # Use nearest neighbor for extrapolation if outside Conv(grid)
        val = val if not math.isnan(val) else vfe(state.continuous)
        return val

    def add_values(self,t: int,ds: Tuple[int, ...],k: int, vals: np.ndarray):
        idx = um.tuple_to_idx(ds,self._discrete_shape)
        if self._grid.shape[0] != len(vals):
            raise RuntimeError("vals should have same length as number of grid points")
        vf = self._vfarray[t][k][idx]
        if vf is not None:
            raise RuntimeError("already added values at this idx and time...")
        # uses rescaling because dims may have very different scales
        self._vfarray[t][k][idx] = LinearNDInterpolator(self._grid,vals,rescale=True)
        self._vfarray_extrap[t][k][idx] = NearestNDInterpolator(self._grid,vals,rescale=True)
