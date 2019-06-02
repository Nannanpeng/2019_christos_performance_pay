from scipy.interpolate import LinearNDInterpolator

import utils
from . import utility
from . import dynamics

__all__ = ['SimplePPModel','GenericPPModel','utility','dynamics']


class BasePPModel(object):

    def __init__(self,spec,**kwargs):
        # some sanity checks for model spec
        # assert 'utilityFunction' in spec, "Need to specify a utility function"
        assert 'parameters' in spec, "Need to specify model parameters"
        assert 'constants' in spec, "Need to specify constants"

        self.constants = utils.struct_factory('%s_Constants' % spec['name'],spec['constants'])
        self.parameters = utils.struct_factory('%s_Parameters' % spec['name'],spec['parameters'])


class GenericPPModel(BasePPModel):

    def __init__(self,spec):
        super().__init__(spec)
        self.utility = getattr(utility,spec['utilityFunction'])
        self.bellman = getattr(dynamics,spec['bellman'])


class SimplePPModel(BasePPModel):

    def __init__(self,spec):
        super().__init__(spec)

    def bellman():
        pass



class MultiDiscretizationLI(object):
    """
    Grid should have one less dimension than vals

    Performs linear interpolation
    """
    def __init__(self,grid,values):
        self._num_grids = values.shape[0]
        self.discretizations = [LinearNDInterpolator(grid,values[i,]) for i in range(self._num_grids)]

    def eval(self,idx,points):
        assert idx < self._num_grids, "idx is out of range"
        return self.discretizations[idx]
