from scipy.interpolate import LinearNDInterpolator
from collections import namedtuple

import utils
from . import utility
from . import dynamics
from . import stats

__all__ = ['SimplePPModel','GenericPPModel','utility','dynamics','stats']


class BasePPModel(object):

    def __init__(self,spec,**kwargs):
        # some sanity checks for model spec
        # assert 'utilityFunction' in spec, "Need to specify a utility function"
        assert 'parameters' in spec, "Need to specify model parameters"
        assert 'constants' in spec, "Need to specify constants"

        self.constants = utils.struct_factory('%s_Constants' % spec['name'],spec['constants'])
        self.parameters = utils.struct_factory('%s_Parameters' % spec['name'],spec['parameters'])



SimpleState = namedtuple('SimpleState',['discrete','continuous'])

class SimplePPModel(BasePPModel):
    """
    Simplified Model implementation.

    State space is an instance of SimpleState, where
     - discrete: (t,O_t,P_t), a length 3 tuple of integers
     - continuous: (z_t,h_t,a_t), a length 3 tuple of floats

    The interpretation of the components of the state are as follows
     - t: time, taking values in [0,30]
     - z_t: earnings process at time t
     - h_t: human capital at time t
     - a_t: assets at time t
    """

    def __init__(self,spec):
        super().__init__(spec)


    def transition(state: SimpleState):
        """
        Takes as input a state, optimal value function and returns the optimal value at the indicated state
        """
        pass


    def utility(state):
        pass


    def _terminal_utility(state):
        pass




class GenericPPModel(BasePPModel):

    def __init__(self,spec):
        super().__init__(spec)
        self.utility = getattr(utility,spec['utilityFunction'])
        self.bellman = getattr(dynamics,spec['bellman'])
