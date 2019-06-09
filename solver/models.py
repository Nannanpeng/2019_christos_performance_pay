import logging
logger = logging.getLogger(__name__)
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from collections import namedtuple

import utils


class BasePPModel(object):

    def __init__(self,spec,**kwargs):
        # some sanity checks for model spec
        # assert 'utilityFunction' in spec, "Need to specify a utility function"
        assert 'parameters' in spec, "Need to specify model parameters"
        assert 'constants' in spec, "Need to specify constants"

        self.constants = utils.struct_factory('%s_Constants' % spec['name'],spec['constants'])
        self.parameters = utils.struct_factory('%s_Parameters' % spec['name'],spec['parameters'])



State = namedtuple('State',['t','discrete','continuous'])
PPControl = namedtuple('PPControl',['consumption','labor','job'])

class SimplePPModel(BasePPModel):
    """
    Simplified Model implementation.

    State space is an instance of SimpleState, where
     - discrete: (e_t,o_t,P_t), a length 3 tuple of integers
     - continuous: (z_t,h_t,a_t), a length 3 tuple of floats

    The interpretation of the components of the state are as follows
     - t: time, taking values in [0,30]
     - z_t: earnings process at time t
     - h_t: human capital at time t
     - a_t: assets at time t
    """

    def __init__(self,spec):
        super().__init__(spec)
        self.T = 31 # T = 31 is terminal state. TODO: This belongs elsewhere


    # shocks are the random part of the system dynamics
    # some transitions may not be deterministic, but could be draws from a multinomial
    # distribution. A list of tuples is returned with successor states and the corresponding
    # probabilities.
    # shocks is a list (in this case length 1), of values
    def transition_list(self,state: State, control: PPControl, shocks = [0.0]):

        tl = []


        return tl


    # need to pass in current state, the selected control, and any random shocks
    # TODO: Check to ensure borrowing constraint is satisfied
    def reward(self,state, control, shocks):
        if state.t == self.T:
            return self._terminal_utility(state)
        if state.t > self.T:
            raise RuntimeError('Requested time greater than T')

        return self.reward_noshock(state,control) + shocks[0]


    def reward_noshock(self,state, control):
        u = control.consumption**(1. - self.constants.iota) / (1. - self.constants.iota)
        u += - self.parameters.chi * (control.labor**(1+self.parameters.psi)) / (1+self.parameters.psi)
        u += shocks[0]

        return u



    # Simplifying assumption -- consume all wealth in final period
    def _terminal_utility(self,state):
        print('here',state,1. - self.constants.iota)
        return (state.continuous[2])**(1. - self.constants.iota) / (1. - self.constants.iota)




class GenericPPModel(BasePPModel):

    def __init__(self,spec):
        super().__init__(spec)
        self.utility = getattr(utility,spec['utilityFunction'])
        self.bellman = getattr(dynamics,spec['bellman'])
