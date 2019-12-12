import numpy as np

from solver import IPOptCallback
from . import bellman, dynamics


class SimonGrowthModel:
    def __init__(self, params):
        self.params = params
        self.N = 3 * params.n_agents
        self.M = 3 * params.n_agents + 1

    # state-action value function
    def value(self, X, U, V_tp1=None):
        if V_tp1 is not None:
            return bellman.EV_F(X, U, self.params)
        return bellman.EV_F_ITER(X, U, self.params, V_tp1)

    # returns gradient of val function (ie deriv transpose)
    def value_deriv(self, X, U, V_tp1=None):
        if V_tp1 is not None:
            return bellman.EV_GRAD_F(X, U, self.params)
        return bellman.EV_GRAD_F_ITER(X, U, self.params, V_tp1)

    @property
    def value_hess_sparsity(self):
        return bellman.sparsity_hess(self.N)

    def constraints(self, X, U):
        return dynamics.EV_G(X, U, self.params)

    # returns derivative (jacobian) of constraint set
    def constraints_deriv(self, X, U):
        return dynamics.EV_JAC_G(X, U, self.params)

    @property
    def constraints_deriv_sparsity(self):
        return dynamics.sparsity_jac_g(self.N, self.M)

    @property
    def bounds_control(self):
        return dynamics.control_bounds(self.params)

    @property
    def bounds_constraint(self):
        return dynamics.constraint_bounds(self.params)
