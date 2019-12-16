import numpy as np

from solver import IPOptCallback
from . import bellman, dynamics
import utils.math as mu
from models import ModelDimensions


class SimonGrowthModel:
    def __init__(self, params):
        self.params = params
        self.dim = ModelDimensions(3 * params.n_agents, 3 * params.n_agents + 1, params.n_agents)

    # state-action value function
    def value(self, X, U, V_tp1=None, **kwargs):
        if V_tp1 is None:
            return bellman.EV_F(X, U, self.params)
        return bellman.EV_F_ITER(X, U, self.params, V_tp1)

    # returns gradient of val function (ie deriv transpose)
    def value_deriv(self, X, U, V_tp1=None, **kwargs):
        F = None
        if V_tp1 is None:
            F = lambda U: bellman.EV_F(X, U, self.params)
        else:
            F = lambda U: bellman.EV_F_ITER(X, U, self.params, V_tp1)
        return mu.derivative_fd(F,U)

    @property
    def value_hess_sparsity(self):
        return mu.dense_hessian(self.dim.control)

    def constraints(self, X, U, **kwargs):
        return dynamics.EV_G(X, U, self.params)

    # returns derivative (jacobian) of constraint set
    def constraints_deriv(self, X, U, **kwargs):
        F = lambda U: dynamics.EV_G(X,U,self.params)
        return mu.jacobian_fd(F, U, self.dim.control,self.dim.constraints)

    @property
    def constraints_deriv_sparsity(self):
        return mu.dense_jacobian(self.dim.control,self.dim.constraints)

    @property
    def bounds_control(self):
        return dynamics.control_bounds(self.params)

    @property
    def bounds_constraint(self):
        return dynamics.constraint_bounds(self.params)
