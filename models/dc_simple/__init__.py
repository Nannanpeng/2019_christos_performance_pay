import numpy as np

from solver import IPOptCallback
from . import bellman, dynamics
import utils.math as mu


class DCSimple:
    def __init__(self, params):
        self.params = params
        self.N = None
        self.M = None

    # state-action value function
    def value(self, X, U, V_tp1=None):
        if V_tp1 is None:
            return bellman.EV_F(X, U, self.params)
        return bellman.EV_F_ITER(X, U, self.params, V_tp1)

    def value_deriv(self, X, U, V_tp1=None):
        F = None
        if V_tp1 is None:
            F = lambda U: bellman.EV_F(X, U, self.params)
        else:
            F = lambda U: bellman.EV_F_ITER(X, U, self.params, V_tp1)
        return mu.derivative_fd(F,U)

    @property
    def value_hess_sparsity(self):
        return mu.dense_hessian(self.N)

    def constraints(self, X, U):
        return dynamics.EV_G(X, U, self.params)

    # returns derivative (jacobian) of constraint set
    def constraints_deriv(self, X, U):
        F = lambda U: dynamics.EV_G(X,U,self.params)
        return mu.jacobian_fd(F, U, self.N,self.M)

    @property
    def constraints_deriv_sparsity(self):
        return mu.dense_jacobian(self.N, self.M)

    @property
    def bounds_control(self):
        return dynamics.control_bounds(self.params)

    @property
    def bounds_constraint(self):
        return dynamics.constraint_bounds(self.params)
