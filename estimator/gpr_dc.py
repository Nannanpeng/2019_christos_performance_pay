import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern

class GPR_DC:

    def __init__(self,num_choices,n_restarts_optimizer=9):
        kernel = RBF()
        self.num_choices = num_choices
        self._impl = [GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9) for i in range(num_choices)]

    def __call__(self,X,k=None,**kwargs):
        if k is not None:
            return self._impl[k].predict(X,**kwargs)

    def fit(self,X,y):
        assert y.shape[1] == self.num_choices, "Y should be array with width num_choices"
        for i in range(self.num_choices):
            self._impl.fit(X,y[,:i])