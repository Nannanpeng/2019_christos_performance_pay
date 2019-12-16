import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern

class GPR:

    def __init__(self,kernel=None,n_restarts_optimizer=9):
        kernel = RBF() if kernel is None else kernel
        self._impl = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    def __call__(self,X,**kwargs):
        return self._impl.predict(X,**kwargs)

    def fit(self,X,y):
        self._impl.fit(X,y)