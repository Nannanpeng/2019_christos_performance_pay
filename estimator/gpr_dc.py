import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern

class GPR_DC:

    def __init__(self,num_choices,n_restarts_optimizer=9):
        kernel = RBF(length_scale=5)
        self.num_choices = num_choices
        self._impl = [GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9) for i in range(num_choices)]

    def _all_preds(self,X,k,**kwargs):
        preds = [self._impl[k].predict(X,**kwargs).reshape(-1,1) for k in range(self.num_choices)]
        preds = np.hstack(preds)
        return preds

    def __call__(self,X,k=None,maximum=False,**kwargs):
        if maximum:
            all_preds = self._all_preds(X,k,**kwargs)
            return np.amax(all_preds,axis=1).reshape(-1,)
        if k is not None:
            return self._impl[k].predict(X,**kwargs)

        res = self._all_preds(X,k,**kwargs)
        res = res.reshape(-1,) # reshape to make sure 1D array
        return res

    def fit(self,X,y):
        assert y.shape[1] == self.num_choices, "Y should be array with width num_choices"
        for i in range(self.num_choices):
            _X = X
            _y = y[:,i]
            idxs = np.logical_not(np.isnan(_y))
            _y = _y[idxs]
            _X = _X[idxs]
            self._impl[i].fit(_X,_y)