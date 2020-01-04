import numpy as np
import logging
logger = logging.getLogger(__name__)
import torch
import copy
import gpytorch
import collections

import third_party.hjmshi.lbfgs as opt_tp

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y,
                                                     likelihood)
        # self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(
            num_mixtures=4)
        # self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


GPR_TYPE = SpectralMixtureGPModel


class GPR_DC:
    @staticmethod
    def from_saved(path):
        val = torch.load(PATH)
        gpr = GPR_DC(val['num_choices'])
        gpr._data = val['data']
        gpr._xnorm = val['xnorm']
        gpr._ynorm = val['ynorm']
        for i in range(gpr.num_choices):
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            gpr._impl[i] = GPR_TYPE(self._data[i][0], self._data[i][1],
                                    likelihood)
        return gpr

    @staticmethod
    def from_state(val):
        gpr = GPR_DC(val['num_choices'])
        gpr._data = val['data']
        gpr._xnorm = val['xnorm']
        gpr._ynorm = val['ynorm']
        for i in range(gpr.num_choices):
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            gpr._impl[i] = GPR_TYPE(gpr._data[i][0], gpr._data[i][1],
                                    likelihood)
            gpr._impl[i].load_state_dict(val['models'][i])
        gpr.eval()
        return gpr

    def __init__(self, num_choices, **kwargs):
        self.num_choices = num_choices
        self._impl = [None] * self.num_choices
        self._data = [None] * self.num_choices
        self._xnorm = [None,None]
        self._ynorm = ([None] * self.num_choices,[None] * self.num_choices)

    def save(self, path):
        val = self.state_dict()
        torch.save(val, path)

    def state_dict(self):
        val = {
            'models': [m.state_dict() for m in self._impl],
            'num_choices': self.num_choices,
            'data': self._data,
            'xnorm': self._xnorm,
            'ynorm': self._ynorm,
        }
        return val

    def eval(self):
        for i in range(self.num_choices):
            self._impl[i].eval()

    def _all_preds(self, X, **kwargs):
        preds = []
        preds = [
            self._impl[k](X, **kwargs).mean.unsqueeze(-1)
            for k in range(self.num_choices)
        ]
        preds_scaled = [self._ynorm[1][k]*p + self._ynorm[0][k] for k,p in enumerate(preds)]
        preds_scaled = torch.cat(preds_scaled, -1)
        return preds_scaled

    def __call__(self, X, k=None, maximum=False, **kwargs):
        X = torch.tensor(X) if not isinstance(X, torch.Tensor) else X
        X = (X - self._xnorm[0]) / self._xnorm[1]
        if maximum:
            all_preds = self._all_preds(X, **kwargs)
            max_preds = torch.max(all_preds, -1).values
            return max_preds
        if k is not None:
            res = self._impl[k](X, **kwargs).mean
            return self._ynorm[1][k]*res + self._ynorm[0][k]

        res = self._all_preds(X, **kwargs)
        return res

    def fit(self, X, y, **kwargs):
        assert y.shape[
            1] == self.num_choices, "Y should be array with width num_choices"
        X = torch.tensor(X)
        self._xnorm[0] = X.mean(dim=0)
        self._xnorm[1] = X.std(dim=0)
        X = (X - self._xnorm[0]) / self._xnorm[1]
        for i in range(self.num_choices):
            _X = X
            _y = y[:, i]
            likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(0.1))
            idxs = np.logical_not(np.isnan(_y))
            _y = torch.tensor(_y[idxs])
            self._ynorm[0][i] = _y.mean(dim=0)
            self._ynorm[1][i] = _y.std(dim=0)
            _X = _X[idxs]
            _y = (_y - self._ynorm[0][i]) / self._ynorm[1][i]
            self._data[i] = (_X, _y)
            self._impl[i] = _fit_gpr(GPR_TYPE, likelihood, _X, _y,**kwargs)

    def __str__(self):
        fstr = '\tDC%d: %s'
        strs = [
            fstr % (i, str(self._impl[i])) for i in range(self.num_choices)
        ]
        strs = '\r\n'.join(strs)
        return 'GPR_DC(%d):\r\n%s' % (self.num_choices, strs)


def _fit_gpr(GPR_TYPE, likelihood, X, y, num_iter=1000, lr=1, debug_log_interval=5, num_restarts = 9):
    logger.info('Fiting GPR (num_iter: %d, num_restarts: %d)' % (num_iter, num_restarts))
    min_loss = 0.0
    min_model = None
    for i in range(num_restarts):
        new_model = GPR_TYPE(X,y,likelihood)
        loss_i = _run_fit_gpr(new_model,likelihood, X, y, num_iter,debug_log_interval)
        if loss_i < min_loss or min_model is None:
            logger.debug('Found new best loss: %.3f' % loss_i)
            min_model = new_model
        else:
            logger.debug('Did not find new best loss')
    return min_model

def _run_fit_gpr(model, likelihood, X, y, num_iter, debug_log_interval):
    model.train()
    likelihood.train()
    eps = 1e-4
    h = 100

    # Use the adam optimizer
    optimizer = opt_tp.FullBatchLBFGS(model.parameters())

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def closure():
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X)
        # Calc loss and backprop gradients
        loss = -mll(output, y)
        return loss

    loss = closure()
    loss.backward()
    loss_min, loss_min_iter = None, -1
    for i in range(num_iter):
        # import pdb; pdb.set_trace()
        options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
        loss, _, lr, _, _, _, _, _ = optimizer.step(options)
        loss_val = loss.item()
        # Log Progress
        log_str = '%%s Iter %d/%d - Loss: %.3f | Noise: %.3f | LR: %.3f  ' % (
            i, num_iter, loss_val, model.likelihood.noise.item(), lr)
        if i == 0:
            logger.debug(log_str % '[START TRAIN]')
        elif i == (num_iter - 1):
            logger.debug(log_str % '[END TRAIN]')
        elif i % debug_log_interval == 0:
            logger.debug(log_str % '[TRAINING]')

        # Early stopping rule
        if loss_min is None or loss_val < loss_min - eps:
            loss_min, loss_min_iter = loss_val, i
        if loss_min_iter < i - h:
            logger.debug('[EARLY STOP] No decrease in %d iterations' % h)
            break

    return loss.item()
