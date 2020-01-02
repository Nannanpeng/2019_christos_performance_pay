import numpy as np
import logging
logger = logging.getLogger(__name__)
import torch
import gpytorch
import collections


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
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(
            num_mixtures=4)
        self.covar_module.initialize_from_data(train_x, train_y)

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
        for i in range(gpr.num_choices):
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            gpr._impl[i] = GPR_TYPE(self._data[i][0], self._data[i][1],
                                    likelihood)
        return gpr

    @staticmethod
    def from_state(val):
        gpr = GPR_DC(val['num_choices'])
        gpr._data = val['data']
        for i in range(gpr.num_choices):
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            gpr._impl[i] = GPR_TYPE(gpr._data[i][0], gpr._data[i][1],
                                    likelihood)
            gpr._impl[i].load_state_dict(val['models'][i])
        gpr.eval()
        return gpr

    def __init__(self, num_choices, **kwargs):
        # kernel = kernel if kernel is not None else [RBF(length_scale=1) for i in range(num_choices)]
        # if not isinstance(kernel,collections.Sized):
        # kernel = [copy.deepcopy(kernel) for i in range(num_choices)]
        # if len(kernel) != num_choices:
        # raise RuntimeError('Incorrect length of kernels provided')
        self.num_choices = num_choices
        self._impl = [None] * self.num_choices
        self._data = [None] * self.num_choices

    def save(self, path):
        val = self.state_dict()
        torch.save(val, path)

    def state_dict(self):
        val = {
            'models': [m.state_dict() for m in self._impl],
            'num_choices': self.num_choices,
            'data': self._data
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
        preds = torch.cat(preds, -1)
        return preds

    def __call__(self, X, k=None, maximum=False, **kwargs):
        X = torch.tensor(X) if not isinstance(X, torch.Tensor) else X
        if maximum:
            all_preds = self._all_preds(X, **kwargs)
            max_preds = torch.max(all_preds, -1).values
            return max_preds
        if k is not None:
            return self._impl[k](X, **kwargs).mean

        res = self._all_preds(X, **kwargs)
        return res

    def fit(self, X, y, **kwargs):
        assert y.shape[
            1] == self.num_choices, "Y should be array with width num_choices"
        for i in range(self.num_choices):
            _X = X
            _y = y[:, i]
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            idxs = np.logical_not(np.isnan(_y))
            _y = torch.tensor(_y[idxs])
            _X = torch.tensor(_X[idxs])
            self._impl[i] = GPR_TYPE(_X, _y, likelihood)
            self._data[i] = (_X, _y)
            _fit_gpr(self._impl[i], likelihood, _X, _y,**kwargs)

    def __str__(self):
        fstr = '\tDC%d: %s'
        strs = [
            fstr % (i, str(self._impl[i])) for i in range(self.num_choices)
        ]
        strs = '\r\n'.join(strs)
        return 'GPR_DC(%d):\r\n%s' % (self.num_choices, strs)


def _fit_gpr(model, likelihood, X, y, num_iter=1500, lr=1, debug_log_interval=50):
    logger.info('Fiting GPR (num_iter: %d, lr: %.4f)' % (num_iter, lr))
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        [
            {
                'params': model.parameters()
            },  # Includes GaussianLikelihood parameters
        ],
        lr=lr)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(num_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X)
        # Calc loss and backprop gradients
        loss = -mll(output, y)
        loss.backward()

        # Log Progress
        log_str = '%%s Iter %d/%d - Loss: %.3f   noise: %.3f' % (
            i, num_iter, loss.item(), model.likelihood.noise.item())
        if i == 0:
            logger.info(log_str % '[START TRAIN]')
        elif i == (num_iter - 1):
            logger.info(log_str % '[START TRAIN]')
        elif i % debug_log_interval == 0:
            logger.debug(log_str % '[TRAINING]')

        optimizer.step()
