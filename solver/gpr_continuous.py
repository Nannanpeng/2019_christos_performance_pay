#======================================================================
#
#     This routine interfaces with Gaussian Process Regression
#     The crucial part is
#
#     y[iI] = solver.initial(Xtraining[iI], n_agents)[0]
#     => at every training point, we solve an optimization problem
#
#     Simon Scheidegger, 01/19
#======================================================================

import numpy as np
import logging
import sys
logger = logging.getLogger(__name__)
logger.write = lambda msg: logger.info(msg.decode('utf-8')) if msg.strip(
) != '' else None
import pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern

from . import nonlinear_solver as solver
from utils import stdout_redirector


def GPR_iter(model, iteration, V_tp1=None, num_samples = 20):
    logger.info("Beginning Step %d" % iteration)

    #fix seed
    np.random.seed(666)

    #generate sample aPoints
    dim = model.params.n_agents
    Xtraining = np.random.uniform(model.params.k_bar, model.params.k_up,
                                  (num_samples, dim))
    y = np.zeros(num_samples, float)  # training targets

    # solve bellman equations at training points
    # with stdout_redirector(logger):
    for iI in range(len(Xtraining)):
        y[iI] = solver.solve(model, Xtraining[iI], V_tp1)[0]

    # Instantiate a Gaussian Process model
    # Fit to data using Maximum Likelihood Estimation of the parameters
    kernel = RBF()
    V_t = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    V_t.fit(Xtraining, y)

    logger.info("Finished Step %d" % iteration)

    return V_t
