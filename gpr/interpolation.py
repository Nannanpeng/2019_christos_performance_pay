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
logger.write = lambda msg: logger.info(msg.decode('utf-8')) if msg.strip() != '' else None
import pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern

from . import nonlinear_solver as solver
from utils import stdout_redirector

def GPR_init(params, iteration, checkpoint_out):
    logger.info("Beginning Step %d" % iteration)

    #fix seed
    np.random.seed(666)

    #generate sample aPoints
    dim = params.n_agents
    Xtraining = np.random.uniform(params.k_bar, params.k_up,
                                  (params.No_samples, dim))
    y = np.zeros(params.No_samples, float)  # training targets

    # solve bellman equations at training points
    # with stdout_redirector(logger):
    for iI in range(len(Xtraining)):
        y[iI] = solver.solve(Xtraining[iI], params.n_agents, params)[0]

    # Instantiate a Gaussian Process model
    # Fit to data using Maximum Likelihood Estimation of the parameters
    kernel = RBF()
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp.fit(Xtraining, y)

    #save the model to a file
    logger.info('Output file: %s' % checkpoint_out)
    with open(checkpoint_out, 'wb') as fd:
        pickle.dump(gp, fd, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Step %d data  written to disk" % iteration)
