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
logger = logging.getLogger(__name__)
logger.write = lambda msg: logger.info(msg.decode('utf-8')) if msg.strip(
) != '' else None
import pickle

from . import solver_ipopt as solver
from utils import stdout_redirector
from estimator.gpr import GPR


def VFI_iter(model, V_tp1=None, num_samples = 20):
    logger.info("Beginning VFI Step")

    #fix seed
    np.random.seed(666)

    #generate sample aPoints
    dim = model.dim.state
    Xtraining = np.random.uniform(model.params.k_bar, model.params.k_up,
                                  (num_samples, dim))
    y = np.zeros(num_samples, float)  # training targets

    # solve bellman equations at training points
    # with stdout_redirector(logger):
    for iI in range(len(Xtraining)):
        y[iI] = solver.solve(model, Xtraining[iI], V_tp1=V_tp1)[0]

    # Instantiate a Gaussian Process model
    # Fit to data using Maximum Likelihood Estimation of the parameters
    V_t = GPR()
    V_t.fit(Xtraining, y)

    logger.info("Finished VFI Step")

    return V_t
