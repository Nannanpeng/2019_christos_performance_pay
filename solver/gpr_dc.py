import numpy as np
import logging
logger = logging.getLogger(__name__)
logger.write = lambda msg: logger.info(msg.decode('utf-8')) if msg.strip(
) != '' else None
import pickle

from . import solver_ipopt as solver
from utils import stdout_redirector
from estimator.gpr_dc import GPR_DC


def VFI_iter(model, V_tp1=None, num_samples = 20):
    logger.info("Beginning VFI Step")

    #fix seed
    np.random.seed(666)

    #generate sample aPoints
    dim = model.dim.state
    K = model.num_choices
    Xtraining = np.random.uniform(model.params.k_bar, model.params.k_up,
                                  (num_samples, dim))
    y_f = np.zeros((num_samples,model.num_choices), float)  # training targets, value function
    y_u = np.zeros((num_samples,model.num_choices), float)  # training targets, policy
    # solve bellman equations at training points
    # with stdout_redirector(logger):
    for k in range(model.num_choices):
        for iI in range(len(Xtraining)):
            y_f[iI,k],y_u[iI,k] = solver.solve(model, Xtraining[iI], V_tp1 = V_tp1, U_k = k)

    # Instantiate a Gaussian Process model
    # Fit to data using Maximum Likelihood Estimation of the parameters
    V_t = GPR_DC(model.num_choices)
    P_t = GPR_DC(model.num_choices)
    V_t.fit(Xtraining, y_f)
    P_t.fit(Xtraining, y_u)

    logger.info("Finished VFI Step")

    return V_t, P_t
