import numpy as np
import logging
import signal
import dill
logger = logging.getLogger(__name__)
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from sklearn.gaussian_process.kernels import RBF, DotProduct, ExpSineSquared, WhiteKernel

from utils import stdout_redirector, ipopt_stdout_filter
from . import solver_ipopt as solver
from estimator.gpr_dc import GPR_DC
logger.write = lambda msg: ipopt_stdout_filter(msg.decode('utf-8'), logger)


def VFI_iter(model, V_tp1=None, num_samples=20):
    logger.info("Beginning VFI Step")

    Xtraining, y_f, y_u = _evaluate_vf(model, num_samples, V_tp1)

    # Instantiate a Gaussian Process model
    # Fit to data using Maximum Likelihood Estimation of the parameters
    V_t = GPR_DC(model.num_choices,
                 kernel=[
                     DotProduct() * RBF(length_scale_bounds=(100, 100000.0)) + WhiteKernel()
                     for i in range(model.num_choices)
                 ],
                 n_restarts_optimizer=20)
    P_t = GPR_DC(model.num_choices,
                 kernel=[
                     DotProduct() * RBF(length_scale_bounds=(100, 100000.0)) + WhiteKernel()
                     for i in range(model.num_choices)
                 ],
                 n_restarts_optimizer=20)
    V_t.fit(Xtraining, y_f)
    P_t.fit(Xtraining, y_u)

    logger.info("Finished VFI Step")

    return V_t, P_t, Xtraining, y_f, y_u


def _run_one(args):
    (x, model_str, V_tp1, k) = args
    model = dill.loads(model_str)
    with stdout_redirector(logger):
        y_f, y_u = solver.solve(model, x, V_tp1=V_tp1, U_k=k)
    return y_f, y_u[0]


def _safe_run_set(X, k, model_str, V_tp1, y_f, y_u, start_idx):
    logger.debug('received start index %d' % start_idx)
    with ProcessPoolExecutor(max_workers=2) as executor:
        args_training = [(x, model_str, V_tp1, k) for x in X[start_idx:]]
        idx = start_idx
        try:
            results = executor.map(_run_one, args_training)
            for idx, (y_f_i, y_u_i) in enumerate(results, start=start_idx):
                logger.debug('Index: %s. Results: (%.3f,%.3f)' %
                             (idx, y_f_i, y_u_i))
                y_f[idx, k] = y_f_i
                y_u[idx, k] = y_u_i
        except BrokenProcessPool as e:
            idx += 1  # exceptions raised before idx incremented
            logger.info('Broken process - Failed index: %d' % idx)
            y_f[idx, k] = np.nan
            y_u[idx, k] = np.nan

        logger.debug("Final idx: %s" % idx)

        return idx


def _run_set(X, k, model_str, V_tp1, y_f, y_u):
    N = X.shape[0]
    start_idx = 0
    logger.debug('Number of samples: %d' % N)
    while start_idx < N:
        end_idx = _safe_run_set(X, k, model_str, V_tp1, y_f, y_u, start_idx)
        start_idx = end_idx + 1  # end_idx should be last *succesful* solve
        logger.debug("Max idx: %d" % end_idx)


def _evaluate_vf(model, num_samples, V_tp1):
    np.random.seed(666)

    dim = model.dim.state
    # Sample on log-scale and on uniform to get good coverage
    X1 = np.random.uniform(np.log(model.params.k_bar), np.log(model.params.k_up),
                          (num_samples // 2, dim))
    X2 = np.random.uniform(model.params.k_bar, model.params.k_up,
                        (num_samples // 2, dim))
    X = np.concatenate((np.exp(X1),X2))
    y_f = np.zeros((num_samples, model.num_choices),
                   float)  # training targets, value function
    y_u = np.zeros((num_samples, model.num_choices),
                   float)  # training targets, policy

    model_str = dill.dumps(
        model)  # dill can correctly serialize model parameters
    for k in range(model.num_choices):
        _run_set(X, k, model_str, V_tp1, y_f, y_u)

    return X, y_f, y_u
