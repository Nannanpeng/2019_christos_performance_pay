import numpy as np
import logging
import signal
import torch.multiprocessing as mp
# if context is 'fork', mp hangs when passing pytorch model
torch_mp_ctx = mp.get_context("spawn") 
import dill
logger = logging.getLogger(__name__)
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import sklearn.gaussian_process.kernels as kr

from utils import stdout_redirector, ipopt_stdout_filter, stderr_redirector
from . import solver_ipopt as solver
from estimator.gpr_dc import GPR_DC
logger.write = lambda msg: ipopt_stdout_filter(msg.decode('utf-8'), logger)


def VFI_iter(model, V_tp1=None, num_samples=20):
    logger.info("Beginning VFI Step")

    Xtraining, y_f, y_u = _evaluate_vf(model, num_samples, V_tp1)

    # Instantiate a Gaussian Process model
    V_t = GPR_DC(model.num_choices)
    P_t = GPR_DC(model.num_choices)
    V_t.fit(Xtraining, y_f)
    P_t.fit(Xtraining, y_u)
    V_t.eval()
    P_t.eval()

    logger.info("Finished VFI Step")

    return V_t, P_t, Xtraining, y_f, y_u


def _run_one(args):
    (x, model_str, V_tp1_str, k) = args
    model = dill.loads(model_str)
    V_tp1 = dill.loads(V_tp1_str)

    with stdout_redirector(logger), stderr_redirector(logger):
        y_f, y_u, status = solver.solve(model, x, V_tp1=V_tp1, U_k=k)

    return x, y_f, y_u[0], status


def _safe_run_set(X, k, model_str, V_tp1, y_f, y_u, start_idx):
    logger.debug('received start index %d' % start_idx)
    with ProcessPoolExecutor(max_workers=1,mp_context=torch_mp_ctx) as executor:
        args_training = [(x, model_str, V_tp1, k) for x in X[start_idx:]]
        idx = start_idx - 1 # decrement in case first iteration fails
        try:
            results = executor.map(_run_one, args_training)
            for idx, (x, y_f_i, y_u_i, status_i) in enumerate(results, start=start_idx):
                logger.debug('Index: %s. Results: (%.3f,%.3f)' %
                             (idx, y_f_i, y_u_i))
                logger.info("[X: %s] Ipopt status: %s" % (str(x),status_i))
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
    X1 = np.random.uniform(np.log(model.params.k_bar),
                           np.log(model.params.k_up), (num_samples // 2, dim))
    X2 = np.random.uniform(model.params.k_bar, model.params.k_up,
                           ((num_samples // 2) - 1, dim))
    X3 = np.array([[0.99*model.params.k_up]]) # add top of range to sample points
    X = np.concatenate((np.exp(X1), X2, X3))
    y_f = np.zeros((num_samples, model.num_choices),
                   float)  # training targets, value function
    y_u = np.zeros((num_samples, model.num_choices),
                   float)  # training targets, policy

    # dill can correctly serialize model parameters
    model_str = dill.dumps(model)  
    V_tp1_str = dill.dumps(V_tp1)
    for k in range(model.num_choices):
        _run_set(X, k, model_str, V_tp1_str, y_f, y_u)
    return X, y_f, y_u
