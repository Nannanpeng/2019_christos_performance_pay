import numpy as np
import logging
import signal
import dill
logger = logging.getLogger(__name__)
# from pathos.multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from . import solver_ipopt as solver
from estimator.gpr_dc import GPR_DC

def VFI_iter(model, V_tp1=None, num_samples = 20):
    logger.info("Beginning VFI Step")

    Xtraining, y_f, y_u = _run_iters(model,num_samples, V_tp1)

    # Instantiate a Gaussian Process model
    # Fit to data using Maximum Likelihood Estimation of the parameters
    V_t = GPR_DC(model.num_choices)
    P_t = GPR_DC(model.num_choices)
    V_t.fit(Xtraining, y_f)
    P_t.fit(Xtraining, y_u)

    logger.info("Finished VFI Step")

    return V_t, P_t

def _run_one(args):
    (x, model_str, V_tp1, k) = args
    model = dill.loads(model_str)
    # print(model.params)
    y_f, y_u = solver.solve(model, x, V_tp1 = V_tp1, U_k = k)
    return y_f, y_u[0]


def _run_set(X,k, model_str, V_tp1, y_f, y_u):
    with ProcessPoolExecutor(max_workers=2) as executor:
        args_training = [(x, model_str, V_tp1, k) for x in X]
        try:
            results = executor.map(_run_one,args_training)
            for idx, (y_f_i, y_u_i) in enumerate(results):
                print('Index: %s. Results: (%.3f,%.3f)' % (idx,y_f_i,y_u_i))
                y_f[idx,k] = y_f_i
                y_u[idx,k] = y_u_i
        except BrokenProcessPool as e:
            print('Broken process')
            print(e)

def _run_iters(model,num_samples, V_tp1):
    np.random.seed(666)

    dim = model.dim.state
    X = np.random.uniform(model.params.k_bar, model.params.k_up,
                                  (num_samples, dim))
    y_f = np.zeros((num_samples,model.num_choices), float)  # training targets, value function
    y_u = np.zeros((num_samples,model.num_choices), float)  # training targets, policy

    model_str = dill.dumps(model) # dill can correctly serialize model parameters
    for k in range(model.num_choices):
        _run_set(X,k,model_str,V_tp1,y_f,y_u)

    print(y_f)

    return X, y_f, y_u

