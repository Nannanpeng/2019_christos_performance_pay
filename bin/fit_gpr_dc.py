import sys
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)
sys.path.append('./')
import os
import time
import numpy as np
from collections import namedtuple
import yaml
import random
import faulthandler

import utils
from solver.gpr_dc import VFI_iter
from models.dc_simple import DCSimple

_ITER_LOG_STR = """
===============================================================
Computation of a lifetime consumption with retirement model
finished after %d steps
===============================================================
"""


def diagnostic_plots(X, y_f, y_u, V, P, t, plot_out, lb=0.1, ub=500):
    fig = plt.figure(figsize=(10, 25))

    # Value Function
    ax = plt.subplot(3, 1, 1)
    ax.set_title('Value Function (T=%d)' % t)
    utils.plot.plot_function(V, 0.01, 500, 'Assets', 'Value', None, True)
    utils.plot.plot_vals(X, y_f, 'Assets', 'Value', None, True)

    # Plot Policy Worker
    ax = plt.subplot(3, 1, 2)
    ax.set_title('Policy for Worker')
    utils.plot.plot_function(P, lb, ub, 'Assets', 'Consumption', 1, False)
    utils.plot.plot_vals(X, y_u, 'Assets', 'Consumption', 1, False)

    # Plot Policy Retired
    ax = plt.subplot(3, 1, 3)
    ax.set_title('Policy for Retiree')
    utils.plot.plot_function(P, lb, ub, 'Assets', 'Consumption', 0, False)
    utils.plot.plot_vals(X, y_u, 'Assets', 'Consumption', 0, False)

    fig.tight_layout()
    plt.savefig(plot_out)
    plt.close()


def configure_run(run_config):
    utils.plot.configure()
    utils.make_directory(run_config['odir'])

    with open(os.path.join(run_config['odir'], 'run_config.yaml'), 'w') as fp:
        yaml.dump(run_config, fp, default_flow_style=False)

    log_level = logging.DEBUG if run_config['debug'] else logging.INFO
    if not run_config['terminal']:
        logging.basicConfig(
            filename=os.path.join(run_config['odir'], 'progress.log'),
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
    else:
        logging.basicConfig(level=log_level)

    # TODO: Check seeding with numpy, torch -- I don't think it will seed correctly as is
    if run_config['seed'] is not None:
        random.seed(run_config['seed'])

    log_str = '\r\n###################################################\r\n' + \
              '\tModel: %s\r\n' % run_config['model'].name + \
              '###################################################'
    logger.info(log_str)


def fit_model(run_config):
    start = time.time()
    model_spec = run_config['model']
    algorithm_config = run_config['algorithm_config']
    parameters = model_spec.parameters
    model = DCSimple(parameters)

    v_fstr = '%s/value_%%d.pcl' % (run_config['odir'])
    plot_fstr = '%s/diagnostic_plots_%%d.pdf' % (run_config['odir'])
    p_fstr = '%s/policy_%%d.pcl' % (run_config['odir'])
    vals_fstr = '%s/%%s_%%d.pcl' % (run_config['odir'])
    V_tp1, V_t = None, None

    # Value Function Iteration
    for i in range(parameters.T, 0, -1):
        # import pdb; pdb.set_trace()
        V_tp1 = V_t
        logger.info("Value Function Iteration -- Step %d" % i)
        V_t, P_t, X, y_f, y_u = VFI_iter(
            model, V_tp1, num_samples=algorithm_config.No_samples)
        V_t.save(v_fstr % i)
        P_t.save(p_fstr % i)
        utils.save_value(X, vals_fstr % ('X', i))
        utils.save_value(y_u, vals_fstr % ('y_u', i))
        utils.save_value(y_f, vals_fstr % ('y_f', i))
        diagnostic_plots(X, y_f, y_u, V_t, P_t, i, plot_fstr % i)

    logger.info(_ITER_LOG_STR % run_config['max_updates'])

    # Compute Value function errors, write to disk
    plots_path = '%s/plot_value.pdf' % (run_config['odir'])

    end = time.time()
    logger.info('Time elapsed: %.3f' % (end - start))


if __name__ == "__main__":
    faulthandler.enable()
    parser = utils.run.gpr_argparser(default_spec='./specs/simple_dc.yaml')
    args = parser.parse_args()
    run_config = utils.run.run_config_from_args(args)
    configure_run(run_config)
    fit_model(run_config)
