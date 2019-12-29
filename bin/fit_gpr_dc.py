import sys
sys.path.append('./')
import os
import logging
logger = logging.getLogger(__name__)
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


def configure_run(run_config):
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

    # TODO: Fix this for numba -- I don't think it will seed correctly as is
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
    p_fstr = '%s/policy_%%d.pcl' % (run_config['odir'])
    V_tp1, V_t = None, None

    # Value Function Iteration
    for i in range(parameters.T,0,-1):
        # import pdb; pdb.set_trace()
        V_tp1 = V_t
        logger.info("Value Function Iteration -- Step %d" % i)
        V_t, P_t = VFI_iter(model, V_tp1, num_samples=algorithm_config.No_samples)
        utils.save_model(V_t, v_fstr % i)
        utils.save_model(P_t, p_fstr % i)

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
