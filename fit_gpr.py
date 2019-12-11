import os
import logging
logger = logging.getLogger(__name__)
import time
import numpy as np
from collections import namedtuple
import yaml
import random

import utils
import gpr.interpolation as interpol  #interface to sparse grid library/terminal VF
import gpr.interpolation_iter as interpol_iter  #interface to sparse grid library/iteration
import gpr.postprocessing as post  #computes the L2 and Linfinity error of the model

def configure_run(run_config):
    utils.make_directory(run_config['odir'])

    with open(os.path.join(run_config['odir'], 'run_config.yaml'), 'w') as fp:
        yaml.dump(run_config, fp, default_flow_style=False)

    log_level = logging.DEBUG if run_config['debug'] else logging.INFO
    if not run_config['console']:
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
              '\tModel: %s\r\n' % run_config['model']['name'] + \
              '###################################################'
    logger.info(log_str)
    #   '\tMax Updates, Save Interval: (%d,%d) \r\n' % (run_config['num_updates'],run_config['save_interval']) + \


_ITER_LOG_STR = """
===============================================================
Computation of a growth model of dimension %d
finished after %d steps
===============================================================
"""


def fit_model(run_config):
    start = time.time()
    parameters = run_config['model']['parameters']

    # Start with Value Function Iteration
    cp_fstr = '%s/restart_%%d.pcl' % (run_config['odir'])
    for i in range(parameters.numstart, parameters.numits):
        # terminal value function
        if (i == 1):
            logger.info("Value Function Iteration -- Initial Step")
            interpol.GPR_init(parameters, i, cp_fstr % i)
        else:
            logger.info("Value Function Iteration -- Step %d" % i)
            interpol_iter.GPR_iter(parameters, i, cp_fstr % (i-1), cp_fstr % i)

    logger.info(_ITER_LOG_STR % (parameters.n_agents, parameters.numits))

    # compute errors
    avg_err = post.ls_error(parameters.n_agents, parameters.numstart,
                            parameters.numits,
                            parameters.No_samples_postprocess,
                            run_config['odir'])

    end = time.time()
    logger.info('Time elapsed: %.3f %' (end - start))


if __name__ == "__main__":
    parser = utils.run.gpr_argparser()
    args = parser.parse_args()
    run_config = utils.run.run_config_from_args(args)
    print(run_config)
    configure_run(run_config)
    fit_model(run_config)
