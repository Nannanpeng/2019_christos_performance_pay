import os
import logging
logger = logging.getLogger(__name__)
import time
import numpy as np
from collections import namedtuple
import yaml

import utils
import solver
import models

def configure_run(run_config):
    vf_result_path = os.path.join(run_config['odir'],'value_function.npy')
    make_directory(run_config['odir'])

    with open(os.path.join(run_config['odir'],'run_config.yaml'), 'w') as fp:
        yaml.dump(run_config, fp, default_flow_style=False)

    log_level = logging.DEBUG if run_config['debug'] else logging.INFO
    if not run_config['console']:
        logging.basicConfig(filename=os.path.join(run_config['odir'],'log_run_%d.log' % run_config['run']),
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',)
    else:
        logging.basicConfig(level=log_level)


    # TODO: Fix this for numba -- I don't think it will seed correctly as is
    if run_config['seed'] is not None:
        random.seed(run_config['seed'])

    log_str = '\r\n###################################################\r\n' + \
              '\tModel: %s\r\n' % run_config['model']['name'] + \
              '\tMax Updates, Save Interval: (%d,%d) \r\n' % (run_config['num_updates'],run_config['save_interval']) + \
              '###################################################'
    logger.info(log_str)



def fit_model(run_config):
    configure_run(run_config)


    # Select the algorithm
    alg = None
    try:
        algorithm_class = getattr(algorithm,run_config['algorithm'])
        alg = algorithm_class()
    except AttributeError as e:
        raise RuntimeError('Algorithm "%s" not found' % run_config['algorithm'])


    # episode_results = np.array([]).reshape((0,6))
    cur_update = 0
    start = time.time()
    while cur_update < run_config['num_updates']:
        # batch,crs,trs,els = smp.sample()
        # algo.update(batch)
        #
        # # save episode results
        # for i,(cr,tr,el) in enumerate(zip(crs,trs,els)):
        #     finished_episodes += 1
        #     total_samples = cur_update * samples_per_update
        #     # stores: total_updates, total_episodes, total_samples, current_episode_length, current_total_reward, current_cumulative_reward
        #     episode_results = np.concatenate((episode_results,np.array([cur_update,finished_episodes,total_samples,el,tr,cr],ndmin=2)),axis=0)
        #     np.save(episode_results_path, episode_results)
        #     logger.info('Update Number: %06d, Finished Episode: %04d ---  Length: %.3f, TR: %.3f, CDR: %.3f'% (cur_update,finished_episodes,el,tr,cr))
        break
        # checkpoint
        if cur_update % run_config['save_interval'] == 0:
            plc.save_model(os.path.join(run_config['odir'],'model_update_%06d.pt' % (cur_update)))

        cur_update += 1

    end = time.time()
    logger.info('Time elapsed: %.3f %' (end - start))



if __name__ == "__main__":
    parser = utils.run.fit_model_argparser()
    args = parser.parse_args()
    run_config = utils.run.run_config_from_args(args)
    # fit_model(run_config)
    # a = utils.load_model_from_yaml('./specs/simplified.yaml')
    # print(run_config['model'])
    model = models.SimplePPModel(run_config['model'])
    solver.algorithm.solve(model)
