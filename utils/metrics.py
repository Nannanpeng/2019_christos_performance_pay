#======================================================================
#
#     This module contains routines to postprocess the VFI
#     solutions.
#
#     Simon Scheidegger, 01/19
#======================================================================

import numpy as np
import pickle
import logging
logger = logging.getLogger(__name__)


#======================================================================
# Routine compute the errors
def vfi_ls_error(n_agents, t1, t2, num_points, restart_fstr, params, output_path):
    with open(output_path, 'w') as file:
        # np.random.seed(0)
        unif = np.random.rand(num_points, n_agents)
        k_sample = params.k_bar + (unif) * (params.k_up - params.k_bar)
        to_print = np.empty((1, 3))

        for i in range(t1, t2-1):
            sum_diffs = 0
            diff = 0

            # Load the model from the previous iteration step
            restart_data = restart_fstr % (i)
            with open(restart_data, 'rb') as fd_old:
                gp_old = pickle.load(fd_old)
                logger.info("data from iteration step %d loaded from disk" % i)

            # Load the model from the previous iteration step
            restart_data = restart_fstr % (i+1)
            with open(restart_data, 'rb') as fd_new:
                gp_new = pickle.load(fd_new)
                logger.info("data from iteration step %d loaded from disk" % (i+1))

            y_pred_old, sigma_old = gp_old.predict(k_sample, return_std=True)
            y_pred_new, sigma_new = gp_new.predict(k_sample, return_std=True)

            # plot predictive mean and 95% quantiles
            #for j in range(num_points):
            #print k_sample[j], " ",y_pred_new[j], " ",y_pred_new[j] + 1.96*sigma_new[j]," ",y_pred_new[j] - 1.96*sigma_new[j]

            diff = y_pred_old - y_pred_new
            max_abs_diff = np.amax(np.fabs(diff))
            average = np.average(np.fabs(diff))

            to_print[0, 0] = i + 1
            to_print[0, 1] = max_abs_diff
            to_print[0, 2] = average

            np.savetxt(file, to_print, fmt='%2.16f')

#======================================================================
