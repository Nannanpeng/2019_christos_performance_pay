#======================================================================
#
#     This routine interfaces with Gaussian Process Regression
#     The crucial part is
#
#     y[iI] = solver.initial(Xtraining[iI], n_agents,gp_old)[0]
#     => at every training point, we solve an optimization problem
#
#     check kernels here: https://scikit-learn.org/stable/auto_examples/gaussian_process
#       /plot_gpr_prior_posterior.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-prior-posterior-py
#
#
#     Simon Scheidegger, 01/19
#======================================================================

import numpy as np
from . import nonlinear_solver as solver
import pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern

#======================================================================


def GPR_iter(params, iteration, checkpoint_in, checkpoint_out):

    # Load the model from the previous iteration step
    restart_data = filename + str(iteration - 1) + ".pcl"
    with open(restart_data, 'rb') as fd_old:
        gp_old = pickle.load(fd_old)
        print("data from iteration step ", iteration - 1, "loaded from disk")


    ##generate sample aPoints
    np.random.seed(666)  #fix seed
    dim = n_agents
    Xtraining = np.random.uniform(k_bar, k_up, (No_samples, dim))
    y = np.zeros(No_samples, float)  # training targets

    # solve bellman equations at training points
    for iI in range(len(Xtraining)):
        y[iI] = solver.solve(Xtraining[iI], n_agents, gp_old)[0]

    #print data for debugging purposes
    #for iI in range(len(Xtraining)):
    #print Xtraining[iI], y[iI]

    # Instantiate a Gaussian Process model
    # Fit to data using Maximum Likelihood Estimation of the parameters
    kernel = RBF()
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(Xtraining, y)

    ##save the model to a file
    output_file = filename + str(iteration) + ".pcl"
    print(output_file)
    with open(output_file, 'wb') as fd:
        pickle.dump(gp, fd, protocol=pickle.HIGHEST_PROTOCOL)
        print("data of step ", iteration, "  written to disk")
        print(" -------------------------------------------")
    fd.close()


#======================================================================
