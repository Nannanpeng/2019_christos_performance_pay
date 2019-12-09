#======================================================================
#
#     This routine solves an infinite horizon growth model
#     with dynamic programming and sparse grids
#
#     The model is described in Scheidegger & Bilionis (2017)
#     https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2927400
#
#     external libraries needed:
#     - IPOPT (https://projects.coin-or.org/Ipopt)
#     - PYIPOPT (https://github.com/xuy/pyipopt)
#     - scikit-learn GPR (https://scikit-learn.org)
#
#     Simon Scheidegger, 01/19
#======================================================================

import gpr.nonlinear_solver_initial as solver  #solves opt. problems for terminal VF
import gpr.nonlinear_solver_iterate as solviter  #solves opt. problems during VFI
import gpr.interpolation as interpol  #interface to sparse grid library/terminal VF
import gpr.interpolation_iter as interpol_iter  #interface to sparse grid library/iteration
import gpr.postprocessing as post  #computes the L2 and Linfinity error of the model
import numpy as np
import utils


def main(parameters):
    #======================================================================
    # Start with Value Function Iteration

    for i in range(parameters.numstart, parameters.numits):
        # terminal value function
        if (i == 1):
            print("start with Value Function Iteration")
            interpol.GPR_init(i,parameters)

        else:
            print("Now, we are in Value Function Iteration step", i)
            interpol_iter.GPR_iter(i,parameters)

    #======================================================================
    print("===============================================================")
    print(" ")
    print(" Computation of a growth model of dimension ", parameters.n_agents,
          " finished after ", parameters.numits, " steps")
    print(" ")
    print("===============================================================")
    #======================================================================

    # compute errors
    avg_err = post.ls_error(parameters.n_agents, parameters.numstart,
                            parameters.numits,
                            parameters.No_samples_postprocess)

    #======================================================================
    print("===============================================================")
    print(" ")
    #print " Errors are computed -- see error.txt"
    print(" ")
    print("===============================================================")
    #======================================================================


if __name__ == '__main__':
    params = utils.load_yaml('./model_specs/simon_gpr.yaml')
    params = utils.struct_factory('gpr_params',params['parameters'])
    main(params)