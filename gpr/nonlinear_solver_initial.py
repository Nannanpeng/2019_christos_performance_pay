#======================================================================
#
#     This routine interfaces with IPOPT
#     It sets the optimization problem for every training point
#     at the beginning of the VFI.
#
#     Simon Scheidegger, 11/16 ; 07/17; 01/19
#======================================================================

# from parameters import *
from .ipopt_wrapper import EV_F, EV_GRAD_F, EV_G, EV_JAC_G, sparsity_hess, sparsity_jac_g
import numpy as np
import ipyopt

#======================================================================


def initial(k_init, n_agents, params):
    # IPOPT PARAMETERS below

    N = 3 * n_agents  # number of vars
    M = 3 * n_agents + 1  # number of constraints

    # Vector of variables -> solution of non-linear equation system
    X = np.empty(N)

    LAM = np.empty(M)  # multipliers
    G = np.empty(M)  # (in-)equality constraints

    # Vector of lower and upper bounds
    G_L = np.empty(M)
    G_U = np.empty(M)

    X_L = np.empty(N)
    X_U = np.empty(N)

    Z_L = np.empty(N)
    Z_U = np.empty(N)

    # get coords of an individual grid points
    grid_pt_box = k_init
    X_L[:n_agents] = params.c_bar
    X_U[:n_agents] = params.c_up

    X_L[n_agents:2 * n_agents] = params.l_bar
    X_U[n_agents:2 * n_agents] = params.l_up

    X_L[2 * n_agents:3 * n_agents] = params.inv_bar
    X_U[2 * n_agents:3 * n_agents] = params.inv_up

    # Set bounds for the constraints
    G_L[:n_agents] = params.c_bar
    G_U[:n_agents] = params.c_up

    G_L[n_agents:2 * n_agents] = params.l_bar
    G_U[n_agents:2 * n_agents] = params.l_up

    G_L[2 * n_agents:3 * n_agents] = params.inv_bar
    G_U[2 * n_agents:3 * n_agents] = params.inv_up

    G_L[3 * n_agents] = 0.0  # both values set to 0 for equality contraints
    G_U[3 * n_agents] = 0.0

    # initial guesses for first iteration
    cons_init = 0.5 * (X_U[:n_agents] - X_L[:n_agents]) + X_L[:n_agents]
    lab_init = 0.5 * (X_U[n_agents:2 * n_agents] -
                      X_L[n_agents:2 * n_agents]) + X_L[n_agents:2 * n_agents]
    inv_init = 0.5 * (X_U[2 * n_agents:3 * n_agents] -
                      X_L[2 * n_agents:3 * n_agents]) + X_L[2 * n_agents:3 *
                                                            n_agents]

    X[:n_agents] = cons_init
    X[n_agents:2 * n_agents] = lab_init
    X[2 * n_agents:3 * n_agents] = inv_init

    # Create callback functions
    def eval_f(X):
        out = EV_F(X, k_init, n_agents, params)
        return out

    def eval_grad_f(X, out):
        out[()] = EV_GRAD_F(X, k_init, n_agents, params)
        return out

    def eval_g(X, out):
        out[()] = EV_G(X, k_init, n_agents,params)
        return out

    def eval_jac_g(X, out):
        out[()] = EV_JAC_G(X, k_init, n_agents,params)
        return out

    ipyopt.set_loglevel(ipyopt.LOGGING_DEBUG)

    # First create a handle for the Ipopt problem
    nlp = ipyopt.Problem(N, X_L, X_U, M, G_L, G_U, sparsity_jac_g(N, M),
                         sparsity_hess(N), eval_f, eval_grad_f, eval_g,
                         eval_jac_g)

    nlp.set(obj_scaling_factor=-1.00,
            tol=1e-6,
            acceptable_tol=1e-5,
            derivative_test='first-order',
            hessian_approximation="limited-memory")
            # print_level=0)

    # x: Solution of the primal variables
    # z_l, z_u: Solution of the bound multipliers
    # constraint_multipliers: Solution of the constraint multipliers
    # obj: Objective value
    # status: Exit Status
    z_l = np.zeros(N)
    z_u = np.zeros(N)
    constraint_multipliers = np.zeros(M)
    x, obj, status = nlp.solve(X,
                               mult_g=constraint_multipliers,
                               mult_x_L=z_l,
                               mult_x_U=z_u)

    # Unpack Consumption, Labor, and Investment
    c = x[:n_agents]
    l = x[n_agents:2 * n_agents]
    inv = x[2 * n_agents:3 * n_agents]

    to_print = np.hstack((obj, x))

    # == debug ==
    #f=open("results.txt", 'a')
    #np.savetxt(f, np.transpose(to_print) #, fmt=len(x)*'%10.10f ')
    #for num in to_print:
    #    f.write(str(num)+"\t")
    #f.write("\n")
    #f.close()

    return obj, c, l, inv
