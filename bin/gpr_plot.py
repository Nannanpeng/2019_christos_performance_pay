import sys
sys.path.append('./')  # allows it to be run from parent dir
import logging
logger = logging.getLogger(__name__)
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sklearn.gaussian_process.kernels as kr

from estimator.gpr_dc import GPR_DC
import utils.plot as up

rqa = {'length_scale_bounds': (10, 1e5), 'alpha_bounds': (1e-5, 10)}
rqa = {'length_scale_bounds': (10, 1e5), 'alpha_bounds': (1e-5, 10)}
essa = {'length_scale_bounds': (1, 1e5), 'periodicity_bounds': (10, 1e5)}

def plot_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_directory",
                        type=str,
                        help="Directory with checkpoints to use as input")
    parser.add_argument("time",
                        type=int,
                        help="Time step to fit models")
    parser.add_argument("-m",'--max_iter',
                        type=int,
                        default=1500,
                        help="Number of iters for fitting GPR")
    parser.add_argument("-l",'--learning_rate',
                    type=float,
                    default=0.1,
                    help="Learning Rate to use")

    parser.add_argument("-d", "--debug", action="store_true", help="debug")
    parser.add_argument("-p", "--policy", action="store_true", help="plot policy only")
    parser.add_argument("-v", "--value", action="store_true", help="plot value function only")

    return parser


def load_values(location):
    with open(location, 'rb') as fd:
        X = pickle.load(fd)
        return X


def load_and_fit(directory, t, xname, yname, kernels = None, num_iter = None, lr = None):
    load_fstr = './%s/%s_%d.pcl'
    X = load_values(load_fstr % (directory, xname, t))
    y = load_values(load_fstr % (directory, yname, t))

    # prep kwargs
    kwargs = {}
    if num_iter is not None:
        kwargs['num_iter'] = num_iter
    if lr is not None:
        kwargs['lr'] = lr

    # create estimator and fit
    V_t = GPR_DC(2)
    V_t.fit(X, y,**kwargs)
    V_t.eval()
    return X, y, V_t

def load_fit_policy(directory,t, num_iter, lr):
    rqa = {'length_scale_bounds': (10, 1e5), 'alpha_bounds': (1e-5, 10)}
    rqa = {'length_scale_bounds': (10, 1e5), 'alpha_bounds': (1e-5, 10)}
    essa = {'length_scale_bounds': (1, 1e5), 'periodicity_bounds': (10, 1e5)}
    kernels = [
        kr.RationalQuadratic(**rqa)
        # + kr.DotProduct()
        # + kr.ConstantKernel() * kr.DotProduct() * kr.ExpSineSquared(**essa)
        for i in range(2)
    ]
    X, y, V = load_and_fit(directory, t, 'X', 'y_u', kernels, num_iter, lr)
    print(V)
    return X,y,V

def plot_policy(X,y,V, dc):
    up.plot_function(V, 0.01, 500, 'Assets', 'Consumption', dc, False)
    up.plot_vals(X, y, 'Assets', 'Consumption', dc, False)

def plot_value(directory, t, num_iter, lr):
    kernels = [
        kr.RationalQuadratic(**rqa)
        # + kr.DotProduct()
        # + kr.ConstantKernel() * kr.DotProduct() * kr.ExpSineSquared(**essa)
        for i in range(2)
    ]
    X, y, V = load_and_fit(directory, t, 'X', 'y_f', kernels, num_iter, lr)
    print(V)
    up.plot_function(V, 0.01, 500, 'Assets', 'Value', None, True)
    up.plot_vals(X, y, 'Assets', 'Value', None, True)


def _config(args):
    sns.set(context='paper', style="darkgrid", rc=up.plot_style)
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=logging.DEBUG)
    make_value, make_policy = False, False
    if not (args.policy or args.value):
        make_policy = make_value = True
    else:
        make_value, make_policy = args.value, args.policy

    return make_value,make_policy

if __name__ == '__main__':
    argparser = plot_argparser()
    args = argparser.parse_args()
    make_value,make_policy = _config(args)
    directory = args.checkpoint_directory
    t = args.time
    log_str = 'CREATING PLOTS\r\n\tDirectory: %s\r\n\tTime: %d\r\n\tFit Policy? %s\r\n\tFit Value? %s'
    logger.info(log_str % (directory,t,make_policy,make_value))


    fig = plt.figure()
    figManager = plt.get_current_fig_manager()

    if make_value:
        logger.info('Fitting Value Function')
        ax = plt.subplot(2, 2, 1)
        ax.set_title('Value Function')
        plot_value(directory, t, args.max_iter, args.learning_rate)

    if make_policy:
        logger.info('Fitting Policy Function')
        X, y, V = load_fit_policy(directory, t, args.max_iter, args.learning_rate)

        # Plot Policy Worker
        ax = plt.subplot(2, 2, 3)
        ax.set_title('Policy for Worker')
        plot_policy(X,y,V,1)

        # Plot Policy Retired
        ax = plt.subplot(2, 2, 4)
        ax.set_title('Policy for Retiree')
        plot_policy(X,y,V,0)

    # display plot
    fig.tight_layout()
    figManager.window.showMaximized()
    plt.show()

    # fig = plt.figure()
    # figManager = plt.get_current_fig_manager()
    # # Plot Value Function
    # # ax = plt.subplot(2, 2, 1)
    # # ax.set_title('Value Function')
    # # plot_value(directory, t)
    # # Fit policy
    # X, y, V = load_fit_policy(directory, t, lr=0.01)
    # # Plot Policy Worker
    # ax = plt.subplot(2, 2, 3)
    # ax.set_title('Policy for Worker')
    # plot_policy(X,y,V,1)
    # # Plot Policy Retired
    # ax = plt.subplot(2, 2, 4)
    # ax.set_title('Policy for Retiree')
    # plot_policy(X,y,V,0)

    # # display plot
    # fig.tight_layout()
    # figManager.window.showMaximized()
    # plt.show()