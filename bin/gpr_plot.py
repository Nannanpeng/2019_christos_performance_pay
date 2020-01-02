import sys
sys.path.append('./')  # allows it to be run from parent dir
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sklearn.gaussian_process.kernels as kr

from estimator.gpr_dc import GPR_DC
import utils.plot as up


def load_values(location):
    with open(location, 'rb') as fd:
        X = pickle.load(fd)
        return X


def load_and_fit(directory, t, xname, yname, kernels):
    load_fstr = './out/%s/%s_%d.pcl'
    X = load_values(load_fstr % (directory, xname, t))
    y = load_values(load_fstr % (directory, yname, t))

    V_t = GPR_DC(2)
    V_t.fit(X, y)
    V_t.eval()
    return X, y, V_t

def load_fit_policy(directory,t):
    rqa = {'length_scale_bounds': (10, 1e5), 'alpha_bounds': (1e-5, 10)}
    rqa = {'length_scale_bounds': (10, 1e5), 'alpha_bounds': (1e-5, 10)}
    essa = {'length_scale_bounds': (1, 1e5), 'periodicity_bounds': (10, 1e5)}
    kernels = [
        kr.RationalQuadratic(**rqa)
        # + kr.DotProduct()
        # + kr.ConstantKernel() * kr.DotProduct() * kr.ExpSineSquared(**essa)
        for i in range(2)
    ]
    X, y, V = load_and_fit(directory, t, 'X', 'y_u', kernels)
    print(V)
    return X,y,V

def plot_policy(X,y,V, dc):
    up.plot_function(V, 0.01, 500, 'Assets', 'Consumption', dc, False)
    up.plot_vals(X, y, 'Assets', 'Consumption', dc, False)

def plot_value(directory, t):
    rqa = {'length_scale_bounds': (10, 1e5), 'alpha_bounds': (1e-5, 10)}
    rqa = {'length_scale_bounds': (10, 1e5), 'alpha_bounds': (1e-5, 10)}
    essa = {'length_scale_bounds': (1, 1e5), 'periodicity_bounds': (10, 1e5)}
    kernels = [
        kr.RationalQuadratic(**rqa)
        # + kr.DotProduct()
        # + kr.ConstantKernel() * kr.DotProduct() * kr.ExpSineSquared(**essa)
        for i in range(2)
    ]
    X, y, V = load_and_fit(directory, t, 'X', 'y_f', kernels)
    print(V)
    up.plot_function(V, 0.01, 500, 'Assets', 'Value', None, True)
    up.plot_vals(X, y, 'Assets', 'Value', None, True)


if __name__ == '__main__':
    sns.set(context='paper', style="darkgrid", rc=up.plot_style)
    directory = 'DC_Simple_2020.01.01_21.32.03'
    t = 20
    fig = plt.figure()
    figManager = plt.get_current_fig_manager()
    # Plot Value Function
    ax = plt.subplot(2, 2, 1)
    ax.set_title('Value Function')
    plot_value(directory, t)
    # Fit policy
    X, y, V = load_fit_policy(directory, t)
    # Plot Policy Worker
    ax = plt.subplot(2, 2, 3)
    ax.set_title('Policy for Worker')
    plot_policy(X,y,V,0)
    # Plot Policy Retired
    ax = plt.subplot(2, 2, 4)
    ax.set_title('Policy for Retiree')
    plot_policy(X,y,V,1)

    # display plot
    fig.tight_layout()
    figManager.window.showMaximized()
    plt.show()