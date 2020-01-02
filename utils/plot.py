# import matplotlib and suppress logs
import matplotlib.pyplot as plt
import logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
import pandas as pd
import numpy as np
import seaborn as sns
import torch

# local
plot_style = {
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'font.size': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'serif',
    'font.serif': 'Computer Modern Roman',
    'ps.useafm': True,
    'pdf.use14corefonts': True,
    'text.usetex': True
}

def plot_vals(X, Y, xlabel, ylabel,which,maximum):
    if maximum:
        Y = np.amax(Y,axis=1).reshape(-1,)
    else:
        Y = Y[:,which]
    X = X.reshape(-1,)
    data = {}
    data[xlabel] = X
    data[ylabel] = Y

    # Create DataFrame
    df = pd.DataFrame(data)
    ax = sns.scatterplot(x=xlabel, y=ylabel, data=df)

def plot_function(V,xmin,xmax,xlabel,ylabel,which,maximum):
    X = np.linspace(xmin, xmax, num=200)
    Y = None
    if maximum:
        Y = V(X.reshape(-1, 1), maximum=True)
    else:
        Y = V(X.reshape(-1, 1))[:,which]
    Y = Y if not isinstance(Y,torch.Tensor) else Y.detach().numpy()
    data = {}
    data[xlabel] = X
    data[ylabel] = Y
    df = pd.DataFrame(data)
    ax = sns.lineplot(x=xlabel, y=ylabel, data=df)

def create_1D(V, X_train, Y_train, xmin, xmax,xlabel,ylabel,plot_out,which=None,maximum=False):
    sns.set(context='paper', style="darkgrid", rc=plot_style)
    assert not (which is None and not maximum), "must specify either which index or to take maximum"
    plot_function(V,xmin,xmax,xlabel,ylabel,which,maximum)
    plot_vals(X_train,Y_train,xlabel,ylabel,which,maximum)
    plt.savefig(plot_out)
    plt.close()
