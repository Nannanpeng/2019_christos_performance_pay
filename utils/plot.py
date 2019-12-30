import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

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

def create_1D(V, xmin, xmax,ylabel,xlabel,plot_out):
    X = np.linspace(xmin, xmax, num=200)
    Y = V(X.reshape(-1, 1), maximum=True)
    data = {}
    data[xlabel] = X
    data[ylabel] = Y
    df = pd.DataFrame(data)
    ax = sns.lineplot(x=xlabel, y=ylabel, data=df)
    plt.savefig(plot_out)
    plt.close()
