import sys
sys.path.append('./') # allows it to be run from parent dir
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import neighbors
import os
import re
import argparse
import pickle
from glob import glob
from estimator.gpr_dc import GPR_DC
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, DotProduct


# local
plot_style = {
   'axes.titlesize' : 14,
   'axes.labelsize': 12,
   'font.size': 12,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'font.family':'serif',
   'font.serif' : 'Computer Modern Roman',
   'ps.useafm' : True,
   'pdf.use14corefonts' : True,
   'text.usetex' : True
}

def load_values(location):
    with open(location,'rb') as fd:
        X = pickle.load(fd)
        return X

def load_and_fit(directory,t):
    load_fstr = './out/%s/%s_%d.pcl'
    X = load_values(load_fstr % (directory,'X',t))
    y_u = load_values(load_fstr % (directory,'y_u',t))
    y_f = load_values(load_fstr % (directory,'y_f',t))
    # kernels = [RBF(length_scale=1) for i in range(2)]
    kernels = [DotProduct() + RBF() for i in range(2)]

    V_t = GPR_DC(2, kernel=kernels)
    V_t.fit(X,y_f)
    return X,y_f,V_t

def plot_vals(X,Y):
    data = {'Assets': X, 'Value': Y} 
  
    # Create DataFrame 
    df = pd.DataFrame(data)
    ax = sns.scatterplot(x='Assets', y='Value', data=df)

def plot_function(V_T):
    X = np.linspace(0,500,num=200)
    Y = V_T(X.reshape(-1,1),maximum=True)
    data = {'Assets': X, 'Value': Y} 
    df = pd.DataFrame(data)
    ax = sns.lineplot(x='Assets', y='Value', data=df)

if __name__ == '__main__':
    sns.set(context='paper',style="darkgrid",rc=plot_style)

    X,y_f,V_T = load_and_fit('DC_Simple_2019.12.29_23.50.58',20)
    plot_vals(X[:,0],y_f[:,0])
    plot_function(V_T)
    plt.show()

    # plt.savefig(plot_out)
    # plt.close()