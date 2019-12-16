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


if __name__ == '__main__':
    # PLOTTING CONFIG
    sns.set(context='paper',style="darkgrid",rc=plot_style)
    V_T = None
    with open('./out/DC_Simple_2019.12.16_18.02.47/restart_20.pcl', 'rb') as fd:
        V_T = pickle.load(fd)

    # SETUP
    X = np.linspace(0,500,num=200)
    # import pdb; pdb.set_trace()
    Y = V_T(X.reshape(-1,1),maximum=True)
    data = {'Consumption': X, 'Value': Y} 
  
    # Create DataFrame 
    df = pd.DataFrame(data)
    ax = sns.lineplot(x='Consumption', y='Value', data=df)
    plt.show()


    # plt.savefig(plot_out)
    # plt.close()