import logging
logger = logging.getLogger(__name__)
import numpy as np
import math
import random

from models import State
from utils import math as um
from .approximation import Multi_LI_VFA, Multi_LI_QFA



def random_grid_sample(num_points,grid):
    rows = random.sample(range(0,grid.shape[0]), num_points)
    return rows,grid[rows]


def simulate_emax(model,points,t,R,vfa):
    noise = model.generate_noise(R)
    import pdb; pdb.set_trace()

    return None

def solve(model):
    ########################################
    # Step 0 - Setup
    R = 100 # number of noise samples at each grid point where emax is simulated
    num_sp = 10 # number locations to sample for emax
    # Define grid over continuous portion of state space
    z_grid = np.linspace(-10,10,3) # on a log scale, variance of noise is 0.06. If the process starts at 0, this range seems OK
    h_grid = np.linspace(0,30,3) # human capital, growth is upper bounded by 1 each time period.
    a_grid = np.exp(np.linspace(0,math.log(1.2e6),3))
    grid = um.lspaces_to_grid(z_grid,h_grid,a_grid)

    # State-Action Value Function Estimates
    vfa = Multi_LI_QFA(model.T,(9,2,2),model.num_discrete_actions,grid)

    ########################################
    # Step 1 - Estimate Terminal values
    # Estimate action-values for terminal state
    for k in range(model.num_discrete_actions):
        for ds in model.discrete_space_iter():
            values = np.array([model.reward(State(model.T,ds,p),None,None) for p in grid])
            vfa.add_values(model.T,ds,k,values)

    for t in reversed(range(model.T)):
        rows,sample_points = random_grid_sample(num_sp,grid)
        emax_sim = simulate_emax(model,sample_points,t,R,vfa)
        values = np.zeros(grid.shape) # allocate empty values array


    return vfa
