import logging
logger = logging.getLogger(__name__)
import numpy as np
import math

from models import State
from utils import math as um
from .approximation import Multi_LI_VFA


def solve(model):
    ########################################
    # Step 0 - Setup

    # Define grid over continuous portion of state space
    z_grid = np.linspace(-10,10,3) # on a log scale, variance of noise is 0.06. If the process starts at 0, this range seems OK
    h_grid = np.linspace(0,30,3) # human capital, growth is upper bounded by 1 each time period.
    a_grid = np.exp(np.linspace(0,math.log(1.2e6),3))
    grid = um.lspaces_to_grid(z_grid,h_grid,a_grid)

    # Value Function Estimate
    vfa = Multi_LI_VFA(model.T,(9,2,2),grid)

    ########################################
    # Step 1 - Estimate Terminal values
    # Estimate action-values for terminal state
    for ds in model.discrete_space_iter():
        values = np.array([model.reward(State(model.T,ds,p),None,None) for p in grid])
        vfa.add_values(model.T,ds,values)

    import pdb; pdb.set_trace()


    return vfa
