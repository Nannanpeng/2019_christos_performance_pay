

def fit_lr_simulated_emax():
    pass




class MultiDiscretizationLI(object):
    """
    Grid should have one less dimension than vals

    Performs linear interpolation
    """
    def __init__(self,grid,values):
        self._num_grids = values.shape[0]
        self.discretizations = [LinearNDInterpolator(grid,values[i,]) for i in range(self._num_grids)]

    def eval(self,idx,points):
        assert idx < self._num_grids, "idx is out of range"
        return self.discretizations[idx]
