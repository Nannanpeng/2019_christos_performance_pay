# from . import utility
# from . import algorithm
# from . import approximation
from collections import namedtuple

# __all__ = ['SimplePPModel','GenericPPModel','utility','dynamics','stats']

IPOptCallback = namedtuple('IPOptCallback', [
    'ev_f', 'ev_grad_f', 'ev_g', 'ev_jac_g', 'hess_sparsity', 'jac_g_sparsity'
])

