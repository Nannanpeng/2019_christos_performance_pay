# Utility functions for optimization

from scipy.optimize import fminbound

# Maximize function V on interval [a,b]
def maximum(V, a, b):
    return float(V(fminbound(lambda x: -V(x), a, b)))

# Return Maximizer of function V on interval [a,b]
def maximizer(V, a, b):
    return float(fminbound(lambda x: -V(x), a, b))
