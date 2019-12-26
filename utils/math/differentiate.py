import numpy as np
import scipy.optimize as opt

def test_inf_nan(val):
    if sum(np.isinf(val)) > 0 or sum(np.isnan(val)) > 0:
        return True
    return False

def derivative_fd(F,X,h = 1e-4):
    N = len(X)
    D = np.zeros(N,float)
    
    for ixN in range(N):
        dX = np.copy(X)
        dX[ixN] = X[ixN] + h
        F2 = F(dX)

        if (dX[ixN] - h >= 0):
            dX[ixN] = X[ixN] - h
            F1 = F(dX)
            D[ixN] = (F2 - F1) / (2.0 * h)
        else:
            dX[ixN] = X[ixN]
            F1 = F(dX)
            D[ixN] = (F2 - F1) / h

    if test_inf_nan(D):
        import pdb; pdb.set_trace()

    return D


def jacobian_fd(F,X,M,eps=1e-8):
    N = len(X)
    NZ = M * N
    D = np.empty(NZ, float)

    # Finite Differences

    for ixM in range(M):
        F_i = lambda x : F(x,constraint=ixM)
        D[ixM*N:(ixM+1)*N] = approx_fprime(X,F_i, eps)

    # for ixM in range(M):
    #     for ixN in range(N):
    #         dX = np.copy(X)
    #         dX[ixN] = dX[ixN] + h
    #         F2 = F(dX)
    #         D[ixN + ixM * N] = (F2[ixM] - F1[ixM]) / h

    # if test_inf_nan(D):
    #     import pdb; pdb.set_trace()

    return D

# Jacobian sparsity pattern
def dense_jacobian(N, M):
    NZ = M * N
    ACON = np.empty(NZ, int)
    AVAR = np.empty(NZ, int)
    for iuM in range(M):
        for ixN in range(N):
            ACON[ixN + (iuM) * N] = iuM
            AVAR[ixN + (iuM) * N] = ixN

    return (ACON, AVAR)


def dense_hessian(N):
    NZ = (N**2 - N) // 2 + N
    A1 = np.empty(NZ, int)
    A2 = np.empty(NZ, int)
    idx = 0
    for ixI in range(N):
        for ixJ in range(ixI + 1):
            A1[idx] = ixI
            A2[idx] = ixJ
            idx += 1

    return (A1, A2)