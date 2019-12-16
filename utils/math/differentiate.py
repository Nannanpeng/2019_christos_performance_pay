import numpy as np

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

    return D


def jacobian_fd(F,X,N,M):
    assert N == len(X), "X is not same dimension as N"
    N = len(X)
    NZ = M * N
    D = np.empty(NZ, float)

    # Finite Differences
    h = 1e-4
    F1 = F(X)

    for ixM in range(M):
        for ixN in range(N):
            dX = np.copy(X)
            dX[ixN] = dX[ixN] + h
            F2 = F(dX)
            D[ixN + ixM * N] = (F2[ixM] - F1[ixM]) / h
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