import numpy as np
from tropiq import _core


def matvec(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Max-plus matrix-vector product: y[i] = max_k(A[i,k] + x[k])"""
    M = A.shape[0]
    K = A.shape[1]
    A_flat = A.flatten().tolist()
    x_list = x.tolist()
    y_list = _core.maxplus_matvec(A_flat, x_list, M, K)
    return np.array(y_list)
    
