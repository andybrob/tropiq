import numpy as np


def allpairs_shortest_path(W: np.ndarray) -> np.ndarray:
    """
    All-pairs shortest path via Floyd-Warshall in the min-plus tropical semiring.

    Args:
        W: (N, N) cost matrix. W[i, j] is the cost of the direct edge from i to j.
           Use np.inf for pairs with no direct edge.
           W[i, i] should be 0.

    Returns:
        (N, N) distance matrix D where D[i, j] is the cost of the shortest
        path from i to j.
    """
D = W.copy()
    N = D.shape[0]
    for k in range(N):
        D = np.minimum(D, D[:, k:k+1] + D[k:k+1, :])
    return D
