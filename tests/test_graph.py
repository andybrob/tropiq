import numpy as np
from tropiq.graph import allpairs_shortest_path


def test_allpairs_basic():
    """
    4-node graph:
      0 -1-> 1 -2-> 3
      0 -4-> 2 -1-> 3

    Shortest path from 0 to 3:
      via 1: 1 + 2 = 3
      via 2: 4 + 1 = 5
      answer: 3
    """
    inf = np.inf
    W = np.array([
        [0,   1,   4,   inf],
        [inf, 0,   inf, 2  ],
        [inf, inf, 0,   1  ],
        [inf, inf, inf, 0  ],
    ], dtype=float)

    D = allpairs_shortest_path(W)

    assert D[0, 3] == 3.0
    assert D[0, 1] == 1.0
    assert D[1, 3] == 2.0
    assert D[2, 3] == 1.0
    assert D[0, 0] == 0.0
