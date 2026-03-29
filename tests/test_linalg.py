import numpy as np
from tropiq.linalg import matvec


def test_matvec_basic():
    A = np.array([[1, 4, 2],
                  [5, 1, 0]], dtype=np.float64)
    x = np.array([0, 1, 3], dtype=np.float64)
    expected_y = np.array([5.0, 5.0], dtype=np.float64)
    y = matvec(A, x)
    np.testing.assert_array_equal(y, expected_y)
