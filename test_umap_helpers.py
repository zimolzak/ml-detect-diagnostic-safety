import numpy as np
from umap_helpers import points_near

EXAMPLE_POINTS = np.array([[4, 4], [4, 5], [4, 6], [5, 4], [5, 5], [5, 6], [6, 4], [6, 5], [6, 6], [4.5, 4.5]])
#                    0       1       2       3       4                               8         9


def test_points_near():
    indices = points_near(5, 5, EXAMPLE_POINTS, 5)
    assert indices[0] == 4  # Point 4 is [5,5] so distance 0.
    indices = points_near(4.6, 4.6, EXAMPLE_POINTS)
    assert indices[0] == 9

    for my_k in range(100):
        indices = points_near(-7.5, -10.2, EXAMPLE_POINTS, k=my_k)
        n_points_returned = indices.shape[0]
        assert n_points_returned <= my_k
        if my_k > 0:
            assert indices[0] == 0  # Point 0 is [4,4] so closest to -7.5, -10.2
