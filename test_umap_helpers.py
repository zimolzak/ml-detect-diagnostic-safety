import numpy as np
from umap_helpers import points_near

MY_ARY = np.array([[4, 4], [4, 5], [4, 6], [5, 4], [5, 5], [5, 6], [6, 4], [6, 5], [6, 6], [4.5, 4.5]])
#                    0       1       2       3       4


def test_points_near():
    my_k = 4
    my_indices = points_near(5, 5, MY_ARY, k=my_k)
    n_points_returned = my_indices.shape[0]
    assert n_points_returned <= my_k
    assert my_indices[0] == 4  # Point 4 in MY_ARY is [5,5] so distance 0.
