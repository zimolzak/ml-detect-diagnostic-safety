import numpy as np
from umap_helpers import points_near


def test_points_near():
    my_ary = np.array([[4, 4], [4, 5], [4, 6], [5, 4], [5, 5], [5, 6], [6, 4], [6, 5], [6, 6], [4.5, 4.5]])
    print()
    points_near(5, 5, my_ary)
