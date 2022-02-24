import numpy as np


def points_near(x0, y0, point_array, k=10):
    """Take a numpy array (n * 2) of ordered pairs. Figure out which are closest to specified x, y.

    :param x0: x-coordinate of center of neighborhood to select
    :param y0: y-coordinate of center of neighborhood to select
    :param point_array: Numpy array of points
    :param k: How many nearest points we want.
    """

    n_total_points = point_array.shape[0]
    x0y0 = np.tile(np.array([x0, y0]), [n_total_points, 1])
    dists = np.linalg.norm(x0y0 - point_array, axis=1)
    indices = dists.argsort()
    return indices[:k]
