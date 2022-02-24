import numpy as np


def points_near(point_of_interest: np.ndarray, point_array: np.ndarray, k: int = 10) -> np.ndarray:
    """Take a numpy array representing N points. Figure out which are closest to a single specified point.
    Note: Returns only the indices (not the points themselves, not the distances).

    Usually point_array would be shape (N, 2) and point_of_interest would be shape (2), representing points in 2-D
    plane. But should work dimensions other than 2 (any number of matrix columns).

    :param point_of_interest: Numpy array to specify a single point (center of neighborhood to select).
    :param point_array: Numpy array of points (x is column 0, y is column 1).
    :param k: How many nearest points we want.
    :return: Array of indices of the nearest points.
    """

    # fixme - should probably check assumptions about .shape and .ndim of inputs.

    poi_repeated = np.broadcast_to(point_of_interest, point_array.shape)
    distances = np.linalg.norm(poi_repeated - point_array, axis=1)
    nearest_indices = distances.argsort()
    return nearest_indices[:k]
