import numpy as np


def points_near(x0: float, y0: float, point_array: np.ndarray, k: int = 10) -> np.ndarray:
    """Take a numpy array (n * 2) of ordered pairs. Figure out which are closest to specified x0, y0.
    Note: Returns only the indices (not the points themselves, not the distances).

    :param x0: x-coordinate of center of neighborhood to select.
    :param y0: y-coordinate of center of neighborhood to select.
    :param point_array: Numpy array of points (x is column 0, y is column 1).
    :param k: How many nearest points we want.
    :return: Array of indices of the nearest points.
    """

    x0y0 = np.broadcast_to(np.array([x0, y0]), point_array.shape)
    distances = np.linalg.norm(x0y0 - point_array, axis=1)
    nearest_indices = distances.argsort()
    return nearest_indices[:k]
