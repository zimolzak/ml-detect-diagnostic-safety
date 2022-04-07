import numpy as np


def repair_icd(icd) -> str:
    """Take a "funny" ICD code composed of numbers, possibly letters, but no dot.
    Return a string w/ the dot in the proper place, given some assumptions

    :param icd: An ICD code without a dot, like 4011. Allow int or str.
    :return: Properly dotted ICD code like 401.1 or H81.01

    :raises: ValueError if icd param isn't an int or str, or if it already contains dots.
    """
    if type(icd) != int and type(icd) != str:
        raise ValueError("Parameter icd has wrong type: %s" % type(icd))
    icd_str = str(icd)
    if '.' in icd_str:
        raise ValueError("There is already a dot '.' in icd: %s" % icd_str)
    if len(icd_str) > 3:
        return icd_str[0:3] + '.' + icd_str[3:]
    else:
        return icd_str


def points_near(point_of_interest: np.ndarray, point_array: np.ndarray, k: int = 10) -> np.ndarray:
    """Take a numpy array representing N points. Figure out which are closest to a single specified point.
    Note: Returns only the indices (not the points themselves, not the distances).

    Usually point_array would be shape (N, 2) and point_of_interest would be shape (2), representing points in 2-D
    plane. But should work dimensions other than 2 (any number of matrix columns).

    :param point_of_interest: Numpy array to specify a single point (center of neighborhood to select).
    :param point_array: Numpy array of points (x is column 0, y is column 1).
    :param k: How many nearest points we want.
    :return: Array of indices of the nearest points.

    :raises: ValueError if point_array has wrong num of axes, or if dimensions don't match in the 2 params.
    """

    if point_array.ndim != 2:
        raise ValueError("point_array has %i axes. Should have exactly 2." % point_array.ndim)
    if point_array.shape[1] != point_of_interest.shape[-1]:
        raise ValueError("Mismatch: point_array has %i dimensional points vs point_of_interest %i" %
                         (point_array.shape[1], point_of_interest.shape[-1]))

    poi_repeated = np.broadcast_to(point_of_interest, point_array.shape)
    distances = np.linalg.norm(poi_repeated - point_array, axis=1)
    nearest_indices = distances.argsort()
    return nearest_indices[:k]
