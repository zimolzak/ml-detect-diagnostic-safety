import numpy as np
from helpers import points_near, repair_icd
import pytest

EXAMPLE_POINTS = np.array([[4, 4], [4, 5], [4, 6], [5, 4], [5, 5], [5, 6], [6, 4], [6, 5], [6, 6], [4.5, 4.5]])
#                            0       1       2       3       4                               8         9


def test_points_near():
    indices = points_near(np.array([5, 5]), EXAMPLE_POINTS, 5)
    assert indices[0] == 4  # Point 4 is [5,5] so distance 0.
    indices = points_near(np.array([4.6, 4.6]), EXAMPLE_POINTS)
    assert indices[0] == 9

    for my_k in range(100):
        indices = points_near(np.array([-7.5, -10.2]), EXAMPLE_POINTS, k=my_k)
        n_points_returned = indices.shape[0]
        assert n_points_returned <= my_k
        if my_k > 0:
            assert indices[0] == 0  # Point 0 is [4,4] so closest to -7.5, -10.2


def test_points_near_exception():
    with pytest.raises(ValueError):
        dummy = points_near(np.array([5]), EXAMPLE_POINTS)
        # That is bad because [5] gets broadcast to shape (10, 2).
        # In other words, it silently changes [5] to [5,5] which is probably a surprise


ICD_EXPECTED = {4010: '401.0',
                40200: '402.00',
                4031: '403.1',
                24900: '249.00',
                7915: '791.5',
                '7915': '791.5',
                436: '436',  # tricky tricky
                'G43601': 'G43.601',
                'I6000': 'I60.00',
                'I604': 'I60.4',
                'I63032': 'I63.032',
                'I10': 'I10',  # also tricky
                }


def test_repair_icd():
    for k, v in ICD_EXPECTED.items():
        assert repair_icd(k) == v


def test_repair_icd_exception_float():
    with pytest.raises(ValueError):
        dummy = repair_icd(456.456)


def test_repair_icd_exception_dot():
    with pytest.raises(ValueError):
        dummy = repair_icd('456.456')


def test_repair_icd_exception_list():
    with pytest.raises(ValueError):
        dummy = repair_icd([1, 2, 3])


def test_repair_icd_exception_short():
    with pytest.raises(ValueError):
        dummy = repair_icd(59)
