"""
Unit tests for `mlpj.numpy_utils`.
"""
import numpy.testing
import numpy as np

from mlpj import numpy_utils as npu


def test_anynan():
    x = np.array([np.nan, 0, 1, 10, 14])
    assert npu.anynan(x)

    x[0] = 6
    assert not npu.anynan(x)


def test_digitize():
    x = np.array([-1000, -0.2, 0.2, 6.4, 3.0, 10, 11, 1000])
    bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
    
    np.testing.assert_equal(
        npu.digitize(x, bins), np.array([0, 0, 1, 4, 3, 4, 4, 4]))
