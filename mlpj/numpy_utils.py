"""
Utilities and convenience functions for using `numpy`.
"""
import numpy as np


def anynan(x):
    """Does the input contain any NaN values?

    Args:
        x (array-like): input array
    Returns:
        bool: whether any NaNs were found
    """
    return np.isnan(x).any()


def digitize(x, bins):
    """An alternative version `np.digitize`. It puts x values greater or
    equal to `bins.max()` on the last index of `bins`. Values smaller than
    `bins.min()` are put on the first index of `bins`.

    If values in `x` are such that they fall outside the bin range,
    attempting to index `bins` with the indices that `digitize` returns
    would result in an IndexError.

    Args:
        x (array-like, 1dim): input array to be binned
        bins (array-like, 1dim): array of bins, must be monotonic
    Returns:
        `np.ndarray`: array of indices; it has the same shape as `x`.
    
    Raises:
        ValueError: If the input is not 1-dimensional, or if `bins` is not monotonic.
        TypeError: If the type of the input is complex.
    """
    bin_numbers = np.digitize(x, bins)
    is_max = x >= bins.max()
    bin_numbers[is_max] -= 1
    return bin_numbers
