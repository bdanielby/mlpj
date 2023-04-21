"""
Utilities and convenience functions for using `numpy`.
"""
import numpy as np
import pandas as pd
import numba


def is_numerical(ser):
    """Does a series contain numerical values?

    Args:
      ser (pd.Series): input series

    Returns:
      bool: whether the series can be used for numerical purposes
    """
    return ser.dtype.kind in "bif"
