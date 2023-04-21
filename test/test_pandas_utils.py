"""
Unit tests for `mlpj.pandas_utils`.
"""
import numpy as np
import pandas as pd
import numba

from mlpj import pandas_utils as pdu

nan = np.nan


def test_is_numerical():
    assert pdu.is_numerical(pd.Series([3, 4]))
    assert pdu.is_numerical(pd.Series([3.5, 4]))
    assert pdu.is_numerical(pd.Series([True, False]))
    assert pdu.is_numerical(pd.Series([3.5, 4, nan]))
    assert not pdu.is_numerical(pd.Series(["foo", "bar"]))
