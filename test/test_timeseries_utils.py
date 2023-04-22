"""
Unit tests for `mlpj.timeseries_utils`.
"""
import numpy as np
import pandas as pd
import pandas.testing as pd_testing

from mlpj import python_utils as pu
from mlpj import pandas_utils as pdu
from mlpj import timeseries_utils as tsu


def test_remove_last_n_days():
    df = pdu.from_items([
        ('date', pd.to_datetime(
            [pu.n_days_ago(20), pu.n_days_ago(5), pu.n_days_ago(10)])),
        ('a', [2, 3, 1])
    ])
    
    df1 = tsu.remove_last_n_days(df, 'date', 10)
    pdu.assert_frame_contents_equal(
        df1,
        pdu.from_items([
            ('date', pd.to_datetime([pu.n_days_ago(20), pu.n_days_ago(10)])),
            ('a', [2, 1])
        ]))

    
def test_ts_train_test_split():
    df = pdu.from_items([
        ('date', pd.to_datetime(['2023-03-05', '2023-01-15', '2023-02-01'])),
        ('a', [2, 3, 1]),
        ('y', [0, 1, -1]),
    ])

    X_train, y_train, X_test, y_test = tsu.ts_train_test_split(
        df, '2023-02-01', 'date', 'y')

    pd_testing.assert_frame_equal(
        X_train,
        pdu.from_items([
            ('date', pd.to_datetime(['2023-01-15'])),
            ('a', [3]),
            ('y', [1]),
        ], index=[1]))

    np.testing.assert_array_equal(y_train, [1])

    pd_testing.assert_frame_equal(
        X_test,
        pdu.from_items([
            ('date', pd.to_datetime(['2023-03-05', '2023-02-01'])),
            ('a', [2, 1]),
            ('y', [0, -1]),
        ], index=[0, 2]))

    np.testing.assert_array_equal(y_test, [0, -1])
