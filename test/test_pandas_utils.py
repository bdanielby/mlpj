"""
Unit tests for `mlpj.pandas_utils`.
"""
import io

import numpy as np
import pandas as pd
import numba
import pandas.testing as pd_testing
import pytest

from mlpj import pandas_utils as pdu

nan = np.nan


def test_from_items():
    pd_testing.assert_frame_equal(
        pdu.from_items([
            ('b', np.array([3, 8])),
            ('a', ['first', 'second']),
        ], index=[3, 4]),
        pd.DataFrame([[3, 'first'], [8, 'second']],
                     columns=['b', 'a'], index=[3, 4]))

    df = pdu.from_items([
        ('b', np.array([3, 8])),
        ('a', ['first', 'second']),
    ])
    pd_testing.assert_frame_equal(
        df,
        pd.DataFrame([[3, 'first'], [8, 'second']],
                     columns=['b', 'a']))
    assert df.a.dtype == np.dtype('O')
    assert df.b.dtype == np.dtype('int64')

    with pytest.raises(ValueError):
        pdu.from_items([
            ('b', np.array([3, 8, 9])),
            ('a', ['first', 'second']),
        ], index=[3, 4])


def test_wide_display():
    out = io.StringIO()
    with pdu.wide_display():
        n_cols = 50
        df = pd.DataFrame(np.random.randint(low=0, high=3, size=(2, n_cols)),
                          columns=np.arange(n_cols))
        print(df, file=out)
    lines = out.getvalue().splitlines()
    assert len(lines) == 3


def test_is_numerical():
    assert pdu.is_numerical(pd.Series([3, 4]))
    assert pdu.is_numerical(pd.Series([3.5, 4]))
    assert pdu.is_numerical(pd.Series([True, False]))
    assert pdu.is_numerical(pd.Series([3.5, 4, nan]))
    assert not pdu.is_numerical(pd.Series(["foo", "bar"]))


def test_get_colnames():
    df = pdu.from_items([
        ('a', [3, 4, 3, 2]),
        ('b', ['a', 'b', 'b', 'a']),
        ('c', ['x', 'y', 'z', 'x'])
    ])
    np.testing.assert_array_equal(pdu.get_colnames(df), ['a', 'b', 'c'])

    dfg = df.groupby('b')
    np.testing.assert_array_equal(pdu.get_colnames(dfg), ['a', 'b', 'c'])

    np.testing.assert_array_equal(pdu.get_colnames(['x', 'y']), ['x', 'y'])


def test_all_colnames_except():
    X = pd.DataFrame(np.eye(4), columns=['x', 'a', 'ba', 'c'])
    assert pdu.all_colnames_except(X, ['a', 'c']) == ['x', 'ba']

    X = pd.DataFrame(np.eye(4), columns=list('abcd'))
    assert pdu.all_colnames_except(X, ['c', 'a']) == ['b', 'd']
    assert pdu.all_colnames_except(X, ['c', 'a', 'b', 'd']) == []

    
def test_category_colnames():
    df = pdu.from_items([
        ('a', [3, 4, 3, 2]),
        ('b', ['a', 'b', 'b', 'a']),
        ('c', ['x', 'y', 'z', 'x'])
    ])
        
    assert pdu.category_colnames(df) == []
    for colname in ['a', 'b']:
        df[colname] = df[colname].astype('category')
    assert pdu.category_colnames(df) == ['a', 'b']

    assert pdu.category_colnames(df, feature_list=('b', 'c')) == ['b']


def test_rename_column():
    df = pdu.from_items([
        ('a', [3, 4, 3, 2]),
        ('b', ['a', 'b', 'b', 'a']),
    ])
    
    pdu.rename_column(df, 'b', 'b1')
    pd_testing.assert_frame_equal(
        df,
        pdu.from_items([
            ('a', [3, 4, 3, 2]),
            ('b1', ['a', 'b', 'b', 'a']),
        ]))

    with pytest.raises(KeyError):
        pdu.rename_column(df, 'foo', 'bar')


def test_drop_index():
    df = pdu.from_items([
        ('b', np.array([3, 8])),
        ('a', ['first', 'second']),
    ], index=[3, 4])

    pdu.drop_index(df)

    pd_testing.assert_frame_equal(
        df,
        pdu.from_items([
            ('b', np.array([3, 8])),
            ('a', ['first', 'second'])
        ]))

    
def test_drop_columns():
    df = pdu.from_items([
        ('a', [3, 4, 3, 2]),
        ('b', ['a', 'b', 'b', 'a']),
        ('c', ['x', 'y', 'z', 'x'])
    ], index=[3, 4, 5, 8])
    
    df1 = df.copy()
    pdu.drop_columns(df1, 'a')

    pd_testing.assert_frame_equal(
        df1,
        pdu.from_items([
            ('b', ['a', 'b', 'b', 'a']),
            ('c', ['x', 'y', 'z', 'x'])
        ], index=[3, 4, 5, 8]))

    df2 = df.copy()
    pdu.drop_columns(df2, ['a', 'c'])
    pd_testing.assert_frame_equal(
        df2,
        pdu.from_items([
            ('b', ['a', 'b', 'b', 'a']),
        ], index=[3, 4, 5, 8]))
        
    with pytest.raises(KeyError):
        pdu.drop_columns(df, 'x')


def test_columns_to_right():
    df = pdu.from_items([
        ('a', [3, 4, 3, 2]),
        ('b', ['a', 'b', 'b', 'a']),
        ('c', ['x', 'y', 'z', 'x'])
    ], index=[3, 4, 5, 8])

    df1 = pdu.columns_to_right(df, ['b', 'a'])
    pd_testing.assert_frame_equal(
        df1,
        pdu.from_items([
            ('c', ['x', 'y', 'z', 'x']),
            ('b', ['a', 'b', 'b', 'a']),
            ('a', [3, 4, 3, 2]),
        ], index=[3, 4, 5, 8]))


def test_shuffle_df_drop_index():
    df = pd.DataFrame(np.random.random(size=(3, 3)), columns=['a', 'b', 'c'])
    df1 = pdu.shuffle_df_drop_index(df)
    np.testing.assert_array_equal(df1.index.values, np.arange(len(df)))
    np.testing.assert_array_equal(df1.sum(), df.sum())


def test_assert_frame_contents_equal():
    df1 = pdu.from_items([
        ('b', np.array([3, 8])),
        ('a', ['first', 'second']),
    ], index=[3, 4])
    
    df2 = pd.DataFrame([[3, 'first'], [8, 'second']],
                       columns=['b', 'a'])

    with pytest.raises(AssertionError):
        pd_testing.assert_frame_equal(df1, df2)

    pdu.assert_frame_contents_equal(df1, df2)


def test_ser_where_defined():
    x = pd.Series([4, 5, nan, 2, nan])

    pd_testing.assert_series_equal(
        pdu.ser_where_defined(x),
        pd.Series([4., 5, 2], index=[0, 1, 3]))

    
def test_n_undefined_and_percentage():
    x = pd.Series([4, 5, nan, 2, nan])
    
    n, perc = pdu.n_undefined_and_percentage(x)
    assert n == 2
    assert perc == 2 / 5 * 100


def test_colname_list():
    assert pdu.colname_list('foo') == ['foo']
    assert pdu.colname_list(['foo', 'bar']) == ['foo', 'bar']


def test_sort():
    df = pdu.from_items([
        ('a', [3, 4, 3, 2]),
        ('b', ['a', 'b', 'c', 'd']),
        ('c', ['x', 'y', 'z', 'w'])
    ], index=[3, 4, 5, 8])

    df1 = pdu.sort(df, colnames='a', inplace=True)
    assert df1 is df
    
    pd_testing.assert_frame_equal(
        df,
        pdu.from_items([
            ('a', [2, 3, 3, 4]),
            ('b', ['d', 'a', 'c', 'b']),
            ('c', ['w', 'x', 'z', 'y'])
        ], index=[0, 1, 2, 3]))
        

def test_sorted_unique_1dim():
    x = pd.Series([4, 3, nan, 8, 4, 3, nan, 2])
    pd_testing.assert_series_equal(
        pdu.sorted_unique_1dim(x),
        pd.Series([2, 3, 4, 8, nan]))


def test_left_merge():
    df = pdu.from_items([('ITEM', [10, 20, 70, 30]),
                         ('Quantity', [3, 4, 8, 9])])
    dfb = pdu.from_items([('ITEM', [10, 20, 90]),
                          ('Quantity_nrm', [8, 9, 7])])
    dfr = pdu.left_merge(df, dfb, on=['ITEM'])
    pd_testing.assert_frame_equal(
        dfr, pdu.from_items([('ITEM', [10, 20, 70, 30]),
                             ('Quantity', [3, 4, 8, 9]),
                             ('Quantity_nrm', [8, 9, nan, nan])]))
    
    df = pdu.from_items([('ITEM', np.zeros(0)),
                         ('Quantity', np.zeros(0))])
    dfb = pdu.from_items([('ITEM', np.zeros(0)),
                          ('Quantity_nrm', np.zeros(0))])
    dfr = pdu.left_merge(df, dfb, on=['ITEM'])
    pd_testing.assert_frame_equal(
        dfr, pdu.from_items([('ITEM', np.zeros(0)),
                             ('Quantity', np.zeros(0)),
                             ('Quantity_nrm', np.zeros(0))]
                             ))


@numba.njit
def add_cumsum_a_to_b(X):
    a = X[:, 0]
    b = X[:, 1]
    b += np.cumsum(a)
    # to test whether overwriting a non-result column has any consequence:
    a[:] = 0.


@numba.njit
def double_a(X):
    X[:, 0] *= 2
    

def test_fast_groupby_multi_transform():
    df = pdu.shuffle_df_drop_index(
        pdu.from_items([
            ('g', [0,   0, 0, 0, 1, 1, 1, 1]),
            ('a', [1,   2, 4, 8, 3, 9, 27, 81]),
            ('b', [nan, 2, 5, 4, 4, 0, 3, -1])
        ]))
    
    pdu.fast_groupby_multi_transform(
        df, 'g', ['a', 'b'], 'b', add_cumsum_a_to_b, further_sort_colnames='a')

    pdu.assert_frame_contents_equal(
        df,
        pdu.from_items([
            ('g', [0,   0, 0, 0,   1, 1, 1, 1]),
            ('a', [1,   2, 4, 8,   3, 9,  27, 81]),
            ('b', [nan, 5, 12, 19, 7, 12, 42, 119])
        ]))

    pdu.fast_groupby_multi_transform(
        df, 'g', 'a', 'a', double_a, already_sorted=True)
    
    pdu.assert_frame_contents_equal(
        df,
        pdu.from_items([
            ('g', [0,   0, 0, 0,   1, 1, 1, 1]),
            ('a', [2,   4, 8, 16,  6, 18, 54, 162]),
            ('b', [nan, 5, 12, 19, 7, 12, 42, 119])
        ]))
    

def test_flatten_multi_columns():
    df = pdu.from_items([
        (('a', '1'), [3, 4, 3, 2]),
        (('b', '2'), ['a', 'b', 'b', 'a']),
        (('c', '1'), ['x', 'y', 'z', 'x'])
    ], index=[3, 4, 5, 8])
    
    pdu.flatten_multi_columns(df)
    pd_testing.assert_frame_equal(
        df,
        pdu.from_items([
            ('a_1', [3, 4, 3, 2]),
            ('b_2', ['a', 'b', 'b', 'a']),
            ('c_1', ['x', 'y', 'z', 'x'])
        ], index=[3, 4, 5, 8]))
