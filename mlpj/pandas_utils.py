"""
Utilities and convenience functions for using `pandas`.
"""
import re
import contextlib

import numpy as np
import pandas as pd
import numba

from . import python_utils as pu


@contextlib.contextmanager
def wide_display(width=250, max_columns=75, max_rows=500):
    """Use this for a wide and long output of `pd.DataFrame`s.

    Args:
        width (int): terminal width in characters
        max_columns (int): maximum number of dataframe columns
        max_rows (int): maximum number of dataframe rows
    """
    with pd.option_context("display.width", width,
                           "display.max_columns", max_columns,
                           "display.max_rows", max_rows,
                           'max_colwidth', 80):
        yield


def is_numerical(ser):
    """Does a series contain numerical values?

    Args:
      ser (pd.Series): input series

    Returns:
      bool: whether the series can be used for numerical purposes
    """
    return ser.dtype.kind in "bif"


def rename_column(X, old_name, new_name):
    """Rename a column in the passed dataframe (in place).

    Args:
        X (`pd.DataFrame): input dataframe
        old_name (str): old column name
        new_name (str): new column name

    Raises:
        `ValueError` if the `old_name` isn't found among the columns.
    """
    if old_name not in X.columns:
        raise ValueError('column {} not found'.format(old_name))
    X.rename(columns={old_name: new_name}, inplace=True)


def print_column_info(col, table_name=None, ignored_columns=(), n_value_counts=50):
    """Print information about a column
    
    For a column, print its `dtype`, call `describe`, count the missing
    values, count the distinct values and print the `value_counts` of up to
    `n_value_counts` most frequent values.

    Args:
        col (`pd.Series`): the input column
        table_name (str): table name of the dataframe for context
        ignored_columns (seq of str): sequence of columns to ignore
        n_value_counts (int): number of values to print the value counts for
    """
    colname = col.name
    print("<hr>")
    if table_name is None:
        full_colname = colname
    else:
        full_colname = f"{table_name}.{colname}"
    internal_link = re.sub(r'\.', '_', full_colname) + '_colinfo'
    print(f'<a name="{internal_link}"><h3><a href="#{internal_link}">'
          f'column: {full_colname}</a></h3>')
    print(f"column dtype: {col.dtype}, length {len(col)}")
    if colname in ignored_columns:
        return
    
    print()
    print("describe:")
    print(col.describe())
    print()
    n_undefined, perc_undefined = n_undefined_and_percentage(col)
    print(f"{n_undefined} missing values ({perc_undefined:.1f} %)")
    value_counts = col.value_counts()
    n_value_counts = len(value_counts)
    print(f"{n_value_counts} distinct values")
    print()
    if n_value_counts < 50:
        print(f"value counts")
        print(value_counts)
    else:
        print("value counts of the 50 most frequent values:")
        print(value_counts.iloc[:50])

        
def print_table_info(df, table_name, ignored_columns=(), n_value_counts=50):
    """Print information about a dataframe's columns.
    
    For a dataframe, print its shape, column dtypes and call
    `print_column_info` for each column (except the ones in `ignored_columns`).

    Args:
        df (`pd.DataFrame`): the input dataframe
        table_name (str): table name of the dataframe for context
        ignored_columns (seq of str): sequence of columns to ignore
        n_value_counts (int): number of values to print the value counts for
    """
    print(f'<h3>table {table_name}</h3>')
    print(f"shape: {df.shape}")
    print(f"columns and dtypes of table {table_name}")
    print(df.dtypes)
    for colname in df.columns:
        print_column_info(
            df[colname], table_name=table_name, ignored_columns=ignored_columns,
            n_value_counts=n_value_counts)


def keys_of_dict_column(ser):
    """Transform a series with dict values into a series with the lists of their
    keys.

    Args:
        ser (`pd.Series`): input series with `dict` entries
    Returns:
        `pd.Series`: a new series containing the list of keys for each entry
    """
    return ser.transform(lambda   dic: list(dic.keys()))


def distinct_keys_of_dict_column(ser):
    """For a series with dict values, extract the distinct lists of keys
    these dicts.

    Args:
        ser (`pd.Series`): input series with `dict` entries
    Returns:
        `np.ndarray`: array of distinct lists of keys
    """
    return np.unique(keys_of_dict_column(ser))

        
def ser_where_defined(ser):
    """Select non-null entries of a series.

    Args:
        ser (`pd.Series`): input series
    Returns:
        `pd.Series`: a new series containing the non-null entries.
    """
    return ser[ser.notnull()]


def n_undefined_and_percentage(ser):
    """Number of undefined values and their percentage

    Args:
       ser (`pd.Series`): input series
    Returns:
       n, perc: the number of undefined values in the series and their
           percentage (scaled to 100)
    """
    return pu.wi_perc(ser.isnull().sum(), len(ser))


def colname_list(colnames):
    """If the input is a string, turn it into a one-element list, otherwise just
    return the input.

    Args:
        colnames (str or list of str): input column names
    Returns:
        list of str: list of oclumn names
    """
    if pu.isstring(colnames):
        return [colnames]
    return colnames
    

def sort(X, colnames=None, inplace=False, kind='stable', ascending=True):
    """Convenience function to sort a dataframe with a stable sort and ignoring
    the index.

    For some applications, the resorted original index is harmful.

    Args:
        X (`pd.DataFrame`): input dataframe
        colnames (str or list of str): column names to sort by
        inplace (bool): whether to sort in place
        kind (str): sorting algorithm, by default a stable algorithm is used.
        ascending (bool): whether to sort in ascending order

    Returns:
        `pd.DataFrame`: result dataframe even if we sort in place
    """
    if colnames is None:
        colnames = list(X.columns)
    else:
        colnames = colname_list(colnames)
    X1 = X.sort_values(colnames, inplace=inplace, kind=kind,
                       ascending=ascending, ignore_index=True)
    if not inplace:
        X = X1
    return X


def left_merge(left, right, **kwargs):
    """Convience function for a left merge in Pandas.
    
    `pd.DataFrame.merge` may produce result columns to in a different order
    if the left dataframe is empty. The (empty) index is also of a different
    type than for nonempty dataframes. This causes problems in `dask`. This
    wrapper function fixes the bug.

    Duplicated rows are not tolerated, i.e. the merge must be unique and the
    result must have rows in one-to-one correspondence with `left`. The
    index is always identical to the one in `left`.

    Args:
        left (`pd.DataFrame`): left dataframe for the left merge
        right (`pd.DataFrame`): right dataframe for the left merge
        kwargs: See `pd.merge`.
    Returns:
        `pd.DataFrame`: result of the left-merge, guaranteed to have the same
        length as `left`. No arrays are shared with `left` or `right`.
    Raises:
        `ValueError` if the result length is different from the length of
        `left`.
    """
    assert 'how' not in kwargs
    df = left.merge(right, how='left', **kwargs)
    
    left_colnames = left.columns.tolist()
    if (len(df) == 0 and
            df.columns.tolist()[:len(left_colnames)] != left_colnames):
        rest_colnames = all_colnames_except(df, left_colnames)
        df = df.loc[:, left_colnames + rest_colnames]
    if len(df) != len(left):
        raise ValueError("The left merge has a different number of rows ({}) "
                         "from the original ({})".format(len(df), len(left)))
    df.index = left.index
    return df


def flatten_multi_columns(df):
    """A multi-index for the column is flattened in place.

    The column levels are combined with underscores.

    Args:
        df (`pd.DataFrame`): input dataframe
    """
    df.columns = ['_'.join(colname).strip() for colname in df.columns.values]


def remove_last_n_days(df, datetime_colname, n_days):
    """Remove the last n_days days from the dataframe.

    Args:
        df (`pd.DataFrame`): input dataframe
        datetime_colname (str): column name containing pd-datetime values
        n_days (int): number of days (prior to today) to remove
    Returns:
        `pd.DataFrame`: filtered dataframe
    """
    last_day = pu.n_days_ago(n_days)
    return df[df[datetime_colname] <= pd.to_datetime(last_day)]
