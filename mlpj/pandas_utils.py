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


def n_undefined_and_percentage(ser):
    """Number of undefined values and their percentage

    Args:
       ser (`pd.Series`): input series
    Returns:
       n, perc: the number of undefined values in the series and their
           percentage (scaled to 100)
    """
    return pu.wi_perc(ser.isnull().sum(), len(ser))
