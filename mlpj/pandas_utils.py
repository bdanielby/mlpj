"""
Utilities and convenience functions for using `pandas`.
"""
import re
import contextlib
import collections

import numpy as np
import pandas as pd
import pandas.testing as pd_testing
import numba

from . import python_utils as pu


def from_items(items, index=None):
    """Convert a sequence of (key, value) pairs to a DataFrame.

    This is similar to `pandas.DataFrame.from_items`, but with support for
    the `index` argument.

    In contrast to the `pandas.DataFrame` constructor using dictionaries,
    the order of the columns in the resulting dataframe will be as specified
    in the ``items`` argument.

    Args:
        items (seq of key-value pairs): Keys are the column names. Values
            should be array-like. All arrays have to be of equal length.
        index (array-like, optional): index to use for resulting dataframe.
            Will default to `np.arange()` if no indexing information is
            part of input data and no index is provided, see `pd.DataFrame`.
    """
    dframe = pd.DataFrame.from_dict(collections.OrderedDict(items))
    if index is not None:
        dframe.index = index
    return dframe


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


def get_colnames(X_or_names):
    """Get columns from an object carrying column names
    
    Args:
        X_or_names (`pd.DataFrame` or `DataFrameGroupBy` or seq of str):
            input carrying column names
    Returns:
        list of str: list of column names
    """
    if isinstance(X_or_names, pd.DataFrame):
        return X_or_names.columns
    elif isinstance(X_or_names, pd.core.groupby.DataFrameGroupBy):
        return X_or_names.obj.columns
    else:
        return X_or_names


def all_colnames_except(X_or_names, colnames):
    """All column names except the ones given in `colnames`.

    Args:
        X_or_names (`pd.DataFrame` or `DataFrameGroupBy` or seq of str):
            input carrying column names
        colnames (str or seq of str): column names to exclude
    Returns:
        list of str: remaining column names
    """
    return pu.all_except(get_colnames(X_or_names), colname_list(colnames))


def category_colnames(df, feature_list=None):
    """The list of columns of category type

    Args:
        df (`pd.DataFrame`): input dataframe
        feature_list (list of str, optional): If specified, consider only these
            columns of the dataframe.
    Returns:
        list of str: the category column names
    """
    selected = []
    for colname in df.columns:
        if df[colname].dtype.name == 'category':
            selected.append(colname)
    if feature_list is not None:
        selected = set(selected)
        return [feature for feature in feature_list if feature in selected]
    else:
        return selected

    
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
        raise KeyError('column {} not found'.format(old_name))
    X.rename(columns={old_name: new_name}, inplace=True)

    
def drop_index(X):
    """Drop the index from a dataframe in place.

    Args:
        X (`pd.DataFrame`): input dataframe
    Returns:
        X: changed in place
    """
    X.reset_index(drop=True, inplace=True)
    return X


def drop_columns(X, columns):
    """Drop columns of a dataframe in place.

    Args:
        columns (str or list of str): column names to remove; don't use tuples in but
        lists.
    Returns:
        X: changed in place
    """
    X.drop(columns, axis=1, inplace=True)
    return X


def columns_to_right(df, colnames):
    """Move some column names to the right.

    Due to a Pandas bug, columns with texts in right-to-left writing order
    are always printed on the right end of a table. Circumvent this by
    moving these columns to the right. Then the values will be consistent
    with the headline output.

    Args:
        df (`pd.DataFrame`): input dataframe
        colnames (str or list of str): column names to move to the right
    Returns:
        `pd.DataFrame` with the columns rearranged as described
    """
    colnames = colname_list(colnames)
    colnames_set = set(colnames)
    return df[[colname for colname in df.columns if colname not in colnames_set] + colnames]


def shuffle_df_drop_index(df):
    """Shuffle the rows of a dataframe and drop the index.

    Args:
        df (`pd.DataFrame`): input dataframe
    Returns:
        `pd.DataFrame`: new dataframe
    """
    import sklearn.utils
    
    df = sklearn.utils.shuffle(df)
    drop_index(df)
    return df


def assert_frame_contents_equal(df1, df2, **kwargs):
    """Convenience function to assert that the contents of two datframes is
    equal.

    That is, the indices of the dataframes are ignored.

    Args:
        df1 (`pd.DataFrame`): first input dataframe
        df2 (`pd.DataFrame`): second input dataframe
        kwargs: See `pd.testing.assert_frame_equal`.
    """
    pd_testing.assert_frame_equal(
        df1.reset_index(drop=True), df2.reset_index(drop=True), **kwargs)

        
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


def sorted_unique_1dim(ser):
    """The sorted series of unique values within the given series or index.

    Args:
        ser (`pd.Series` | `pd.Index`): input series or index
    Returns:
        `pd.Series`: unique values of the original series as a series with the
            original name
    """
    arr = np.unique(ser.values)
    return pd.Series(arr, name=ser.name)


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


def _fast_groupby_multi_transform1(igroup, work_columns, f):
    """Backend for fast_groupby_multi_transform

    Args:
        igroup (`np.ndarray` of int): array containing the group numbers
        nrows (int): number of rows
        work_columns (`np.ndarray` of shape `(n, d)`): the work columns
            to read and write, the result columns among them
        f (user function, preferably a `numba`-function): called for each
            row group on the respective slice of `work_columns`
    """
    nrows = len(igroup)
    ngroups = igroup[-1] + 1
    row_of_group_start = 0

    for i in range(ngroups):
        # Find the end of the current row group.
        cur_igroup = igroup[row_of_group_start]
        row_of_group_end = row_of_group_start + 1
        while row_of_group_end < nrows and igroup[row_of_group_end] == cur_igroup:
            row_of_group_end += 1

        work_columns_slice = (
            work_columns[row_of_group_start:row_of_group_end, :])

        f(work_columns_slice)
        row_of_group_start = row_of_group_end
        

_fast_groupby_multi_transform1_jit = numba.njit(_fast_groupby_multi_transform1)


def fast_groupby_multi_transform(
        df, group_colnames, work_colnames, result_colnames,
        f, further_sort_colnames=[], already_sorted=False
):
    """Efficiently transform multiple columns as a whole for each row group.

    In contrast to Pandas's groupby-transform, the work columns are transformed
    as a whole rather than independently of each other. The dataframe is changed
    in place.

    Args:
        df (`pd.DataFrame`): input dataframe
        group_colnames (str or seq of str): group column names
        work_colnames (str or seq of str): work column names for the user function
            If you need group columns as work columns, copy them under a
            new name first. If you need extra result columns, create them
            with default values in `df` first.
        result_colnames (sr or seq of str): These represent the result column
            names which are written back into the df at the end.
            They must be a subset of `work_colnames`.
        f (user function, preferably a `numba` JIT function): called for
            each row group.
            Its sole param is the 2D numpy array for containing the slice of the
            work columns (in the order of `work_colnames`) for the current
            row group. The type of the work columns is chosen by
            `pd.DataFrame.to_numpy`.
            The function can read and write these columns. Only the
            `result_colnames` are written back in the end. The rest of the
            changes has no effect. The return value of the function is ignored.
        further_sort_colnames (str or seq of str): `df` will be
            resorted in place by the `group_colnames + further_sort_colnames`
            first unless `already_sorted == True`. A stable sort method
            is used.
        already_sorted (bool): If `True`, turn the initial sorting off because
            `df` is already sorted as specified above.
    """
    import numba.extending

    group_colnames = colname_list(group_colnames)
    work_colnames = colname_list(work_colnames)
    result_colnames = colname_list(result_colnames)
    if not set(result_colnames) <= set(work_colnames):
        raise ValueError(
            f"result_colnames {result_colnames} is not a subset of work_colnames "
            f"{work_colnames}")
        
    further_sort_colnames = colname_list(further_sort_colnames)
    if not already_sorted:
        df.sort_values(group_colnames + further_sort_colnames, inplace=True,
                       kind='stable')

    igroup = df.groupby(group_colnames).ngroup().values

    work_columns = df[work_colnames].to_numpy(copy=True)

    if numba.extending.is_jitted(f):
        backend = _fast_groupby_multi_transform1_jit
    else:
        backend = _fast_groupby_multi_transform1
    backend(igroup, work_columns, f)

    for result_colname in result_colnames:
        i = work_colnames.index(result_colname)
        df[result_colname] = work_columns[:, i]


def flatten_multi_columns(df):
    """A multi-index for the column is flattened in place.

    The column levels are combined with underscores.

    Args:
        df (`pd.DataFrame`): input dataframe
    """
    df.columns = ['_'.join(colname).strip() for colname in df.columns.values]


def rename_groupby_colnames(df, name_for_count=None, renamings=()):
    """Rename column names as they arise from groupby-calls.

    The dataframe is changed in place.
    
    If a column has multiple aggregations, the tuple column names are combined
    with underscores. Otherwise just the column is used.

    Args:
        df (`pd.DataFrame`): input dataframe
        name_for_count (str, optional): If given, this aggregation is treated
            separately because such an aggregation is often not specific to
            the column it was used on
        renamings (seq of key-value pairs or dict):
            These renamings are applied after all the other steps. They are
            applied to the name of the index (or its levels for a multiindex).
    """
    assert name_for_count is None or pu.isstring(name_for_count)
    if not isinstance(renamings, dict):
        renamings = dict(renamings)
        
    aggs_for_colname = collections.defaultdict(list)
    for colname, agg_name in df.columns:
        if name_for_count is not None and agg_name == 'count':
            continue
        aggs_for_colname[colname].append(agg_name)
        
    new_colnames = []
    for colname, agg_name in df.columns:
        if name_for_count is not None and agg_name == 'count':
            new_colname = name_for_count
        else:
            aggs = aggs_for_colname[colname]
            if len(aggs) == 1:
                new_colname = colname
            else:
                new_colname = '{}__{}'.format(colname, agg_name)
        new_colname = renamings.get(new_colname, new_colname)
        new_colnames.append(new_colname)
    df.columns = new_colnames
    
    if len(renamings) > 0:
        if isinstance(df.index, pd.MultiIndex):
            for i_level in range(len(df.index.levels)):
                name = df.index.levels[i_level].name
                if name in renamings:
                    df.index.levels[i_level].name = renamings[name]
        else:
            if df.index.name in renamings:
                df.index.name = renamings[df.index.name]


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


def consistency_check(df, colname_a, colname_b, n_examples=50, extra_colnames=[]):
    """Print a report about inconsistencies between two columns.

    Args:
        df (`pd.DataFrame`): input dataframe
        colname_a (str): first column name to compare
        colname_b (str): second column name to compare
        n_examples (int): number of inconsistency examples to show
        extra_colnames (list of str): extra columns to show for context in the
            inconsistency results
    """
    ser_a = df[colname_a]
    ser_b = df[colname_b]
    
    print(f"consistency check of {ser_a.name} and {ser_b.name}")
    a_isnull = ser_a.isnull()
    is_same = (ser_a == ser_b) | (a_isnull & (a_isnull == ser_b.isnull()))
    if is_same.all():
        print("always the same")
    else:
        print("not always the same; value counts of equality:")
        print(is_same.value_counts())
        print()
        extra_colnames = colname_list(extra_colnames)
        examples = df.loc[~is_same, extra_colnames + [colname_a, colname_b]]
        n_examples, n_examples_perc = pu.wi_perc(len(examples), len(df))
        print(f"{n_examples} inconsistencies found ({n_examples_perc:.1f} %).")
        if len(examples) <= n_examples:
            print("all inconsistencies:")
        else:
            examples = examples.iloc[-n_examples:]
            print(f"the last {len(examples)} inconsistencies:")
        print(examples)

        
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


def truncate_datetime_to_freq(ser, freq):
    """Truncate datetime values to the specified period.

    Args:
        ser (`pd.Series`): input series
        freq (str): Pandas frequency or frequency alias
    Returns:
        `pd.Series` with truncated datetime values
    """
    return ser.dt.to_period(freq).dt.start_time


def truncate_datetime_to_month(ser):
    """Truncate datetime values to the beginning of the month.

    Special case of `truncate_datetime_to_freq` with frequency 'M'.
    
    Args:
        ser (`pd.Series`): input series
    Returns:
        `pd.Series` with datetime values truncated to the beginning of the
        respective month
    """
    return truncate_datetime_to_freq(ser, 'M')


def truncate_datetime_to_week(ser, sunday_first=False):
    """Truncate datetime values to the beginning of the week.

    Special case of `truncate_datetime_to_freq`.
    
    Args:
        ser (`pd.Series`): input series
        sunday_first (bool): whether to consider Sunday the beginning of the week;
            by default the week starts with Monday
    Returns:
        `pd.Series` with datetime values truncated to the beginning of the
        respective week
    """
    if sunday_first:
        freq = 'W-SAT'
    else:
        freq = 'W'
    return truncate_datetime_to_freq(ser, freq)


MIN_DATETIME = pd.to_datetime("1970-01-01")


def datetime_to_epoch(ser):
    """Convert a Pandas datetime series to a float epoch, including the
    nanoseconds.
    
    NaT values are converted to NaT values.

    Args:
        ser (`pd.Series`): input deries
    Returns:
        `pd.Series`: output series containing float epochs
    """
    return (ser - MIN_DATETIME).dt.total_seconds()


def to_csv(df, filepath, **kwargs):
    """Convenience function to call `pd.DataFrame.to_csv` with common
    defaults.

    Args:
        df (`pd.DataFrame`): input dataframe
        file (str or stream): CSV filepath to create or stream to write to
        kwargs: See `pd.DataFrame.to_csv`.
    """
    kwargs.setdefault('sep', ';')
    kwargs.setdefault('index', False)
    kwargs.setdefault('encoding', 'utf-8')
    df.to_csv(filepath, **kwargs)
