"""
Utilities and convenience functions for timeseries models
"""
import pandas as pd

from . import python_utils as pu


def remove_last_n_days(df, datetime_colname, n_days):
    """Remove the data later than `n_days` before today from the dataframe.

    Args:
        df (`pd.DataFrame`): input dataframe
        datetime_colname (str): column name containing pd-datetime values
        n_days (int): Data up to the date this many days ago will be kept.
    Returns:
        `pd.DataFrame`: filtered dataframe
    """
    last_day = pu.n_days_ago(n_days)
    return df[df[datetime_colname] <= pd.to_datetime(last_day)]


def ts_train_test_split(df, train_test_split_date, date_colname, target_colname):
    """Train-test split for timeseries data

    All data before the `train_test_split_date` is taken as training data,
    the data starting from that time is taken as test data.

    Args:
        df (`pd.DataFrame`): input dataframe
        train_test_split_date (str or `datetime.datetime`): train-test split
            date; must be convertible with `pd.to_datetime`.
        date_colname (str): column name with the date or datetime values to
            compare
        target_colname (str): the target column name to return as extra arrays;
            it isn't removed from X-matrix
    Returns:
        `X_train, y_train, X_test, y_test`
    """
    condition = df[date_colname] < pd.to_datetime(train_test_split_date)
    
    X_train = df[condition].copy()
    y_train = X_train[target_colname]
    X_test = df[~condition].copy()
    y_test = X_test[target_colname]

    return X_train, y_train, X_test, y_test
