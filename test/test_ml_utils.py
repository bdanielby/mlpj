"""
Unit tests for `mlpj.ml_utils`.
"""
import collections

import numpy as np
import pandas as pd
import pandas.testing as pd_testing
from sklearn import linear_model, preprocessing, feature_extraction, pipeline

from mlpj import pandas_utils as pdu
from mlpj import ml_utils


def test_Estimator() -> None:
    assert issubclass(linear_model.LinearRegression, ml_utils.Estimator)
    assert not issubclass(feature_extraction.DictVectorizer,
                        ml_utils.Estimator)


def test_Transformer() -> None:
    assert issubclass(preprocessing.StandardScaler, ml_utils.Transformer)
    assert not issubclass(preprocessing.StandardScaler, ml_utils.Estimator)
    
    assert issubclass(feature_extraction.DictVectorizer, ml_utils.Transformer)
    assert not issubclass(linear_model.LinearRegression,
                        ml_utils.Transformer)


def test_find_cls_in_sklearn_obj() -> None:
    scaler = preprocessing.StandardScaler()
    assert ml_utils.find_cls_in_sklearn_obj(
        scaler, preprocessing.StandardScaler) is scaler
    
    lin = linear_model.LinearRegression()
    assert ml_utils.find_cls_in_sklearn_obj(
        lin, linear_model.LinearRegression) is lin
    
    pipe = pipeline.Pipeline([
        ('scaler', scaler),
        ('lin', lin)
    ])
    assert ml_utils.find_cls_in_sklearn_obj(
        pipe, preprocessing.StandardScaler) is scaler
    assert ml_utils.find_cls_in_sklearn_obj(
        pipe, linear_model.LinearRegression) is lin
    
    oncols = ml_utils.OnCols(pipe, ['feature_a', 'feature_b'])
    assert ml_utils.find_cls_in_sklearn_obj(
        oncols, ml_utils.OnCols) is oncols
    assert ml_utils.find_cls_in_sklearn_obj(
        oncols, preprocessing.StandardScaler) is scaler
    assert ml_utils.find_cls_in_sklearn_obj(
        oncols, linear_model.LinearRegression) is lin


def test_get_used_features() -> None:
    assert ml_utils.get_used_features(
        collections.OrderedDict([('feature_b', 1), ('feature_a', 2)])
    ) == ['feature_b', 'feature_a']
    
    assert ml_utils.get_used_features(
        collections.OrderedDict([
            ('feature_b', 1),
            (('feature_b', 'feature_a'), 2)
        ])
    ) == ['feature_b', 'feature_a']


def test_oncols() -> None:
    N = 100
    df = pdu.from_items([
        ('x1', np.random.random(N)),
        ('x2', np.random.random(N)),
        ('c', np.random.random(N))
    ])
    df['y'] = 5.0 * df['x1'] + 3.0 * df['x2'] - 1.0 + 0.1 * np.random.random(N)
    
    model1 = ml_utils.OnCols(linear_model.LinearRegression(), ['x2', 'x1'])
    model1.fit(df, df['y'])
    est1 = model1._est
    y_pred1 = model1.predict(df)
    
    est2 = linear_model.LinearRegression()
    est2.fit(df[['x2', 'x1']], df['y'])
    y_pred2 = est2.predict(df[['x2', 'x1']])

    np.testing.assert_allclose(est1.coef_, est2.coef_)
    np.testing.assert_allclose(est1.intercept_, est2.intercept_)
    np.testing.assert_allclose(y_pred1, y_pred2)
