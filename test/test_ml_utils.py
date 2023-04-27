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


def test_OnCols() -> None:
    N = 100
    df = pdu.from_items([
        ('x1', np.random.random(N)),
        ('x2', np.random.random(N)),
        ('c', np.random.random(N))
    ])
    df['y'] = 5.0 * df['x1'] + 3.0 * df['x2'] - 1.0 + 0.1 * np.random.random(N)
    
    model1 = ml_utils.OnCols(linear_model.LinearRegression(), ['x2', 'x1'])
    model1.fit(df, df['y'])
    est1 = model1.est
    y_pred1 = model1.predict(df)
    
    est2 = linear_model.LinearRegression()
    est2.fit(df[['x2', 'x1']], df['y'])
    y_pred2 = est2.predict(df[['x2', 'x1']])

    np.testing.assert_allclose(est1.coef_, est2.coef_)
    np.testing.assert_allclose(est1.intercept_, est2.intercept_)
    np.testing.assert_allclose(y_pred1, y_pred2)


def test_OnColsTrans_same_output_columns() -> None:
    N = 100
    df = pdu.from_items([
        ('x1', np.random.random(N)),
        ('x2', np.random.random(N)),
        ('c', np.random.random(N))
    ])
    
    wrapped_trans = ml_utils.OnColsTrans(
        preprocessing.StandardScaler(), ['x1', 'x2'])
    
    assert hasattr(wrapped_trans, 'get_params')
    assert hasattr(wrapped_trans, 'set_params')
        
    wrapped_trans.fit(df)
    df_trans1 = wrapped_trans.transform(df)
    assert isinstance(df_trans1, pd.DataFrame)
    df_trans2 = wrapped_trans.fit_transform(df)
    assert isinstance(df_trans2, pd.DataFrame)
    pd_testing.assert_frame_equal(df_trans1, df_trans2)
    scaler1 = wrapped_trans.est
    
    scaler2 = preprocessing.StandardScaler()
    scaler2.fit(df[['x1', 'x2']])
    df_trans3 = scaler2.transform(df[['x1', 'x2']])
    df_trans4 = scaler2.fit_transform(df[['x1', 'x2']])
    np.testing.assert_allclose(df_trans3, df_trans4)

    np.testing.assert_allclose(scaler1.scale_, scaler2.scale_)
    np.testing.assert_allclose(scaler1.mean_, scaler2.mean_)

    np.testing.assert_allclose(df_trans1[['x1', 'x2']].values, df_trans3)


def test_OnColsTrans_different_output_columns() -> None:
    N = 100
    df = pdu.from_items([
        ('x1', np.random.random(N)),
        ('x2', np.random.random(N)),
        ('c', np.random.random(N))
    ])

    for keep_originals in [False, True]:
        wrapped_trans = ml_utils.OnColsTrans(
            preprocessing.StandardScaler(), ['x1', 'x2'],
            output_features=['x1s', 'x2s'], keep_originals=keep_originals)
        
        wrapped_trans.fit(df)
        df_trans1 = wrapped_trans.transform(df)
        assert isinstance(df_trans1, pd.DataFrame)
        df_trans2 = wrapped_trans.fit_transform(df)
        assert isinstance(df_trans2, pd.DataFrame)
        pd_testing.assert_frame_equal(df_trans1, df_trans2)
        scaler1 = wrapped_trans.est

        if keep_originals:
            assert df_trans1.columns.to_list() == ['x1', 'x2', 'c', 'x1s', 'x2s']
        else:
            assert df_trans1.columns.to_list() == ['c', 'x1s', 'x2s']
        
        scaler2 = preprocessing.StandardScaler()
        scaler2.fit(df[['x1', 'x2']])
        df_trans3 = scaler2.transform(df[['x1', 'x2']])
        df_trans4 = scaler2.fit_transform(df[['x1', 'x2']])
        np.testing.assert_allclose(df_trans3, df_trans4)
    
        np.testing.assert_allclose(scaler1.scale_, scaler2.scale_)
        np.testing.assert_allclose(scaler1.mean_, scaler2.mean_)
    
        np.testing.assert_allclose(df_trans1[['x1s', 'x2s']].values, df_trans3)
