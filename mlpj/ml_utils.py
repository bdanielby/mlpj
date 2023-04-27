"""
Utilities and convenience functions for using `sklearn` and other standard
machine learning libraries
"""
import re
import collections
import os
from typing import Optional, List, Union, Type, Any, Dict, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import sklearn.base
import sklearn.pipeline

from . import python_utils as pu
from . import pandas_utils as pdu
from . import project_utils


@runtime_checkable
class Estimator(Protocol):
    """Protocol to declare the type of sklearn-style estimators in function
    signatures
    """
    def fit(self, X, y, **fit_params): ...
    
    def predict(self, X): ...
    

@runtime_checkable
class Transformer(Protocol):
    """Protocol to declare the type of sklearn-style transformers in function
    signatures
    """
    def fit(self, X, **fit_params): ...
    
    def transform(self, X): ...

    def fit_transform(self, X, **fit_params): ...


def find_cls_in_sklearn_obj(
        est_or_trans: Union[Estimator, Transformer],
        cls: Union[Type[Estimator], Type[Transformer]]
) -> Union[Estimator, Transformer]:
    """Find a transformer or estimator of the given class within the given
    structured scikit-learn estimator or transformer.

    `isinstance(cls)` is used as the criterion.
    
    So far, the meta-estimators `OnCols`, `OnColsTrans and
    `sklearn.pipeline.Pipeline` are supported.

    Args:
        est_or_trans (`Estimator` | `Transformer`):
            structured estimator or transformer
        cls (class object): to compare against
    Returns:
        object of the given class
    Raises:
        `ValueError` if no match was found
    """
    def loop(est_or_trans: Union[Estimator, Transformer]) -> Optional[Any]:
        if isinstance(est_or_trans, cls):
            return est_or_trans
        elif isinstance(est_or_trans, (OnCols, OnColsTrans)):
            return loop(est_or_trans._est)
        elif isinstance(est_or_trans, sklearn.pipeline.Pipeline):
            for _, est in est_or_trans.steps:
                res = loop(est)
                if res is not None:
                    return res
            return None
        else:
            return None

    res = loop(est_or_trans)
    if res is None:
        raise ValueError(f'no match found for {cls} in {est_or_trans}')
    else:
        return res
        

def cyclic_boosting_analysis(
        pj: project_utils.Manager, est: Estimator, model_name: str
) -> None:
    """Create the analysis plots for a trained Cyclic Boosting model based on
    the plot observers in the estimator.

    The links to the PDFs are printed. Put this into a call of
    `project_utils.Manager.printer`.
    
    Args:
        pj (`project_utils.Manager`): project manager object
        est (`Estimator`): Scikit-learn-style estimator containing a
            Cyclic Boosting model
        model_name (str): model name

    Using this function introduces a new dependency: `cyclic_boosting`.
    """
    import cyclic_boosting
    import cyclic_boosting.plots
    from cyclic_boosting import binning

    binner = find_cls_in_sklearn_obj(est, binning.BinNumberTransformer)
    cb_est = find_cls_in_sklearn_obj(
        est, cyclic_boosting.base.CyclicBoostingBase)
    
    for iteration0, plob in enumerate(cb_est.observers):
        iteration = iteration0 + 1
        if iteration == len(cb_est.observers):
            iteration = -1
            remark = " for the last iteration"
        else:
            remark = f" for iteration {iteration}"
        filepath = pj.get_analysis_pdf_filepath(model_name, iteration=iteration)
        filepath_wo_ext = re.sub(r'(?i)\.pdf$', '', filepath)
        cyclic_boosting.plots.plot_analysis(
            plot_observer=plob, file_obj=filepath_wo_ext, use_tightlayout=False,
            binners=[binner])
        filename = os.path.basename(filepath)
        print(f'Cyclic Boosting training plots{remark}: '
              f'<a href="../image/{filename}">{filename}</a>')


def xgboost_analysis(est: Estimator, model_name: str, used_features: List[str]) -> None:
    """Print analysis information about a trained XGBoost model.

    Args:
        est (`Estimator`): Scikit-learn-style estimator containing an
            XGBoost model
        model_name (str): model name

    Using this function introduces a new dependency: `xgboost`.
    """
    import xgboost
    
    xg_est = find_cls_in_sklearn_obj(est, xgboost.XGBModel)
        
    print(f"<h3>feature importances of model {model_name}</h3>")
    importances = xg_est.feature_importances_
    ind = np.argsort(importances, kind='stable')[::-1]
    for i in ind:
        print(f"{used_features[i]}: {importances[i]:.3f}")    

        
def get_used_features(feature_properties: Dict[Any, int]) -> List[str]:
    """Get the column names in the Cyclic Boosting feature properties. They
    are the features used by the model.
    
    Args:
        feature_properties (dict): Cyclic Boosting's feature properties dict
    Returns:
        used_features (list of str): columns in `X` to pass as features
    """
    res = []
    for key in feature_properties.keys():
        if pu.isstring(key):
            res.append(key)
        elif isinstance(key, collections.abc.Sequence):
            for subkey in key:
                if subkey not in feature_properties:
                    res.append(subkey)
        else:
            raise ValueError("only strings and sequences of strings supported "
                             f"as keys in feature_properties, found: {key}")
    return res
        

def create_ordinal_encoder(
        df: pd.DataFrame, used_features: List[str]
) -> Transformer:
    """OrdinalEncoder acting on all categorical features

    Args:
        df (`pd.DataFrame`): input dataframe
        used_features (list of str): columns in `X` to pass as features
    Returns:
        `Transformer` for the categorical features (wrapped in `OnColsTrans)

    Using this function introduces a new dependency: `category_encoders`.
    """
    from category_encoders import OrdinalEncoder

    return OnColsTrans(
        OrdinalEncoder(handle_missing='return_nan',
                       handle_unknown='return_nan'),
        pdu.category_colnames(df, feature_list=used_features))


def create_target_encoder(
        df: pd.DataFrame, used_features: List[str],
        min_samples_leaf: int = 20
) -> Transformer:
    """TargetEncoder acting on all categorical features

    Args:
        df (`pd.DataFrame`): input dataframe
        used_features (list of str): columns in `X` to pass as features
        The rest of the params are hyperparams of
        `category_encoders.TargetEncoder`.
    Returns:
        `Transformer` for the categorical features (wrapped in `OnColsTrans)

    Using this function introduces a new dependency: `category_encoders`.
    """
    from category_encoders import TargetEncoder

    return OnColsTrans(
        TargetEncoder(
            handle_missing='return_nan', handle_unknown='return_nan',
            min_samples_leaf=min_samples_leaf),
        pdu.category_colnames(df, feature_list=used_features))


class OnCols(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin,
         sklearn.base.RegressorMixin
):
    """Scikit-learn-style meta-estimator restricting a given estimator to
    a given subset of the columns of the feature matrix.

    The feature matrix is expected to be a `pd.DataFrame`.

    Args:
        est (`Estimator`): base estimator to be wrapped
        used_features (list of str): column names to restrict to
    """
    def __init__(self, est: Estimator, used_features: List[str]):
        self._est = est
        self._used_features = used_features
        self._estimator_type = est._estimator_type

    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("OnCols's methods require dataframes as "
                             "feature matrices")
        return X[self._used_features].copy()

    def fit(self, X: pd.DataFrame, y: ArrayLike, **fit_params) -> Estimator:
        X_passed = self._select_features(X)
        self._est = sklearn.base.clone(self._est)
        self._est.fit(X_passed, y, **fit_params)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_passed = self._select_features(X)
        return self._est.predict(X_passed)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_passed = self._select_features(X)
        return self._est.predict_proba(X_passed)

    
class OnColsTrans(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Scikit-learn-style meta-transformer restricting a given transformer to
    a given subset of the columns of the feature matrix.

    The feature matrix is expected to be a `pd.DataFrame`.

    Args:
        trans: base transformer to be wrapped
        used_features: column names to restrict to
        output_features: column names for the result of the base transformer,
            defaulting to `used_features`

            If the base transformer transforms a `pd.DataFrame` into a
            `pd.DataFrame` instead of an `np.ndarray`, the output features
            are taken from it and this param is ignored.
        keep_originals: whether to keep columns in `used_features` which aren't
            present among `output_features` (or the output dataframe of
            the base transformer)
    """
    def __init__(
            self, trans: Transformer, used_features: List[str],
            output_features: Optional[List[str]] = None, keep_originals: bool = False
    ):
        self._est = trans
        self._used_features = used_features
        if output_features is None:
            self._output_features = used_features
        else:
            self._output_features = output_features
        self._keep_originals = keep_originals

    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "OnColsTrans's methods require dataframes as "
                "feature matrices")
        return X[self._used_features].copy()

    def _result_dataframe(
            self, X: pd.DataFrame,
            trans_result: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        X = X.copy()
        
        if isinstance(trans_result, pd.DataFrame):
            output_features = trans_result.columns.to_list()
            X_trans = trans_result.iloc
        else:
            output_features = self._output_features
            
            n_cols_found = trans_result.shape[1]
            if len(output_features) != n_cols_found:
                raise ValueError(
                    f"The transformer result has {n_cols_found} columns but "
                    f"we expected the output features {output_features}, i.e. "
                    f"{len(output_features)} columns")
            
            X_trans = trans_result

        if not self._keep_originals:
            # Drop the used features which aren't present in the base
            #   transformer's output.
            dropped_colnames = set(self._used_features) - set(output_features)
            if len(dropped_colnames) > 0:
                X = pdu.drop_columns(X, dropped_colnames)
    
        # Copy all output features into the dataframe.
        for i, colname in enumerate(output_features):
            X[colname] = X_trans[:, i]
        return X

    def fit(self, X: pd.DataFrame, **fit_params) -> Transformer:
        X_passed = self._select_features(X)
        self._est = sklearn.base.clone(self._est)
        self._est.fit(X_passed, **fit_params)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_passed = self._select_features(X)
        return self._result_dataframe(X, self._est.transform(X_passed))

    def fit_transform(self, X: pd.DataFrame, **fit_transform) -> pd.DataFrame:
        X_passed = self._select_features(X)
        return self._result_dataframe(X, self._est.fit_transform(X_passed))
