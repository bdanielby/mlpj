"""
Utilities for statistical modeling
"""
from typing import Union, Tuple

import numpy as np
from numpy.typing import ArrayLike
from sklearn import linear_model


def estimate_overdispersion(y: ArrayLike, mu: ArrayLike) -> float:
    r"""Estimate a constant negative binomial overdispersion from target values
    and estimates for the mean with a Poisson model.

    The variance is assumed to be the following function of the mean
    $\sigma^2 = \mu + \alpha \mu^2, \alpha = 1 / n$ (NB2).

    Args:
        y: target values from which to estimate the overdispersion
        mu: individualized means estimated by an earlier model
    Returns:
        alpha: overdispersion parameter

    References:
        * Cameron and Trivedi: Regression Analysis of Count Data
          https://faculty.econ.ucdavis.edu/faculty/cameron/racd2/
    """
    y = np.asarray(y)
    mu = np.asarray(mu)
    ols_X = mu[:, None]
    ols_y = (np.square(y - mu) - mu) / mu

    ols_model = linear_model.LinearRegression(fit_intercept=False)
    ols_model.fit(X=ols_X, y=ols_y)
    assert ols_model.intercept_ == 0

    return ols_model.coef_[0]


def nbinom_mu_alpha(mu: Union[ArrayLike, float], alpha: Union[ArrayLike, float]
                  ) -> Tuple[Union[ArrayLike, float], Union[ArrayLike, float]]:
    r"""Calculate the parameters of the negative-binomial distribution as
    required by `scipy.stats.binom` based on the mean and the overdispersion
    parameter.

    The variance is assumed to be the following function of the mean
    $\sigma^2 = \mu + \alpha \mu^2, \alpha = 1 / n$

    Args:
        mu: (individualized) means
        alpha: (individualized) overdispersion param
    Returns:
        n, p (individualized)
    """
    var = mu * (1 + alpha * mu)
    n = mu * mu / (var - mu)
    p = mu / var
    return n, p
