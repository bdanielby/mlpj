"""
Unit tests for `mlpj.stats_utils`.
"""
import numpy as np
import numpy.testing

import scipy.stats

from mlpj import stats_utils


def test_nbinom_mu_alpha() -> None:
    for n, p in [(5, 0.5), ([2, 4, 3.5], [0.5, 0.2, 0.1])]:
        n = np.asarray(n)
        p = np.asarray(p)

        dist = scipy.stats.nbinom(n, p)
        mu = dist.mean()
        alpha = 1 / n

        n2, p2 = stats_utils.nbinom_mu_alpha(mu, alpha)
        numpy.testing.assert_allclose(n2, n)
        numpy.testing.assert_allclose(p2, p)


def test_overdispersion() -> None:
    for n, p in [(5, [0.5, 0.3]), (2.5, np.linspace(0.01, 0.99, num=10))]:
        p = np.asarray(p)
        N1, N2 = 10000, p.shape[0]
        alpha = 1 / n
        dist = scipy.stats.nbinom(n * np.ones_like(p), p)
        mu = dist.mean()
        y = dist.rvs(size=(N1, N2), random_state=23)
        alpha_est = stats_utils.estimate_overdispersion(
            y.flatten(),
            np.tile(mu, (N1, 1)).flatten())
        np.testing.assert_allclose(alpha_est, alpha, rtol=0.2)


