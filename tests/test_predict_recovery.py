from __future__ import annotations

import numpy as np

from mleml import predict


def test_predict_recovers_shallow_univariate_eml():
    x = np.linspace(0.8, 2.1, 72)
    y = np.exp(1.0) - np.log(x)

    result = predict(x, y, max_depth=1)

    holdout = np.linspace(0.85, 2.05, 40)
    expected = np.exp(1.0) - np.log(holdout)
    actual = result(holdout)

    assert str(result).startswith("eml(")
    assert result.depth <= 1
    assert result.mse < 1e-8
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_predict_recovers_shallow_bivariate_eml():
    x1 = np.linspace(0.7, 1.9, 64)
    x2 = np.linspace(1.1, 2.3, 64)
    y = np.exp(x1) - np.log(x2)

    result = predict((x1, x2), y, max_depth=1)

    holdout_x1 = np.linspace(0.8, 1.8, 32)
    holdout_x2 = np.linspace(1.2, 2.2, 32)
    expected = np.exp(holdout_x1) - np.log(holdout_x2)
    actual = result(holdout_x1, holdout_x2)

    assert str(result).startswith("eml(")
    assert result.depth <= 1
    assert result.mse < 1e-8
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_predict_returns_best_fit_for_noisy_data():
    rng = np.random.default_rng(42)
    x = np.linspace(0.8, 2.0, 72)
    y = np.exp(1.0) - np.log(x) + rng.normal(scale=0.02, size=x.shape)

    result = predict(x, y, max_depth=2)

    assert str(result).startswith("eml(")
    assert np.isfinite(result.mse)

