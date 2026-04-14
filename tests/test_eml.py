from __future__ import annotations

import numpy as np

from mleml import eml


def test_eml_scalar_matches_definition():
    x = 1.7
    y = 2.3
    expected = np.exp(x) - np.log(y)
    assert np.isclose(eml(x, y), expected)


def test_eml_array_matches_definition():
    x = np.array([0.5, 1.0, 1.5], dtype=np.float64)
    y = np.array([1.1, 1.3, 1.7], dtype=np.float64)
    expected = np.exp(x) - np.log(y)
    actual = eml(x, y)
    np.testing.assert_allclose(actual, expected)

