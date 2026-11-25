import numpy as np
import pytest

from psionic import normalize_column


def test_columns_have_unit_length():
    x = np.random.default_rng(0).normal(size=(100, 5))
    y = normalize_column(x)
    assert np.allclose(np.linalg.norm(y, axis=0), 1.0)


def test_zero_column_stays_zero():
    x = np.random.default_rng(1).normal(size=(20, 3))
    x[:, 1] = 0.0
    y = normalize_column(x)
    assert np.all(y[:, 1] == 0.0)
    assert np.all(np.isfinite(y))


def test_rejects_1d_input():
    with pytest.raises(ValueError):
        normalize_column(np.ones(10))
