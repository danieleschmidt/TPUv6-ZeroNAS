"""Tests for ZeroNASScorer."""

import numpy as np
import pytest

from tpuv6_zeronas import ZeroNASScorer


def identity_model(x: np.ndarray) -> np.ndarray:
    return x.copy()


def constant_model(x: np.ndarray) -> np.ndarray:
    return np.zeros_like(x)


def relu_model(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def linear_model(x: np.ndarray) -> np.ndarray:
    flat = x.reshape(x.shape[0], -1)
    return flat @ (np.ones((flat.shape[1], 10)) * 0.01)


@pytest.fixture
def scorer():
    return ZeroNASScorer()


def test_score_returns_float(scorer):
    result = scorer.score(identity_model)
    assert isinstance(result, float)


def test_score_nonnegative(scorer):
    result = scorer.score(relu_model)
    assert result >= 0.0


def test_constant_model_low_score(scorer):
    """A constant model (all zeros) should have zero or near-zero score."""
    result = scorer.score(constant_model)
    assert result < 1e-3


def test_identity_model_has_positive_score(scorer):
    result = scorer.score(identity_model)
    assert result > 0.0


def test_score_custom_input_shape(scorer):
    result = scorer.score(identity_model, input_shape=(1, 16, 16))
    assert isinstance(result, float)
    assert result >= 0.0


def test_score_different_models_differ(scorer):
    s1 = scorer.score(identity_model)
    s2 = scorer.score(linear_model)
    assert abs(s1 - s2) > 1e-9


def test_scorer_default_input_shape(scorer):
    """Default input_shape=(1,3,32,32) should work without errors."""
    result = scorer.score(relu_model)
    assert np.isfinite(result)
