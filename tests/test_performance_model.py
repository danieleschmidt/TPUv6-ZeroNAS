"""Tests for PerformanceModel."""

import numpy as np
import pytest

from tpuv6_zeronas import PerformanceModel


@pytest.fixture
def simple_data():
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 10, (20, 3))
    y_lat = X[:, 0] * 0.5 + X[:, 1] * 0.3 + 1.0
    y_eng = X[:, 2] * 0.4 + 0.5
    return X, y_lat, y_eng


def test_fit_returns_self(simple_data):
    X, y_lat, y_eng = simple_data
    model = PerformanceModel()
    result = model.fit(X, y_lat, y_eng)
    assert result is model


def test_predict_latency_shape(simple_data):
    X, y_lat, y_eng = simple_data
    model = PerformanceModel().fit(X, y_lat, y_eng)
    preds = model.predict_latency(X)
    assert preds.shape == (len(X),)


def test_predict_energy_shape(simple_data):
    X, y_lat, y_eng = simple_data
    model = PerformanceModel().fit(X, y_lat, y_eng)
    preds = model.predict_energy(X)
    assert preds.shape == (len(X),)


def test_predict_latency_reasonable(simple_data):
    """Fitted model should track training labels reasonably well."""
    X, y_lat, y_eng = simple_data
    model = PerformanceModel(degree=2).fit(X, y_lat, y_eng)
    preds = model.predict_latency(X)
    # Correlation should be high on noiseless data
    corr = np.corrcoef(preds, y_lat)[0, 1]
    assert corr > 0.95


def test_predict_energy_reasonable(simple_data):
    X, y_lat, y_eng = simple_data
    model = PerformanceModel(degree=2).fit(X, y_lat, y_eng)
    preds = model.predict_energy(X)
    corr = np.corrcoef(preds, y_eng)[0, 1]
    assert corr > 0.95


def test_single_sample_prediction(simple_data):
    X, y_lat, y_eng = simple_data
    model = PerformanceModel().fit(X, y_lat, y_eng)
    x_single = X[[0]]
    lat = model.predict_latency(x_single)
    eng = model.predict_energy(x_single)
    assert lat.shape == (1,)
    assert eng.shape == (1,)


def test_predict_before_fit_raises():
    model = PerformanceModel()
    with pytest.raises(RuntimeError):
        model.predict_latency(np.ones((3, 3)))


def test_predict_energy_before_fit_raises():
    model = PerformanceModel()
    with pytest.raises(RuntimeError):
        model.predict_energy(np.ones((3, 3)))


def test_degree_1_works(simple_data):
    X, y_lat, y_eng = simple_data
    model = PerformanceModel(degree=1).fit(X, y_lat, y_eng)
    preds = model.predict_latency(X)
    assert preds.shape == (len(X),)


def test_different_degrees_give_different_predictions():
    """Degree-1 vs degree-3 should differ on nonlinear targets."""
    rng = np.random.default_rng(1)
    X = rng.uniform(1, 10, (30, 3))
    # Nonlinear target: includes squared interactions
    y_lat = X[:, 0] ** 2 + X[:, 1] * X[:, 2] + 1.0
    y_eng = X[:, 2] ** 2 - X[:, 0] + 2.0

    m1 = PerformanceModel(degree=1).fit(X, y_lat, y_eng)
    m3 = PerformanceModel(degree=3).fit(X, y_lat, y_eng)
    p1 = m1.predict_latency(X)
    p3 = m3.predict_latency(X)
    # Higher-degree model captures quadratic terms; predictions will differ
    assert not np.allclose(p1, p3)
