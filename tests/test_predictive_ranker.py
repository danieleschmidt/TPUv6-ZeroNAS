"""Tests for PredictiveRanker."""

import numpy as np
import pytest

from tpuv6_zeronas import NASSearchSpace, PerformanceModel, PredictiveRanker


@pytest.fixture
def fitted_model_and_space():
    space = NASSearchSpace()
    archs = space.all_architectures()
    X = np.array([space.feature_vector(a) for a in archs])
    rng = np.random.default_rng(99)
    y_lat = X[:, 0] * 0.4 + rng.uniform(0, 0.1, len(archs))
    y_eng = X[:, 1] * 0.3 + rng.uniform(0, 0.05, len(archs))
    model = PerformanceModel(degree=2).fit(X, y_lat, y_eng)
    return model, space, archs


def test_rank_returns_same_length(fitted_model_and_space):
    model, space, archs = fitted_model_and_space
    ranker = PredictiveRanker(model)
    ranked = ranker.rank(archs, space)
    assert len(ranked) == len(archs)


def test_rank_contains_all_architectures(fitted_model_and_space):
    model, space, archs = fitted_model_and_space
    ranker = PredictiveRanker(model)
    ranked = ranker.rank(archs, space)
    assert sorted(str(a) for a in ranked) == sorted(str(a) for a in archs)


def test_top_k_length(fitted_model_and_space):
    model, space, archs = fitted_model_and_space
    ranker = PredictiveRanker(model)
    top5 = ranker.top_k(archs, space, k=5)
    assert len(top5) == 5


def test_top_k_subset_of_ranked(fitted_model_and_space):
    model, space, archs = fitted_model_and_space
    ranker = PredictiveRanker(model)
    ranked = ranker.rank(archs, space)
    top5 = ranker.top_k(archs, space, k=5)
    assert top5 == ranked[:5]


def test_rank_empty_input(fitted_model_and_space):
    model, space, _ = fitted_model_and_space
    ranker = PredictiveRanker(model)
    assert ranker.rank([], space) == []


def test_top_k_with_k_greater_than_archs(fitted_model_and_space):
    model, space, _ = fitted_model_and_space
    tiny_archs = space.all_architectures()[:3]
    ranker = PredictiveRanker(model)
    result = ranker.top_k(tiny_archs, space, k=10)
    assert len(result) == 3


def test_speedup_changes_ranking():
    """Different tpuv6_speedup values shouldn't change order (scores scaled equally),
    but sanity-check that ranker instantiates with custom speedup."""
    space = NASSearchSpace(conv_channels=[32, 64], attention_heads=[4], mlp_ratios=[2])
    archs = space.all_architectures()
    X = np.array([space.feature_vector(a) for a in archs])
    y_lat = np.array([1.0, 2.0])
    y_eng = np.array([1.0, 1.0])
    model = PerformanceModel(degree=1).fit(X, y_lat, y_eng)

    ranker1 = PredictiveRanker(model, tpuv6_speedup=1.0)
    ranker2 = PredictiveRanker(model, tpuv6_speedup=4.0)
    r1 = ranker1.rank(archs, space)
    r2 = ranker2.rank(archs, space)
    # Order should be the same (speedup is a global scale)
    assert r1 == r2


def test_rank_best_arch_has_lowest_score(fitted_model_and_space):
    """The first ranked arch should have the minimum latency*energy product."""
    model, space, archs = fitted_model_and_space
    ranker = PredictiveRanker(model)
    ranked = ranker.rank(archs, space)

    X_all = np.array([space.feature_vector(a) for a in archs])
    lat = model.predict_latency(X_all)
    eng = model.predict_energy(X_all)
    scores = (lat / ranker.tpuv6_speedup) * eng

    best_idx = np.argmin(scores)
    best_arch = archs[best_idx]
    assert ranked[0] == best_arch
