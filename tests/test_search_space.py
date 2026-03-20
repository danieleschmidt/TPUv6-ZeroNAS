"""Tests for NASSearchSpace."""

import itertools

import numpy as np
import pytest

from tpuv6_zeronas import NASSearchSpace


@pytest.fixture
def default_space():
    return NASSearchSpace()


def test_all_architectures_count(default_space):
    archs = default_space.all_architectures()
    expected = len(default_space.conv_channels) * len(default_space.attention_heads) * len(default_space.mlp_ratios)
    assert len(archs) == expected


def test_all_architectures_unique(default_space):
    archs = default_space.all_architectures()
    tuples = [
        (a["conv_channels"], a["attention_heads"], a["mlp_ratio"])
        for a in archs
    ]
    assert len(set(tuples)) == len(tuples)


def test_sample_returns_valid_config(default_space):
    rng = np.random.default_rng(42)
    arch = default_space.sample(rng=rng)
    assert arch["conv_channels"] in default_space.conv_channels
    assert arch["attention_heads"] in default_space.attention_heads
    assert arch["mlp_ratio"] in default_space.mlp_ratios


def test_sample_reproducible():
    space = NASSearchSpace()
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    assert space.sample(rng1) == space.sample(rng2)


def test_sample_no_rng(default_space):
    arch = default_space.sample()
    assert "conv_channels" in arch
    assert "attention_heads" in arch
    assert "mlp_ratio" in arch


def test_feature_vector_shape(default_space):
    arch = default_space.all_architectures()[0]
    fv = default_space.feature_vector(arch)
    assert fv.shape == (3,)


def test_feature_vector_values_positive(default_space):
    for arch in default_space.all_architectures():
        fv = default_space.feature_vector(arch)
        assert np.all(fv > 0), f"Non-positive feature in {arch}: {fv}"


def test_custom_search_space():
    space = NASSearchSpace(conv_channels=[16, 32], attention_heads=[2], mlp_ratios=[1])
    archs = space.all_architectures()
    assert len(archs) == 2


def test_all_architectures_contain_expected_keys(default_space):
    for arch in default_space.all_architectures():
        assert set(arch.keys()) == {"conv_channels", "attention_heads", "mlp_ratio"}


def test_feature_vector_differs_for_different_archs(default_space):
    archs = default_space.all_architectures()
    fv0 = default_space.feature_vector(archs[0])
    fv1 = default_space.feature_vector(archs[-1])
    assert not np.allclose(fv0, fv1)
