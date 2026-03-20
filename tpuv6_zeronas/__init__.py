"""
tpuv6_zeronas — Predictive Neural Architecture Search for TPU v6.

Exports
-------
PerformanceModel   : Polynomial regression for latency/energy prediction.
NASSearchSpace     : Discrete search space (conv channels, heads, MLP ratios).
ZeroNASScorer      : Training-free architecture scoring proxy.
PredictiveRanker   : Rank architectures by predicted TPU v6 efficiency.
"""

from .performance_model import PerformanceModel
from .search_space import NASSearchSpace
from .zero_nas_scorer import ZeroNASScorer
from .predictive_ranker import PredictiveRanker

__all__ = [
    "PerformanceModel",
    "NASSearchSpace",
    "ZeroNASScorer",
    "PredictiveRanker",
]

__version__ = "0.1.0"
