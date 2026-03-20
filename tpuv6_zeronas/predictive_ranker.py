"""
PredictiveRanker: Rank NAS architectures by predicted TPU v6 performance
using a fitted PerformanceModel.
"""

from typing import Dict, List

import numpy as np

from .performance_model import PerformanceModel
from .search_space import NASSearchSpace


class PredictiveRanker:
    """
    Ranks architectures by predicted efficiency on TPU v6, accounting for the
    hardware speedup factor relative to baseline TPU generation.

    Ranking criterion: lower ``latency × energy / tpuv6_speedup`` is better.
    """

    def __init__(self, performance_model: PerformanceModel, tpuv6_speedup: float = 2.0):
        """
        Parameters
        ----------
        performance_model : PerformanceModel
            A fitted performance model used for latency/energy prediction.
        tpuv6_speedup : float
            Relative throughput improvement of TPU v6 vs. previous generation.
            Predicted latency is divided by this factor before ranking.
        """
        self.performance_model = performance_model
        self.tpuv6_speedup = tpuv6_speedup

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_architectures(
        self, architectures: List[Dict], search_space: NASSearchSpace
    ) -> np.ndarray:
        """Return a 1-D array of scores (lower = better) for each architecture."""
        X = np.array([search_space.feature_vector(a) for a in architectures])
        latency = self.performance_model.predict_latency(X)
        energy = self.performance_model.predict_energy(X)
        # Apply TPU v6 speedup: effective latency is reduced by the speedup factor
        effective_latency = latency / self.tpuv6_speedup
        return effective_latency * energy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(
        self, architectures: List[Dict], search_space: NASSearchSpace
    ) -> List[Dict]:
        """
        Return architectures sorted from best (lowest latency×energy) to worst.

        Parameters
        ----------
        architectures : list of dict
            Architecture configs as returned by ``NASSearchSpace.sample`` /
            ``NASSearchSpace.all_architectures``.
        search_space : NASSearchSpace
            Used to convert configs to feature vectors.

        Returns
        -------
        list of dict
            Sorted architectures (best first).
        """
        if not architectures:
            return []
        scores = self._score_architectures(architectures, search_space)
        order = np.argsort(scores)
        return [architectures[i] for i in order]

    def top_k(
        self, architectures: List[Dict], search_space: NASSearchSpace, k: int = 5
    ) -> List[Dict]:
        """
        Return the top-k architectures by predicted TPU v6 performance.

        Parameters
        ----------
        k : int
            Number of architectures to return.

        Returns
        -------
        list of dict
            Top-k architectures (best first).
        """
        ranked = self.rank(architectures, search_space)
        return ranked[:k]
