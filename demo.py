#!/usr/bin/env python3
"""
demo.py — Toy NAS search demonstrating TPUv6-ZeroNAS pipeline.

Steps
-----
1. Define a search space.
2. Enumerate all architectures.
3. Generate synthetic latency/energy training data.
4. Fit a PerformanceModel.
5. Rank architectures with PredictiveRanker.
6. Print the top 5.
"""

import numpy as np

from tpuv6_zeronas import (
    NASSearchSpace,
    PerformanceModel,
    PredictiveRanker,
    ZeroNASScorer,
)


def main():
    print("=" * 60)
    print("  TPUv6-ZeroNAS — Predictive NAS Demo")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Search space
    # ------------------------------------------------------------------ #
    search_space = NASSearchSpace(
        conv_channels=[32, 64, 128, 256],
        attention_heads=[4, 8, 16],
        mlp_ratios=[2, 4, 8],
    )
    all_archs = search_space.all_architectures()
    print(f"\n[1] Search space: {len(all_archs)} candidate architectures")

    # ------------------------------------------------------------------ #
    # 2. Synthetic training data for PerformanceModel
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(42)
    sample_archs = rng.choice(all_archs, size=min(30, len(all_archs)), replace=False).tolist()

    X_train = np.array([search_space.feature_vector(a) for a in sample_archs])

    # Synthetic targets: latency ~ linear in params+ops, energy ~ quadratic
    noise_lat = rng.normal(0, 0.5, len(sample_archs))
    noise_eng = rng.normal(0, 0.3, len(sample_archs))
    y_latency = 0.5 * X_train[:, 0] + 0.3 * X_train[:, 1] + 1.0 + noise_lat
    y_energy = 0.2 * X_train[:, 0] ** 2 + 0.1 * X_train[:, 2] + 0.5 + noise_eng

    # ------------------------------------------------------------------ #
    # 3. Fit PerformanceModel
    # ------------------------------------------------------------------ #
    perf_model = PerformanceModel(degree=2)
    perf_model.fit(X_train, y_latency, y_energy)
    print("\n[2] PerformanceModel fitted (degree=2 polynomial regression)")

    # ------------------------------------------------------------------ #
    # 4. ZeroNAS proxy score on a few sampled architectures
    # ------------------------------------------------------------------ #
    scorer = ZeroNASScorer()

    def simple_model_fn(x: np.ndarray) -> np.ndarray:
        """Simulated model: ReLU + global average pool + linear."""
        flat = np.maximum(x, 0).reshape(x.shape[0], -1)  # ReLU + flatten
        return flat.mean(axis=0, keepdims=True)            # global avg pool

    sample_score = scorer.score(simple_model_fn, input_shape=(1, 3, 32, 32))
    print(f"\n[3] ZeroNAS proxy score (example model): {sample_score:.4f}")

    # ------------------------------------------------------------------ #
    # 5. Rank all architectures with PredictiveRanker
    # ------------------------------------------------------------------ #
    ranker = PredictiveRanker(perf_model, tpuv6_speedup=2.0)
    top5 = ranker.top_k(all_archs, search_space, k=5)

    print("\n[4] Top-5 architectures by predicted TPU v6 efficiency:")
    print(f"{'Rank':<6} {'conv_channels':<16} {'attention_heads':<18} {'mlp_ratio':<12}")
    print("-" * 56)
    for rank, arch in enumerate(top5, start=1):
        print(
            f"{rank:<6} {arch['conv_channels']:<16} "
            f"{arch['attention_heads']:<18} {arch['mlp_ratio']:<12}"
        )

    print("\nDone! 🎉")


if __name__ == "__main__":
    main()
