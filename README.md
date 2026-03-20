# TPUv6-ZeroNAS

**Predictive Neural Architecture Search optimised for Google TPU v6**

---

## Overview

TPUv6-ZeroNAS is a lightweight Python library for finding high-efficiency neural network architectures without lengthy training runs. It combines two complementary approaches:

| Component | What it does |
|-----------|-------------|
| `PerformanceModel` | Polynomial regression: maps architecture features → predicted latency & energy |
| `NASSearchSpace` | Defines the discrete search space (conv channels, attention heads, MLP ratios) |
| `ZeroNASScorer` | Training-free proxy score (gradient norm + activation variance) |
| `PredictiveRanker` | Ranks candidates by predicted TPU v6 efficiency |

The key insight: **we never need to train candidate architectures**. Instead, we use:
1. A *performance model* fitted on hardware measurements (or synthetic proxies) to predict cost.  
2. A *zero-shot proxy* (gradient norm + activation variance) to estimate representation capacity.

Combined, these let us rank hundreds of architectures in seconds.

---

## Installation

```bash
pip install -e .
# or
pip install tpuv6-zeronas
```

Requires **Python ≥ 3.9** and **NumPy ≥ 1.22**. No PyTorch or JAX dependency.

---

## Quick Start

```python
import numpy as np
from tpuv6_zeronas import NASSearchSpace, PerformanceModel, PredictiveRanker

# 1. Define the search space
space = NASSearchSpace(
    conv_channels=[32, 64, 128, 256],
    attention_heads=[4, 8, 16],
    mlp_ratios=[2, 4, 8],
)

# 2. Enumerate candidates
all_archs = space.all_architectures()   # 36 configs

# 3. Fit a performance model on hardware measurements
#    (here we use synthetic data for illustration)
rng = np.random.default_rng(0)
X_train = np.array([space.feature_vector(a) for a in all_archs])
y_lat = X_train[:, 0] * 0.5 + rng.normal(0, 0.1, len(all_archs))
y_eng = X_train[:, 1] * 0.3 + rng.normal(0, 0.05, len(all_archs))

perf_model = PerformanceModel(degree=2).fit(X_train, y_lat, y_eng)

# 4. Rank and pick the top 5
ranker = PredictiveRanker(perf_model, tpuv6_speedup=2.0)
top5 = ranker.top_k(all_archs, space, k=5)
for i, arch in enumerate(top5, 1):
    print(f"{i}. {arch}")
```

Run the bundled demo:

```bash
python demo.py
```

---

## API Reference

### `PerformanceModel(degree=2)`
Polynomial regression surrogate for hardware cost.

| Method | Description |
|--------|-------------|
| `fit(X, y_latency, y_energy)` | Train on measured data |
| `predict_latency(X) → np.array` | Predict latency (ms) |
| `predict_energy(X) → np.array` | Predict energy (mJ) |

`X` rows: `[num_params, num_ops, memory_mb]`

---

### `NASSearchSpace(conv_channels, attention_heads, mlp_ratios)`
Discrete architecture search space.

| Method | Description |
|--------|-------------|
| `sample(rng=None) → dict` | Random architecture config |
| `all_architectures() → list` | Full Cartesian product |
| `feature_vector(arch) → np.array` | Encode config as 3-D feature |

---

### `ZeroNASScorer()`
Training-free proxy (no model weights needed).

| Method | Description |
|--------|-------------|
| `score(model_fn, input_shape) → float` | Gradient norm + activation variance |

`model_fn` is any callable `(np.ndarray) → np.ndarray`.

---

### `PredictiveRanker(performance_model, tpuv6_speedup=2.0)`
Ranks architectures by predicted TPU v6 efficiency.

| Method | Description |
|--------|-------------|
| `rank(architectures, search_space) → list` | Sorted best→worst |
| `top_k(architectures, search_space, k=5) → list` | Top-k only |

---

## How It Works

```
Search Space ──► all_architectures() ──► feature_vectors
                                                │
                                    PerformanceModel.predict()
                                                │
                                    latency × energy / speedup
                                                │
                                    PredictiveRanker.rank()
                                                │
                                           Top-5 Archs ✓
```

TPU v6's improved compute density is modelled by dividing predicted latency by `tpuv6_speedup` (default 2×), reflecting Google's published throughput gains over TPU v5.

---

## Running Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/ -v
```

---

## License

MIT
