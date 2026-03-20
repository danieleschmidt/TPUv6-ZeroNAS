"""
NASSearchSpace: Defines the discrete architecture search space for
TPU v6-optimised networks (conv channels, attention heads, MLP ratios).
"""

import itertools
from typing import Dict, List, Optional

import numpy as np


class NASSearchSpace:
    """
    Search space covering convolutional channel widths, attention head counts,
    and MLP expansion ratios.

    Each architecture is represented as a dict::

        {
            "conv_channels": int,
            "attention_heads": int,
            "mlp_ratio": int | float,
        }
    """

    def __init__(
        self,
        conv_channels: List[int] = None,
        attention_heads: List[int] = None,
        mlp_ratios: List[float] = None,
    ):
        self.conv_channels = conv_channels if conv_channels is not None else [32, 64, 128, 256]
        self.attention_heads = attention_heads if attention_heads is not None else [4, 8, 16]
        self.mlp_ratios = mlp_ratios if mlp_ratios is not None else [2, 4, 8]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, rng: Optional[np.random.Generator] = None) -> Dict:
        """Return a randomly sampled architecture configuration."""
        if rng is None:
            rng = np.random.default_rng()
        return {
            "conv_channels": int(rng.choice(self.conv_channels)),
            "attention_heads": int(rng.choice(self.attention_heads)),
            "mlp_ratio": float(rng.choice(self.mlp_ratios)),
        }

    # ------------------------------------------------------------------
    # Enumeration
    # ------------------------------------------------------------------

    def all_architectures(self) -> List[Dict]:
        """Return all architecture configurations (Cartesian product)."""
        configs = []
        for cc, ah, mr in itertools.product(
            self.conv_channels, self.attention_heads, self.mlp_ratios
        ):
            configs.append({
                "conv_channels": int(cc),
                "attention_heads": int(ah),
                "mlp_ratio": float(mr),
            })
        return configs

    # ------------------------------------------------------------------
    # Feature encoding
    # ------------------------------------------------------------------

    def feature_vector(self, arch_config: Dict) -> np.ndarray:
        """
        Convert an architecture config dict to a numeric feature vector
        suitable for ``PerformanceModel``.

        Returns
        -------
        np.ndarray, shape (3,)
            [num_params_proxy, num_ops_proxy, memory_mb_proxy]
        """
        cc = arch_config["conv_channels"]
        ah = arch_config["attention_heads"]
        mr = arch_config["mlp_ratio"]

        # Simple proxies (order-of-magnitude realistic for small networks)
        num_params = (cc ** 2) * ah * mr / 1e3          # in thousands → treat as M-scale proxy
        num_ops = cc * ah * mr * 4.0                     # GFLOPs proxy
        memory_mb = cc * mr * ah / 8.0                   # MB proxy

        return np.array([num_params, num_ops, memory_mb], dtype=float)
