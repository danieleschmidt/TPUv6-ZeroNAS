"""
ZeroNASScorer: Training-free NAS proxy based on gradient norm and
activation variance — no PyTorch required (pure numpy simulation).
"""

from typing import Callable, Tuple

import numpy as np


class ZeroNASScorer:
    """
    Computes a training-free score for a candidate architecture without
    any gradient-based training.

    The proxy combines two signals:
    1. **Gradient norm** — magnitude of the numerical Jacobian of the
       model output w.r.t. the input (higher ≈ richer representation).
    2. **Activation variance** — variance across the output activations
       (higher ≈ less dead/collapsed neurons).

    A higher combined score is considered better.
    """

    def __init__(self):
        self._eps = 1e-5

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _numerical_gradient_norm(
        self,
        model_fn: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
    ) -> float:
        """
        Estimate gradient norm via finite differences (central diff).
        We sample a random direction to keep cost manageable.
        """
        rng = np.random.default_rng(42)
        direction = rng.standard_normal(x.shape)
        direction /= np.linalg.norm(direction) + 1e-12

        x_plus = x + self._eps * direction
        x_minus = x - self._eps * direction

        y_plus = model_fn(x_plus).ravel()
        y_minus = model_fn(x_minus).ravel()

        directional_grad = (y_plus - y_minus) / (2 * self._eps)
        return float(np.linalg.norm(directional_grad))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        model_fn: Callable[[np.ndarray], np.ndarray],
        input_shape: Tuple[int, ...] = (1, 3, 32, 32),
    ) -> float:
        """
        Compute a training-free proxy score for the given model function.

        Parameters
        ----------
        model_fn : callable
            A function that accepts a numpy array of shape ``input_shape``
            and returns a numpy array (the model's output / activations).
        input_shape : tuple
            Shape of the random input tensor. Default: (1, 3, 32, 32).

        Returns
        -------
        float
            Combined score (gradient_norm + activation_variance).
            Higher is better.
        """
        rng = np.random.default_rng(0)
        x = rng.standard_normal(input_shape).astype(np.float32)

        # Evaluate model
        activations = np.array(model_fn(x), dtype=float).ravel()

        # Signal 1: gradient norm (finite difference along random direction)
        grad_norm = self._numerical_gradient_norm(model_fn, x)

        # Signal 2: activation variance
        act_variance = float(np.var(activations))

        return grad_norm + act_variance
