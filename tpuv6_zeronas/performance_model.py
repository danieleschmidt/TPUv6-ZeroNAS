"""
PerformanceModel: Polynomial regression mapping layer parameters to
latency and energy predictions for TPU v6 workloads.
"""

import numpy as np
from itertools import combinations_with_replacement


class PerformanceModel:
    """
    Polynomial regression model that predicts latency and energy consumption
    from architecture feature vectors.

    Input features X: array of shape (n_samples, 3)
        - num_params: number of model parameters (millions)
        - num_ops: number of operations (GFLOPs)
        - memory_mb: peak memory usage in MB
    """

    def __init__(self, degree: int = 2):
        self.degree = degree
        self._coef_latency = None
        self._coef_energy = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _poly_features(self, X: np.ndarray) -> np.ndarray:
        """Expand raw features into polynomial feature matrix (no sklearn needed)."""
        X = np.atleast_2d(X).astype(float)
        n_samples, n_features = X.shape
        feature_cols = [np.ones((n_samples, 1))]

        # Generate all monomial terms up to `self.degree`
        for d in range(1, self.degree + 1):
            for combo in combinations_with_replacement(range(n_features), d):
                col = np.ones(n_samples)
                for idx in combo:
                    col = col * X[:, idx]
                feature_cols.append(col.reshape(-1, 1))

        return np.hstack(feature_cols)

    @staticmethod
    def _lstsq_fit(Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Least-squares regression via numpy."""
        coef, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)
        return coef

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y_latency: np.ndarray, y_energy: np.ndarray):
        """
        Fit polynomial regression models for latency and energy.

        Parameters
        ----------
        X : array-like, shape (n_samples, 3)
            Feature matrix [num_params, num_ops, memory_mb].
        y_latency : array-like, shape (n_samples,)
            Target latency values (ms).
        y_energy : array-like, shape (n_samples,)
            Target energy values (mJ).
        """
        X = np.array(X, dtype=float)
        y_latency = np.array(y_latency, dtype=float)
        y_energy = np.array(y_energy, dtype=float)

        Phi = self._poly_features(X)
        self._coef_latency = self._lstsq_fit(Phi, y_latency)
        self._coef_energy = self._lstsq_fit(Phi, y_energy)
        self._fitted = True
        return self

    def predict_latency(self, X: np.ndarray) -> np.ndarray:
        """Predict latency (ms) for given feature matrix X."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before calling predict_latency.")
        Phi = self._poly_features(np.array(X, dtype=float))
        return Phi @ self._coef_latency

    def predict_energy(self, X: np.ndarray) -> np.ndarray:
        """Predict energy (mJ) for given feature matrix X."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before calling predict_energy.")
        Phi = self._poly_features(np.array(X, dtype=float))
        return Phi @ self._coef_energy
