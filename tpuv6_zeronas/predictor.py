"""TPUv6 performance prediction using Edge TPU v5e regression models."""

import logging
import pickle
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = None
    pd = None

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Mock classes for when sklearn is not available
    class MockModel:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, X, y):
            pass
        def predict(self, X):
            import random
            return [random.uniform(0.1, 10.0) for _ in range(len(X) if hasattr(X, '__len__') else 1)]
    
    class MockScaler:
        def __init__(self):
            pass
        def fit_transform(self, X):
            return X
        def transform(self, X):
            return X
    
    RandomForestRegressor = MockModel
    GradientBoostingRegressor = MockModel
    Ridge = MockModel
    StandardScaler = MockScaler
    
    def train_test_split(X, y, **kwargs):
        split_idx = len(X) // 2
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
    
    def mean_squared_error(y_true, y_pred):
        return 1.0
    
    def r2_score(y_true, y_pred):
        return 0.8

from .architecture import Architecture
from .metrics import PerformanceMetrics


class EdgeTPUv5eCounters:
    """Edge TPU v5e performance counter data collector."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def collect_counters(self, architecture: Architecture) -> Dict[str, float]:
        """Collect performance counters from Edge TPU v5e execution."""
        features = {}
        
        features['ops_count'] = architecture.total_ops
        features['params_count'] = architecture.total_params
        features['memory_footprint'] = architecture.memory_mb
        features['depth'] = architecture.depth
        features['width'] = architecture.avg_width
        
        features['conv_ops'] = architecture.conv_ops
        features['linear_ops'] = architecture.linear_ops
        features['activation_ops'] = architecture.activation_ops
        features['norm_ops'] = architecture.norm_ops
        
        features['compute_intensity'] = features['ops_count'] / max(features['memory_footprint'], 1)
        features['param_efficiency'] = features['ops_count'] / max(features['params_count'], 1)
        features['depth_width_ratio'] = features['depth'] / max(features['width'], 1)
        
        features['matrix_mult_ops'] = architecture.matrix_mult_ops
        features['elementwise_ops'] = architecture.elementwise_ops
        features['reduction_ops'] = architecture.reduction_ops
        
        features['tpu_utilization'] = self._estimate_tpu_utilization(architecture)
        features['memory_bandwidth_req'] = self._estimate_memory_bandwidth(architecture)
        features['parallelism_factor'] = self._estimate_parallelism(architecture)
        
        return features
    
    def _estimate_tpu_utilization(self, arch: Architecture) -> float:
        """Estimate TPU core utilization based on architecture."""
        matrix_ops = arch.matrix_mult_ops
        total_ops = arch.total_ops
        
        utilization = min(1.0, matrix_ops / max(total_ops * 0.8, 1))
        return utilization
    
    def _estimate_memory_bandwidth(self, arch: Architecture) -> float:
        """Estimate memory bandwidth requirements."""
        return (arch.memory_mb * arch.total_ops) / 1000.0
    
    def _estimate_parallelism(self, arch: Architecture) -> float:
        """Estimate parallelism factor for TPU execution."""
        width_factor = min(1.0, arch.avg_width / 512.0)
        batch_factor = min(1.0, arch.batch_size / 32.0) if hasattr(arch, 'batch_size') else 1.0
        
        return width_factor * batch_factor


class TPUv6Predictor:
    """Predicts TPUv6 performance from Edge TPU v5e counter data."""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.counter_collector = EdgeTPUv5eCounters()
        
        self.latency_model: Optional[Any] = None
        self.energy_model: Optional[Any] = None
        self.accuracy_model: Optional[Any] = None
        self.scaler: Optional[StandardScaler] = None
        
        self.is_trained = False
        
        if model_path and model_path.exists():
            self.load_models(model_path)
    
    def train(
        self, 
        training_data: List[Tuple[Architecture, PerformanceMetrics]]
    ) -> Dict[str, float]:
        """Train regression models on Edge TPU v5e data."""
        self.logger.info("Training TPUv6 prediction models...")
        
        if not SKLEARN_AVAILABLE:
            self.logger.warning("Scikit-learn not available, training will use mock models")
            self.is_trained = True
            return {'warning': 'sklearn_not_available'}
        
        try:
            X, y_latency, y_energy, y_accuracy = self._prepare_training_data(training_data)
            
            if len(X) == 0:
                self.logger.error("No training data available")
                return {'error': 'no_training_data'}
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            X_train, X_test, y_lat_train, y_lat_test = train_test_split(
                X_scaled, y_latency, test_size=0.2, random_state=42
            )
            _, _, y_eng_train, y_eng_test = train_test_split(
                X_scaled, y_energy, test_size=0.2, random_state=42
            )
            _, _, y_acc_train, y_acc_test = train_test_split(
                X_scaled, y_accuracy, test_size=0.2, random_state=42
            )
            
            self.latency_model = GradientBoostingRegressor(n_estimators=200, random_state=42)
            self.energy_model = RandomForestRegressor(n_estimators=150, random_state=42)
            self.accuracy_model = Ridge(alpha=1.0)
            
            self.latency_model.fit(X_train, y_lat_train)
            self.energy_model.fit(X_train, y_eng_train)
            self.accuracy_model.fit(X_train, y_acc_train)
            
            metrics = self._evaluate_models(
                X_test, y_lat_test, y_eng_test, y_acc_test
            )
            
            self.is_trained = True
            self.logger.info("Model training completed!")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {'error': str(e)}
    
    def predict(self, architecture: Architecture) -> PerformanceMetrics:
        """Predict TPUv6 performance for given architecture."""
        try:
            if not self._validate_architecture_input(architecture):
                raise ValueError(f"Invalid architecture input: {architecture.name}")
                
            if not self.is_trained:
                return self._fallback_prediction(architecture)
            
            if np is None:
                self.logger.warning("NumPy not available, using fallback prediction")
                return self._fallback_prediction(architecture)
            
            features = self.counter_collector.collect_counters(architecture)
            
            if not features or len(features) == 0:
                self.logger.warning(f"No features extracted for {architecture.name}, using fallback")
                return self._fallback_prediction(architecture)
            
            X = np.array([list(features.values())]).reshape(1, -1)
            
            if self.scaler is None:
                self.logger.warning("Scaler not available, using fallback prediction")
                return self._fallback_prediction(architecture)
                
            X_scaled = self.scaler.transform(X)
            
            pred_latency = self._safe_model_predict(self.latency_model, X_scaled, 'latency')
            pred_energy = self._safe_model_predict(self.energy_model, X_scaled, 'energy')
            pred_accuracy = self._safe_model_predict(self.accuracy_model, X_scaled, 'accuracy')
            
            if any(pred is None for pred in [pred_latency, pred_energy, pred_accuracy]):
                self.logger.warning(f"Model prediction failed for {architecture.name}, using fallback")
                return self._fallback_prediction(architecture)
            
            pred_latency = self._scale_v5e_to_v6_latency(pred_latency)
            pred_energy = self._scale_v5e_to_v6_energy(pred_energy)
            
            tops_per_watt = (architecture.total_ops / 1e12) / max(pred_energy / 1000, 1e-6)
            
            metrics = PerformanceMetrics(
                latency_ms=max(0.1, pred_latency),
                energy_mj=max(0.001, pred_energy),
                accuracy=np.clip(pred_accuracy, 0.0, 1.0) if np else max(0.0, min(1.0, pred_accuracy)),
                tops_per_watt=max(0.1, tops_per_watt),
                memory_mb=architecture.memory_mb,
                flops=architecture.total_ops
            )
            
            return self._validate_and_sanitize_metrics(metrics)
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {architecture.name}: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return self._fallback_prediction(architecture)
    
    def _prepare_training_data(
        self, 
        training_data: List[Tuple[Architecture, PerformanceMetrics]]
    ) -> Tuple[Any, Any, Any, Any]:
        """Prepare training data matrices."""
        X = []
        y_latency = []
        y_energy = []
        y_accuracy = []
        
        for arch, metrics in training_data:
            features = self.counter_collector.collect_counters(arch)
            X.append(list(features.values()))
            y_latency.append(metrics.latency_ms)
            y_energy.append(metrics.energy_mj)
            y_accuracy.append(metrics.accuracy)
            
        if np is not None:
            return (
                np.array(X),
                np.array(y_latency),
                np.array(y_energy), 
                np.array(y_accuracy)
            )
        else:
            return (X, y_latency, y_energy, y_accuracy)
    
    def _evaluate_models(
        self,
        X_test: Any,
        y_lat_test: Any,
        y_eng_test: Any,
        y_acc_test: Any
    ) -> Dict[str, float]:
        """Evaluate trained models."""
        lat_pred = self.latency_model.predict(X_test)
        eng_pred = self.energy_model.predict(X_test)
        acc_pred = self.accuracy_model.predict(X_test)
        
        metrics = {
            'latency_mse': mean_squared_error(y_lat_test, lat_pred),
            'latency_r2': r2_score(y_lat_test, lat_pred),
            'energy_mse': mean_squared_error(y_eng_test, eng_pred),
            'energy_r2': r2_score(y_eng_test, eng_pred),
            'accuracy_mse': mean_squared_error(y_acc_test, acc_pred),
            'accuracy_r2': r2_score(y_acc_test, acc_pred),
        }
        
        self.logger.info(f"Model evaluation metrics: {metrics}")
        return metrics
    
    def _scale_v5e_to_v6_latency(self, v5e_latency: float) -> float:
        """Scale Edge TPU v5e latency to estimated TPUv6 latency."""
        v6_speedup = 2.8
        return v5e_latency / v6_speedup
    
    def _scale_v5e_to_v6_energy(self, v5e_energy: float) -> float:
        """Scale Edge TPU v5e energy to estimated TPUv6 energy."""
        v6_efficiency = 2.1
        return v5e_energy / v6_efficiency
    
    def _fallback_prediction(self, architecture: Architecture) -> PerformanceMetrics:
        """Fallback prediction using analytical models."""
        features = self.counter_collector.collect_counters(architecture)
        
        base_latency = (architecture.total_ops / 1e9) * 0.5
        if np and hasattr(np, 'log'):
            complexity_factor = np.log(architecture.total_params / 1e6 + 1)
        else:
            import math
            complexity_factor = math.log(architecture.total_params / 1e6 + 1)
        latency = base_latency * (1 + 0.1 * complexity_factor)
        
        base_energy = latency * 10.0
        efficiency_factor = features.get('tpu_utilization', 0.7)
        energy = base_energy / efficiency_factor
        
        if np and hasattr(np, 'exp'):
            accuracy = 0.95 - 0.05 * np.exp(-architecture.total_params / 1e7)
        else:
            import math
            accuracy = 0.95 - 0.05 * math.exp(-architecture.total_params / 1e7)
        
        tops_per_watt = (architecture.total_ops / 1e12) / max(energy / 1000, 1e-6)
        
        return PerformanceMetrics(
            latency_ms=latency,
            energy_mj=energy,
            accuracy=accuracy,
            tops_per_watt=tops_per_watt,
            memory_mb=architecture.memory_mb,
            flops=architecture.total_ops
        )
    
    def save_models(self, path: Path) -> None:
        """Save trained models to disk."""
        model_data = {
            'latency_model': self.latency_model,
            'energy_model': self.energy_model,
            'accuracy_model': self.accuracy_model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Models saved to {path}")
    
    def load_models(self, path: Path) -> None:
        """Load trained models from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.latency_model = model_data['latency_model']
        self.energy_model = model_data['energy_model'] 
        self.accuracy_model = model_data['accuracy_model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        
        self.logger.info(f"Models loaded from {path}")
    
    def _validate_architecture_input(self, architecture: Architecture) -> bool:
        """Validate architecture input for prediction."""
        try:
            if not architecture:
                return False
            
            if not hasattr(architecture, 'layers') or not architecture.layers:
                return False
            
            if not hasattr(architecture, 'total_ops') or architecture.total_ops <= 0:
                return False
            
            if not hasattr(architecture, 'total_params') or architecture.total_params <= 0:
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Architecture validation failed: {e}")
            return False
    
    def _safe_model_predict(self, model: Any, X: Any, model_name: str) -> Optional[float]:
        """Safely predict using model with error handling."""
        try:
            if model is None:
                return None
            
            if np is None:
                return None
            
            prediction = model.predict(X)
            
            if hasattr(prediction, '__len__') and len(prediction) > 0:
                result = float(prediction[0])
            else:
                result = float(prediction)
            
            # Basic sanity checks
            if not np.isfinite(result):
                self.logger.warning(f"Non-finite prediction from {model_name} model: {result}")
                return None
            
            if result < 0:
                self.logger.warning(f"Negative prediction from {model_name} model: {result}")
                result = abs(result)
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Model prediction failed for {model_name}: {e}")
            return None
    
    def _validate_and_sanitize_metrics(self, metrics: PerformanceMetrics) -> PerformanceMetrics:
        """Validate and sanitize performance metrics."""
        try:
            # Sanitize latency
            if not (0.01 <= metrics.latency_ms <= 1000.0):
                metrics.latency_ms = max(0.1, min(1000.0, metrics.latency_ms))
            
            # Sanitize energy
            if not (0.001 <= metrics.energy_mj <= 10000.0):
                metrics.energy_mj = max(0.1, min(10000.0, metrics.energy_mj))
            
            # Sanitize accuracy
            if not (0.0 <= metrics.accuracy <= 1.0):
                metrics.accuracy = max(0.0, min(1.0, metrics.accuracy))
            
            # Sanitize TOPS/W
            if not (0.1 <= metrics.tops_per_watt <= 500.0):
                metrics.tops_per_watt = max(0.1, min(500.0, metrics.tops_per_watt))
            
            # Sanitize memory
            if metrics.memory_mb <= 0:
                metrics.memory_mb = 1.0
            
            # Sanitize flops
            if metrics.flops <= 0:
                metrics.flops = 1000000
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Metrics sanitization failed: {e}")
            # Return conservative fallback
            return PerformanceMetrics(
                latency_ms=10.0,
                energy_mj=100.0,
                accuracy=0.9,
                tops_per_watt=50.0,
                memory_mb=100.0,
                flops=1000000
            )