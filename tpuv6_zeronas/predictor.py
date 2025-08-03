"""TPUv6 performance prediction using Edge TPU v5e regression models."""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

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
        
        X, y_latency, y_energy, y_accuracy = self._prepare_training_data(training_data)
        
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
    
    def predict(self, architecture: Architecture) -> PerformanceMetrics:
        """Predict TPUv6 performance for given architecture."""
        if not self.is_trained:
            return self._fallback_prediction(architecture)
            
        features = self.counter_collector.collect_counters(architecture)
        X = np.array([list(features.values())]).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        pred_latency = self.latency_model.predict(X_scaled)[0]
        pred_energy = self.energy_model.predict(X_scaled)[0]
        pred_accuracy = self.accuracy_model.predict(X_scaled)[0]
        
        pred_latency = self._scale_v5e_to_v6_latency(pred_latency)
        pred_energy = self._scale_v5e_to_v6_energy(pred_energy)
        
        tops_per_watt = (architecture.total_ops / 1e12) / max(pred_energy, 1e-6)
        
        return PerformanceMetrics(
            latency_ms=max(0.1, pred_latency),
            energy_mj=max(0.001, pred_energy),
            accuracy=np.clip(pred_accuracy, 0.0, 1.0),
            tops_per_watt=tops_per_watt,
            memory_mb=architecture.memory_mb,
            flops=architecture.total_ops
        )
    
    def _prepare_training_data(
        self, 
        training_data: List[Tuple[Architecture, PerformanceMetrics]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            
        return (
            np.array(X),
            np.array(y_latency),
            np.array(y_energy), 
            np.array(y_accuracy)
        )
    
    def _evaluate_models(
        self,
        X_test: np.ndarray,
        y_lat_test: np.ndarray,
        y_eng_test: np.ndarray,
        y_acc_test: np.ndarray
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
        complexity_factor = np.log(architecture.total_params / 1e6 + 1)
        latency = base_latency * (1 + 0.1 * complexity_factor)
        
        base_energy = latency * 10.0
        efficiency_factor = features.get('tpu_utilization', 0.7)
        energy = base_energy / efficiency_factor
        
        accuracy = 0.95 - 0.05 * np.exp(-architecture.total_params / 1e7)
        
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