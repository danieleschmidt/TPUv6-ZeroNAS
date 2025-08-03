"""Tests for TPUv6 performance predictor."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from tpuv6_zeronas.predictor import TPUv6Predictor, EdgeTPUv5eCounters
from tpuv6_zeronas.architecture import Architecture, Layer, LayerType
from tpuv6_zeronas.metrics import PerformanceMetrics


class TestEdgeTPUv5eCounters:
    """Test Edge TPU v5e counter collection."""
    
    def create_sample_architecture(self):
        """Create sample architecture for testing."""
        layers = [
            Layer(LayerType.CONV2D, 3, 64, (3, 3)),
            Layer(LayerType.CONV2D, 64, 128, (3, 3)),
            Layer(LayerType.LINEAR, 128, 1000)
        ]
        
        return Architecture(
            layers=layers,
            input_shape=(224, 224, 3),
            num_classes=1000
        )
    
    def test_counter_collection(self):
        """Test counter collection from architecture."""
        counter_collector = EdgeTPUv5eCounters()
        arch = self.create_sample_architecture()
        
        features = counter_collector.collect_counters(arch)
        
        assert 'ops_count' in features
        assert 'params_count' in features
        assert 'memory_footprint' in features
        assert 'depth' in features
        assert 'width' in features
        assert 'conv_ops' in features
        assert 'linear_ops' in features
        
        assert features['ops_count'] > 0
        assert features['params_count'] > 0
        assert features['depth'] == 3
        
    def test_derived_features(self):
        """Test derived feature calculations."""
        counter_collector = EdgeTPUv5eCounters()
        arch = self.create_sample_architecture()
        
        features = counter_collector.collect_counters(arch)
        
        assert 'compute_intensity' in features
        assert 'param_efficiency' in features
        assert 'depth_width_ratio' in features
        assert 'tpu_utilization' in features
        assert 'memory_bandwidth_req' in features
        assert 'parallelism_factor' in features
        
        assert features['compute_intensity'] > 0
        assert 0 <= features['tpu_utilization'] <= 1
        assert features['parallelism_factor'] > 0


class TestTPUv6Predictor:
    """Test TPUv6 performance predictor."""
    
    def create_sample_architecture(self):
        """Create sample architecture for testing."""
        layers = [
            Layer(LayerType.CONV2D, 3, 64, (3, 3)),
            Layer(LayerType.CONV2D, 64, 128, (3, 3)),
            Layer(LayerType.LINEAR, 128, 1000)
        ]
        
        return Architecture(
            layers=layers,
            input_shape=(224, 224, 3),
            num_classes=1000
        )
    
    def test_predictor_creation(self):
        """Test predictor creation."""
        predictor = TPUv6Predictor()
        
        assert predictor is not None
        assert not predictor.is_trained
        assert predictor.counter_collector is not None
        
    def test_fallback_prediction(self):
        """Test fallback prediction without trained model."""
        predictor = TPUv6Predictor()
        arch = self.create_sample_architecture()
        
        metrics = predictor.predict(arch)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.latency_ms > 0
        assert metrics.energy_mj > 0
        assert 0 <= metrics.accuracy <= 1
        assert metrics.tops_per_watt > 0
        assert metrics.memory_mb > 0
        
    def test_prediction_consistency(self):
        """Test that predictions are consistent for same architecture."""
        predictor = TPUv6Predictor()
        arch = self.create_sample_architecture()
        
        metrics1 = predictor.predict(arch)
        metrics2 = predictor.predict(arch)
        
        assert metrics1.latency_ms == metrics2.latency_ms
        assert metrics1.energy_mj == metrics2.energy_mj
        assert metrics1.accuracy == metrics2.accuracy
        
    def test_prediction_variation(self):
        """Test that different architectures give different predictions."""
        predictor = TPUv6Predictor()
        
        arch1 = self.create_sample_architecture()
        
        layers2 = [
            Layer(LayerType.CONV2D, 3, 32, (3, 3)),
            Layer(LayerType.LINEAR, 32, 1000)
        ]
        arch2 = Architecture(layers2, (224, 224, 3), 1000)
        
        metrics1 = predictor.predict(arch1)
        metrics2 = predictor.predict(arch2)
        
        assert metrics1.latency_ms != metrics2.latency_ms or metrics1.energy_mj != metrics2.energy_mj
        
    def test_v5e_to_v6_scaling(self):
        """Test v5e to v6 scaling functions."""
        predictor = TPUv6Predictor()
        
        v5e_latency = 10.0
        v6_latency = predictor._scale_v5e_to_v6_latency(v5e_latency)
        assert v6_latency < v5e_latency
        
        v5e_energy = 100.0
        v6_energy = predictor._scale_v5e_to_v6_energy(v5e_energy)
        assert v6_energy < v5e_energy
        
    def test_model_save_load(self):
        """Test model save and load functionality."""
        predictor = TPUv6Predictor()
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = Path(f.name)
        
        try:
            predictor.save_models(model_path)
            assert model_path.exists()
            
            new_predictor = TPUv6Predictor()
            new_predictor.load_models(model_path)
            
        finally:
            if model_path.exists():
                model_path.unlink()
                
    def create_mock_training_data(self):
        """Create mock training data."""
        training_data = []
        
        for i in range(10):
            layers = [
                Layer(LayerType.CONV2D, 3, 32 + i*16, (3, 3)),
                Layer(LayerType.LINEAR, 32 + i*16, 1000)
            ]
            arch = Architecture(layers, (224, 224, 3), 1000)
            
            metrics = PerformanceMetrics(
                latency_ms=5.0 + i,
                energy_mj=50.0 + i*10,
                accuracy=0.9 + i*0.01,
                tops_per_watt=60.0 + i*2,
                memory_mb=100.0 + i*20,
                flops=arch.total_ops
            )
            
            training_data.append((arch, metrics))
            
        return training_data
        
    def test_training_data_preparation(self):
        """Test training data preparation."""
        predictor = TPUv6Predictor()
        training_data = self.create_mock_training_data()
        
        X, y_lat, y_eng, y_acc = predictor._prepare_training_data(training_data)
        
        assert X.shape[0] == len(training_data)
        assert len(y_lat) == len(training_data)
        assert len(y_eng) == len(training_data)
        assert len(y_acc) == len(training_data)
        
        assert X.shape[1] > 0  # Should have features
        assert np.all(y_lat > 0)
        assert np.all(y_eng > 0)
        assert np.all((y_acc >= 0) & (y_acc <= 1))
        
    def test_model_training(self):
        """Test model training process."""
        predictor = TPUv6Predictor()
        training_data = self.create_mock_training_data()
        
        metrics = predictor.train(training_data)
        
        assert predictor.is_trained
        assert 'latency_mse' in metrics
        assert 'energy_mse' in metrics
        assert 'accuracy_mse' in metrics
        assert 'latency_r2' in metrics
        assert 'energy_r2' in metrics
        assert 'accuracy_r2' in metrics
        
        assert predictor.latency_model is not None
        assert predictor.energy_model is not None
        assert predictor.accuracy_model is not None
        assert predictor.scaler is not None


if __name__ == '__main__':
    pytest.main([__file__])