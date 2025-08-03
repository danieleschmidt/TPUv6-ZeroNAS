"""Tests for TPUv6-specific optimizations."""

import pytest
import numpy as np

from tpuv6_zeronas.optimizations import (
    TPUv6Optimizer, TPUv6Config, QuantizationOptimizer, 
    SparsityOptimizer, TPUv6OptimizationType
)
from tpuv6_zeronas.architecture import Architecture, Layer, LayerType, ActivationType


class TestTPUv6Config:
    """Test TPUv6 hardware configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TPUv6Config()
        
        assert config.matrix_units == 4
        assert config.vector_units == 2
        assert config.memory_bandwidth_gbps == 900.0
        assert config.peak_tops == 275.0
        assert config.target_tops_per_watt == 75.0
        assert config.matrix_unit_size == (256, 256)
        assert config.support_bf16 == True
        assert config.support_int8 == True
        assert config.support_sparsity == True
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TPUv6Config(
            matrix_units=8,
            target_tops_per_watt=80.0,
            support_bf16=False
        )
        
        assert config.matrix_units == 8
        assert config.target_tops_per_watt == 80.0
        assert config.support_bf16 == False


class TestTPUv6Optimizer:
    """Test TPUv6 optimizer functionality."""
    
    def create_sample_architecture(self):
        """Create sample architecture for testing."""
        layers = [
            Layer(LayerType.CONV2D, 3, 64, (3, 3), activation=ActivationType.RELU),
            Layer(LayerType.CONV2D, 64, 128, (3, 3), activation=ActivationType.SIGMOID),
            Layer(LayerType.LINEAR, 128, 1000, activation=None)
        ]
        
        return Architecture(
            layers=layers,
            input_shape=(224, 224, 3),
            num_classes=1000,
            name="test_arch"
        )
    
    def test_optimizer_creation(self):
        """Test optimizer creation."""
        optimizer = TPUv6Optimizer()
        
        assert optimizer.config is not None
        assert isinstance(optimizer.config, TPUv6Config)
        
    def test_optimizer_with_custom_config(self):
        """Test optimizer with custom configuration."""
        config = TPUv6Config(target_tops_per_watt=80.0)
        optimizer = TPUv6Optimizer(config)
        
        assert optimizer.config.target_tops_per_watt == 80.0
        
    def test_architecture_optimization(self):
        """Test full architecture optimization."""
        optimizer = TPUv6Optimizer()
        original_arch = self.create_sample_architecture()
        
        optimized_arch = optimizer.optimize_architecture(original_arch)
        
        assert isinstance(optimized_arch, Architecture)
        assert optimized_arch.name.endswith("_tpuv6_optimized")
        assert len(optimized_arch.layers) >= len(original_arch.layers)
        
    def test_matrix_operation_optimization(self):
        """Test matrix operation optimization."""
        optimizer = TPUv6Optimizer()
        original_layers = self.create_sample_architecture().layers.copy()
        
        optimized_layers = optimizer._optimize_matrix_operations(original_layers)
        
        assert len(optimized_layers) == len(original_layers)
        
        for layer in optimized_layers:
            if layer.layer_type in [LayerType.CONV2D, LayerType.LINEAR]:
                assert layer.output_channels > 0
                
    def test_optimal_size_rounding(self):
        """Test optimal size rounding for matrix units."""
        optimizer = TPUv6Optimizer()
        
        rounded_64 = optimizer._round_to_optimal_size(64, 256)
        rounded_100 = optimizer._round_to_optimal_size(100, 256)
        rounded_300 = optimizer._round_to_optimal_size(300, 256)
        
        assert rounded_64 == 64
        assert rounded_100 >= 100
        assert rounded_300 >= 300
        
    def test_memory_layout_optimization(self):
        """Test memory layout optimization."""
        optimizer = TPUv6Optimizer()
        original_layers = self.create_sample_architecture().layers.copy()
        
        optimized_layers = optimizer._optimize_memory_layout(original_layers)
        
        assert len(optimized_layers) >= len(original_layers)
        
    def test_layer_memory_estimation(self):
        """Test layer memory estimation."""
        optimizer = TPUv6Optimizer()
        layer = Layer(LayerType.CONV2D, 64, 128, (3, 3))
        
        memory_mb = optimizer._estimate_layer_memory(layer)
        
        assert memory_mb > 0
        assert isinstance(memory_mb, float)
        
    def test_activation_optimization(self):
        """Test activation function optimization."""
        optimizer = TPUv6Optimizer()
        
        layers = [
            Layer(LayerType.CONV2D, 3, 64, activation=ActivationType.RELU),
            Layer(LayerType.CONV2D, 64, 128, activation=ActivationType.SIGMOID),
            Layer(LayerType.LINEAR, 128, 1000, activation=ActivationType.TANH)
        ]
        
        optimized_layers = optimizer._optimize_activation_functions(layers)
        
        assert optimized_layers[0].activation == ActivationType.GELU
        assert optimized_layers[1].activation == ActivationType.SWISH
        
    def test_precision_optimization(self):
        """Test precision optimization."""
        optimizer = TPUv6Optimizer()
        layers = self.create_sample_architecture().layers.copy()
        
        optimized_layers = optimizer._optimize_precision(layers)
        
        for layer in optimized_layers:
            if layer.layer_type in [LayerType.CONV2D, LayerType.LINEAR]:
                assert hasattr(layer, 'precision')
                
    def test_tops_efficiency_estimation(self):
        """Test TOPS efficiency estimation."""
        optimizer = TPUv6Optimizer()
        arch = self.create_sample_architecture()
        
        efficiency = optimizer._estimate_tops_efficiency(arch)
        
        assert isinstance(efficiency, float)
        assert efficiency > 0
        assert efficiency <= optimizer.config.target_tops_per_watt * 1.3
        
    def test_optimization_report(self):
        """Test optimization report generation."""
        optimizer = TPUv6Optimizer()
        original = self.create_sample_architecture()
        optimized = optimizer.optimize_architecture(original)
        
        report = optimizer.get_optimization_report(original, optimized)
        
        assert 'original_ops' in report
        assert 'optimized_ops' in report
        assert 'ops_change_pct' in report
        assert 'original_params' in report
        assert 'optimized_params' in report
        assert 'params_change_pct' in report
        assert 'estimated_tops_efficiency' in report
        assert 'optimization_applied' in report
        
        assert isinstance(report['estimated_tops_efficiency'], float)
        assert isinstance(report['optimization_applied'], list)


class TestQuantizationOptimizer:
    """Test quantization optimizer."""
    
    def create_sample_architecture(self):
        """Create sample architecture for testing."""
        layers = [
            Layer(LayerType.CONV2D, 3, 64, (3, 3)),
            Layer(LayerType.LINEAR, 64, 1000)
        ]
        
        return Architecture(
            layers=layers,
            input_shape=(224, 224, 3),
            num_classes=1000
        )
    
    def test_quantization_optimizer_creation(self):
        """Test quantization optimizer creation."""
        optimizer = QuantizationOptimizer()
        
        assert optimizer.config is not None
        assert isinstance(optimizer.config, TPUv6Config)
        
    def test_quantization_application(self):
        """Test quantization application."""
        optimizer = QuantizationOptimizer()
        arch = self.create_sample_architecture()
        
        quantized_arch = optimizer.apply_quantization(arch)
        
        assert isinstance(quantized_arch, Architecture)
        assert quantized_arch.name.endswith("_quantized")
        
    def test_layer_quantization(self):
        """Test individual layer quantization."""
        optimizer = QuantizationOptimizer()
        layer = Layer(LayerType.CONV2D, 3, 64, (3, 3))
        
        quantized_layer = optimizer._quantize_layer(layer)
        
        assert hasattr(quantized_layer, 'quantized')
        assert hasattr(quantized_layer, 'precision')
        assert hasattr(quantized_layer, 'scale_factor')
        assert getattr(quantized_layer, 'quantized') == True
        assert getattr(quantized_layer, 'precision') == 'int8'
        
    def test_quantization_without_support(self):
        """Test quantization when INT8 not supported."""
        config = TPUv6Config(support_int8=False)
        optimizer = QuantizationOptimizer(config)
        arch = self.create_sample_architecture()
        
        result_arch = optimizer.apply_quantization(arch)
        
        assert result_arch == arch


class TestSparsityOptimizer:
    """Test sparsity optimizer."""
    
    def create_sample_architecture(self):
        """Create sample architecture for testing."""
        layers = [
            Layer(LayerType.CONV2D, 3, 64, (3, 3)),
            Layer(LayerType.LINEAR, 64, 1000)
        ]
        
        return Architecture(
            layers=layers,
            input_shape=(224, 224, 3),
            num_classes=1000
        )
    
    def test_sparsity_optimizer_creation(self):
        """Test sparsity optimizer creation."""
        optimizer = SparsityOptimizer()
        
        assert optimizer.config is not None
        assert isinstance(optimizer.config, TPUv6Config)
        
    def test_sparsity_application(self):
        """Test sparsity application."""
        optimizer = SparsityOptimizer()
        arch = self.create_sample_architecture()
        
        sparse_arch = optimizer.apply_sparsity(arch, sparsity_ratio=0.5)
        
        assert isinstance(sparse_arch, Architecture)
        assert sparse_arch.name.endswith("_sparse_50pct")
        
    def test_layer_sparsity(self):
        """Test individual layer sparsity."""
        optimizer = SparsityOptimizer()
        layer = Layer(LayerType.CONV2D, 3, 64, (3, 3))
        
        sparse_layer = optimizer._apply_layer_sparsity(layer, 0.3)
        
        assert hasattr(sparse_layer, 'sparse')
        assert hasattr(sparse_layer, 'sparsity_ratio')
        assert hasattr(sparse_layer, 'effective_params')
        assert getattr(sparse_layer, 'sparse') == True
        assert getattr(sparse_layer, 'sparsity_ratio') == 0.3
        
        original_params = layer.params_count
        effective_params = getattr(sparse_layer, 'effective_params')
        expected_params = int(original_params * 0.7)
        assert effective_params == expected_params
        
    def test_sparsity_without_support(self):
        """Test sparsity when not supported."""
        config = TPUv6Config(support_sparsity=False)
        optimizer = SparsityOptimizer(config)
        arch = self.create_sample_architecture()
        
        result_arch = optimizer.apply_sparsity(arch)
        
        assert result_arch == arch
        
    def test_various_sparsity_ratios(self):
        """Test different sparsity ratios."""
        optimizer = SparsityOptimizer()
        arch = self.create_sample_architecture()
        
        for ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
            sparse_arch = optimizer.apply_sparsity(arch, sparsity_ratio=ratio)
            
            assert isinstance(sparse_arch, Architecture)
            assert sparse_arch.name.endswith(f"_sparse_{int(ratio*100)}pct")


if __name__ == '__main__':
    pytest.main([__file__])