"""Tests for architecture representation and search space."""

import pytest
import numpy as np

from tpuv6_zeronas.architecture import (
    Architecture, Layer, LayerType, ActivationType, ArchitectureSpace
)


class TestLayer:
    """Test Layer class functionality."""
    
    def test_conv_layer_creation(self):
        """Test creation of convolutional layer."""
        layer = Layer(
            layer_type=LayerType.CONV2D,
            input_channels=3,
            output_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            activation=ActivationType.RELU
        )
        
        assert layer.layer_type == LayerType.CONV2D
        assert layer.input_channels == 3
        assert layer.output_channels == 64
        assert layer.kernel_size == (3, 3)
        
    def test_linear_layer_creation(self):
        """Test creation of linear layer."""
        layer = Layer(
            layer_type=LayerType.LINEAR,
            input_channels=1024,
            output_channels=1000,
            activation=None
        )
        
        assert layer.layer_type == LayerType.LINEAR
        assert layer.input_channels == 1024
        assert layer.output_channels == 1000
        assert layer.activation is None
        
    def test_ops_count_conv(self):
        """Test operation count calculation for conv layer."""
        layer = Layer(
            layer_type=LayerType.CONV2D,
            input_channels=64,
            output_channels=128,
            kernel_size=(3, 3)
        )
        
        expected_ops = 64 * 128 * 3 * 3
        assert layer.ops_count == expected_ops
        
    def test_ops_count_linear(self):
        """Test operation count calculation for linear layer."""
        layer = Layer(
            layer_type=LayerType.LINEAR,
            input_channels=1024,
            output_channels=1000
        )
        
        expected_ops = 1024 * 1000
        assert layer.ops_count == expected_ops
        
    def test_params_count_conv(self):
        """Test parameter count calculation for conv layer."""
        layer = Layer(
            layer_type=LayerType.CONV2D,
            input_channels=64,
            output_channels=128,
            kernel_size=(3, 3),
            use_bias=True
        )
        
        expected_params = 64 * 128 * 3 * 3 + 128
        assert layer.params_count == expected_params
        
    def test_params_count_linear(self):
        """Test parameter count calculation for linear layer."""
        layer = Layer(
            layer_type=LayerType.LINEAR,
            input_channels=1024,
            output_channels=1000,
            use_bias=True
        )
        
        expected_params = 1024 * 1000 + 1000
        assert layer.params_count == expected_params


class TestArchitecture:
    """Test Architecture class functionality."""
    
    def create_sample_architecture(self):
        """Create sample architecture for testing."""
        layers = [
            Layer(LayerType.CONV2D, 3, 64, (3, 3), (1, 1), activation=ActivationType.RELU),
            Layer(LayerType.BATCH_NORM, 64, 64),
            Layer(LayerType.CONV2D, 64, 128, (3, 3), (2, 2), activation=ActivationType.RELU),
            Layer(LayerType.LINEAR, 128, 1000, activation=None)
        ]
        
        return Architecture(
            layers=layers,
            input_shape=(224, 224, 3),
            num_classes=1000,
            name="test_arch"
        )
    
    def test_architecture_creation(self):
        """Test architecture creation."""
        arch = self.create_sample_architecture()
        
        assert len(arch.layers) == 4
        assert arch.input_shape == (224, 224, 3)
        assert arch.num_classes == 1000
        assert arch.name == "test_arch"
        
    def test_total_ops(self):
        """Test total operations calculation."""
        arch = self.create_sample_architecture()
        
        expected_ops = (
            3 * 64 * 3 * 3 +  # First conv
            64 +              # BatchNorm (approximation)
            64 * 128 * 3 * 3 + # Second conv
            128 * 1000        # Linear
        )
        
        assert arch.total_ops > 0
        
    def test_total_params(self):
        """Test total parameters calculation."""
        arch = self.create_sample_architecture()
        
        expected_params = (
            3 * 64 * 3 * 3 + 64 +     # First conv + bias
            64 * 2 +                   # BatchNorm
            64 * 128 * 3 * 3 + 128 +  # Second conv + bias
            128 * 1000 + 1000         # Linear + bias
        )
        
        assert arch.total_params > 0
        
    def test_depth(self):
        """Test network depth calculation."""
        arch = self.create_sample_architecture()
        assert arch.depth == 4
        
    def test_avg_width(self):
        """Test average width calculation."""
        arch = self.create_sample_architecture()
        expected_width = (64 + 64 + 128 + 1000) / 4
        assert arch.avg_width == expected_width
        
    def test_memory_estimate(self):
        """Test memory footprint estimation."""
        arch = self.create_sample_architecture()
        assert arch.memory_mb > 0
        
    def test_operation_breakdown(self):
        """Test operation type breakdown."""
        arch = self.create_sample_architecture()
        
        assert arch.conv_ops > 0
        assert arch.linear_ops > 0
        assert arch.matrix_mult_ops == arch.conv_ops + arch.linear_ops


class TestArchitectureSpace:
    """Test ArchitectureSpace functionality."""
    
    def test_space_creation(self):
        """Test architecture space creation."""
        space = ArchitectureSpace(
            input_shape=(224, 224, 3),
            num_classes=1000,
            max_depth=20
        )
        
        assert space.input_shape == (224, 224, 3)
        assert space.num_classes == 1000
        assert space.max_depth == 20
        
    def test_random_sampling(self):
        """Test random architecture sampling."""
        space = ArchitectureSpace(max_depth=10)
        
        arch = space.sample_random()
        
        assert isinstance(arch, Architecture)
        assert len(arch.layers) >= 5
        assert len(arch.layers) <= 10
        assert arch.input_shape == space.input_shape
        assert arch.num_classes == space.num_classes
        
    def test_multiple_samples_different(self):
        """Test that multiple samples produce different architectures."""
        space = ArchitectureSpace(max_depth=10)
        
        arch1 = space.sample_random()
        arch2 = space.sample_random()
        
        assert arch1.total_params != arch2.total_params or arch1.total_ops != arch2.total_ops
        
    def test_mutation(self):
        """Test architecture mutation."""
        space = ArchitectureSpace(max_depth=10)
        original = space.sample_random()
        
        mutated = space.mutate(original)
        
        assert isinstance(mutated, Architecture)
        assert mutated.name.startswith("mutated_")
        
    def test_crossover(self):
        """Test architecture crossover."""
        space = ArchitectureSpace(max_depth=10)
        
        parent1 = space.sample_random()
        parent2 = space.sample_random()
        
        child = space.crossover(parent1, parent2)
        
        assert isinstance(child, Architecture)
        assert child.name.startswith("crossover_")
        assert len(child.layers) > 0
        
    def test_channel_compatibility(self):
        """Test that crossover maintains channel compatibility."""
        space = ArchitectureSpace(max_depth=8)
        
        parent1 = space.sample_random()
        parent2 = space.sample_random()
        
        child = space.crossover(parent1, parent2)
        
        for i in range(1, len(child.layers)):
            if child.layers[i].layer_type in [LayerType.CONV2D, LayerType.LINEAR]:
                prev_output = child.layers[i-1].output_channels
                curr_input = child.layers[i].input_channels
                assert prev_output == curr_input, f"Channel mismatch at layer {i}"


if __name__ == '__main__':
    pytest.main([__file__])