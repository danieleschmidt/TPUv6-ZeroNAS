"""Neural architecture representation and search space definition."""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

try:
    import numpy as np
except ImportError:
    # Mock numpy for basic operations
    class MockNumPy:
        @staticmethod
        def clip(x, a, b):
            return max(a, min(b, x))
    np = MockNumPy()


class LayerType(Enum):
    """Supported layer types for TPUv6 optimization."""
    CONV2D = "conv2d"
    DEPTHWISE_CONV = "depthwise_conv"
    POINTWISE_CONV = "pointwise_conv"
    LINEAR = "linear"
    BATCH_NORM = "batch_norm"
    LAYER_NORM = "layer_norm"
    RELU = "relu"
    GELU = "gelu"
    SWISH = "swish"
    ATTENTION = "attention"
    RESIDUAL = "residual"
    POOLING = "pooling"


class ActivationType(Enum):
    """Activation function types."""
    RELU = "relu"
    GELU = "gelu"
    SWISH = "swish"
    SIGMOID = "sigmoid"
    TANH = "tanh"


@dataclass
class Layer:
    """Individual layer specification."""
    layer_type: LayerType
    input_channels: int
    output_channels: int
    kernel_size: Optional[Tuple[int, int]] = None
    stride: Optional[Tuple[int, int]] = None
    padding: Optional[str] = None
    activation: Optional[ActivationType] = None
    use_bias: bool = True
    
    @property
    def ops_count(self) -> int:
        """Estimate operation count for this layer."""
        if self.layer_type == LayerType.CONV2D:
            if self.kernel_size:
                return (self.input_channels * self.output_channels * 
                       self.kernel_size[0] * self.kernel_size[1])
            return self.input_channels * self.output_channels
            
        elif self.layer_type == LayerType.LINEAR:
            return self.input_channels * self.output_channels
            
        elif self.layer_type == LayerType.ATTENTION:
            return self.input_channels * self.output_channels * 4
        
        return self.input_channels
    
    @property
    def params_count(self) -> int:
        """Estimate parameter count for this layer."""
        if self.layer_type in [LayerType.CONV2D, LayerType.DEPTHWISE_CONV]:
            if self.kernel_size:
                params = (self.input_channels * self.output_channels * 
                         self.kernel_size[0] * self.kernel_size[1])
            else:
                params = self.input_channels * self.output_channels
                
            if self.use_bias:
                params += self.output_channels
            return params
            
        elif self.layer_type == LayerType.LINEAR:
            params = self.input_channels * self.output_channels
            if self.use_bias:
                params += self.output_channels
            return params
            
        elif self.layer_type in [LayerType.BATCH_NORM, LayerType.LAYER_NORM]:
            return self.output_channels * 2
        
        return 0


@dataclass
class Architecture:
    """Complete neural architecture specification."""
    layers: List[Layer]
    input_shape: Tuple[int, int, int]
    num_classes: int
    name: Optional[str] = None
    
    @property
    def total_ops(self) -> int:
        """Total operation count across all layers."""
        return sum(layer.ops_count for layer in self.layers)
    
    @property
    def total_params(self) -> int:
        """Total parameter count across all layers."""
        return sum(layer.params_count for layer in self.layers)
    
    @property 
    def depth(self) -> int:
        """Network depth (number of layers)."""
        return len(self.layers)
    
    @property
    def avg_width(self) -> float:
        """Average layer width (channels)."""
        if not self.layers:
            return 0.0
        return sum(layer.output_channels for layer in self.layers) / len(self.layers)
    
    @property
    def max_channels(self) -> int:
        """Maximum number of channels across all layers."""
        if not self.layers:
            return 0
        return max(layer.output_channels for layer in self.layers)
    
    @property
    def memory_mb(self) -> float:
        """Estimated memory footprint in MB."""
        h, w, c = self.input_shape
        current_size = h * w * c
        total_memory = current_size
        
        for layer in self.layers:
            if layer.layer_type == LayerType.CONV2D and layer.stride:
                h = h // layer.stride[0]
                w = w // layer.stride[1]
            current_size = h * w * layer.output_channels
            total_memory += current_size
            
        return (total_memory * 4) / (1024 * 1024)
    
    def copy(self) -> 'Architecture':
        """Create a deep copy of the architecture."""
        import copy
        copied_layers = []
        for layer in self.layers:
            # Create new layer with same attributes
            new_layer = Layer(
                layer_type=layer.layer_type,
                input_channels=layer.input_channels,
                output_channels=layer.output_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                activation=layer.activation,
                use_bias=layer.use_bias
            )
            # Copy any additional attributes
            for attr_name, attr_value in layer.__dict__.items():
                if not hasattr(new_layer, attr_name):
                    setattr(new_layer, attr_name, copy.deepcopy(attr_value))
            copied_layers.append(new_layer)
        
        return Architecture(
            layers=copied_layers,
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            name=self.name
        )
    
    @property
    def conv_ops(self) -> int:
        """Convolution operation count."""
        return sum(layer.ops_count for layer in self.layers 
                  if layer.layer_type in [LayerType.CONV2D, LayerType.DEPTHWISE_CONV])
    
    @property
    def linear_ops(self) -> int:
        """Linear operation count."""
        return sum(layer.ops_count for layer in self.layers 
                  if layer.layer_type == LayerType.LINEAR)
    
    @property
    def activation_ops(self) -> int:
        """Activation operation count."""
        return sum(layer.output_channels for layer in self.layers 
                  if layer.activation is not None)
    
    @property
    def norm_ops(self) -> int:
        """Normalization operation count.""" 
        return sum(layer.output_channels for layer in self.layers
                  if layer.layer_type in [LayerType.BATCH_NORM, LayerType.LAYER_NORM])
    
    @property
    def matrix_mult_ops(self) -> int:
        """Matrix multiplication operation count."""
        return self.conv_ops + self.linear_ops
    
    @property
    def elementwise_ops(self) -> int:
        """Element-wise operation count."""
        return self.activation_ops + self.norm_ops
    
    @property
    def reduction_ops(self) -> int:
        """Reduction operation count."""
        return sum(layer.ops_count for layer in self.layers
                  if layer.layer_type == LayerType.POOLING)


class ArchitectureSpace:
    """Search space definition for neural architectures."""
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 1000,
        max_depth: int = 20,
        channel_choices: List[int] = None,
        kernel_choices: List[Tuple[int, int]] = None
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.max_depth = max_depth
        
        self.channel_choices = channel_choices or [16, 32, 64, 128, 256, 512, 1024]
        self.kernel_choices = kernel_choices or [(1, 1), (3, 3), (5, 5), (7, 7)]
        
        self.layer_types = [
            LayerType.CONV2D,
            LayerType.DEPTHWISE_CONV,
            LayerType.LINEAR,
            LayerType.BATCH_NORM,
            LayerType.RELU,
            LayerType.GELU,
        ]
        
        self.activations = [
            ActivationType.RELU,
            ActivationType.GELU,
            ActivationType.SWISH,
        ]
    
    def sample_random(self) -> Architecture:
        """Sample a random architecture from the search space with enhanced layer compatibility."""
        max_attempts = 10
        
        for attempt in range(max_attempts):
            try:
                depth = random.randint(3, min(8, self.max_depth))  # Reasonable depth range
                layers = []
                
                current_channels = self.input_shape[2]  # Start with input channels (e.g., 3 for RGB)
                current_spatial_size = max(self.input_shape[:2])  # Track spatial dimensions
                
                for i in range(depth):
                    # Choose layer type with better logic
                    if i == 0:
                        # First layer must be conv to handle image input
                        layer_type = LayerType.CONV2D
                    elif i == depth - 1:
                        # Final layer is typically linear for classification
                        layer_type = LayerType.LINEAR
                    else:
                        # Middle layers can be varied
                        layer_type = random.choice([
                            LayerType.CONV2D, LayerType.DEPTHWISE_CONV, 
                            LayerType.BATCH_NORM, LayerType.RELU
                        ])
                    
                    try:
                        if layer_type == LayerType.CONV2D:
                            # Progressive channel expansion
                            if i == 0:
                                output_channels = random.choice([16, 32, 64])  # Start small
                            else:
                                output_channels = min(
                                    random.choice(self.channel_choices),
                                    current_channels * 4  # Reasonable expansion
                                )
                            
                            kernel_size = random.choice([(1, 1), (3, 3), (5, 5)])
                            stride = (1, 1) if i == 0 else random.choice([(1, 1), (2, 2)])
                            activation = random.choice(self.activations)
                            
                            layer = Layer(
                                layer_type=layer_type,
                                input_channels=current_channels,
                                output_channels=output_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding="same",
                                activation=activation
                            )
                            current_channels = output_channels
                            if stride == (2, 2):
                                current_spatial_size //= 2
                            
                        elif layer_type == LayerType.DEPTHWISE_CONV:
                            # Depthwise keeps same channels
                            kernel_size = random.choice([(3, 3), (5, 5)])
                            stride = random.choice([(1, 1), (2, 2)])
                            
                            layer = Layer(
                                layer_type=layer_type,
                                input_channels=current_channels,
                                output_channels=current_channels,  # Depthwise preserves channels
                                kernel_size=kernel_size,
                                stride=stride,
                                padding="same",
                                activation=random.choice(self.activations)
                            )
                            if stride == (2, 2):
                                current_spatial_size //= 2
                                
                        elif layer_type == LayerType.LINEAR:
                            # For final layer, output num_classes
                            if i == depth - 1:
                                output_channels = self.num_classes
                            else:
                                # Intermediate linear layer
                                output_channels = random.choice([256, 512, 1024])
                                
                            layer = Layer(
                                layer_type=layer_type,
                                input_channels=current_channels,
                                output_channels=output_channels,
                                activation=None if i == depth - 1 else random.choice(self.activations)
                            )
                            current_channels = output_channels
                            
                        else:  # Activation/norm layers
                            layer = Layer(
                                layer_type=layer_type,
                                input_channels=current_channels,
                                output_channels=current_channels,  # Preserve channels
                                activation=random.choice(self.activations) if layer_type in [LayerType.RELU, LayerType.GELU] else None
                            )
                        
                        if self._validate_layer(layer, current_spatial_size):
                            layers.append(layer)
                        else:
                            # If validation fails, try a simpler layer
                            simple_layer = Layer(
                                layer_type=LayerType.RELU,
                                input_channels=current_channels,
                                output_channels=current_channels
                            )
                            if self._validate_layer(simple_layer, current_spatial_size):
                                layers.append(simple_layer)
                            
                    except Exception as e:
                        continue  # Skip this layer and try again
                
                # Ensure we have a valid architecture
                if len(layers) >= 3:
                    # Ensure architecture ends with classification layer
                    if layers[-1].layer_type != LayerType.LINEAR or layers[-1].output_channels != self.num_classes:
                        layers.append(Layer(
                            layer_type=LayerType.LINEAR,
                            input_channels=current_channels,
                            output_channels=self.num_classes,
                            activation=None
                        ))
                    
                    arch = Architecture(
                        layers=layers,
                        input_shape=self.input_shape,
                        num_classes=self.num_classes,
                        name=f"random_arch_{random.randint(1000, 9999)}"
                    )
                    
                    if self._validate_architecture(arch):
                        return arch
                        
            except Exception as e:
                continue  # Try again
        
        # Fallback: create minimal valid architecture
        return self._create_minimal_architecture()
    
    def _validate_layer(self, layer: Layer, spatial_size: int = 224) -> bool:
        """Validate individual layer with enhanced checks."""
        try:
            # Channel validation
            if layer.input_channels <= 0 or layer.output_channels <= 0:
                return False
            
            if layer.input_channels > 2048 or layer.output_channels > 2048:
                return False  # Reasonable bounds
            
            # Layer-specific validation
            if layer.layer_type == LayerType.CONV2D:
                if layer.kernel_size is None:
                    return False
                # Kernel size should not exceed spatial dimensions
                if max(layer.kernel_size) > spatial_size:
                    return False
                    
            elif layer.layer_type == LayerType.DEPTHWISE_CONV:
                # Depthwise conv should preserve input channels
                if layer.input_channels != layer.output_channels:
                    return False
                    
            elif layer.layer_type == LayerType.LINEAR:
                # Linear layers should have reasonable size
                if layer.input_channels * layer.output_channels > 50_000_000:  # 50M parameters max
                    return False
            
            return True
            
        except:
            return False
    
    def _validate_architecture(self, arch: Architecture) -> bool:
        """Validate complete architecture with channel flow validation."""
        try:
            if not arch.layers or len(arch.layers) < 3:
                return False
            
            # Validate parameter and operation counts
            if arch.total_params <= 0 or arch.total_params > 1e9:  # Reasonable bounds
                return False
            
            if arch.total_ops <= 0 or arch.total_ops > 1e15:  # Reasonable bounds
                return False
            
            # Validate channel flow compatibility
            expected_channels = self.input_shape[2]  # Start with input channels
            
            for i, layer in enumerate(arch.layers):
                # Check input channel compatibility
                if layer.input_channels != expected_channels:
                    # Allow some flexibility for first layer
                    if i == 0 and layer.layer_type == LayerType.CONV2D:
                        continue  # First conv can adapt
                    else:
                        return False
                
                # Update expected channels for next layer
                if layer.layer_type in [LayerType.CONV2D, LayerType.LINEAR, LayerType.DEPTHWISE_CONV]:
                    expected_channels = layer.output_channels
                # Other layers (norm, activation) preserve channel count
            
            # Architecture should end with classification output
            final_layer = arch.layers[-1]
            if final_layer.layer_type == LayerType.LINEAR and final_layer.output_channels != self.num_classes:
                return False
            
            return True
            
        except Exception as e:
            return False
    
    def _create_minimal_architecture(self) -> Architecture:
        """Create minimal valid architecture as fallback."""
        layers = [
            Layer(
                layer_type=LayerType.CONV2D,
                input_channels=self.input_shape[2],
                output_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding="same",
                activation=ActivationType.RELU
            ),
            Layer(
                layer_type=LayerType.CONV2D,
                input_channels=64,
                output_channels=128,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding="same",
                activation=ActivationType.RELU
            ),
            Layer(
                layer_type=LayerType.LINEAR,
                input_channels=128,
                output_channels=self.num_classes,
                activation=None
            )
        ]
        
        return Architecture(
            layers=layers,
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            name="minimal_fallback_arch"
        )
    
    def _repair_channel_flow(self, layers: List[Layer]) -> List[Layer]:
        """Repair channel flow inconsistencies in layer sequence."""
        if not layers:
            return layers
            
        try:
            repaired_layers = []
            expected_channels = self.input_shape[2]  # Start with input channels
            
            for i, layer in enumerate(layers):
                # Create a corrected layer if needed
                if layer.input_channels != expected_channels:
                    # Fix input channels to match expected flow
                    corrected_layer = Layer(
                        layer_type=layer.layer_type,
                        input_channels=expected_channels,
                        output_channels=layer.output_channels,
                        kernel_size=layer.kernel_size,
                        stride=layer.stride,
                        padding=layer.padding,
                        activation=layer.activation
                    )
                    repaired_layers.append(corrected_layer)
                else:
                    repaired_layers.append(layer)
                
                # Update expected channels for next layer
                if layer.layer_type in [LayerType.CONV2D, LayerType.LINEAR, LayerType.DEPTHWISE_CONV]:
                    expected_channels = layer.output_channels
                # Other layers preserve channel count
            
            return repaired_layers
            
        except Exception as e:
            # If repair fails, return original layers
            return layers
    
    def _create_safe_mutation_fallback(self, architecture: Architecture) -> Architecture:
        """Create a safe mutation fallback when all mutation attempts fail."""
        try:
            # Simple safe mutation: change activation function of a random layer
            new_layers = architecture.layers.copy()
            
            # Find a layer that can have its activation changed
            mutable_indices = [
                i for i, layer in enumerate(new_layers)
                if layer.layer_type in [LayerType.CONV2D, LayerType.LINEAR] and i < len(new_layers) - 1
            ]
            
            if mutable_indices:
                idx = random.choice(mutable_indices)
                new_activation = random.choice(self.activations)
                
                new_layers[idx] = Layer(
                    layer_type=new_layers[idx].layer_type,
                    input_channels=new_layers[idx].input_channels,
                    output_channels=new_layers[idx].output_channels,
                    kernel_size=new_layers[idx].kernel_size,
                    stride=new_layers[idx].stride,
                    padding=new_layers[idx].padding,
                    activation=new_activation
                )
            
            return Architecture(
                layers=new_layers,
                input_shape=architecture.input_shape,
                num_classes=architecture.num_classes,
                name=f"safe_mutated_{architecture.name}"
            )
            
        except:
            # Ultimate fallback: return copy with different name
            return Architecture(
                layers=architecture.layers.copy(),
                input_shape=architecture.input_shape,
                num_classes=architecture.num_classes,
                name=f"fallback_mutated_{architecture.name}"
            )
    
    def mutate(self, architecture: Architecture) -> Architecture:
        """Mutate an existing architecture with enhanced channel flow validation."""
        max_attempts = 10  # Increased attempts for better success rate
        
        for attempt in range(max_attempts):
            try:
                new_layers = architecture.layers.copy()
                
                # Choose mutation strategy based on architecture characteristics
                if len(new_layers) < 5:
                    # Small architectures: prefer adding layers
                    mutation_ops = [self._add_layer, self._mutate_channels, self._mutate_activation]
                elif len(new_layers) > 15:
                    # Large architectures: prefer removing layers or changing existing
                    mutation_ops = [self._remove_layer, self._mutate_channels, self._mutate_kernel_size]
                else:
                    # Medium architectures: balanced mutations
                    mutation_ops = [
                        self._mutate_channels,
                        self._mutate_kernel_size,
                        self._mutate_activation,
                        self._add_layer,
                        self._remove_layer,
                    ]
                
                # Single mutation per attempt for better control
                mutation_op = random.choice(mutation_ops)
                
                try:
                    new_layers = mutation_op(new_layers)
                    
                    # Repair channel flow after mutation
                    new_layers = self._repair_channel_flow(new_layers)
                    
                except Exception as e:
                    continue  # Skip failed mutations
                
                if len(new_layers) < 3:  # Ensure minimum viable architecture
                    continue
                
                # Ensure final layer outputs correct number of classes
                if new_layers and new_layers[-1].layer_type != LayerType.LINEAR:
                    new_layers.append(Layer(
                        layer_type=LayerType.LINEAR,
                        input_channels=new_layers[-1].output_channels,
                        output_channels=self.num_classes,
                        activation=None
                    ))
                elif new_layers and new_layers[-1].output_channels != self.num_classes:
                    new_layers[-1] = Layer(
                        layer_type=LayerType.LINEAR,
                        input_channels=new_layers[-1].input_channels,
                        output_channels=self.num_classes,
                        activation=None
                    )
                
                mutated_arch = Architecture(
                    layers=new_layers,
                    input_shape=architecture.input_shape,
                    num_classes=architecture.num_classes,
                    name=f"mutated_{architecture.name}_{attempt}"
                )
                
                if self._validate_architecture(mutated_arch):
                    return mutated_arch
                    
            except Exception as e:
                continue  # Try again
        
        # Enhanced fallback: create a slightly modified version
        return self._create_safe_mutation_fallback(architecture)
    
    def crossover(self, parent1: Architecture, parent2: Architecture) -> Architecture:
        """Crossover two parent architectures."""
        max_attempts = 5
        
        for attempt in range(max_attempts):
            try:
                if not parent1.layers or not parent2.layers:
                    break
                
                min_len = min(len(parent1.layers), len(parent2.layers))
                if min_len < 2:
                    break
                
                crossover_point = random.randint(1, min_len - 1)
                
                # Try different crossover strategies
                if attempt % 2 == 0:
                    # Standard crossover
                    new_layers = (parent1.layers[:crossover_point] + 
                                 parent2.layers[crossover_point:])
                else:
                    # Interleaved crossover
                    new_layers = []
                    max_len = max(len(parent1.layers), len(parent2.layers))
                    for i in range(max_len):
                        if i % 2 == 0 and i < len(parent1.layers):
                            new_layers.append(parent1.layers[i])
                        elif i < len(parent2.layers):
                            new_layers.append(parent2.layers[i])
                
                if len(new_layers) < 3:
                    continue
                
                try:
                    self._fix_channel_compatibility(new_layers)
                except:
                    continue
                
                child_arch = Architecture(
                    layers=new_layers,
                    input_shape=parent1.input_shape,
                    num_classes=parent1.num_classes,
                    name=f"crossover_{parent1.name}_{parent2.name}_{attempt}"
                )
                
                if self._validate_architecture(child_arch):
                    return child_arch
                    
            except Exception as e:
                continue
        
        # Fallback: return mutated version of better parent
        better_parent = parent1 if len(parent1.layers) >= len(parent2.layers) else parent2
        return self.mutate(better_parent)
    
    def _mutate_channels(self, layers: List[Layer]) -> List[Layer]:
        """Mutate channel counts in layers."""
        if not layers:
            return layers
            
        layer_idx = random.randint(0, len(layers) - 1)
        layer = layers[layer_idx]
        
        if layer.layer_type in [LayerType.CONV2D, LayerType.LINEAR]:
            new_channels = random.choice(self.channel_choices)
            layer.output_channels = new_channels
            
            if layer_idx < len(layers) - 1:
                layers[layer_idx + 1].input_channels = new_channels
        
        return layers
    
    def _mutate_kernel_size(self, layers: List[Layer]) -> List[Layer]:
        """Mutate kernel sizes in conv layers."""
        conv_layers = [i for i, layer in enumerate(layers) 
                      if layer.layer_type == LayerType.CONV2D]
        
        if conv_layers:
            layer_idx = random.choice(conv_layers)
            layers[layer_idx].kernel_size = random.choice(self.kernel_choices)
        
        return layers
    
    def _mutate_activation(self, layers: List[Layer]) -> List[Layer]:
        """Mutate activation functions."""
        activatable_layers = [i for i, layer in enumerate(layers)
                             if layer.activation is not None]
        
        if activatable_layers:
            layer_idx = random.choice(activatable_layers)
            layers[layer_idx].activation = random.choice(self.activations)
        
        return layers
    
    def _add_layer(self, layers: List[Layer]) -> List[Layer]:
        """Add a new layer to the architecture."""
        if len(layers) >= self.max_depth:
            return layers
        
        insert_idx = random.randint(0, len(layers))
        
        if insert_idx == 0:
            input_channels = self.input_shape[2]
        else:
            input_channels = layers[insert_idx - 1].output_channels
        
        output_channels = random.choice(self.channel_choices)
        layer_type = random.choice([LayerType.CONV2D, LayerType.BATCH_NORM, LayerType.RELU])
        
        if layer_type == LayerType.CONV2D:
            new_layer = Layer(
                layer_type=layer_type,
                input_channels=input_channels,
                output_channels=output_channels,
                kernel_size=random.choice(self.kernel_choices),
                stride=(1, 1),
                padding="same",
                activation=random.choice(self.activations)
            )
        else:
            new_layer = Layer(
                layer_type=layer_type,
                input_channels=input_channels,
                output_channels=input_channels if layer_type != LayerType.CONV2D else output_channels
            )
        
        layers.insert(insert_idx, new_layer)
        
        if insert_idx < len(layers) - 1:
            layers[insert_idx + 1].input_channels = new_layer.output_channels
        
        return layers
    
    def _remove_layer(self, layers: List[Layer]) -> List[Layer]:
        """Remove a layer from the architecture."""
        if len(layers) <= 3:
            return layers
        
        remove_idx = random.randint(0, len(layers) - 2)
        removed_layer = layers.pop(remove_idx)
        
        if remove_idx < len(layers):
            layers[remove_idx].input_channels = removed_layer.input_channels
        
        return layers
    
    def _fix_channel_compatibility(self, layers: List[Layer]) -> None:
        """Fix channel compatibility after crossover."""
        for i in range(1, len(layers)):
            layers[i].input_channels = layers[i - 1].output_channels