"""Validation and error handling for TPUv6-ZeroNAS."""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import numpy as np

from .architecture import Architecture, Layer, LayerType
from .metrics import PerformanceMetrics


@dataclass
class ValidationError:
    """Validation error details."""
    error_type: str
    severity: str  # 'error', 'warning', 'info'
    message: str
    context: Dict[str, Any]
    component: str


class ArchitectureValidator:
    """Validates neural architecture specifications."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
    
    def validate_architecture(self, architecture: Architecture) -> Dict[str, Any]:
        """Validate complete architecture specification."""
        self.errors.clear()
        self.warnings.clear()
        
        self._validate_basic_structure(architecture)
        self._validate_layer_compatibility(architecture)
        self._validate_resource_constraints(architecture)
        self._validate_performance_feasibility(architecture)
        
        return {
            'is_valid': len(self.errors) == 0,
            'errors': self.errors,
            'warnings': self.warnings,
            'summary': {
                'total_errors': len(self.errors),
                'total_warnings': len(self.warnings),
                'validation_passed': len(self.errors) == 0
            }
        }
    
    def _validate_basic_structure(self, architecture: Architecture) -> None:
        """Validate basic architecture structure."""
        if not architecture.layers:
            self.errors.append(ValidationError(
                error_type='empty_architecture',
                severity='error',
                message='Architecture must have at least one layer',
                context={'layer_count': 0},
                component='architecture'
            ))
            return
        
        if len(architecture.layers) < 2:
            self.warnings.append(ValidationError(
                error_type='minimal_architecture',
                severity='warning',
                message='Architecture has very few layers, may underperform',
                context={'layer_count': len(architecture.layers)},
                component='architecture'
            ))
        
        if len(architecture.layers) > 100:
            self.warnings.append(ValidationError(
                error_type='excessive_depth',
                severity='warning',
                message='Architecture is very deep, may cause training difficulties',
                context={'layer_count': len(architecture.layers)},
                component='architecture'
            ))
        
        if not all(isinstance(layer, Layer) for layer in architecture.layers):
            self.errors.append(ValidationError(
                error_type='invalid_layer_type',
                severity='error',
                message='All layers must be Layer instances',
                context={'layer_types': [type(layer) for layer in architecture.layers]},
                component='architecture'
            ))
    
    def _validate_layer_compatibility(self, architecture: Architecture) -> None:
        """Validate layer input/output compatibility."""
        for i, layer in enumerate(architecture.layers):
            if i == 0:
                # First layer input compatibility
                if hasattr(layer, 'input_channels'):
                    expected_channels = architecture.input_shape[2]
                    if layer.input_channels != expected_channels:
                        self.errors.append(ValidationError(
                            error_type='input_mismatch',
                            severity='error',
                            message=f'First layer input channels mismatch',
                            context={
                                'layer_index': i,
                                'expected': expected_channels,
                                'actual': layer.input_channels
                            },
                            component='layer_compatibility'
                        ))
            
            else:
                # Inter-layer compatibility
                prev_layer = architecture.layers[i-1]
                if (hasattr(layer, 'input_channels') and 
                    hasattr(prev_layer, 'output_channels')):
                    if layer.input_channels != prev_layer.output_channels:
                        self.errors.append(ValidationError(
                            error_type='channel_mismatch',
                            severity='error',
                            message=f'Channel mismatch between layers {i-1} and {i}',
                            context={
                                'prev_layer_output': prev_layer.output_channels,
                                'curr_layer_input': layer.input_channels,
                                'layer_index': i
                            },
                            component='layer_compatibility'
                        ))
        
        # Final layer compatibility
        final_layer = architecture.layers[-1]
        if (hasattr(final_layer, 'output_channels') and 
            final_layer.output_channels != architecture.num_classes):
            if final_layer.layer_type == LayerType.LINEAR:
                self.errors.append(ValidationError(
                    error_type='output_mismatch',
                    severity='error',
                    message='Final layer output does not match num_classes',
                    context={
                        'final_layer_output': final_layer.output_channels,
                        'num_classes': architecture.num_classes
                    },
                    component='layer_compatibility'
                ))
    
    def _validate_resource_constraints(self, architecture: Architecture) -> None:
        """Validate resource usage within reasonable bounds."""
        # Memory constraints
        memory_mb = architecture.memory_mb
        if memory_mb > 8000:  # 8GB
            self.warnings.append(ValidationError(
                error_type='high_memory_usage',
                severity='warning',
                message='Architecture requires excessive memory',
                context={'memory_mb': memory_mb, 'limit_mb': 8000},
                component='resource_constraints'
            ))
        
        # Parameter constraints
        total_params = architecture.total_params
        if total_params > 100_000_000:  # 100M parameters
            self.warnings.append(ValidationError(
                error_type='high_parameter_count',
                severity='warning',
                message='Architecture has very large parameter count',
                context={'total_params': total_params, 'limit': 100_000_000},
                component='resource_constraints'
            ))
        
        # Computation constraints
        total_ops = architecture.total_ops
        if total_ops > 10_000_000_000:  # 10B operations
            self.warnings.append(ValidationError(
                error_type='high_computation',
                severity='warning',
                message='Architecture requires excessive computation',
                context={'total_ops': total_ops, 'limit': 10_000_000_000},
                component='resource_constraints'
            ))
    
    def _validate_performance_feasibility(self, architecture: Architecture) -> None:
        """Validate performance feasibility."""
        # Check for degenerate cases
        if architecture.total_params == 0:
            self.errors.append(ValidationError(
                error_type='no_parameters',
                severity='error',
                message='Architecture has no trainable parameters',
                context={'total_params': 0},
                component='performance_feasibility'
            ))
        
        if architecture.total_ops == 0:
            self.errors.append(ValidationError(
                error_type='no_operations',
                severity='error',
                message='Architecture performs no operations',
                context={'total_ops': 0},
                component='performance_feasibility'
            ))
        
        # Check for reasonable ratios
        if architecture.total_params > 0 and architecture.total_ops > 0:
            ops_per_param = architecture.total_ops / architecture.total_params
            if ops_per_param < 1:
                self.warnings.append(ValidationError(
                    error_type='low_ops_per_param',
                    severity='warning',
                    message='Very low operations per parameter ratio',
                    context={'ops_per_param': ops_per_param},
                    component='performance_feasibility'
                ))
            elif ops_per_param > 1000:
                self.warnings.append(ValidationError(
                    error_type='high_ops_per_param',
                    severity='warning',
                    message='Very high operations per parameter ratio',
                    context={'ops_per_param': ops_per_param},
                    component='performance_feasibility'
                ))


class MetricsValidator:
    """Validates performance metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_metrics(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Validate performance metrics values."""
        errors = []
        warnings = []
        
        # Latency validation
        if metrics.latency_ms < 0:
            errors.append(ValidationError(
                error_type='negative_latency',
                severity='error',
                message='Latency cannot be negative',
                context={'latency_ms': metrics.latency_ms},
                component='metrics'
            ))
        elif metrics.latency_ms > 1000:  # 1 second
            warnings.append(ValidationError(
                error_type='high_latency',
                severity='warning',
                message='Very high latency prediction',
                context={'latency_ms': metrics.latency_ms},
                component='metrics'
            ))
        
        # Energy validation
        if metrics.energy_mj < 0:
            errors.append(ValidationError(
                error_type='negative_energy',
                severity='error',
                message='Energy cannot be negative',
                context={'energy_mj': metrics.energy_mj},
                component='metrics'
            ))
        
        # Accuracy validation
        if not (0.0 <= metrics.accuracy <= 1.0):
            errors.append(ValidationError(
                error_type='invalid_accuracy',
                severity='error',
                message='Accuracy must be between 0 and 1',
                context={'accuracy': metrics.accuracy},
                component='metrics'
            ))
        
        # TOPS/W validation
        if metrics.tops_per_watt < 0:
            errors.append(ValidationError(
                error_type='negative_efficiency',
                severity='error',
                message='TOPS/W efficiency cannot be negative',
                context={'tops_per_watt': metrics.tops_per_watt},
                component='metrics'
            ))
        elif metrics.tops_per_watt > 1000:  # Theoretical maximum
            warnings.append(ValidationError(
                error_type='unrealistic_efficiency',
                severity='warning',
                message='TOPS/W efficiency seems unrealistically high',
                context={'tops_per_watt': metrics.tops_per_watt},
                component='metrics'
            ))
        
        # Cross-metric validation
        if metrics.energy_mj > 0 and metrics.latency_ms > 0:
            power_w = metrics.energy_mj / metrics.latency_ms
            if power_w > 100:  # 100W seems high for edge device
                warnings.append(ValidationError(
                    error_type='high_power',
                    severity='warning',
                    message='Estimated power consumption is high',
                    context={'power_w': power_w},
                    component='metrics'
                ))
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'summary': {
                'total_errors': len(errors),
                'total_warnings': len(warnings),
                'validation_passed': len(errors) == 0
            }
        }


class SearchConfigValidator:
    """Validates search configuration parameters."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_search_config(self, config: Any) -> Dict[str, Any]:
        """Validate search configuration."""
        errors = []
        warnings = []
        
        # Iteration constraints
        if config.max_iterations <= 0:
            errors.append(ValidationError(
                error_type='invalid_iterations',
                severity='error',
                message='Max iterations must be positive',
                context={'max_iterations': config.max_iterations},
                component='search_config'
            ))
        elif config.max_iterations > 10000:
            warnings.append(ValidationError(
                error_type='high_iterations',
                severity='warning',
                message='Very high iteration count may take long time',
                context={'max_iterations': config.max_iterations},
                component='search_config'
            ))
        
        # Population size constraints
        if config.population_size <= 0:
            errors.append(ValidationError(
                error_type='invalid_population',
                severity='error',
                message='Population size must be positive',
                context={'population_size': config.population_size},
                component='search_config'
            ))
        elif config.population_size < 5:
            warnings.append(ValidationError(
                error_type='small_population',
                severity='warning',
                message='Small population may limit search diversity',
                context={'population_size': config.population_size},
                component='search_config'
            ))
        
        # Rate constraints
        if not (0.0 <= config.mutation_rate <= 1.0):
            errors.append(ValidationError(
                error_type='invalid_mutation_rate',
                severity='error',
                message='Mutation rate must be between 0 and 1',
                context={'mutation_rate': config.mutation_rate},
                component='search_config'
            ))
        
        if not (0.0 <= config.crossover_rate <= 1.0):
            errors.append(ValidationError(
                error_type='invalid_crossover_rate',
                severity='error',
                message='Crossover rate must be between 0 and 1',
                context={'crossover_rate': config.crossover_rate},
                component='search_config'
            ))
        
        # Target constraints
        if config.target_tops_w <= 0:
            errors.append(ValidationError(
                error_type='invalid_target_tops',
                severity='error',
                message='Target TOPS/W must be positive',
                context={'target_tops_w': config.target_tops_w},
                component='search_config'
            ))
        
        if config.max_latency_ms <= 0:
            errors.append(ValidationError(
                error_type='invalid_latency_constraint',
                severity='error',
                message='Max latency must be positive',
                context={'max_latency_ms': config.max_latency_ms},
                component='search_config'
            ))
        
        if not (0.0 <= config.min_accuracy <= 1.0):
            errors.append(ValidationError(
                error_type='invalid_accuracy_constraint',
                severity='error',
                message='Min accuracy must be between 0 and 1',
                context={'min_accuracy': config.min_accuracy},
                component='search_config'
            ))
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'summary': {
                'total_errors': len(errors),
                'total_warnings': len(warnings),
                'validation_passed': len(errors) == 0
            }
        }


def validate_input(input_data: Any, input_type: str) -> Dict[str, Any]:
    """Universal input validation function."""
    if input_type == 'architecture':
        validator = ArchitectureValidator()
        return validator.validate_architecture(input_data)
    elif input_type == 'metrics':
        validator = MetricsValidator()
        return validator.validate_metrics(input_data)
    elif input_type == 'search_config':
        validator = SearchConfigValidator()
        return validator.validate_search_config(input_data)
    else:
        return {
            'is_valid': False,
            'errors': [ValidationError(
                error_type='unknown_input_type',
                severity='error',
                message=f'Unknown input type: {input_type}',
                context={'input_type': input_type},
                component='validation'
            )],
            'warnings': [],
            'summary': {
                'total_errors': 1,
                'total_warnings': 0,
                'validation_passed': False
            }
        }