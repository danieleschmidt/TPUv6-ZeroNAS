#!/usr/bin/env python3
"""Generate synthetic training data for TPUv6 predictor."""

import json
import logging
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from tpuv6_zeronas.architecture import ArchitectureSpace, Architecture
from tpuv6_zeronas.metrics import PerformanceMetrics
from tpuv6_zeronas.predictor import EdgeTPUv5eCounters


def simulate_v5e_measurements(architecture: Architecture) -> PerformanceMetrics:
    """Simulate Edge TPU v5e measurements for architecture."""
    counter_collector = EdgeTPUv5eCounters()
    features = counter_collector.collect_counters(architecture)
    
    base_latency = (architecture.total_ops / 1e9) * np.random.uniform(0.8, 1.2)
    complexity_penalty = np.log(architecture.total_params / 1e6 + 1) * 0.1
    latency = base_latency * (1 + complexity_penalty)
    
    base_energy = latency * np.random.uniform(8.0, 12.0)
    efficiency_boost = features.get('tpu_utilization', 0.7) * 1.5
    energy = base_energy / efficiency_boost
    
    accuracy_base = 0.97
    param_penalty = max(0, np.log(architecture.total_params / 1e7))
    accuracy = accuracy_base - param_penalty * 0.02 + np.random.normal(0, 0.01)
    accuracy = np.clip(accuracy, 0.85, 0.99)
    
    tops_per_watt = (architecture.total_ops / 1e12) / max(energy / 1000, 1e-6)
    
    latency += np.random.normal(0, latency * 0.05)
    energy += np.random.normal(0, energy * 0.05)
    
    return PerformanceMetrics(
        latency_ms=max(0.1, latency),
        energy_mj=max(0.1, energy),
        accuracy=accuracy,
        tops_per_watt=tops_per_watt,
        memory_mb=architecture.memory_mb,
        flops=architecture.total_ops
    )


def generate_diverse_architectures(arch_space: ArchitectureSpace, count: int) -> List[Architecture]:
    """Generate diverse set of architectures for training."""
    architectures = []
    
    depth_ranges = [(3, 8), (8, 15), (15, 25)]
    channel_strategies = ['small', 'medium', 'large']
    
    for i in tqdm(range(count), desc="Generating architectures"):
        if i < count // 3:
            depth_range = depth_ranges[0]
            channels = [16, 32, 64, 128]
        elif i < 2 * count // 3:
            depth_range = depth_ranges[1] 
            channels = [64, 128, 256, 512]
        else:
            depth_range = depth_ranges[2]
            channels = [128, 256, 512, 1024]
        
        arch_space.max_depth = random.randint(*depth_range)
        arch_space.channel_choices = channels
        
        arch = arch_space.sample_random()
        architectures.append(arch)
    
    return architectures


def serialize_training_data(data: List[Tuple[Architecture, PerformanceMetrics]]) -> List[dict]:
    """Serialize training data to JSON format."""
    serialized = []
    
    for arch, metrics in data:
        arch_data = {
            'layers': [
                {
                    'type': layer.layer_type.value,
                    'input_channels': layer.input_channels,
                    'output_channels': layer.output_channels,
                    'kernel_size': layer.kernel_size,
                    'stride': layer.stride,
                    'activation': layer.activation.value if layer.activation else None,
                    'ops_count': layer.ops_count,
                    'params_count': layer.params_count
                }
                for layer in arch.layers
            ],
            'input_shape': arch.input_shape,
            'num_classes': arch.num_classes,
            'total_ops': arch.total_ops,
            'total_params': arch.total_params,
            'memory_mb': arch.memory_mb
        }
        
        metrics_data = {
            'latency_ms': metrics.latency_ms,
            'energy_mj': metrics.energy_mj,
            'accuracy': metrics.accuracy,
            'tops_per_watt': metrics.tops_per_watt,
            'memory_mb': metrics.memory_mb,
            'flops': metrics.flops
        }
        
        serialized.append({
            'architecture': arch_data,
            'metrics': metrics_data
        })
    
    return serialized


def main():
    """Generate training data for TPUv6 predictor."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Generating synthetic training data for TPUv6 predictor...")
    
    arch_space = ArchitectureSpace(
        input_shape=(224, 224, 3),
        num_classes=1000,
        max_depth=20
    )
    
    num_samples = 5000
    logger.info(f"Generating {num_samples} architecture samples...")
    
    architectures = generate_diverse_architectures(arch_space, num_samples)
    
    logger.info("Simulating Edge TPU v5e measurements...")
    training_data = []
    
    for arch in tqdm(architectures, desc="Simulating measurements"):
        try:
            metrics = simulate_v5e_measurements(arch)
            training_data.append((arch, metrics))
        except Exception as e:
            logger.warning(f"Failed to simulate architecture {arch.name}: {e}")
            continue
    
    logger.info(f"Generated {len(training_data)} training samples")
    
    serialized_data = serialize_training_data(training_data)
    
    output_path = Path("training_data_v5e_to_v6.json")
    with open(output_path, 'w') as f:
        json.dump(serialized_data, f, indent=2)
    
    logger.info(f"Training data saved to {output_path}")
    
    stats = {
        'total_samples': len(training_data),
        'avg_latency_ms': np.mean([m.latency_ms for _, m in training_data]),
        'avg_energy_mj': np.mean([m.energy_mj for _, m in training_data]),
        'avg_accuracy': np.mean([m.accuracy for _, m in training_data]),
        'avg_tops_per_watt': np.mean([m.tops_per_watt for _, m in training_data]),
        'depth_range': (min(len(a.layers) for a, _ in training_data), 
                       max(len(a.layers) for a, _ in training_data)),
        'param_range': (min(a.total_params for a, _ in training_data),
                       max(a.total_params for a, _ in training_data))
    }
    
    logger.info(f"Dataset statistics: {json.dumps(stats, indent=2)}")


if __name__ == '__main__':
    main()