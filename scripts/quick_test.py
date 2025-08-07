#!/usr/bin/env python3
"""Quick test without external dependencies."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test basic imports
try:
    from tpuv6_zeronas.architecture import Architecture, ArchitectureSpace, Layer, LayerType
    from tpuv6_zeronas.metrics import PerformanceMetrics
    print("✓ Basic imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test architecture creation
try:
    layer = Layer(
        layer_type=LayerType.CONV2D,
        input_channels=3,
        output_channels=64,
        kernel_size=(3, 3)
    )
    print(f"✓ Layer created: {layer.layer_type.value}, ops={layer.ops_count}")
except Exception as e:
    print(f"✗ Layer creation failed: {e}")
    sys.exit(1)

# Test architecture space
try:
    arch_space = ArchitectureSpace(
        input_shape=(224, 224, 3),
        num_classes=1000,
        max_depth=5
    )
    
    arch = arch_space.sample_random()
    print(f"✓ Architecture sampled: {arch.name}, {len(arch.layers)} layers")
    print(f"✓ Architecture stats: {arch.total_params:,} params, {arch.total_ops:,} ops")
except Exception as e:
    print(f"✗ Architecture sampling failed: {e}")
    sys.exit(1)

# Test metrics
try:
    metrics = PerformanceMetrics(
        latency_ms=5.0,
        energy_mj=50.0,
        accuracy=0.96,
        tops_per_watt=70.0,
        memory_mb=100.0,
        flops=arch.total_ops
    )
    print(f"✓ Metrics created: efficiency={metrics.efficiency_score:.3f}")
except Exception as e:
    print(f"✗ Metrics creation failed: {e}")
    sys.exit(1)

print("\n🎉 Quick test completed successfully!")
print("TPUv6-ZeroNAS core functionality is working.")
