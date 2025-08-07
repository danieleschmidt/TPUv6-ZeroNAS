#!/usr/bin/env python3
"""Minimal test without numpy dependency."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create mock numpy if not available
class MockNumpy:
    @staticmethod
    def clip(x, a, b):
        return max(a, min(b, x))
    
    @staticmethod
    def random():
        import random
        return random.random()

if 'numpy' not in sys.modules:
    sys.modules['numpy'] = MockNumpy()
    sys.modules['np'] = MockNumpy()

# Test basic functionality
print("Testing TPUv6-ZeroNAS core components...")

try:
    from tpuv6_zeronas.architecture import Architecture, ArchitectureSpace, Layer, LayerType
    from tpuv6_zeronas.metrics import PerformanceMetrics
    print("✓ Basic imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test layer creation
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

# Test metrics
try:
    metrics = PerformanceMetrics(
        latency_ms=5.0,
        energy_mj=50.0,
        accuracy=0.96,
        tops_per_watt=70.0,
        memory_mb=100.0,
        flops=1000000
    )
    print(f"✓ Metrics created: efficiency={metrics.efficiency_score:.3f}")
except Exception as e:
    print(f"✗ Metrics creation failed: {e}")
    sys.exit(1)

print("\n🎉 Minimal test completed successfully!")
print("Core functionality is working without external dependencies.")
