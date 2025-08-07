#!/usr/bin/env python3
"""Validate TPUv6-ZeroNAS installation and run basic functionality tests."""

import logging
import sys
from pathlib import Path

try:
    from tpuv6_zeronas import ZeroNASSearcher, TPUv6Predictor, ArchitectureSpace, PerformanceMetrics
    from tpuv6_zeronas.core import SearchConfig
    from tpuv6_zeronas.optimizations import TPUv6Optimizer
except ImportError as e:
    print(f"ERROR: Failed to import TPUv6-ZeroNAS modules: {e}")
    print("Please ensure the package is properly installed.")
    sys.exit(1)


def test_architecture_space():
    """Test architecture space functionality."""
    print("Testing ArchitectureSpace...")
    
    arch_space = ArchitectureSpace(
        input_shape=(224, 224, 3),
        num_classes=1000,
        max_depth=10
    )
    
    arch = arch_space.sample_random()
    print(f"  ✓ Generated random architecture: {arch.name}")
    print(f"  ✓ Architecture stats: {len(arch.layers)} layers, {arch.total_params:,} params")
    
    mutated = arch_space.mutate(arch)
    print(f"  ✓ Mutation successful: {mutated.name}")
    
    arch2 = arch_space.sample_random()
    crossover = arch_space.crossover(arch, arch2)
    print(f"  ✓ Crossover successful: {crossover.name}")
    
    return True


def test_predictor():
    """Test TPUv6 predictor functionality."""
    print("Testing TPUv6Predictor...")
    
    arch_space = ArchitectureSpace(max_depth=8)
    arch = arch_space.sample_random()
    
    predictor = TPUv6Predictor()
    metrics = predictor.predict(arch)
    
    print(f"  ✓ Prediction successful")
    print(f"  ✓ Latency: {metrics.latency_ms:.2f} ms")
    print(f"  ✓ Energy: {metrics.energy_mj:.2f} mJ")
    print(f"  ✓ Accuracy: {metrics.accuracy:.3f}")
    print(f"  ✓ TOPS/W: {metrics.tops_per_watt:.1f}")
    
    assert metrics.latency_ms > 0
    assert metrics.energy_mj > 0
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.tops_per_watt > 0
    
    return True


def test_optimizer():
    """Test TPUv6 optimizer functionality."""
    print("Testing TPUv6Optimizer...")
    
    arch_space = ArchitectureSpace(max_depth=8)
    arch = arch_space.sample_random()
    
    optimizer = TPUv6Optimizer()
    optimized = optimizer.optimize_architecture(arch)
    
    print(f"  ✓ Optimization successful: {optimized.name}")
    
    report = optimizer.get_optimization_report(arch, optimized)
    print(f"  ✓ Optimization report generated")
    print(f"  ✓ Parameter change: {report['params_change_pct']:.1f}%")
    
    return True


def test_search():
    """Test basic search functionality."""
    print("Testing ZeroNAS search (short run)...")
    
    arch_space = ArchitectureSpace(
        input_shape=(224, 224, 3),
        num_classes=1000,
        max_depth=8
    )
    
    predictor = TPUv6Predictor()
    
    config = SearchConfig(
        max_iterations=5,
        population_size=8,
        target_tops_w=75.0
    )
    
    searcher = ZeroNASSearcher(arch_space, predictor, config)
    best_arch, best_metrics = searcher.search()
    
    print(f"  ✓ Search completed successfully")
    print(f"  ✓ Best architecture: {best_arch.name}")
    print(f"  ✓ Best metrics: {best_metrics}")
    print(f"  ✓ Search history: {len(searcher.search_history)} evaluations")
    
    assert best_arch is not None
    assert best_metrics is not None
    
    return True


def test_metrics():
    """Test performance metrics functionality."""
    print("Testing PerformanceMetrics...")
    
    metrics = PerformanceMetrics(
        latency_ms=5.0,
        energy_mj=50.0,
        accuracy=0.96,
        tops_per_watt=70.0,
        memory_mb=100.0,
        flops=1000000
    )
    
    print(f"  ✓ Metrics created: {metrics}")
    print(f"  ✓ Efficiency score: {metrics.efficiency_score:.3f}")
    
    metrics_dict = metrics.to_dict()
    print(f"  ✓ Serialization successful: {len(metrics_dict)} fields")
    
    return True


def test_cli_availability():
    """Test CLI module availability."""
    print("Testing CLI availability...")
    
    try:
        from tpuv6_zeronas.cli import main
        print("  ✓ CLI module imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ CLI import failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("TPUv6-ZeroNAS Installation Validation")
    print("=" * 40)
    
    tests = [
        test_architecture_space,
        test_predictor,
        test_optimizer,
        test_metrics,
        test_cli_availability,
        test_search,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            print(f"\n{test_func.__name__.replace('test_', '').replace('_', ' ').title()}:")
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 40)
    print(f"Validation Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! TPUv6-ZeroNAS is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the installation.")
        return 1


if __name__ == '__main__':
    sys.exit(main())