#!/usr/bin/env python3
"""Basic example of using TPUv6-ZeroNAS for architecture search."""

import logging
from pathlib import Path

from tpuv6_zeronas import ZeroNASSearcher, TPUv6Predictor, ArchitectureSpace
from tpuv6_zeronas.core import SearchConfig
from tpuv6_zeronas.optimizations import TPUv6Optimizer


def main():
    """Run basic architecture search example."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("TPUv6-ZeroNAS Basic Search Example")
    
    arch_space = ArchitectureSpace(
        input_shape=(224, 224, 3),
        num_classes=1000,
        max_depth=15
    )
    
    predictor = TPUv6Predictor()
    
    search_config = SearchConfig(
        max_iterations=100,
        population_size=20,
        target_tops_w=75.0,
        max_latency_ms=8.0,
        min_accuracy=0.95
    )
    
    searcher = ZeroNASSearcher(arch_space, predictor, search_config)
    
    logger.info("Starting architecture search...")
    best_arch, best_metrics = searcher.search()
    
    logger.info(f"Search completed!")
    logger.info(f"Best architecture: {best_arch.name}")
    logger.info(f"Layers: {len(best_arch.layers)}")
    logger.info(f"Parameters: {best_arch.total_params:,}")
    logger.info(f"Operations: {best_arch.total_ops:,}")
    logger.info(f"Memory: {best_arch.memory_mb:.1f} MB")
    
    logger.info(f"Performance metrics:")
    logger.info(f"  Latency: {best_metrics.latency_ms:.2f} ms")
    logger.info(f"  Energy: {best_metrics.energy_mj:.2f} mJ")
    logger.info(f"  Accuracy: {best_metrics.accuracy:.3f}")
    logger.info(f"  TOPS/W: {best_metrics.tops_per_watt:.1f}")
    
    logger.info("Applying TPUv6 optimizations...")
    optimizer = TPUv6Optimizer()
    optimized_arch = optimizer.optimize_architecture(best_arch)
    optimized_metrics = predictor.predict(optimized_arch)
    
    logger.info(f"Optimized performance:")
    logger.info(f"  Latency: {optimized_metrics.latency_ms:.2f} ms")
    logger.info(f"  Energy: {optimized_metrics.energy_mj:.2f} mJ") 
    logger.info(f"  Accuracy: {optimized_metrics.accuracy:.3f}")
    logger.info(f"  TOPS/W: {optimized_metrics.tops_per_watt:.1f}")
    
    optimization_report = optimizer.get_optimization_report(best_arch, optimized_arch)
    logger.info(f"Optimization gains:")
    logger.info(f"  Parameters change: {optimization_report['params_change_pct']:.1f}%")
    logger.info(f"  Memory change: {optimization_report['memory_change_pct']:.1f}%")
    logger.info(f"  Estimated TOPS efficiency: {optimization_report['estimated_tops_efficiency']:.1f}")


if __name__ == '__main__':
    main()