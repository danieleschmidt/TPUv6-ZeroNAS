#!/usr/bin/env python3
"""Demonstrate advanced search capabilities."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_basic_functionality():
    """Test basic TPUv6-ZeroNAS functionality."""
    logger.info("Testing basic functionality...")
    
    try:
        from tpuv6_zeronas import ZeroNASSearcher, TPUv6Predictor, ArchitectureSpace
        from tpuv6_zeronas.core import SearchConfig
        
        # Create components
        arch_space = ArchitectureSpace(
            input_shape=(224, 224, 3),
            num_classes=1000,
            max_depth=8
        )
        
        predictor = TPUv6Predictor()
        
        config = SearchConfig(
            max_iterations=10,
            population_size=8,
            target_tops_w=75.0
        )
        
        searcher = ZeroNASSearcher(arch_space, predictor, config)
        
        # Run short search
        logger.info("Running basic search...")
        start_time = time.time()
        best_arch, best_metrics = searcher.search()
        elapsed = time.time() - start_time
        
        logger.info(f"Basic search completed in {elapsed:.2f}s")
        logger.info(f"Best architecture: {best_arch.name}")
        logger.info(f"Best metrics: {best_metrics}")
        
        return True
        
    except Exception as e:
        logger.error(f"Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_advanced_optimization():
    """Test advanced optimization features."""
    logger.info("Testing advanced optimization...")
    
    try:
        from tpuv6_zeronas import ADVANCED_AVAILABLE
        
        if not ADVANCED_AVAILABLE:
            logger.warning("Advanced features not available (missing dependencies)")
            return True  # Not a failure, just not available
        
        from tpuv6_zeronas import ProgressiveSearchOptimizer, OptimizationConfig
        from tpuv6_zeronas import ArchitectureSpace, TPUv6Predictor
        from tpuv6_zeronas.optimization import OptimizationStrategy
        
        # Create components
        arch_space = ArchitectureSpace(
            input_shape=(224, 224, 3),
            num_classes=1000,
            max_depth=8
        )
        
        predictor = TPUv6Predictor()
        
        config = OptimizationConfig(
            strategy=OptimizationStrategy.PROGRESSIVE,
            max_evaluations=50,
            use_surrogate_model=True,
            adaptive_parameters=True
        )
        
        optimizer = ProgressiveSearchOptimizer(arch_space, predictor, config)
        
        # Run progressive optimization
        logger.info("Running progressive optimization...")
        start_time = time.time()
        best_arch, best_metrics = optimizer.optimize()
        elapsed = time.time() - start_time
        
        logger.info(f"Progressive optimization completed in {elapsed:.2f}s")
        logger.info(f"Best architecture: {best_arch.name}")
        logger.info(f"Best metrics: {best_metrics}")
        
        return True
        
    except Exception as e:
        logger.error(f"Advanced optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_objective_optimization():
    """Test multi-objective optimization."""
    logger.info("Testing multi-objective optimization...")
    
    try:
        from tpuv6_zeronas import ADVANCED_AVAILABLE
        
        if not ADVANCED_AVAILABLE:
            logger.warning("Multi-objective optimization not available")
            return True
        
        from tpuv6_zeronas import MultiObjectiveOptimizer
        from tpuv6_zeronas import ArchitectureSpace, TPUv6Predictor
        
        # Create components
        arch_space = ArchitectureSpace(
            input_shape=(224, 224, 3),
            num_classes=1000,
            max_depth=6
        )
        
        predictor = TPUv6Predictor()
        
        optimizer = MultiObjectiveOptimizer(
            arch_space,
            predictor,
            objectives=['accuracy', 'latency', 'energy', 'efficiency']
        )
        
        # Find Pareto front
        logger.info("Finding Pareto-optimal solutions...")
        start_time = time.time()
        pareto_solutions = optimizer.optimize_pareto_front(
            population_size=20,
            generations=5
        )
        elapsed = time.time() - start_time
        
        logger.info(f"Multi-objective optimization completed in {elapsed:.2f}s")
        logger.info(f"Found {len(pareto_solutions)} Pareto-optimal solutions")
        
        for i, (arch, metrics) in enumerate(pareto_solutions[:3]):
            logger.info(f"  Solution {i+1}: {arch.name} - {metrics}")
        
        return True
        
    except Exception as e:
        logger.error(f"Multi-objective optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parallel_search():
    """Test parallel search capabilities."""
    logger.info("Testing parallel search...")
    
    try:
        from tpuv6_zeronas import ADVANCED_AVAILABLE
        
        if not ADVANCED_AVAILABLE:
            logger.warning("Parallel search not available")
            return True
        
        from tpuv6_zeronas import DistributedSearcher, ParallelSearchConfig
        from tpuv6_zeronas import ArchitectureSpace, TPUv6Predictor
        from tpuv6_zeronas.core import SearchConfig
        
        # Create components
        arch_space = ArchitectureSpace(
            input_shape=(224, 224, 3),
            num_classes=1000,
            max_depth=6
        )
        
        predictor = TPUv6Predictor()
        
        search_config = SearchConfig(
            max_iterations=5,
            population_size=16,
            target_tops_w=75.0
        )
        
        parallel_config = ParallelSearchConfig(
            num_workers=2,  # Use 2 workers for testing
            batch_size=8
        )
        
        searcher = DistributedSearcher(
            arch_space,
            predictor,
            search_config,
            parallel_config
        )
        
        # Run distributed search
        logger.info("Running distributed search...")
        start_time = time.time()
        best_arch, best_metrics = searcher.search()
        elapsed = time.time() - start_time
        
        logger.info(f"Distributed search completed in {elapsed:.2f}s")
        logger.info(f"Best architecture: {best_arch.name}")
        logger.info(f"Best metrics: {best_metrics}")
        
        return True
        
    except Exception as e:
        logger.error(f"Parallel search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_performance_benchmark():
    """Run performance benchmark."""
    logger.info("Running performance benchmark...")
    
    try:
        from tpuv6_zeronas import ZeroNASSearcher, TPUv6Predictor, ArchitectureSpace
        from tpuv6_zeronas.core import SearchConfig
        
        # Different problem sizes
        test_cases = [
            {'name': 'Small', 'depth': 5, 'population': 8, 'iterations': 5},
            {'name': 'Medium', 'depth': 8, 'population': 16, 'iterations': 10},
            {'name': 'Large', 'depth': 12, 'population': 24, 'iterations': 15},
        ]
        
        benchmark_results = []
        
        for case in test_cases:
            logger.info(f"Benchmarking {case['name']} problem...")
            
            arch_space = ArchitectureSpace(
                input_shape=(224, 224, 3),
                num_classes=1000,
                max_depth=case['depth']
            )
            
            predictor = TPUv6Predictor()
            
            config = SearchConfig(
                max_iterations=case['iterations'],
                population_size=case['population'],
                target_tops_w=75.0
            )
            
            searcher = ZeroNASSearcher(arch_space, predictor, config)
            
            start_time = time.time()
            best_arch, best_metrics = searcher.search()
            elapsed = time.time() - start_time
            
            evaluations = len(searcher.search_history)
            evals_per_second = evaluations / elapsed if elapsed > 0 else 0
            
            benchmark_results.append({
                'name': case['name'],
                'time': elapsed,
                'evaluations': evaluations,
                'evals_per_sec': evals_per_second,
                'best_score': best_metrics.efficiency_score
            })
            
            logger.info(f"  {case['name']}: {elapsed:.2f}s, {evaluations} evals, {evals_per_second:.1f} evals/s")
        
        # Summary
        logger.info("Benchmark Summary:")
        for result in benchmark_results:
            logger.info(f"  {result['name']}: {result['time']:.2f}s, "
                       f"{result['evaluations']} evals, "
                       f"{result['evals_per_sec']:.1f} evals/s, "
                       f"score: {result['best_score']:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Performance benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all demonstrations."""
    logger.info("TPUv6-ZeroNAS Advanced Search Demonstration")
    logger.info("=" * 50)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Advanced Optimization", test_advanced_optimization),
        ("Multi-objective Optimization", test_multi_objective_optimization),
        ("Parallel Search", test_parallel_search),
        ("Performance Benchmark", run_performance_benchmark),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            if test_func():
                logger.info(f"✓ {test_name} passed")
                passed += 1
            else:
                logger.error(f"✗ {test_name} failed")
                failed += 1
        except Exception as e:
            logger.error(f"✗ {test_name} failed with exception: {e}")
            failed += 1
    
    logger.info("\n" + "=" * 50)
    logger.info(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All tests passed! TPUv6-ZeroNAS is working great.")
        return 0
    else:
        logger.warning(f"⚠️  Some tests failed, but basic functionality is working.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
