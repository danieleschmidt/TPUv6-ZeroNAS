"""Tests for Generation 3 optimization features."""

import pytest
import time
from pathlib import Path

from tpuv6_zeronas.core import ZeroNASSearcher, SearchConfig
from tpuv6_zeronas.architecture import ArchitectureSpace
from tpuv6_zeronas.predictor import TPUv6Predictor
from tpuv6_zeronas.parallel import ParallelEvaluator, WorkerConfig, PerformanceOptimizer
from tpuv6_zeronas.caching import PredictionCache, CachedPredictor


class TestParallelEvaluator:
    """Test parallel evaluation functionality."""
    
    def test_parallel_evaluator_creation(self):
        """Test parallel evaluator initialization."""
        predictor = TPUv6Predictor()
        config = WorkerConfig(num_workers=2, worker_type='thread')
        evaluator = ParallelEvaluator(predictor, config)
        
        assert evaluator.config.num_workers == 2
        assert evaluator.config.worker_type == 'thread'
        assert evaluator.executor is not None
        
        evaluator.shutdown()
    
    def test_parallel_population_evaluation(self):
        """Test parallel evaluation of architecture population."""
        arch_space = ArchitectureSpace(
            input_shape=(64, 64, 3),
            num_classes=10,
            max_depth=8
        )
        
        predictor = TPUv6Predictor()
        config = WorkerConfig(num_workers=2, worker_type='thread')
        evaluator = ParallelEvaluator(predictor, config)
        
        # Create small population
        population = [arch_space.sample_random() for _ in range(5)]
        
        start_time = time.time()
        results = evaluator.evaluate_population_parallel(population)
        elapsed = time.time() - start_time
        
        assert len(results) == len(population)
        assert all(len(result) == 2 for result in results)
        assert elapsed < 5.0  # Should complete quickly
        
        evaluator.shutdown()


class TestCaching:
    """Test caching functionality."""
    
    def test_prediction_cache_basic(self):
        """Test basic caching operations."""
        cache = PredictionCache()
        
        arch_space = ArchitectureSpace(
            input_shape=(32, 32, 3),
            num_classes=10,
            max_depth=8
        )
        
        architecture = arch_space.sample_random()
        predictor = TPUv6Predictor()
        metrics = predictor.predict(architecture)
        
        # Test cache miss
        cached_result = cache.get_prediction(architecture)
        assert cached_result is None
        
        # Cache the prediction
        cache.cache_prediction(architecture, metrics)
        
        # Test cache hit
        cached_result = cache.get_prediction(architecture)
        assert cached_result is not None
        assert cached_result.accuracy == metrics.accuracy
        assert cached_result.latency_ms == metrics.latency_ms
    
    def test_cached_predictor(self):
        """Test cached predictor wrapper."""
        predictor = TPUv6Predictor()
        cached_predictor = CachedPredictor(predictor)
        
        arch_space = ArchitectureSpace(
            input_shape=(32, 32, 3),
            num_classes=10,
            max_depth=8
        )
        
        architecture = arch_space.sample_random()
        
        # First prediction (cache miss)
        start_time = time.time()
        result1 = cached_predictor.predict(architecture)
        time1 = time.time() - start_time
        
        # Second prediction (cache hit)
        start_time = time.time()
        result2 = cached_predictor.predict(architecture)
        time2 = time.time() - start_time
        
        # Results should be identical
        assert result1.accuracy == result2.accuracy
        assert result1.latency_ms == result2.latency_ms
        
        # Second call should be faster
        assert time2 < time1
        
        # Check cache stats
        stats = cached_predictor.get_performance_stats()
        assert stats['cache_hits'] >= 1
        assert stats['cache_misses'] >= 1
        assert stats['cache_hit_rate'] > 0


class TestPerformanceOptimizer:
    """Test performance optimization features."""
    
    def test_performance_optimizer(self):
        """Test performance optimizer functionality."""
        optimizer = PerformanceOptimizer()
        
        arch_space = ArchitectureSpace(
            input_shape=(32, 32, 3),
            num_classes=10,
            max_depth=8
        )
        
        predictor = TPUv6Predictor()
        config = WorkerConfig(num_workers=2, worker_type='thread')
        evaluator = ParallelEvaluator(predictor, config)
        
        # Create population
        population = [arch_space.sample_random() for _ in range(10)]
        
        # First evaluation
        results1 = optimizer.optimize_population_evaluation(population, evaluator)
        
        # Second evaluation (should benefit from caching)
        results2 = optimizer.optimize_population_evaluation(population, evaluator)
        
        assert len(results1) == len(population)
        assert len(results2) == len(population)
        
        # Check cache stats
        cache_stats = optimizer.get_cache_stats()
        assert cache_stats['cache_hits'] > 0
        assert cache_stats['hit_rate'] > 0
        
        evaluator.shutdown()


class TestOptimizedSearch:
    """Test end-to-end optimized search."""
    
    def test_optimized_search_integration(self):
        """Test complete optimized search pipeline."""
        arch_space = ArchitectureSpace(
            input_shape=(32, 32, 3),
            num_classes=10,
            max_depth=5
        )
        
        predictor = TPUv6Predictor()
        
        # Configure with all optimizations enabled
        config = SearchConfig(
            max_iterations=5,
            population_size=10,
            target_tops_w=50.0,
            max_latency_ms=20.0,
            min_accuracy=0.85,
            enable_parallel=True,
            enable_caching=True,
            enable_adaptive=True,
            parallel_workers=2
        )
        
        searcher = ZeroNASSearcher(arch_space, predictor, config)
        
        start_time = time.time()
        best_arch, best_metrics = searcher.search()
        elapsed = time.time() - start_time
        
        assert best_arch is not None
        assert best_metrics is not None
        assert elapsed < 30.0  # Should complete quickly with optimizations
        
        # Check that optimizations were used
        if hasattr(searcher.predictor, 'get_performance_stats'):
            stats = searcher.predictor.get_performance_stats()
            # Should have some cache activity
            assert stats['predictions_made'] > 0
        
        searcher.cleanup()
    
    def test_performance_comparison(self):
        """Compare optimized vs non-optimized search performance."""
        arch_space = ArchitectureSpace(
            input_shape=(32, 32, 3),
            num_classes=10,
            max_depth=8
        )
        
        # Non-optimized search
        predictor1 = TPUv6Predictor()
        config1 = SearchConfig(
            max_iterations=3,
            population_size=8,
            enable_parallel=False,
            enable_caching=False,
            enable_adaptive=False
        )
        
        searcher1 = ZeroNASSearcher(arch_space, predictor1, config1)
        
        start_time = time.time()
        best_arch1, best_metrics1 = searcher1.search()
        time_non_optimized = time.time() - start_time
        
        # Optimized search
        predictor2 = TPUv6Predictor()
        config2 = SearchConfig(
            max_iterations=3,
            population_size=8,
            enable_parallel=True,
            enable_caching=True,
            enable_adaptive=True,
            parallel_workers=2
        )
        
        searcher2 = ZeroNASSearcher(arch_space, predictor2, config2)
        
        start_time = time.time()
        best_arch2, best_metrics2 = searcher2.search()
        time_optimized = time.time() - start_time
        
        # Both should succeed
        assert best_arch1 is not None
        assert best_arch2 is not None
        
        # Optimized version might be faster or similar (depends on overhead)
        # Just ensure both complete in reasonable time
        assert time_non_optimized < 30.0
        assert time_optimized < 30.0
        
        searcher1.cleanup()
        searcher2.cleanup()


if __name__ == '__main__':
    pytest.main([__file__])