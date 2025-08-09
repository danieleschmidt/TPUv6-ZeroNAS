#!/usr/bin/env python3
"""
TPUv6-ZeroNAS Scaling and Performance Demonstration

This script demonstrates the Generation 3 scaling capabilities including:
- Auto-scaling based on load
- Load balancing across workers
- Performance optimization and caching
- Adaptive parameter tuning
"""

import sys
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tpuv6_zeronas import (
    ZeroNASSearcher, TPUv6Predictor, ArchitectureSpace, SearchConfig
)
from tpuv6_zeronas.advanced_monitoring import get_advanced_monitor
from tpuv6_zeronas.scaling import (
    get_auto_scaler, get_load_balancer, get_adaptive_optimizer,
    ScalingPolicy, enable_auto_scaling
)
from tpuv6_zeronas.error_handling import get_error_handler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_auto_scaling():
    """Demonstrate auto-scaling capabilities."""
    logger.info("🚀 GENERATION 3: Auto-Scaling Demonstration")
    logger.info("=" * 60)
    
    # Enable auto-scaling with aggressive policy for demo
    scaler = enable_auto_scaling(
        policy=ScalingPolicy.AGGRESSIVE,
        min_workers=1,
        max_workers=4
    )
    
    # Start advanced monitoring
    monitor = get_advanced_monitor()
    monitor.start_monitoring()
    
    try:
        # Run multiple search sessions to trigger scaling
        configs = [
            SearchConfig(max_iterations=5, population_size=8),
            SearchConfig(max_iterations=10, population_size=12),
            SearchConfig(max_iterations=8, population_size=16),
        ]
        
        space = ArchitectureSpace(max_depth=10)
        predictor = TPUv6Predictor()
        
        for i, config in enumerate(configs):
            logger.info(f"\n📊 Running search session {i+1}/3")
            logger.info(f"   Configuration: {config.max_iterations} iterations, {config.population_size} population")
            
            # Record start metrics
            start_metrics = scaler.collect_metrics()
            logger.info(f"   Load before: {start_metrics.current_load:.2f}")
            logger.info(f"   Workers: {scaler.current_workers}")
            
            # Run search
            start_time = time.time()
            searcher = ZeroNASSearcher(space, predictor, config)
            best_arch, best_metrics = searcher.search()
            duration = time.time() - start_time
            
            # Record end metrics
            end_metrics = scaler.collect_metrics()
            logger.info(f"   Search completed in {duration:.1f}s")
            logger.info(f"   Best architecture: {best_arch.name}")
            logger.info(f"   Performance: {best_metrics.latency_ms:.2f}ms latency, {best_metrics.accuracy:.1%} accuracy")
            logger.info(f"   Load after: {end_metrics.current_load:.2f}")
            
            # Wait for auto-scaler to potentially adjust
            time.sleep(2)
            
            searcher.cleanup()
        
        # Get scaling recommendations
        recommendations = scaler.get_scaling_recommendations()
        logger.info(f"\n🎯 Scaling Analysis:")
        logger.info(f"   Average load: {recommendations['avg_load']:.2f}")
        logger.info(f"   Average response time: {recommendations['avg_response_time_ms']:.1f}ms")
        logger.info(f"   Current workers: {recommendations['current_workers']}")
        logger.info(f"   Recent actions: {', '.join(recommendations['recent_scaling_actions']) or 'none'}")
        
        if recommendations['recommendations']:
            logger.info(f"   Recommendations:")
            for rec in recommendations['recommendations']:
                logger.info(f"     • {rec}")
        
    finally:
        scaler.stop_monitoring()
        monitor.stop_monitoring()


def demonstrate_load_balancing():
    """Demonstrate load balancing capabilities."""
    logger.info("\n⚖️  Load Balancing Demonstration")
    logger.info("=" * 40)
    
    load_balancer = get_load_balancer()
    
    # Simulate distributing tasks across workers
    tasks = [f"task_{i}" for i in range(20)]
    workers = [f"worker_{i}" for i in range(3)]
    
    # Test different balancing strategies
    strategies = ["round_robin", "least_connections", "weighted_round_robin"]
    
    for strategy in strategies:
        load_balancer.balancing_strategy = strategy
        distribution = load_balancer.distribute_tasks(tasks, workers)
        
        logger.info(f"\n   Strategy: {strategy}")
        for worker_id, worker_tasks in distribution.items():
            logger.info(f"     Worker {worker_id}: {len(worker_tasks)} tasks")


def demonstrate_adaptive_optimization():
    """Demonstrate adaptive optimization capabilities.""" 
    logger.info("\n🧠 Adaptive Optimization Demonstration")
    logger.info("=" * 45)
    
    optimizer = get_adaptive_optimizer()
    
    # Simulate different performance scenarios
    scenarios = [
        {
            'name': 'High Performance',
            'performance': {
                'avg_latency_ms': 300,
                'success_rate': 0.98,
                'throughput': 8.5
            }
        },
        {
            'name': 'Low Performance',
            'performance': {
                'avg_latency_ms': 2500,
                'success_rate': 0.75,
                'throughput': 0.8
            }
        },
        {
            'name': 'Balanced Performance',
            'performance': {
                'avg_latency_ms': 1200,
                'success_rate': 0.90,
                'throughput': 3.2
            }
        }
    ]
    
    for scenario in scenarios:
        logger.info(f"\n   Scenario: {scenario['name']}")
        perf = scenario['performance']
        logger.info(f"     Input: {perf['avg_latency_ms']:.0f}ms latency, "
                   f"{perf['success_rate']:.1%} success, "
                   f"{perf['throughput']:.1f} ops/s")
        
        optimizations = optimizer.optimize_search_parameters(perf)
        logger.info(f"     Optimizations:")
        for key, value in optimizations.items():
            if isinstance(value, float):
                logger.info(f"       {key}: {value:.2f}")
            else:
                logger.info(f"       {key}: {value}")


def demonstrate_comprehensive_system():
    """Demonstrate the complete optimized system."""
    logger.info("\n🌟 Comprehensive System Demonstration")
    logger.info("=" * 50)
    
    # Enable all optimization features
    scaler = enable_auto_scaling(ScalingPolicy.BALANCED, 1, 3)
    monitor = get_advanced_monitor()
    monitor.start_monitoring()
    
    try:
        # Run a comprehensive search with all features enabled
        config = SearchConfig(
            max_iterations=12,
            population_size=15,
            enable_parallel=True,
            enable_caching=True,
            enable_adaptive=True
        )
        
        space = ArchitectureSpace(max_depth=8)
        predictor = TPUv6Predictor()
        
        logger.info("   🔧 Configuration:")
        logger.info(f"     Max iterations: {config.max_iterations}")
        logger.info(f"     Population size: {config.population_size}")
        logger.info(f"     Parallel processing: {config.enable_parallel}")
        logger.info(f"     Caching enabled: {config.enable_caching}")
        logger.info(f"     Adaptive optimization: {config.enable_adaptive}")
        
        # Record initial system state
        initial_health = monitor.get_health_summary()
        logger.info(f"\n   📊 Initial System Status: {initial_health['overall_status']}")
        
        # Run the optimized search
        start_time = time.time()
        searcher = ZeroNASSearcher(space, predictor, config)
        
        logger.info("   🔍 Running optimized architecture search...")
        best_arch, best_metrics = searcher.search()
        
        duration = time.time() - start_time
        
        # Collect final results
        final_health = monitor.get_health_summary()
        error_stats = get_error_handler().get_error_statistics()
        
        logger.info(f"\n   ✅ Search Results:")
        logger.info(f"     Duration: {duration:.1f} seconds")
        logger.info(f"     Best architecture: {best_arch.name}")
        logger.info(f"     Architecture complexity:")
        logger.info(f"       • Layers: {len(best_arch.layers)}")
        logger.info(f"       • Parameters: {best_arch.total_params:,}")
        logger.info(f"       • Operations: {best_arch.total_ops:,}")
        logger.info(f"       • Memory: {best_arch.memory_mb:.1f} MB")
        
        logger.info(f"\n   📈 Performance Metrics:")
        logger.info(f"     • Latency: {best_metrics.latency_ms:.2f} ms")
        logger.info(f"     • Energy: {best_metrics.energy_mj:.2f} mJ")
        logger.info(f"     • Accuracy: {best_metrics.accuracy:.1%}")
        logger.info(f"     • Efficiency: {best_metrics.tops_per_watt:.1f} TOPS/W")
        logger.info(f"     • Overall Score: {best_metrics.efficiency_score:.3f}")
        
        logger.info(f"\n   🔧 System Performance:")
        logger.info(f"     • Total evaluations: {len(searcher.search_history)}")
        logger.info(f"     • Final system status: {final_health['overall_status']}")
        logger.info(f"     • Total errors: {error_stats.get('total_errors', 0)}")
        
        if hasattr(searcher.predictor, 'get_performance_stats'):
            cache_stats = searcher.predictor.get_performance_stats()
            logger.info(f"     • Cache hit rate: {cache_stats.get('cache_hit_rate', 0):.1%}")
            logger.info(f"     • Predictions made: {cache_stats.get('predictions_made', 0)}")
        
        searcher.cleanup()
        
    finally:
        scaler.stop_monitoring()
        monitor.stop_monitoring()


def main():
    """Main demonstration function."""
    logger.info("🚀 TPUv6-ZeroNAS Generation 3: Scaling & Performance Optimization")
    logger.info("Advanced features demonstration for production-ready deployment")
    logger.info("=" * 80)
    
    try:
        # Demonstrate individual scaling features
        demonstrate_auto_scaling()
        demonstrate_load_balancing()
        demonstrate_adaptive_optimization()
        
        # Demonstrate comprehensive optimized system
        demonstrate_comprehensive_system()
        
        logger.info("\n🎉 GENERATION 3 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        logger.info("🏆 TPUv6-ZeroNAS is now production-ready with:")
        logger.info("   ✅ Auto-scaling based on system load")
        logger.info("   ✅ Intelligent load balancing")
        logger.info("   ✅ Performance optimization and caching")
        logger.info("   ✅ Adaptive parameter tuning")
        logger.info("   ✅ Comprehensive monitoring and health checks")
        logger.info("   ✅ Robust error handling and recovery")
        logger.info("   ✅ Security measures and input validation")
        logger.info("\n🚀 Ready for large-scale neural architecture search!")
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)