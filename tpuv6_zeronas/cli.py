"""Command-line interface for TPUv6-ZeroNAS."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from .core import ZeroNASSearcher, SearchConfig
from .architecture import ArchitectureSpace
from .predictor import TPUv6Predictor
from .optimizations import TPUv6Optimizer, TPUv6Config
from .metrics import MetricsAggregator


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('tpuv6_zeronas.log')
        ]
    )


def create_search_config(args) -> SearchConfig:
    """Create search configuration from arguments."""
    return SearchConfig(
        max_iterations=args.max_iterations,
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        target_tops_w=args.target_tops_w,
        max_latency_ms=args.max_latency,
        min_accuracy=args.min_accuracy
    )


def run_search(args) -> None:
    """Run neural architecture search with enhanced validation."""
    logger = logging.getLogger(__name__)
    
    # Enhanced input validation
    try:
        from .security import get_resource_guard
        from .validation import validate_input
        
        guard = get_resource_guard()
        
        # Validate numeric inputs
        if args.max_iterations <= 0 or args.max_iterations > 10000:
            raise ValueError(f"Invalid max_iterations: {args.max_iterations}. Must be 1-10000")
        
        if args.population_size <= 0 or args.population_size > 500:
            raise ValueError(f"Invalid population_size: {args.population_size}. Must be 1-500")
        
        if args.target_tops_w <= 0 or args.target_tops_w > 1000:
            raise ValueError(f"Invalid target_tops_w: {args.target_tops_w}. Must be 1-1000")
        
        if args.max_latency <= 0 or args.max_latency > 10000:
            raise ValueError(f"Invalid max_latency: {args.max_latency}. Must be 1-10000ms")
        
        if args.min_accuracy < 0 or args.min_accuracy > 1:
            raise ValueError(f"Invalid min_accuracy: {args.min_accuracy}. Must be 0-1")
        
        # Validate file paths
        if hasattr(args, 'predictor_model') and args.predictor_model:
            args.predictor_model = guard.sanitize_file_path(args.predictor_model)
        
        if args.output:
            args.output = guard.sanitize_file_path(args.output)
            
    except Exception as e:
        logger.error(f"Input validation failed: {e}")
        raise
    
    logger.info("Starting TPUv6-ZeroNAS search...")
    logger.info(f"Target: {args.target_tops_w} TOPS/W")
    logger.info(f"Max iterations: {args.max_iterations}")
    
    arch_space = ArchitectureSpace(
        input_shape=(args.input_height, args.input_width, args.input_channels),
        num_classes=args.num_classes,
        max_depth=min(args.max_depth, 50)  # Security cap
    )
    
    predictor = TPUv6Predictor()
    if args.predictor_model and Path(args.predictor_model).exists():
        predictor.load_models(Path(args.predictor_model))
        logger.info(f"Loaded predictor model from {args.predictor_model}")
    else:
        logger.warning("No predictor model found, using fallback predictions")
    
    search_config = create_search_config(args)
    searcher = ZeroNASSearcher(arch_space, predictor, search_config)
    
    best_arch, best_metrics = searcher.search()
    
    if args.optimize_for_tpuv6:
        logger.info("Applying TPUv6-specific optimizations...")
        optimizer = TPUv6Optimizer()
        optimized_arch = optimizer.optimize_architecture(best_arch)
        
        optimization_report = optimizer.get_optimization_report(best_arch, optimized_arch)
        logger.info(f"Optimization report: {optimization_report}")
        
        best_arch = optimized_arch
        best_metrics = predictor.predict(optimized_arch)
    
    output_path = Path(args.output) if args.output else Path("best_architecture.json")
    
    result = {
        'architecture': {
            'name': best_arch.name,
            'layers': [
                {
                    'type': layer.layer_type.value,
                    'input_channels': layer.input_channels,
                    'output_channels': layer.output_channels,
                    'kernel_size': layer.kernel_size,
                    'stride': layer.stride,
                    'activation': layer.activation.value if layer.activation else None
                }
                for layer in best_arch.layers
            ],
            'input_shape': best_arch.input_shape,
            'num_classes': best_arch.num_classes,
            'total_ops': best_arch.total_ops,
            'total_params': best_arch.total_params,
            'memory_mb': best_arch.memory_mb
        },
        'metrics': best_metrics.to_dict(),
        'search_config': {
            'max_iterations': search_config.max_iterations,
            'population_size': search_config.population_size,
            'target_tops_w': search_config.target_tops_w
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Best architecture: {best_arch.name}")
    logger.info(f"Performance: {best_metrics}")


def train_predictor(args) -> None:
    """Train the TPUv6 predictor model."""
    logger = logging.getLogger(__name__)
    
    logger.info("Training TPUv6 predictor model...")
    
    if not Path(args.training_data).exists():
        logger.error(f"Training data not found: {args.training_data}")
        sys.exit(1)
    
    with open(args.training_data, 'r') as f:
        training_data = json.load(f)
    
    predictor = TPUv6Predictor()
    metrics = predictor.train(training_data)
    
    logger.info(f"Training completed. Metrics: {metrics}")
    
    output_path = Path(args.output) if args.output else Path("tpuv6_predictor.pkl")
    predictor.save_models(output_path)
    
    logger.info(f"Predictor model saved to {output_path}")


def benchmark_architecture(args) -> None:
    """Benchmark a specific architecture."""
    logger = logging.getLogger(__name__)
    
    if not Path(args.architecture).exists():
        logger.error(f"Architecture file not found: {args.architecture}")
        sys.exit(1)
    
    with open(args.architecture, 'r') as f:
        arch_data = json.load(f)
    
    predictor = TPUv6Predictor()
    if args.predictor_model and Path(args.predictor_model).exists():
        predictor.load_models(Path(args.predictor_model))
    
    metrics = predictor.predict(arch_data['architecture'])
    
    logger.info(f"Architecture benchmark results:")
    logger.info(f"Latency: {metrics.latency_ms:.2f} ms")
    logger.info(f"Energy: {metrics.energy_mj:.2f} mJ")
    logger.info(f"Accuracy: {metrics.accuracy:.3f}")
    logger.info(f"TOPS/W: {metrics.tops_per_watt:.1f}")
    logger.info(f"Memory: {metrics.memory_mb:.1f} MB")


def run_health_check(args) -> None:
    """Run comprehensive system health check."""
    logger = logging.getLogger(__name__)
    
    logger.info("🏥 TPUv6-ZeroNAS Health Check")
    logger.info("=" * 50)
    
    try:
        # Initialize components for health checking
        predictor = TPUv6Predictor()
        
        # Basic health status
        health_status = predictor.get_health_status()
        
        logger.info(f"✅ Predictor Status: {health_status['status']}")
        logger.info(f"📊 Prediction Count: {health_status['prediction_count']}")
        logger.info(f"💾 Cache Size: {health_status['cache_size']}")
        logger.info(f"🎯 Cache Hit Rate: {health_status['cache_hit_rate']:.2%}")
        
        if args.detailed:
            logger.info("\n📋 Detailed Health Information:")
            logger.info(f"⏱️  Average Prediction Time: {health_status['avg_prediction_time']:.4f}s")
            logger.info(f"❌ Error Rate: {health_status['error_rate']:.2%}")
            logger.info(f"🔬 Novel Patterns Found: {health_status['novel_patterns_found']}")
            logger.info(f"⚠️  Scaling Violations: {health_status['scaling_violations']}")
            
            # Test basic functionality
            logger.info("\n🧪 Running Basic Functionality Tests:")
            
            from .architecture import ArchitectureSpace
            arch_space = ArchitectureSpace()
            test_arch = arch_space.sample_random()
            
            test_metrics = predictor.predict(test_arch)
            logger.info(f"✅ Test Prediction: {test_metrics.latency_ms:.2f}ms, {test_metrics.accuracy:.3f} acc")
            
        if args.repair:
            logger.info("\n🔧 Attempting System Repairs:")
            repairs_successful = predictor.validate_and_repair()
            if repairs_successful:
                logger.info("✅ System repairs completed successfully")
            else:
                logger.warning("⚠️  Some repairs failed, check logs for details")
        
        # System resource check
        try:
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024
            logger.info(f"🖥️  Memory Usage: {memory_usage:.1f} MB")
            logger.info(f"⚡ CPU Usage: {process.cpu_percent():.1f}%")
        except ImportError:
            logger.info("🖥️  Resource monitoring requires 'psutil' package")
        
        logger.info("\n🎉 Health check completed!")
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TPUv6-ZeroNAS: Neural Architecture Search for TPUv6 Optimization"
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    search_parser = subparsers.add_parser('search', help='Run architecture search')
    search_parser.add_argument('--max-iterations', type=int, default=1000, 
                              help='Maximum search iterations')
    search_parser.add_argument('--population-size', type=int, default=50,
                              help='Population size for genetic algorithm')
    search_parser.add_argument('--mutation-rate', type=float, default=0.1,
                              help='Mutation rate')
    search_parser.add_argument('--crossover-rate', type=float, default=0.7,
                              help='Crossover rate')
    search_parser.add_argument('--target-tops-w', type=float, default=75.0,
                              help='Target TOPS/W efficiency')
    search_parser.add_argument('--max-latency', type=float, default=10.0,
                              help='Maximum latency constraint (ms)')
    search_parser.add_argument('--min-accuracy', type=float, default=0.95,
                              help='Minimum accuracy constraint')
    search_parser.add_argument('--input-height', type=int, default=224,
                              help='Input image height')
    search_parser.add_argument('--input-width', type=int, default=224,
                              help='Input image width')
    search_parser.add_argument('--input-channels', type=int, default=3,
                              help='Input channels')
    search_parser.add_argument('--num-classes', type=int, default=1000,
                              help='Number of output classes')
    search_parser.add_argument('--max-depth', type=int, default=20,
                              help='Maximum network depth')
    search_parser.add_argument('--predictor-model', type=str,
                              help='Path to trained predictor model')
    search_parser.add_argument('--optimize-for-tpuv6', action='store_true',
                              help='Apply TPUv6-specific optimizations')
    search_parser.add_argument('--output', '-o', type=str,
                              help='Output file path')
    
    train_parser = subparsers.add_parser('train', help='Train predictor model')
    train_parser.add_argument('--training-data', type=str, required=True,
                             help='Path to training data JSON file')
    train_parser.add_argument('--output', '-o', type=str,
                             help='Output model file path')
    
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark architecture')
    benchmark_parser.add_argument('--architecture', type=str, required=True,
                                 help='Path to architecture JSON file')
    benchmark_parser.add_argument('--predictor-model', type=str,
                                 help='Path to trained predictor model')
    
    # Health check command for robustness monitoring
    health_parser = subparsers.add_parser('health', help='Check system health and status')
    health_parser.add_argument('--detailed', action='store_true', help='Show detailed health information')
    health_parser.add_argument('--repair', action='store_true', help='Attempt automatic repairs')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    if args.command == 'search':
        run_search(args)
    elif args.command == 'train':
        train_predictor(args)
    elif args.command == 'benchmark':
        benchmark_architecture(args)
    elif args.command == 'health':
        run_health_check(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()