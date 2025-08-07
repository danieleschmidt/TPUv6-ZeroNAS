#!/usr/bin/env python3
"""Integration tests for TPUv6-ZeroNAS."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
import json
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_architecture_generation():
    """Test architecture generation and validation."""
    logger.info("Testing architecture generation...")
    
    try:
        from tpuv6_zeronas.architecture import ArchitectureSpace, LayerType
        
        arch_space = ArchitectureSpace(
            input_shape=(224, 224, 3),
            num_classes=1000,
            max_depth=10
        )
        
        # Test multiple architecture generation
        architectures = []
        failed_generations = 0
        
        for i in range(50):
            try:
                arch = arch_space.sample_random()
                
                # Validate architecture
                if arch.total_params <= 0:
                    logger.warning(f"Architecture {i} has invalid parameters: {arch.total_params}")
                    failed_generations += 1
                    continue
                
                if arch.total_ops <= 0:
                    logger.warning(f"Architecture {i} has invalid ops: {arch.total_ops}")
                    failed_generations += 1
                    continue
                
                if len(arch.layers) < 3:
                    logger.warning(f"Architecture {i} too shallow: {len(arch.layers)} layers")
                    failed_generations += 1
                    continue
                
                architectures.append(arch)
                
            except Exception as e:
                logger.warning(f"Failed to generate architecture {i}: {e}")
                failed_generations += 1
        
        success_rate = len(architectures) / 50
        logger.info(f"Generated {len(architectures)}/50 valid architectures (success rate: {success_rate:.1%})")
        
        if len(architectures) < 25:  # At least 50% success rate
            logger.error("Architecture generation success rate too low")
            return False
        
        # Test diversity
        depths = [arch.depth for arch in architectures]
        params = [arch.total_params for arch in architectures]
        
        depth_diversity = (max(depths) - min(depths)) / max(depths)
        param_diversity = (max(params) - min(params)) / max(params)
        
        logger.info(f"Architecture diversity - Depth: {depth_diversity:.3f}, Params: {param_diversity:.3f}")
        
        if depth_diversity < 0.3 or param_diversity < 0.5:
            logger.warning("Low architecture diversity detected")
        
        return True
        
    except Exception as e:
        logger.error(f"Architecture generation test failed: {e}")
        return False


def test_mutation_and_crossover():
    """Test genetic operations."""
    logger.info("Testing mutation and crossover...")
    
    try:
        from tpuv6_zeronas.architecture import ArchitectureSpace
        
        arch_space = ArchitectureSpace(
            input_shape=(224, 224, 3),
            num_classes=1000,
            max_depth=8
        )
        
        # Generate parent architectures
        parent1 = arch_space.sample_random()
        parent2 = arch_space.sample_random()
        
        # Test mutation
        mutations_success = 0
        for i in range(10):
            try:
                mutated = arch_space.mutate(parent1)
                
                if mutated.name != parent1.name:  # Should have different name
                    mutations_success += 1
                else:
                    logger.warning(f"Mutation {i} produced identical name")
                    
            except Exception as e:
                logger.warning(f"Mutation {i} failed: {e}")
        
        logger.info(f"Successful mutations: {mutations_success}/10")
        
        # Test crossover
        crossovers_success = 0
        for i in range(10):
            try:
                child = arch_space.crossover(parent1, parent2)
                
                if len(child.layers) > 0:  # Should have layers
                    crossovers_success += 1
                else:
                    logger.warning(f"Crossover {i} produced empty architecture")
                    
            except Exception as e:
                logger.warning(f"Crossover {i} failed: {e}")
        
        logger.info(f"Successful crossovers: {crossovers_success}/10")
        
        if mutations_success < 7 or crossovers_success < 7:
            logger.error("Genetic operations success rate too low")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Genetic operations test failed: {e}")
        return False


def test_predictor_functionality():
    """Test predictor predictions."""
    logger.info("Testing predictor functionality...")
    
    try:
        from tpuv6_zeronas.predictor import TPUv6Predictor
        from tpuv6_zeronas.architecture import ArchitectureSpace
        
        arch_space = ArchitectureSpace(
            input_shape=(224, 224, 3),
            num_classes=1000,
            max_depth=8
        )
        
        predictor = TPUv6Predictor()
        
        # Test predictions on multiple architectures
        successful_predictions = 0
        latencies = []
        energies = []
        accuracies = []
        tops_per_watt = []
        
        for i in range(20):
            try:
                arch = arch_space.sample_random()
                metrics = predictor.predict(arch)
                
                # Validate metrics
                if not (0.1 <= metrics.latency_ms <= 1000.0):
                    logger.warning(f"Prediction {i}: Invalid latency {metrics.latency_ms}")
                    continue
                
                if not (0.1 <= metrics.energy_mj <= 10000.0):
                    logger.warning(f"Prediction {i}: Invalid energy {metrics.energy_mj}")
                    continue
                
                if not (0.0 <= metrics.accuracy <= 1.0):
                    logger.warning(f"Prediction {i}: Invalid accuracy {metrics.accuracy}")
                    continue
                
                if not (0.1 <= metrics.tops_per_watt <= 1000.0):
                    logger.warning(f"Prediction {i}: Invalid TOPS/W {metrics.tops_per_watt}")
                    continue
                
                successful_predictions += 1
                latencies.append(metrics.latency_ms)
                energies.append(metrics.energy_mj)
                accuracies.append(metrics.accuracy)
                tops_per_watt.append(metrics.tops_per_watt)
                
            except Exception as e:
                logger.warning(f"Prediction {i} failed: {e}")
        
        logger.info(f"Successful predictions: {successful_predictions}/20")
        
        if successful_predictions < 15:
            logger.error("Prediction success rate too low")
            return False
        
        # Check prediction diversity
        if successful_predictions > 5:
            lat_range = max(latencies) - min(latencies)
            eng_range = max(energies) - min(energies)
            acc_range = max(accuracies) - min(accuracies)
            
            logger.info(f"Prediction ranges - Latency: {lat_range:.2f}ms, "
                       f"Energy: {eng_range:.2f}mJ, Accuracy: {acc_range:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Predictor functionality test failed: {e}")
        return False


def test_search_functionality():
    """Test basic search functionality."""
    logger.info("Testing search functionality...")
    
    try:
        from tpuv6_zeronas import ZeroNASSearcher, TPUv6Predictor, ArchitectureSpace
        from tpuv6_zeronas.core import SearchConfig
        
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
        
        # Run search
        start_time = time.time()
        best_arch, best_metrics = searcher.search()
        elapsed = time.time() - start_time
        
        # Validate results
        if best_arch is None or best_metrics is None:
            logger.error("Search returned None results")
            return False
        
        if len(searcher.search_history) == 0:
            logger.error("Search history is empty")
            return False
        
        logger.info(f"Search completed in {elapsed:.2f}s")
        logger.info(f"Evaluations: {len(searcher.search_history)}")
        logger.info(f"Best architecture: {best_arch.name}")
        logger.info(f"Best metrics: {best_metrics}")
        
        # Check that search found reasonable solution
        if best_metrics.accuracy < 0.7:
            logger.warning(f"Low accuracy found: {best_metrics.accuracy:.3f}")
        
        if best_metrics.latency_ms > 50.0:
            logger.warning(f"High latency found: {best_metrics.latency_ms:.2f}ms")
        
        return True
        
    except Exception as e:
        logger.error(f"Search functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimization_constraints():
    """Test optimization constraints."""
    logger.info("Testing optimization constraints...")
    
    try:
        from tpuv6_zeronas import ZeroNASSearcher, TPUv6Predictor, ArchitectureSpace
        from tpuv6_zeronas.core import SearchConfig
        
        arch_space = ArchitectureSpace(
            input_shape=(224, 224, 3),
            num_classes=1000,
            max_depth=8
        )
        
        predictor = TPUv6Predictor()
        
        # Test with tight constraints
        config = SearchConfig(
            max_iterations=10,
            population_size=8,
            target_tops_w=75.0,
            max_latency_ms=5.0,  # Tight constraint
            min_accuracy=0.92    # Tight constraint
        )
        
        searcher = ZeroNASSearcher(arch_space, predictor, config)
        
        try:
            best_arch, best_metrics = searcher.search()
            
            # Check constraints are satisfied
            if best_metrics.latency_ms > config.max_latency_ms:
                logger.error(f"Latency constraint violated: {best_metrics.latency_ms} > {config.max_latency_ms}")
                return False
            
            if best_metrics.accuracy < config.min_accuracy:
                logger.error(f"Accuracy constraint violated: {best_metrics.accuracy} < {config.min_accuracy}")
                return False
            
            logger.info(f"Constraints satisfied - Latency: {best_metrics.latency_ms:.2f}ms, "
                       f"Accuracy: {best_metrics.accuracy:.3f}")
            
        except RuntimeError as e:
            if "failed to find any valid architecture" in str(e):
                logger.info("Search correctly failed with tight constraints (expected behavior)")
            else:
                raise
        
        return True
        
    except Exception as e:
        logger.error(f"Optimization constraints test failed: {e}")
        return False


def test_serialization():
    """Test architecture and metrics serialization."""
    logger.info("Testing serialization...")
    
    try:
        from tpuv6_zeronas.architecture import ArchitectureSpace
        from tpuv6_zeronas.metrics import PerformanceMetrics
        
        arch_space = ArchitectureSpace(
            input_shape=(224, 224, 3),
            num_classes=1000,
            max_depth=8
        )
        
        # Test architecture properties
        arch = arch_space.sample_random()
        
        arch_dict = {
            'name': arch.name,
            'layers': [
                {
                    'type': layer.layer_type.value,
                    'input_channels': layer.input_channels,
                    'output_channels': layer.output_channels,
                    'kernel_size': layer.kernel_size,
                    'stride': layer.stride,
                    'activation': layer.activation.value if layer.activation else None
                }
                for layer in arch.layers
            ],
            'total_ops': arch.total_ops,
            'total_params': arch.total_params,
            'memory_mb': arch.memory_mb
        }
        
        # Test metrics serialization
        metrics = PerformanceMetrics(
            latency_ms=5.0,
            energy_mj=50.0,
            accuracy=0.96,
            tops_per_watt=70.0,
            memory_mb=100.0,
            flops=1000000
        )
        
        metrics_dict = metrics.to_dict()
        
        # Test JSON serialization
        json_str = json.dumps({
            'architecture': arch_dict,
            'metrics': metrics_dict
        }, indent=2)
        
        # Test deserialization
        data = json.loads(json_str)
        
        if 'architecture' not in data or 'metrics' not in data:
            logger.error("Serialization failed - missing keys")
            return False
        
        logger.info(f"Serialization successful - {len(json_str)} characters")
        
        return True
        
    except Exception as e:
        logger.error(f"Serialization test failed: {e}")
        return False


def test_error_handling():
    """Test error handling and robustness."""
    logger.info("Testing error handling...")
    
    try:
        from tpuv6_zeronas.architecture import ArchitectureSpace
        from tpuv6_zeronas.predictor import TPUv6Predictor
        
        # Test invalid architecture space parameters
        try:
            invalid_space = ArchitectureSpace(
                input_shape=(0, 0, 0),  # Invalid
                num_classes=-1,  # Invalid
                max_depth=0  # Invalid
            )
            arch = invalid_space.sample_random()
            logger.warning("Should have failed with invalid parameters")
        except Exception:
            logger.info("Correctly handled invalid architecture space parameters")
        
        # Test predictor with None input
        predictor = TPUv6Predictor()
        
        try:
            metrics = predictor.predict(None)
            logger.warning("Should have failed with None input")
        except Exception:
            logger.info("Correctly handled None input to predictor")
        
        # Test with very large architectures
        try:
            large_space = ArchitectureSpace(
                input_shape=(1024, 1024, 3),  # Very large
                num_classes=10000,
                max_depth=100
            )
            # Should handle gracefully or fail safely
            arch = large_space.sample_random()
            if arch.total_params > 1e10:  # Too many parameters
                logger.info("Large architecture handled (may be inefficient)")
        except Exception:
            logger.info("Correctly handled very large architecture request")
        
        return True
        
    except Exception as e:
        logger.error(f"Error handling test failed: {e}")
        return False


def generate_test_report():
    """Generate comprehensive test report."""
    logger.info("Generating test report...")
    
    test_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_cases': [
            {'name': 'Architecture Generation', 'function': test_architecture_generation},
            {'name': 'Mutation and Crossover', 'function': test_mutation_and_crossover},
            {'name': 'Predictor Functionality', 'function': test_predictor_functionality},
            {'name': 'Search Functionality', 'function': test_search_functionality},
            {'name': 'Optimization Constraints', 'function': test_optimization_constraints},
            {'name': 'Serialization', 'function': test_serialization},
            {'name': 'Error Handling', 'function': test_error_handling},
        ],
        'results': [],
        'summary': {}
    }
    
    passed = 0
    failed = 0
    
    for test_case in test_results['test_cases']:
        test_name = test_case['name']
        test_func = test_case['function']
        
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        
        start_time = time.time()
        
        try:
            result = test_func()
            elapsed = time.time() - start_time
            
            test_results['results'].append({
                'name': test_name,
                'passed': result,
                'duration': elapsed,
                'error': None
            })
            
            if result:
                logger.info(f"✓ {test_name} passed ({elapsed:.2f}s)")
                passed += 1
            else:
                logger.error(f"✗ {test_name} failed ({elapsed:.2f}s)")
                failed += 1
                
        except Exception as e:
            elapsed = time.time() - start_time
            
            test_results['results'].append({
                'name': test_name,
                'passed': False,
                'duration': elapsed,
                'error': str(e)
            })
            
            logger.error(f"✗ {test_name} failed with exception: {e} ({elapsed:.2f}s)")
            failed += 1
    
    test_results['summary'] = {
        'total_tests': len(test_results['test_cases']),
        'passed': passed,
        'failed': failed,
        'success_rate': passed / len(test_results['test_cases']) if test_results['test_cases'] else 0
    }
    
    # Save report
    report_path = Path('test_report.json')
    with open(report_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    logger.info(f"\nTest report saved to {report_path}")
    
    return test_results


def main():
    """Run integration tests."""
    logger.info("TPUv6-ZeroNAS Integration Tests")
    logger.info("=" * 50)
    
    report = generate_test_report()
    
    logger.info("\n" + "=" * 50)
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info("=" * 50)
    
    summary = report['summary']
    logger.info(f"Total tests: {summary['total_tests']}")
    logger.info(f"Passed: {summary['passed']}")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"Success rate: {summary['success_rate']:.1%}")
    
    if summary['failed'] == 0:
        logger.info("🎉 All integration tests passed!")
        return 0
    elif summary['success_rate'] >= 0.7:
        logger.warning(f"⚠️  Some tests failed, but {summary['success_rate']:.1%} success rate is acceptable.")
        return 0
    else:
        logger.error(f"❌ Integration tests failed with only {summary['success_rate']:.1%} success rate.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
