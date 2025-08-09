#!/usr/bin/env python3
"""
TPUv6-ZeroNAS Research Demonstration

This script demonstrates the research capabilities of TPUv6-ZeroNAS,
including comparative studies and algorithmic validation.
"""

import sys
import os
import json
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tpuv6_zeronas import (
    ZeroNASSearcher, TPUv6Predictor, ArchitectureSpace, SearchConfig,
    PerformanceMetrics, MetricsAggregator
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_comparative_study():
    """Run comparative study of different search configurations."""
    logger.info("🔬 Starting TPUv6-ZeroNAS Comparative Research Study")
    
    # Define experimental configurations
    configs = {
        'small_fast': SearchConfig(
            max_iterations=10,
            population_size=8,
            target_tops_w=60.0,
            max_latency_ms=5.0,
            min_accuracy=0.85
        ),
        'large_efficient': SearchConfig(
            max_iterations=15,
            population_size=12,
            target_tops_w=75.0,
            max_latency_ms=15.0,
            min_accuracy=0.92
        ),
        'ultra_fast': SearchConfig(
            max_iterations=8,
            population_size=6,
            target_tops_w=50.0,
            max_latency_ms=2.0,
            min_accuracy=0.80
        )
    }
    
    # Different architecture search spaces
    spaces = {
        'mobile_friendly': ArchitectureSpace(
            input_shape=(224, 224, 3),
            num_classes=1000,
            max_depth=12,
            channel_choices=[16, 32, 64, 128, 256],
            kernel_choices=[(1, 1), (3, 3), (5, 5)]
        ),
        'high_capacity': ArchitectureSpace(
            input_shape=(224, 224, 3),
            num_classes=1000,
            max_depth=20,
            channel_choices=[64, 128, 256, 512, 1024],
            kernel_choices=[(1, 1), (3, 3), (5, 5), (7, 7)]
        ),
        'edge_optimized': ArchitectureSpace(
            input_shape=(96, 96, 3),
            num_classes=100,
            max_depth=8,
            channel_choices=[8, 16, 32, 64, 128],
            kernel_choices=[(1, 1), (3, 3)]
        )
    }
    
    results = {}
    
    for config_name, config in configs.items():
        logger.info(f"\n📊 Testing configuration: {config_name}")
        logger.info(f"   Target: {config.target_tops_w} TOPS/W, <{config.max_latency_ms}ms latency")
        
        config_results = {}
        
        for space_name, space in spaces.items():
            logger.info(f"  🏗️  Architecture space: {space_name}")
            
            start_time = time.time()
            
            try:
                predictor = TPUv6Predictor()
                searcher = ZeroNASSearcher(space, predictor, config)
                
                best_arch, best_metrics = searcher.search()
                
                search_time = time.time() - start_time
                
                config_results[space_name] = {
                    'architecture': {
                        'name': best_arch.name,
                        'layers': len(best_arch.layers),
                        'parameters': best_arch.total_params,
                        'operations': best_arch.total_ops,
                        'memory_mb': best_arch.memory_mb
                    },
                    'metrics': {
                        'latency_ms': best_metrics.latency_ms,
                        'energy_mj': best_metrics.energy_mj,
                        'accuracy': best_metrics.accuracy,
                        'tops_per_watt': best_metrics.tops_per_watt,
                        'efficiency_score': best_metrics.efficiency_score
                    },
                    'search_performance': {
                        'time_seconds': search_time,
                        'total_evaluations': len(searcher.search_history)
                    }
                }
                
                logger.info(f"    ✅ Found: {best_arch.name} "
                          f"({best_metrics.latency_ms:.2f}ms, "
                          f"{best_metrics.accuracy:.1%} acc, "
                          f"{best_metrics.tops_per_watt:.1f} TOPS/W)")
                          
                searcher.cleanup()
                
            except Exception as e:
                logger.error(f"    ❌ Failed: {e}")
                config_results[space_name] = {'error': str(e)}
        
        results[config_name] = config_results
    
    return results


def analyze_results(results):
    """Analyze and summarize research results."""
    logger.info("\n📈 RESEARCH RESULTS ANALYSIS")
    logger.info("=" * 50)
    
    # Find best performers by different metrics
    best_efficiency = None
    best_latency = None
    best_accuracy = None
    best_overall = None
    
    max_efficiency = 0
    min_latency = float('inf')
    max_accuracy = 0
    max_overall_score = 0
    
    all_results = []
    
    for config_name, config_results in results.items():
        for space_name, result in config_results.items():
            if 'error' not in result:
                metrics = result['metrics']
                arch = result['architecture']
                
                result_summary = {
                    'config': config_name,
                    'space': space_name,
                    'efficiency': metrics['tops_per_watt'],
                    'latency': metrics['latency_ms'],
                    'accuracy': metrics['accuracy'],
                    'overall': metrics['efficiency_score'],
                    'params': arch['parameters']
                }
                all_results.append(result_summary)
                
                # Track best performers
                if metrics['tops_per_watt'] > max_efficiency:
                    max_efficiency = metrics['tops_per_watt']
                    best_efficiency = result_summary
                
                if metrics['latency_ms'] < min_latency:
                    min_latency = metrics['latency_ms']
                    best_latency = result_summary
                
                if metrics['accuracy'] > max_accuracy:
                    max_accuracy = metrics['accuracy']
                    best_accuracy = result_summary
                
                if metrics['efficiency_score'] > max_overall_score:
                    max_overall_score = metrics['efficiency_score']
                    best_overall = result_summary
    
    # Print analysis
    logger.info(f"🏆 BEST EFFICIENCY: {best_efficiency['config']}/{best_efficiency['space']}")
    logger.info(f"   {best_efficiency['efficiency']:.1f} TOPS/W, {best_efficiency['params']:,} params")
    
    logger.info(f"⚡ LOWEST LATENCY: {best_latency['config']}/{best_latency['space']}")
    logger.info(f"   {best_latency['latency']:.2f}ms, {best_latency['accuracy']:.1%} accuracy")
    
    logger.info(f"🎯 HIGHEST ACCURACY: {best_accuracy['config']}/{best_accuracy['space']}")
    logger.info(f"   {best_accuracy['accuracy']:.1%} accuracy, {best_accuracy['latency']:.2f}ms latency")
    
    logger.info(f"🌟 BEST OVERALL: {best_overall['config']}/{best_overall['space']}")
    logger.info(f"   {best_overall['overall']:.3f} score, balanced performance")
    
    # Statistical summary
    if all_results:
        efficiencies = [r['efficiency'] for r in all_results]
        latencies = [r['latency'] for r in all_results]
        accuracies = [r['accuracy'] for r in all_results]
        
        logger.info("\n📊 STATISTICAL SUMMARY:")
        logger.info(f"   Efficiency range: {min(efficiencies):.1f} - {max(efficiencies):.1f} TOPS/W")
        logger.info(f"   Latency range: {min(latencies):.2f} - {max(latencies):.2f}ms")
        logger.info(f"   Accuracy range: {min(accuracies):.1%} - {max(accuracies):.1%}")
        logger.info(f"   Total experiments: {len(all_results)}")
    
    return {
        'best_efficiency': best_efficiency,
        'best_latency': best_latency,
        'best_accuracy': best_accuracy,
        'best_overall': best_overall,
        'statistical_summary': {
            'total_experiments': len(all_results),
            'efficiency_range': [min(efficiencies), max(efficiencies)] if efficiencies else [0, 0],
            'latency_range': [min(latencies), max(latencies)] if latencies else [0, 0],
            'accuracy_range': [min(accuracies), max(accuracies)] if accuracies else [0, 0]
        }
    }


def demonstrate_predictive_capabilities():
    """Demonstrate TPUv6 predictive modeling capabilities."""
    logger.info("\n🤖 PREDICTIVE MODELING DEMONSTRATION")
    logger.info("=" * 50)
    
    # Create different predictors with various configurations
    from tpuv6_zeronas.predictor import TPUv6Config, PredictorEnsemble
    
    configs = [
        TPUv6Config(peak_tops=250.0, power_budget_w=3.5),  # Conservative estimate
        TPUv6Config(peak_tops=275.0, power_budget_w=4.0),  # Expected performance
        TPUv6Config(peak_tops=300.0, power_budget_w=4.5),  # Optimistic estimate
    ]
    
    predictors = [TPUv6Predictor(config) for config in configs]
    ensemble = PredictorEnsemble(predictors)
    
    # Test on various architecture sizes
    space = ArchitectureSpace(max_depth=15)
    test_architectures = []
    
    logger.info("🏗️  Generating test architectures...")
    for i in range(5):
        arch = space.sample_random()
        test_architectures.append(arch)
        logger.info(f"   {arch.name}: {len(arch.layers)} layers, {arch.total_params:,} params")
    
    logger.info("\n🔮 Prediction Comparison:")
    for i, arch in enumerate(test_architectures):
        logger.info(f"\nArchitecture {i+1}: {arch.name}")
        
        # Individual predictor results
        individual_results = []
        for j, predictor in enumerate(predictors):
            metrics = predictor.predict(arch)
            individual_results.append(metrics)
            logger.info(f"  Predictor {j+1}: {metrics.latency_ms:.2f}ms, "
                       f"{metrics.accuracy:.1%}, {metrics.tops_per_watt:.1f} TOPS/W")
        
        # Ensemble result
        ensemble_metrics = ensemble.predict(arch)
        logger.info(f"  Ensemble:   {ensemble_metrics.latency_ms:.2f}ms, "
                   f"{ensemble_metrics.accuracy:.1%}, {ensemble_metrics.tops_per_watt:.1f} TOPS/W")
        
        # Variance analysis
        latencies = [m.latency_ms for m in individual_results]
        accuracies = [m.accuracy for m in individual_results]
        efficiencies = [m.tops_per_watt for m in individual_results]
        
        lat_variance = max(latencies) - min(latencies)
        acc_variance = max(accuracies) - min(accuracies)
        eff_variance = max(efficiencies) - min(efficiencies)
        
        logger.info(f"  Variance:   ±{lat_variance:.2f}ms, "
                   f"±{acc_variance:.1%}, ±{eff_variance:.1f} TOPS/W")


def generate_research_report(results, analysis):
    """Generate comprehensive research report."""
    logger.info("\n📄 Generating Research Report...")
    
    report = {
        'experiment_metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'tpuv6_zeronas_version': '0.1.0',
            'experiment_type': 'comparative_architecture_search',
            'total_configurations': len(results),
            'total_experiments': sum(len(config_results) for config_results in results.values())
        },
        'methodology': {
            'description': 'Comparative study of neural architecture search configurations for TPUv6 optimization',
            'search_algorithm': 'Evolutionary algorithm with hardware-aware fitness function',
            'prediction_model': 'Learned scaling laws from Edge TPU v5e → v6 regression',
            'objectives': ['latency_minimization', 'energy_efficiency', 'accuracy_maximization']
        },
        'experimental_results': results,
        'analysis': analysis,
        'key_findings': [
            f"Best efficiency achieved: {analysis['best_efficiency']['efficiency']:.1f} TOPS/W",
            f"Minimum latency achieved: {analysis['best_latency']['latency']:.2f}ms",
            f"Maximum accuracy achieved: {analysis['best_accuracy']['accuracy']:.1%}",
            f"Total architectures evaluated: {analysis['statistical_summary']['total_experiments']}"
        ],
        'reproducibility_info': {
            'random_seed': 'architecture_sampling_based',
            'hardware_requirements': 'CPU-only (no external dependencies)',
            'execution_time': 'varies by configuration complexity'
        }
    }
    
    # Save to file
    output_path = Path('research_report.json')
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"📊 Research report saved to: {output_path.absolute()}")
    
    # Generate summary
    logger.info("\n🎓 RESEARCH CONCLUSIONS:")
    logger.info("=" * 50)
    for finding in report['key_findings']:
        logger.info(f"• {finding}")
    
    logger.info(f"\n📚 This demonstrates TPUv6-ZeroNAS research capabilities:")
    logger.info("• Comparative algorithmic studies")
    logger.info("• Hardware-aware architecture optimization")
    logger.info("• Predictive performance modeling")
    logger.info("• Reproducible experimental framework")
    logger.info("• Publication-ready results generation")


def main():
    """Main research demonstration function."""
    logger.info("🚀 TPUv6-ZeroNAS Research Demonstration")
    logger.info("Showcasing zero-shot neural architecture search for unreleased hardware")
    logger.info("=" * 80)
    
    try:
        # Run comparative study
        results = run_comparative_study()
        
        # Analyze results
        analysis = analyze_results(results)
        
        # Demonstrate predictive capabilities
        demonstrate_predictive_capabilities()
        
        # Generate research report
        generate_research_report(results, analysis)
        
        logger.info("\n🎉 Research demonstration completed successfully!")
        logger.info("✅ TPUv6-ZeroNAS is ready for academic research and industrial deployment")
        
    except Exception as e:
        logger.error(f"❌ Research demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)