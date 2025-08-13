#!/usr/bin/env python3
"""Quantum-Inspired NAS Research Demo: Advanced optimization with quantum computing principles."""

import logging
import time
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def main():
    """Demonstrate quantum-inspired neural architecture search."""
    print("\n" + "="*60)
    print("🧠 QUANTUM-INSPIRED NEURAL ARCHITECTURE SEARCH DEMO")
    print("="*60)
    
    try:
        from tpuv6_zeronas import ArchitectureSpace, TPUv6Predictor, SearchConfig
        from tpuv6_zeronas.quantum_nas import create_quantum_nas_searcher
        from tpuv6_zeronas.federated_nas import create_federated_nas_searcher
        from tpuv6_zeronas.neuromorphic_nas import create_neuromorphic_nas_searcher
        
        # Demo 1: Quantum-Inspired Search
        print("\n🚀 Demo 1: Quantum-Inspired Architecture Search")
        print("-" * 50)
        
        # Initialize components
        arch_space = ArchitectureSpace(
            input_shape=(224, 224, 3),
            num_classes=1000,
            max_depth=15
        )
        
        predictor = TPUv6Predictor()
        
        config = SearchConfig(
            max_iterations=8,
            population_size=12,
            target_tops_w=75.0,
            max_latency_ms=8.0,
            min_accuracy=0.92,
            enable_research=True
        )
        
        # Create quantum-inspired searcher
        quantum_searcher = create_quantum_nas_searcher(arch_space, predictor, config)
        
        print("🔬 Initializing quantum superposition state...")
        
        # Execute quantum search
        start_time = time.time()
        best_arch, best_metrics = quantum_searcher.search()
        search_time = time.time() - start_time
        
        print(f"\n✅ Quantum Search Results:")
        print(f"   Best Architecture: {best_arch.name if best_arch else 'None'}")
        if best_metrics:
            print(f"   Latency: {best_metrics.latency_ms:.2f}ms")
            print(f"   Accuracy: {best_metrics.accuracy:.3f}")
            print(f"   Energy: {best_metrics.energy_mj:.2f}mJ")
            print(f"   TOPS/W: {best_metrics.tops_per_watt:.1f}")
        print(f"   Search Time: {search_time:.2f}s")
        
        # Get quantum statistics
        quantum_stats = quantum_searcher.get_search_statistics()
        print(f"\n🔬 Quantum Search Statistics:")
        print(f"   Quantum Coherence: {quantum_stats['quantum_state']['coherence_time']:.3f}")
        print(f"   Entangled Pairs: {quantum_stats['quantum_state']['entangled_pairs']}")
        print(f"   RL Epsilon: {quantum_stats['rl_controller']['epsilon']:.3f}")
        print(f"   Strategy Performance: {len(quantum_stats['neuro_evolution']['strategy_performance'])} strategies")
        
        # Demo 2: Federated Learning Search
        print("\n🌐 Demo 2: Federated Neural Architecture Search")
        print("-" * 50)
        
        # Create federated searcher
        federated_searcher = create_federated_nas_searcher(
            arch_space, predictor, config, 
            num_nodes=8, privacy_epsilon=0.5
        )
        
        print("🔐 Initializing federated nodes with differential privacy...")
        
        # Execute federated search
        start_time = time.time()
        fed_best_arch, fed_best_metrics = federated_searcher.federated_search()
        fed_search_time = time.time() - start_time
        
        print(f"\n✅ Federated Search Results:")
        print(f"   Best Architecture: {fed_best_arch.name if fed_best_arch else 'None'}")
        if fed_best_metrics:
            print(f"   Latency: {fed_best_metrics.latency_ms:.2f}ms")
            print(f"   Accuracy: {fed_best_metrics.accuracy:.3f}")
            print(f"   Energy: {fed_best_metrics.energy_mj:.2f}mJ")
            print(f"   TOPS/W: {fed_best_metrics.tops_per_watt:.1f}")
        print(f"   Search Time: {fed_search_time:.2f}s")
        
        # Get federated statistics
        fed_stats = federated_searcher.get_federated_statistics()
        print(f"\n🌐 Federated Search Statistics:")
        print(f"   Participating Nodes: {fed_stats['federated_state']['participating_nodes']}")
        print(f"   Global Rounds: {fed_stats['federated_state']['global_round']}")
        print(f"   Privacy Budget Used: {fed_stats['privacy']['budget_consumed']:.1%}")
        print(f"   Communication Cost: {fed_stats['communication']['total_cost_mb']:.2f} MB")
        print(f"   Pareto Front Size: {fed_stats['federated_state']['pareto_front_size']}")
        
        # Demo 3: Neuromorphic Computing Search
        print("\n🧠 Demo 3: Neuromorphic Neural Architecture Search")
        print("-" * 50)
        
        # Create neuromorphic searcher
        neuromorphic_searcher = create_neuromorphic_nas_searcher(arch_space, predictor, config)
        
        print("⚡ Initializing spiking neural networks...")
        
        # Execute neuromorphic search
        start_time = time.time()
        neuro_best_arch, neuro_best_metrics = neuromorphic_searcher.search()
        neuro_search_time = time.time() - start_time
        
        print(f"\n✅ Neuromorphic Search Results:")
        print(f"   Best Architecture: {neuro_best_arch.name if neuro_best_arch else 'None'}")
        if neuro_best_metrics:
            print(f"   Latency: {neuro_best_metrics.latency_ms:.2f}ms")
            print(f"   Accuracy: {neuro_best_metrics.accuracy:.3f}")
            print(f"   Energy: {neuro_best_metrics.energy_mj:.2f}mJ")
            print(f"   TOPS/W: {neuro_best_metrics.tops_per_watt:.1f}")
            print(f"   Spike Sparsity: {getattr(neuro_best_metrics, 'spike_sparsity', 0.0):.3f}")
        print(f"   Search Time: {neuro_search_time:.2f}s")
        
        # Get neuromorphic statistics
        neuro_stats = neuromorphic_searcher.get_neuromorphic_statistics()
        if neuro_stats:
            print(f"\n🧠 Neuromorphic Search Statistics:")
            print(f"   Generations: {neuro_stats['search_progress']['generations_completed']}")
            print(f"   Avg Spike Sparsity: {neuro_stats['neuromorphic_metrics']['avg_spike_sparsity']:.3f}")
            print(f"   Neuromorphic Layers: {neuro_stats['neuromorphic_metrics']['total_neuromorphic_layers']}")
            print(f"   Population Diversity: {neuro_stats['search_progress']['population_diversity']:.3f}")
        
        # Comparative Analysis
        print("\n📊 COMPARATIVE ANALYSIS")
        print("-" * 50)
        
        results = []
        if best_metrics:
            results.append(("Quantum-Inspired", best_metrics, search_time))
        if fed_best_metrics:
            results.append(("Federated", fed_best_metrics, fed_search_time))
        if neuro_best_metrics:
            results.append(("Neuromorphic", neuro_best_metrics, neuro_search_time))
        
        if results:
            print("Method           | Latency(ms) | Accuracy | Energy(mJ) | TOPS/W | Time(s)")
            print("-" * 70)
            for method, metrics, search_time in results:
                print(f"{method:<15} | {metrics.latency_ms:>9.2f} | {metrics.accuracy:>8.3f} | "
                      f"{metrics.energy_mj:>9.2f} | {metrics.tops_per_watt:>6.1f} | {search_time:>6.1f}")
        
        # Save comprehensive results
        research_results = {
            'experiment_type': 'quantum_research_demo',
            'timestamp': time.time(),
            'quantum_search': {
                'architecture': best_arch.name if best_arch else None,
                'metrics': {
                    'latency_ms': best_metrics.latency_ms if best_metrics else None,
                    'accuracy': best_metrics.accuracy if best_metrics else None,
                    'energy_mj': best_metrics.energy_mj if best_metrics else None,
                    'tops_per_watt': best_metrics.tops_per_watt if best_metrics else None
                } if best_metrics else None,
                'search_time': search_time,
                'statistics': quantum_stats
            },
            'federated_search': {
                'architecture': fed_best_arch.name if fed_best_arch else None,
                'metrics': {
                    'latency_ms': fed_best_metrics.latency_ms if fed_best_metrics else None,
                    'accuracy': fed_best_metrics.accuracy if fed_best_metrics else None,
                    'energy_mj': fed_best_metrics.energy_mj if fed_best_metrics else None,
                    'tops_per_watt': fed_best_metrics.tops_per_watt if fed_best_metrics else None
                } if fed_best_metrics else None,
                'search_time': fed_search_time,
                'statistics': fed_stats
            },
            'neuromorphic_search': {
                'architecture': neuro_best_arch.name if neuro_best_arch else None,
                'metrics': {
                    'latency_ms': neuro_best_metrics.latency_ms if neuro_best_metrics else None,
                    'accuracy': neuro_best_metrics.accuracy if neuro_best_metrics else None,
                    'energy_mj': neuro_best_metrics.energy_mj if neuro_best_metrics else None,
                    'tops_per_watt': neuro_best_metrics.tops_per_watt if neuro_best_metrics else None,
                    'spike_sparsity': getattr(neuro_best_metrics, 'spike_sparsity', None) if neuro_best_metrics else None
                } if neuro_best_metrics else None,
                'search_time': neuro_search_time,
                'statistics': neuro_stats
            }
        }
        
        # Save results
        output_file = Path('quantum_research_results.json')
        with open(output_file, 'w') as f:
            json.dump(research_results, f, indent=2, default=str)
        
        print(f"\n💾 Research results saved to: {output_file}")
        
        # Research insights
        print("\n🔬 RESEARCH INSIGHTS")
        print("-" * 50)
        print("1. Quantum-inspired search leverages superposition and entanglement")
        print("   principles for enhanced exploration of architecture space.")
        print("2. Federated search enables privacy-preserving collaborative")
        print("   architecture discovery across distributed nodes.")
        print("3. Neuromorphic search optimizes for spike sparsity and")
        print("   temporal dynamics, achieving ultra-low energy consumption.")
        print("4. Each approach offers unique advantages for different")
        print("   deployment scenarios and hardware targets.")
        
        print(f"\n🎉 Quantum Research Demo completed successfully!")
        print("   Advanced search algorithms demonstrate significant potential")
        print("   for next-generation neural architecture optimization.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Please ensure tpuv6_zeronas is properly installed:")
        print("   pip install -e .")
    except Exception as e:
        logging.error(f"Demo error: {e}")
        print(f"❌ Demo failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())