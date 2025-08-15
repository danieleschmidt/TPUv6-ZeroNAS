#!/usr/bin/env python3
"""Autonomous Research Demo: Self-improving NAS with Meta-Learning and Scientific Discovery."""

import asyncio
import logging
import time
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

async def main():
    """Demonstrate autonomous research capabilities with self-improving NAS."""
    print("\n" + "="*70)
    print("🧬 AUTONOMOUS RESEARCH ACCELERATOR DEMO")
    print("="*70)
    
    try:
        from tpuv6_zeronas import ArchitectureSpace, TPUv6Predictor, SearchConfig
        from tpuv6_zeronas.autonomous_research_accelerator import create_autonomous_research_accelerator
        
        # Initialize components for research
        print("\n🔬 Initializing Research Infrastructure")
        print("-" * 50)
        
        arch_space = ArchitectureSpace(
            input_shape=(224, 224, 3),
            num_classes=1000,
            max_depth=20
        )
        
        predictor = TPUv6Predictor(enable_uncertainty=True, enable_caching=True)
        
        config = SearchConfig(
            max_iterations=5,  # Short for demo
            population_size=8,
            target_tops_w=75.0,
            max_latency_ms=10.0,
            min_accuracy=0.90,
            enable_research=True,
            enable_parallel=True
        )
        
        # Create autonomous research accelerator
        research_accelerator = create_autonomous_research_accelerator(
            arch_space, predictor, config
        )
        
        print("✅ Autonomous Research Accelerator initialized")
        print(f"📊 Available research algorithms: {len(research_accelerator.search_algorithms)}")
        print(f"🔬 Discovery engines active: {len(research_accelerator.discovery_engines)}")
        
        # Run autonomous research campaign
        print("\n🧪 Starting Autonomous Research Campaign")
        print("-" * 50)
        
        campaign_results = await research_accelerator.run_autonomous_research_campaign(
            campaign_duration_hours=0.1,  # 6 minutes for demo
            max_parallel_experiments=2
        )
        
        print("\n📊 RESEARCH CAMPAIGN RESULTS")
        print("=" * 50)
        
        # Display campaign summary
        print(f"Campaign ID: {campaign_results['campaign_id']}")
        print(f"Duration: {campaign_results['actual_duration']/60:.1f} minutes")
        print(f"Experiments Completed: {len(campaign_results['experiments'])}")
        print(f"Novel Discoveries: {len(campaign_results['discoveries'])}")
        print(f"Meta-Learning Updates: {len(campaign_results['meta_learning_improvements'])}")
        
        # Display experiments
        if campaign_results['experiments']:
            print("\n🧪 EXPERIMENTAL RESULTS")
            print("-" * 40)
            for i, experiment in enumerate(campaign_results['experiments'], 1):
                print(f"\nExperiment {i}: {experiment.experiment_id}")
                print(f"  Hypothesis: {experiment.hypothesis.description[:60]}...")
                
                # Display conclusions
                conclusions = experiment.results.get('conclusions', [])
                if conclusions:
                    print("  Key Findings:")
                    for conclusion in conclusions[:3]:  # Show first 3
                        print(f"    • {conclusion}")
                
                # Display statistical significance
                stats = experiment.results.get('statistical_analysis', {})
                if 'accuracy' in stats:
                    acc_stats = stats['accuracy']
                    print(f"  Accuracy Effect: {acc_stats.get('difference', 0):.3f} ({acc_stats.get('significance', 'unknown')})")
        
        # Display discoveries
        if campaign_results['discoveries']:
            print("\n💡 NOVEL DISCOVERIES")
            print("-" * 40)
            for i, discovery in enumerate(campaign_results['discoveries'], 1):
                print(f"\n{i}. {discovery['type'].title()}")
                print(f"   {discovery['description']}")
                print(f"   Confidence: {discovery.get('confidence', 'unknown')}")
        
        # Display novel insights
        if campaign_results['novel_insights']:
            print("\n🎯 SCIENTIFIC INSIGHTS")
            print("-" * 40)
            for i, insight in enumerate(campaign_results['novel_insights'], 1):
                print(f"{i}. {insight}")
        
        # Display meta-learning improvements
        if campaign_results['meta_learning_improvements']:
            print("\n🔄 META-LEARNING IMPROVEMENTS")
            print("-" * 40)
            for improvement in campaign_results['meta_learning_improvements']:
                print(f"  • {improvement}")
        
        # Save research results
        results_path = Path('autonomous_research_results.json')
        with open(results_path, 'w') as f:
            # Convert experiment objects to dictionaries for JSON serialization
            serializable_results = campaign_results.copy()
            serializable_results['experiments'] = [
                {
                    'experiment_id': exp.experiment_id,
                    'hypothesis': {
                        'description': exp.hypothesis.description,
                        'variables': exp.hypothesis.variables,
                        'confidence_level': exp.hypothesis.confidence_level
                    },
                    'results': exp.results,
                    'duration': (exp.end_time - exp.start_time) if exp.end_time and exp.start_time else 0
                }
                for exp in campaign_results['experiments']
            ]
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"\n💾 Research results saved to {results_path}")
        
        # Performance summary
        total_experiments = len(campaign_results['experiments'])
        significant_findings = sum(
            1 for exp in campaign_results['experiments'] 
            if any('significant' in c.lower() for c in exp.results.get('conclusions', []))
        )
        
        print("\n📈 PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"Research Efficiency: {significant_findings}/{total_experiments} experiments with significant findings")
        print(f"Discovery Rate: {len(campaign_results['discoveries'])} novel discoveries")
        print(f"Insight Generation: {len(campaign_results['novel_insights'])} scientific insights")
        print(f"Meta-Learning: {len(campaign_results['meta_learning_improvements'])} algorithm improvements")
        
        # Research quality assessment
        if total_experiments > 0:
            quality_score = (
                (significant_findings / total_experiments) * 0.4 +
                (min(len(campaign_results['discoveries']), 5) / 5) * 0.3 +
                (min(len(campaign_results['novel_insights']), 5) / 5) * 0.3
            )
            print(f"Research Quality Score: {quality_score:.2f}/1.00")
        
        print("\n🎯 Autonomous research campaign completed successfully!")
        print("The system demonstrated self-improving capabilities with:")
        print("  • Automated hypothesis generation")
        print("  • Controlled experimental design")
        print("  • Statistical analysis and validation")
        print("  • Meta-learning algorithm adaptation")
        print("  • Novel scientific insight discovery")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Some advanced features may not be available.")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())