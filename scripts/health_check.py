#!/usr/bin/env python3
"""Health check script for TPUv6-ZeroNAS deployment monitoring."""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add the package to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tpuv6_zeronas import ZeroNASSearcher, TPUv6Predictor, ArchitectureSpace, SearchConfig


def check_import_health() -> Dict[str, Any]:
    """Check if all modules can be imported."""
    try:
        import tpuv6_zeronas
        import numpy
        import sklearn
        import pandas
        
        return {
            'status': 'healthy',
            'message': 'All imports successful',
            'tpuv6_version': tpuv6_zeronas.__version__
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'message': f'Import error: {str(e)}'
        }


def check_basic_functionality() -> Dict[str, Any]:
    """Test basic functionality."""
    try:
        # Create minimal components
        arch_space = ArchitectureSpace(
            input_shape=(32, 32, 3),
            num_classes=10,
            max_depth=5
        )
        
        predictor = TPUv6Predictor()
        
        # Test architecture generation
        arch = arch_space.sample_random()
        assert arch is not None
        
        # Test prediction
        metrics = predictor.predict(arch)
        assert metrics is not None
        assert hasattr(metrics, 'accuracy')
        assert hasattr(metrics, 'latency_ms')
        
        return {
            'status': 'healthy',
            'message': 'Basic functionality working',
            'test_results': {
                'architecture_generated': True,
                'prediction_made': True,
                'accuracy': float(metrics.accuracy),
                'latency_ms': float(metrics.latency_ms)
            }
        }
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'message': f'Functionality error: {str(e)}'
        }


def check_search_capability() -> Dict[str, Any]:
    """Test minimal search capability."""
    try:
        arch_space = ArchitectureSpace(
            input_shape=(32, 32, 3),
            num_classes=10,
            max_depth=5
        )
        
        predictor = TPUv6Predictor()
        
        config = SearchConfig(
            max_iterations=2,
            population_size=3,
            min_accuracy=0.1,  # Lower threshold for health check
            enable_parallel=True,
            enable_caching=True
        )
        
        searcher = ZeroNASSearcher(arch_space, predictor, config)
        
        start_time = time.time()
        best_arch, best_metrics = searcher.search()
        search_time = time.time() - start_time
        
        searcher.cleanup()
        
        return {
            'status': 'healthy',
            'message': 'Search capability working',
            'test_results': {
                'search_completed': True,
                'search_time_seconds': round(search_time, 2),
                'best_accuracy': float(best_metrics.accuracy) if best_metrics else None,
                'optimizations_enabled': {
                    'parallel': config.enable_parallel,
                    'caching': config.enable_caching,
                    'adaptive': config.enable_adaptive
                }
            }
        }
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'message': f'Search error: {str(e)}'
        }


def check_system_resources() -> Dict[str, Any]:
    """Check system resource availability."""
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        disk = psutil.disk_usage('/')
        
        # Check minimum requirements
        min_memory_gb = 2
        min_disk_gb = 1
        
        memory_gb = memory.total / (1024**3)
        disk_gb = disk.free / (1024**3)
        
        issues = []
        if memory_gb < min_memory_gb:
            issues.append(f'Low memory: {memory_gb:.1f}GB < {min_memory_gb}GB required')
        
        if disk_gb < min_disk_gb:
            issues.append(f'Low disk space: {disk_gb:.1f}GB < {min_disk_gb}GB required')
        
        status = 'healthy' if not issues else 'warning'
        
        return {
            'status': status,
            'message': 'System resources checked',
            'resources': {
                'memory_gb': round(memory_gb, 1),
                'memory_available_gb': round(memory.available / (1024**3), 1),
                'cpu_count': cpu_count,
                'disk_free_gb': round(disk_gb, 1)
            },
            'issues': issues
        }
        
    except ImportError:
        return {
            'status': 'warning',
            'message': 'psutil not available for resource monitoring'
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'message': f'Resource check error: {str(e)}'
        }


def main():
    """Run comprehensive health checks."""
    print("🏥 TPUv6-ZeroNAS Health Check")
    print("=" * 40)
    
    checks = [
        ("Import Health", check_import_health),
        ("Basic Functionality", check_basic_functionality),
        ("Search Capability", check_search_capability),
        ("System Resources", check_system_resources)
    ]
    
    overall_status = 'healthy'
    results = {}
    
    for check_name, check_func in checks:
        print(f"\n🔍 Running {check_name}...")
        
        try:
            result = check_func()
            results[check_name.lower().replace(' ', '_')] = result
            
            status_emoji = {
                'healthy': '✅',
                'warning': '⚠️',
                'unhealthy': '❌'
            }.get(result['status'], '❓')
            
            print(f"{status_emoji} {result['message']}")
            
            if result['status'] == 'unhealthy':
                overall_status = 'unhealthy'
            elif result['status'] == 'warning' and overall_status == 'healthy':
                overall_status = 'warning'
                
        except Exception as e:
            error_result = {
                'status': 'unhealthy',
                'message': f'Health check failed: {str(e)}'
            }
            results[check_name.lower().replace(' ', '_')] = error_result
            print(f"❌ Health check failed: {str(e)}")
            overall_status = 'unhealthy'
    
    # Summary
    print(f"\n📋 Overall Status: {overall_status.upper()}")
    
    status_messages = {
        'healthy': '🎉 All systems operational!',
        'warning': '⚠️  System operational with warnings',
        'unhealthy': '🚨 Critical issues detected'
    }
    
    print(status_messages.get(overall_status, 'Unknown status'))
    
    # Output JSON for programmatic use
    if '--json' in sys.argv:
        health_report = {
            'timestamp': time.time(),
            'overall_status': overall_status,
            'checks': results
        }
        print(json.dumps(health_report, indent=2))
    
    # Exit with appropriate code
    exit_code = {
        'healthy': 0,
        'warning': 1,
        'unhealthy': 2
    }.get(overall_status, 2)
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()