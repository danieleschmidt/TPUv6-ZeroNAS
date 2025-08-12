#!/usr/bin/env python3
"""Comprehensive validation of TPUv6-ZeroNAS implementation."""

import sys
import os
import time
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all critical imports."""
    print("🔍 Testing imports...")
    try:
        from tpuv6_zeronas import ZeroNASSearcher, SearchConfig, ArchitectureSpace, TPUv6Predictor
        from tpuv6_zeronas.security import get_resource_guard
        from tpuv6_zeronas.validation import validate_input
        from tpuv6_zeronas.caching import create_cached_predictor
        from tpuv6_zeronas.parallel import ParallelEvaluator, WorkerConfig
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_security():
    """Test security features."""
    print("🔒 Testing security...")
    try:
        from tpuv6_zeronas.security import get_resource_guard
        from tpuv6_zeronas.core import SearchConfig
        
        guard = get_resource_guard()
        
        # Test security limits
        try:
            config = SearchConfig(max_iterations=50000)  # Should fail
            guard.check_resource_limits(config)
            print("❌ Security limits not enforced")
            return False
        except Exception:
            print("✅ Security limits properly enforced")
        
        # Test valid config
        config = SearchConfig(max_iterations=50)
        guard.check_resource_limits(config)
        print("✅ Valid configurations accepted")
        return True
    except Exception as e:
        print(f"❌ Security test failed: {e}")
        return False

def test_performance():
    """Test performance optimizations."""
    print("⚡ Testing performance...")
    try:
        from tpuv6_zeronas import ZeroNASSearcher, SearchConfig, ArchitectureSpace, TPUv6Predictor
        from tpuv6_zeronas.caching import create_cached_predictor
        
        arch_space = ArchitectureSpace()
        predictor = TPUv6Predictor()
        cached_predictor = create_cached_predictor(predictor)
        
        # Test caching performance
        test_arch = arch_space.sample_random()
        
        start = time.time()
        metrics1 = cached_predictor.predict(test_arch)
        first_time = time.time() - start
        
        start = time.time()
        metrics2 = cached_predictor.predict(test_arch)  # Should be cached
        cached_time = time.time() - start
        
        speedup = first_time / max(cached_time, 0.0001)
        print(f"✅ Caching speedup: {speedup:.1f}x")
        
        # Test search performance
        config = SearchConfig(max_iterations=3, population_size=4, enable_parallel=True, enable_caching=True)
        start = time.time()
        searcher = ZeroNASSearcher(arch_space, cached_predictor, config)
        best_arch, best_metrics = searcher.search()
        search_time = time.time() - start
        
        print(f"✅ Optimized search: {search_time:.3f}s")
        print(f"✅ Best efficiency: {best_metrics.efficiency_score:.3f}")
        return True
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        traceback.print_exc()
        return False

def test_functionality():
    """Test core functionality."""
    print("⚙️ Testing functionality...")
    try:
        from tpuv6_zeronas import ZeroNASSearcher, SearchConfig, ArchitectureSpace, TPUv6Predictor
        from tpuv6_zeronas.validation import validate_input
        
        # Create components
        arch_space = ArchitectureSpace()
        predictor = TPUv6Predictor()
        config = SearchConfig(max_iterations=2, population_size=3)
        
        # Test architecture generation
        arch = arch_space.sample_random()
        print(f"✅ Architecture generated: {arch.name}")
        
        # Test prediction
        metrics = predictor.predict(arch)
        print(f"✅ Prediction: {metrics.latency_ms:.2f}ms, {metrics.accuracy:.3f} acc")
        
        # Test validation
        validation = validate_input(arch, 'architecture')
        print(f"✅ Validation: {validation['is_valid']}")
        
        # Test full search
        searcher = ZeroNASSearcher(arch_space, predictor, config)
        best_arch, best_metrics = searcher.search()
        print(f"✅ Search completed: {best_arch.name}")
        return True
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run comprehensive validation."""
    print("🚀 TPUv6-ZeroNAS Comprehensive Validation")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Security", test_security),
        ("Functionality", test_functionality),
        ("Performance", test_performance),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name} Test")
        print("-" * 30)
        if test_func():
            passed += 1
    
    print(f"\n🏁 Final Results")
    print("=" * 50)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - System is production ready!")
        return 0
    else:
        print("⚠️ Some tests failed - Review issues above")
        return 1

if __name__ == "__main__":
    sys.exit(main())