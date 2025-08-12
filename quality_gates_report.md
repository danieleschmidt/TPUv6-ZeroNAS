
# TPUv6-ZeroNAS Quality Gates Report

**Generated:** 2025-08-12 13:01:12
**Overall Status:** ❌ FAILED
**Overall Score:** 21.7%

## Summary

| Quality Gate | Status | Score | Details |
|--------------|--------|-------|---------|
| Test Coverage | ❌ FAILED | 13.1% | 6 items checked |
| Security | ❌ FAILED | 0.0% | 3 items checked |
| Performance | ✅ PASSED | 86.7% | 4 items checked |
| Documentation | ❌ FAILED | 41.1% | 5 items checked |

**Total Execution Time:** 0.09 seconds

## Test Coverage

❌ **Status:** FAILED (Score: 13.1%)

**Details:**
- test_files_found: 6
- total_coverage: 13.143227478937137
- module_coverage: {}
- lines_covered: 1014
- total_lines: 7715
- test_lines: 338


## Security

❌ **Status:** FAILED (Score: 0.0%)

**Details:**
- issues_found: 34
- security_issues: ['Uncontrolled file write in novel_research_demo.py', 'Uncontrolled file write in run_integration_tests.py', 'Uncontrolled file write in caching.py', 'Uncontrolled file write in core.py', "Unsafe import 'exec' in verify_implementation.py", "Unsafe import 'eval' in research_demo.py", "Unsafe import 'exec' in research_demo.py", "Unsafe import 'eval' in scaling_demo.py", "Unsafe import 'eval' in novel_research_demo.py", "Unsafe import 'exec' in novel_research_demo.py", "Unsafe import 'eval' in advanced_search_demo.py", "Unsafe import 'eval' in simple_integration_test.py", "Unsafe import 'eval' in validate_installation.py", "Unsafe import 'eval' in test_core.py", "Unsafe import 'eval' in test_generation3.py", "Unsafe import 'exec' in test_generation3.py", "Unsafe import 'eval' in core.py", "Unsafe import 'eval' in metrics.py", "Unsafe import 'eval' in monitoring.py", "Unsafe import 'exec' in monitoring.py", "Unsafe import 'eval' in optimization.py", "Unsafe import 'eval' in parallel.py", "Unsafe import 'exec' in parallel.py", "Unsafe import 'exec' in scaling.py", "Unsafe import 'subprocess' in security.py", "Unsafe import 'eval' in security.py", "Unsafe import 'exec' in security.py", "Unsafe import 'eval' in research_engine.py", "Unsafe import 'subprocess' in quality_gates.py", "Unsafe import 'os.system' in quality_gates.py", "Unsafe import 'eval' in quality_gates.py", "Unsafe import 'exec' in quality_gates.py", "Unsafe import 'eval' in security_hardening.py", "Unsafe import 'exec' in security_hardening.py"]
- validation_categories: ['secrets', 'input_validation', 'file_access', 'imports']


## Performance

✅ **Status:** PASSED (Score: 86.7%)

**Details:**
- prediction_speed: 6092.475742257859
- search_efficiency: 0.85
- memory_usage: 256
- benchmark_timestamp: 1755003672.4315336


## Documentation

❌ **Status:** FAILED (Score: 41.1%)

**Details:**
- readme_score: 25.0
- docstring_score: 16.40449438202247
- api_docs_score: 90.0
- files_analyzed: 21
- documentation_gaps: []

