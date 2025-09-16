# Feature Extraction Performance Testing Framework

## Overview

This module provides a comprehensive performance testing framework specifically designed to validate the <100ms feature extraction requirement for the AI Trading Platform's CNN+LSTM feature extraction system.

The framework includes:

1. **Performance Testing** - Single extraction latency validation
2. **Load Testing** - Concurrent user and constant load scenarios
3. **Stress Testing** - Extreme load and resource exhaustion scenarios
4. **Metrics Collection** - Comprehensive performance metrics
5. **Reporting** - JSON and HTML report generation
6. **Integration** - Seamless integration with existing monitoring systems

## Components

### 1. Framework (`framework.py`)
Core performance testing framework with:
- `FeatureExtractionPerformanceTester` - Main testing class
- `PerformanceTestConfig` - Configuration for performance tests
- `PerformanceTestResult` - Results container

### 2. Load Testing (`load_testing.py`)
Load testing capabilities with:
- `LoadTester` - Load testing class
- `LoadTestConfig` - Configuration for load tests
- `LoadTestResult` - Results container

### 3. Stress Testing (`stress_testing.py`)
Stress testing capabilities with:
- `StressTester` - Stress testing class
- `StressTestConfig` - Configuration for stress tests
- `StressTestResult` - Results container

### 4. Metrics (`metrics.py`)
Metrics collection and analysis with:
- `PerformanceMetricsCollector` - Metrics collection from test results
- `PerformanceReport` - Standardized performance report
- `LatencyStats` - Latency statistics container

### 5. Reporting (`reporting.py`)
Performance reporting capabilities with:
- `JSONReportGenerator` - JSON report generation
- `HTMLReportGenerator` - HTML report generation
- `generate_comparison_report` - Multi-format report generation

### 6. Integration (`integration.py`)
Integration with existing monitoring systems with:
- `PerformanceTestingIntegration` - Integration class
- `create_performance_test_suite` - Test suite creation

## Usage Examples

### Basic Performance Test
```python
from src.ml.feature_extraction.performance_testing.framework import (
    FeatureExtractionPerformanceTester,
    PerformanceTestConfig
)

# Configure test
config = PerformanceTestConfig(
    num_iterations=1000,
    warmup_iterations=100,
    target_latency_ms=100.0
)

# Create tester and run test
tester = FeatureExtractionPerformanceTester(config)
result = tester.run_performance_test(extractor)

# Check results
if result.meets_performance_requirements():
    print("Performance requirements met!")
else:
    print("Performance requirements not met!")
```

### Load Testing
```python
from src.ml.feature_extraction.performance_testing.load_testing import (
    LoadTester,
    LoadTestConfig
)

# Configure load test
config = LoadTestConfig(
    concurrent_users=10,
    requests_per_user=100,
    target_response_time_ms=100.0,
    target_throughput_rps=50.0
)

# Create load tester and run test
load_tester = LoadTester(config)
result = load_tester.run_concurrent_users_test(extractor)

# Check throughput
throughput = result.get_throughput_rps()
print(f"Throughput: {throughput:.2f} RPS")
```

### Stress Testing
```python
from src.ml.feature_extraction.performance_testing.stress_testing import (
    StressTester,
    StressTestConfig
)

# Configure stress test
config = StressTestConfig(
    max_concurrent_users=50,
    max_requests_per_user=200,
    test_duration_seconds=300
)

# Create stress tester and run test
stress_tester = StressTester(config)
result = stress_tester.run_stress_test(extractor)

# Check resource usage
print(f"Peak memory: {result.peak_memory_mb:.2f}MB")
```

## Integration with Monitoring Systems

The framework integrates with existing monitoring systems:

```python
from src.ml.feature_extraction.performance_testing.integration import PerformanceTestingIntegration

# Create integration
integration = PerformanceTestingIntegration()

# Integrate test results
integration.integrate_performance_test_result(result)

# Validate requirements
validation = integration.validate_100ms_requirement()
print(f"Meets <100ms requirement: {validation['meets_100ms_requirement']}")

# Check alerts
alerts = integration.check_performance_alerts()
print(f"Alert count: {alerts['alert_count']}")
```

## Performance Requirements Validation

The framework specifically validates the <100ms feature extraction requirement:

1. **Single Extraction**: <100ms P95 latency
2. **Cached Extraction**: <50ms P95 latency
3. **Fallback Extraction**: <20ms P95 latency
4. **Concurrent Load**: <100ms P95 latency under load
5. **Stress Conditions**: Graceful degradation under extreme load

## Running Tests

To run the performance tests:

```bash
# Run all performance tests
pytest tests/test_feature_extraction_performance.py -v

# Run <100ms validation tests
pytest tests/test_feature_extraction_100ms_validation.py -v

# Run framework validation tests
pytest tests/test_performance_testing_framework_validation.py -v
```

## Demo

A demo script is available to show how to use the framework:

```bash
python examples/performance_testing_demo.py
```

This will generate JSON and HTML reports in the current directory.