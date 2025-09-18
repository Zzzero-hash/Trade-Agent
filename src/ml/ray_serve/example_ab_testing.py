"""Example usage of the A/B testing framework for Ray Serve deployments."""

import asyncio
import numpy as np
from datetime import datetime

from src.ml.ray_serve.ab_testing import (
    VariantConfig, 
    VariantStatus, 
    ABTestExperiment, 
    RayServeABTestManager,
    create_sample_experiment
)


async def example_basic_usage():
    """Example of basic A/B testing usage."""
    print("=== Basic A/B Testing Example ===")
    
    # Create A/B test manager
    ab_test_manager = RayServeABTestManager()
    
    # Define model variants
    variants = [
        VariantConfig(
            name="control",
            model_path="models/production_model_v1.pth",
            weight=0.5,  # 50% traffic
            metadata={"description": "Current production model"}
        ),
        VariantConfig(
            name="variant_a",
            model_path="models/experimental_model_v2.pth",
            weight=0.3,  # 30% traffic
            metadata={"description": "Experimental model with new features"}
        ),
        VariantConfig(
            name="variant_b",
            model_path="models/experimental_model_v3.pth",
            weight=0.2,  # 20% traffic
            metadata={"description": "Experimental model with different architecture"}
        )
    ]
    
    # Create experiment
    experiment = ab_test_manager.create_experiment(
        experiment_id="model_comparison_2023",
        variants=variants,
        duration_hours=48,  # Run for 48 hours
        confidence_level=0.95  # 95% confidence level for statistical tests
    )
    
    print(f"Created experiment: {experiment.experiment_id}")
    print(f"Variants: {[v.name for v in variants]}")
    print(f"Traffic split: {[f'{v.name}: {v.weight*100}%' for v in variants]}")
    
    # Simulate some requests
    print("\n--- Simulating Requests ---")
    request_ids = [f"req_{i}" for i in range(100)]
    
    for request_id in request_ids:
        # Get variant assignment for this request
        variant = ab_test_manager.get_variant_for_request(
            experiment.experiment_id, 
            request_id
        )
        
        if variant:
            print(f"Request {request_id} assigned to variant: {variant}")
            
            # Simulate model prediction (in real usage, this would be actual inference)
            latency = np.random.normal(50, 10)  # Random latency around 50ms
            confidence = np.random.uniform(0.7, 0.95)  # Random confidence score
            
            # Record metrics
            ab_test_manager.record_metrics(
                experiment_id=experiment.experiment_id,
                variant_name=variant,
                latency_ms=latency,
                processing_time_ms=latency * 0.9,  # Processing time is 90% of latency
                confidence_score=confidence
            )
    
    # Get experiment results
    print("\n--- Experiment Results ---")
    results = ab_test_manager.get_experiment_results(experiment.experiment_id)
    
    if results:
        print(f"Experiment ID: {results['experiment_id']}")
        print(f"Status: {results['status']}")
        print(f"Duration: {results['duration_hours']:.1f} hours")
        
        print("\nVariant Performance:")
        for variant_name, variant_data in results['variants'].items():
            metrics = variant_data['metrics']
            print(f"\n{variant_name}:")
            print(f"  Requests: {metrics['requests']}")
            print(f"  Error Rate: {metrics['error_rate']:.2%}")
            print(f"  Avg Latency: {metrics['avg_latency_ms']:.2f} ms")
            print(f"  Avg Confidence: {metrics['avg_confidence']:.3f}")
    
    # Get statistical summary
    print("\n--- Statistical Analysis ---")
    summary = ab_test_manager.get_statistical_summary(experiment.experiment_id)
    
    if summary:
        print(f"Total Statistical Tests: {summary['total_tests']}")
        print(f"Significant Tests: {summary['significant_tests']}")
        print(f"Confidence Level: {summary['confidence_level']:.0%}")
        
        if summary['tests']:
            print("\nTest Results:")
            for test in summary['tests']:
                significance = "SIGNIFICANT" if test['significant'] else "NOT SIGNIFICANT"
                print(f"  {test['test_name']}: {significance} (p={test['p_value']:.4f})")


async def example_sample_experiment():
    """Example using the sample experiment creation function."""
    print("\n\n=== Sample Experiment Example ===")
    
    # Create a sample experiment
    experiment = create_sample_experiment()
    
    print(f"Created sample experiment: {experiment.experiment_id}")
    print(f"Variants: {list(experiment.variants.keys())}")
    
    # Simulate a few requests
    for i in range(10):
        request_id = f"sample_req_{i}"
        variant = experiment.get_variant_for_request(request_id)
        
        if variant:
            # Simulate metrics
            latency = np.random.normal(45 + i, 5)  # Varying latency
            experiment.record_metrics(
                variant_name=variant,
                latency_ms=latency,
                processing_time_ms=latency * 0.85,
                confidence_score=0.8 + (i * 0.01)  # Increasing confidence
            )
    
    # Show results
    results = experiment.get_results()
    print(f"\nSample experiment results for {results['experiment_id']}:")
    for variant_name, variant_data in results['variants'].items():
        metrics = variant_data['metrics']
        print(f"  {variant_name}: {metrics['requests']} requests, "
              f"avg latency {metrics['avg_latency_ms']:.2f}ms")


async def example_traffic_splitting():
    """Example demonstrating traffic splitting."""
    print("\n\n=== Traffic Splitting Example ===")
    
    # Create experiment with custom traffic split
    ab_test_manager = RayServeABTestManager()
    
    # Uneven traffic split - favoring the control variant
    variants = [
        VariantConfig(name="control", model_path="models/control.pth", weight=0.7),      # 70%
        VariantConfig(name="challenger", model_path="models/challenger.pth", weight=0.2), # 20%
        VariantConfig(name="experimental", model_path="models/experimental.pth", weight=0.1) # 10%
    ]
    
    experiment = ab_test_manager.create_experiment(
        experiment_id="traffic_split_test",
        variants=variants
    )
    
    print("Traffic Split Configuration:")
    for variant in variants:
        print(f"  {variant.name}: {variant.weight*100:.0f}%")
    
    # Simulate many requests to verify traffic split
    print("\nSimulating 1000 requests...")
    variant_counts = {"control": 0, "challenger": 0, "experimental": 0}
    
    for i in range(1000):
        request_id = f"split_test_{i}"
        variant = ab_test_manager.get_variant_for_request(
            experiment.experiment_id, 
            request_id
        )
        
        if variant:
            variant_counts[variant] += 1
    
    print("\nActual Traffic Distribution:")
    for variant, count in variant_counts.items():
        percentage = (count / 1000) * 100
        expected = next(v.weight * 100 for v in variants if v.name == variant)
        print(f"  {variant}: {count}/1000 ({percentage:.1f}%) - Expected: {expected:.0f}%")


async def example_statistical_significance():
    """Example demonstrating statistical significance testing."""
    print("\n\n=== Statistical Significance Example ===")
    
    ab_test_manager = RayServeABTestManager()
    
    # Create variants with different performance characteristics
    variants = [
        VariantConfig(name="baseline", model_path="models/baseline.pth", weight=0.5),
        VariantConfig(name="improved", model_path="models/improved.pth", weight=0.5)
    ]
    
    experiment = ab_test_manager.create_experiment(
        experiment_id="significance_test",
        variants=variants
    )
    
    # Simulate data for baseline variant (slower, less confident)
    print("Generating baseline performance data...")
    for i in range(100):
        latency = np.random.normal(60, 15)  # Mean 60ms, higher variance
        confidence = np.random.uniform(0.6, 0.85)
        
        ab_test_manager.record_metrics(
            experiment_id=experiment.experiment_id,
            variant_name="baseline",
            latency_ms=latency,
            processing_time_ms=latency * 0.9,
            confidence_score=confidence
        )
    
    # Simulate data for improved variant (faster, more confident)
    print("Generating improved performance data...")
    for i in range(100):
        latency = np.random.normal(45, 10)  # Mean 45ms, lower variance
        confidence = np.random.uniform(0.75, 0.95)
        
        ab_test_manager.record_metrics(
            experiment_id=experiment.experiment_id,
            variant_name="improved",
            latency_ms=latency,
            processing_time_ms=latency * 0.85,
            confidence_score=confidence
        )
    
    # Perform statistical analysis
    print("\nPerforming statistical analysis...")
    summary = ab_test_manager.get_statistical_summary(experiment.experiment_id)
    
    if summary and summary['tests']:
        print(f"\nStatistical Test Results (Confidence Level: {summary['confidence_level']:.0%}):")
        for test in summary['tests']:
            significance = "SIGNIFICANT" if test['significant'] else "NOT SIGNIFICANT"
            test_type = "Latency Comparison" if "latency" in test['test_name'] else "Error Rate Comparison"
            print(f"\n {test_type}:")
            print(f"    Result: {significance}")
            print(f"    P-value: {test['p_value']:.6f}")
            if test['effect_size'] is not None:
                print(f"    Effect Size: {test['effect_size']:.4f}")
            print(f"    Description: {test['description']}")
    else:
        print("Not enough data for statistical testing or no tests performed.")


async def main():
    """Run all examples."""
    print("A/B Testing Framework Examples")
    print("=" * 50)
    
    # Run examples
    await example_basic_usage()
    await example_sample_experiment()
    await example_traffic_splitting()
    await example_statistical_significance()
    
    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())