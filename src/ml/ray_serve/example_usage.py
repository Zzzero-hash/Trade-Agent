"""Example usage of Ray Serve CNN+LSTM deployment.

This module demonstrates how to deploy, use, and manage CNN+LSTM models
with Ray Serve, including auto-scaling and monitoring.
"""

import ray
from ray import serve
import numpy as np
import asyncio
import time

from src.ml.ray_serve.deployment_manager import DeploymentManager
from src.ml.ray_serve.config import (
    AutoscalingConfig, 
    ResourceConfig, 
    TradingWorkloadAutoscaler
)
from src.ml.ray_serve.model_loader import RayServeModelLoader, GPUOptimizer


async def deploy_cnn_lstm_model():
    """Example of deploying a CNN+LSTM model with Ray Serve."""
    print("=== CNN+LSTM Ray Serve Deployment Example ===\n")
    
    # Initialize deployment manager
    deployment_manager = DeploymentManager()
    
    # Initialize Ray Serve deployment
    print("1. Initializing Ray Serve deployment...")
    success = await deployment_manager.initialize()
    if not success:
        print("Failed to initialize deployment")
        return
    
    print("✓ Deployment initialized successfully\n")
    
    # Check health
    print("2. Checking deployment health...")
    health = await deployment_manager.health_check()
    print(f"Health status: {health['status']}")
    print(f"Is healthy: {health['is_healthy']}\n")
    
    # Get deployment stats
    print("3. Getting deployment statistics...")
    stats = await deployment_manager.get_deployment_stats()
    print(f"Request count: {stats.get('request_count', 0)}")
    print(f"Average processing time: {stats.get('avg_processing_time_ms', 0):.2f}ms\n")
    
    # Test prediction with sample data
    print("4. Testing prediction with sample data...")
    try:
        # Create sample input data (batch_size=1, features=10, sequence_length=60)
        sample_data = np.random.rand(1, 10, 60).astype(np.float32)
        
        # Time the prediction
        start_time = time.time()
        result = await deployment_manager.predict(sample_data)
        prediction_time = (time.time() - start_time) * 1000
        
        print(f"✓ Prediction completed in {prediction_time:.2f}ms")
        print(f"Result keys: {list(result.keys()) if result else 'No result'}\n")
        
    except Exception as e:
        print(f"Prediction failed: {e}\n")
    
    # Check performance requirements
    print("5. Checking performance requirements...")
    perf_reqs = deployment_manager.check_performance_requirements()
    meets_requirement = perf_reqs["meets_100ms_requirement"]
    avg_latency = perf_reqs["avg_latency_ms"]
    
    print(f"Average latency: {avg_latency:.2f}ms")
    print(f"Meets <100ms requirement: {meets_requirement}")
    if meets_requirement:
        print("✓ Performance requirement satisfied\n")
    else:
        print("⚠ Performance requirement not met\n")
    
    # Apply market hours scaling
    print("6. Applying market hours auto-scaling configuration...")
    success = deployment_manager.apply_market_hours_scaling()
    if success:
        print("✓ Market hours scaling applied\n")
    else:
        print("⚠ Failed to apply market hours scaling\n")
    
    # Scale deployment
    print("7. Scaling deployment to 5 replicas...")
    success = deployment_manager.scale_deployment(5)
    if success:
        print("✓ Deployment scaled to 5 replicas\n")
    else:
        print("⚠ Failed to scale deployment\n")
    
    # Final health check
    print("8. Final health check...")
    health = await deployment_manager.health_check()
    print(f"Final health status: {health['status']}\n")
    
    # Shutdown deployment
    print("9. Shutting down deployment...")
    deployment_manager.shutdown()
    print("✓ Deployment shut down successfully\n")
    
    print("=== Example completed ===")


def configure_autoscaling():
    """Example of configuring auto-scaling for different scenarios."""
    print("=== Auto-scaling Configuration Examples ===\n")
    
    # Default configuration
    default_config = AutoscalingConfig()
    print("1. Default auto-scaling configuration:")
    print(f"   Min replicas: {default_config.min_replicas}")
    print(f"   Max replicas: {default_config.max_replicas}")
    print(f"   Target requests per replica: {default_config.target_num_ongoing_requests_per_replica}\n")
    
    # Market hours configuration
    market_config = TradingWorkloadAutoscaler.get_market_hours_config()
    print("2. Market hours auto-scaling configuration:")
    print(f"   Min replicas: {market_config.min_replicas}")
    print(f"   Max replicas: {market_config.max_replicas}")
    print(f"   Upscale delay: {market_config.upscale_delay_s}s")
    print(f"   Downscale delay: {market_config.downscale_delay_s}s\n")
    
    # Off hours configuration
    off_config = TradingWorkloadAutoscaler.get_off_hours_config()
    print("3. Off-hours auto-scaling configuration:")
    print(f"   Min replicas: {off_config.min_replicas}")
    print(f"   Max replicas: {off_config.max_replicas}")
    print(f"   Upscale delay: {off_config.upscale_delay_s}s")
    print(f"   Downscale delay: {off_config.downscale_delay_s}s\n")
    
    # Resource configurations
    print("4. Resource configurations:")
    for size, config in [("Small", ResourceConfig(num_cpus=1, num_gpus=0.25)),
                         ("Medium", ResourceConfig(num_cpus=2, num_gpus=0.5)),
                         ("Large", ResourceConfig(num_cpus=4, num_gpus=1.0))]:
        print(f"   {size}: {config.num_cpus} CPUs, {config.num_gpus} GPUs")


def setup_gpu_optimization():
    """Example of setting up GPU optimization."""
    print("=== GPU Optimization Setup ===\n")
    
    # Setup GPU optimizations
    print("1. Setting up GPU optimizations...")
    GPUOptimizer.setup_gpu_settings()
    print("✓ GPU optimizations configured\n")
    
    # Get GPU memory info
    print("2. Getting GPU memory information...")
    gpu_info = GPUOptimizer.get_gpu_memory_info()
    print(f"   Allocated memory: {gpu_info['allocated_mb']:.2f} MB")
    print(f"   Reserved memory: {gpu_info['reserved_mb']:.2f} MB")
    print(f"   Utilization: {gpu_info['utilization_pct']:.1f}%\n")


async def main():
    """Main example function."""
    # Configure auto-scaling
    configure_autoscaling()
    print()
    
    # Setup GPU optimization
    setup_gpu_optimization()
    print()
    
    # Deploy and test model (this would require a real Ray environment)
    # For demonstration purposes, we'll just show the code structure
    print("=== Model Deployment Example ===")
    print("Note: Actual deployment requires a running Ray cluster\n")
    
    # Show example deployment code
    print("Example deployment code:")
    print("""
    # Initialize Ray
    ray.init(address="auto")
    
    # Start Serve
    serve.start(detached=True)
    
    # Deploy model
    from src.ml.ray_serve.cnn_lstm_deployment import cnn_lstm_deployment
    serve.run(cnn_lstm_deployment, name="cnn_lstm_predictor")
    
    # Get handle and make predictions
    handle = serve.get_deployment("cnn_lstm_predictor").get_handle()
    result = await handle.remote(input_data)
    """)
    print()
    
    # Run the full deployment example (commented out for safety)
    print("To run the full deployment example, uncomment the following line:")
    print("# await deploy_cnn_lstm_model()")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())