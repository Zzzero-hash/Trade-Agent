"""Model Serving Demo

This script demonstrates how to use the model serving infrastructure
including loading models, making predictions, and running A/B tests.

Requirements: 6.2, 11.1
"""

import asyncio
import numpy as np
import requests
import json
import time
from typing import Dict, Any, List
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8080/api/v1"
DEMO_MODEL_CONFIG = {
    "input_dim": 50,
    "sequence_length": 60,
    "prediction_horizon": 10,
    "num_classes": 3,
    "regression_targets": 1,
    "feature_fusion_dim": 256,
    "device": "cpu"
}


def check_api_health() -> bool:
    """Check if the API is running and healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"API health check failed: {e}")
        return False


def get_model_types() -> Dict[str, Any]:
    """Get available model types"""
    try:
        response = requests.get(f"{API_BASE_URL}/models/types")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Failed to get model types: {e}")
        return {}


def load_demo_model() -> bool:
    """Load a demo model (this would normally load from a file)"""
    try:
        # In a real scenario, you would have actual model files
        # For demo purposes, we'll try to load a model
        payload = {
            "model_type": "cnn_lstm_hybrid",
            "version": "demo_v1.0",
            "file_path": "/path/to/demo/model.pth",  # This would be a real path
            "config": DEMO_MODEL_CONFIG
        }
        
        response = requests.post(f"{API_BASE_URL}/models/load", json=payload)
        
        if response.status_code == 200:
            print("✓ Demo model loaded successfully")
            return True
        else:
            print(f"✗ Failed to load demo model: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Error loading demo model: {e}")
        return False


def make_single_prediction() -> Dict[str, Any]:
    """Make a single prediction request"""
    try:
        # Generate sample data
        sample_data = np.random.randn(1, 50).tolist()  # 1 sample, 50 features
        
        payload = {
            "model_type": "cnn_lstm_hybrid",
            "model_version": "demo_v1.0",
            "data": sample_data,
            "return_uncertainty": True,
            "use_ensemble": True
        }
        
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/predict", json=payload)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Single prediction successful (latency: {latency_ms:.2f}ms)")
            print(f"  Request ID: {result.get('request_id', 'N/A')}")
            print(f"  Model: {result.get('model_type', 'N/A')}:{result.get('model_version', 'N/A')}")
            print(f"  Processing time: {result.get('processing_time_ms', 'N/A')}ms")
            return result
        else:
            print(f"✗ Single prediction failed: {response.status_code}")
            print(f"Response: {response.text}")
            return {}
            
    except Exception as e:
        print(f"✗ Error making single prediction: {e}")
        return {}


def make_batch_predictions() -> Dict[str, Any]:
    """Make batch prediction requests"""
    try:
        # Generate multiple sample requests
        requests_data = []
        for i in range(5):
            sample_data = np.random.randn(1, 50).tolist()
            requests_data.append({
                "model_type": "cnn_lstm_hybrid",
                "model_version": "demo_v1.0",
                "data": sample_data,
                "return_uncertainty": False,
                "use_ensemble": True
            })
        
        payload = {
            "requests": requests_data,
            "priority": 1
        }
        
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/predict/batch", json=payload)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Batch prediction successful (latency: {latency_ms:.2f}ms)")
            print(f"  Batch ID: {result.get('batch_id', 'N/A')}")
            print(f"  Total requests: {result.get('total_requests', 'N/A')}")
            print(f"  Successful: {result.get('successful_predictions', 'N/A')}")
            print(f"  Failed: {result.get('failed_predictions', 'N/A')}")
            print(f"  Total processing time: {result.get('total_processing_time_ms', 'N/A')}ms")
            return result
        else:
            print(f"✗ Batch prediction failed: {response.status_code}")
            print(f"Response: {response.text}")
            return {}
            
    except Exception as e:
        print(f"✗ Error making batch predictions: {e}")
        return {}


def create_ab_test_experiment() -> bool:
    """Create an A/B test experiment"""
    try:
        experiment_config = {
            "experiment_id": "demo_experiment_" + str(int(time.time())),
            "model_variants": {
                "control": {"model_type": "cnn_lstm_hybrid", "version": "demo_v1.0"},
                "treatment": {"model_type": "cnn_lstm_hybrid", "version": "demo_v1.0"}  # Same model for demo
            },
            "traffic_split": {
                "control": 0.5,
                "treatment": 0.5
            },
            "duration_hours": 1  # Short duration for demo
        }
        
        response = requests.post(f"{API_BASE_URL}/experiments/create", json=experiment_config)
        
        if response.status_code == 201:
            result = response.json()
            print(f"✓ A/B test experiment created successfully")
            print(f"  Experiment ID: {experiment_config['experiment_id']}")
            print(f"  Duration: {experiment_config['duration_hours']} hours")
            return True
        else:
            print(f"✗ Failed to create A/B test: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Error creating A/B test: {e}")
        return False


def list_experiments() -> List[Dict[str, Any]]:
    """List all A/B test experiments"""
    try:
        response = requests.get(f"{API_BASE_URL}/experiments")
        
        if response.status_code == 200:
            result = response.json()
            experiments = result.get("experiments", {})
            
            print(f"✓ Found {len(experiments)} experiments")
            for exp_id, exp_data in experiments.items():
                print(f"  - {exp_id}: {exp_data.get('status', 'unknown')} "
                      f"({exp_data.get('total_requests', 0)} requests)")
            
            return list(experiments.values())
        else:
            print(f"✗ Failed to list experiments: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"✗ Error listing experiments: {e}")
        return []


def get_cache_stats() -> Dict[str, Any]:
    """Get model cache statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/models/cache/stats")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Cache statistics retrieved")
            print(f"  Total models: {result.get('total_models', 0)}")
            print(f"  Max capacity: {result.get('max_capacity', 0)}")
            print(f"  Total usage: {result.get('total_usage', 0)}")
            return result
        else:
            print(f"✗ Failed to get cache stats: {response.status_code}")
            return {}
            
    except Exception as e:
        print(f"✗ Error getting cache stats: {e}")
        return {}


def get_metrics() -> Dict[str, Any]:
    """Get serving metrics"""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Metrics retrieved")
            print(f"  Timestamp: {result.get('timestamp', 'N/A')}")
            return result
        else:
            print(f"✗ Failed to get metrics: {response.status_code}")
            return {}
            
    except Exception as e:
        print(f"✗ Error getting metrics: {e}")
        return {}


def performance_test(num_requests: int = 10) -> Dict[str, float]:
    """Run a simple performance test"""
    print(f"\n🚀 Running performance test with {num_requests} requests...")
    
    latencies = []
    successful_requests = 0
    
    for i in range(num_requests):
        sample_data = np.random.randn(1, 50).tolist()
        payload = {
            "model_type": "cnn_lstm_hybrid",
            "model_version": "demo_v1.0",
            "data": sample_data,
            "return_uncertainty": False,
            "use_ensemble": False
        }
        
        start_time = time.time()
        try:
            response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            if response.status_code == 200:
                successful_requests += 1
                
        except Exception as e:
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            print(f"  Request {i+1} failed: {e}")
    
    # Calculate statistics
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        success_rate = successful_requests / num_requests
        
        print(f"✓ Performance test completed")
        print(f"  Successful requests: {successful_requests}/{num_requests} ({success_rate:.1%})")
        print(f"  Average latency: {avg_latency:.2f}ms")
        print(f"  Min latency: {min_latency:.2f}ms")
        print(f"  Max latency: {max_latency:.2f}ms")
        
        return {
            "avg_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "success_rate": success_rate,
            "total_requests": num_requests
        }
    else:
        print("✗ No latency data collected")
        return {}


def main():
    """Main demo function"""
    print("🤖 AI Trading Platform - Model Serving Demo")
    print("=" * 50)
    
    # Check API health
    print("\n1. Checking API health...")
    if not check_api_health():
        print("❌ API is not running. Please start the model serving API first.")
        print("   Run: python -m src.api.app")
        return
    
    print("✅ API is healthy and running")
    
    # Get model types
    print("\n2. Getting available model types...")
    model_types = get_model_types()
    if model_types:
        print("✅ Available model types:")
        for model_type, info in model_types.get("model_types", {}).items():
            print(f"   - {model_type}: {info.get('name', 'N/A')}")
    
    # Note: In a real demo, you would load actual models
    # For this demo, we'll skip model loading since we don't have actual model files
    print("\n3. Model loading...")
    print("ℹ️  Skipping model loading (no actual model files in demo)")
    print("   In production, you would load models using the /models/load endpoint")
    
    # Make predictions (these will likely fail without loaded models, but demonstrate the API)
    print("\n4. Making single prediction...")
    prediction_result = make_single_prediction()
    
    print("\n5. Making batch predictions...")
    batch_result = make_batch_predictions()
    
    # A/B testing demo
    print("\n6. A/B testing demo...")
    ab_test_created = create_ab_test_experiment()
    
    if ab_test_created:
        print("\n7. Listing experiments...")
        experiments = list_experiments()
    
    # Cache and metrics
    print("\n8. Getting cache statistics...")
    cache_stats = get_cache_stats()
    
    print("\n9. Getting metrics...")
    metrics = get_metrics()
    
    # Performance test
    print("\n10. Performance test...")
    perf_results = performance_test(num_requests=5)
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed!")
    print("\nNote: Some operations may fail in this demo environment")
    print("because actual trained models are not loaded. In a production")
    print("environment with loaded models, all operations would succeed.")
    
    print("\n📚 Next steps:")
    print("1. Load actual trained models using the /models/load endpoint")
    print("2. Set up Redis for caching and A/B testing persistence")
    print("3. Configure monitoring and alerting")
    print("4. Set up production deployment with proper security")


if __name__ == "__main__":
    main()