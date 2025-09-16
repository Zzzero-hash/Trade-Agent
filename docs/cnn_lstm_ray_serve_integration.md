# Integration Approach for CNN+LSTM Models with Ray Serve

## 1. Overview

This document outlines the integration approach for incorporating existing CNN+LSTM models into the Ray Serve deployment architecture. The approach ensures seamless integration with the current model loading pipeline, maintains compatibility with existing APIs, and leverages Ray Serve's distributed capabilities for improved performance.

## 2. Current CNN+LSTM Model Architecture

### 2.1 Model Components

The existing CNN+LSTM hybrid model consists of:

1. **CNN Feature Extractor**: Extracts spatial patterns from multi-dimensional market data
2. **LSTM Temporal Processor**: Captures long-term dependencies in price movements
3. **Feature Fusion Module**: Combines CNN and LSTM features using cross-attention
4. **Multi-task Learning Heads**: Simultaneously optimize classification and regression objectives
5. **Ensemble Components**: Learnable ensemble weights with dynamic adjustment
6. **Uncertainty Quantification**: Monte Carlo dropout for prediction confidence intervals

### 2.2 Model Loading Pipeline

The current model loading pipeline includes:
- Model registry for version management
- Configuration loading and validation
- Model instantiation and weight loading
- Device placement (CPU/GPU)
- Model validation and warmup

## 3. Integration Strategy

### 3.1 Model Compatibility Layer

A compatibility layer will be implemented to bridge the existing model architecture with Ray Serve:

```python
# model_compatibility_layer.py
import torch
import numpy as np
from typing import Dict, Any, Optional
from src.ml.hybrid_model import CNNLSTMHybridModel
from src.ml.feature_extraction.cnn_lstm_extractor import CNNLSTMExtractor

class RayServeModelAdapter:
    """Adapter to integrate existing CNN+LSTM models with Ray Serve."""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize the adapter with a pre-trained model.
        
        Args:
            model_path: Path to the saved model
            device: Device to run inference on
        """
        self.device = device
        self.model = self._load_model(model_path)
        self.extractor = CNNLSTMExtractor(self.model, device)
        
    def _load_model(self, model_path: str) -> CNNLSTMHybridModel:
        """
        Load the CNN+LSTM model from the specified path.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded CNN+LSTM hybrid model
        """
        try:
            # Load model configuration
            config_path = model_path.replace(".pth", "_config.json")
            
            # Load the model
            model = CNNLSTMHybridModel.load_from_path(model_path)
            model.to(self.device)
            model.eval()
            
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    
    def predict(self, input_data: np.ndarray, 
                return_uncertainty: bool = True,
                use_ensemble: bool = True) -> Dict[str, Any]:
        """
        Make predictions using the CNN+LSTM model.
        
        Args:
            input_data: Input data array
            return_uncertainty: Whether to return uncertainty estimates
            use_ensemble: Whether to use ensemble predictions
            
        Returns:
            Dictionary containing predictions and metadata
        """
        # Validate input
        self._validate_input(input_data)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(input_data).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model.forward(
                input_tensor,
                return_features=True,
                use_ensemble=use_ensemble
            )
        
        # Process outputs
        results = self._process_outputs(outputs, return_uncertainty)
        
        return results
    
    def _validate_input(self, data: np.ndarray) -> None:
        """Validate input data format and dimensions."""
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a numpy array")
        
        if data.ndim not in [2, 3]:
            raise ValueError(f"Expected 2D or 3D input array, got {data.ndim}D")
        
        # Add model-specific validation
        if data.ndim == 3:
            expected_channels = self.model.config.input_dim
            if data.shape[1] != expected_channels:
                raise ValueError(
                    f"Expected {expected_channels} channels, got {data.shape[1]}"
                )
    
    def _process_outputs(self, outputs: Dict[str, torch.Tensor], 
                        return_uncertainty: bool) -> Dict[str, Any]:
        """
        Process model outputs into standardized format.
        
        Args:
            outputs: Raw model outputs
            return_uncertainty: Whether to include uncertainty estimates
            
        Returns:
            Processed outputs dictionary
        """
        results = {
            'classification_probs': outputs['classification_probs'].cpu().numpy(),
            'regression_pred': outputs['regression_mean'].cpu().numpy(),
            'ensemble_classification': outputs['ensemble_classification'].cpu().numpy(),
            'ensemble_regression': outputs['ensemble_regression'].cpu().numpy()
        }
        
        if return_uncertainty:
            results['regression_uncertainty'] = outputs['regression_uncertainty'].cpu().numpy()
            results['ensemble_weights'] = (
                outputs['ensemble_weights'].cpu().numpy() 
                if outputs['ensemble_weights'] is not None else None
            )
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for monitoring and debugging."""
        return {
            'model_type': 'CNNLSTMHybridModel',
            'input_dim': self.model.config.input_dim,
            'sequence_length': self.model.config.sequence_length,
            'num_classes': self.model.config.num_classes,
            'device': self.device,
            'is_trained': self.model.is_trained
        }
```

### 3.2 Model Registry Integration

The integration will maintain compatibility with the existing model registry:

```python
# model_registry_integration.py
import os
from typing import Optional
from src.config.settings import get_settings

class ModelRegistryAdapter:
    """Adapter for integrating with the existing model registry."""
    
    def __init__(self):
        """Initialize the model registry adapter."""
        self.settings = get_settings()
        self.registry_path = self.settings.ml.model_registry_path
    
    def get_model_path(self, model_name: str, version: str = "latest") -> str:
        """
        Get the path to a specific model version.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            
        Returns:
            Path to the model file
        """
        if version == "latest":
            version = self._get_latest_version(model_name)
        
        return os.path.join(self.registry_path, model_name, version, "model.pth")
    
    def _get_latest_version(self, model_name: str) -> str:
        """
        Get the latest version of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Latest version string
        """
        model_dir = os.path.join(self.registry_path, model_name)
        if not os.path.exists(model_dir):
            raise ValueError(f"Model {model_name} not found in registry")
        
        versions = [
            d for d in os.listdir(model_dir) 
            if os.path.isdir(os.path.join(model_dir, d))
        ]
        
        if not versions:
            raise ValueError(f"No versions found for model {model_name}")
        
        # Sort versions and return the latest
        versions.sort(reverse=True)
        return versions[0]
    
    def validate_model(self, model_path: str) -> bool:
        """
        Validate that a model file exists and is accessible.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if model is valid, False otherwise
        """
        return os.path.exists(model_path) and os.access(model_path, os.R_OK)
```

## 4. API Integration

### 4.1 Request/Response Format Compatibility

The Ray Serve deployment will maintain compatibility with existing API formats:

```python
# api_compatibility.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    """Compatible request model for predictions."""
    model_type: str = "cnn_lstm_hybrid"
    model_version: Optional[str] = "latest"
    data: List[List[float]]
    batch_size: Optional[int] = 32
    return_uncertainty: bool = True
    use_ensemble: bool = True

class PredictionResponse(BaseModel):
    """Compatible response model for predictions."""
    request_id: str
    model_type: str
    model_version: str
    predictions: Dict[str, Any]
    uncertainty: Optional[Dict[str, Any]] = None
    confidence_scores: Optional[List[float]] = None
    processing_time_ms: float
    timestamp: str
    ab_test_group: Optional[str] = None

class BatchPredictionRequest(BaseModel):
    """Compatible request model for batch predictions."""
    requests: List[PredictionRequest]
    priority: int = 1

class BatchPredictionResponse(BaseModel):
    """Compatible response model for batch predictions."""
    batch_id: str
    total_requests: int
    successful_predictions: int
    failed_predictions: int
    results: List[PredictionResponse]
    total_processing_time_ms: float
    timestamp: str
```

### 4.2 Gateway Integration

The integration with the existing API gateway will be maintained:

```python
# gateway_integration.py
from fastapi import APIRouter, HTTPException, Depends
import ray
from ray import serve
import numpy as np
from typing import Dict, Any

# Initialize Ray Serve client
ray.init(ignore_reinit_error=True)
serve.start(detached=True)

# Get reference to the deployed model
cnn_lstm_handle = serve.get_deployment("cnn_lstm_predictor").get_handle()

class GatewayAdapter:
    """Adapter for integrating Ray Serve with existing API gateway."""
    
    async def forward_prediction_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward a prediction request to the Ray Serve deployment.
        
        Args:
            request_data: Prediction request data
            
        Returns:
            Prediction response from Ray Serve
        """
        try:
            # Convert data to numpy array
            input_data = np.array(request_data["data"], dtype=np.float32)
            
            # Call Ray Serve deployment
            result = await cnn_lstm_handle.remote(
                input_data,
                request_data.get("return_uncertainty", True),
                request_data.get("use_ensemble", True)
            )
            
            return result
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def forward_batch_request(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Forward a batch prediction request to the Ray Serve deployment.
        
        Args:
            batch_data: List of prediction requests
            
        Returns:
            List of prediction responses
        """
        try:
            # Group requests by model configuration for batch optimization
            grouped_requests = self._group_requests(batch_data)
            results = []
            
            # Process each group
            for group in grouped_requests:
                # Convert to batch input
                batch_input = np.array([req["data"] for req in group], dtype=np.float32)
                
                # Call Ray Serve deployment with batch
                batch_results = await cnn_lstm_handle.batch_predict.remote(batch_input)
                
                # Split results back to individual responses
                for i, result in enumerate(batch_results):
                    results.append({
                        "request_id": group[i].get("request_id", f"batch_{i}"),
                        "predictions": result,
                        "processing_time_ms": result.get("processing_time_ms", 0)
                    })
            
            return results
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    def _group_requests(self, requests: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group requests by model configuration for batch optimization.
        
        Args:
            requests: List of prediction requests
            
        Returns:
            List of grouped requests
        """
        # Group by model configuration (simplified implementation)
        groups = {}
        for req in requests:
            key = (
                req.get("model_type", "cnn_lstm_hybrid"),
                req.get("model_version", "latest"),
                req.get("return_uncertainty", True),
                req.get("use_ensemble", True)
            )
            
            if key not in groups:
                groups[key] = []
            groups[key].append(req)
        
        return list(groups.values())
```

## 5. Feature Extraction Integration

### 5.1 Cached Feature Extraction

The integration will leverage existing feature extraction components:

```python
# feature_extraction_integration.py
from src.ml.feature_extraction import (
    FeatureExtractorFactory,
    FeatureExtractionConfig,
    CachedFeatureExtractor,
    FallbackFeatureExtractor
)
from src.ml.feature_extraction.cnn_lstm_extractor import CNNLSTMExtractor
import numpy as np

class IntegratedFeatureExtractor:
    """Integrated feature extractor combining CNN+LSTM with caching."""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """Initialize the integrated feature extractor."""
        # Create configuration
        config = FeatureExtractionConfig(
            hybrid_model_path=model_path,
            fused_feature_dim=256,
            enable_caching=True,
            cache_size=1000,
            cache_ttl_seconds=3600,
            enable_fallback=True,
            fallback_feature_dim=15
        )
        
        # Create feature extractor using factory
        self.extractor = FeatureExtractorFactory.create_extractor(config)
        
        # Set device
        self.extractor.model.to(device)
    
    def extract_features(self, market_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract features from market data.
        
        Args:
            market_data: Market data array
            
        Returns:
            Dictionary of extracted features
        """
        return self.extractor.extract_features(market_data)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if hasattr(self.extractor, 'cache_hits'):
            return {
                'cache_hits': getattr(self.extractor, 'cache_hits', 0),
                'cache_misses': getattr(self.extractor, 'cache_misses', 0),
                'cache_size': getattr(self.extractor, 'cache_size', 0)
            }
        return {}
```

## 6. Monitoring and Observability Integration

### 6.1 Metrics Integration

The integration will maintain compatibility with existing monitoring systems:

```python
# monitoring_integration.py
from src.utils.monitoring import get_metrics_collector
from prometheus_client import Counter, Histogram, Gauge
import time

class MetricsAdapter:
    """Adapter for integrating Ray Serve metrics with existing monitoring."""
    
    def __init__(self):
        """Initialize the metrics adapter."""
        self.metrics = get_metrics_collector()
        
        # Ray Serve specific metrics
        self.prediction_requests = Counter(
            'ray_serve_cnn_lstm_requests_total',
            'Total CNN+LSTM prediction requests via Ray Serve',
            ['model_version']
        )
        
        self.prediction_latency = Histogram(
            'ray_serve_cnn_lstm_latency_seconds',
            'CNN+LSTM prediction latency via Ray Serve',
            ['model_version'],
            buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
        )
        
        self.prediction_errors = Counter(
            'ray_serve_cnn_lstm_errors_total',
            'Total CNN+LSTM prediction errors via Ray Serve',
            ['model_version', 'error_type']
        )
        
        self.replica_count = Gauge(
            'ray_serve_cnn_lstm_replicas',
            'Number of active CNN+LSTM replicas'
        )
    
    def record_prediction(self, model_version: str, latency: float):
        """
        Record a successful prediction.
        
        Args:
            model_version: Model version used
            latency: Prediction latency in seconds
        """
        self.prediction_requests.labels(model_version=model_version).inc()
        self.prediction_latency.labels(model_version=model_version).observe(latency)
        self.metrics.increment_counter("model_predictions_total", {"model_type": "cnn_lstm"})
        self.metrics.record_histogram("prediction_latency_ms", latency * 1000)
    
    def record_error(self, model_version: str, error_type: str):
        """
        Record a prediction error.
        
        Args:
            model_version: Model version used
            error_type: Type of error
        """
        self.prediction_errors.labels(
            model_version=model_version, 
            error_type=error_type
        ).inc()
        self.metrics.increment_counter(
            "model_prediction_errors_total", 
            {"model_type": "cnn_lstm", "error_type": error_type}
        )
    
    def update_replica_count(self, count: int):
        """
        Update the replica count metric.
        
        Args:
            count: Number of active replicas
        """
        self.replica_count.set(count)
```

## 7. Configuration Integration

### 7.1 Settings Integration

The integration will maintain compatibility with existing configuration systems:

```python
# config_integration.py
from src.config.settings import get_settings
from ray.serve.config import AutoscalingConfig
import os

class ConfigurationAdapter:
    """Adapter for integrating Ray Serve configuration with existing settings."""
    
    def __init__(self):
        """Initialize the configuration adapter."""
        self.settings = get_settings()
    
    def get_autoscaling_config(self) -> AutoscalingConfig:
        """
        Get autoscaling configuration based on environment settings.
        
        Returns:
            Ray Serve autoscaling configuration
        """
        # Determine if we're in market hours or off hours
        is_market_hours = self._is_market_hours()
        
        if is_market_hours:
            return AutoscalingConfig(
                min_replicas=self.settings.ray.get("min_replicas_market", 5),
                max_replicas=self.settings.ray.get("max_replicas_market", 30),
                target_num_ongoing_requests_per_replica=3,
                upscale_delay_s=15,
                downscale_delay_s=120,
                upscale_smoothing_factor=1.5,
                downscale_smoothing_factor=0.3
            )
        else:
            return AutoscalingConfig(
                min_replicas=self.settings.ray.get("min_replicas_off", 2),
                max_replicas=self.settings.ray.get("max_replicas_off", 10),
                target_num_ongoing_requests_per_replica=10,
                upscale_delay_s=60,
                downscale_delay_s=300,
                upscale_smoothing_factor=1.0,
                downscale_smoothing_factor=0.5
            )
    
    def get_resource_config(self) -> Dict[str, Any]:
        """
        Get resource configuration based on environment settings.
        
        Returns:
            Resource configuration dictionary
        """
        return {
            "num_cpus": self.settings.ray.get("num_cpus", 2),
            "num_gpus": self.settings.ray.get("num_gpus", 0.5 if self._has_gpu() else 0),
            "memory": self.settings.ray.get("memory", 2 * 1024 * 1024 * 1024),
            "object_store_memory": self.settings.ray.get("object_store_memory", 1 * 1024 * 1024 * 1024)
        }
    
    def _is_market_hours(self) -> bool:
        """
        Determine if current time is within market hours.
        
        Returns:
            True if within market hours, False otherwise
        """
        import datetime
        # Simplified market hours check (9:30 AM - 4:00 PM EST)
        current_time = datetime.datetime.now()
        current_hour = current_time.hour
        current_weekday = current_time.weekday()
        
        # Monday-Friday, 9:30 AM - 4:00 PM
        return (0 <= current_weekday <= 4 and 
                9 <= current_hour <= 16)
    
    def _has_gpu(self) -> bool:
        """
        Check if GPU is available.
        
        Returns:
            True if GPU is available, False otherwise
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
```

## 8. Deployment Integration

### 8.1 Deployment Lifecycle Management

The integration will provide lifecycle management for deployments:

```python
# deployment_lifecycle.py
import ray
from ray import serve
from typing import Optional
import asyncio

class DeploymentLifecycleManager:
    """Manager for CNN+LSTM deployment lifecycle."""
    
    def __init__(self):
        """Initialize the deployment lifecycle manager."""
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Start Serve if not already started
        try:
            serve.start(detached=True)
        except RuntimeError:
            # Serve already started
            pass
    
    async def deploy_model(self, model_name: str, model_path: str, 
                          version: str = "latest") -> bool:
        """
        Deploy a CNN+LSTM model.
        
        Args:
            model_name: Name of the model
            model_path: Path to the model file
            version: Version of the model
            
        Returns:
            True if deployment successful, False otherwise
        """
        try:
            # Import deployment definition
            from ray_serve_cnn_lstm_deployment import CNNLSTMPredictor
            
            # Create deployment
            deployment = CNNLSTMPredictor.bind(model_path=model_path)
            
            # Deploy
            serve.run(deployment, name=f"{model_name}_{version}")
            
            return True
            
        except Exception as e:
            print(f"Failed to deploy model {model_name}: {e}")
            return False
    
    async def undeploy_model(self, model_name: str, version: str = "latest") -> bool:
        """
        Undeploy a CNN+LSTM model.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            
        Returns:
            True if undeployment successful, False otherwise
        """
        try:
            serve.delete(f"{model_name}_{version}")
            return True
        except Exception as e:
            print(f"Failed to undeploy model {model_name}: {e}")
            return False
    
    def get_deployment_status(self, model_name: str, version: str = "latest") -> Dict[str, Any]:
        """
        Get deployment status.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            
        Returns:
            Deployment status dictionary
        """
        try:
            deployment_name = f"{model_name}_{version}"
            deployment = serve.get_deployment(deployment_name)
            
            return {
                "status": "deployed",
                "name": deployment_name,
                "config": deployment.config
            }
        except Exception as e:
            return {
                "status": "not_deployed",
                "error": str(e)
            }
```

## 9. Testing Integration

### 9.1 Integration Testing Framework

The integration will include testing capabilities:

```python
# integration_testing.py
import unittest
import numpy as np
from unittest.mock import Mock, patch

class CNNLSTMIntegrationTests(unittest.TestCase):
    """Integration tests for CNN+LSTM Ray Serve integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_input = np.random.rand(1, 50, 60).astype(np.float32)
        self.batch_input = np.random.rand(5, 50, 60).astype(np.float32)
    
    @patch('ray_serve_cnn_lstm_deployment.CNNLSTMPredictor')
    def test_single_prediction(self, mock_predictor):
        """Test single prediction integration."""
        # Mock the predictor
        mock_instance = Mock()
        mock_predictor.return_value = mock_instance
        mock_instance.predict.return_value = {
            'classification_probs': np.array([[0.5, 0.3, 0.2]]),
            'regression_pred': np.array([[100.5]]),
            'processing_time_ms': 45.2
        }
        
        # Test the integration
        from ray_serve_cnn_lstm_deployment import CNNLSTMPredictor
        predictor = CNNLSTMPredictor()
        result = predictor.predict(self.sample_input)
        
        # Verify results
        self.assertIn('classification_probs', result)
        self.assertIn('regression_pred', result)
        self.assertGreater(result['processing_time_ms'], 0)
    
    @patch('ray_serve_cnn_lstm_deployment.CNNLSTMPredictor')
    def test_batch_prediction(self, mock_predictor):
        """Test batch prediction integration."""
        # Mock the predictor
        mock_instance = Mock()
        mock_predictor.return_value = mock_instance
        mock_instance.batch_predict.return_value = [
            {
                'classification_probs': np.array([[0.5, 0.3, 0.2]]),
                'regression_pred': np.array([[100.5]]),
                'processing_time_ms': 45.2
            } for _ in range(5)
        ]
        
        # Test the integration
        from ray_serve_cnn_lstm_deployment import CNNLSTMPredictor
        predictor = CNNLSTMPredictor()
        results = predictor.batch_predict([self.sample_input] * 5)
        
        # Verify results
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIn('classification_probs', result)
            self.assertIn('regression_pred', result)
    
    def test_input_validation(self):
        """Test input validation."""
        from ray_serve_cnn_lstm_deployment import CNNLSTMPredictor
        predictor = CNNLSTMPredictor()
        
        # Test invalid input types
        with self.assertRaises(ValueError):
            predictor._validate_input("invalid_input")
        
        # Test invalid dimensions
        with self.assertRaises(ValueError):
            predictor._validate_input(np.array([1, 2, 3]))  # 1D array

if __name__ == '__main__':
    unittest.main()
```

## 10. Migration Strategy

### 10.1 Gradual Migration Approach

The integration will support a gradual migration from existing serving to Ray Serve:

```python
# migration_strategy.py
from typing import Dict, Any, Optional
import asyncio

class MigrationStrategy:
    """Strategy for migrating from existing serving to Ray Serve."""
    
    def __init__(self, traffic_split: float = 0.0):
        """
        Initialize migration strategy.
        
        Args:
            traffic_split: Percentage of traffic to route to Ray Serve (0.0-1.0)
        """
        self.traffic_split = traffic_split
        self.ray_serve_enabled = traffic_split > 0.0
    
    async def route_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route request based on migration strategy.
        
        Args:
            request_data: Request data
            
        Returns:
            Response from appropriate serving system
        """
        import random
        
        # Determine routing
        if not self.ray_serve_enabled or random.random() > self.traffic_split:
            # Route to existing FastAPI serving
            return await self._route_to_fastapi(request_data)
        else:
            # Route to Ray Serve
            return await self._route_to_ray_serve(request_data)
    
    async def _route_to_fastapi(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route request to existing FastAPI serving.
        
        Args:
            request_data: Request data
            
        Returns:
            Response from FastAPI serving
        """
        # Implementation would call existing FastAPI endpoints
        # This is a placeholder
        return {"source": "fastapi", "data": request_data}
    
    async def _route_to_ray_serve(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route request to Ray Serve.
        
        Args:
            request_data: Request data
            
        Returns:
            Response from Ray Serve
        """
        # Implementation would call Ray Serve deployment
        # This is a placeholder
        return {"source": "ray_serve", "data": request_data}
    
    def update_traffic_split(self, new_split: float):
        """
        Update traffic split percentage.
        
        Args:
            new_split: New traffic split percentage (0.0-1.0)
        """
        self.traffic_split = max(0.0, min(1.0, new_split))
        self.ray_serve_enabled = self.traffic_split > 0.0
```

This integration approach ensures seamless compatibility between existing CNN+LSTM models and the new Ray Serve deployment architecture while maintaining all the benefits of distributed serving, auto-scaling, and GPU acceleration.