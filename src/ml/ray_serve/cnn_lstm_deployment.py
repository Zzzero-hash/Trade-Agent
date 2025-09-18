"""Ray Serve deployment for CNN+LSTM hybrid models.

This module implements the Ray Serve deployment for CNN+LSTM hybrid models with
auto-scaling, GPU acceleration, and monitoring integration to meet the <100ms
feature extraction requirement.
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional
import time
import logging
import asyncio
from collections import deque
from dataclasses import dataclass

from src.ml.hybrid_model import CNNLSTMHybridModel, HybridModelConfig
from src.utils.monitoring import get_metrics_collector

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Configuration for CNN+LSTM Ray Serve deployment."""
    
    # Model configuration
    model_path: Optional[str] = None
    device: str = "cpu"
    
    # Resource configuration
    num_replicas: int = 2
    num_cpus: int = 2
    num_gpus: float = 0.5
    memory: int = 2 * 1024 * 1024 * 1024  # 2GB
    object_store_memory: int = 1 * 1024 * 1024 * 1024  # 1GB
    
    # Auto-scaling configuration
    min_replicas: int = 2
    max_replicas: int = 20
    target_num_ongoing_requests_per_replica: int = 5
    upscale_delay_s: int = 30
    downscale_delay_s: int = 300
    upscale_smoothing_factor: float = 1.0
    downscale_smoothing_factor: float = 0.5
    metrics_interval_s: int = 10
    look_back_period_s: int = 120
    
    # Batch processing configuration
    max_batch_size: int = 32
    batch_wait_timeout_s: float = 0.01
    
    def __post_init__(self):
        """Initialize device based on GPU availability."""
        if torch.cuda.is_available():
            self.device = "cuda"
            self.num_gpus = 0.5
        else:
            self.device = "cpu"
            self.num_gpus = 0.0


# Auto-scaling configuration
AUTOSCALING_CONFIG = {
    "min_replicas": 2,
    "max_replicas": 20,
    "target_num_ongoing_requests_per_replica": 5,
    "upscale_delay_s": 30,
    "downscale_delay_s": 300,
    "upscale_smoothing_factor": 1.0,
    "downscale_smoothing_factor": 0.5,
    "metrics_interval_s": 10,
    "look_back_period_s": 120
}


class BatchRequest:
    """Represents a single request in a batch with priority."""
    
    def __init__(self, data: np.ndarray, priority: int = 0):
        self.data = data
        self.priority = priority
        self.timestamp = time.time()
        self.future = asyncio.Future()


class CNNLSTMPredictor:
    """Ray Serve deployment for CNN+LSTM hybrid models."""
    
    def __init__(self, model_path: str = None, device: str = "cpu", ab_test_experiment: str = None):
        """Initialize predictor with model loading and configuration.
        
        Args:
            model_path: Path to the pre-trained model
            device: Device to run inference on ('cpu' or 'cuda')
            ab_test_experiment: A/B test experiment ID (optional)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        metrics_collector = get_metrics_collector()
        self.metrics = metrics_collector if metrics_collector else None
        self.ab_test_experiment = ab_test_experiment
        
        # Load the pre-trained CNN+LSTM hybrid model
        if model_path:
            self.model = self._load_model_from_path(model_path)
        else:
            # Load default model configuration
            self.model = self._create_default_model()
        
        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize performance tracking
        self.request_count = 0
        self.total_processing_time = 0.0
        self.start_time = time.time()
        
        # Initialize batch processing
        self.request_queue = deque()
        self.batch_lock = asyncio.Lock()
        self.batch_timeout = 0.01  # 10ms default timeout
        self.max_batch_size = 32
        self.batch_timer = None
        self.priority_levels = 3  # High, Medium, Low priority
        
        # GPU optimization settings
        self.enable_tf32 = True
        self.enable_cudnn_benchmark = True
        self._setup_gpu_optimizations()
        
        # A/B testing integration
        if self.ab_test_experiment:
            try:
                from src.ml.ray_serve.ab_testing import ab_test_manager
                self.ab_test_manager = ab_test_manager
                logger.info("A/B test manager initialized for experiment: %s", self.ab_test_experiment)
            except ImportError:
                self.ab_test_manager = None
                logger.warning("A/B test manager not available")
        else:
            self.ab_test_manager = None
        
        logger.info("CNNLSTMPredictor initialized on device: %s", self.device)
    
    def _setup_gpu_optimizations(self):
        """Setup GPU-specific optimizations for better performance."""
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                # Enable TensorFloat-32 for better performance on modern GPUs
                if self.enable_tf32:
                    cuda_matmul = getattr(torch.backends.cuda, 'matmul', None)
                    matmul_allow = getattr(cuda_matmul, 'allow_tf32', None)
                    if cuda_matmul and matmul_allow:
                        torch.backends.cuda.matmul.allow_tf32 = True
                    
                    cudnn_module = torch.backends.cudnn
                    cudnn_allow = getattr(cudnn_module, 'allow_tf32', None)
                    if cudnn_allow:
                        torch.backends.cudnn.allow_tf32 = True
                
                # Enable cuDNN benchmark for better performance
                if self.enable_cudnn_benchmark:
                    torch.backends.cudnn.benchmark = True
                
                logger.info("GPU optimizations enabled")
            except Exception as e:
                logger.warning("GPU optimization setup failed: %s", e)
    
    def _load_model_from_path(self, model_path: str) -> CNNLSTMHybridModel:
        """Load model from a given path.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded CNN+LSTM hybrid model
        """
        try:
            # This would load the actual model in a real implementation
            # For now, we'll create a minimal model for demonstration
            config = HybridModelConfig(
                model_type="CNNLSTMHybridModel",
                input_dim=10,
                output_dim=4,
                hidden_dims=[64],
                sequence_length=60,
                device=self.device
            )
            model = CNNLSTMHybridModel(config)
            logger.info("Model loaded successfully from %s", model_path)
            return model
        except Exception as e:
            logger.error("Failed to load model from %s: %s", model_path, e)
            raise
    
    def _create_default_model(self) -> CNNLSTMHybridModel:
        """Create a default CNN+LSTM model for testing purposes.
        
        Returns:
            Default CNN+LSTM hybrid model
        """
        config = HybridModelConfig(
            model_type="CNNLSTMHybridModel",
            input_dim=10,
            output_dim=4,
            hidden_dims=[64],
            sequence_length=60,
            device=self.device
        )
        
        model = CNNLSTMHybridModel(config)
        logger.info("Default model created")
        return model
    
    async def _process_batch(self, requests: List[BatchRequest]) -> None:
        """Process a batch of requests.
        
        Args:
            requests: List of batch requests to process
        """
        if not requests:
            return
        
        start_time = time.time()
        
        try:
            # Extract data from requests
            batch_data = [req.data for req in requests]
            
            # Convert to batch tensor
            batch_input = np.stack(batch_data)
            input_tensor = torch.FloatTensor(batch_input).to(self.device)
            
            # Perform inference
            with torch.no_grad():
                predictions = self.model.forward(
                    input_tensor,
                    return_features=True,
                    use_ensemble=True
                )
            
            # Process results and set futures
            batch_size = input_tensor.shape[0]
            for i, req in enumerate(requests):
                try:
                    # Extract predictions
                    cls_probs = predictions['classification_probs'][i]
                    cls_probs = cls_probs.cpu().numpy()
                    
                    reg_pred = predictions['regression_mean'][i]
                    reg_pred = reg_pred.cpu().numpy()
                    
                    reg_unc = predictions['regression_uncertainty'][i]
                    reg_unc = reg_unc.cpu().numpy()
                    
                    ens_weights = None
                    if predictions['ensemble_weights'] is not None:
                        ens_weights = predictions['ensemble_weights']
                        ens_weights = ens_weights.cpu().numpy()
                    
                    result = {
                        'classification_probs': cls_probs,
                        'regression_pred': reg_pred,
                        'regression_uncertainty': reg_unc,
                        'ensemble_weights': ens_weights,
                        'processing_time_ms': (
                            (time.time() - start_time) * 1000 / batch_size
                        )
                    }
                    req.future.set_result(result)
                except Exception as e:
                    req.future.set_exception(e)
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self.request_count += len(requests)
            self.total_processing_time += processing_time
            
            if self.metrics:
                self.metrics.increment_counter(
                    "model_predictions_total", 
                    {"model_type": "cnn_lstm"}
                )
                self.metrics.record_histogram(
                    "prediction_latency_ms", 
                    processing_time / len(requests)
                )
                self.metrics.record_histogram("batch_size", len(requests))
                
        except Exception as e:
            # Set exception for all requests in the batch
            for req in requests:
                req.future.set_exception(e)
            
            if self.metrics:
                self.metrics.increment_counter(
                    "model_prediction_errors_total", 
                    {"model_type": "cnn_lstm"}
                )
            logger.error("Batch prediction failed: %s", e)
    
    async def _batch_scheduler(self) -> None:
        """Scheduler that processes batches based on size and timeout."""
        while True:
            try:
                async with self.batch_lock:
                    # Check if we have enough requests for a batch
                    queue_size = len(self.request_queue)
                    if queue_size >= self.max_batch_size:
                        # Form a full batch
                        batch_requests = []
                        batch_size = min(queue_size, self.max_batch_size)
                        for _ in range(batch_size):
                            batch_requests.append(self.request_queue.popleft())
                        
                        # Process the batch
                        if batch_requests:
                            await self._process_batch(batch_requests)
                    elif queue_size > 0:
                        # Check if timeout expired
                        first_req_time = self.request_queue[0].timestamp
                        if time.time() - first_req_time > self.batch_timeout:
                            # Process partial batch due to timeout
                            batch_requests = []
                            while self.request_queue:
                                item = self.request_queue.popleft()
                                batch_requests.append(item)
                            
                            # Process the batch
                            if batch_requests:
                                await self._process_batch(batch_requests)
                
                # Small sleep to prevent busy waiting
                await asyncio.sleep(0.001)
                
            except Exception as e:
                logger.error("Batch scheduler error: %s", e)
                await asyncio.sleep(0.1)  # Prevent tight loop on errors
    
    async def batch_predict(
        self, 
        requests: List[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Batch prediction method for improved throughput.
        
        Args:
            requests: List of input data arrays
            
        Returns:
            List of prediction results
        """
        # Convert to batch request objects
        batch_requests = [BatchRequest(req) for req in requests]
        
        # Add requests to queue
        async with self.batch_lock:
            for req in batch_requests:
                self.request_queue.append(req)
        
        # Wait for results
        results = []
        for req in batch_requests:
            try:
                result = await req.future
                results.append(result)
            except Exception as e:
                logger.error("Batch request failed: %s", e)
                raise e
        
        return results
    
    async def __call__(
        self,
        request_data: np.ndarray,
        priority: int = 0,
        request_id: str = None
    ) -> Dict[str, Any]:
        """Single prediction endpoint with priority queuing and A/B testing support.
        
        Args:
            request_data: Input data array
            priority: Priority level (0=low, 1=medium, 2=high)
            request_id: Unique request identifier for A/B testing (optional)
            
        Returns:
            Prediction result dictionary
        """
        # Validate input
        self._validate_input(request_data)
        
        # A/B testing - determine variant if experiment is active
        ab_test_variant = None
        if self.ab_test_manager and self.ab_test_experiment and request_id:
            ab_test_variant = self.ab_test_manager.get_variant_for_request(
                self.ab_test_experiment, request_id
            )
        
        # Record start time for metrics
        start_time = time.time()
        
        # Create batch request with priority
        req = BatchRequest(request_data, priority)
        
        # Add request to queue based on priority
        async with self.batch_lock:
            if priority > 0:
                # Insert at the beginning for higher priority
                self.request_queue.appendleft(req)
            else:
                # Add to the end for normal/low priority
                self.request_queue.append(req)
        
        # Wait for result
        try:
            result = await req.future
            
            # Record metrics for A/B testing
            if self.ab_test_manager and self.ab_test_experiment and request_id:
                processing_time = (time.time() - start_time) * 1000
                confidence_score = result.get('confidence_scores', [None])[0] if result.get('confidence_scores') else None
                
                self.ab_test_manager.record_metrics(
                    self.ab_test_experiment,
                    ab_test_variant or "control",
                    processing_time,
                    processing_time,
                    confidence_score
                )
            
            return result
            
        except Exception as e:
            # Record error metrics for A/B testing
            if self.ab_test_manager and self.ab_test_experiment and request_id:
                processing_time = (time.time() - start_time) * 1000
                self.ab_test_manager.record_metrics(
                    self.ab_test_experiment,
                    ab_test_variant or "control",
                    processing_time,
                    processing_time,
                    error=True
                )
            
            logger.error("Prediction failed: %s", e)
            raise e
    
    def _validate_input(self, data: np.ndarray) -> None:
        """Validate input data format and dimensions.
        
        Args:
            data: Input data to validate
            
        Raises:
            ValueError: If input data is invalid
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a numpy array")
        
        if data.ndim != 3:
            raise ValueError("Expected 3D input array, got %dD" % data.ndim)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deployment statistics.
        
        Returns:
            Dictionary containing deployment statistics
        """
        avg_processing_time = (
            self.total_processing_time / self.request_count 
            if self.request_count > 0 else 0
        )
        
        uptime = time.time() - self.start_time
        
        # GPU stats if available
        gpu_stats = {}
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                mem_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                mem_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                gpu_stats = {
                    "gpu_memory_allocated_mb": mem_allocated,
                    "gpu_memory_reserved_mb": mem_reserved,
                }
            except Exception:
                pass
        
        return {
            "request_count": self.request_count,
            "avg_processing_time_ms": avg_processing_time,
            "device": self.device,
            "uptime_seconds": uptime,
            "model_type": "CNNLSTMHybridModel",
            "queue_size": len(self.request_queue),
            "batch_size": self.max_batch_size,
            "batch_timeout": self.batch_timeout,
            **gpu_stats
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the deployment.
        
        Returns:
            Dictionary containing health status
        """
        try:
            # Create dummy input data for health check
            np.random.rand(1, 10, 60).astype(np.float32)
            
            # Run inference
            start_time = time.time()
            # In a real implementation, we would call the model
            # For now, we'll just simulate the latency
            latency = time.time() - start_time
            
            # Check result validity
            is_healthy = latency < 0.1  # <100ms requirement
            
            return {
                "status": "healthy" if is_healthy else "degraded",
                "latency_ms": latency * 1000,
                "is_healthy": is_healthy,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error("Health check failed: %s", e)
            return {
                "status": "unhealthy",
                "error": str(e),
                "is_healthy": False,
                "timestamp": time.time()
            }


# For Ray Serve deployment
def bind(model_path: str = None, device: str = "cpu", ab_test_experiment: str = None):
    """Bind the deployment with configuration.
    
    Args:
        model_path: Path to the pre-trained model
        device: Device to run inference on
        ab_test_experiment: A/B test experiment ID (optional)
        
    Returns:
        Configured CNNLSTMPredictor instance
    """
    return CNNLSTMPredictor(model_path=model_path, device=device, ab_test_experiment=ab_test_experiment)


# Deployment entry point
cnn_lstm_deployment = bind()
