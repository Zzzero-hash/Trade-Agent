"""Monitoring and health check integration for Ray Serve CNN+LSTM deployments.

This module provides monitoring utilities, metrics collection, and health checking
for CNN+LSTM model deployments in Ray Serve.
"""

import time
import torch
import numpy as np
from typing import Dict, Any
import logging

# Try to import Prometheus metrics (optional dependency)
try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Histogram = Gauge = object  # Fallback to avoid import errors

# Configure logging
logger = logging.getLogger(__name__)


class ModelMetrics:
    """Metrics collection for CNN+LSTM deployments."""
    
    def __init__(self):
        """Initialize metrics collectors."""
        if PROMETHEUS_AVAILABLE:
            self.prediction_requests = Counter(
                'cnn_lstm_prediction_requests_total',
                'Total number of prediction requests',
                ['model_version']
            )
            
            self.prediction_latency = Histogram(
                'cnn_lstm_prediction_latency_seconds',
                'Prediction latency in seconds',
                ['model_version'],
                buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
            )
            
            self.prediction_errors = Counter(
                'cnn_lstm_prediction_errors_total',
                'Total number of prediction errors',
                ['model_version', 'error_type']
            )
            
            self.gpu_utilization = Gauge(
                'cnn_lstm_gpu_utilization_percent',
                'GPU utilization percentage',
                ['gpu_id']
            )
            
            self.replica_count = Gauge(
                'cnn_lstm_replicas_total',
                'Number of active replicas'
            )
        else:
            # Create mock metrics objects
            self.prediction_requests = None
            self.prediction_latency = None
            self.prediction_errors = None
            self.gpu_utilization = None
            self.replica_count = None
    
    def increment_counter(self, name: str, labels: Dict[str, str] = None):
        """Increment a counter metric.
        
        Args:
            name: Name of the counter
            labels: Label dictionary for the metric
        """
        if hasattr(self, name) and getattr(self, name) is not None:
            if labels:
                getattr(self, name).labels(**labels).inc()
            else:
                getattr(self, name).inc()
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram metric.
        
        Args:
            name: Name of the histogram
            value: Value to record
            labels: Label dictionary for the metric
        """
        if hasattr(self, name) and getattr(self, name) is not None:
            if labels:
                getattr(self, name).labels(**labels).observe(value)
            else:
                getattr(self, name).observe(value)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric.
        
        Args:
            name: Name of the gauge
            value: Value to set
            labels: Label dictionary for the metric
        """
        if hasattr(self, name) and getattr(self, name) is not None:
            if labels:
                getattr(self, name).labels(**labels).set(value)
            else:
                getattr(self, name).set(value)


class HealthChecker:
    """Health checking for CNN+LSTM deployments."""
    
    def __init__(self, model_deployment):
        """Initialize health checker.
        
        Args:
            model_deployment: The model deployment to check
        """
        self.model_deployment = model_deployment
        self.last_health_check = 0
        self.health_status = "unknown"
    
    async def check_model_health(self) -> Dict[str, Any]:
        """Check model health by running a simple inference.
        
        Returns:
            Health status dictionary
        """
        try:
            # Create dummy input data
            dummy_input = np.random.rand(1, 10, 60).astype(np.float32)
            
            # Run inference
            start_time = time.time()
            result = await self.model_deployment.__call__(dummy_input)
            latency = time.time() - start_time
            
            # Check result validity
            is_healthy = (
                result is not None and
                'classification_probs' in result and
                latency < 0.1  # <100ms requirement
            )
            
            self.health_status = "healthy" if is_healthy else "degraded"
            self.last_health_check = time.time()
            
            return {
                "status": self.health_status,
                "latency_ms": latency * 1000,
                "last_check": self.last_health_check,
                "is_healthy": is_healthy
            }
            
        except Exception as e:
            self.health_status = "unhealthy"
            self.last_health_check = time.time()
            
            return {
                "status": self.health_status,
                "error": str(e),
                "last_check": self.last_health_check,
                "is_healthy": False
            }
    
    def get_gpu_health(self) -> Dict[str, Any]:
        """Get GPU health information.
        
        Returns:
            Dictionary with GPU health information
        """
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                max_memory = torch.cuda.get_device_properties(0).total_memory
                
                return {
                    "gpu_available": True,
                    "allocated_mb": allocated / 1024 / 1024,
                    "reserved_mb": reserved / 1024 / 1024,
                    "max_memory_mb": max_memory / 1024 / 1024,
                    "utilization_pct": (allocated / max_memory * 100) if max_memory > 0 else 0
                }
            except Exception as e:
                logger.warning(f"Failed to get GPU health info: {e}")
                return {"gpu_available": True, "error": str(e)}
        
        return {"gpu_available": False}
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health information.
        
        Returns:
            Dictionary with system health information
        """
        health_info = {
            "timestamp": time.time(),
            "model_status": self.health_status,
            "last_health_check": self.last_health_check
        }
        
        # Add GPU health if available
        gpu_health = self.get_gpu_health()
        health_info.update(gpu_health)
        
        return health_info


# Global metrics instance
metrics = ModelMetrics()


def get_metrics_collector():
    """Get the global metrics collector.
    
    Returns:
        ModelMetrics instance
    """
    return metrics


class PerformanceMonitor:
    """Performance monitoring for CNN+LSTM deployments."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.request_count = 0
        self.total_latency = 0.0
        self.error_count = 0
        self.start_time = time.time()
    
    def record_request(self, latency_ms: float, success: bool = True):
        """Record a request for performance monitoring.
        
        Args:
            latency_ms: Request latency in milliseconds
            success: Whether the request was successful
        """
        self.request_count += 1
        self.total_latency += latency_ms
        
        if not success:
            self.error_count += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        uptime = time.time() - self.start_time
        avg_latency = self.total_latency / self.request_count if self.request_count > 0 else 0
        
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "avg_latency_ms": avg_latency,
            "uptime_seconds": uptime,
            "requests_per_second": self.request_count / uptime if uptime > 0 else 0,
            "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0
        }
    
    def check_performance_requirements(self) -> Dict[str, Any]:
        """Check if performance requirements are met.
        
        Returns:
            Dictionary with performance requirement status
        """
        stats = self.get_performance_stats()
        avg_latency = stats["avg_latency_ms"]
        
        return {
            "meets_100ms_requirement": avg_latency < 100,
            "avg_latency_ms": avg_latency,
            "target_latency_ms": 100,
            "latency_margin_ms": 100 - avg_latency
        }