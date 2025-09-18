"""Configuration for Ray Serve CNN+LSTM deployments.

This module provides configuration classes and utilities for Ray Serve deployments
of CNN+LSTM models with auto-scaling capabilities.
"""

from dataclasses import dataclass
from typing import Dict, Any
import torch


@dataclass
class AutoscalingConfig:
    """Auto-scaling configuration for CNN+LSTM deployments."""
    
    # Replica configuration
    min_replicas: int = 2
    max_replicas: int = 20
    target_num_ongoing_requests_per_replica: int = 5
    
    # Scaling timing
    upscale_delay_s: int = 30
    downscale_delay_s: int = 300
    
    # Scaling aggressiveness
    upscale_smoothing_factor: float = 1.0
    downscale_smoothing_factor: float = 0.5
    
    # Metrics collection
    metrics_interval_s: int = 10
    look_back_period_s: int = 120
    
    def to_ray_config(self) -> Dict[str, Any]:
        """Convert to Ray Serve autoscaling config format.
        
        Returns:
            Dictionary compatible with Ray Serve autoscaling configuration
        """
        return {
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "target_num_ongoing_requests_per_replica": self.target_num_ongoing_requests_per_replica,
            "upscale_delay_s": self.upscale_delay_s,
            "downscale_delay_s": self.downscale_delay_s,
            "upscale_smoothing_factor": self.upscale_smoothing_factor,
            "downscale_smoothing_factor": self.downscale_smoothing_factor,
            "metrics_interval_s": self.metrics_interval_s,
            "look_back_period_s": self.look_back_period_s
        }


@dataclass
class ResourceConfig:
    """Resource configuration for CNN+LSTM deployments."""
    
    num_cpus: int = 2
    num_gpus: float = 0.5
    memory: int = 2 * 1024 * 1024 * 1024  # 2GB
    object_store_memory: int = 1 * 1024 * 1024 * 1024  # 1GB
    
    def __post_init__(self):
        """Initialize GPU configuration based on availability."""
        if not torch.cuda.is_available():
            self.num_gpus = 0.0
    
    def to_ray_config(self) -> Dict[str, Any]:
        """Convert to Ray Serve resource config format.
        
        Returns:
            Dictionary compatible with Ray Serve resource configuration
        """
        return {
            "num_cpus": self.num_cpus,
            "num_gpus": self.num_gpus,
            "memory": self.memory,
            "object_store_memory": self.object_store_memory
        }


@dataclass
class BatchConfig:
    """Batch processing configuration."""
    
    max_batch_size: int = 32
    batch_wait_timeout_s: float = 0.01
    
    def to_ray_config(self) -> Dict[str, Any]:
        """Convert to Ray Serve batch config format.
        
        Returns:
            Dictionary with batch configuration parameters
        """
        return {
            "max_batch_size": self.max_batch_size,
            "batch_wait_timeout_s": self.batch_wait_timeout_s
        }


class TradingWorkloadAutoscaler:
    """Auto-scaling policies optimized for trading workloads."""
    
    @staticmethod
    def get_market_hours_config() -> AutoscalingConfig:
        """Auto-scaling configuration for active market hours.
        
        Returns:
            Autoscaling configuration optimized for market hours
        """
        return AutoscalingConfig(
            min_replicas=5,
            max_replicas=30,
            target_num_ongoing_requests_per_replica=3,
            upscale_delay_s=15,      # Quick upscale during market hours
            downscale_delay_s=120,   # Slower downscale to avoid oscillation
            upscale_smoothing_factor=1.5,  # More aggressive upscaling
            downscale_smoothing_factor=0.3   # Conservative downscaling
        )
    
    @staticmethod
    def get_off_hours_config() -> AutoscalingConfig:
        """Auto-scaling configuration for off-market hours.
        
        Returns:
            Autoscaling configuration optimized for off-hours
        """
        return AutoscalingConfig(
            min_replicas=2,
            max_replicas=10,
            target_num_ongoing_requests_per_replica=10,
            upscale_delay_s=60,      # Slower upscale during off hours
            downscale_delay_s=300,   # Much slower downscale
            upscale_smoothing_factor=1.0,
            downscale_smoothing_factor=0.5
        )
    
    @staticmethod
    def get_stress_test_config() -> AutoscalingConfig:
        """Auto-scaling configuration for stress testing.
        
        Returns:
            Autoscaling configuration for stress testing
        """
        return AutoscalingConfig(
            min_replicas=10,
            max_replicas=50,
            target_num_ongoing_requests_per_replica=2,
            upscale_delay_s=5,       # Very quick upscale for testing
            downscale_delay_s=60,
            upscale_smoothing_factor=2.0,   # Very aggressive upscaling
            downscale_smoothing_factor=0.1   # Very conservative downscaling
        )


# Default configurations
DEFAULT_AUTOSCALING_CONFIG = AutoscalingConfig()
DEFAULT_RESOURCE_CONFIG = ResourceConfig()
DEFAULT_BATCH_CONFIG = BatchConfig()

# Resource configurations for different deployment sizes
RESOURCE_CONFIGS = {
    "small": ResourceConfig(
        num_cpus=1,
        num_gpus=0.25,
        memory=1 * 1024 * 1024 * 1024,  # 1GB
        object_store_memory=512 * 1024 * 1024  # 512MB
    ),
    "medium": ResourceConfig(
        num_cpus=2,
        num_gpus=0.5,
        memory=2 * 1024 * 1024 * 1024,  # 2GB
        object_store_memory=1 * 1024 * 1024 * 1024  # 1GB
    ),
    "large": ResourceConfig(
        num_cpus=4,
        num_gpus=1.0,
        memory=4 * 1024 * 1024 * 1024,  # 4GB
        object_store_memory=2 * 1024 * 1024 * 1024  # 2GB
    )
}