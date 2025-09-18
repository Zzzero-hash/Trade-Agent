"""Configuration classes for feature extraction

This module provides configuration dataclasses for the feature extraction
system with proper validation and type hints.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FeatureExtractionConfig:
    """Configuration for CNN+LSTM feature extraction"""
    
    # Model paths
    hybrid_model_path: Optional[str] = None
    
    # Feature dimensions
    fused_feature_dim: int = 256
    uncertainty_dim: int = 1
    
    # Caching configuration
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl_seconds: int = 60
    
    # Batch processing
    batch_size: int = 32
    max_batch_wait_time: float = 0.1  # seconds
    
    # Fallback configuration
    enable_fallback: bool = True
    fallback_feature_dim: int = 15  # Basic technical indicators
    
    # Performance monitoring
    log_performance: bool = True
    performance_log_interval: int = 100
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.fused_feature_dim <= 0:
            raise ValueError("fused_feature_dim must be positive")
        
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        
        if self.cache_ttl_seconds <= 0:
            raise ValueError("cache_ttl_seconds must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.max_batch_wait_time < 0:
            raise ValueError("max_batch_wait_time must be non-negative")