"""Factory for creating configured feature extractors

This module provides a factory class for creating properly configured
feature extractors with caching, fallback, and performance tracking.
"""

from typing import Optional
import logging

from .base import FeatureExtractor
from .config import FeatureExtractionConfig
from .cnn_lstm_extractor import CNNLSTMExtractor
from .cached_extractor import CachedFeatureExtractor
from .intelligent_cached_extractor import IntelligentCachedFeatureExtractor
from .fallback_extractor import FallbackFeatureExtractor


class FeatureExtractorFactory:
    """Factory for creating configured feature extractors"""
    
    @staticmethod
    def create_extractor(
        hybrid_model,
        config: Optional[FeatureExtractionConfig] = None,
        device: Optional[str] = None
    ) -> FeatureExtractor:
        """Create a fully configured feature extractor
        
        Args:
            hybrid_model: Pre-trained CNN+LSTM hybrid model
            config: Feature extraction configuration
            device: Device for model inference
            
        Returns:
            Configured feature extractor with caching and fallback
        """
        if config is None:
            config = FeatureExtractionConfig()
        
        logger = logging.getLogger(__name__)
        logger.info("Creating CNN+LSTM feature extractor")
        
        # Create base CNN+LSTM extractor
        base_extractor = CNNLSTMExtractor(hybrid_model, device)
        
        # Add caching if enabled
        if config.enable_caching:
            logger.info(
                f"Adding caching layer (size={config.cache_size}, "
                f"ttl={config.cache_ttl_seconds}s)"
            )
            base_extractor = CachedFeatureExtractor(
                base_extractor,
                cache_size=config.cache_size,
                ttl_seconds=config.cache_ttl_seconds,
                enable_metrics=config.log_performance
            )
        
        # Add fallback if enabled
        if config.enable_fallback:
            logger.info(
                "Adding fallback layer with basic technical indicators"
            )
            base_extractor = FallbackFeatureExtractor(base_extractor)
        
        return base_extractor
    
    @staticmethod
    def create_basic_extractor(
        hybrid_model,
        device: Optional[str] = None
    ) -> FeatureExtractor:
        """Create a basic CNN+LSTM extractor without caching or fallback
        
        Args:
            hybrid_model: Pre-trained CNN+LSTM hybrid model
            device: Device for model inference
            
        Returns:
            Basic CNN+LSTM feature extractor
        """
        return CNNLSTMExtractor(hybrid_model, device)
    
    @staticmethod
    def create_cached_extractor(
        hybrid_model,
        cache_size: int = 1000,
        ttl_seconds: int = 60,
        device: Optional[str] = None
    ) -> FeatureExtractor:
        """Create a cached CNN+LSTM extractor
        
        Args:
            hybrid_model: Pre-trained CNN+LSTM hybrid model
            cache_size: Maximum cache size
            ttl_seconds: Cache time-to-live in seconds
            device: Device for model inference
            
        Returns:
            Cached CNN+LSTM feature extractor
        """
        base_extractor = CNNLSTMExtractor(hybrid_model, device)
        return CachedFeatureExtractor(
            base_extractor,
            cache_size=cache_size,
            ttl_seconds=ttl_seconds
        )
    
    @staticmethod
    def create_intelligent_cached_extractor(
        hybrid_model,
        cache_size: int = 1000,
        ttl_seconds: int = 60,
        device: Optional[str] = None
    ) -> FeatureExtractor:
        """Create an intelligent cached CNN+LSTM extractor
        
        Args:
            hybrid_model: Pre-trained CNN+LSTM hybrid model
            cache_size: Maximum cache size
            ttl_seconds: Cache time-to-live in seconds
            device: Device for model inference
            
        Returns:
            Intelligent cached CNN+LSTM feature extractor
        """
        base_extractor = CNNLSTMExtractor(hybrid_model, device)
        return IntelligentCachedFeatureExtractor(
            base_extractor,
            config=FeatureExtractionConfig(
                cache_size=cache_size,
                cache_ttl_seconds=ttl_seconds
            )
        )