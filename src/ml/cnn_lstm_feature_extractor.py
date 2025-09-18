"""CNN+LSTM Feature Extractor for RL Environment Integration

This module provides a feature extractor that combines CNN and LSTM models
to generate rich features for the RL trading environment, implementing
task 12 requirements for CNN+LSTM integration.

This is a legacy interface that wraps the new modular architecture for
backward compatibility.

Requirements: 1.4, 2.4, 9.1
"""

import logging
from typing import Dict, Any, Optional
import numpy as np

from .feature_extraction import (
    FeatureExtractorFactory,
    FeatureExtractionConfig,
    PerformanceTracker,
    create_feature_extraction_config
)

logger = logging.getLogger(__name__)


# Re-export for backward compatibility
FeatureExtractionConfig = FeatureExtractionConfig


# Legacy cache class - deprecated, use CachedFeatureExtractor instead
class FeatureCache:
    """Deprecated: Use CachedFeatureExtractor instead"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 60):
        logger.warning("FeatureCache is deprecated. Use CachedFeatureExtractor instead.")
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds


class CNNLSTMFeatureExtractor:
    """Legacy wrapper for the new modular feature extraction architecture
    
    This class maintains backward compatibility while using the improved
    modular architecture under the hood.
    """
    
    def __init__(self, config: FeatureExtractionConfig):
        logger.warning(
            "CNNLSTMFeatureExtractor is deprecated. "
            "Use FeatureExtractorFactory.create_extractor() instead."
        )
        
        self.config = config
        
        # Create the new modular extractor
        self.extractor, self.performance_tracker = (
            FeatureExtractorFactory.create_with_performance_tracking(config)
        )
        
        # Legacy compatibility properties
        self.is_model_loaded = bool(config.hybrid_model_path)
        self.fallback_mode = False
    
    def load_hybrid_model(self, model_path: str) -> bool:
        """Legacy method - model loading is now handled in the factory
        
        Args:
            model_path: Path to the saved hybrid model
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        logger.warning("load_hybrid_model is deprecated. Use FeatureExtractorFactory instead.")
        try:
            # Recreate extractor with new model path
            new_config = FeatureExtractionConfig(
                hybrid_model_path=model_path,
                fused_feature_dim=self.config.fused_feature_dim,
                enable_caching=self.config.enable_caching,
                cache_size=self.config.cache_size,
                cache_ttl_seconds=self.config.cache_ttl_seconds,
                enable_fallback=self.config.enable_fallback,
                fallback_feature_dim=self.config.fallback_feature_dim,
                log_performance=self.config.log_performance,
                performance_log_interval=self.config.performance_log_interval
            )
            
            self.extractor, self.performance_tracker = (
                FeatureExtractorFactory.create_with_performance_tracking(new_config)
            )
            
            self.is_model_loaded = True
            self.fallback_mode = False
            
            logger.info(f"Successfully loaded CNN+LSTM hybrid model from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load hybrid model from {model_path}: {e}")
            self.is_model_loaded = False
            self.fallback_mode = True
            return False
    
    def extract_features(self, market_window: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract CNN+LSTM features from market data window
        
        Args:
            market_window: Market data window of shape (sequence_length, features)
                          or (batch_size, sequence_length, features)
            
        Returns:
            Dictionary containing extracted features:
            - 'fused_features': Rich CNN+LSTM fused features
            - 'classification_confidence': Model confidence for classification
            - 'regression_uncertainty': Uncertainty estimates for regression
            - 'ensemble_weights': Ensemble model weights (if available)
            - 'fallback_used': Whether fallback features were used
        """
        # Use the new modular architecture
        self.performance_tracker.start_extraction()
        
        try:
            features = self.extractor.extract_features(market_window)
            
            # Track performance
            used_cache = hasattr(self.extractor, 'cache_hits')
            used_fallback = features.get('fallback_used', [False])[0]
            
            self.performance_tracker.end_extraction(
                used_cache=used_cache,
                used_fallback=used_fallback,
                had_error=False
            )
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            
            # Track error
            self.performance_tracker.end_extraction(
                used_cache=False,
                used_fallback=True,
                had_error=True
            )
            
            # Return empty fallback features
            return {
                'fused_features': np.zeros((1, self.config.fallback_feature_dim)),
                'classification_confidence': np.array([0.5]),
                'regression_uncertainty': np.array([1.0]),
                'ensemble_weights': None,
                'fallback_used': np.array([True])
            }
    
    # Legacy methods - deprecated
    def _extract_cnn_lstm_features(self, market_window: np.ndarray) -> Dict[str, np.ndarray]:
        """Deprecated: Use the new modular architecture instead"""
        logger.warning("_extract_cnn_lstm_features is deprecated")
        return self.extractor.extract_features(market_window)
    
    def _extract_fallback_features(self, market_window: np.ndarray) -> Dict[str, np.ndarray]:
        """Deprecated: Use FallbackFeatureExtractor instead"""
        logger.warning("_extract_fallback_features is deprecated")
        return self.extractor.extract_features(market_window)
    
    def _log_performance(self) -> None:
        """Log performance statistics using the new tracker"""
        self.performance_tracker.log_summary()
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get dimensions of extracted features"""
        return self.extractor.get_feature_dimensions()
    
    def reset_performance_stats(self) -> None:
        """Reset performance tracking statistics"""
        self.performance_tracker.reset_metrics()
        
        # Reset cache stats if available
        if hasattr(self.extractor, 'reset_stats'):
            self.extractor.reset_stats()
    
    def enable_fallback_mode(self, enable: bool = True) -> None:
        """Manually enable or disable fallback mode"""
        logger.warning("enable_fallback_mode is deprecated. Fallback is handled automatically.")
        self.fallback_mode = enable
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the feature extractor"""
        metrics = self.performance_tracker.get_metrics()
        
        status = {
            'model_loaded': self.is_model_loaded,
            'fallback_mode': self.fallback_mode,
            'extraction_count': metrics['total_extractions'],
            'cache_enabled': hasattr(self.extractor, 'cache'),
            'fallback_count': metrics['fallback_count'],
            'cache_hits': metrics['cache_hits']
        }
        
        # Add cache size if available
        if hasattr(self.extractor, 'get_cache_stats'):
            cache_stats = self.extractor.get_cache_stats()
            status['cache_size'] = cache_stats['cache_size']
        
        return status


def create_feature_extraction_config(
    hybrid_model_path: Optional[str] = None,
    fused_feature_dim: int = 256,
    enable_caching: bool = True,
    cache_size: int = 1000,
    enable_fallback: bool = True,
    fallback_feature_dim: int = 15,
    **kwargs
) -> FeatureExtractionConfig:
    """Create configuration for CNN+LSTM feature extraction
    
    Args:
        hybrid_model_path: Path to pre-trained hybrid model
        fused_feature_dim: Dimension of fused CNN+LSTM features
        enable_caching: Whether to enable feature caching
        cache_size: Maximum cache size
        enable_fallback: Whether to enable fallback to basic features
        fallback_feature_dim: Dimension of fallback features
        **kwargs: Additional configuration parameters
        
    Returns:
        FeatureExtractionConfig object
    """
    return FeatureExtractionConfig(
        hybrid_model_path=hybrid_model_path,
        fused_feature_dim=fused_feature_dim,
        enable_caching=enable_caching,
        cache_size=cache_size,
        enable_fallback=enable_fallback,
        fallback_feature_dim=fallback_feature_dim,
        **kwargs
    )