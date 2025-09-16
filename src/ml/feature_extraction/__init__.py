"""Feature extraction module for CNN+LSTM integration with RL environments

This module provides a clean, modular architecture for feature extraction
with proper separation of concerns, efficient caching, and robust error
handling.
"""

from .base import (
    FeatureExtractor,
    FeatureExtractionError,
    DataValidationError,
    FeatureComputationError,
    ModelLoadError
)
from .config import FeatureExtractionConfig
from .cnn_lstm_extractor import CNNLSTMExtractor
from .cached_extractor import CachedFeatureExtractor
from .intelligent_cached_extractor import (
    IntelligentCachedFeatureExtractor,
    CacheTTLConfig,
    CacheInvalidationConfig
)
from .fallback_extractor import FallbackFeatureExtractor
from .factory import FeatureExtractorFactory
from .metrics import PerformanceTracker, PerformanceMetrics

__all__ = [
    'FeatureExtractor',
    'FeatureExtractionError',
    'DataValidationError',
    'FeatureComputationError',
    'ModelLoadError',
    'FeatureExtractionConfig',
    'CNNLSTMExtractor',
    'CachedFeatureExtractor',
    'IntelligentCachedFeatureExtractor',
    'CacheTTLConfig',
    'CacheInvalidationConfig',
    'FallbackFeatureExtractor',
    'FeatureExtractorFactory',
    'PerformanceTracker',
    'PerformanceMetrics'
]