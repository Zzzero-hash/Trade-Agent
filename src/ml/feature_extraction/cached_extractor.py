"""Cached feature extractor implementation

This module provides a caching wrapper for feature extractors using
TTLCache for efficient caching with time-to-live expiration.
"""

import hashlib
import time
from typing import Dict, Any
import numpy as np

try:
    from cachetools import TTLCache
except ImportError:
    raise ImportError(
        "cachetools is required for cached feature extraction. "
        "Install with: pip install cachetools"
    )

from .base import FeatureExtractor, FeatureExtractionError
from .metrics import PerformanceTracker


class CachedFeatureExtractor(FeatureExtractor):
    """Feature extractor with TTL-based caching
    
    This wrapper adds efficient caching to any feature extractor using
    TTLCache with configurable size and time-to-live settings.
    """
    
    def __init__(self, extractor: FeatureExtractor, cache_size: int = 1000,
                 ttl_seconds: int = 60, enable_metrics: bool = True):
        """Initialize cached feature extractor
        
        Args:
            extractor: Base feature extractor to wrap
            cache_size: Maximum number of cached entries
            ttl_seconds: Time-to-live for cache entries in seconds
            enable_metrics: Whether to track performance metrics
        """
        self.extractor = extractor
        self.cache = TTLCache(maxsize=cache_size, ttl=ttl_seconds)
        self.performance_tracker = PerformanceTracker() if enable_metrics else None
        
    def extract_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract features with caching
        
        Args:
            data: Input market data as numpy array
            
        Returns:
            Dictionary containing extracted features
            
        Raises:
            FeatureExtractionError: If feature extraction fails
        """
        start_time = time.time()
        cache_hit = False
        error = False
        
        try:
            # Generate cache key from data hash
            cache_key = self._generate_cache_key(data)
            
            # Check cache first
            if cache_key in self.cache:
                cache_hit = True
                result = self.cache[cache_key]
            else:
                # Extract features and cache result
                result = self.extractor.extract_features(data)
                self.cache[cache_key] = result
                
            return result
            
        except Exception as e:
            error = True
            raise FeatureExtractionError(f"Cached feature extraction failed: {e}") from e
            
        finally:
            if self.performance_tracker:
                execution_time = time.time() - start_time
                self.performance_tracker.track_execution(
                    execution_time, cache_hit, error
                )
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get feature dimensions from wrapped extractor"""
        return self.extractor.get_feature_dimensions()
    
    def _generate_cache_key(self, data: np.ndarray) -> str:
        """Generate cache key from data hash
        
        Args:
            data: Input data array
            
        Returns:
            Hash string to use as cache key
        """
        # Use MD5 hash of data bytes for cache key
        return hashlib.md5(data.tobytes()).hexdigest()
    
    def clear_cache(self):
        """Clear all cached entries"""
        self.cache.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'cache_size': len(self.cache),
            'max_size': self.cache.maxsize,
            'ttl': self.cache.ttl,
            'performance_summary': (
                self.performance_tracker.get_summary() 
                if self.performance_tracker else None
            )
        }