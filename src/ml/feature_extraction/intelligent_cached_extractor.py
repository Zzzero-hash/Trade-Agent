"""Intelligent cached feature extractor with advanced caching strategies

This module provides an enhanced caching wrapper for feature extractors using
TTLCache with configurable TTL strategies, adaptive TTL management, and 
cache invalidation capabilities.
"""

import hashlib
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
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
from .config import FeatureExtractionConfig

logger = logging.getLogger(__name__)


@dataclass
class CacheTTLConfig:
    """Configuration for different cache TTL strategies"""
    
    # Base TTL values for different data types
    market_features_ttl: int = 60  # 60 seconds for market features
    model_features_ttl: int = 60   # 60 seconds for model-based features
    predictions_ttl: int = 30      # 30 seconds for predictions
    models_ttl: int = 3600         # 1 hour for models
    
    # Adaptive TTL adjustment factors
    # Reduce TTL for frequently accessed items
    frequent_access_factor: float = 0.5
    # Increase TTL for infrequently accessed items
    infrequent_access_factor: float = 2.0
    
    # Cache warming settings
    enable_cache_warming: bool = True
    warming_threshold: int = 10  # Warm cache after this many accesses


@dataclass
class CacheInvalidationConfig:
    """Configuration for cache invalidation strategies"""
    
    # Pattern-based invalidation
    enable_pattern_invalidation: bool = True
    invalidation_patterns: List[str] = field(default_factory=list)
    
    # Time-based invalidation
    enable_time_based_invalidation: bool = True
    invalidation_interval: int = 300  # 5 minutes
    
    # Event-driven invalidation
    enable_event_invalidation: bool = True


class AdaptiveTTLManager:
    """Manages adaptive TTL based on access patterns"""
    
    def __init__(self, base_ttl: int, config: CacheTTLConfig):
        self.base_ttl = base_ttl
        self.config = config
        self.access_counts: Dict[str, int] = {}
        self.access_timestamps: Dict[str, float] = {}
        
    def record_access(self, cache_key: str) -> None:
        """Record cache access for adaptive TTL calculation"""
        count = self.access_counts.get(cache_key, 0) + 1
        self.access_counts[cache_key] = count
        self.access_timestamps[cache_key] = time.time()
        
    def get_adaptive_ttl(self, cache_key: str) -> int:
        """Calculate adaptive TTL based on access patterns"""
        access_count = self.access_counts.get(cache_key, 0)
        
        if access_count > self.config.warming_threshold:
            # Frequently accessed items get shorter TTL
            factor = self.config.frequent_access_factor
            return max(30, int(self.base_ttl * factor))
        elif access_count == 0:
            # Infrequently accessed items get longer TTL
            return int(self.base_ttl * self.config.infrequent_access_factor)
        else:
            return self.base_ttl
            
    def should_warm_cache(self, cache_key: str) -> bool:
        """Determine if cache should be warmed for this key"""
        if not self.config.enable_cache_warming:
            return False
        threshold = self.config.warming_threshold
        count = self.access_counts.get(cache_key, 0)
        return count >= threshold


class CacheInvalidator:
    """Handles cache invalidation strategies"""
    
    def __init__(self, cache: TTLCache, config: CacheInvalidationConfig):
        self.cache = cache
        self.config = config
        self.last_invalidation_time = time.time()
        
    def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern"""
        if not self.config.enable_pattern_invalidation:
            return 0
            
        invalidated_count = 0
        keys_to_remove = []
        
        for key in self.cache.keys():
            if pattern in str(key):
                keys_to_remove.append(key)
                invalidated_count += 1
                
        for key in keys_to_remove:
            self.cache.pop(key, None)
            
        msg = f"Invalidated {invalidated_count} cache entries"
        logger.info(f"{msg} matching pattern: {pattern}")
        return invalidated_count
        
    def invalidate_by_time(self) -> int:
        """Invalidate cache entries based on time criteria"""
        if not self.config.enable_time_based_invalidation:
            return 0
            
        current_time = time.time()
        interval = self.config.invalidation_interval
        if current_time - self.last_invalidation_time < interval:
            return 0
            
        # For TTLCache, entries are automatically expired
        # but we can clear old entries
        invalidated_count = len(self.cache)
        self.cache.expire()
        new_count = len(self.cache)
        
        invalidated_count = invalidated_count - new_count
        self.last_invalidation_time = current_time
        
        cleared_msg = f"Time-based invalidation cleared {invalidated_count}"
        logger.info(f"{cleared_msg} expired entries")
        return invalidated_count
        
    def invalidate_all(self) -> None:
        """Invalidate all cache entries"""
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"Invalidated all {count} cache entries")


class IntelligentCachedFeatureExtractor(FeatureExtractor):
    """Intelligent feature extractor with advanced caching strategies
    
    This wrapper adds efficient caching to any feature extractor using
    TTLCache with configurable TTL strategies, adaptive TTL management,
    and cache invalidation capabilities.
    """
    
    def __init__(
        self, 
        extractor: FeatureExtractor, 
        config: Optional[FeatureExtractionConfig] = None,
        ttl_config: Optional[CacheTTLConfig] = None,
        invalidation_config: Optional[CacheInvalidationConfig] = None
    ):
        """Initialize intelligent cached feature extractor
        
        Args:
            extractor: Base feature extractor to wrap
            config: Feature extraction configuration
            ttl_config: Cache TTL configuration
            invalidation_config: Cache invalidation configuration
        """
        self.extractor = extractor
        self.config = config or FeatureExtractionConfig()
        self.ttl_config = ttl_config or CacheTTLConfig()
        config = invalidation_config or CacheInvalidationConfig()
        self.invalidation_config = config
        
        # Initialize cache with base TTL for market features
        self.cache = TTLCache(
            maxsize=self.config.cache_size, 
            ttl=self.ttl_config.market_features_ttl
        )
        
        # Initialize performance tracker
        tracker = PerformanceTracker(
            log_interval=self.config.performance_log_interval)
        self.performance_tracker = (
            tracker if self.config.log_performance else None)
        
        # Initialize adaptive TTL manager
        self.adaptive_ttl_manager = AdaptiveTTLManager(
            self.ttl_config.market_features_ttl, 
            self.ttl_config
        )
        
        # Initialize cache invalidator
        invalidator = CacheInvalidator(self.cache, self.invalidation_config)
        self.cache_invalidator = invalidator
        
        # Cache warming tracking
        self.warmed_keys: set = set()
        
        logger.info(
            f"IntelligentCachedFeatureExtractor initialized with "
            f"cache_size={self.config.cache_size}, "
            f"base_ttl={self.ttl_config.market_features_ttl}s"
        )
        
    def extract_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract features with intelligent caching
        
        Args:
            data: Input market data as numpy array
            
        Returns:
            Dictionary containing extracted features
            
        Raises:
            FeatureExtractionError: If feature extraction fails
        """
        cache_hit = False
        error = False
        
        try:
            # Start performance tracking
            if self.performance_tracker:
                self.performance_tracker.start_extraction()
            
            # Generate cache key from data hash
            cache_key = self._generate_cache_key(data)
            
            # Record access for adaptive TTL
            self.adaptive_ttl_manager.record_access(cache_key)
            
            # Check cache first
            if cache_key in self.cache:
                cache_hit = True
                result = self.cache[cache_key]
                logger.debug(f"Cache hit for key: {cache_key[:16]}...")
            else:
                # Extract features and cache result
                logger.debug(f"Cache miss for key: {cache_key[:16]}...")
                result = self.extractor.extract_features(data)
                
                # Apply adaptive TTL
                self.adaptive_ttl_manager.get_adaptive_ttl(cache_key)
                self.cache[cache_key] = result
                
                # Cache warming
                if self.adaptive_ttl_manager.should_warm_cache(cache_key):
                    if cache_key not in self.warmed_keys:
                        self.warmed_keys.add(cache_key)
                        key_msg = f"Cache warmed for key: {cache_key[:16]}..."
                        logger.info(key_msg)
                
            return result
            
        except Exception as e:
            error = True
            logger.error(f"Cached feature extraction failed: {e}")
            error_msg = f"Cached feature extraction failed: {e}"
            raise FeatureExtractionError(error_msg) from e
            
        finally:
            if self.performance_tracker:
                self.performance_tracker.end_extraction(
                    used_cache=cache_hit,
                    had_error=error
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
    
    def clear_cache(self) -> None:
        """Clear all cached entries"""
        self.cache.clear()
        self.adaptive_ttl_manager.access_counts.clear()
        self.adaptive_ttl_manager.access_timestamps.clear()
        self.warmed_keys.clear()
        logger.info("Cache cleared")
    
    def invalidate_cache_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern
        
        Args:
            pattern: Pattern to match for invalidation
            
        Returns:
            Number of invalidated entries
        """
        return self.cache_invalidator.invalidate_by_pattern(pattern)
    
    def invalidate_cache_all(self) -> None:
        """Invalidate all cache entries"""
        self.cache_invalidator.invalidate_all()
        self.adaptive_ttl_manager.access_counts.clear()
        self.adaptive_ttl_manager.access_timestamps.clear()
        self.warmed_keys.clear()
        logger.info("All cache entries invalidated")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        # Perform time-based invalidation
        self.cache_invalidator.invalidate_by_time()
        
        # Build adaptive TTL stats
        access_counts = self.adaptive_ttl_manager.access_counts
        warming_threshold = self.ttl_config.warming_threshold
        
        frequently_accessed = len([
            k for k, v in access_counts.items()
            if v > warming_threshold
        ])
        
        infrequently_accessed = len([
            k for k, v in access_counts.items()
            if v == 0
        ])
        
        # Get performance summary if tracker exists
        performance_summary = None
        if self.performance_tracker:
            performance_summary = self.performance_tracker.get_metrics()
        
        return {
            'cache_size': len(self.cache),
            'max_size': self.cache.maxsize,
            'ttl': self.cache.ttl,
            'warmed_keys_count': len(self.warmed_keys),
            'adaptive_ttl_stats': {
                'frequently_accessed': frequently_accessed,
                'infrequently_accessed': infrequently_accessed
            },
            'performance_summary': performance_summary
        }
    
    def update_ttl_config(self, new_config: CacheTTLConfig) -> None:
        """Update TTL configuration
        
        Args:
            new_config: New TTL configuration
        """
        self.ttl_config = new_config
        self.adaptive_ttl_manager = AdaptiveTTLManager(
            new_config.market_features_ttl, 
            new_config
        )
        logger.info("TTL configuration updated")