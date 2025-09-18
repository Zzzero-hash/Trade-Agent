"""Performance tracking for feature extraction"""

import logging
import time
from typing import Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for feature extraction"""
    extraction_count: int = 0
    total_time: float = 0.0
    cache_hits: int = 0
    fallback_count: int = 0
    error_count: int = 0
    
    def get_average_time(self) -> float:
        """Get average extraction time"""
        return self.total_time / max(self.extraction_count, 1)
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate"""
        return self.cache_hits / max(self.extraction_count, 1)
    
    def get_fallback_rate(self) -> float:
        """Get fallback usage rate"""
        return self.fallback_count / max(self.extraction_count, 1)
    
    def get_error_rate(self) -> float:
        """Get error rate"""
        return self.error_count / max(self.extraction_count, 1)


class PerformanceTracker:
    """Performance tracking for feature extraction operations
    
    This class provides comprehensive performance monitoring with
    structured logging and metrics collection.
    """
    
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.metrics = PerformanceMetrics()
        self._start_time: float = 0.0
    
    def start_extraction(self) -> None:
        """Start timing an extraction operation"""
        self._start_time = time.time()
    
    def end_extraction(
        self, 
        used_cache: bool = False, 
        used_fallback: bool = False,
        had_error: bool = False
    ) -> float:
        """End timing and record metrics
        
        Args:
            used_cache: Whether cache was used
            used_fallback: Whether fallback was used
            had_error: Whether an error occurred
            
        Returns:
            Duration of the extraction in seconds
        """
        duration = time.time() - self._start_time
        
        # Update metrics
        self.metrics.extraction_count += 1
        self.metrics.total_time += duration
        
        if used_cache:
            self.metrics.cache_hits += 1
        if used_fallback:
            self.metrics.fallback_count += 1
        if had_error:
            self.metrics.error_count += 1
        
        # Log periodically
        if self.metrics.extraction_count % self.log_interval == 0:
            self._log_performance()
        
        return duration
    
    def _log_performance(self) -> None:
        """Log current performance statistics"""
        metrics = self.get_metrics()
        
        logger.info("Feature Extraction Performance:")
        logger.info(f"  Total extractions: {metrics['total_extractions']}")
        logger.info(f"  Average time: {metrics['avg_extraction_time']:.4f}s")
        logger.info(f"  Cache hit rate: {metrics['cache_hit_rate']:.2%}")
        logger.info(f"  Fallback rate: {metrics['fallback_rate']:.2%}")
        logger.info(f"  Error rate: {metrics['error_rate']:.2%}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics
        
        Returns:
            Dictionary with performance statistics
        """
        return {
            'total_extractions': self.metrics.extraction_count,
            'total_time': self.metrics.total_time,
            'avg_extraction_time': self.metrics.get_average_time(),
            'cache_hit_rate': self.metrics.get_cache_hit_rate(),
            'fallback_rate': self.metrics.get_fallback_rate(),
            'error_rate': self.metrics.get_error_rate(),
            'cache_hits': self.metrics.cache_hits,
            'fallback_count': self.metrics.fallback_count,
            'error_count': self.metrics.error_count
        }
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics"""
        self.metrics = PerformanceMetrics()
        logger.info("Performance metrics reset")
    
    def log_summary(self) -> None:
        """Log a comprehensive performance summary"""
        metrics = self.get_metrics()
        
        logger.info("=== Feature Extraction Performance Summary ===")
        logger.info(f"Total Operations: {metrics['total_extractions']}")
        logger.info(f"Total Time: {metrics['total_time']:.2f}s")
        logger.info(f"Average Time: {metrics['avg_extraction_time']:.4f}s")
        logger.info(f"Cache Hit Rate: {metrics['cache_hit_rate']:.2%}")
        logger.info(f"Fallback Rate: {metrics['fallback_rate']:.2%}")
        logger.info(f"Error Rate: {metrics['error_rate']:.2%}")
        logger.info("=" * 45)