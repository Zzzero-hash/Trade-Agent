"""
Resource management for monitoring service with async context managers.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, AsyncGenerator
import numpy as np
from datetime import datetime, timedelta

from src.utils.logging import get_logger
from .exceptions import InsufficientDataError

logger = get_logger("resource_manager")


class DataBuffer:
    """Efficient circular buffer for monitoring data."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.data: list = []
        self._lock = asyncio.Lock()
    
    async def append(self, item: Any) -> None:
        """Thread-safe append with size limit."""
        async with self._lock:
            self.data.append(item)
            if len(self.data) > self.max_size:
                # Remove oldest items efficiently
                self.data = self.data[-self.max_size:]
    
    async def get_recent(self, count: Optional[int] = None) -> list:
        """Get recent items thread-safely."""
        async with self._lock:
            if count is None:
                return self.data.copy()
            return self.data[-count:] if count <= len(self.data) else self.data.copy()
    
    async def get_size(self) -> int:
        """Get current buffer size."""
        async with self._lock:
            return len(self.data)
    
    async def clear_old(self, cutoff_date: datetime) -> int:
        """Clear old data based on timestamp."""
        async with self._lock:
            original_size = len(self.data)
            self.data = [
                item for item in self.data
                if item.get('timestamp', datetime.now()) > cutoff_date
            ]
            removed_count = original_size - len(self.data)
            return removed_count


class FeatureCache:
    """Efficient caching for feature arrays with TTL."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[np.ndarray]:
        """Get cached features if not expired."""
        async with self._lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            if datetime.now() - entry['timestamp'] > timedelta(seconds=self.ttl_seconds):
                del self.cache[key]
                return None
            
            return entry['features']
    
    async def set(self, key: str, features: np.ndarray) -> None:
        """Cache features with timestamp."""
        async with self._lock:
            # Implement LRU eviction if cache is full
            if len(self.cache) >= self.max_size:
                oldest_key = min(
                    self.cache.keys(),
                    key=lambda k: self.cache[k]['timestamp']
                )
                del self.cache[oldest_key]
            
            self.cache[key] = {
                'features': features.copy(),
                'timestamp': datetime.now()
            }
    
    async def clear_expired(self) -> int:
        """Clear expired cache entries."""
        async with self._lock:
            now = datetime.now()
            expired_keys = [
                key for key, entry in self.cache.items()
                if now - entry['timestamp'] > timedelta(seconds=self.ttl_seconds)
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            return len(expired_keys)


class MonitoringResourceManager:
    """Manages resources for monitoring service."""
    
    def __init__(self, max_buffer_size: int = 1000):
        self.prediction_buffers: Dict[str, DataBuffer] = {}
        self.feature_buffers: Dict[str, DataBuffer] = {}
        self.feature_cache = FeatureCache()
        self.max_buffer_size = max_buffer_size
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info("Resource manager started")
    
    async def stop(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Resource manager stopped")
    
    @asynccontextmanager
    async def get_model_data(self, model_name: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Context manager for accessing model data."""
        try:
            # Ensure buffers exist
            if model_name not in self.prediction_buffers:
                self.prediction_buffers[model_name] = DataBuffer(self.max_buffer_size)
            if model_name not in self.feature_buffers:
                self.feature_buffers[model_name] = DataBuffer(self.max_buffer_size)
            
            # Get data
            predictions = await self.prediction_buffers[model_name].get_recent()
            features = await self.feature_buffers[model_name].get_recent()
            
            yield {
                'predictions': predictions,
                'features': features,
                'prediction_buffer': self.prediction_buffers[model_name],
                'feature_buffer': self.feature_buffers[model_name]
            }
        
        except Exception as e:
            logger.error(f"Error accessing model data for {model_name}: {e}")
            raise
    
    async def add_prediction(
        self, 
        model_name: str, 
        prediction_data: Dict[str, Any],
        features: np.ndarray
    ) -> None:
        """Add prediction and features to buffers."""
        
        # Ensure buffers exist
        if model_name not in self.prediction_buffers:
            self.prediction_buffers[model_name] = DataBuffer(self.max_buffer_size)
        if model_name not in self.feature_buffers:
            self.feature_buffers[model_name] = DataBuffer(self.max_buffer_size)
        
        # Add to buffers
        await self.prediction_buffers[model_name].append(prediction_data)
        await self.feature_buffers[model_name].append(features)
        
        # Cache features for quick access
        cache_key = f"{model_name}_{datetime.now().timestamp()}"
        await self.feature_cache.set(cache_key, features)
    
    async def get_sufficient_data(
        self, 
        model_name: str, 
        min_samples: int,
        data_type: str = 'predictions'
    ) -> list:
        """Get data ensuring sufficient samples."""
        
        buffer = (self.prediction_buffers.get(model_name) if data_type == 'predictions'
                 else self.feature_buffers.get(model_name))
        
        if not buffer:
            raise InsufficientDataError(
                f"get_{data_type}", min_samples, 0
            )
        
        data = await buffer.get_recent()
        if len(data) < min_samples:
            raise InsufficientDataError(
                f"get_{data_type}", min_samples, len(data)
            )
        
        return data
    
    async def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old data from all buffers."""
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cleanup_stats = {}
        
        # Clean prediction buffers
        for model_name, buffer in self.prediction_buffers.items():
            removed = await buffer.clear_old(cutoff_date)
            cleanup_stats[f"{model_name}_predictions"] = removed
        
        # Clean feature buffers
        for model_name, buffer in self.feature_buffers.items():
            removed = await buffer.clear_old(cutoff_date)
            cleanup_stats[f"{model_name}_features"] = removed
        
        # Clean feature cache
        cache_removed = await self.feature_cache.clear_expired()
        cleanup_stats['cache_entries'] = cache_removed
        
        logger.info(f"Cleanup completed: {cleanup_stats}")
        return cleanup_stats
    
    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup task."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self.cleanup_old_data()
                await self.feature_cache.clear_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        stats = {
            'prediction_buffers': {},
            'feature_buffers': {},
            'cache_size': len(self.feature_cache.cache)
        }
        
        for model_name, buffer in self.prediction_buffers.items():
            stats['prediction_buffers'][model_name] = await buffer.get_size()
        
        for model_name, buffer in self.feature_buffers.items():
            stats['feature_buffers'][model_name] = await buffer.get_size()
        
        return stats