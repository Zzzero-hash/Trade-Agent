"""
Resource manager for monitoring system.

This module provides efficient resource management for monitoring data,
including memory management, data buffering, and cleanup operations.
"""

import asyncio
import psutil
import gc
from typing import Dict, List, Any, Optional, AsyncContextManager
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import dataclass
from contextlib import asynccontextmanager
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor

from src.utils.logging import get_logger

logger = get_logger("resource_manager")


@dataclass
class ResourceUsage:
    """Resource usage metrics"""
    timestamp: datetime
    memory_mb: float
    cpu_percent: float
    buffer_sizes: Dict[str, int]
    total_predictions: int
    total_features: int
    cleanup_operations: int


@dataclass
class BufferConfig:
    """Configuration for data buffers"""
    max_size: int = 1000
    cleanup_threshold: float = 0.8  # Cleanup when 80% full
    retention_hours: int = 24
    compression_enabled: bool = True


class DataBuffer:
    """Thread-safe data buffer with automatic cleanup"""
    
    def __init__(self, name: str, config: BufferConfig):
        self.name = name
        self.config = config
        self.data: deque = deque(maxlen=config.max_size)
        self.metadata: Dict[str, Any] = {}
        self.lock = threading.RLock()
        self.last_cleanup = datetime.now()
        self.access_count = 0
        self.total_added = 0
        self.total_removed = 0
    
    def add(self, item: Any, timestamp: Optional[datetime] = None) -> None:
        """Add item to buffer"""
        with self.lock:
            entry = {
                'data': item,
                'timestamp': timestamp or datetime.now(),
                'access_count': 0
            }
            self.data.append(entry)
            self.total_added += 1
            
            # Check if cleanup is needed
            if len(self.data) >= self.config.max_size * self.config.cleanup_threshold:
                self._cleanup_old_data()
    
    def get_recent(self, count: int) -> List[Any]:
        """Get most recent items"""
        with self.lock:
            recent_items = list(self.data)[-count:] if count > 0 else list(self.data)
            
            # Update access counts
            for item in recent_items:
                item['access_count'] += 1
            
            self.access_count += 1
            return [item['data'] for item in recent_items]
    
    def get_by_timerange(self, start_time: datetime, end_time: datetime) -> List[Any]:
        """Get items within time range"""
        with self.lock:
            filtered_items = [
                item for item in self.data
                if start_time <= item['timestamp'] <= end_time
            ]
            
            # Update access counts
            for item in filtered_items:
                item['access_count'] += 1
            
            self.access_count += 1
            return [item['data'] for item in filtered_items]
    
    def size(self) -> int:
        """Get current buffer size"""
        with self.lock:
            return len(self.data)
    
    def clear(self) -> None:
        """Clear all data from buffer"""
        with self.lock:
            removed_count = len(self.data)
            self.data.clear()
            self.total_removed += removed_count
            logger.debug(f"Buffer {self.name} cleared: {removed_count} items removed")
    
    def _cleanup_old_data(self) -> None:
        """Remove old data based on retention policy"""
        if datetime.now() - self.last_cleanup < timedelta(minutes=5):
            return  # Don't cleanup too frequently
        
        cutoff_time = datetime.now() - timedelta(hours=self.config.retention_hours)
        original_size = len(self.data)
        
        # Remove old items
        while self.data and self.data[0]['timestamp'] < cutoff_time:
            self.data.popleft()
            self.total_removed += 1
        
        # Remove least accessed items if still over threshold
        if len(self.data) >= self.config.max_size * 0.9:
            # Sort by access count and remove least accessed
            sorted_data = sorted(self.data, key=lambda x: x['access_count'])
            items_to_remove = int(len(self.data) * 0.2)  # Remove 20%
            
            for _ in range(items_to_remove):
                if sorted_data:
                    item_to_remove = sorted_data.pop(0)
                    try:
                        self.data.remove(item_to_remove)
                        self.total_removed += 1
                    except ValueError:
                        pass  # Item already removed
        
        removed_count = original_size - len(self.data)
        if removed_count > 0:
            logger.debug(f"Buffer {self.name} cleanup: {removed_count} items removed")
        
        self.last_cleanup = datetime.now()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self.lock:
            return {
                'name': self.name,
                'current_size': len(self.data),
                'max_size': self.config.max_size,
                'utilization': len(self.data) / self.config.max_size,
                'total_added': self.total_added,
                'total_removed': self.total_removed,
                'access_count': self.access_count,
                'last_cleanup': self.last_cleanup.isoformat(),
                'oldest_item': self.data[0]['timestamp'].isoformat() if self.data else None,
                'newest_item': self.data[-1]['timestamp'].isoformat() if self.data else None
            }


class MonitoringResourceManager:
    """Resource manager for monitoring system data and operations"""
    
    def __init__(self, max_buffer_size: int = 10000):
        self.max_buffer_size = max_buffer_size
        self.buffers: Dict[str, DataBuffer] = {}
        self.resource_usage_history: deque = deque(maxlen=100)
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="monitoring_")
        
        # Resource monitoring
        self.memory_threshold_mb = 1000  # 1GB threshold
        self.cpu_threshold_percent = 80.0
        self.cleanup_interval_minutes = 15
        
        # Statistics
        self.total_cleanup_operations = 0
        self.last_resource_check = datetime.now()
        
        # Start background resource monitoring
        self._start_resource_monitoring()
        
        logger.info(f"Monitoring resource manager initialized with max buffer size: {max_buffer_size}")
    
    async def start(self) -> None:
        """Start the resource manager"""
        logger.info("Monitoring resource manager started")
    
    async def stop(self) -> None:
        """Stop the resource manager and cleanup resources"""
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Clear all buffers
        for buffer in self.buffers.values():
            buffer.clear()
        
        logger.info("Monitoring resource manager stopped")
    
    def get_buffer(self, name: str, config: Optional[BufferConfig] = None) -> DataBuffer:
        """Get or create a data buffer"""
        
        if name not in self.buffers:
            buffer_config = config or BufferConfig(max_size=self.max_buffer_size)
            self.buffers[name] = DataBuffer(name, buffer_config)
            logger.debug(f"Created new buffer: {name}")
        
        return self.buffers[name]
    
    async def add_prediction(self, model_name: str, prediction_data: Dict[str, Any], features: Optional[np.ndarray] = None) -> None:
        """Add prediction data to model buffer"""
        
        # Get or create buffers
        prediction_buffer = self.get_buffer(f"{model_name}_predictions")
        
        # Add prediction data
        prediction_buffer.add(prediction_data)
        
        # Add features if provided
        if features is not None:
            feature_buffer = self.get_buffer(f"{model_name}_features")
            feature_buffer.add(features)
        
        # Check resource usage periodically
        if datetime.now() - self.last_resource_check > timedelta(minutes=5):
            await self._check_resource_usage()
    
    @asynccontextmanager
    async def get_model_data(self, model_name: str) -> AsyncContextManager[Dict[str, Any]]:
        """Get model data with automatic resource management"""
        
        try:
            # Get data from buffers
            prediction_buffer = self.get_buffer(f"{model_name}_predictions")
            feature_buffer = self.get_buffer(f"{model_name}_features")
            
            model_data = {
                'predictions': prediction_buffer.get_recent(1000),  # Last 1000 predictions
                'features': feature_buffer.get_recent(1000),       # Last 1000 feature sets
                'prediction_count': prediction_buffer.size(),
                'feature_count': feature_buffer.size()
            }
            
            yield model_data
            
        except Exception as e:
            logger.error(f"Error accessing model data for {model_name}: {e}")
            yield {'predictions': [], 'features': [], 'prediction_count': 0, 'feature_count': 0}
        
        finally:
            # Cleanup if needed
            await self._cleanup_if_needed()
    
    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Calculate buffer memory usage
        buffer_memory = 0
        buffer_stats = {}
        
        for name, buffer in self.buffers.items():
            stats = buffer.get_stats()
            buffer_stats[name] = stats
            # Rough estimate of memory usage
            buffer_memory += stats['current_size'] * 1024  # Assume 1KB per item
        
        return {
            'process_memory_mb': memory_info.rss / (1024 * 1024),
            'buffer_memory_kb': buffer_memory,
            'buffer_count': len(self.buffers),
            'buffer_stats': buffer_stats,
            'total_predictions': sum(b.total_added for b in self.buffers.values() if 'predictions' in b.name),
            'total_features': sum(b.total_added for b in self.buffers.values() if 'features' in b.name)
        }
    
    async def cleanup_old_data(self, hours: int = 24) -> Dict[str, int]:
        """Cleanup old data from all buffers"""
        
        cleanup_stats = {}
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        for name, buffer in self.buffers.items():
            original_size = buffer.size()
            
            # Force cleanup of old data
            buffer._cleanup_old_data()
            
            cleaned_count = original_size - buffer.size()
            cleanup_stats[name] = cleaned_count
            
            if cleaned_count > 0:
                logger.info(f"Cleaned {cleaned_count} old items from buffer: {name}")
        
        self.total_cleanup_operations += 1
        
        # Force garbage collection
        gc.collect()
        
        return cleanup_stats
    
    async def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage by cleaning up and compacting data"""
        
        optimization_stats = {
            'buffers_optimized': 0,
            'memory_freed_mb': 0,
            'items_removed': 0
        }
        
        # Get initial memory usage
        initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Optimize each buffer
        for name, buffer in self.buffers.items():
            original_size = buffer.size()
            
            # Remove least accessed items if buffer is large
            if original_size > buffer.config.max_size * 0.7:
                buffer._cleanup_old_data()
                optimization_stats['buffers_optimized'] += 1
                optimization_stats['items_removed'] += original_size - buffer.size()
        
        # Force garbage collection
        gc.collect()
        
        # Calculate memory freed
        final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        optimization_stats['memory_freed_mb'] = max(0, initial_memory - final_memory)
        
        logger.info(f"Memory optimization completed: {optimization_stats}")
        
        return optimization_stats
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get comprehensive resource statistics"""
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Buffer statistics
        buffer_stats = {}
        total_items = 0
        total_capacity = 0
        
        for name, buffer in self.buffers.items():
            stats = buffer.get_stats()
            buffer_stats[name] = stats
            total_items += stats['current_size']
            total_capacity += stats['max_size']
        
        # Recent resource usage
        recent_usage = list(self.resource_usage_history)[-10:] if self.resource_usage_history else []
        
        return {
            'timestamp': datetime.now().isoformat(),
            'memory': {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'percent': psutil.virtual_memory().percent
            },
            'cpu': {
                'percent': psutil.cpu_percent(interval=1),
                'count': psutil.cpu_count()
            },
            'buffers': {
                'count': len(self.buffers),
                'total_items': total_items,
                'total_capacity': total_capacity,
                'utilization': total_items / total_capacity if total_capacity > 0 else 0,
                'stats': buffer_stats
            },
            'operations': {
                'cleanup_operations': self.total_cleanup_operations,
                'last_resource_check': self.last_resource_check.isoformat()
            },
            'recent_usage': recent_usage
        }
    
    async def _check_resource_usage(self) -> None:
        """Check and record resource usage"""
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get buffer sizes
            buffer_sizes = {name: buffer.size() for name, buffer in self.buffers.items()}
            
            # Create resource usage record
            usage = ResourceUsage(
                timestamp=datetime.now(),
                memory_mb=memory_mb,
                cpu_percent=cpu_percent,
                buffer_sizes=buffer_sizes,
                total_predictions=sum(b.total_added for b in self.buffers.values() if 'predictions' in b.name),
                total_features=sum(b.total_added for b in self.buffers.values() if 'features' in b.name),
                cleanup_operations=self.total_cleanup_operations
            )
            
            self.resource_usage_history.append(usage)
            self.last_resource_check = datetime.now()
            
            # Check if cleanup is needed
            if memory_mb > self.memory_threshold_mb or cpu_percent > self.cpu_threshold_percent:
                logger.warning(f"High resource usage detected: Memory={memory_mb:.1f}MB, CPU={cpu_percent:.1f}%")
                await self._cleanup_if_needed()
            
        except Exception as e:
            logger.error(f"Error checking resource usage: {e}")
    
    async def _cleanup_if_needed(self) -> None:
        """Perform cleanup if resource usage is high"""
        
        try:
            memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            
            if memory_mb > self.memory_threshold_mb:
                logger.info(f"Performing cleanup due to high memory usage: {memory_mb:.1f}MB")
                
                # Cleanup old data
                await self.cleanup_old_data(hours=12)  # More aggressive cleanup
                
                # Optimize memory
                await self.optimize_memory()
                
                self.total_cleanup_operations += 1
        
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _start_resource_monitoring(self) -> None:
        """Start background resource monitoring"""
        
        async def monitor_loop():
            while True:
                try:
                    await self._check_resource_usage()
                    await asyncio.sleep(300)  # Check every 5 minutes
                except Exception as e:
                    logger.error(f"Error in resource monitoring loop: {e}")
                    await asyncio.sleep(60)  # Shorter sleep on error
        
        # Start monitoring task
        asyncio.create_task(monitor_loop())
        logger.info("Background resource monitoring started")
    
    def clear_all_buffers(self) -> None:
        """Clear all data buffers"""
        
        total_cleared = 0
        for buffer in self.buffers.values():
            total_cleared += buffer.size()
            buffer.clear()
        
        logger.info(f"All buffers cleared: {total_cleared} items removed")
    
    def get_buffer_names(self) -> List[str]:
        """Get list of all buffer names"""
        return list(self.buffers.keys())
    
    def remove_buffer(self, name: str) -> bool:
        """Remove a specific buffer"""
        
        if name in self.buffers:
            self.buffers[name].clear()
            del self.buffers[name]
            logger.info(f"Buffer removed: {name}")
            return True
        
        return False