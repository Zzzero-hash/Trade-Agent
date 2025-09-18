"""
Performance optimization utilities for the AI trading platform.

This module provides utilities for memory management, caching,
and performance monitoring to optimize the trading system.
"""

import gc
import time
import psutil
import functools
import threading
from typing import Any, Callable, Dict, Optional, Union, List
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from contextlib import contextmanager
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    peak_memory_mb: Optional[float] = None
    cache_hits: int = 0
    cache_misses: int = 0


class LRUCache:
    """Thread-safe LRU cache implementation with size limits."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: Optional[float] = None):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items to cache
            ttl_seconds: Time-to-live for cache entries (None for no expiration)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {} if ttl_seconds else None
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: Any) -> Any:
        """Get item from cache."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            # Check TTL if enabled
            if self.timestamps and key in self.timestamps:
                if time.time() - self.timestamps[key] > self.ttl_seconds:
                    del self.cache[key]
                    del self.timestamps[key]
                    self.misses += 1
                    return None
            
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            self.hits += 1
            return value
    
    def put(self, key: Any, value: Any) -> None:
        """Put item in cache."""
        with self.lock:
            if key in self.cache:
                # Update existing item
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used item
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                if self.timestamps and oldest_key in self.timestamps:
                    del self.timestamps[oldest_key]
            
            self.cache[key] = value
            if self.timestamps is not None:
                self.timestamps[key] = time.time()
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            if self.timestamps:
                self.timestamps.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'ttl_seconds': self.ttl_seconds
            }


class MemoryManager:
    """Memory management utilities."""
    
    @staticmethod
    def get_memory_usage() -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def get_cpu_usage() -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)
    
    @staticmethod
    def force_garbage_collection() -> Dict[str, int]:
        """Force garbage collection and return statistics."""
        collected = {}
        for generation in range(3):
            collected[f'gen_{generation}'] = gc.collect(generation)
        
        return collected
    
    @staticmethod
    def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage by downcasting numeric types."""
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype
            
            if col_type != 'object':
                c_min = optimized_df[col].min()
                c_max = optimized_df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        optimized_df[col] = optimized_df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        optimized_df[col] = optimized_df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        optimized_df[col] = optimized_df[col].astype(np.int32)
                
                elif str(col_type)[:5] == 'float':
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        optimized_df[col] = optimized_df[col].astype(np.float32)
        
        return optimized_df
    
    @staticmethod
    def get_object_size(obj: Any) -> int:
        """Get approximate size of object in bytes."""
        import sys
        
        if isinstance(obj, pd.DataFrame):
            return obj.memory_usage(deep=True).sum()
        elif isinstance(obj, np.ndarray):
            return obj.nbytes
        else:
            return sys.getsizeof(obj)


class PerformanceMonitor:
    """Performance monitoring and profiling utilities."""
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.active_timers = {}
    
    @contextmanager
    def monitor_performance(self, operation_name: str):
        """Context manager for monitoring operation performance."""
        start_time = time.time()
        start_memory = MemoryManager.get_memory_usage()
        start_cpu = MemoryManager.get_cpu_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = MemoryManager.get_memory_usage()
            end_cpu = MemoryManager.get_cpu_usage()
            
            metrics = PerformanceMetrics(
                execution_time=end_time - start_time,
                memory_usage_mb=end_memory - start_memory,
                cpu_usage_percent=(start_cpu + end_cpu) / 2
            )
            
            self.metrics_history[operation_name].append(metrics)
            
            logger.debug(
                f"Performance - {operation_name}: "
                f"Time={metrics.execution_time:.3f}s, "
                f"Memory={metrics.memory_usage_mb:.1f}MB, "
                f"CPU={metrics.cpu_usage_percent:.1f}%"
            )
    
    def get_performance_summary(self, operation_name: str) -> Dict[str, float]:
        """Get performance summary for an operation."""
        if operation_name not in self.metrics_history:
            return {}
        
        metrics_list = self.metrics_history[operation_name]
        
        execution_times = [m.execution_time for m in metrics_list]
        memory_usages = [m.memory_usage_mb for m in metrics_list]
        cpu_usages = [m.cpu_usage_percent for m in metrics_list]
        
        return {
            'count': len(metrics_list),
            'avg_execution_time': np.mean(execution_times),
            'max_execution_time': np.max(execution_times),
            'min_execution_time': np.min(execution_times),
            'avg_memory_usage': np.mean(memory_usages),
            'max_memory_usage': np.max(memory_usages),
            'avg_cpu_usage': np.mean(cpu_usages),
            'max_cpu_usage': np.max(cpu_usages)
        }
    
    def clear_history(self, operation_name: Optional[str] = None) -> None:
        """Clear performance history."""
        if operation_name:
            self.metrics_history.pop(operation_name, None)
        else:
            self.metrics_history.clear()


def performance_cache(
    max_size: int = 1000, 
    ttl_seconds: Optional[float] = None
):
    """Decorator for caching function results with performance monitoring."""
    def decorator(func: Callable) -> Callable:
        cache = LRUCache(max_size=max_size, ttl_seconds=ttl_seconds)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = str(args) + str(sorted(kwargs.items()))
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.put(key, result)
            
            return result
        
        # Add cache management methods
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = cache.get_stats
        
        return wrapper
    
    return decorator


def memory_efficient_batch_processor(
    data: Union[pd.DataFrame, np.ndarray, List],
    batch_size: int = 1000,
    process_func: Callable = None
) -> List[Any]:
    """
    Process large datasets in memory-efficient batches.
    
    Args:
        data: Data to process
        batch_size: Size of each batch
        process_func: Function to apply to each batch
        
    Returns:
        List of processed results
    """
    if process_func is None:
        process_func = lambda x: x
    
    results = []
    
    if isinstance(data, pd.DataFrame):
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i + batch_size]
            result = process_func(batch)
            results.append(result)
            
            # Force garbage collection periodically
            if i % (batch_size * 10) == 0:
                gc.collect()
    
    elif isinstance(data, np.ndarray):
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            result = process_func(batch)
            results.append(result)
            
            if i % (batch_size * 10) == 0:
                gc.collect()
    
    elif isinstance(data, list):
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            result = process_func(batch)
            results.append(result)
            
            if i % (batch_size * 10) == 0:
                gc.collect()
    
    return results


class DataFrameOptimizer:
    """Utilities for optimizing DataFrame operations."""
    
    @staticmethod
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for memory efficiency."""
        optimized = df.copy()
        
        # Optimize numeric columns
        for col in optimized.select_dtypes(include=[np.number]).columns:
            col_min = optimized[col].min()
            col_max = optimized[col].max()
            
            if optimized[col].dtype == 'int64':
                if col_min >= -128 and col_max <= 127:
                    optimized[col] = optimized[col].astype('int8')
                elif col_min >= -32768 and col_max <= 32767:
                    optimized[col] = optimized[col].astype('int16')
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    optimized[col] = optimized[col].astype('int32')
            
            elif optimized[col].dtype == 'float64':
                if col_min >= np.finfo(np.float32).min and col_max <= np.finfo(np.float32).max:
                    optimized[col] = optimized[col].astype('float32')
        
        # Optimize string columns to category if beneficial
        for col in optimized.select_dtypes(include=['object']).columns:
            if optimized[col].nunique() / len(optimized) < 0.5:  # Less than 50% unique
                optimized[col] = optimized[col].astype('category')
        
        return optimized
    
    @staticmethod
    def efficient_groupby_apply(
        df: pd.DataFrame, 
        groupby_cols: List[str], 
        apply_func: Callable,
        chunk_size: int = 10000
    ) -> pd.DataFrame:
        """Perform memory-efficient groupby operations on large DataFrames."""
        if len(df) <= chunk_size:
            return df.groupby(groupby_cols).apply(apply_func)
        
        results = []
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            chunk_result = chunk.groupby(groupby_cols).apply(apply_func)
            results.append(chunk_result)
            
            # Periodic garbage collection
            if i % (chunk_size * 5) == 0:
                gc.collect()
        
        return pd.concat(results, ignore_index=True)


# Global instances
performance_monitor = PerformanceMonitor()
memory_manager = MemoryManager()


# Convenience decorators
def monitor_performance(operation_name: str):
    """Decorator for monitoring function performance."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with performance_monitor.monitor_performance(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def memory_profile(func: Callable) -> Callable:
    """Decorator for profiling memory usage of functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_memory = memory_manager.get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_memory = memory_manager.get_memory_usage()
            memory_delta = end_memory - start_memory
            
            if memory_delta > 10:  # Log if memory usage > 10MB
                logger.info(
                    f"Memory usage for {func.__name__}: {memory_delta:.1f}MB"
                )
    
    return wrapper