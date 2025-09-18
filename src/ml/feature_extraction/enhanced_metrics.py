"""Enhanced metrics collection for feature extraction with real-time monitoring.

This module provides enhanced metrics collection that integrates with
the existing monitoring infrastructure while providing specialized
metrics for feature extraction operations.
"""

import time
import psutil
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque
import asyncio
import logging

from src.utils.monitoring import get_metrics_collector
from src.ml.feature_extraction.monitoring import FeatureExtractionMetrics

logger = logging.getLogger(__name__)


@dataclass
class RealTimeMetrics:
    """Real-time metrics for feature extraction"""
    timestamp: datetime
    extraction_rate: float # extractions per second
    avg_latency_ms: float
    memory_usage_mb: float
    cpu_percent: float
    gpu_utilization: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0


class EnhancedMetricsCollector:
    """Enhanced metrics collector for feature extraction with real-time capabilities"""
    
    def __init__(self, window_size: int = 1000):
        """Initialize enhanced metrics collector.
        
        Args:
            window_size: Size of sliding window for metrics calculation
        """
        self.window_size = window_size
        self.metrics_history: deque = deque(maxlen=window_size)
        self.metrics_collector = get_metrics_collector()
        self._lock = asyncio.Lock()
        
        # System metrics tracking
        self._last_collection_time = time.time()
        self._last_extraction_count = 0
        
        logger.info("Enhanced metrics collector initialized")
    
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics.
        
        Returns:
            Dictionary with system metrics
        """
        metrics = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_mb': psutil.virtual_memory().used / (1024 * 1024),
            'disk_usage_percent': psutil.disk_usage('/').percent
        }
        
        # GPU metrics if available
        if torch.cuda.is_available():
            try:
                metrics['gpu_utilization'] = torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else 0
                metrics['gpu_memory_mb'] = torch.cuda.memory_allocated(0) / (1024 * 1024)
            except Exception:
                metrics['gpu_utilization'] = 0
                metrics['gpu_memory_mb'] = 0
        
        return metrics
    
    def add_feature_extraction_metrics(self, metrics: FeatureExtractionMetrics) -> None:
        """Add feature extraction metrics to history.
        
        Args:
            metrics: Feature extraction metrics to add
        """
        self.metrics_history.append(metrics)
        
        # Record to general metrics collector
        if self.metrics_collector:
            self.metrics_collector.record_timer(
                "feature_extraction_duration_ms",
                metrics.duration_ms
            )
            
            self.metrics_collector.set_gauge(
                "feature_extraction_memory_mb",
                metrics.memory_mb
            )
            
            self.metrics_collector.set_gauge(
                "feature_extraction_cpu_percent",
                metrics.cpu_percent
            )
            
            if metrics.gpu_utilization > 0:
                self.metrics_collector.set_gauge(
                    "feature_extraction_gpu_utilization",
                    metrics.gpu_utilization
                )
    
    def get_real_time_metrics(self) -> RealTimeMetrics:
        """Get real-time metrics for feature extraction.
        
        Returns:
            Real-time metrics
        """
        if not self.metrics_history:
            system_metrics = self.collect_system_metrics()
            return RealTimeMetrics(
                timestamp=datetime.now(),
                extraction_rate=0.0,
                avg_latency_ms=0.0,
                memory_usage_mb=system_metrics.get('memory_mb', 0),
                cpu_percent=system_metrics.get('cpu_percent', 0),
                gpu_utilization=system_metrics.get('gpu_utilization', 0)
            )
        
        # Get recent metrics (last 100)
        recent_metrics = list(self.metrics_history)[-100:]
        
        # Calculate rates and averages
        current_time = time.time()
        time_delta = current_time - self._last_collection_time
        extraction_count = len(self.metrics_history)
        extraction_delta = extraction_count - self._last_extraction_count
        
        extraction_rate = extraction_delta / time_delta if time_delta > 0 else 0
        
        avg_latency_ms = np.mean([m.duration_ms for m in recent_metrics])
        memory_usage_mb = np.mean([m.memory_mb for m in recent_metrics])
        cpu_percent = np.mean([m.cpu_percent for m in recent_metrics])
        gpu_utilization = np.mean([m.gpu_utilization for m in recent_metrics])
        
        cache_hits = sum(1 for m in recent_metrics if m.used_cache)
        cache_hit_rate = cache_hits / len(recent_metrics) if recent_metrics else 0
        
        errors = sum(1 for m in recent_metrics if m.had_error)
        error_rate = errors / len(recent_metrics) if recent_metrics else 0
        
        # Update counters
        self._last_collection_time = current_time
        self._last_extraction_count = extraction_count
        
        # Collect current system metrics
        system_metrics = self.collect_system_metrics()
        
        return RealTimeMetrics(
            timestamp=datetime.now(),
            extraction_rate=extraction_rate,
            avg_latency_ms=avg_latency_ms,
            memory_usage_mb=system_metrics.get('memory_mb', memory_usage_mb),
            cpu_percent=system_metrics.get('cpu_percent', cpu_percent),
            gpu_utilization=system_metrics.get('gpu_utilization', gpu_utilization),
            cache_hit_rate=cache_hit_rate,
            error_rate=error_rate
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary.
        
        Returns:
            Dictionary with performance summary
        """
        if not self.metrics_history:
            return {
                'total_extractions': 0,
                'avg_latency_ms': 0,
                'p95_latency_ms': 0,
                'p99_latency_ms': 0,
                'max_latency_ms': 0,
                'min_latency_ms': 0,
                'cache_hit_rate': 0,
                'error_rate': 0,
                'throughput_extractions_per_second': 0
            }
        
        durations = [m.duration_ms for m in self.metrics_history]
        cache_hits = sum(1 for m in self.metrics_history if m.used_cache)
        errors = sum(1 for m in self.metrics_history if m.had_error)
        
        # Calculate time-based metrics
        if self.metrics_history:
            first_time = self.metrics_history[0].timestamp.timestamp()
            last_time = self.metrics_history[-1].timestamp.timestamp()
            time_span = last_time - first_time
            throughput = len(self.metrics_history) / time_span if time_span > 0 else 0
        else:
            throughput = 0
        
        return {
            'total_extractions': len(self.metrics_history),
            'avg_latency_ms': np.mean(durations),
            'median_latency_ms': np.median(durations),
            'p95_latency_ms': np.percentile(durations, 95) if len(durations) >= 20 else 0,
            'p99_latency_ms': np.percentile(durations, 99) if len(durations) >= 100 else 0,
            'max_latency_ms': np.max(durations),
            'min_latency_ms': np.min(durations),
            'cache_hit_rate': cache_hits / len(self.metrics_history),
            'error_rate': errors / len(self.metrics_history),
            'throughput_extractions_per_second': throughput,
            'peak_memory_mb': np.max([m.memory_mb for m in self.metrics_history]),
            'avg_memory_mb': np.mean([m.memory_mb for m in self.metrics_history])
        }
    
    def meets_performance_requirements(self) -> Dict[str, Any]:
        """Check if performance requirements are met.
        
        Returns:
            Dictionary with performance requirement status
        """
        summary = self.get_performance_summary()
        
        return {
            'meets_100ms_requirement': summary.get('p95_latency_ms', 0) < 100,
            'meets_50ms_target': summary.get('avg_latency_ms', 0) < 50,
            'current_avg_latency_ms': summary.get('avg_latency_ms', 0),
            'current_p95_latency_ms': summary.get('p95_latency_ms', 0),
            'target_latency_ms': 100,
            'warning_latency_ms': 50,
            'latency_margin_ms': 100 - summary.get('p95_latency_ms', 0),
            'cache_hit_rate': summary.get('cache_hit_rate', 0),
            'min_required_cache_hit_rate': 0.7,
            'meets_cache_requirement': summary.get('cache_hit_rate', 0) > 0.7
        }
    
    def get_resource_utilization(self) -> Dict[str, Any]:
        """Get current resource utilization.
        
        Returns:
            Dictionary with resource utilization metrics
        """
        system_metrics = self.collect_system_metrics()
        
        if self.metrics_history:
            recent_metrics = list(self.metrics_history)[-100:]
            avg_memory_mb = np.mean([m.memory_mb for m in recent_metrics])
            peak_memory_mb = np.max([m.memory_mb for m in recent_metrics])
        else:
            avg_memory_mb = system_metrics.get('memory_mb', 0)
            peak_memory_mb = avg_memory_mb
        
        return {
            'cpu_percent': system_metrics.get('cpu_percent', 0),
            'memory_mb': avg_memory_mb,
            'peak_memory_mb': peak_memory_mb,
            'disk_usage_percent': system_metrics.get('disk_usage_percent', 0),
            'gpu_utilization': system_metrics.get('gpu_utilization', 0),
            'gpu_memory_mb': system_metrics.get('gpu_memory_mb', 0)
        }
    
    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        self.metrics_history.clear()
        self._last_collection_time = time.time()
        self._last_extraction_count = 0
        logger.info("Enhanced metrics collector reset")