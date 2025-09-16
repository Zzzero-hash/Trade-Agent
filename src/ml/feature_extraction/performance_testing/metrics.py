"""Performance metrics collection and analysis for feature extraction.

This module provides comprehensive metrics collection, analysis, and
reporting capabilities for feature extraction performance testing.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import psutil
import torch

from .framework import PerformanceTestResult
from .load_testing import LoadTestResult
from .stress_testing import StressTestResult

logger = logging.getLogger(__name__)


@dataclass
class LatencyStats:
    """Latency statistics."""
    avg_ms: float = 0.0
    median_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    std_dev_ms: float = 0.0


@dataclass
class ResourceStats:
    """Resource usage statistics."""
    avg_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    peak_cpu_percent: float = 0.0
    gpu_utilization_percent: float = 0.0
    gpu_memory_mb: float = 0.0


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    # Test metadata
    test_name: str
    test_type: str # performance, load, stress
    timestamp: datetime
    duration_seconds: float
    
    # Performance metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    success_rate: float = 0.0
    
    # Latency metrics
    latency_stats: LatencyStats = field(default_factory=LatencyStats)
    
    # Throughput metrics
    throughput_rps: float = 0.0
    
    # Resource metrics
    resource_stats: ResourceStats = field(default_factory=ResourceStats)
    
    # Requirements validation
    meets_latency_requirement: bool = False
    meets_throughput_requirement: bool = False
    meets_resource_requirement: bool = False
    
    # Test-specific metrics
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of performance report."""
        return {
            'test_name': self.test_name,
            'test_type': self.test_type,
            'timestamp': self.timestamp.isoformat(),
            'duration_seconds': self.duration_seconds,
            'total_requests': self.total_requests,
            'success_rate': self.success_rate,
            'avg_latency_ms': self.latency_stats.avg_ms,
            'p95_latency_ms': self.latency_stats.p95_ms,
            'throughput_rps': self.throughput_rps,
            'peak_memory_mb': self.resource_stats.peak_memory_mb,
            'meets_requirements': (
                self.meets_latency_requirement and 
                self.meets_throughput_requirement and 
                self.meets_resource_requirement
            )
        }


class PerformanceMetricsCollector:
    """Collector for performance metrics from various test types."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.logger = logging.getLogger(__name__)
    
    def collect_from_performance_test(self, result: PerformanceTestResult) -> PerformanceReport:
        """Collect metrics from performance test result.
        
        Args:
            result: Performance test result
            
        Returns:
            Performance report
        """
        # Calculate latency statistics
        latency_stats = LatencyStats()
        if result.latencies_ms:
            latencies = np.array(result.latencies_ms)
            latency_stats = LatencyStats(
                avg_ms=float(np.mean(latencies)),
                median_ms=float(np.median(latencies)),
                min_ms=float(np.min(latencies)),
                max_ms=float(np.max(latencies)),
                p95_ms=float(np.percentile(latencies, 95)),
                p99_ms=float(np.percentile(latencies, 9)),
                std_dev_ms=float(np.std(latencies))
            )
        
        # Calculate resource statistics
        resource_stats = ResourceStats()
        if result.memory_usage_mb or result.cpu_usage_percent:
            if result.memory_usage_mb:
                memory_array = np.array(result.memory_usage_mb)
                resource_stats.avg_memory_mb = float(np.mean(memory_array))
                resource_stats.peak_memory_mb = float(np.max(memory_array))
            
            if result.cpu_usage_percent:
                cpu_array = np.array(result.cpu_usage_percent)
                resource_stats.avg_cpu_percent = float(np.mean(cpu_array))
                resource_stats.peak_cpu_percent = float(np.max(cpu_array))
        
        # Calculate throughput
        throughput_rps = 0.0
        if result.total_duration_seconds > 0:
            throughput_rps = result.successful_extractions / result.total_duration_seconds
        
        # Calculate success rate
        total_requests = result.successful_extractions + result.failed_extractions
        success_rate = 0.0
        if total_requests > 0:
            success_rate = result.successful_extractions / total_requests
        
        # Check requirements
        meets_latency_requirement = latency_stats.p95_ms < result.config.target_latency_ms
        meets_throughput_requirement = throughput_rps >= 10.0  # Minimum throughput requirement
        meets_resource_requirement = resource_stats.peak_memory_mb < result.config.max_memory_mb
        
        return PerformanceReport(
            test_name=result.test_name,
            test_type="performance",
            timestamp=result.test_timestamp,
            duration_seconds=result.total_duration_seconds,
            total_requests=total_requests,
            successful_requests=result.successful_extractions,
            failed_requests=result.failed_extractions,
            success_rate=success_rate,
            latency_stats=latency_stats,
            throughput_rps=throughput_rps,
            resource_stats=resource_stats,
            meets_latency_requirement=meets_latency_requirement,
            meets_throughput_requirement=meets_throughput_requirement,
            meets_resource_requirement=meets_resource_requirement,
            additional_metrics={
                'cache_hits': result.cache_hits,
                'fallback_uses': result.fallback_uses,
                'warmup_duration': result.warmup_duration_seconds
            }
        )
    
    def collect_from_load_test(self, result: LoadTestResult) -> PerformanceReport:
        """Collect metrics from load test result.
        
        Args:
            result: Load test result
            
        Returns:
            Performance report
        """
        # Calculate latency statistics
        latency_stats = LatencyStats()
        if result.request_latencies_ms:
            latencies = np.array(result.request_latencies_ms)
            latency_stats = LatencyStats(
                avg_ms=float(np.mean(latencies)),
                median_ms=float(np.median(latencies)),
                min_ms=float(np.min(latencies)),
                max_ms=float(np.max(latencies)),
                p95_ms=float(np.percentile(latencies, 95)),
                p99_ms=float(np.percentile(latencies, 99)),
                std_dev_ms=float(np.std(latencies))
            )
        
        # Calculate resource statistics
        resource_stats = ResourceStats(
            avg_memory_mb=result.avg_memory_mb,
            peak_memory_mb=result.peak_memory_mb,
            avg_cpu_percent=result.avg_cpu_percent,
            peak_cpu_percent=result.peak_cpu_percent
        )
        
        # Calculate throughput
        throughput_rps = result.get_throughput_rps()
        
        # Calculate success rate
        success_rate = 0.0
        if result.total_requests > 0:
            success_rate = result.successful_requests / result.total_requests
        
        # Check requirements
        meets_latency_requirement = latency_stats.p95_ms < result.config.target_response_time_ms
        meets_throughput_requirement = throughput_rps >= result.config.target_throughput_rps
        meets_resource_requirement = (
            resource_stats.peak_memory_mb < result.config.max_memory_mb and
            resource_stats.peak_cpu_percent < result.config.max_cpu_percent
        )
        
        return PerformanceReport(
            test_name=result.test_name,
            test_type="load",
            timestamp=result.test_timestamp,
            duration_seconds=result.end_time - result.start_time if result.end_time > result.start_time else 0,
            total_requests=result.total_requests,
            successful_requests=result.successful_requests,
            failed_requests=result.failed_requests,
            success_rate=success_rate,
            latency_stats=latency_stats,
            throughput_rps=throughput_rps,
            resource_stats=resource_stats,
            meets_latency_requirement=meets_latency_requirement,
            meets_throughput_requirement=meets_throughput_requirement,
            meets_resource_requirement=meets_resource_requirement,
            additional_metrics={
                'user_count': len(result.user_metrics),
                'error_rate': result.get_error_rate()
            }
        )
    
    def collect_from_stress_test(self, result: StressTestResult) -> PerformanceReport:
        """Collect metrics from stress test result.
        
        Args:
            result: Stress test result
            
        Returns:
            Performance report
        """
        # Calculate latency statistics
        latency_stats = LatencyStats()
        if result.request_latencies_ms:
            latencies = np.array(result.request_latencies_ms)
            latency_stats = LatencyStats(
                avg_ms=float(np.mean(latencies)),
                median_ms=float(np.median(latencies)),
                min_ms=float(np.min(latencies)),
                max_ms=float(np.max(latencies)),
                p95_ms=float(np.percentile(latencies, 95)),
                p99_ms=float(np.percentile(latencies, 9)),
                std_dev_ms=float(np.std(latencies))
            )
        
        # Calculate resource statistics
        resource_stats = ResourceStats(
            peak_memory_mb=result.peak_memory_mb,
            avg_cpu_percent=result.peak_cpu_percent / 2,  # Approximate average
            peak_cpu_percent=result.peak_cpu_percent
        )
        
        # Add GPU metrics if available
        if torch.cuda.is_available():
            try:
                resource_stats.gpu_utilization_percent = torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else 0
                resource_stats.gpu_memory_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
            except Exception:
                pass  # Ignore GPU metrics if not available
        
        # Calculate throughput
        throughput_rps = result.get_throughput_rps()
        
        # Calculate success rate
        success_rate = 0.0
        if result.total_requests > 0:
            success_rate = result.successful_requests / result.total_requests
        
        # Check requirements
        meets_latency_requirement = latency_stats.p95_ms < result.config.max_acceptable_latency_ms
        meets_throughput_requirement = throughput_rps >= result.config.min_required_throughput_rps
        meets_resource_requirement = (
            resource_stats.peak_memory_mb < result.config.critical_memory_mb and
            resource_stats.peak_cpu_percent < result.config.critical_cpu_percent
        )
        
        return PerformanceReport(
            test_name=result.test_name,
            test_type="stress",
            timestamp=result.test_timestamp,
            duration_seconds=result.end_time - result.start_time if result.end_time > result.start_time else 0,
            total_requests=result.total_requests,
            successful_requests=result.successful_requests,
            failed_requests=result.failed_requests,
            success_rate=success_rate,
            latency_stats=latency_stats,
            throughput_rps=throughput_rps,
            resource_stats=resource_stats,
            meets_latency_requirement=meets_latency_requirement,
            meets_throughput_requirement=meets_throughput_requirement,
            meets_resource_requirement=meets_resource_requirement,
            additional_metrics={
                'failure_points': len(result.failure_points),
                'memory_growth_mb': result.memory_growth_mb,
                'error_rate': result.get_error_rate()
            }
        )
    
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics.
        
        Returns:
            Dictionary with system metrics
        """
        metrics = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_mb': psutil.virtual_memory().available / (1024 * 1024),
            'disk_usage_percent': psutil.disk_usage('/').percent
        }
        
        # Add GPU metrics if available
        if torch.cuda.is_available():
            try:
                metrics['gpu_utilization_percent'] = torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else 0
                metrics['gpu_memory_mb'] = torch.cuda.memory_allocated(0) / (1024 * 1024)
                metrics['gpu_memory_cached_mb'] = torch.cuda.memory_reserved(0) / (1024 * 1024)
            except Exception:
                pass  # Ignore GPU metrics if not available
        
        return metrics
    
    def compare_test_results(self, reports: List[PerformanceReport]) -> Dict[str, Any]:
        """Compare multiple test results.
        
        Args:
            reports: List of performance reports
            
        Returns:
            Comparison results
        """
        if not reports:
            return {}
        
        comparison = {
            'test_count': len(reports),
            'tests': [],
            'summary': {
                'all_meet_requirements': all(r.meets_latency_requirement and r.meets_throughput_requirement for r in reports),
                'avg_success_rate': np.mean([r.success_rate for r in reports]),
                'avg_latency_ms': np.mean([r.latency_stats.avg_ms for r in reports]),
                'avg_throughput_rps': np.mean([r.throughput_rps for r in reports])
            }
        }
        
        # Add individual test details
        for report in reports:
            comparison['tests'].append({
                'test_name': report.test_name,
                'test_type': report.test_type,
                'success_rate': report.success_rate,
                'avg_latency_ms': report.latency_stats.avg_ms,
                'p95_latency_ms': report.latency_stats.p95_ms,
                'throughput_rps': report.throughput_rps,
                'meets_requirements': (
                    report.meets_latency_requirement and 
                    report.meets_throughput_requirement and 
                    report.meets_resource_requirement
                )
            })
        
        return comparison