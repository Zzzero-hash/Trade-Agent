"""
Performance Monitoring Service

Tracks latency, throughput, error rates, and system performance metrics
for production trading operations.
"""

import asyncio
import logging
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import statistics

from src.services.monitoring_service import MonitoringService
from src.services.alert_service import ProductionAlertService


@dataclass
class PerformanceMetric:
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class LatencyStats:
    min_latency: float
    max_latency: float
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    sample_count: int


@dataclass
class ThroughputStats:
    requests_per_second: float
    peak_rps: float
    avg_rps: float
    total_requests: int
    time_window_seconds: int


@dataclass
class ErrorRateStats:
    error_rate: float
    total_errors: int
    total_requests: int
    error_types: Dict[str, int]


class PerformanceTracker:
    """Tracks performance metrics for a specific operation or service."""
    
    def __init__(self, name: str, window_size: int = 1000):
        self.name = name
        self.window_size = window_size
        
        # Circular buffers for metrics
        self.latencies: deque = deque(maxlen=window_size)
        self.timestamps: deque = deque(maxlen=window_size)
        self.errors: deque = deque(maxlen=window_size)
        self.error_types: defaultdict = defaultdict(int)
        
        # Counters
        self.total_requests = 0
        self.total_errors = 0
        
        # Performance thresholds
        self.latency_threshold_ms = 1000  # 1 second
        self.error_rate_threshold = 0.05  # 5%
        self.throughput_threshold_rps = 100
    
    def record_request(self, latency_ms: float, error: Optional[str] = None):
        """Record a request with its latency and optional error."""
        now = datetime.utcnow()
        
        self.latencies.append(latency_ms)
        self.timestamps.append(now)
        self.total_requests += 1
        
        if error:
            self.errors.append(True)
            self.error_types[error] += 1
            self.total_errors += 1
        else:
            self.errors.append(False)
    
    def get_latency_stats(self) -> LatencyStats:
        """Calculate latency statistics."""
        if not self.latencies:
            return LatencyStats(0, 0, 0, 0, 0, 0, 0)
        
        latencies = list(self.latencies)
        latencies.sort()
        
        return LatencyStats(
            min_latency=min(latencies),
            max_latency=max(latencies),
            avg_latency=statistics.mean(latencies),
            p50_latency=self._percentile(latencies, 50),
            p95_latency=self._percentile(latencies, 95),
            p99_latency=self._percentile(latencies, 99),
            sample_count=len(latencies)
        )
    
    def get_throughput_stats(self, window_seconds: int = 60) -> ThroughputStats:
        """Calculate throughput statistics for the specified time window."""
        if not self.timestamps:
            return ThroughputStats(0, 0, 0, 0, window_seconds)
        
        now = datetime.utcnow()
        cutoff_time = now - timedelta(seconds=window_seconds)
        
        # Count requests in the time window
        recent_requests = sum(1 for ts in self.timestamps if ts > cutoff_time)
        
        # Calculate RPS
        current_rps = recent_requests / window_seconds if window_seconds > 0 else 0
        
        # Calculate peak RPS (using 10-second windows)
        peak_rps = self._calculate_peak_rps(10)
        
        # Calculate average RPS over all time
        if self.timestamps:
            total_time = (max(self.timestamps) - min(self.timestamps)).total_seconds()
            avg_rps = len(self.timestamps) / max(total_time, 1)
        else:
            avg_rps = 0
        
        return ThroughputStats(
            requests_per_second=current_rps,
            peak_rps=peak_rps,
            avg_rps=avg_rps,
            total_requests=self.total_requests,
            time_window_seconds=window_seconds
        )
    
    def get_error_rate_stats(self) -> ErrorRateStats:
        """Calculate error rate statistics."""
        if not self.errors:
            return ErrorRateStats(0, 0, 0, {})
        
        error_count = sum(self.errors)
        total_count = len(self.errors)
        error_rate = error_count / total_count if total_count > 0 else 0
        
        return ErrorRateStats(
            error_rate=error_rate,
            total_errors=error_count,
            total_requests=total_count,
            error_types=dict(self.error_types)
        )
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0
        
        index = (percentile / 100) * (len(data) - 1)
        if index.is_integer():
            return data[int(index)]
        else:
            lower = data[int(index)]
            upper = data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def _calculate_peak_rps(self, window_seconds: int) -> float:
        """Calculate peak RPS using sliding window."""
        if len(self.timestamps) < 2:
            return 0
        
        max_rps = 0
        timestamps = list(self.timestamps)
        
        for i in range(len(timestamps)):
            window_start = timestamps[i]
            window_end = window_start + timedelta(seconds=window_seconds)
            
            # Count requests in this window
            count = sum(1 for ts in timestamps[i:] if ts <= window_end)
            rps = count / window_seconds
            max_rps = max(max_rps, rps)
        
        return max_rps
    
    def is_healthy(self) -> Tuple[bool, List[str]]:
        """Check if performance is within healthy thresholds."""
        issues = []
        
        # Check latency
        latency_stats = self.get_latency_stats()
        if latency_stats.p95_latency > self.latency_threshold_ms:
            issues.append(f"High latency: P95 {latency_stats.p95_latency:.1f}ms > {self.latency_threshold_ms}ms")
        
        # Check error rate
        error_stats = self.get_error_rate_stats()
        if error_stats.error_rate > self.error_rate_threshold:
            issues.append(f"High error rate: {error_stats.error_rate:.2%} > {self.error_rate_threshold:.2%}")
        
        # Check throughput (if we have enough data)
        if len(self.timestamps) > 10:
            throughput_stats = self.get_throughput_stats()
            if throughput_stats.requests_per_second < self.throughput_threshold_rps:
                issues.append(f"Low throughput: {throughput_stats.requests_per_second:.1f} RPS < {self.throughput_threshold_rps} RPS")
        
        return len(issues) == 0, issues


class SystemResourceMonitor:
    """Monitor system resource usage."""
    
    def __init__(self):
        self.cpu_threshold = 80.0  # 80%
        self.memory_threshold = 85.0  # 85%
        self.disk_threshold = 90.0  # 90%
        
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=1)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percentage": memory.percent
        }
    
    def get_disk_usage(self) -> Dict[str, float]:
        """Get disk usage statistics."""
        disk = psutil.disk_usage('/')
        return {
            "total_gb": disk.total / (1024**3),
            "used_gb": disk.used / (1024**3),
            "free_gb": disk.free / (1024**3),
            "percentage": (disk.used / disk.total) * 100
        }
    
    def get_network_stats(self) -> Dict[str, int]:
        """Get network I/O statistics."""
        net_io = psutil.net_io_counters()
        return {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv
        }
    
    def is_healthy(self) -> Tuple[bool, List[str]]:
        """Check if system resources are within healthy thresholds."""
        issues = []
        
        # Check CPU
        cpu_usage = self.get_cpu_usage()
        if cpu_usage > self.cpu_threshold:
            issues.append(f"High CPU usage: {cpu_usage:.1f}% > {self.cpu_threshold}%")
        
        # Check memory
        memory_stats = self.get_memory_usage()
        if memory_stats["percentage"] > self.memory_threshold:
            issues.append(f"High memory usage: {memory_stats['percentage']:.1f}% > {self.memory_threshold}%")
        
        # Check disk
        disk_stats = self.get_disk_usage()
        if disk_stats["percentage"] > self.disk_threshold:
            issues.append(f"High disk usage: {disk_stats['percentage']:.1f}% > {self.disk_threshold}%")
        
        return len(issues) == 0, issues


class PerformanceMonitoringService:
    """Production performance monitoring service."""
    
    def __init__(
        self,
        monitoring_service: MonitoringService,
        alert_service: ProductionAlertService
    ):
        self.monitoring_service = monitoring_service
        self.alert_service = alert_service
        self.logger = logging.getLogger(__name__)
        
        # Performance trackers for different operations
        self.trackers: Dict[str, PerformanceTracker] = {}
        
        # System resource monitor
        self.resource_monitor = SystemResourceMonitor()
        
        # Monitoring intervals
        self.performance_check_interval = 30  # seconds
        self.resource_check_interval = 60     # seconds
        self.alert_cooldown = 300             # 5 minutes between similar alerts
        
        # Alert tracking
        self.last_alerts: Dict[str, datetime] = {}
        
    async def start_monitoring(self):
        """Start performance monitoring tasks."""
        self.logger.info("Starting performance monitoring service")
        
        await asyncio.gather(
            self._performance_monitoring_loop(),
            self._resource_monitoring_loop(),
            self._metrics_collection_loop()
        )
    
    def get_tracker(self, name: str) -> PerformanceTracker:
        """Get or create a performance tracker."""
        if name not in self.trackers:
            self.trackers[name] = PerformanceTracker(name)
        return self.trackers[name]
    
    async def record_request(
        self,
        operation: str,
        latency_ms: float,
        error: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a request performance metric."""
        tracker = self.get_tracker(operation)
        tracker.record_request(latency_ms, error)
        
        # Also record in Prometheus
        labels = labels or {}
        labels["operation"] = operation
        
        await self.monitoring_service.record_metric(
            "request_duration_ms",
            latency_ms,
            labels
        )
        
        if error:
            labels["error_type"] = error
            await self.monitoring_service.record_metric(
                "request_errors_total",
                1,
                labels
            )
    
    async def _performance_monitoring_loop(self):
        """Monitor performance metrics and trigger alerts."""
        while True:
            try:
                for name, tracker in self.trackers.items():
                    await self._check_tracker_health(name, tracker)
                
                await asyncio.sleep(self.performance_check_interval)
                
            except Exception as e:
                self.logger.error(f"Performance monitoring loop error: {e}")
                await asyncio.sleep(self.performance_check_interval)
    
    async def _check_tracker_health(self, name: str, tracker: PerformanceTracker):
        """Check health of a performance tracker."""
        is_healthy, issues = tracker.is_healthy()
        
        if not is_healthy:
            alert_key = f"performance_{name}"
            
            # Check alert cooldown
            if self._should_send_alert(alert_key):
                await self.alert_service.send_warning_alert(
                    f"Performance issues detected: {name}",
                    f"Performance degradation in {name}: {', '.join(issues)}"
                )
                self.last_alerts[alert_key] = datetime.utcnow()
        
        # Record health metrics
        await self.monitoring_service.record_metric(
            "performance_health_score",
            1.0 if is_healthy else 0.0,
            {"operation": name}
        )
    
    async def _resource_monitoring_loop(self):
        """Monitor system resource usage."""
        while True:
            try:
                # Check system health
                is_healthy, issues = self.resource_monitor.is_healthy()
                
                if not is_healthy:
                    alert_key = "system_resources"
                    
                    if self._should_send_alert(alert_key):
                        await self.alert_service.send_error_alert(
                            "System resource issues detected",
                            f"System resource problems: {', '.join(issues)}"
                        )
                        self.last_alerts[alert_key] = datetime.utcnow()
                
                # Record resource health
                await self.monitoring_service.record_metric(
                    "system_health_score",
                    1.0 if is_healthy else 0.0
                )
                
                await asyncio.sleep(self.resource_check_interval)
                
            except Exception as e:
                self.logger.error(f"Resource monitoring loop error: {e}")
                await asyncio.sleep(self.resource_check_interval)
    
    async def _metrics_collection_loop(self):
        """Collect and record performance metrics."""
        while True:
            try:
                # Collect system resource metrics
                await self._collect_system_metrics()
                
                # Collect performance tracker metrics
                await self._collect_performance_metrics()
                
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                self.logger.error(f"Metrics collection loop error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_system_metrics(self):
        """Collect system resource metrics."""
        # CPU metrics
        cpu_usage = self.resource_monitor.get_cpu_usage()
        await self.monitoring_service.record_metric("system_cpu_usage_percent", cpu_usage)
        
        # Memory metrics
        memory_stats = self.resource_monitor.get_memory_usage()
        await self.monitoring_service.record_metric("system_memory_usage_percent", memory_stats["percentage"])
        await self.monitoring_service.record_metric("system_memory_used_gb", memory_stats["used_gb"])
        await self.monitoring_service.record_metric("system_memory_available_gb", memory_stats["available_gb"])
        
        # Disk metrics
        disk_stats = self.resource_monitor.get_disk_usage()
        await self.monitoring_service.record_metric("system_disk_usage_percent", disk_stats["percentage"])
        await self.monitoring_service.record_metric("system_disk_used_gb", disk_stats["used_gb"])
        await self.monitoring_service.record_metric("system_disk_free_gb", disk_stats["free_gb"])
        
        # Network metrics
        network_stats = self.resource_monitor.get_network_stats()
        await self.monitoring_service.record_metric("system_network_bytes_sent", network_stats["bytes_sent"])
        await self.monitoring_service.record_metric("system_network_bytes_recv", network_stats["bytes_recv"])
    
    async def _collect_performance_metrics(self):
        """Collect performance tracker metrics."""
        for name, tracker in self.trackers.items():
            labels = {"operation": name}
            
            # Latency metrics
            latency_stats = tracker.get_latency_stats()
            if latency_stats.sample_count > 0:
                await self.monitoring_service.record_metric("latency_p50_ms", latency_stats.p50_latency, labels)
                await self.monitoring_service.record_metric("latency_p95_ms", latency_stats.p95_latency, labels)
                await self.monitoring_service.record_metric("latency_p99_ms", latency_stats.p99_latency, labels)
                await self.monitoring_service.record_metric("latency_avg_ms", latency_stats.avg_latency, labels)
            
            # Throughput metrics
            throughput_stats = tracker.get_throughput_stats()
            await self.monitoring_service.record_metric("throughput_rps", throughput_stats.requests_per_second, labels)
            await self.monitoring_service.record_metric("throughput_peak_rps", throughput_stats.peak_rps, labels)
            
            # Error rate metrics
            error_stats = tracker.get_error_rate_stats()
            await self.monitoring_service.record_metric("error_rate", error_stats.error_rate, labels)
            await self.monitoring_service.record_metric("total_errors", error_stats.total_errors, labels)
    
    def _should_send_alert(self, alert_key: str) -> bool:
        """Check if we should send an alert based on cooldown."""
        if alert_key not in self.last_alerts:
            return True
        
        time_since_last = datetime.utcnow() - self.last_alerts[alert_key]
        return time_since_last.total_seconds() > self.alert_cooldown
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of all performance metrics."""
        summary = {
            "system_resources": {
                "cpu_usage": self.resource_monitor.get_cpu_usage(),
                "memory_usage": self.resource_monitor.get_memory_usage(),
                "disk_usage": self.resource_monitor.get_disk_usage(),
                "network_stats": self.resource_monitor.get_network_stats()
            },
            "operations": {}
        }
        
        for name, tracker in self.trackers.items():
            latency_stats = tracker.get_latency_stats()
            throughput_stats = tracker.get_throughput_stats()
            error_stats = tracker.get_error_rate_stats()
            is_healthy, issues = tracker.is_healthy()
            
            summary["operations"][name] = {
                "latency": {
                    "p50_ms": latency_stats.p50_latency,
                    "p95_ms": latency_stats.p95_latency,
                    "p99_ms": latency_stats.p99_latency,
                    "avg_ms": latency_stats.avg_latency
                },
                "throughput": {
                    "current_rps": throughput_stats.requests_per_second,
                    "peak_rps": throughput_stats.peak_rps,
                    "total_requests": throughput_stats.total_requests
                },
                "errors": {
                    "error_rate": error_stats.error_rate,
                    "total_errors": error_stats.total_errors,
                    "error_types": error_stats.error_types
                },
                "health": {
                    "is_healthy": is_healthy,
                    "issues": issues
                }
            }
        
        return summary
    
    def set_thresholds(
        self,
        operation: str,
        latency_threshold_ms: Optional[float] = None,
        error_rate_threshold: Optional[float] = None,
        throughput_threshold_rps: Optional[float] = None
    ):
        """Set performance thresholds for an operation."""
        tracker = self.get_tracker(operation)
        
        if latency_threshold_ms is not None:
            tracker.latency_threshold_ms = latency_threshold_ms
        if error_rate_threshold is not None:
            tracker.error_rate_threshold = error_rate_threshold
        if throughput_threshold_rps is not None:
            tracker.throughput_threshold_rps = throughput_threshold_rps
        
        self.logger.info(f"Updated thresholds for {operation}")
    
    def clear_tracker(self, operation: str):
        """Clear performance data for an operation."""
        if operation in self.trackers:
            del self.trackers[operation]
            self.logger.info(f"Cleared performance tracker for {operation}")