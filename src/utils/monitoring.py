"""Monitoring and metrics infrastructure"""

import time
import psutil
import threading
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import requests
from config.settings import MonitoringConfig, get_settings
from utils.logging import get_logger

logger = get_logger("monitoring")


@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    timestamp: datetime


class MetricsCollector:
    """Collects and stores application metrics"""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        if config is None:
            settings = get_settings()
            config = settings.monitoring
        
        self.config = config
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._lock = threading.Lock()
        
        # System metrics tracking
        self._system_metrics_history: deque = deque(maxlen=100)
        self._last_network_stats = psutil.net_io_counters()
        
        # Health check status
        self.health_checks: Dict[str, bool] = {}
        
        # Start background monitoring if enabled
        if self.config.enabled:
            self._start_monitoring()
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric"""
        with self._lock:
            key = self._make_key(name, tags)
            self.counters[key] += value
            
            metric = Metric(
                name=name,
                value=self.counters[key],
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self.metrics[name].append(metric)
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value"""
        with self._lock:
            key = self._make_key(name, tags)
            self.gauges[key] = value
            
            metric = Metric(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self.metrics[name].append(metric)
    
    def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timing metric"""
        with self._lock:
            key = self._make_key(name, tags)
            self.timers[key].append(duration)
            
            metric = Metric(
                name=name,
                value=duration,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self.metrics[name].append(metric)
    
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations"""
        return TimerContext(self, name, tags)
    
    def get_metrics(self, name: str, since: Optional[datetime] = None) -> list:
        """Get metrics for a specific name"""
        with self._lock:
            metrics = list(self.metrics[name])
            
            if since:
                metrics = [m for m in metrics if m.timestamp >= since]
            
            return metrics
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Network usage
        network = psutil.net_io_counters()
        
        metrics = SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            disk_usage_percent=disk.percent,
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
            timestamp=datetime.now()
        )
        
        self._system_metrics_history.append(metrics)
        return metrics
    
    def register_health_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """Register a health check function"""
        try:
            result = check_func()
            self.health_checks[name] = result
            logger.debug(f"Health check '{name}': {'PASS' if result else 'FAIL'}")
        except Exception as e:
            self.health_checks[name] = False
            logger.error(f"Health check '{name}' failed with error: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        all_healthy = all(self.health_checks.values()) if self.health_checks else True
        
        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "checks": self.health_checks.copy(),
            "system": self._get_system_health()
        }
    
    def send_alert(self, message: str, severity: str = "warning") -> None:
        """Send alert notification"""
        if not self.config.alert_webhook_url:
            logger.warning(f"Alert webhook not configured. Alert: {message}")
            return
        
        alert_data = {
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "service": "ai_trading_platform"
        }
        
        try:
            response = requests.post(
                self.config.alert_webhook_url,
                json=alert_data,
                timeout=10
            )
            response.raise_for_status()
            logger.info(f"Alert sent successfully: {message}")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
    
    def _make_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Create a unique key for metric storage"""
        if not tags:
            return name
        
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get system health indicators"""
        if not self._system_metrics_history:
            return {"status": "unknown"}
        
        latest = self._system_metrics_history[-1]
        
        # Define thresholds
        cpu_threshold = 80.0
        memory_threshold = 85.0
        disk_threshold = 90.0
        
        issues = []
        if latest.cpu_percent > cpu_threshold:
            issues.append(f"High CPU usage: {latest.cpu_percent:.1f}%")
        
        if latest.memory_percent > memory_threshold:
            issues.append(f"High memory usage: {latest.memory_percent:.1f}%")
        
        if latest.disk_usage_percent > disk_threshold:
            issues.append(f"High disk usage: {latest.disk_usage_percent:.1f}%")
        
        return {
            "status": "healthy" if not issues else "degraded",
            "issues": issues,
            "cpu_percent": latest.cpu_percent,
            "memory_percent": latest.memory_percent,
            "disk_percent": latest.disk_usage_percent
        }
    
    def _start_monitoring(self) -> None:
        """Start background monitoring thread"""
        def monitor_loop():
            while self.config.enabled:
                try:
                    # Collect system metrics
                    self.get_system_metrics()
                    
                    # Run health checks
                    for name, check_func in self.health_checks.items():
                        if callable(check_func):
                            self.register_health_check(name, check_func)
                    
                    time.sleep(self.config.health_check_interval)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(5)  # Short sleep on error
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info("Background monitoring started")


class TimerContext:
    """Context manager for timing operations"""
    
    def __init__(self, collector: MetricsCollector, name: str, tags: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.record_timer(self.name, duration, self.tags)


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    global _metrics_collector
    
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    
    return _metrics_collector


def setup_monitoring(config: Optional[MonitoringConfig] = None) -> MetricsCollector:
    """Setup monitoring infrastructure"""
    global _metrics_collector
    
    _metrics_collector = MetricsCollector(config)
    logger.info("Monitoring infrastructure initialized")
    
    return _metrics_collector