"""Alerting system for feature extraction performance monitoring.

This module provides specialized alerting mechanisms for feature extraction
performance that integrate with the existing monitoring infrastructure
and provide real-time notifications for performance degradation.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

from src.services.monitoring.alert_system import AlertSubject, AlertFactory, AlertSeverity
from src.ml.feature_extraction.monitoring import FeatureExtractionMetrics
from src.ml.feature_extraction.enhanced_metrics import EnhancedMetricsCollector

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Types of alerts for feature extraction"""
    LATENCY_THRESHOLD = "latency_threshold"
    RESOURCE_THRESHOLD = "resource_threshold"
    CACHE_DEGRADATION = "cache_degradation"
    ERROR_RATE_SPIKE = "error_rate_spike"
    THROUGHPUT_DROP = "throughput_drop"


@dataclass
class FeatureExtractionAlert:
    """Alert for feature extraction performance issues"""
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    metrics: Optional[Dict[str, Any]] = None
    threshold: Optional[float] = None
    current_value: Optional[float] = None


class FeatureExtractionAlertingSystem:
    """Alerting system for feature extraction performance monitoring"""
    
    def __init__(self, metrics_collector: EnhancedMetricsCollector):
        """Initialize alerting system.
        
        Args:
            metrics_collector: Enhanced metrics collector for feature extraction
        """
        self.metrics_collector = metrics_collector
        self.alert_subject = AlertSubject()
        self.alert_handlers: Dict[AlertType, List[Callable]] = {}
        self.alert_history: List[FeatureExtractionAlert] = []
        self.alert_cooldown: Dict[str, datetime] = {}
        self.cooldown_period = timedelta(minutes=5)
        
        # Alert thresholds
        self.thresholds = {
            'max_latency_ms': 100.0,      # <100ms requirement
            'critical_latency_ms': 80.0,  # Critical at 80ms
            'warning_latency_ms': 50.0,   # Warning at 50ms
            'max_error_rate': 0.01,       # Maximum 1% error rate
            'min_cache_hit_rate': 0.7,    # Minimum 70% cache hit rate
            'max_memory_mb': 1000.0,      # Maximum 1GB memory usage
            'critical_cpu_percent': 80.0, # Critical CPU usage
            'min_throughput': 10.0        # Minimum 10 extractions per second
        }
        
        logger.info("Feature extraction alerting system initialized")
    
    def register_alert_handler(self, alert_type: AlertType, handler: Callable) -> None:
        """Register an alert handler for a specific alert type.
        
        Args:
            alert_type: Type of alert to handle
            handler: Function to call when alert is triggered
        """
        if alert_type not in self.alert_handlers:
            self.alert_handlers[alert_type] = []
        self.alert_handlers[alert_type].append(handler)
        logger.info(f"Alert handler registered for {alert_type.value}")
    
    def check_performance_alerts(self, metrics: Optional[FeatureExtractionMetrics] = None) -> List[FeatureExtractionAlert]:
        """Check for performance alerts based on current metrics.
        
        Args:
            metrics: Optional specific metrics to check, otherwise uses latest metrics
            
        Returns:
            List of alerts that were triggered
        """
        alerts = []
        
        # Get current performance summary
        performance_summary = self.metrics_collector.get_performance_summary()
        resource_utilization = self.metrics_collector.get_resource_utilization()
        requirements_status = self.metrics_collector.meets_performance_requirements()
        
        # Check latency thresholds
        alerts.extend(self._check_latency_alerts(performance_summary))
        
        # Check resource thresholds
        alerts.extend(self._check_resource_alerts(resource_utilization))
        
        # Check cache degradation
        alerts.extend(self._check_cache_alerts(performance_summary))
        
        # Check error rate spikes
        alerts.extend(self._check_error_rate_alerts(performance_summary))
        
        # Check throughput drops
        alerts.extend(self._check_throughput_alerts(performance_summary))
        
        # Process and send alerts
        for alert in alerts:
            self._process_alert(alert)
        
        return alerts
    
    def _check_latency_alerts(self, performance_summary: Dict[str, Any]) -> List[FeatureExtractionAlert]:
        """Check for latency-related alerts.
        
        Args:
            performance_summary: Performance summary metrics
            
        Returns:
            List of latency alerts
        """
        alerts = []
        avg_latency = performance_summary.get('avg_latency_ms', 0)
        p95_latency = performance_summary.get('p95_latency_ms', 0)
        
        # Check if we're violating the <100ms requirement
        if p95_latency > self.thresholds['max_latency_ms']:
            alerts.append(FeatureExtractionAlert(
                alert_type=AlertType.LATENCY_THRESHOLD,
                severity=AlertSeverity.CRITICAL,
                title="Feature Extraction Latency Violation",
                message=f"95th percentile latency ({p95_latency:.2f}ms) exceeds maximum threshold ({self.thresholds['max_latency_ms']}ms)",
                timestamp=datetime.now(),
                metrics=performance_summary,
                threshold=self.thresholds['max_latency_ms'],
                current_value=p95_latency
            ))
        elif p95_latency > self.thresholds['critical_latency_ms']:
            alerts.append(FeatureExtractionAlert(
                alert_type=AlertType.LATENCY_THRESHOLD,
                severity=AlertSeverity.HIGH,
                title="Feature Extraction Latency Critical",
                message=f"95th percentile latency ({p95_latency:.2f}ms) exceeds critical threshold ({self.thresholds['critical_latency_ms']}ms)",
                timestamp=datetime.now(),
                metrics=performance_summary,
                threshold=self.thresholds['critical_latency_ms'],
                current_value=p95_latency
            ))
        elif avg_latency > self.thresholds['warning_latency_ms']:
            alerts.append(FeatureExtractionAlert(
                alert_type=AlertType.LATENCY_THRESHOLD,
                severity=AlertSeverity.MEDIUM,
                title="Feature Extraction Latency Warning",
                message=f"Average latency ({avg_latency:.2f}ms) exceeds warning threshold ({self.thresholds['warning_latency_ms']}ms)",
                timestamp=datetime.now(),
                metrics=performance_summary,
                threshold=self.thresholds['warning_latency_ms'],
                current_value=avg_latency
            ))
        
        return alerts
    
    def _check_resource_alerts(self, resource_utilization: Dict[str, Any]) -> List[FeatureExtractionAlert]:
        """Check for resource-related alerts.
        
        Args:
            resource_utilization: Resource utilization metrics
            
        Returns:
            List of resource alerts
        """
        alerts = []
        memory_mb = resource_utilization.get('memory_mb', 0)
        cpu_percent = resource_utilization.get('cpu_percent', 0)
        gpu_utilization = resource_utilization.get('gpu_utilization', 0)
        
        # Check memory usage
        if memory_mb > self.thresholds['max_memory_mb']:
            alerts.append(FeatureExtractionAlert(
                alert_type=AlertType.RESOURCE_THRESHOLD,
                severity=AlertSeverity.HIGH,
                title="Feature Extraction Memory Usage High",
                message=f"Memory usage ({memory_mb:.1f}MB) exceeds maximum threshold ({self.thresholds['max_memory_mb']}MB)",
                timestamp=datetime.now(),
                metrics=resource_utilization,
                threshold=self.thresholds['max_memory_mb'],
                current_value=memory_mb
            ))
        
        # Check CPU usage
        if cpu_percent > self.thresholds['critical_cpu_percent']:
            alerts.append(FeatureExtractionAlert(
                alert_type=AlertType.RESOURCE_THRESHOLD,
                severity=AlertSeverity.HIGH,
                title="Feature Extraction CPU Usage High",
                message=f"CPU usage ({cpu_percent:.1f}%) exceeds critical threshold ({self.thresholds['critical_cpu_percent']}%)",
                timestamp=datetime.now(),
                metrics=resource_utilization,
                threshold=self.thresholds['critical_cpu_percent'],
                current_value=cpu_percent
            ))
        
        # Check GPU usage if available
        if gpu_utilization > 95:  # GPU near capacity
            alerts.append(FeatureExtractionAlert(
                alert_type=AlertType.RESOURCE_THRESHOLD,
                severity=AlertSeverity.MEDIUM,
                title="Feature Extraction GPU Near Capacity",
                message=f"GPU utilization ({gpu_utilization:.1f}%) is near maximum capacity",
                timestamp=datetime.now(),
                metrics=resource_utilization,
                threshold=95.0,
                current_value=gpu_utilization
            ))
        
        return alerts
    
    def _check_cache_alerts(self, performance_summary: Dict[str, Any]) -> List[FeatureExtractionAlert]:
        """Check for cache-related alerts.
        
        Args:
            performance_summary: Performance summary metrics
            
        Returns:
            List of cache alerts
        """
        alerts = []
        cache_hit_rate = performance_summary.get('cache_hit_rate', 0)
        
        # Check cache hit rate
        if cache_hit_rate < self.thresholds['min_cache_hit_rate']:
            alerts.append(FeatureExtractionAlert(
                alert_type=AlertType.CACHE_DEGRADATION,
                severity=AlertSeverity.HIGH,
                title="Feature Extraction Cache Hit Rate Low",
                message=f"Cache hit rate ({cache_hit_rate:.1%}) below minimum threshold ({self.thresholds['min_cache_hit_rate']:.0%})",
                timestamp=datetime.now(),
                metrics=performance_summary,
                threshold=self.thresholds['min_cache_hit_rate'],
                current_value=cache_hit_rate
            ))
        elif cache_hit_rate < self.thresholds['min_cache_hit_rate'] * 1.2:  # 20% buffer
            alerts.append(FeatureExtractionAlert(
                alert_type=AlertType.CACHE_DEGRADATION,
                severity=AlertSeverity.MEDIUM,
                title="Feature Extraction Cache Hit Rate Warning",
                message=f"Cache hit rate ({cache_hit_rate:.1%}) approaching minimum threshold ({self.thresholds['min_cache_hit_rate']:.0%})",
                timestamp=datetime.now(),
                metrics=performance_summary,
                threshold=self.thresholds['min_cache_hit_rate'] * 1.2,
                current_value=cache_hit_rate
            ))
        
        return alerts
    
    def _check_error_rate_alerts(self, performance_summary: Dict[str, Any]) -> List[FeatureExtractionAlert]:
        """Check for error rate alerts.
        
        Args:
            performance_summary: Performance summary metrics
            
        Returns:
            List of error rate alerts
        """
        alerts = []
        error_rate = performance_summary.get('error_rate', 0)
        
        # Check error rate
        if error_rate > self.thresholds['max_error_rate']:
            alerts.append(FeatureExtractionAlert(
                alert_type=AlertType.ERROR_RATE_SPIKE,
                severity=AlertSeverity.CRITICAL,
                title="Feature Extraction Error Rate Spike",
                message=f"Error rate ({error_rate:.2%}) exceeds maximum threshold ({self.thresholds['max_error_rate']:.1%})",
                timestamp=datetime.now(),
                metrics=performance_summary,
                threshold=self.thresholds['max_error_rate'],
                current_value=error_rate
            ))
        elif error_rate > self.thresholds['max_error_rate'] * 0.5:  # 50% of threshold
            alerts.append(FeatureExtractionAlert(
                alert_type=AlertType.ERROR_RATE_SPIKE,
                severity=AlertSeverity.MEDIUM,
                title="Feature Extraction Error Rate Warning",
                message=f"Error rate ({error_rate:.2%}) approaching maximum threshold ({self.thresholds['max_error_rate']:.1%})",
                timestamp=datetime.now(),
                metrics=performance_summary,
                threshold=self.thresholds['max_error_rate'] * 0.5,
                current_value=error_rate
            ))
        
        return alerts
    
    def _check_throughput_alerts(self, performance_summary: Dict[str, Any]) -> List[FeatureExtractionAlert]:
        """Check for throughput-related alerts.
        
        Args:
            performance_summary: Performance summary metrics
            
        Returns:
            List of throughput alerts
        """
        alerts = []
        throughput = performance_summary.get('throughput_extractions_per_second', 0)
        
        # Check throughput
        if throughput < self.thresholds['min_throughput'] and throughput > 0:  # Only alert if we have data
            alerts.append(FeatureExtractionAlert(
                alert_type=AlertType.THROUGHPUT_DROP,
                severity=AlertSeverity.HIGH,
                title="Feature Extraction Throughput Drop",
                message=f"Throughput ({throughput:.1f} extractions/sec) below minimum threshold ({self.thresholds['min_throughput']} extractions/sec)",
                timestamp=datetime.now(),
                metrics=performance_summary,
                threshold=self.thresholds['min_throughput'],
                current_value=throughput
            ))
        
        return alerts
    
    def _process_alert(self, alert: FeatureExtractionAlert) -> None:
        """Process and send an alert.
        
        Args:
            alert: Alert to process
        """
        # Check cooldown
        cooldown_key = f"{alert.alert_type.value}_{alert.severity.value}"
        now = datetime.now()
        
        if cooldown_key in self.alert_cooldown:
            if now - self.alert_cooldown[cooldown_key] < self.cooldown_period:
                logger.debug(f"Alert suppressed due to cooldown: {alert.title}")
                return
        
        # Update cooldown
        self.alert_cooldown[cooldown_key] = now
        
        # Add to history
        self.alert_history.append(alert)
        
        # Trim history to last 1000 alerts
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        # Convert to standard alert format
        standard_alert = AlertFactory.create_performance_alert(
            "feature_extraction",
            alert.alert_type.value,
            alert.current_value or 0,
            alert.threshold or 0
        )
        standard_alert.severity = alert.severity
        standard_alert.title = alert.title
        standard_alert.message = alert.message
        standard_alert.timestamp = alert.timestamp
        
        # Send through alert system
        asyncio.create_task(self.alert_subject.notify_observers(standard_alert))
        
        # Call registered handlers
        if alert.alert_type in self.alert_handlers:
            for handler in self.alert_handlers[alert.alert_type]:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Error in alert handler for {alert.alert_type.value}: {e}")
        
        logger.warning(f"Feature extraction alert: {alert.title} - {alert.message}")
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent alerts.
        
        Args:
            hours: Number of hours to look back for alerts
            
        Returns:
            Dictionary with alert summary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [alert for alert in self.alert_history if alert.timestamp > cutoff_time]
        
        # Group by alert type and severity
        alert_counts = {}
        for alert in recent_alerts:
            key = f"{alert.alert_type.value}_{alert.severity.value}"
            alert_counts[key] = alert_counts.get(key, 0) + 1
        
        return {
            'total_alerts': len(recent_alerts),
            'alert_counts': alert_counts,
            'most_common_alert': max(alert_counts.items(), key=lambda x: x[1])[0] if alert_counts else None,
            'critical_alerts': len([a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]),
            'high_alerts': len([a for a in recent_alerts if a.severity == AlertSeverity.HIGH]),
            'medium_alerts': len([a for a in recent_alerts if a.severity == AlertSeverity.MEDIUM])
        }
    
    def clear_alert_history(self) -> None:
        """Clear alert history."""
        self.alert_history.clear()
        self.alert_cooldown.clear()
        logger.info("Feature extraction alert history cleared")