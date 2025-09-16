"""
Monitoring Dashboard Service

Provides real-time monitoring dashboards and performance visualization
for the AI trading platform.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

from src.services.model_monitoring_service import (
    ModelMonitoringService, 
    ModelPerformanceMetrics,
    Alert,
    AlertSeverity
)
from src.utils.monitoring import get_metrics_collector
from src.utils.logging import get_logger

logger = get_logger("monitoring_dashboard")


@dataclass
class DashboardMetrics:
    """Dashboard metrics container"""
    timestamp: datetime
    system_health: str
    active_models: int
    total_predictions: int
    alerts_last_24h: int
    avg_model_accuracy: float
    avg_prediction_confidence: float
    system_uptime: float
    cpu_usage: float
    memory_usage: float
    active_alerts: List[Dict[str, Any]]


@dataclass
class ModelDashboardData:
    """Model-specific dashboard data"""
    model_name: str
    status: str
    health_score: float
    last_prediction: Optional[datetime]
    predictions_today: int
    accuracy_trend: List[float]
    confidence_trend: List[float]
    recent_alerts: List[Dict[str, Any]]
    performance_metrics: Optional[Dict[str, Any]]


class MonitoringDashboardService:
    """
    Service for generating monitoring dashboards and real-time metrics
    for system administrators and traders.
    """

    def __init__(self, monitoring_service: ModelMonitoringService):
        self.monitoring_service = monitoring_service
        self.metrics_collector = get_metrics_collector()
        self.dashboard_cache: Dict[str, Any] = {}
        self.cache_ttl = timedelta(seconds=30)  # 30-second cache
        self.last_cache_update: Dict[str, datetime] = {}

    async def get_system_dashboard(self) -> DashboardMetrics:
        """Get comprehensive system dashboard metrics"""
        
        cache_key = "system_dashboard"
        now = datetime.now()
        
        # Check cache
        if (cache_key in self.dashboard_cache and 
            cache_key in self.last_cache_update and
            now - self.last_cache_update[cache_key] < self.cache_ttl):
            return self.dashboard_cache[cache_key]
        
        # Calculate system metrics
        system_health = await self._calculate_system_health()
        active_models = len(self.monitoring_service.performance_history)
        
        # Get prediction counts
        total_predictions = sum(
            len(predictions) 
            for predictions in self.monitoring_service.prediction_history.values()
        )
        
        # Count recent alerts
        alerts_24h = len([
            alert for alert in self.monitoring_service.alert_history
            if alert.timestamp > now - timedelta(hours=24)
        ])
        
        # Calculate average model accuracy
        avg_accuracy = await self._calculate_average_accuracy()
        
        # Calculate average prediction confidence
        avg_confidence = await self._calculate_average_confidence()
        
        # Get system resource metrics
        system_metrics = self.metrics_collector.get_system_metrics()
        
        # Get active alerts (last 1 hour, high severity)
        active_alerts = [
            asdict(alert) for alert in self.monitoring_service.alert_history
            if (alert.timestamp > now - timedelta(hours=1) and 
                alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL])
        ]
        
        dashboard_metrics = DashboardMetrics(
            timestamp=now,
            system_health=system_health,
            active_models=active_models,
            total_predictions=total_predictions,
            alerts_last_24h=alerts_24h,
            avg_model_accuracy=avg_accuracy,
            avg_prediction_confidence=avg_confidence,
            system_uptime=99.9,  # Placeholder - would calculate from actual uptime
            cpu_usage=system_metrics.cpu_percent,
            memory_usage=system_metrics.memory_percent,
            active_alerts=active_alerts
        )
        
        # Cache result
        self.dashboard_cache[cache_key] = dashboard_metrics
        self.last_cache_update[cache_key] = now
        
        return dashboard_metrics

    async def get_model_dashboard(self, model_name: str) -> Optional[ModelDashboardData]:
        """Get detailed dashboard data for a specific model"""
        
        cache_key = f"model_dashboard_{model_name}"
        now = datetime.now()
        
        # Check cache
        if (cache_key in self.dashboard_cache and 
            cache_key in self.last_cache_update and
            now - self.last_cache_update[cache_key] < self.cache_ttl):
            return self.dashboard_cache[cache_key]
        
        # Get model status
        model_status = await self.monitoring_service.get_model_status(model_name)
        
        if not model_status:
            return None
        
        # Get prediction history
        predictions = self.monitoring_service.prediction_history.get(model_name, [])
        
        # Calculate predictions today
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        predictions_today = len([
            p for p in predictions 
            if p['timestamp'] > today_start
        ])
        
        # Get last prediction time
        last_prediction = None
        if predictions:
            last_prediction = max(p['timestamp'] for p in predictions)
        
        # Calculate accuracy trend (last 10 performance measurements)
        accuracy_trend = []
        performance_history = self.monitoring_service.performance_history.get(model_name, [])
        if performance_history:
            recent_performance = performance_history[-10:]
            accuracy_trend = [p.accuracy for p in recent_performance]
        
        # Calculate confidence trend
        confidence_trend = []
        if predictions:
            recent_predictions = predictions[-50:]  # Last 50 predictions
            confidences = [
                p['confidence'] for p in recent_predictions 
                if p['confidence'] is not None
            ]
            if confidences:
                # Group by time windows and average
                confidence_trend = self._calculate_trend_windows(confidences, window_size=10)
        
        # Get recent alerts
        recent_alerts = [
            asdict(alert) for alert in self.monitoring_service.alert_history
            if (alert.model_name == model_name and 
                alert.timestamp > now - timedelta(hours=24))
        ]
        
        dashboard_data = ModelDashboardData(
            model_name=model_name,
            status=model_status['drift_status'],
            health_score=model_status['health_score'],
            last_prediction=last_prediction,
            predictions_today=predictions_today,
            accuracy_trend=accuracy_trend,
            confidence_trend=confidence_trend,
            recent_alerts=recent_alerts,
            performance_metrics=model_status['performance_metrics']
        )
        
        # Cache result
        self.dashboard_cache[cache_key] = dashboard_data
        self.last_cache_update[cache_key] = now
        
        return dashboard_data

    async def get_performance_trends(self, model_name: str, 
                                   days: int = 7) -> Dict[str, List[Dict[str, Any]]]:
        """Get performance trends for a model over specified days"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        performance_history = self.monitoring_service.performance_history.get(model_name, [])
        recent_performance = [
            p for p in performance_history 
            if p.timestamp > cutoff_date
        ]
        
        if not recent_performance:
            return {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1_score': [],
                'confidence': []
            }
        
        # Group by day and calculate daily averages
        daily_metrics = {}
        for metrics in recent_performance:
            day_key = metrics.timestamp.date()
            if day_key not in daily_metrics:
                daily_metrics[day_key] = []
            daily_metrics[day_key].append(metrics)
        
        trends = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'confidence': []
        }
        
        for day, day_metrics in sorted(daily_metrics.items()):
            day_data = {
                'date': day.isoformat(),
                'accuracy': np.mean([m.accuracy for m in day_metrics]),
                'precision': np.mean([m.precision for m in day_metrics]),
                'recall': np.mean([m.recall for m in day_metrics]),
                'f1_score': np.mean([m.f1_score for m in day_metrics]),
            }
            
            # Calculate average confidence if available
            confidences = [
                m.prediction_confidence for m in day_metrics 
                if m.prediction_confidence is not None
            ]
            if confidences:
                day_data['confidence'] = np.mean(confidences)
            else:
                day_data['confidence'] = None
            
            for metric_name in trends:
                if day_data[metric_name] is not None:
                    trends[metric_name].append({
                        'date': day_data['date'],
                        'value': day_data[metric_name]
                    })
        
        return trends

    async def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary for the specified time period"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.monitoring_service.alert_history
            if alert.timestamp > cutoff_time
        ]
        
        # Group by severity
        severity_counts = {severity.value: 0 for severity in AlertSeverity}
        for alert in recent_alerts:
            severity_counts[alert.severity.value] += 1
        
        # Group by model
        model_counts = {}
        for alert in recent_alerts:
            model_name = alert.model_name or 'system'
            model_counts[model_name] = model_counts.get(model_name, 0) + 1
        
        # Group by hour for trend
        hourly_counts = {}
        for alert in recent_alerts:
            hour_key = alert.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_counts[hour_key] = hourly_counts.get(hour_key, 0) + 1
        
        # Convert to time series
        alert_trend = []
        for i in range(hours):
            hour = datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(hours=i)
            count = hourly_counts.get(hour, 0)
            alert_trend.append({
                'hour': hour.isoformat(),
                'count': count
            })
        
        alert_trend.reverse()  # Chronological order
        
        return {
            'total_alerts': len(recent_alerts),
            'severity_breakdown': severity_counts,
            'model_breakdown': model_counts,
            'hourly_trend': alert_trend,
            'most_recent': [asdict(alert) for alert in recent_alerts[-5:]]  # Last 5 alerts
        }

    async def get_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report"""
        
        now = datetime.now()
        
        # Get system metrics
        system_metrics = self.metrics_collector.get_system_metrics()
        health_status = self.metrics_collector.get_health_status()
        
        # Calculate model health scores
        model_health = {}
        for model_name in self.monitoring_service.performance_history:
            status = await self.monitoring_service.get_model_status(model_name)
            model_health[model_name] = {
                'health_score': status['health_score'],
                'status': status['drift_status'],
                'last_update': status['timestamp'].isoformat()
            }
        
        # Calculate overall system health score
        if model_health:
            avg_model_health = np.mean([m['health_score'] for m in model_health.values()])
        else:
            avg_model_health = 100.0
        
        # Factor in system resources
        resource_health = 100.0
        if system_metrics.cpu_percent > 80:
            resource_health -= 20
        if system_metrics.memory_percent > 85:
            resource_health -= 20
        if system_metrics.disk_usage_percent > 90:
            resource_health -= 30
        
        overall_health = min(avg_model_health, resource_health)
        
        # Determine overall status
        if overall_health >= 90:
            overall_status = "excellent"
        elif overall_health >= 75:
            overall_status = "good"
        elif overall_health >= 50:
            overall_status = "degraded"
        else:
            overall_status = "critical"
        
        return {
            'timestamp': now.isoformat(),
            'overall_health_score': overall_health,
            'overall_status': overall_status,
            'system_resources': {
                'cpu_percent': system_metrics.cpu_percent,
                'memory_percent': system_metrics.memory_percent,
                'disk_percent': system_metrics.disk_usage_percent,
                'status': health_status['system']['status']
            },
            'model_health': model_health,
            'active_issues': health_status['system'].get('issues', []),
            'recommendations': await self._generate_health_recommendations(overall_health, system_metrics)
        }

    async def _calculate_system_health(self) -> str:
        """Calculate overall system health status"""
        
        # Check system resources
        system_metrics = self.metrics_collector.get_system_metrics()
        
        # Check recent alerts
        recent_alerts = [
            alert for alert in self.monitoring_service.alert_history
            if alert.timestamp > datetime.now() - timedelta(hours=1)
        ]
        
        critical_alerts = [
            alert for alert in recent_alerts 
            if alert.severity == AlertSeverity.CRITICAL
        ]
        
        high_alerts = [
            alert for alert in recent_alerts 
            if alert.severity == AlertSeverity.HIGH
        ]
        
        # Determine health status
        if critical_alerts:
            return "critical"
        elif high_alerts or system_metrics.cpu_percent > 90 or system_metrics.memory_percent > 95:
            return "degraded"
        elif len(recent_alerts) > 10 or system_metrics.cpu_percent > 80:
            return "warning"
        else:
            return "healthy"

    async def _calculate_average_accuracy(self) -> float:
        """Calculate average accuracy across all models"""
        
        all_accuracies = []
        
        for model_name, performance_history in self.monitoring_service.performance_history.items():
            if performance_history:
                # Get recent accuracy (last 5 measurements)
                recent_performance = performance_history[-5:]
                accuracies = [p.accuracy for p in recent_performance]
                all_accuracies.extend(accuracies)
        
        return np.mean(all_accuracies) if all_accuracies else 0.0

    async def _calculate_average_confidence(self) -> float:
        """Calculate average prediction confidence across all models"""
        
        all_confidences = []
        
        for model_name, predictions in self.monitoring_service.prediction_history.items():
            # Get recent predictions (last 50)
            recent_predictions = predictions[-50:]
            confidences = [
                p['confidence'] for p in recent_predictions 
                if p['confidence'] is not None
            ]
            all_confidences.extend(confidences)
        
        return np.mean(all_confidences) if all_confidences else 0.0

    def _calculate_trend_windows(self, values: List[float], window_size: int = 10) -> List[float]:
        """Calculate windowed averages for trend visualization"""
        
        if len(values) < window_size:
            return values
        
        trends = []
        for i in range(0, len(values), window_size):
            window = values[i:i + window_size]
            trends.append(np.mean(window))
        
        return trends

    async def _generate_health_recommendations(self, health_score: float, 
                                             system_metrics) -> List[str]:
        """Generate health improvement recommendations"""
        
        recommendations = []
        
        if health_score < 50:
            recommendations.append("System health is critical. Immediate attention required.")
        
        if system_metrics.cpu_percent > 80:
            recommendations.append("High CPU usage detected. Consider scaling compute resources.")
        
        if system_metrics.memory_percent > 85:
            recommendations.append("High memory usage detected. Consider increasing memory allocation.")
        
        if system_metrics.disk_usage_percent > 90:
            recommendations.append("Disk space is running low. Clean up old data or expand storage.")
        
        # Check for model-specific issues
        for model_name in self.monitoring_service.performance_history:
            status = self.monitoring_service.get_model_status(model_name)
            if status['health_score'] < 60:
                recommendations.append(f"Model {model_name} performance is degraded. Consider retraining.")
        
        # Check alert frequency
        recent_alerts = [
            alert for alert in self.monitoring_service.alert_history
            if alert.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        if len(recent_alerts) > 20:
            recommendations.append("High alert frequency detected. Review system configuration.")
        
        if not recommendations:
            recommendations.append("System is operating normally. Continue monitoring.")
        
        return recommendations

    def clear_cache(self) -> None:
        """Clear dashboard cache to force refresh"""
        self.dashboard_cache.clear()
        self.last_cache_update.clear()
        logger.info("Dashboard cache cleared")

    async def export_metrics(self, format_type: str = "json", 
                           time_range: Optional[timedelta] = None) -> str:
        """Export monitoring metrics in specified format"""
        
        if time_range is None:
            time_range = timedelta(days=7)
        
        cutoff_time = datetime.now() - time_range
        
        # Collect all metrics
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'time_range_days': time_range.days,
            'system_health': await self.get_system_health_report(),
            'models': {},
            'alerts': []
        }
        
        # Export model data
        for model_name in self.monitoring_service.performance_history:
            model_data = await self.get_model_dashboard(model_name)
            if model_data:
                export_data['models'][model_name] = asdict(model_data)
        
        # Export alerts
        recent_alerts = [
            asdict(alert) for alert in self.monitoring_service.alert_history
            if alert.timestamp > cutoff_time
        ]
        export_data['alerts'] = recent_alerts
        
        if format_type.lower() == "json":
            return json.dumps(export_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")