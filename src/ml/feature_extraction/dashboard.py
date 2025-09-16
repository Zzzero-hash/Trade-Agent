"""Dashboard components for feature extraction performance monitoring.

This module provides dashboard components for visualizing feature extraction
performance metrics, alerts, and resource utilization in a unified interface.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from src.ml.feature_extraction.enhanced_metrics import EnhancedMetricsCollector
from src.ml.feature_extraction.alerting import FeatureExtractionAlertingSystem
from src.ml.feature_extraction.ray_integration import RayServeIntegration
from src.ml.feature_extraction.cache_connection_integration import CacheConnectionIntegration

logger = logging.getLogger(__name__)


@dataclass
class DashboardMetrics:
    """Metrics for dashboard display"""
    timestamp: str
    feature_extraction: Dict[str, Any]
    ray_serve: Dict[str, Any]
    cache: Dict[str, Any]
    connection_pools: Dict[str, Any]
    alerts: Dict[str, Any]
    resource_utilization: Dict[str, Any]


class FeatureExtractionDashboard:
    """Dashboard for feature extraction performance monitoring"""
    
    def __init__(self):
        """Initialize dashboard components."""
        self.enhanced_metrics_collector = EnhancedMetricsCollector()
        self.alerting_system = FeatureExtractionAlertingSystem(self.enhanced_metrics_collector)
        self.ray_integration = RayServeIntegration()
        self.cache_connection_integration = CacheConnectionIntegration()
        
        logger.info("Feature extraction dashboard initialized")
    
    def get_real_time_dashboard_data(self) -> DashboardMetrics:
        """Get real-time dashboard data from all integrated systems.
        
        Returns:
            DashboardMetrics with current data from all systems
        """
        # Get real-time metrics
        real_time_metrics = self.enhanced_metrics_collector.get_real_time_metrics()
        
        # Get performance summary
        performance_summary = self.enhanced_metrics_collector.get_performance_summary()
        
        # Get Ray Serve integration data
        ray_stats = self.ray_integration.get_unified_performance_stats()
        
        # Get cache and connection pool data
        resource_stats = self.cache_connection_integration.get_unified_resource_stats()
        
        # Get alert summary
        alert_summary = self.alerting_system.get_alert_summary()
        
        # Get resource utilization
        resource_utilization = self.enhanced_metrics_collector.get_resource_utilization()
        
        # Create dashboard metrics
        dashboard_metrics = DashboardMetrics(
            timestamp=datetime.now().isoformat(),
            feature_extraction={
                'real_time': asdict(real_time_metrics) if hasattr(real_time_metrics, '__dict__') else {},
                'summary': performance_summary,
                'requirements_status': self.enhanced_metrics_collector.meets_performance_requirements()
            },
            ray_serve=ray_stats.get('ray_serve', {}),
            cache=resource_stats.get('cache', {}),
            connection_pools=resource_stats.get('connection_pools', {}),
            alerts=alert_summary,
            resource_utilization=resource_utilization
        )
        
        return dashboard_metrics
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends over time.
        
        Args:
            hours: Number of hours to look back for trends
            
        Returns:
            Dictionary with performance trends data
        """
        # Calculate time window
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Get metrics history within time window
        metrics_history = [
            m for m in self.enhanced_metrics_collector.metrics_history 
            if m.timestamp >= start_time
        ]
        
        if not metrics_history:
            return {
                'time_series': [],
                'trends': {}
            }
        
        # Group metrics by time intervals (1 hour)
        interval_data = {}
        for metric in metrics_history:
            interval_key = metric.timestamp.replace(minute=0, second=0, microsecond=0)
            if interval_key not in interval_data:
                interval_data[interval_key] = []
            interval_data[interval_key].append(metric)
        
        # Calculate averages for each interval
        time_series = []
        for interval_time, interval_metrics in sorted(interval_data.items()):
            if interval_metrics:
                avg_latency = sum(m.duration_ms for m in interval_metrics) / len(interval_metrics)
                cache_hit_rate = sum(1 if m.used_cache else 0 for m in interval_metrics) / len(interval_metrics)
                error_rate = sum(1 if m.had_error else 0 for m in interval_metrics) / len(interval_metrics)
                
                time_series.append({
                    'timestamp': interval_time.isoformat(),
                    'avg_latency_ms': avg_latency,
                    'cache_hit_rate': cache_hit_rate,
                    'error_rate': error_rate,
                    'count': len(interval_metrics)
                })
        
        # Calculate trends
        if len(time_series) >= 2:
            first_half = time_series[:len(time_series)//2]
            second_half = time_series[len(time_series)//2:]
            
            first_avg_latency = sum(d['avg_latency_ms'] for d in first_half) / len(first_half)
            second_avg_latency = sum(d['avg_latency_ms'] for d in second_half) / len(second_half)
            
            latency_trend = 'improving' if second_avg_latency < first_avg_latency else 'degrading' if second_avg_latency > first_avg_latency else 'stable'
            
            trends = {
                'latency_trend': latency_trend,
                'latency_change_percent': ((second_avg_latency - first_avg_latency) / first_avg_latency * 100) if first_avg_latency > 0 else 0,
                'total_extractions': sum(d['count'] for d in time_series)
            }
        else:
            trends = {
                'latency_trend': 'insufficient_data',
                'latency_change_percent': 0,
                'total_extractions': sum(d['count'] for d in time_series)
            }
        
        return {
            'time_series': time_series,
            'trends': trends
        }
    
    def get_system_health_overview(self) -> Dict[str, Any]:
        """Get system health overview.
        
        Returns:
            Dictionary with system health overview
        """
        # Get performance requirements status
        requirements_status = self.enhanced_metrics_collector.meets_performance_requirements()
        
        # Get Ray Serve requirements status
        ray_requirements = self.ray_integration.get_performance_requirements_status()
        
        # Get resource utilization
        resource_utilization = self.enhanced_metrics_collector.get_resource_utilization()
        
        # Get alert summary
        alert_summary = self.alerting_system.get_alert_summary()
        
        # Calculate overall health score
        health_score = self._calculate_health_score(
            requirements_status, 
            resource_utilization, 
            alert_summary
        )
        
        # Determine health status
        if health_score >= 80:
            health_status = 'healthy'
        elif health_score >= 60:
            health_status = 'warning'
        elif health_score >= 40:
            health_status = 'degraded'
        else:
            health_status = 'critical'
        
        return {
            'timestamp': datetime.now().isoformat(),
            'health_score': health_score,
            'health_status': health_status,
            'performance_requirements': requirements_status,
            'ray_requirements': ray_requirements,
            'resource_utilization': resource_utilization,
            'alerts': alert_summary,
            'recommendations': self._generate_health_recommendations(
                health_status, 
                requirements_status, 
                resource_utilization, 
                alert_summary
            )
        }
    
    def _calculate_health_score(self, requirements_status: Dict[str, Any], 
                              resource_utilization: Dict[str, Any], 
                              alert_summary: Dict[str, Any]) -> float:
        """Calculate overall health score.
        
        Args:
            requirements_status: Performance requirements status
            resource_utilization: Resource utilization metrics
            alert_summary: Alert summary
            
        Returns:
            Health score between 0-100
        """
        score = 100.0
        
        # Penalty for not meeting performance requirements
        if not requirements_status.get('meets_100ms_requirement', True):
            score -= 30
        
        # Penalty for high resource utilization
        cpu_percent = resource_utilization.get('cpu_percent', 0)
        memory_mb = resource_utilization.get('memory_mb', 0)
        
        if cpu_percent > 80:
            score -= 10
        elif cpu_percent > 60:
            score -= 5
            
        if memory_mb > 800:  # 800MB threshold
            score -= 10
        elif memory_mb > 600:
            score -= 5
        
        # Penalty for alerts
        critical_alerts = alert_summary.get('critical_alerts', 0)
        high_alerts = alert_summary.get('high_alerts', 0)
        medium_alerts = alert_summary.get('medium_alerts', 0)
        
        score -= (critical_alerts * 15 + high_alerts * 10 + medium_alerts * 5)
        
        return max(0, min(100, score))
    
    def _generate_health_recommendations(self, health_status: str, 
                                       requirements_status: Dict[str, Any], 
                                       resource_utilization: Dict[str, Any], 
                                       alert_summary: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate health recommendations.
        
        Args:
            health_status: Current health status
            requirements_status: Performance requirements status
            resource_utilization: Resource utilization metrics
            alert_summary: Alert summary
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Recommendations based on performance requirements
        if not requirements_status.get('meets_100ms_requirement', True):
            recommendations.append({
                'category': 'performance',
                'priority': 'high',
                'message': 'Feature extraction latency exceeds 100ms requirement. Consider optimizing model or increasing resources.'
            })
        
        # Recommendations based on resource utilization
        cpu_percent = resource_utilization.get('cpu_percent', 0)
        memory_mb = resource_utilization.get('memory_mb', 0)
        
        if cpu_percent > 80:
            recommendations.append({
                'category': 'resources',
                'priority': 'high',
                'message': f'CPU utilization is high ({cpu_percent:.1f}%). Consider scaling up or optimizing resource usage.'
            })
        
        if memory_mb > 800:
            recommendations.append({
                'category': 'resources',
                'priority': 'medium',
                'message': f'Memory usage is high ({memory_mb:.1f}MB). Consider optimizing memory usage or increasing limits.'
            })
        
        # Recommendations based on alerts
        critical_alerts = alert_summary.get('critical_alerts', 0)
        if critical_alerts > 0:
            recommendations.append({
                'category': 'alerts',
                'priority': 'critical',
                'message': f'There are {critical_alerts} critical alerts. Immediate attention required.'
            })
        
        return recommendations
    
    def get_alerts_dashboard(self, hours: int = 24) -> Dict[str, Any]:
        """Get alerts dashboard data.
        
        Args:
            hours: Number of hours to look back for alerts
            
        Returns:
            Dictionary with alerts dashboard data
        """
        # Get recent alerts
        alert_summary = self.alerting_system.get_alert_summary(hours)
        
        # Get alert history
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [
            {
                'timestamp': alert.timestamp.isoformat(),
                'type': alert.alert_type.value if hasattr(alert, 'alert_type') else 'unknown',
                'severity': alert.severity.value if hasattr(alert, 'severity') else 'unknown',
                'title': getattr(alert, 'title', 'Unknown Alert'),
                'message': getattr(alert, 'message', '')
            }
            for alert in self.alerting_system.alert_history 
            if hasattr(alert, 'timestamp') and alert.timestamp > cutoff_time
        ]
        
        # Sort by timestamp (newest first)
        recent_alerts.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'summary': alert_summary,
            'recent_alerts': recent_alerts[:50],  # Limit to last 50 alerts
            'total_recent_alerts': len(recent_alerts)
        }
    
    def get_resource_dashboard(self) -> Dict[str, Any]:
        """Get resource utilization dashboard.
        
        Returns:
            Dictionary with resource utilization dashboard data
        """
        # Get resource utilization
        resource_utilization = self.enhanced_metrics_collector.get_resource_utilization()
        
        # Get cache statistics
        cache_stats = self.cache_connection_integration._get_cache_statistics()
        
        # Get connection pool statistics
        pool_stats = self.cache_connection_integration._get_connection_pool_statistics()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'resource_utilization': resource_utilization,
            'cache_statistics': cache_stats,
            'connection_pool_statistics': pool_stats
        }
    
    def export_dashboard_data(self) -> str:
        """Export current dashboard data as JSON.
        
        Returns:
            JSON string with current dashboard data
        """
        try:
            # Get all dashboard data
            dashboard_data = {
                'real_time': asdict(self.get_real_time_dashboard_data()),
                'performance_trends': self.get_performance_trends(),
                'system_health': self.get_system_health_overview(),
                'alerts': self.get_alerts_dashboard(),
                'resources': self.get_resource_dashboard()
            }
            
            return json.dumps(dashboard_data, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error exporting dashboard data: {e}")
            return json.dumps({"error": str(e)}, indent=2)
    
    def reset_dashboard_data(self) -> None:
        """Reset all dashboard data."""
        # Reset all integrated systems
        self.enhanced_metrics_collector.reset_metrics()
        self.alerting_system.clear_alert_history()
        self.ray_integration.reset_monitoring()
        self.cache_connection_integration.reset_monitoring()
        
        logger.info("Dashboard data reset")