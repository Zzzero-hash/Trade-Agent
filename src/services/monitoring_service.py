"""
Production Monitoring Service

Provides comprehensive monitoring with Prometheus metrics, custom business metrics,
performance tracking, and integration with alerting systems.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager

import aiohttp
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from prometheus_client.exposition import MetricsHandler
from http.server import HTTPServer
import threading

from src.services.alert_service import ProductionAlertService


@dataclass
class MetricThreshold:
    metric_name: str
    threshold_value: float
    comparison: str  # 'gt', 'lt', 'eq'
    alert_level: str
    description: str


class PrometheusMetrics:
    """Prometheus metrics collection for the trading platform."""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # System metrics
        self.request_count = Counter(
            'trading_platform_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'trading_platform_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'trading_platform_active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        # Trading metrics
        self.trades_executed = Counter(
            'trading_platform_trades_executed_total',
            'Total number of trades executed',
            ['symbol', 'side', 'customer_id'],
            registry=self.registry
        )
        
        self.trade_latency = Histogram(
            'trading_platform_trade_latency_seconds',
            'Trade execution latency in seconds',
            ['broker', 'symbol'],
            registry=self.registry
        )
        
        self.position_count = Gauge(
            'trading_platform_active_positions',
            'Number of active positions',
            ['customer_id'],
            registry=self.registry
        )
        
        self.portfolio_value = Gauge(
            'trading_platform_portfolio_value',
            'Portfolio value in USD',
            ['customer_id'],
            registry=self.registry
        )
        
        self.portfolio_pnl = Gauge(
            'trading_platform_portfolio_pnl',
            'Portfolio P&L in USD',
            ['customer_id'],
            registry=self.registry
        )
        
        # Risk metrics
        self.risk_violations = Counter(
            'trading_platform_risk_violations_total',
            'Total number of risk violations',
            ['violation_type', 'customer_id'],
            registry=self.registry
        )
        
        self.stop_loss_executions = Counter(
            'trading_platform_stop_loss_executions_total',
            'Total number of stop-loss executions',
            ['symbol', 'customer_id'],
            registry=self.registry
        )
        
        self.circuit_breaker_activations = Counter(
            'trading_platform_circuit_breaker_activations_total',
            'Total number of circuit breaker activations',
            ['customer_id', 'reason'],
            registry=self.registry
        )
        
        # ML model metrics
        self.model_predictions = Counter(
            'trading_platform_model_predictions_total',
            'Total number of model predictions',
            ['model_name', 'symbol'],
            registry=self.registry
        )
        
        self.model_accuracy = Gauge(
            'trading_platform_model_accuracy',
            'Model prediction accuracy',
            ['model_name', 'timeframe'],
            registry=self.registry
        )
        
        self.feature_extraction_latency = Histogram(
            'trading_platform_feature_extraction_latency_seconds',
            'Feature extraction latency in seconds',
            ['symbol'],
            registry=self.registry
        )
        
        # External service metrics
        self.external_api_calls = Counter(
            'trading_platform_external_api_calls_total',
            'Total number of external API calls',
            ['service', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.external_api_latency = Histogram(
            'trading_platform_external_api_latency_seconds',
            'External API call latency in seconds',
            ['service', 'endpoint'],
            registry=self.registry
        )
        
        # Business metrics
        self.customer_signups = Counter(
            'trading_platform_customer_signups_total',
            'Total number of customer signups',
            ['source'],
            registry=self.registry
        )
        
        self.revenue = Gauge(
            'trading_platform_revenue_usd',
            'Revenue in USD',
            ['revenue_type'],
            registry=self.registry
        )
        
        self.customer_churn = Counter(
            'trading_platform_customer_churn_total',
            'Total number of churned customers',
            ['reason'],
            registry=self.registry
        )


class MonitoringService:
    """Production monitoring service with Prometheus integration."""
    
    def __init__(self, alert_service: ProductionAlertService, metrics_port: int = 8000):
        self.alert_service = alert_service
        self.metrics_port = metrics_port
        self.logger = logging.getLogger(__name__)
        
        # Initialize Prometheus metrics
        self.metrics = PrometheusMetrics()
        
        # Metric thresholds for alerting
        self.thresholds: List[MetricThreshold] = []
        self._setup_default_thresholds()
        
        # Performance tracking
        self.performance_data: Dict[str, List[float]] = {}
        
        # Metrics server
        self.metrics_server = None
        self.metrics_thread = None
        
    def _setup_default_thresholds(self):
        """Setup default metric thresholds for alerting."""
        self.thresholds = [
            MetricThreshold(
                "trading_platform_request_duration_seconds",
                2.0, "gt", "warning",
                "Request latency exceeds 2 seconds"
            ),
            MetricThreshold(
                "trading_platform_trade_latency_seconds",
                1.0, "gt", "critical",
                "Trade execution latency exceeds 1 second"
            ),
            MetricThreshold(
                "trading_platform_external_api_latency_seconds",
                5.0, "gt", "warning",
                "External API latency exceeds 5 seconds"
            ),
            MetricThreshold(
                "trading_platform_feature_extraction_latency_seconds",
                0.1, "gt", "warning",
                "Feature extraction latency exceeds 100ms"
            ),
        ]
    
    async def start_monitoring(self):
        """Start the monitoring service and metrics server."""
        self.logger.info("Starting production monitoring service")
        
        # Start Prometheus metrics server
        self._start_metrics_server()
        
        # Start monitoring tasks
        await asyncio.gather(
            self._performance_monitoring_loop(),
            self._threshold_monitoring_loop(),
            self._health_check_loop()
        )
    
    def _start_metrics_server(self):
        """Start Prometheus metrics HTTP server."""
        try:
            class MetricsHTTPHandler(MetricsHandler):
                def __init__(self, registry):
                    self.registry = registry
                
                def do_GET(self):
                    if self.path == '/metrics':
                        self.send_response(200)
                        self.send_header('Content-Type', 'text/plain; charset=utf-8')
                        self.end_headers()
                        self.wfile.write(generate_latest(self.registry))
                    else:
                        self.send_response(404)
                        self.end_headers()
            
            def run_server():
                handler = lambda *args: MetricsHTTPHandler(self.metrics.registry)(*args)
                self.metrics_server = HTTPServer(('', self.metrics_port), handler)
                self.metrics_server.serve_forever()
            
            self.metrics_thread = threading.Thread(target=run_server, daemon=True)
            self.metrics_thread.start()
            
            self.logger.info(f"Prometheus metrics server started on port {self.metrics_port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start metrics server: {e}")
    
    async def record_metric(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        try:
            labels = labels or {}
            
            # Map metric names to Prometheus metrics
            if metric_name == "request_count":
                self.metrics.request_count.labels(**labels).inc(value)
            elif metric_name == "request_duration":
                self.metrics.request_duration.labels(**labels).observe(value)
            elif metric_name == "trades_executed":
                self.metrics.trades_executed.labels(**labels).inc(value)
            elif metric_name == "trade_latency":
                self.metrics.trade_latency.labels(**labels).observe(value)
            elif metric_name == "portfolio_value":
                self.metrics.portfolio_value.labels(**labels).set(value)
            elif metric_name == "portfolio_pnl":
                self.metrics.portfolio_pnl.labels(**labels).set(value)
            elif metric_name == "risk_violations_total":
                self.metrics.risk_violations.labels(**labels).inc(value)
            elif metric_name == "stop_loss_executions_total":
                self.metrics.stop_loss_executions.labels(**labels).inc(value)
            elif metric_name == "model_predictions":
                self.metrics.model_predictions.labels(**labels).inc(value)
            elif metric_name == "model_accuracy":
                self.metrics.model_accuracy.labels(**labels).set(value)
            elif metric_name == "feature_extraction_latency_seconds":
                self.metrics.feature_extraction_latency.labels(**labels).observe(value)
            elif metric_name == "external_api_calls":
                self.metrics.external_api_calls.labels(**labels).inc(value)
            elif metric_name == "external_api_latency":
                self.metrics.external_api_latency.labels(**labels).observe(value)
            elif metric_name == "customer_signups":
                self.metrics.customer_signups.labels(**labels).inc(value)
            elif metric_name == "revenue":
                self.metrics.revenue.labels(**labels).set(value)
            elif metric_name == "customer_churn":
                self.metrics.customer_churn.labels(**labels).inc(value)
            else:
                # Generic metric recording
                self.logger.debug(f"Recording generic metric: {metric_name} = {value}")
            
            # Store for performance analysis
            if metric_name not in self.performance_data:
                self.performance_data[metric_name] = []
            self.performance_data[metric_name].append(value)
            
            # Keep only last 1000 data points
            if len(self.performance_data[metric_name]) > 1000:
                self.performance_data[metric_name] = self.performance_data[metric_name][-1000:]
                
        except Exception as e:
            self.logger.error(f"Failed to record metric {metric_name}: {e}")
    
    @asynccontextmanager
    async def track_request_duration(self, method: str, endpoint: str):
        """Context manager to track request duration."""
        start_time = time.time()
        try:
            yield
            status = "success"
        except Exception as e:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            await self.record_metric(
                "request_duration",
                duration,
                {"method": method, "endpoint": endpoint}
            )
            await self.record_metric(
                "request_count",
                1,
                {"method": method, "endpoint": endpoint, "status": status}
            )
    
    @asynccontextmanager
    async def track_trade_execution(self, broker: str, symbol: str):
        """Context manager to track trade execution latency."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            await self.record_metric(
                "trade_latency",
                duration,
                {"broker": broker, "symbol": symbol}
            )
    
    @asynccontextmanager
    async def track_external_api_call(self, service: str, endpoint: str):
        """Context manager to track external API call latency."""
        start_time = time.time()
        try:
            yield
            status = "success"
        except Exception as e:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            await self.record_metric(
                "external_api_latency",
                duration,
                {"service": service, "endpoint": endpoint}
            )
            await self.record_metric(
                "external_api_calls",
                1,
                {"service": service, "endpoint": endpoint, "status": status}
            )
    
    async def _performance_monitoring_loop(self):
        """Monitor performance metrics and detect anomalies."""
        while True:
            try:
                # Analyze performance trends
                for metric_name, values in self.performance_data.items():
                    if len(values) >= 10:  # Need minimum data points
                        await self._analyze_performance_trend(metric_name, values)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _analyze_performance_trend(self, metric_name: str, values: List[float]):
        """Analyze performance trends and detect anomalies."""
        try:
            recent_values = values[-10:]  # Last 10 values
            historical_values = values[-100:-10] if len(values) >= 100 else values[:-10]
            
            if not historical_values:
                return
            
            # Calculate statistics
            recent_avg = sum(recent_values) / len(recent_values)
            historical_avg = sum(historical_values) / len(historical_values)
            
            # Detect significant performance degradation (>50% increase)
            if recent_avg > historical_avg * 1.5:
                await self.alert_service.send_warning_alert(
                    f"Performance degradation detected",
                    f"Metric {metric_name}: recent avg {recent_avg:.3f} vs historical avg {historical_avg:.3f}"
                )
                
        except Exception as e:
            self.logger.error(f"Performance trend analysis error for {metric_name}: {e}")
    
    async def _threshold_monitoring_loop(self):
        """Monitor metric thresholds and trigger alerts."""
        while True:
            try:
                # Check each threshold
                for threshold in self.thresholds:
                    await self._check_metric_threshold(threshold)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Threshold monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _check_metric_threshold(self, threshold: MetricThreshold):
        """Check if a metric exceeds its threshold."""
        try:
            # Get recent values for the metric
            if threshold.metric_name not in self.performance_data:
                return
            
            values = self.performance_data[threshold.metric_name]
            if not values:
                return
            
            # Check last 5 values
            recent_values = values[-5:]
            avg_value = sum(recent_values) / len(recent_values)
            
            # Check threshold
            threshold_exceeded = False
            if threshold.comparison == "gt" and avg_value > threshold.threshold_value:
                threshold_exceeded = True
            elif threshold.comparison == "lt" and avg_value < threshold.threshold_value:
                threshold_exceeded = True
            elif threshold.comparison == "eq" and abs(avg_value - threshold.threshold_value) < 0.001:
                threshold_exceeded = True
            
            if threshold_exceeded:
                if threshold.alert_level == "critical":
                    await self.alert_service.send_critical_alert(
                        f"Critical threshold exceeded: {threshold.metric_name}",
                        f"{threshold.description}. Current value: {avg_value:.3f}, Threshold: {threshold.threshold_value}"
                    )
                else:
                    await self.alert_service.send_warning_alert(
                        f"Warning threshold exceeded: {threshold.metric_name}",
                        f"{threshold.description}. Current value: {avg_value:.3f}, Threshold: {threshold.threshold_value}"
                    )
                    
        except Exception as e:
            self.logger.error(f"Threshold check error for {threshold.metric_name}: {e}")
    
    async def _health_check_loop(self):
        """Perform system health checks."""
        while True:
            try:
                # Check system health indicators
                health_status = await self._perform_health_checks()
                
                # Record health metrics
                await self.record_metric("system_health_score", health_status["score"])
                
                # Alert on poor health
                if health_status["score"] < 0.8:
                    await self.alert_service.send_warning_alert(
                        "System health degraded",
                        f"Health score: {health_status['score']:.2f}. Issues: {health_status['issues']}"
                    )
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(300)
    
    async def _perform_health_checks(self) -> Dict[str, Any]:
        """Perform comprehensive system health checks."""
        health_score = 1.0
        issues = []
        
        try:
            # Check database connectivity
            # (This would be implemented with actual database checks)
            db_healthy = True  # Placeholder
            if not db_healthy:
                health_score -= 0.3
                issues.append("Database connectivity issues")
            
            # Check external API availability
            # (This would check actual broker APIs)
            apis_healthy = True  # Placeholder
            if not apis_healthy:
                health_score -= 0.2
                issues.append("External API issues")
            
            # Check memory usage
            # (This would check actual system resources)
            memory_ok = True  # Placeholder
            if not memory_ok:
                health_score -= 0.1
                issues.append("High memory usage")
            
            # Check disk space
            disk_ok = True  # Placeholder
            if not disk_ok:
                health_score -= 0.1
                issues.append("Low disk space")
            
        except Exception as e:
            health_score = 0.0
            issues.append(f"Health check failed: {str(e)}")
        
        return {
            "score": max(0.0, health_score),
            "issues": issues,
            "timestamp": datetime.utcnow()
        }
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        summary = {}
        
        for metric_name, values in self.performance_data.items():
            if values:
                summary[metric_name] = {
                    "current": values[-1],
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        
        return summary
    
    def add_threshold(self, threshold: MetricThreshold):
        """Add a new metric threshold for monitoring."""
        self.thresholds.append(threshold)
        self.logger.info(f"Added threshold for {threshold.metric_name}")
    
    def stop_monitoring(self):
        """Stop the monitoring service."""
        if self.metrics_server:
            self.metrics_server.shutdown()
        self.logger.info("Monitoring service stopped")