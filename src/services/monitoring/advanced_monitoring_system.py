"""
Advanced monitoring and alerting system with real-time anomaly detection,
model drift detection, performance degradation alerts, and predictive maintenance.
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from scipy import stats
import logging

from src.models.monitoring import (
    Alert, AlertSeverity, DriftType, DriftDetectionResult, 
    ModelPerformanceMetrics
)
from src.services.monitoring.drift_strategies import DriftDetectionContext
from src.services.monitoring.alert_system import AlertSubject, AlertFactory
from src.services.monitoring.performance_tracker import PerformanceTracker
from src.utils.logging import get_logger

logger = get_logger("advanced_monitoring")


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection."""
    z_score_threshold: float = 3.0
    isolation_forest_contamination: float = 0.1
    statistical_window_size: int = 100
    min_samples_for_detection: int = 30
    enable_real_time_alerts: bool = True


@dataclass
class ModelDriftConfig:
    """Configuration for model drift detection."""
    data_drift_threshold: float = 0.05
    performance_drift_threshold: float = 0.1
    concept_drift_threshold: float = 0.15
    statistical_significance_level: float = 0.05
    drift_detection_window: int = 50
    enable_statistical_tests: bool = True


@dataclass
class PerformanceDegradationConfig:
    """Configuration for performance degradation monitoring."""
    accuracy_threshold: float = 0.05
    precision_threshold: float = 0.05
    recall_threshold: float = 0.05
    f1_threshold: float = 0.05
    sharpe_ratio_threshold: float = 0.2
    max_drawdown_threshold: float = 0.1
    consecutive_failures_threshold: int = 5
    enable_automated_retraining: bool = True


@dataclass
class SystemHealthConfig:
    """Configuration for system health monitoring."""
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    disk_threshold: float = 90.0
    response_time_threshold: float = 5.0
    error_rate_threshold: float = 0.05
    enable_predictive_maintenance: bool = True


@dataclass
class AlertRoutingConfig:
    """Configuration for alert routing and escalation."""
    escalation_levels: List[str] = field(default_factory=lambda: ["team", "manager", "executive"])
    escalation_timeouts: List[int] = field(default_factory=lambda: [15, 30, 60])  # minutes
    severity_routing: Dict[str, List[str]] = field(default_factory=lambda: {
        "low": ["team"],
        "medium": ["team", "manager"],
        "high": ["team", "manager"],
        "critical": ["team", "manager", "executive"]
    })


class RealTimeAnomalyDetector:
    """Real-time anomaly detection with multiple statistical methods."""
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.data_windows: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=config.statistical_window_size)
        )
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        
    async def detect_anomalies(
        self, 
        model_name: str, 
        data_point: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in real-time data."""
        anomalies = []
        
        for feature_name, value in data_point.items():
            if not isinstance(value, (int, float)):
                continue
                
            key = f"{model_name}_{feature_name}"
            self.data_windows[key].append(value)
            
            if len(self.data_windows[key]) >= self.config.min_samples_for_detection:
                # Z-score based detection
                z_score_anomaly = self._detect_z_score_anomaly(key, value)
                if z_score_anomaly:
                    anomalies.append(z_score_anomaly)
                
                # Statistical change point detection
                change_point = self._detect_change_point(key)
                if change_point:
                    anomalies.append(change_point)
                
                # Update baseline statistics
                self._update_baseline_stats(key)
        
        return anomalies
    
    def _detect_z_score_anomaly(self, key: str, value: float) -> Optional[Dict[str, Any]]:
        """Detect anomaly using Z-score method."""
        window_data = list(self.data_windows[key])
        
        if len(window_data) < self.config.min_samples_for_detection:
            return None
        
        mean = np.mean(window_data[:-1])  # Exclude current value
        std = np.std(window_data[:-1])
        
        if std == 0:
            return None
        
        z_score = abs((value - mean) / std)
        
        if z_score > self.config.z_score_threshold:
            return {
                'type': 'z_score_anomaly',
                'key': key,
                'value': value,
                'z_score': z_score,
                'threshold': self.config.z_score_threshold,
                'mean': mean,
                'std': std,
                'severity': self._calculate_anomaly_severity(z_score)
            }
        
        return None
    
    def _detect_change_point(self, key: str) -> Optional[Dict[str, Any]]:
        """Detect statistical change points using CUSUM."""
        window_data = np.array(list(self.data_windows[key]))
        
        if len(window_data) < self.config.min_samples_for_detection:
            return None
        
        # Simple CUSUM implementation
        mean = np.mean(window_data)
        cusum_pos = np.zeros(len(window_data))
        cusum_neg = np.zeros(len(window_data))
        
        for i in range(1, len(window_data)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + window_data[i] - mean - 0.5)
            cusum_neg[i] = max(0, cusum_neg[i-1] - window_data[i] + mean - 0.5)
        
        # Check for change point
        threshold = 3 * np.std(window_data)
        if cusum_pos[-1] > threshold or cusum_neg[-1] > threshold:
            return {
                'type': 'change_point',
                'key': key,
                'cusum_pos': cusum_pos[-1],
                'cusum_neg': cusum_neg[-1],
                'threshold': threshold,
                'severity': 'medium'
            }
        
        return None
    
    def _update_baseline_stats(self, key: str) -> None:
        """Update baseline statistics for a feature."""
        window_data = list(self.data_windows[key])
        self.baseline_stats[key] = {
            'mean': np.mean(window_data),
            'std': np.std(window_data),
            'min': np.min(window_data),
            'max': np.max(window_data),
            'last_updated': datetime.now().timestamp()
        }
    
    def _calculate_anomaly_severity(self, z_score: float) -> str:
        """Calculate severity based on Z-score magnitude."""
        if z_score > 5:
            return 'critical'
        elif z_score > 4:
            return 'high'
        elif z_score > 3:
            return 'medium'
        else:
            return 'low'


class StatisticalDriftDetector:
    """Enhanced drift detection with statistical significance testing."""
    
    def __init__(self, config: ModelDriftConfig):
        self.config = config
        self.drift_context = DriftDetectionContext()
        
    async def detect_model_drift(
        self, 
        model_name: str, 
        reference_data: np.ndarray,
        current_data: np.ndarray
    ) -> Optional[DriftDetectionResult]:
        """Detect model drift with statistical significance testing."""
        
        if len(reference_data) < self.config.drift_detection_window or \
           len(current_data) < self.config.drift_detection_window:
            return None
        
        # Kolmogorov-Smirnov test for distribution drift
        ks_statistic, ks_p_value = stats.ks_2samp(reference_data.flatten(), current_data.flatten())
        
        # Mann-Whitney U test for median shift
        mw_statistic, mw_p_value = stats.mannwhitneyu(
            reference_data.flatten(), 
            current_data.flatten(),
            alternative='two-sided'
        )
        
        # Population Stability Index (PSI)
        psi_score = self._calculate_psi(reference_data, current_data)
        
        # Determine if drift is detected
        drift_detected = (
            ks_p_value < self.config.statistical_significance_level or
            mw_p_value < self.config.statistical_significance_level or
            psi_score > self.config.data_drift_threshold
        )
        
        if drift_detected:
            severity = self._calculate_drift_severity(ks_p_value, mw_p_value, psi_score)
            
            return DriftDetectionResult(
                drift_type=DriftType.DATA_DRIFT,
                severity=severity,
                drift_score=max(ks_statistic, psi_score),
                threshold=self.config.data_drift_threshold,
                detected=True,
                timestamp=datetime.now(),
                details={
                    'ks_statistic': ks_statistic,
                    'ks_p_value': ks_p_value,
                    'mw_statistic': float(mw_statistic),
                    'mw_p_value': mw_p_value,
                    'psi_score': psi_score,
                    'reference_samples': len(reference_data),
                    'current_samples': len(current_data)
                }
            )
        
        return None
    
    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index."""
        try:
            # Create bins based on reference data
            bin_edges = np.histogram_bin_edges(reference.flatten(), bins=bins)
            
            # Calculate distributions
            ref_counts, _ = np.histogram(reference.flatten(), bins=bin_edges)
            cur_counts, _ = np.histogram(current.flatten(), bins=bin_edges)
            
            # Convert to proportions
            ref_props = ref_counts / len(reference.flatten())
            cur_props = cur_counts / len(current.flatten())
            
            # Avoid division by zero
            ref_props = np.where(ref_props == 0, 0.0001, ref_props)
            cur_props = np.where(cur_props == 0, 0.0001, cur_props)
            
            # Calculate PSI
            psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
            
            return psi
        except Exception as e:
            logger.warning(f"Error calculating PSI: {e}")
            return 0.0
    
    def _calculate_drift_severity(self, ks_p: float, mw_p: float, psi: float) -> AlertSeverity:
        """Calculate drift severity based on statistical tests."""
        min_p_value = min(ks_p, mw_p)
        
        if min_p_value < 0.001 or psi > 0.25:
            return AlertSeverity.CRITICAL
        elif min_p_value < 0.01 or psi > 0.1:
            return AlertSeverity.HIGH
        elif min_p_value < 0.05 or psi > 0.05:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW


class PerformanceDegradationMonitor:
    """Monitor for performance degradation with automated retraining triggers."""
    
    def __init__(
        self,
        config: PerformanceDegradationConfig,
        retraining_handler: Optional[Callable[[str, Dict[str, Any]], Awaitable[Optional[str]]]] = None,
    ):
        self.config = config
        self.performance_history: Dict[str, List[ModelPerformanceMetrics]] = defaultdict(list)
        self.consecutive_failures: Dict[str, int] = defaultdict(int)
        self.retraining_triggers: Dict[str, datetime] = {}
        self.retraining_handler = retraining_handler
        
    def set_retraining_handler(
        self,
        retraining_handler: Callable[[str, Dict[str, Any]], Awaitable[Optional[str]]]
    ) -> None:
        """Register or replace the automated retraining handler."""
        self.retraining_handler = retraining_handler
        
    async def check_performance_degradation(
        self, 
        model_name: str, 
        current_metrics: ModelPerformanceMetrics,
        baseline_metrics: Optional[ModelPerformanceMetrics] = None
    ) -> List[Alert]:
        """Check for performance degradation and generate alerts."""
        alerts = []
        
        # Store current metrics
        self.performance_history[model_name].append(current_metrics)
        
        if baseline_metrics is None:
            baseline_metrics = self._get_baseline_metrics(model_name)
        
        if baseline_metrics is None:
            return alerts
        
        # Check individual metric degradations
        degradations = self._calculate_degradations(current_metrics, baseline_metrics)
        
        for metric_name, degradation in degradations.items():
            threshold = getattr(self.config, f"{metric_name}_threshold", 0.05)
            
            if degradation > threshold:
                severity = self._calculate_degradation_severity(degradation, threshold)
                
                alert = Alert(
                    id=f"performance_degradation_{model_name}_{metric_name}_{datetime.now().timestamp()}",
                    severity=severity,
                    title=f"Performance Degradation: {metric_name.replace('_', ' ').title()}",
                    message=f"Model {model_name} {metric_name} degraded by {degradation:.2%} "
                           f"(threshold: {threshold:.2%})",
                    timestamp=datetime.now(),
                    model_name=model_name,
                    metric_name=metric_name,
                    current_value=getattr(current_metrics, metric_name),
                    threshold=getattr(baseline_metrics, metric_name) * (1 - threshold),
                    metadata={
                        'degradation_percentage': degradation,
                        'baseline_value': getattr(baseline_metrics, metric_name),
                        'current_value': getattr(current_metrics, metric_name)
                    }
                )
                alerts.append(alert)
                
                # Track consecutive failures
                self.consecutive_failures[model_name] += 1
        
        # Check for automated retraining trigger
        failure_count = self.consecutive_failures[model_name]
        if failure_count >= self.config.consecutive_failures_threshold:
            if self.config.enable_automated_retraining:
                await self._trigger_automated_retraining(model_name, degradations, failure_count)

            # Reset counter after triggering
            self.consecutive_failures[model_name] = 0
        
        # Reset consecutive failures if performance is good
        if not alerts:
            self.consecutive_failures[model_name] = 0
        
        return alerts
    
    def _get_baseline_metrics(self, model_name: str) -> Optional[ModelPerformanceMetrics]:
        """Get baseline metrics for comparison."""
        history = self.performance_history[model_name]
        if len(history) < 10:  # Need sufficient history
            return None
        
        # Use best performance from recent history as baseline
        recent_history = history[-50:]  # Last 50 measurements
        return max(recent_history, key=lambda m: m.f1_score)
    
    def _calculate_degradations(
        self, 
        current: ModelPerformanceMetrics, 
        baseline: ModelPerformanceMetrics
    ) -> Dict[str, float]:
        """Calculate performance degradations."""
        degradations = {}
        
        metrics_to_check = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for metric in metrics_to_check:
            current_value = getattr(current, metric, 0)
            baseline_value = getattr(baseline, metric, 0)
            
            if baseline_value > 0:
                degradation = (baseline_value - current_value) / baseline_value
                degradations[metric] = max(0, degradation)  # Only positive degradations
        
        # Trading-specific metrics
        if current.sharpe_ratio is not None and baseline.sharpe_ratio is not None:
            if baseline.sharpe_ratio > 0:
                sharpe_degradation = (baseline.sharpe_ratio - current.sharpe_ratio) / baseline.sharpe_ratio
                degradations['sharpe_ratio'] = max(0, sharpe_degradation)
        
        if current.max_drawdown is not None and baseline.max_drawdown is not None:
            # For drawdown, higher is worse, so we check if current > baseline
            if baseline.max_drawdown > 0:
                drawdown_increase = (current.max_drawdown - baseline.max_drawdown) / baseline.max_drawdown
                degradations['max_drawdown'] = max(0, drawdown_increase)
        
        return degradations
    
    def _calculate_degradation_severity(self, degradation: float, threshold: float) -> AlertSeverity:
        """Calculate severity based on degradation magnitude."""
        if degradation > threshold * 3:
            return AlertSeverity.CRITICAL
        elif degradation > threshold * 2:
            return AlertSeverity.HIGH
        elif degradation > threshold * 1.5:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    async def _trigger_automated_retraining(
        self,
        model_name: str,
        degradations: Dict[str, float],
        failure_count: int,
    ) -> None:
        """Trigger automated retraining for a model."""
        if model_name in self.retraining_triggers:
            last_trigger = self.retraining_triggers[model_name]
            if datetime.now() - last_trigger < timedelta(hours=6):
                logger.info('Skipping retraining for %s due to cooldown', model_name)
                return

        if not self.retraining_handler:
            logger.warning('No retraining handler configured; skipping automated retraining for %s', model_name)
            return

        triggered_at = datetime.now()
        history = self.performance_history.get(model_name, [])
        latest_metrics = history[-1] if history else None

        payload: Dict[str, Any] = {
            'reason': 'performance_degradation',
            'triggered_at': triggered_at,
            'consecutive_failures': failure_count,
            'degradations': degradations,
        }
        if isinstance(latest_metrics, ModelPerformanceMetrics):
            payload['latest_metrics'] = asdict(latest_metrics)

        try:
            job_id = await self.retraining_handler(model_name, payload)
        except Exception as exc:
            logger.error('Failed to schedule automated retraining for %s: %s', model_name, exc, exc_info=True)
            return

        self.retraining_triggers[model_name] = triggered_at
        if job_id:
            logger.info('Automated retraining scheduled for %s (job_id=%s)', model_name, job_id)
        else:
            logger.info('Automated retraining triggered for %s', model_name)



class SystemHealthMonitor:
    """Comprehensive system health monitoring with predictive maintenance."""
    
    def __init__(self, config: SystemHealthConfig):
        self.config = config
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.predictive_models: Dict[str, Any] = {}
        
    async def monitor_system_health(self) -> List[Alert]:
        """Monitor overall system health and generate alerts."""
        alerts = []
        
        # Collect current health metrics
        health_metrics = await self._collect_health_metrics()
        
        # Store metrics in history
        for metric_name, value in health_metrics.items():
            self.health_history[metric_name].append({
                'timestamp': datetime.now(),
                'value': value
            })
        
        # Check thresholds
        threshold_alerts = self._check_health_thresholds(health_metrics)
        alerts.extend(threshold_alerts)
        
        # Predictive maintenance
        if self.config.enable_predictive_maintenance:
            predictive_alerts = await self._run_predictive_maintenance()
            alerts.extend(predictive_alerts)
        
        return alerts
    
    async def _collect_health_metrics(self) -> Dict[str, float]:
        """Collect system health metrics."""
        import psutil
        
        try:
            metrics = {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'network_io_bytes': psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv,
                'process_count': len(psutil.pids()),
                'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Error collecting health metrics: {e}")
            return {}
    
    def _check_health_thresholds(self, metrics: Dict[str, float]) -> List[Alert]:
        """Check health metrics against thresholds."""
        alerts = []
        
        threshold_checks = [
            ('cpu_percent', self.config.cpu_threshold, 'CPU Usage'),
            ('memory_percent', self.config.memory_threshold, 'Memory Usage'),
            ('disk_percent', self.config.disk_threshold, 'Disk Usage')
        ]
        
        for metric_name, threshold, display_name in threshold_checks:
            value = metrics.get(metric_name, 0)
            
            if value > threshold:
                severity = AlertSeverity.CRITICAL if value > threshold * 1.1 else AlertSeverity.HIGH
                
                alert = Alert(
                    id=f"system_health_{metric_name}_{datetime.now().timestamp()}",
                    severity=severity,
                    title=f"System Health Alert: {display_name}",
                    message=f"{display_name} is {value:.1f}% (threshold: {threshold}%)",
                    timestamp=datetime.now(),
                    metric_name=metric_name,
                    current_value=value,
                    threshold=threshold
                )
                alerts.append(alert)
        
        return alerts
    
    async def _run_predictive_maintenance(self) -> List[Alert]:
        """Run predictive maintenance analysis."""
        alerts = []
        
        # Simple trend analysis for predictive maintenance
        for metric_name, history in self.health_history.items():
            if len(history) < 20:  # Need sufficient history
                continue
            
            # Extract values and timestamps
            values = [entry['value'] for entry in history]
            timestamps = [entry['timestamp'].timestamp() for entry in history]
            
            # Simple linear regression to detect trends
            if len(values) >= 10:
                trend_alert = self._analyze_trend(metric_name, values, timestamps)
                if trend_alert:
                    alerts.append(trend_alert)
        
        return alerts
    
    def _analyze_trend(self, metric_name: str, values: List[float], timestamps: List[float]) -> Optional[Alert]:
        """Analyze trend for predictive maintenance."""
        try:
            # Simple linear regression
            x = np.array(timestamps)
            y = np.array(values)
            
            # Normalize timestamps
            x = x - x[0]
            
            # Calculate slope
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Check if trend is significant and concerning
            if abs(r_value) > 0.7 and p_value < 0.05:  # Strong correlation
                # Predict future value (1 hour ahead)
                future_time = x[-1] + 3600  # 1 hour in seconds
                predicted_value = slope * future_time + intercept
                
                # Check if predicted value exceeds thresholds
                threshold_map = {
                    'cpu_percent': self.config.cpu_threshold,
                    'memory_percent': self.config.memory_threshold,
                    'disk_percent': self.config.disk_threshold
                }
                
                threshold = threshold_map.get(metric_name)
                if threshold and predicted_value > threshold:
                    return Alert(
                        id=f"predictive_maintenance_{metric_name}_{datetime.now().timestamp()}",
                        severity=AlertSeverity.MEDIUM,
                        title=f"Predictive Maintenance Alert: {metric_name.replace('_', ' ').title()}",
                        message=f"Trend analysis predicts {metric_name} will exceed threshold "
                               f"({threshold}%) in approximately 1 hour. "
                               f"Predicted value: {predicted_value:.1f}%",
                        timestamp=datetime.now(),
                        metric_name=metric_name,
                        current_value=values[-1],
                        threshold=threshold,
                        metadata={
                            'trend_slope': slope,
                            'correlation': r_value,
                            'p_value': p_value,
                            'predicted_value': predicted_value,
                            'prediction_horizon_hours': 1
                        }
                    )
        except Exception as e:
            logger.warning(f"Error in trend analysis for {metric_name}: {e}")
        
        return None


class AlertRoutingSystem:
    """Advanced alert routing and escalation system."""
    
    def __init__(self, config: AlertRoutingConfig, alert_subject: AlertSubject):
        self.config = config
        self.alert_subject = alert_subject
        self.escalation_timers: Dict[str, datetime] = {}
        self.acknowledged_alerts: Dict[str, datetime] = {}
        
    async def route_alert(self, alert: Alert) -> None:
        """Route alert based on severity and escalation rules."""
        # Get routing rules for severity
        routing_rules = self.config.severity_routing.get(alert.severity.value, ["team"])
        
        # Send to initial recipients
        await self._send_to_recipients(alert, routing_rules[0:1])
        
        # Set up escalation if needed
        if len(routing_rules) > 1:
            await self._setup_escalation(alert, routing_rules[1:])
    
    async def _send_to_recipients(self, alert: Alert, recipients: List[str]) -> None:
        """Send alert to specified recipients."""
        # Notify through alert subject
        await self.alert_subject.notify_observers(alert)
        
        logger.info(f"Alert {alert.id} sent to recipients: {recipients}")
    
    async def _setup_escalation(self, alert: Alert, escalation_recipients: List[str]) -> None:
        """Set up escalation timers for alert."""
        self.escalation_timers[alert.id] = datetime.now()
        
        # Schedule escalation check
        asyncio.create_task(self._handle_escalation(alert, escalation_recipients))
    
    async def _handle_escalation(self, alert: Alert, escalation_recipients: List[str]) -> None:
        """Handle alert escalation after timeout."""
        for i, recipients_level in enumerate(escalation_recipients):
            # Wait for escalation timeout
            timeout_minutes = self.config.escalation_timeouts[min(i, len(self.config.escalation_timeouts) - 1)]
            await asyncio.sleep(timeout_minutes * 60)
            
            # Check if alert was acknowledged
            if alert.id in self.acknowledged_alerts:
                logger.info(f"Alert {alert.id} was acknowledged, stopping escalation")
                break
            
            # Escalate to next level
            escalated_alert = Alert(
                id=f"{alert.id}_escalated_{i+1}",
                severity=alert.severity,
                title=f"[ESCALATED] {alert.title}",
                message=f"ESCALATION LEVEL {i+1}: {alert.message}",
                timestamp=datetime.now(),
                model_name=alert.model_name,
                metric_name=alert.metric_name,
                current_value=alert.current_value,
                threshold=alert.threshold,
                metadata={**alert.metadata, 'escalation_level': i+1, 'original_alert_id': alert.id}
            )
            
            await self._send_to_recipients(escalated_alert, [recipients_level])
    
    def acknowledge_alert(self, alert_id: str) -> None:
        """Acknowledge an alert to stop escalation."""
        self.acknowledged_alerts[alert_id] = datetime.now()
        logger.info(f"Alert {alert_id} acknowledged")


class AdvancedMonitoringSystem:
    """Main orchestrator for advanced monitoring and alerting."""
    
    def __init__(
        self,
        anomaly_config: AnomalyDetectionConfig,
        drift_config: ModelDriftConfig,
        performance_config: PerformanceDegradationConfig,
        health_config: SystemHealthConfig,
        routing_config: AlertRoutingConfig,
        retraining_handler: Optional[Callable[[str, Dict[str, Any]], Awaitable[Optional[str]]]] = None,
    ):
        self.anomaly_detector = RealTimeAnomalyDetector(anomaly_config)
        self.drift_detector = StatisticalDriftDetector(drift_config)
        self.performance_monitor = PerformanceDegradationMonitor(
            performance_config, retraining_handler=retraining_handler
        )
        self.health_monitor = SystemHealthMonitor(health_config)
        
        self.alert_subject = AlertSubject()
        self.routing_system = AlertRoutingSystem(routing_config, self.alert_subject)
        self._retraining_handler = retraining_handler
        
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self) -> None:
        """Start the monitoring system."""
        if self.monitoring_active:
            logger.warning("Monitoring system is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Advanced monitoring system started")
    
    async def stop_monitoring(self) -> None:
        """Stop the monitoring system."""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Advanced monitoring system stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # System health monitoring
                health_alerts = await self.health_monitor.monitor_system_health()
                for alert in health_alerts:
                    await self.routing_system.route_alert(alert)
                
                # Sleep before next cycle
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def process_model_data(
        self, 
        model_name: str, 
        data_point: Dict[str, Any],
        performance_metrics: Optional[ModelPerformanceMetrics] = None,
        reference_data: Optional[np.ndarray] = None,
        current_data: Optional[np.ndarray] = None
    ) -> None:
        """Process model data for monitoring."""
        try:
            # Real-time anomaly detection
            anomalies = await self.anomaly_detector.detect_anomalies(model_name, data_point)
            for anomaly in anomalies:
                alert = self._create_anomaly_alert(model_name, anomaly)
                await self.routing_system.route_alert(alert)
            
            # Model drift detection
            if reference_data is not None and current_data is not None:
                drift_result = await self.drift_detector.detect_model_drift(
                    model_name, reference_data, current_data
                )
                if drift_result and drift_result.detected:
                    alert = AlertFactory.create_drift_alert(model_name, drift_result)
                    await self.routing_system.route_alert(alert)
            
            # Performance degradation monitoring
            if performance_metrics:
                degradation_alerts = await self.performance_monitor.check_performance_degradation(
                    model_name, performance_metrics
                )
                for alert in degradation_alerts:
                    await self.routing_system.route_alert(alert)
        
        except Exception as e:
            logger.error(f"Error processing model data for {model_name}: {e}")
    
    def _create_anomaly_alert(self, model_name: str, anomaly: Dict[str, Any]) -> Alert:
        """Create alert from anomaly detection result."""
        severity_map = {
            'low': AlertSeverity.LOW,
            'medium': AlertSeverity.MEDIUM,
            'high': AlertSeverity.HIGH,
            'critical': AlertSeverity.CRITICAL
        }
        
        severity = severity_map.get(anomaly.get('severity', 'medium'), AlertSeverity.MEDIUM)
        
        return Alert(
            id=f"anomaly_{model_name}_{anomaly['type']}_{datetime.now().timestamp()}",
            severity=severity,
            title=f"Anomaly Detected: {anomaly['type'].replace('_', ' ').title()}",
            message=f"Anomaly detected in {model_name} for {anomaly['key']}: "
                   f"value={anomaly['value']:.4f}",
            timestamp=datetime.now(),
            model_name=model_name,
            metadata=anomaly
        )
    
    def add_alert_observer(self, observer) -> None:
        """Add an alert observer."""
        self.alert_subject.attach(observer)
    
    def register_retraining_handler(
        self,
        retraining_handler: Callable[[str, Dict[str, Any]], Awaitable[Optional[str]]],
    ) -> None:
        """Register a retraining handler for automated scheduling.

        Handlers receive `(model_name, payload)` where `payload` includes
        the degradation summary produced by :class:PerformanceDegradationMonitor.
        The payload contains keys such as `reason`, `consecutive_failures`,
        `degradations` (metric deltas), `triggered_at` (datetime), and an
        optional `latest_metrics` snapshot for auditing.
        """
        self._retraining_handler = retraining_handler
        self.performance_monitor.set_retraining_handler(retraining_handler)

    def acknowledge_alert(self, alert_id: str) -> None:
        """Acknowledge an alert."""
        self.routing_system.acknowledge_alert(alert_id)