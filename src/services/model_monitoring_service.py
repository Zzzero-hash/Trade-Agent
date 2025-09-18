"""
Model Monitoring and Alerting Service

This service provides comprehensive monitoring for model performance,
drift detection, and automated retraining triggers.

Refactored version using improved architecture with:
- Separation of concerns
- Strategy pattern for drift detection
- Observer pattern for alerts
- Better resource management
- Improved error handling
"""

import asyncio
import numpy as np
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from src.utils.logging import get_logger
from src.utils.monitoring import get_metrics_collector, MetricsCollector
from src.config.settings import get_settings
from src.models.trading_signal import TradingSignal
from src.models.market_data import MarketData
from src.models.monitoring import (
    Alert,
    AlertSeverity,
    DriftDetectionResult,
    DriftType,
    ModelPerformanceMetrics,
)

# Import refactored components
from .monitoring.performance_tracker import PerformanceTracker
from .monitoring.drift_strategies import DriftDetectionContext
from .monitoring.alert_system import AlertSubject, AlertFactory
from .monitoring.resource_manager import MonitoringResourceManager
from .monitoring.config import MonitoringConfig, ConfigManager
from .monitoring.exceptions import (
    MonitoringError, 
    InsufficientDataError, 
    DriftDetectionError
)
from src.ml.feature_extraction.monitoring import FeatureExtractionPerformanceMonitor
from src.ml.feature_extraction.alerting import FeatureExtractionAlertingSystem

logger = get_logger("model_monitoring")

class ModelMonitoringService:
    """
    Refactored model monitoring service with improved architecture.
    
    Uses composition and dependency injection for better testability
    and maintainability.
    """

    def __init__(
        self, 
        config: Optional[MonitoringConfig] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        # Configuration management
        self.config_manager = ConfigManager(config)
        self.settings = get_settings()
        self.metrics_collector = metrics_collector or get_metrics_collector()
        
        # Core components using dependency injection
        self.performance_tracker = PerformanceTracker()
        self.drift_detector = DriftDetectionContext()
        self.alert_system = AlertSubject()
        self.resource_manager = MonitoringResourceManager(
            max_buffer_size=self.config_manager.config.max_prediction_history
        )
        self.feature_extraction_monitor = FeatureExtractionPerformanceMonitor(self.alert_system)
        self.feature_extraction_alerter = FeatureExtractionAlertingSystem(
            metrics_collector=self.feature_extraction_monitor.metrics_collector, # Assuming metrics_collector is accessible
            alert_subject=self.alert_system
        )
        
        # Retraining callbacks
        self.retraining_callbacks: Dict[str, Callable[[str, Any], None]] = {}
        
        logger.info("Model monitoring service initialized with improved architecture")
        self._register_default_alert_observers()

    def _register_default_alert_observers(self) -> None:
        """Register default alert observers."""
        # Example: Register EmailAlertObserver
        # In a real application, configuration for these would come from settings
        email_config = self.settings.email_alerts
        if email_config.enabled:
            email_observer = EmailAlertObserver(
                smtp_config=email_config.smtp_settings,
                recipient_email=email_config.recipient_email
            )
            self.register_alert_observer(email_observer)
        
        # Register MetricsAlertObserver to ensure alerts are recorded as metrics
        metrics_observer = MetricsAlertObserver(self.metrics_collector)
        self.register_alert_observer(metrics_observer)
        
        logger.info("Default alert observers registered.")

    async def start(self) -> None:
        """Start the monitoring service."""
        await self.resource_manager.start()
        logger.info("Model monitoring service started")
    
    async def stop(self) -> None:
        """Stop the monitoring service."""
        await self.resource_manager.stop()
        logger.info("Model monitoring service stopped")
    
    def register_alert_observer(self, observer) -> None:
        """Register an alert observer using Observer pattern."""
        self.alert_system.attach(observer)
        logger.info(f"Alert observer registered: {observer.__class__.__name__}")

    def register_retraining_callback(self, model_name: str, 
                                   callback: Callable[[str, Any], None]) -> None:
        """Register a callback for automated retraining."""
        self.retraining_callbacks[model_name] = callback
        logger.info(f"Retraining callback registered for model: {model_name}")

    def set_baseline_metrics(self, model_name: str, metrics) -> None:
        """Set baseline performance metrics for a model."""
        self.performance_tracker.set_baseline_metrics(model_name, metrics)
        logger.info(f"Baseline metrics set for model: {model_name}")

    async def track_prediction(
        self, 
        model_name: str, 
        model_version: str,
        features: np.ndarray, 
        prediction: Any,
        actual: Optional[Any] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track a model prediction for monitoring with improved resource management."""
        
        try:
            prediction_data = {
                'timestamp': datetime.now(),
                'model_version': model_version,
                'prediction': prediction,
                'actual': actual,
                'confidence': confidence,
                'metadata': metadata or {}
            }
            
            # Use resource manager for efficient storage
            await self.resource_manager.add_prediction(
                model_name, prediction_data, features
            )
            
            # Record metrics
            self.metrics_collector.increment_counter(
                "model_predictions_total",
                tags={"model": model_name, "version": model_version}
            )
            
            if confidence is not None:
                self.metrics_collector.set_gauge(
                    "model_prediction_confidence",
                    confidence,
                    tags={"model": model_name}
                )
                
        except Exception as e:
            logger.error(f"Error tracking prediction for {model_name}: {e}")
            raise MonitoringError(f"Failed to track prediction: {e}") from e

    async def calculate_performance_metrics(self, model_name: str,
                                          window_size: Optional[int] = None) -> Optional[ModelPerformanceMetrics]:
        """Calculate current performance metrics for a model"""
        
        if model_name not in self.prediction_history:
            return None
        
        predictions = self.prediction_history[model_name]
        if window_size:
            predictions = predictions[-window_size:]
        
        # Filter predictions with actual values
        labeled_predictions = [p for p in predictions if p['actual'] is not None]
        
        if len(labeled_predictions) < 10:  # Need minimum samples
            return None
        
        # Extract predictions and actuals
        y_pred = [p['prediction'] for p in labeled_predictions]
        y_true = [p['actual'] for p in labeled_predictions]
        
        # Get latest model version
        model_version = labeled_predictions[-1]['model_version']
        
        # Calculate classification metrics
        try:
            # Convert to binary if needed (for trading signals)
            if isinstance(y_pred[0], (int, float)) and isinstance(y_true[0], (int, float)):
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            else:
                # For more complex predictions, use custom logic
                accuracy = sum(1 for p, t in zip(y_pred, y_true) if p == t) / len(y_pred)
                precision = recall = f1 = accuracy  # Simplified
        except Exception as e:
            logger.warning(f"Error calculating classification metrics: {e}")
            accuracy = precision = recall = f1 = 0.0
        
        # Calculate trading-specific metrics if available
        sharpe_ratio = None
        max_drawdown = None
        total_return = None
        win_rate = None
        avg_trade_duration = None
        prediction_confidence = None
        
        # Calculate confidence if available
        confidences = [p['confidence'] for p in labeled_predictions if p['confidence'] is not None]
        if confidences:
            prediction_confidence = np.mean(confidences)
        
        metrics = ModelPerformanceMetrics(
            timestamp=datetime.now(),
            model_name=model_name,
            model_version=model_version,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_return=total_return,
            win_rate=win_rate,
            avg_trade_duration=avg_trade_duration,
            prediction_confidence=prediction_confidence
        )
        
        # Store metrics
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        self.performance_history[model_name].append(metrics)
        
        # Record metrics for monitoring
        self.metrics_collector.set_gauge("model_accuracy", accuracy, tags={"model": model_name})
        self.metrics_collector.set_gauge("model_precision", precision, tags={"model": model_name})
        self.metrics_collector.set_gauge("model_recall", recall, tags={"model": model_name})
        self.metrics_collector.set_gauge("model_f1_score", f1, tags={"model": model_name})
        
        if prediction_confidence is not None:
            self.metrics_collector.set_gauge(
                "model_avg_confidence", 
                prediction_confidence, 
                tags={"model": model_name}
            )
        
        return metrics

    async def detect_data_drift(self, model_name: str) -> Optional[DriftDetectionResult]:
        """Detect data drift using statistical tests"""
        
        if model_name not in self.feature_history:
            return None
        
        features = self.feature_history[model_name]
        if len(features) < self.drift_detection_window:
            return None
        
        # Split into reference (baseline) and current windows
        split_point = len(features) // 2
        reference_features = features[:split_point]
        current_features = features[split_point:]
        
        if len(reference_features) < 20 or len(current_features) < 20:
            return None
        
        # Convert to arrays
        reference_array = np.array(reference_features)
        current_array = np.array(current_features)
        
        # Perform Kolmogorov-Smirnov test for each feature
        drift_scores = []
        
        for i in range(reference_array.shape[1]):
            try:
                ks_stat, p_value = stats.ks_2samp(
                    reference_array[:, i], 
                    current_array[:, i]
                )
                drift_scores.append(p_value)
            except Exception as e:
                logger.warning(f"Error in KS test for feature {i}: {e}")
                drift_scores.append(1.0)  # No drift detected
        
        # Calculate overall drift score (minimum p-value)
        overall_drift_score = min(drift_scores) if drift_scores else 1.0
        threshold = self.drift_thresholds[DriftType.DATA_DRIFT]
        
        detected = overall_drift_score < threshold
        severity = self._calculate_drift_severity(overall_drift_score, threshold)
        
        result = DriftDetectionResult(
            drift_type=DriftType.DATA_DRIFT,
            severity=severity,
            drift_score=overall_drift_score,
            threshold=threshold,
            detected=detected,
            timestamp=datetime.now(),
            details={
                'feature_drift_scores': drift_scores,
                'num_features': len(drift_scores),
                'reference_samples': len(reference_features),
                'current_samples': len(current_features)
            }
        )
        
        if detected:
            await self._handle_drift_detection(model_name, result)
        
        return result

    async def detect_performance_drift(self, model_name: str) -> Optional[DriftDetectionResult]:
        """Detect performance drift by comparing current vs baseline metrics"""
        
        if model_name not in self.baseline_metrics:
            return None
        
        current_metrics = await self.calculate_performance_metrics(model_name, window_size=50)
        if not current_metrics:
            return None
        
        baseline = self.baseline_metrics[model_name]
        
        # Calculate relative performance change
        performance_changes = {}
        for metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
            baseline_value = getattr(baseline, metric_name)
            current_value = getattr(current_metrics, metric_name)
            
            if baseline_value > 0:
                change = (baseline_value - current_value) / baseline_value
                performance_changes[metric_name] = change
        
        # Use worst performance degradation
        max_degradation = max(performance_changes.values()) if performance_changes else 0.0
        threshold = self.drift_thresholds[DriftType.PERFORMANCE_DRIFT]
        
        detected = max_degradation > threshold
        severity = self._calculate_drift_severity(max_degradation, threshold)
        
        result = DriftDetectionResult(
            drift_type=DriftType.PERFORMANCE_DRIFT,
            severity=severity,
            drift_score=max_degradation,
            threshold=threshold,
            detected=detected,
            timestamp=datetime.now(),
            details={
                'performance_changes': performance_changes,
                'baseline_metrics': baseline.__dict__,
                'current_metrics': current_metrics.__dict__
            }
        )
        
        if detected:
            await self._handle_drift_detection(model_name, result)
        
        return result

    async def run_monitoring_cycle(self, model_name: str) -> Dict[str, Any]:
        """Run a complete monitoring cycle for a model with improved error handling."""
        
        results = {
            'model_name': model_name,
            'timestamp': datetime.now(),
            'performance_metrics': None,
            'drift_detection': {
                'data_drift': None,
                'performance_drift': None,
                'data_quality_drift': None
            },
            'alerts_generated': []
        }
        
        try:
            async with self.resource_manager.get_model_data(model_name) as model_data:
                # Calculate performance metrics
                results['performance_metrics'] = await self._calculate_performance_metrics(
                    model_name, model_data['predictions']
                )
                
                # Check performance thresholds
                if results['performance_metrics']:
                    await self._check_performance_thresholds(
                        model_name, results['performance_metrics']
                    )
                
                # Detect drift using strategy pattern
                results['drift_detection'] = await self._detect_all_drift_types(
                    model_name, model_data
                )
            
            logger.info(f"Monitoring cycle completed for model: {model_name}")
            
        except InsufficientDataError as e:
            logger.warning(f"Insufficient data for monitoring {model_name}: {e}")
            results['error'] = str(e)
            
        except Exception as e:
            logger.error(f"Error in monitoring cycle for {model_name}: {e}")
            error_alert = AlertFactory.create_system_error_alert(model_name, e)
            await self.alert_system.notify_observers(error_alert)
            results['error'] = str(e)
        
        return results

    async def _check_performance_thresholds(self, model_name: str, 
                                          metrics: ModelPerformanceMetrics) -> None:
        """Check if performance metrics exceed alert thresholds"""
        
        # Define performance thresholds
        thresholds = {
            'accuracy': 0.6,      # Minimum acceptable accuracy
            'precision': 0.6,     # Minimum acceptable precision
            'recall': 0.6,        # Minimum acceptable recall
            'f1_score': 0.6,      # Minimum acceptable F1 score
        }
        
        for metric_name, threshold in thresholds.items():
            current_value = getattr(metrics, metric_name)
            
            if current_value < threshold:
                severity = AlertSeverity.HIGH if current_value < threshold * 0.8 else AlertSeverity.MEDIUM
                
                alert = Alert(
                    id=f"performance_threshold_{model_name}_{metric_name}_{datetime.now().timestamp()}",
                    severity=severity,
                    title=f"Performance Threshold Breach: {metric_name.title()}",
                    message=f"Model {model_name} {metric_name} ({current_value:.3f}) is below threshold ({threshold})",
                    timestamp=datetime.now(),
                    model_name=model_name,
                    metric_name=metric_name,
                    current_value=current_value,
                    threshold=threshold
                )
                
                await self._send_alert(alert)

    async def _handle_drift_detection(self, model_name: str, 
                                    drift_result: DriftDetectionResult) -> None:
        """Handle detected model drift"""
        
        # Send alert
        alert = Alert(
            id=f"drift_detection_{model_name}_{drift_result.drift_type.value}_{datetime.now().timestamp()}",
            severity=drift_result.severity,
            title=f"Model Drift Detected: {drift_result.drift_type.value.replace('_', ' ').title()}",
            message=f"Model {model_name} shows {drift_result.drift_type.value} "
                   f"(score: {drift_result.drift_score:.4f}, threshold: {drift_result.threshold})",
            timestamp=drift_result.timestamp,
            model_name=model_name,
            metadata=drift_result.details
        )
        
        await self._send_alert(alert)
        
        # Trigger retraining if callback is registered and severity is high enough
        if (model_name in self.retraining_callbacks and 
            drift_result.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]):
            
            try:
                callback = self.retraining_callbacks[model_name]
                callback(model_name, drift_result)
                
                logger.info(f"Retraining triggered for model {model_name} due to {drift_result.drift_type.value}")
                
                # Send retraining notification
                retraining_alert = Alert(
                    id=f"retraining_triggered_{model_name}_{datetime.now().timestamp()}",
                    severity=AlertSeverity.MEDIUM,
                    title="Automated Retraining Triggered",
                    message=f"Retraining initiated for model {model_name} due to detected {drift_result.drift_type.value}",
                    timestamp=datetime.now(),
                    model_name=model_name
                )
                
                await self._send_alert(retraining_alert)
                
            except Exception as e:
                logger.error(f"Error triggering retraining for {model_name}: {e}")

    async def _send_alert(self, alert: Alert) -> None:
        """Send alert through all registered channels"""
        
        # Check cooldown to prevent spam
        cooldown_key = f"{alert.model_name}_{alert.metric_name}_{alert.severity.value}"
        now = datetime.now()
        
        if cooldown_key in self.alert_cooldown:
            if now - self.alert_cooldown[cooldown_key] < self.cooldown_period:
                return  # Skip alert due to cooldown
        
        self.alert_cooldown[cooldown_key] = now
        
        # Store alert
        self.alert_history.append(alert)
        
        # Send through all channels
        for channel in self.alert_channels:
            try:
                channel(alert)
            except Exception as e:
                logger.error(f"Error sending alert through channel: {e}")
        
        # Record alert metric
        self.metrics_collector.increment_counter(
            "alerts_sent_total",
            tags={
                "severity": alert.severity.value,
                "model": alert.model_name or "system"
            }
        )
        
        logger.info(f"Alert sent: {alert.title} (Severity: {alert.severity.value})")

    def _calculate_drift_severity(self, score: float, threshold: float) -> AlertSeverity:
        """Calculate alert severity based on drift score"""
        
        if score > threshold * 2:
            return AlertSeverity.CRITICAL
        elif score > threshold * 1.5:
            return AlertSeverity.HIGH
        elif score > threshold:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW

    async def _calculate_performance_metrics(self, model_name: str, predictions: list):
        """Calculate performance metrics using performance tracker."""
        return await self.performance_tracker.calculate_performance_metrics(
            model_name, predictions
        )
    
    async def _check_performance_thresholds(self, model_name: str, metrics) -> None:
        """Check performance thresholds and send alerts if needed."""
        thresholds = self.config_manager.config.performance_thresholds
        
        for metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
            current_value = getattr(metrics, metric_name)
            threshold = getattr(thresholds, metric_name)
            
            if current_value < threshold:
                alert = AlertFactory.create_performance_alert(
                    model_name, metric_name, current_value, threshold
                )
                await self.alert_system.notify_observers(alert)
    
    async def _detect_all_drift_types(self, model_name: str, model_data: dict) -> Dict[str, Any]:
        """Detect all types of drift using strategy pattern."""
        drift_results = {}
        
        # Data drift detection
        try:
            data_drift_data = {'feature_history': model_data['features']}
            drift_results['data_drift'] = await self.drift_detector.detect_drift(
                DriftType.DATA_DRIFT,
                model_name,
                data_drift_data,
                self.config_manager.get_drift_threshold(DriftType.DATA_DRIFT)
            )
            
            if drift_results['data_drift'] and drift_results['data_drift'].detected:
                await self._handle_drift_detection(model_name, drift_results['data_drift'])
                
        except Exception as e:
            logger.error(f"Data drift detection failed for {model_name}: {e}")
            drift_results['data_drift'] = None
        
        # Performance drift detection
        try:
            performance_drift_data = {
                'baseline_metrics': self.performance_tracker.baseline_metrics.get(model_name),
                'current_metrics': await self._calculate_performance_metrics(
                    model_name, model_data['predictions'][-50:]  # Last 50 predictions
                )
            }
            
            drift_results['performance_drift'] = await self.drift_detector.detect_drift(
                DriftType.PERFORMANCE_DRIFT,
                model_name,
                performance_drift_data,
                self.config_manager.get_drift_threshold(DriftType.PERFORMANCE_DRIFT)
            )
            
            if drift_results['performance_drift'] and drift_results['performance_drift'].detected:
                await self._handle_drift_detection(model_name, drift_results['performance_drift'])
                
        except Exception as e:
            logger.error(f"Performance drift detection failed for {model_name}: {e}")
            drift_results['performance_drift'] = None
        
        # Data quality drift detection
        try:
            data_quality_drift_data = {'feature_history': model_data['features']}
            drift_results['data_quality_drift'] = await self.drift_detector.detect_drift(
                DriftType.DATA_QUALITY_DRIFT,
                model_name,
                data_quality_drift_data,
                self.config_manager.get_drift_threshold(DriftType.DATA_QUALITY_DRIFT)
            )
            
            if drift_results['data_quality_drift'] and drift_results['data_quality_drift'].detected:
                await self._handle_drift_detection(model_name, drift_results['data_quality_drift'])
                
        except Exception as e:
            logger.error(f"Data quality drift detection failed for {model_name}: {e}")
            drift_results['data_quality_drift'] = None
        
        return drift_results
    
    async def _handle_drift_detection(self, model_name: str, drift_result) -> None:
        """Handle detected drift with improved logic."""
        # Send drift alert
        drift_alert = AlertFactory.create_drift_alert(model_name, drift_result)
        await self.alert_system.notify_observers(drift_alert)
        
        # Trigger retraining if conditions are met
        if (model_name in self.retraining_callbacks and 
            self.config_manager.should_trigger_retraining(drift_result.severity)):
            
            try:
                callback = self.retraining_callbacks[model_name]
                callback(model_name, drift_result)
                
                logger.info(f"Retraining triggered for model {model_name} due to {drift_result.drift_type.value}")
                
            except Exception as e:
                logger.error(f"Error triggering retraining for {model_name}: {e}")

    async def get_model_status(self, model_name: str) -> Dict[str, Any]:
        """Get comprehensive status for a model with improved health scoring."""
        
        status = {
            'model_name': model_name,
            'timestamp': datetime.now(),
            'performance_metrics': None,
            'recent_alerts': [],
            'drift_status': 'unknown',
            'health_score': 0.0,
            'resource_usage': {}
        }
        
        try:
            # Get latest performance metrics
            if model_name in self.performance_tracker.performance_history:
                history = self.performance_tracker.performance_history[model_name]
                if history:
                    status['performance_metrics'] = history[-1].__dict__
            
            # Get recent alerts from alert system
            recent_alerts = [
                alert for alert in self.alert_system.alert_history[-50:]
                if alert.model_name == model_name and 
                   alert.timestamp > datetime.now() - timedelta(hours=24)
            ]
            status['recent_alerts'] = [alert.__dict__ for alert in recent_alerts]
            
            # Get resource usage
            status['resource_usage'] = await self.resource_manager.get_memory_usage()
            
            # Calculate health score using improved algorithm
            status['health_score'] = self._calculate_health_score(
                status['performance_metrics'], recent_alerts
            )
            
            status['drift_status'] = self._determine_drift_status(status['health_score'])
            
        except Exception as e:
            logger.error(f"Error getting model status for {model_name}: {e}")
            status['error'] = str(e)
        
        return status
    
    def _calculate_health_score(self, performance_metrics: Optional[dict], recent_alerts: list) -> float:
        """Calculate model health score with improved algorithm."""
        health_score = 100.0
        
        if performance_metrics:
            # Weight different metrics
            accuracy_weight = 0.4
            precision_weight = 0.3
            recall_weight = 0.2
            confidence_weight = 0.1
            
            accuracy = performance_metrics.get('accuracy', 0)
            precision = performance_metrics.get('precision', 0)
            recall = performance_metrics.get('recall', 0)
            confidence = performance_metrics.get('prediction_confidence', 0.5)
            
            weighted_performance = (
                accuracy * accuracy_weight +
                precision * precision_weight +
                recall * recall_weight +
                confidence * confidence_weight
            )
            
            health_score *= weighted_performance
        
        # Penalize based on alert severity
        for alert in recent_alerts:
            severity = alert.get('severity', 'low')
            penalty = {'low': 2, 'medium': 5, 'high': 10, 'critical': 20}.get(severity, 5)
            health_score = max(0, health_score - penalty)
        
        return min(100.0, max(0.0, health_score))
    
    def _determine_drift_status(self, health_score: float) -> str:
        """Determine drift status based on health score."""
        if health_score > 80:
            return 'healthy'
        elif health_score > 60:
            return 'warning'
        elif health_score > 40:
            return 'degraded'
        else:
            return 'critical'

    async def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old monitoring data using resource manager."""
        return await self.resource_manager.cleanup_old_data(days_to_keep)


# Alert channel implementations moved to alert_system.py
# Import them from there if needed:
# from .monitoring.alert_system import EmailAlertObserver, SlackAlertObserver
