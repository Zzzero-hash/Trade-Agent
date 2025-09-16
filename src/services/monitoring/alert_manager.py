"""
Alert manager for handling performance thresholds and system errors.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict

from src.models.monitoring import Alert, AlertSeverity, ModelPerformanceMetrics
from src.services.monitoring.alert_system import AlertSubject, AlertFactory
from src.utils.logging import get_logger

logger = get_logger("alert_manager")


class AlertManager:
    """Manages alerts for performance thresholds and system errors."""
    
    def __init__(self, alert_subject: AlertSubject):
        self.alert_subject = alert_subject
        self.performance_thresholds = {
            'accuracy': 0.8,
            'precision': 0.75,
            'recall': 0.75,
            'f1_score': 0.75,
            'sharpe_ratio': 1.0,
            'max_drawdown': 0.15
        }
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.cooldown_period = timedelta(minutes=15)
    
    async def check_performance_thresholds(
        self, 
        model_name: str, 
        metrics: ModelPerformanceMetrics
    ) -> None:
        """Check performance metrics against thresholds and generate alerts."""
        
        threshold_breaches = []
        
        # Check standard ML metrics
        for metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
            current_value = getattr(metrics, metric_name, 0)
            threshold = self.performance_thresholds.get(metric_name, 0)
            
            if current_value < threshold:
                threshold_breaches.append((metric_name, current_value, threshold))
        
        # Check trading-specific metrics
        if metrics.sharpe_ratio is not None:
            if metrics.sharpe_ratio < self.performance_thresholds['sharpe_ratio']:
                threshold_breaches.append(('sharpe_ratio', metrics.sharpe_ratio, self.performance_thresholds['sharpe_ratio']))
        
        if metrics.max_drawdown is not None:
            if metrics.max_drawdown > self.performance_thresholds['max_drawdown']:
                threshold_breaches.append(('max_drawdown', metrics.max_drawdown, self.performance_thresholds['max_drawdown']))
        
        # Generate alerts for breaches
        for metric_name, current_value, threshold in threshold_breaches:
            if not self._is_in_cooldown(model_name, metric_name):
                alert = AlertFactory.create_performance_alert(
                    model_name, metric_name, current_value, threshold
                )
                await self.alert_subject.notify_observers(alert)
                self._update_cooldown(model_name, metric_name)
    
    async def send_system_error_alert(self, model_name: str, error: Exception) -> None:
        """Send system error alert."""
        if not self._is_in_cooldown(model_name, 'system_error'):
            alert = AlertFactory.create_system_error_alert(model_name, error)
            await self.alert_subject.notify_observers(alert)
            self._update_cooldown(model_name, 'system_error')
    
    def _is_in_cooldown(self, model_name: str, metric_name: str) -> bool:
        """Check if alert is in cooldown period."""
        cooldown_key = f"{model_name}_{metric_name}"
        
        if cooldown_key in self.alert_cooldowns:
            return datetime.now() - self.alert_cooldowns[cooldown_key] < self.cooldown_period
        
        return False
    
    def _update_cooldown(self, model_name: str, metric_name: str) -> None:
        """Update cooldown timestamp."""
        cooldown_key = f"{model_name}_{metric_name}"
        self.alert_cooldowns[cooldown_key] = datetime.now()
    
    def set_performance_threshold(self, metric_name: str, threshold: float) -> None:
        """Set performance threshold for a metric."""
        self.performance_thresholds[metric_name] = threshold
        logger.info(f"Performance threshold set for {metric_name}: {threshold}")
    
    def get_performance_thresholds(self) -> Dict[str, float]:
        """Get current performance thresholds."""
        return self.performance_thresholds.copy()