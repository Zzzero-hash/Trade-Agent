"""
Configuration management for monitoring service.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import timedelta

from src.models.monitoring import DriftType, AlertSeverity


@dataclass
class DriftDetectionConfig:
    """Configuration for drift detection."""
    window_size: int = 100
    min_samples: int = 20
    thresholds: Dict[DriftType, float] = field(default_factory=lambda: {
        DriftType.DATA_DRIFT: 0.05,
        DriftType.CONCEPT_DRIFT: 0.1,
        DriftType.PERFORMANCE_DRIFT: 0.15
    })


@dataclass
class PerformanceThresholds:
    """Performance metric thresholds."""
    accuracy: float = 0.6
    precision: float = 0.6
    recall: float = 0.6
    f1_score: float = 0.6
    confidence: float = 0.5


@dataclass
class AlertConfig:
    """Alert system configuration."""
    cooldown_period: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    max_history_size: int = 1000
    severity_thresholds: Dict[str, Dict[AlertSeverity, float]] = field(
        default_factory=lambda: {
            'performance': {
                AlertSeverity.CRITICAL: 0.4,
                AlertSeverity.HIGH: 0.5,
                AlertSeverity.MEDIUM: 0.6,
                AlertSeverity.LOW: 0.7
            }
        }
    )


@dataclass
class MonitoringConfig:
    """Main monitoring configuration."""
    drift_detection: DriftDetectionConfig = field(default_factory=DriftDetectionConfig)
    performance_thresholds: PerformanceThresholds = field(default_factory=PerformanceThresholds)
    alert_config: AlertConfig = field(default_factory=AlertConfig)
    max_prediction_history: int = 1000
    cleanup_interval_days: int = 30
    enable_auto_retraining: bool = True
    retraining_severity_threshold: AlertSeverity = AlertSeverity.HIGH


class ConfigManager:
    """Manages monitoring configuration with validation."""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        if self.config.drift_detection.window_size < self.config.drift_detection.min_samples * 2:
            raise ValueError(
                "Drift detection window size must be at least twice the minimum samples"
            )
        
        if self.config.max_prediction_history < 100:
            raise ValueError("Max prediction history must be at least 100")
        
        # Validate performance thresholds are between 0 and 1
        thresholds = self.config.performance_thresholds
        for attr_name in ['accuracy', 'precision', 'recall', 'f1_score', 'confidence']:
            value = getattr(thresholds, attr_name)
            if not 0 <= value <= 1:
                raise ValueError(f"Performance threshold {attr_name} must be between 0 and 1")
    
    def update_drift_threshold(self, drift_type: DriftType, threshold: float) -> None:
        """Update drift detection threshold."""
        if not 0 <= threshold <= 1:
            raise ValueError("Drift threshold must be between 0 and 1")
        
        self.config.drift_detection.thresholds[drift_type] = threshold
    
    def update_performance_threshold(self, metric: str, threshold: float) -> None:
        """Update performance threshold."""
        if not 0 <= threshold <= 1:
            raise ValueError("Performance threshold must be between 0 and 1")
        
        if not hasattr(self.config.performance_thresholds, metric):
            raise ValueError(f"Unknown performance metric: {metric}")
        
        setattr(self.config.performance_thresholds, metric, threshold)
    
    def get_drift_threshold(self, drift_type: DriftType) -> float:
        """Get drift detection threshold."""
        return self.config.drift_detection.thresholds.get(drift_type, 0.05)
    
    def get_performance_threshold(self, metric: str) -> float:
        """Get performance threshold."""
        return getattr(self.config.performance_thresholds, metric, 0.6)
    
    def should_trigger_retraining(self, severity: AlertSeverity) -> bool:
        """Check if retraining should be triggered based on severity."""
        if not self.config.enable_auto_retraining:
            return False
        
        severity_levels = {
            AlertSeverity.LOW: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.HIGH: 3,
            AlertSeverity.CRITICAL: 4
        }
        
        return (severity_levels.get(severity, 0) >= 
                severity_levels.get(self.config.retraining_severity_threshold, 3))