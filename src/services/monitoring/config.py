"""
Configuration management for advanced monitoring system.

This module provides centralized configuration management for all monitoring
components with environment-specific settings and runtime adjustments.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from datetime import timedelta

from src.models.monitoring import AlertSeverity, DriftType
from src.utils.logging import get_logger

logger = get_logger("monitoring_config")


class MonitoringEnvironment(Enum):
    """Monitoring environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class PerformanceThresholds:
    """Performance thresholds for alerting"""
    accuracy: float = 0.8
    precision: float = 0.75
    recall: float = 0.75
    f1_score: float = 0.75
    sharpe_ratio: float = 1.0
    max_drawdown: float = 0.15
    latency_ms: float = 100.0
    throughput_min: float = 10.0
    error_rate_max: float = 0.01
    cache_hit_rate_min: float = 0.7


@dataclass
class DriftThresholds:
    """Drift detection thresholds"""
    data_drift: float = 0.05
    performance_drift: float = 0.1
    concept_drift: float = 0.15
    data_quality_drift: float = 5.0  # Number of anomalies
    statistical_significance: float = 0.05
    effect_size_small: float = 0.2
    effect_size_medium: float = 0.5
    effect_size_large: float = 0.8


@dataclass
class AnomalyDetectionSettings:
    """Anomaly detection configuration"""
    enabled: bool = True
    window_size: int = 100
    min_samples: int = 20
    z_score_threshold: float = 3.0
    sensitivity: float = 1.0
    isolation_forest_contamination: float = 0.1
    enable_statistical_tests: bool = True
    enable_ml_detection: bool = True
    bootstrap_iterations: int = 1000


@dataclass
class AlertingSettings:
    """Alerting system configuration"""
    enabled: bool = True
    cooldown_minutes: int = 30
    max_alerts_per_hour: int = 10
    escalation_enabled: bool = True
    auto_resolve_minutes: int = 60
    severity_levels: List[str] = field(default_factory=lambda: ["low", "medium", "high", "critical"])
    
    # Channel configurations
    email_enabled: bool = True
    slack_enabled: bool = True
    pagerduty_enabled: bool = False
    webhook_enabled: bool = False
    
    # Channel settings
    email_recipients: List[str] = field(default_factory=list)
    slack_webhook_url: Optional[str] = None
    pagerduty_integration_key: Optional[str] = None
    custom_webhook_url: Optional[str] = None


@dataclass
class SystemHealthSettings:
    """System health monitoring configuration"""
    enabled: bool = True
    check_interval_seconds: int = 60
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    disk_threshold: float = 90.0
    network_latency_threshold_ms: float = 100.0
    predictive_maintenance_enabled: bool = True
    health_score_threshold: float = 0.7


@dataclass
class DataQualitySettings:
    """Data quality monitoring configuration"""
    enabled: bool = True
    check_interval_seconds: int = 300  # 5 minutes
    quality_score_threshold: float = 0.8
    max_missing_data_percent: float = 5.0
    max_duplicate_data_percent: float = 1.0
    price_anomaly_z_score: float = 5.0
    volume_anomaly_multiplier: float = 10.0
    timestamp_gap_tolerance_minutes: int = 5
    stale_data_threshold_hours: int = 1


@dataclass
class RetrainingSettings:
    """Automated retraining configuration"""
    enabled: bool = True
    cooldown_hours: int = 6
    max_concurrent_jobs: int = 2
    min_samples_required: int = 1000
    performance_improvement_threshold: float = 0.05
    max_training_time_hours: int = 24
    auto_deploy_enabled: bool = False
    backup_models: bool = True
    
    # Trigger conditions
    trigger_on_data_drift: bool = True
    trigger_on_performance_drift: bool = True
    trigger_on_consecutive_failures: int = 3
    trigger_on_degradation_percent: float = 0.15


@dataclass
class MonitoringConfig:
    """Main monitoring configuration"""
    environment: MonitoringEnvironment = MonitoringEnvironment.DEVELOPMENT
    enabled: bool = True
    debug_mode: bool = False
    log_level: str = "INFO"
    
    # Component configurations
    performance_thresholds: PerformanceThresholds = field(default_factory=PerformanceThresholds)
    drift_thresholds: DriftThresholds = field(default_factory=DriftThresholds)
    anomaly_detection: AnomalyDetectionSettings = field(default_factory=AnomalyDetectionSettings)
    alerting: AlertingSettings = field(default_factory=AlertingSettings)
    system_health: SystemHealthSettings = field(default_factory=SystemHealthSettings)
    data_quality: DataQualitySettings = field(default_factory=DataQualitySettings)
    retraining: RetrainingSettings = field(default_factory=RetrainingSettings)
    
    # Storage settings
    max_prediction_history: int = 10000
    max_alert_history: int = 1000
    max_drift_history: int = 100
    data_retention_days: int = 30
    
    # Advanced settings
    enable_predictive_maintenance: bool = True
    enable_statistical_significance: bool = True
    enable_early_warning: bool = True
    monitoring_mode: str = "normal"  # normal, high_sensitivity, maintenance, emergency


class ConfigManager:
    """Configuration manager for monitoring system"""
    
    def __init__(self, config: Optional[MonitoringConfig] = None, config_file: Optional[str] = None):
        self.config_file = config_file
        self.config = config or self._load_config()
        self._validate_config()
    
    def _load_config(self) -> MonitoringConfig:
        """Load configuration from file or environment"""
        
        # Try to load from file
        if self.config_file and os.path.exists(self.config_file):
            return self._load_from_file(self.config_file)
        
        # Try default config files
        default_files = [
            "config/monitoring.yaml",
            "config/monitoring.yml",
            "config/monitoring.json",
            ".kiro/settings/monitoring.yaml"
        ]
        
        for file_path in default_files:
            if os.path.exists(file_path):
                return self._load_from_file(file_path)
        
        # Load from environment variables
        config = self._load_from_environment()
        
        # Use defaults if nothing found
        if not config:
            logger.info("Using default monitoring configuration")
            config = MonitoringConfig()
        
        return config
    
    def _load_from_file(self, file_path: str) -> MonitoringConfig:
        """Load configuration from YAML or JSON file"""
        
        try:
            with open(file_path, 'r') as f:
                if file_path.endswith(('.yaml', '.yml')):
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            config = self._dict_to_config(data)
            logger.info(f"Monitoring configuration loaded from: {file_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config from {file_path}: {e}")
            return MonitoringConfig()
    
    def _load_from_environment(self) -> Optional[MonitoringConfig]:
        """Load configuration from environment variables"""
        
        env_config = {}
        
        # Map environment variables to config structure
        env_mappings = {
            'MONITORING_ENABLED': ('enabled', bool),
            'MONITORING_ENVIRONMENT': ('environment', str),
            'MONITORING_DEBUG': ('debug_mode', bool),
            'MONITORING_LOG_LEVEL': ('log_level', str),
            
            # Performance thresholds
            'MONITORING_ACCURACY_THRESHOLD': ('performance_thresholds.accuracy', float),
            'MONITORING_LATENCY_THRESHOLD': ('performance_thresholds.latency_ms', float),
            'MONITORING_ERROR_RATE_THRESHOLD': ('performance_thresholds.error_rate_max', float),
            
            # Drift thresholds
            'MONITORING_DATA_DRIFT_THRESHOLD': ('drift_thresholds.data_drift', float),
            'MONITORING_PERFORMANCE_DRIFT_THRESHOLD': ('drift_thresholds.performance_drift', float),
            
            # Alerting
            'MONITORING_ALERTS_ENABLED': ('alerting.enabled', bool),
            'MONITORING_EMAIL_ENABLED': ('alerting.email_enabled', bool),
            'MONITORING_SLACK_ENABLED': ('alerting.slack_enabled', bool),
            'MONITORING_SLACK_WEBHOOK': ('alerting.slack_webhook_url', str),
            
            # Retraining
            'MONITORING_RETRAINING_ENABLED': ('retraining.enabled', bool),
            'MONITORING_AUTO_DEPLOY': ('retraining.auto_deploy_enabled', bool),
        }
        
        for env_var, (config_path, value_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    # Convert value to appropriate type
                    if value_type == bool:
                        converted_value = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif value_type == float:
                        converted_value = float(env_value)
                    elif value_type == int:
                        converted_value = int(env_value)
                    else:
                        converted_value = env_value
                    
                    # Set nested config value
                    self._set_nested_value(env_config, config_path, converted_value)
                    
                except ValueError as e:
                    logger.warning(f"Invalid value for {env_var}: {env_value} ({e})")
        
        if env_config:
            config = self._dict_to_config(env_config)
            logger.info("Monitoring configuration loaded from environment variables")
            return config
        
        return None
    
    def _dict_to_config(self, data: Dict[str, Any]) -> MonitoringConfig:
        """Convert dictionary to MonitoringConfig object"""
        
        try:
            # Handle environment enum
            if 'environment' in data and isinstance(data['environment'], str):
                data['environment'] = MonitoringEnvironment(data['environment'])
            
            # Create nested dataclass objects
            if 'performance_thresholds' in data:
                data['performance_thresholds'] = PerformanceThresholds(**data['performance_thresholds'])
            
            if 'drift_thresholds' in data:
                data['drift_thresholds'] = DriftThresholds(**data['drift_thresholds'])
            
            if 'anomaly_detection' in data:
                data['anomaly_detection'] = AnomalyDetectionSettings(**data['anomaly_detection'])
            
            if 'alerting' in data:
                data['alerting'] = AlertingSettings(**data['alerting'])
            
            if 'system_health' in data:
                data['system_health'] = SystemHealthSettings(**data['system_health'])
            
            if 'data_quality' in data:
                data['data_quality'] = DataQualitySettings(**data['data_quality'])
            
            if 'retraining' in data:
                data['retraining'] = RetrainingSettings(**data['retraining'])
            
            return MonitoringConfig(**data)
            
        except Exception as e:
            logger.error(f"Failed to convert dict to config: {e}")
            return MonitoringConfig()
    
    def _set_nested_value(self, config_dict: Dict[str, Any], path: str, value: Any) -> None:
        """Set nested dictionary value using dot notation"""
        
        keys = path.split('.')
        current = config_dict
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _validate_config(self) -> None:
        """Validate configuration values"""
        
        # Validate thresholds
        perf = self.config.performance_thresholds
        if not (0.0 <= perf.accuracy <= 1.0):
            logger.warning(f"Invalid accuracy threshold: {perf.accuracy}")
        
        if perf.latency_ms <= 0:
            logger.warning(f"Invalid latency threshold: {perf.latency_ms}")
        
        # Validate drift thresholds
        drift = self.config.drift_thresholds
        if not (0.0 < drift.statistical_significance < 1.0):
            logger.warning(f"Invalid statistical significance: {drift.statistical_significance}")
        
        # Validate alerting settings
        alerting = self.config.alerting
        if alerting.cooldown_minutes <= 0:
            logger.warning(f"Invalid cooldown period: {alerting.cooldown_minutes}")
        
        logger.info("Configuration validation completed")
    
    def save_config(self, file_path: Optional[str] = None) -> None:
        """Save current configuration to file"""
        
        if file_path is None:
            file_path = self.config_file or "config/monitoring.yaml"
        
        try:
            # Convert config to dictionary
            config_dict = asdict(self.config)
            
            # Handle enum serialization
            config_dict['environment'] = self.config.environment.value
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save to file
            with open(file_path, 'w') as f:
                if file_path.endswith(('.yaml', '.yml')):
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config to {file_path}: {e}")
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        
        try:
            # Apply updates
            for key, value in updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    logger.warning(f"Unknown config key: {key}")
            
            # Re-validate
            self._validate_config()
            
            logger.info(f"Configuration updated with {len(updates)} changes")
            
        except Exception as e:
            logger.error(f"Failed to update config: {e}")
    
    def get_drift_threshold(self, drift_type: DriftType) -> float:
        """Get drift threshold for specific drift type"""
        
        thresholds = {
            DriftType.DATA_DRIFT: self.config.drift_thresholds.data_drift,
            DriftType.PERFORMANCE_DRIFT: self.config.drift_thresholds.performance_drift,
            DriftType.CONCEPT_DRIFT: self.config.drift_thresholds.concept_drift,
            DriftType.DATA_QUALITY_DRIFT: self.config.drift_thresholds.data_quality_drift
        }
        
        return thresholds.get(drift_type, 0.05)
    
    def get_performance_threshold(self, metric_name: str) -> float:
        """Get performance threshold for specific metric"""
        
        return getattr(self.config.performance_thresholds, metric_name, 0.0)
    
    def should_trigger_retraining(self, severity: AlertSeverity) -> bool:
        """Check if retraining should be triggered based on alert severity"""
        
        if not self.config.retraining.enabled:
            return False
        
        # Trigger on high or critical severity
        return severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
    
    def get_alert_cooldown(self) -> timedelta:
        """Get alert cooldown period"""
        return timedelta(minutes=self.config.alerting.cooldown_minutes)
    
    def is_environment(self, environment: MonitoringEnvironment) -> bool:
        """Check if current environment matches specified environment"""
        return self.config.environment == environment
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for monitoring dashboard"""
        
        return {
            'environment': self.config.environment.value,
            'enabled': self.config.enabled,
            'monitoring_mode': self.config.monitoring_mode,
            'performance_thresholds': {
                'accuracy': self.config.performance_thresholds.accuracy,
                'latency_ms': self.config.performance_thresholds.latency_ms,
                'error_rate_max': self.config.performance_thresholds.error_rate_max
            },
            'drift_detection': {
                'data_drift_threshold': self.config.drift_thresholds.data_drift,
                'performance_drift_threshold': self.config.drift_thresholds.performance_drift,
                'statistical_significance': self.config.drift_thresholds.statistical_significance
            },
            'alerting': {
                'enabled': self.config.alerting.enabled,
                'email_enabled': self.config.alerting.email_enabled,
                'slack_enabled': self.config.alerting.slack_enabled,
                'cooldown_minutes': self.config.alerting.cooldown_minutes
            },
            'retraining': {
                'enabled': self.config.retraining.enabled,
                'auto_deploy': self.config.retraining.auto_deploy_enabled,
                'cooldown_hours': self.config.retraining.cooldown_hours
            }
        }


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_file: Optional[str] = None) -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_file=config_file)
    
    return _config_manager


def setup_monitoring_config(config: Optional[MonitoringConfig] = None, config_file: Optional[str] = None) -> ConfigManager:
    """Setup monitoring configuration"""
    global _config_manager
    
    _config_manager = ConfigManager(config=config, config_file=config_file)
    logger.info("Monitoring configuration initialized")
    
    return _config_manager