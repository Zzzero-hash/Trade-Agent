"""
Custom exceptions for monitoring service.
"""


class MonitoringError(Exception):
    """Base exception for monitoring service."""
    pass


class InsufficientDataError(MonitoringError):
    """Raised when there's insufficient data for monitoring operations."""
    
    def __init__(self, operation: str, required_samples: int, actual_samples: int):
        self.operation = operation
        self.required_samples = required_samples
        self.actual_samples = actual_samples
        super().__init__(
            f"Insufficient data for {operation}: "
            f"required {required_samples}, got {actual_samples}"
        )


class DriftDetectionError(MonitoringError):
    """Raised when drift detection fails."""
    
    def __init__(self, model_name: str, drift_type: str, reason: str):
        self.model_name = model_name
        self.drift_type = drift_type
        self.reason = reason
        super().__init__(
            f"Drift detection failed for model {model_name} "
            f"({drift_type}): {reason}"
        )


class MetricsCalculationError(MonitoringError):
    """Raised when metrics calculation fails."""
    
    def __init__(self, model_name: str, metric_type: str, reason: str):
        self.model_name = model_name
        self.metric_type = metric_type
        self.reason = reason
        super().__init__(
            f"Metrics calculation failed for model {model_name} "
            f"({metric_type}): {reason}"
        )


class AlertDeliveryError(MonitoringError):
    """Raised when alert delivery fails."""
    
    def __init__(self, channel: str, alert_id: str, reason: str):
        self.channel = channel
        self.alert_id = alert_id
        self.reason = reason
        super().__init__(
            f"Alert delivery failed via {channel} "
            f"for alert {alert_id}: {reason}"
        )