"""Core monitoring data models used across services."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class AlertSeverity(str, Enum):
    """Severity levels supported by the monitoring system."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftType(str, Enum):
    """Types of drift that can be detected for a model."""

    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PERFORMANCE_DRIFT = "performance_drift"
    DATA_QUALITY_DRIFT = "data_quality_drift"


@dataclass(slots=True)
class ModelPerformanceMetrics:
    """Snapshot of model performance for monitoring dashboards."""

    timestamp: datetime
    model_name: str
    model_version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    total_return: Optional[float] = None
    win_rate: Optional[float] = None
    avg_trade_duration: Optional[float] = None
    prediction_confidence: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DriftDetectionResult:
    """Result of a drift detection evaluation."""

    drift_type: DriftType
    severity: AlertSeverity
    drift_score: float
    threshold: float
    detected: bool
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Alert:
    """Structured alert emitted by monitoring components."""

    id: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    model_name: Optional[str] = None
    metric_name: Optional[str] = None
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "Alert",
    "AlertSeverity",
    "DriftDetectionResult",
    "DriftType",
    "ModelPerformanceMetrics",
]
