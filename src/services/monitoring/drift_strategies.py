"""
Drift detection strategies using Strategy pattern.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
import numpy as np
from scipy import stats
from datetime import datetime

from src.models.monitoring import DriftDetectionResult, DriftType, AlertSeverity
from src.utils.logging import get_logger

logger = get_logger("drift_strategies")


class DriftDetectionStrategy(ABC):
    """Abstract base class for drift detection strategies."""
    
    @abstractmethod
    async def detect_drift(
        self, 
        model_name: str, 
        data: dict,
        threshold: float
    ) -> Optional[DriftDetectionResult]:
        """Detect drift using specific strategy."""
        pass


class KolmogorovSmirnovDriftStrategy(DriftDetectionStrategy):
    """Kolmogorov-Smirnov test for data drift detection."""
    
    def __init__(self, min_samples: int = 20):
        self.min_samples = min_samples
    
    async def detect_drift(
        self, 
        model_name: str, 
        data: dict,
        threshold: float
    ) -> Optional[DriftDetectionResult]:
        """Detect data drift using KS test."""
        
        features = data.get('feature_history', [])
        if len(features) < self.min_samples * 2:
            return None
        
        # Split into reference and current windows
        split_point = len(features) // 2
        reference_features = np.array(features[:split_point])
        current_features = np.array(features[split_point:])
        
        if len(reference_features) < self.min_samples or len(current_features) < self.min_samples:
            return None
        
        # Perform KS test for each feature
        drift_scores = []
        for i in range(reference_features.shape[1]):
            try:
                _, p_value = stats.ks_2samp(
                    reference_features[:, i], 
                    current_features[:, i]
                )
                drift_scores.append(p_value)
            except Exception as e:
                logger.warning(f"Error in KS test for feature {i}: {e}")
                drift_scores.append(1.0)
        
        # Calculate overall drift score
        overall_drift_score = min(drift_scores) if drift_scores else 1.0
        detected = overall_drift_score < threshold
        severity = self._calculate_severity(overall_drift_score, threshold)
        
        return DriftDetectionResult(
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
    
    def _calculate_severity(self, score: float, threshold: float) -> AlertSeverity:
        """Calculate alert severity based on drift score."""
        if score < threshold * 0.2:
            return AlertSeverity.CRITICAL
        elif score < threshold * 0.5:
            return AlertSeverity.HIGH
        elif score < threshold:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW


class PerformanceDriftStrategy(DriftDetectionStrategy):
    """Performance-based drift detection strategy."""
    
    async def detect_drift(
        self, 
        model_name: str, 
        data: dict,
        threshold: float
    ) -> Optional[DriftDetectionResult]:
        """Detect performance drift."""
        
        baseline_metrics = data.get('baseline_metrics')
        current_metrics = data.get('current_metrics')
        
        if not baseline_metrics or not current_metrics:
            return None
        
        # Calculate performance degradation
        performance_changes = {}
        for metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
            baseline_value = getattr(baseline_metrics, metric_name, 0)
            current_value = getattr(current_metrics, metric_name, 0)
            
            if baseline_value > 0:
                change = (baseline_value - current_value) / baseline_value
                performance_changes[metric_name] = change
        
        max_degradation = max(performance_changes.values()) if performance_changes else 0.0
        detected = max_degradation > threshold
        severity = self._calculate_severity(max_degradation, threshold)
        
        return DriftDetectionResult(
            drift_type=DriftType.PERFORMANCE_DRIFT,
            severity=severity,
            drift_score=max_degradation,
            threshold=threshold,
            detected=detected,
            timestamp=datetime.now(),
            details={
                'performance_changes': performance_changes,
                'baseline_metrics': baseline_metrics.__dict__,
                'current_metrics': current_metrics.__dict__
            }
        )
    
    def _calculate_severity(self, score: float, threshold: float) -> AlertSeverity:
        """Calculate alert severity based on performance degradation."""
        if score > threshold * 2:
            return AlertSeverity.CRITICAL
        elif score > threshold * 1.5:
            return AlertSeverity.HIGH
        elif score > threshold:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW


class DriftDetectionContext:
    """Context class for drift detection strategies."""
    
    def __init__(self):
        self.strategies = {
            DriftType.DATA_DRIFT: KolmogorovSmirnovDriftStrategy(),
            DriftType.PERFORMANCE_DRIFT: PerformanceDriftStrategy()
        }
    
    def set_strategy(self, drift_type: DriftType, strategy: DriftDetectionStrategy):
        """Set a specific strategy for a drift type."""
        self.strategies[drift_type] = strategy
    
    async def detect_drift(
        self, 
        drift_type: DriftType,
        model_name: str, 
        data: dict,
        threshold: float
    ) -> Optional[DriftDetectionResult]:
        """Detect drift using the appropriate strategy."""
        
        strategy = self.strategies.get(drift_type)
        if not strategy:
            logger.warning(f"No strategy found for drift type: {drift_type}")
            return None
        
        return await strategy.detect_drift(model_name, data, threshold)