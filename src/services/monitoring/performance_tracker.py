"""
Performance tracking component for model monitoring.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils.logging import get_logger
from src.models.monitoring import ModelPerformanceMetrics

logger = get_logger("performance_tracker")


class PerformanceTracker:
    """Handles model performance calculation and tracking."""
    
    def __init__(self):
        self.performance_history: Dict[str, List[ModelPerformanceMetrics]] = {}
        self.baseline_metrics: Dict[str, ModelPerformanceMetrics] = {}
    
    def set_baseline_metrics(self, model_name: str, metrics: ModelPerformanceMetrics) -> None:
        """Set baseline performance metrics for a model."""
        self.baseline_metrics[model_name] = metrics
        logger.info(f"Baseline metrics set for model: {model_name}")
    
    async def calculate_performance_metrics(
        self, 
        model_name: str,
        predictions_data: List[Dict[str, Any]],
        window_size: Optional[int] = None
    ) -> Optional[ModelPerformanceMetrics]:
        """Calculate current performance metrics for a model."""
        
        if window_size:
            predictions_data = predictions_data[-window_size:]
        
        # Filter predictions with actual values
        labeled_predictions = [p for p in predictions_data if p['actual'] is not None]
        
        if len(labeled_predictions) < 10:  # Need minimum samples
            return None
        
        # Extract predictions and actuals
        y_pred = [p['prediction'] for p in labeled_predictions]
        y_true = [p['actual'] for p in labeled_predictions]
        model_version = labeled_predictions[-1]['model_version']
        
        # Calculate metrics using strategy pattern
        metrics_calculator = self._get_metrics_calculator(y_pred[0], y_true[0])
        accuracy, precision, recall, f1 = metrics_calculator.calculate(y_pred, y_true)
        
        # Calculate confidence if available
        confidences = [p['confidence'] for p in labeled_predictions if p['confidence'] is not None]
        prediction_confidence = np.mean(confidences) if confidences else None
        
        metrics = ModelPerformanceMetrics(
            timestamp=datetime.now(),
            model_name=model_name,
            model_version=model_version,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            prediction_confidence=prediction_confidence
        )
        
        # Store metrics
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        self.performance_history[model_name].append(metrics)
        
        return metrics
    
    def _get_metrics_calculator(self, pred_sample: Any, true_sample: Any) -> 'MetricsCalculator':
        """Factory method to get appropriate metrics calculator."""
        if isinstance(pred_sample, (int, float)) and isinstance(true_sample, (int, float)):
            return ClassificationMetricsCalculator()
        else:
            return CustomMetricsCalculator()


class MetricsCalculator:
    """Abstract base for metrics calculation strategies."""
    
    def calculate(self, y_pred: List[Any], y_true: List[Any]) -> tuple[float, float, float, float]:
        raise NotImplementedError


class ClassificationMetricsCalculator(MetricsCalculator):
    """Calculate standard classification metrics."""
    
    def calculate(self, y_pred: List[Any], y_true: List[Any]) -> tuple[float, float, float, float]:
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            return accuracy, precision, recall, f1
        except Exception as e:
            logger.warning(f"Error calculating classification metrics: {e}")
            return 0.0, 0.0, 0.0, 0.0


class CustomMetricsCalculator(MetricsCalculator):
    """Calculate custom metrics for complex predictions."""
    
    def calculate(self, y_pred: List[Any], y_true: List[Any]) -> tuple[float, float, float, float]:
        accuracy = sum(1 for p, t in zip(y_pred, y_true) if p == t) / len(y_pred)
        return accuracy, accuracy, accuracy, accuracy  # Simplified