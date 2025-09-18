"""
Monitoring orchestrator that coordinates different monitoring components.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass

from src.utils.logging import get_logger
from .performance_tracker import PerformanceTracker
from .drift_detector import DriftDetector
from .alert_manager import AlertManager

logger = get_logger("monitoring_orchestrator")


@dataclass
class MonitoringResult:
    """Result of a monitoring cycle."""
    model_name: str
    timestamp: datetime
    performance_metrics: Optional[Any] = None
    drift_detection: Dict[str, Any] = None
    alerts_generated: list = None
    
    def __post_init__(self):
        if self.drift_detection is None:
            self.drift_detection = {'data_drift': None, 'performance_drift': None}
        if self.alerts_generated is None:
            self.alerts_generated = []


class MonitoringOrchestrator:
    """Orchestrates the monitoring workflow."""
    
    def __init__(
        self,
        performance_tracker: PerformanceTracker,
        drift_detector: DriftDetector,
        alert_manager: AlertManager
    ):
        self.performance_tracker = performance_tracker
        self.drift_detector = drift_detector
        self.alert_manager = alert_manager
    
    async def run_monitoring_cycle(self, model_name: str, prediction_data: list) -> MonitoringResult:
        """Run a complete monitoring cycle for a model."""
        
        result = MonitoringResult(
            model_name=model_name,
            timestamp=datetime.now()
        )
        
        try:
            # Step 1: Calculate performance metrics
            result.performance_metrics = await self._calculate_performance(
                model_name, prediction_data
            )
            
            # Step 2: Check performance thresholds
            if result.performance_metrics:
                await self._check_performance_thresholds(model_name, result.performance_metrics)
            
            # Step 3: Detect drift
            result.drift_detection = await self._detect_drift(model_name)
            
            logger.info(f"Monitoring cycle completed for model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error in monitoring cycle for {model_name}: {e}")
            await self._handle_monitoring_error(model_name, e)
        
        return result
    
    async def _calculate_performance(self, model_name: str, prediction_data: list) -> Optional[Any]:
        """Calculate performance metrics step."""
        return await self.performance_tracker.calculate_performance_metrics(
            model_name, prediction_data
        )
    
    async def _check_performance_thresholds(self, model_name: str, metrics: Any) -> None:
        """Check performance thresholds step."""
        await self.alert_manager.check_performance_thresholds(model_name, metrics)
    
    async def _detect_drift(self, model_name: str) -> Dict[str, Any]:
        """Detect drift step."""
        return {
            'data_drift': await self.drift_detector.detect_data_drift(model_name),
            'performance_drift': await self.drift_detector.detect_performance_drift(model_name)
        }
    
    async def _handle_monitoring_error(self, model_name: str, error: Exception) -> None:
        """Handle monitoring errors."""
        await self.alert_manager.send_system_error_alert(model_name, error)