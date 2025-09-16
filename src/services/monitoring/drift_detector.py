"""
Drift detector that coordinates different drift detection strategies.
"""

from typing import Dict, Optional, Any
from datetime import datetime

from src.models.monitoring import DriftType, DriftDetectionResult
from src.services.monitoring.drift_strategies import DriftDetectionContext
from src.services.monitoring.exceptions import DriftDetectionError
from src.utils.logging import get_logger

logger = get_logger("drift_detector")


class DriftDetector:
    """Coordinates drift detection using different strategies."""
    
    def __init__(self):
        self.drift_context = DriftDetectionContext()
        self.drift_thresholds = {
            DriftType.DATA_DRIFT: 0.05,
            DriftType.PERFORMANCE_DRIFT: 0.1,
            DriftType.CONCEPT_DRIFT: 0.15,
            DriftType.DATA_QUALITY_DRIFT: 5.0  # Number of anomalies threshold
        }
    
    async def detect_data_drift(self, model_name: str, data: Optional[Dict[str, Any]] = None) -> Optional[DriftDetectionResult]:
        """Detect data drift for a model."""
        try:
            if data is None:
                # In a real implementation, this would fetch data from storage
                logger.warning(f"No data provided for data drift detection for {model_name}")
                return None
            
            threshold = self.drift_thresholds[DriftType.DATA_DRIFT]
            
            result = await self.drift_context.detect_drift(
                DriftType.DATA_DRIFT,
                model_name,
                data,
                threshold
            )
            
            if result:
                logger.info(f"Data drift detection completed for {model_name}: detected={result.detected}")
            
            return result
            
        except Exception as e:
            logger.error(f"Data drift detection failed for {model_name}: {e}")
            raise DriftDetectionError(model_name, "data_drift", str(e))
    
    async def detect_performance_drift(self, model_name: str, data: Optional[Dict[str, Any]] = None) -> Optional[DriftDetectionResult]:
        """Detect performance drift for a model."""
        try:
            if data is None:
                # In a real implementation, this would fetch performance data from storage
                logger.warning(f"No data provided for performance drift detection for {model_name}")
                return None
            
            threshold = self.drift_thresholds[DriftType.PERFORMANCE_DRIFT]
            
            result = await self.drift_context.detect_drift(
                DriftType.PERFORMANCE_DRIFT,
                model_name,
                data,
                threshold
            )
            
            if result:
                logger.info(f"Performance drift detection completed for {model_name}: detected={result.detected}")
            
            return result
            
        except Exception as e:
            logger.error(f"Performance drift detection failed for {model_name}: {e}")
            raise DriftDetectionError(model_name, "performance_drift", str(e))
    
    async def detect_concept_drift(self, model_name: str, data: Optional[Dict[str, Any]] = None) -> Optional[DriftDetectionResult]:
        """Detect concept drift for a model."""
        try:
            if data is None:
                logger.warning(f"No data provided for concept drift detection for {model_name}")
                return None
            
            threshold = self.drift_thresholds[DriftType.CONCEPT_DRIFT]
            
            result = await self.drift_context.detect_drift(
                DriftType.CONCEPT_DRIFT,
                model_name,
                data,
                threshold
            )
            
            if result:
                logger.info(f"Concept drift detection completed for {model_name}: detected={result.detected}")
            
            return result
            
        except Exception as e:
            logger.error(f"Concept drift detection failed for {model_name}: {e}")
            raise DriftDetectionError(model_name, "concept_drift", str(e))
    
    async def detect_data_quality_drift(self, model_name: str, data: Optional[Dict[str, Any]] = None) -> Optional[DriftDetectionResult]:
        """Detect data quality drift for a model."""
        try:
            if data is None:
                logger.warning(f"No data provided for data quality drift detection for {model_name}")
                return None
            
            threshold = self.drift_thresholds[DriftType.DATA_QUALITY_DRIFT]
            
            result = await self.drift_context.detect_drift(
                DriftType.DATA_QUALITY_DRIFT,
                model_name,
                data,
                threshold
            )
            
            if result:
                logger.info(f"Data quality drift detection completed for {model_name}: detected={result.detected}")
            
            return result
            
        except Exception as e:
            logger.error(f"Data quality drift detection failed for {model_name}: {e}")
            raise DriftDetectionError(model_name, "data_quality_drift", str(e))
    
    async def detect_drift(self, drift_type: DriftType, model_name: str, data: Dict[str, Any], threshold: Optional[float] = None) -> Optional[DriftDetectionResult]:
        """Generic drift detection method."""
        if threshold is None:
            threshold = self.drift_thresholds.get(drift_type, 0.05)
        
        return await self.drift_context.detect_drift(drift_type, model_name, data, threshold)
    
    def set_drift_threshold(self, drift_type: DriftType, threshold: float) -> None:
        """Set drift threshold for a specific drift type."""
        self.drift_thresholds[drift_type] = threshold
        logger.info(f"Drift threshold set for {drift_type.value}: {threshold}")
    
    def get_drift_thresholds(self) -> Dict[DriftType, float]:
        """Get current drift thresholds."""
        return self.drift_thresholds.copy()