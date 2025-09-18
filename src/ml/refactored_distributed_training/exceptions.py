"""Custom exceptions for distributed training system.

This module provides a comprehensive exception hierarchy for handling errors
in the distributed training system of the AI trading platform.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import logging


class TrainingError(Exception):
    """Base exception for all distributed training errors.
    
    Provides common functionality for error tracking, logging, and context
    management across the distributed training system.
    
    Attributes:
        message: Human-readable error message
        error_code: Machine-readable error code for programmatic handling
        context: Additional context information for debugging
        recoverable: Whether the error might be recoverable with retry
        timestamp: When the error occurred
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: str = "TRAINING_ERROR",
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = False
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.recoverable = recoverable
        self.timestamp = datetime.utcnow()
        
        # Log the error automatically
        logger = logging.getLogger(__name__)
        logger.error(
            f"Training error [{self.error_code}]: {message}", 
            extra=self.context
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp.isoformat()
        }


class ModelLoadError(TrainingError):
    """Raised when model loading fails during distributed training.
    
    This exception is typically raised when:
    - Model checkpoint files are corrupted or missing
    - Model architecture doesn't match checkpoint
    - Insufficient memory to load model on worker nodes
    - Model file format is incompatible
    
    Example:
        try:
            model = load_checkpoint(checkpoint_path)
        except ModelLoadError as e:
            logger.error(f"Failed to load model: {e}")
            # Implement fallback or retry logic
    """
    
    def __init__(
        self, 
        message: str, 
        model_path: Optional[str] = None,
        model_type: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(message, error_code="MODEL_LOAD_FAILED", **kwargs)
        self.model_path = model_path
        self.model_type = model_type


class DataPreparationError(TrainingError):
    """Raised when data preparation fails before training.
    
    Common scenarios include:
    - Data validation failures
    - Feature engineering pipeline errors
    - Data format inconsistencies
    - Missing or corrupted data files
    """
    
    def __init__(
        self, 
        message: str, 
        data_source: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
        original_error: Optional[Exception] = None,
        **kwargs
    ) -> None:
        super().__init__(message, error_code="DATA_PREP_ERROR", **kwargs)
        self.data_source = data_source
        self.validation_errors = validation_errors or []
        self.original_error = original_error
        if original_error:
            self.__cause__ = original_error


class ResourceAllocationError(TrainingError):
    """Raised when resource allocation fails during distributed training setup.
    
    This includes failures in:
    - GPU/CPU allocation
    - Memory allocation
    - Storage allocation
    - Network bandwidth allocation
    """
    
    def __init__(
        self, 
        message: str, 
        resource_type: Optional[str] = None,
        requested_amount: Optional[int] = None,
        available_amount: Optional[int] = None,
        node_id: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(
            message, 
            error_code="RESOURCE_ALLOCATION_FAILED", 
            **kwargs
        )
        self.resource_type = resource_type
        self.requested_amount = requested_amount
        self.available_amount = available_amount
        self.node_id = node_id


class JobSubmissionError(TrainingError):
    """Raised when distributed training job submission fails.
    
    Common causes:
    - Ray cluster unavailable or overloaded
    - Invalid job configuration
    - Insufficient cluster resources
    - Network connectivity issues
    """
    
    def __init__(
        self, 
        message: str, 
        job_id: Optional[str] = None,
        cluster_status: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(
            message, 
            error_code="JOB_SUBMISSION_FAILED", 
            recoverable=True,
            **kwargs
        )
        self.job_id = job_id
        self.cluster_status = cluster_status
    
    def get_recovery_suggestions(self) -> List[str]:
        """Get suggested recovery actions."""
        return [
            "Check Ray cluster status and available resources",
            "Verify job configuration parameters",
            "Retry with exponential backoff",
            "Consider reducing resource requirements"
        ]


class WorkerInitializationError(TrainingError):
    """Raised when worker node initialization fails.
    
    This can occur due to:
    - Environment setup failures
    - Dependency installation issues
    - Configuration synchronization problems
    - Worker node hardware issues
    """
    
    def __init__(
        self, 
        message: str, 
        worker_id: Optional[str] = None,
        initialization_stage: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(message, error_code="WORKER_INIT_FAILED", **kwargs)
        self.worker_id = worker_id
        self.initialization_stage = initialization_stage


class HealthMonitorError(TrainingError):
    """Raised when health monitoring system fails.
    
    This includes failures in:
    - Metric collection
    - Health check execution
    - Monitoring service communication
    - Alert system failures
    """
    
    def __init__(
        self, 
        message: str, 
        monitor_type: Optional[str] = None,
        affected_components: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        super().__init__(message, error_code="HEALTH_MONITOR_FAILED", **kwargs)
        self.monitor_type = monitor_type
        self.affected_components = affected_components or []


class ConfigurationError(TrainingError):
    """Raised when training configuration is invalid or incomplete.
    
    Common issues:
    - Missing required configuration parameters
    - Invalid parameter values or types
    - Conflicting configuration settings
    - Environment-specific configuration errors
    """
    
    def __init__(
        self, 
        message: str, 
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs
    ) -> None:
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)
        self.config_key = config_key
        self.config_value = config_value


class NetworkError(TrainingError):
    """Raised when network communication fails between training nodes.
    
    This includes:
    - Inter-node communication failures
    - Parameter server connectivity issues
    - Data transfer failures
    - Network timeout errors
    """
    
    def __init__(
        self, 
        message: str, 
        source_node: Optional[str] = None,
        target_node: Optional[str] = None,
        network_operation: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(
            message, 
            error_code="NETWORK_ERROR", 
            recoverable=True, 
            **kwargs
        )
        self.source_node = source_node
        self.target_node = target_node
        self.network_operation = network_operation


class CheckpointError(TrainingError):
    """Raised when model checkpointing fails.
    
    Common scenarios:
    - Insufficient storage space
    - File system permission issues
    - Checkpoint corruption during write
    - Network storage unavailability
    """
    
    def __init__(
        self, 
        message: str, 
        checkpoint_path: Optional[str] = None,
        checkpoint_type: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(message, error_code="CHECKPOINT_ERROR", **kwargs)
        self.checkpoint_path = checkpoint_path
        self.checkpoint_type = checkpoint_type
