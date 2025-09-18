"""Refactored distributed training system for the AI trading platform.

This package provides a modular, maintainable implementation of distributed 
training capabilities with proper separation of concerns, strategy patterns, 
and resource management.
"""

from .data_classes import DistributedTrainingConfig, TrainingJob
from .orchestrator import DistributedTrainingOrchestrator
from .factory import create_distributed_training_system
from .exceptions import (
    TrainingError,
    ModelLoadError,
    DataPreparationError,
    ResourceAllocationError,
    JobSubmissionError,
    WorkerInitializationError,
    HealthMonitorError,
    ConfigurationError,
    NetworkError,
    CheckpointError,
)

__all__ = [
    "DistributedTrainingConfig",
    "TrainingJob",
    "DistributedTrainingOrchestrator",
    "create_distributed_training_system",
    "TrainingError",
    "ModelLoadError",
    "DataPreparationError",
    "ResourceAllocationError",
    "JobSubmissionError",
    "WorkerInitializationError",
    "HealthMonitorError",
    "ConfigurationError",
    "NetworkError",
    "CheckpointError",
]
