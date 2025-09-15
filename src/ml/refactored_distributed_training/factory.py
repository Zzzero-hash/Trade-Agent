"""Factory for creating distributed training system components."""

from typing import Optional

from .data_classes import DistributedTrainingConfig
from .orchestrator import DistributedTrainingOrchestrator


def create_distributed_training_system(
    config: Optional[DistributedTrainingConfig] = None,
) -> DistributedTrainingOrchestrator:
    """Create a distributed training system

    Args:
        config: Optional configuration (uses defaults if None)

    Returns:
        Configured distributed training orchestrator
    """
    if config is None:
        config = DistributedTrainingConfig()

    orchestrator = DistributedTrainingOrchestrator(config)
    return orchestrator
