"""Data classes for distributed training system"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime, timedelta


@dataclass
class DistributedTrainingConfig:
    """Configuration for distributed training"""

    # Resource allocation
    num_workers: int = 4
    cpus_per_worker: int = 2
    gpus_per_worker: float = 0.25
    memory_per_worker: str = "4GB"

    # Training parameters
    max_concurrent_trials: int = 8
    training_timeout: timedelta = timedelta(hours=6)
    checkpoint_frequency: int = 10  # epochs

    # Fault tolerance
    max_retries: int = 3
    retry_delay: float = 30.0  # seconds
    health_check_interval: float = 60.0  # seconds

    # Storage
    checkpoint_dir: str = "distributed_checkpoints"
    log_dir: str = "distributed_logs"
    results_dir: str = "distributed_results"

    # Optimization
    use_mixed_precision: bool = True
    gradient_compression: bool = True
    async_checkpointing: bool = True


@dataclass
class TrainingJob:
    """Represents a single training job"""

    job_id: str
    model_type: str
    config: Dict[str, Any]
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed, cancelled
    worker_id: Optional[str] = None
    retry_count: int = 0
    error_message: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
