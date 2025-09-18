"""Refactored Distributed Training Orchestration System using Ray

This module provides a comprehensive distributed training system that 
orchestrates model training across multiple workers, handles fault 
tolerance, and manages resource allocation efficiently.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from .data_classes import DistributedTrainingConfig, TrainingJob
from .job_manager import ThreadSafeJobManager
from .worker_manager import WorkerManager
from .health_monitor import HealthMonitor
from .resource_manager import ResourceManager
from .model_training_strategies import TrainingStrategyFactory


class SetupService:
    """Handles initialization of distributed training system"""
    def __init__(self, config: DistributedTrainingConfig) -> None:
        self.config = config
        
    def initialize(self) -> None:
        self._create_directories()
        self._setup_logging()
        

class DistributedTrainingOrchestrator:
    """Main orchestrator for distributed training operations"""

    def __init__(self, 
                 config: DistributedTrainingConfig,
                 setup_service: Optional[SetupService] = None
                 ) -> None:
        """Initialize distributed training orchestrator

        Args:
            config: Distributed training configuration
            setup_service: Optional setup service for distributed training
        """
        self.config = config
        self.setup_service = setup_service or SetupService(config)
        
        # Delegate setup responsibilities
        self.setup_service.initialize()

        # Focus on orchestration only
        self.job_manager = ThreadSafeJobManager()
        self.resource_manager = ResourceManager(config)
        self.worker_manager = WorkerManager(config)
        self.health_monitor = HealthMonitor(self.job_manager, config)

    def _create_directories(self) -> None:
        """Create necessary directories"""
        for directory in [
            self.config.checkpoint_dir,
            self.config.log_dir,
            self.config.results_dir,
        ]:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> None:
        """Setup distributed logging"""
        log_file = Path(self.config.log_dir) / "orchestrator.log"

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)

    def submit_training_job(
        self,
        model_type: str,
        config: Dict[str, Any],
        priority: int = 1,
        job_id: Optional[str] = None,
    ) -> str:
        """Submit a training job to the queue

        Args:
            model_type: Type of model to train ('cnn', 'lstm', 'hybrid', 'rl')
            config: Training configuration
            priority: Job priority (lower = higher priority)
            job_id: Optional custom job ID

        Returns:
            Job ID
            
        Raises:
            JobSubmissionError: If job submission fails
            ValueError: If model type is not supported
        """
        # Validate model type
        if model_type not in TrainingStrategyFactory._strategies:
            raise ValueError(f"Unsupported model type: {model_type}")

        if job_id is None:
            job_id = f"{model_type}_{int(time.time() * 1000000)}"

        job = TrainingJob(
            job_id=job_id, 
            model_type=model_type, 
            config=config, 
            priority=priority
        )

        return self.job_manager.submit_job(job)

    def start_training_workers(self) -> None:
        """Start distributed training workers"""
        try:
            self.worker_manager.start_workers()
            self.health_monitor.start_monitoring()
            self.logger.info("Training workers and health monitor started")
        except Exception as e:
            self.logger.error(f"Failed to start training system: {e}")
            raise

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a training job

        Args:
            job_id: Job identifier

        Returns:
            Job status dictionary or None if not found
        """
        return self.job_manager.get_job_status(job_id)

    def wait_for_completion(
        self, job_ids: List[str], timeout: Optional[float] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Wait for jobs to complete

        Args:
            job_ids: List of job IDs to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            Dictionary of job results
            
        Raises:
            TimeoutError: If timeout is exceeded before all jobs complete
        """
        start_time: float = time.time()
        results: Dict[str, Dict[str, Any]] = {}

        while len(results) < len(job_ids):
            if timeout and (time.time() - start_time) > timeout:
                incomplete_jobs = set(job_ids) - set(results.keys())
                raise TimeoutError(
                    f"Timeout waiting for jobs: {incomplete_jobs}"
                )

            for job_id in job_ids:
                if job_id not in results:
                    status = self.get_job_status(job_id)
                    if status and status["status"] in ["completed", "failed"]:
                        results[job_id] = status

            time.sleep(1.0)  # Explicit float for clarity

        return results

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job

        Args:
            job_id: Job identifier

        Returns:
            True if job was cancelled, False otherwise
        """
        return self.job_manager.cancel_job(job_id)

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get cluster status information

        Returns:
            Cluster status dictionary
        """
        return self.job_manager.get_cluster_status()

    def shutdown(self) -> None:
        """Shutdown the distributed training system"""
        self.logger.info("Shutting down distributed training system")

        try:
            # Stop health monitoring
            self.health_monitor.stop_monitoring()
        except Exception as e:
            self.logger.warning(f"Error stopping health monitor: {e}")

        try:
            # Stop workers
            self.worker_manager.stop_workers()
        except Exception as e:
            self.logger.warning(f"Error stopping workers: {e}")

        try:
            # Cleanup resources
            self.resource_manager.cleanup_resources()
        except Exception as e:
            self.logger.warning(f"Error cleaning up resources: {e}")

        self.logger.info("Distributed training system shutdown complete")
