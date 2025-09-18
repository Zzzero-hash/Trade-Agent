"""Worker management for distributed training system"""

import threading
import time
import logging
from typing import List, Dict, Any

from .data_classes import DistributedTrainingConfig
from .exceptions import WorkerInitializationError, TrainingError
from .model_training_strategies import TrainingStrategyFactory


class WorkerManager:
    """Manages distributed workers for training jobs"""

    def __init__(self, config: DistributedTrainingConfig):
        """Initialize worker manager
        
        Args:
            config: Distributed training configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.worker_pool: List[threading.Thread] = []
        self.is_running = False

    def start_workers(self) -> None:
        """Start distributed training workers"""
        try:
            self._start_local_workers()
            self.is_running = True
            self.logger.info(f"Started {self.config.num_workers} workers")
        except Exception as e:
            raise WorkerInitializationError(
                f"Failed to start workers: {e}"
            ) from e

    def _start_local_workers(self) -> None:
        """Start local thread-based workers as fallback"""

        def worker_thread(worker_id: str, job_manager):
            """Worker thread function"""
            logger = logging.getLogger(f"worker_{worker_id}")

            while self.is_running:
                try:
                    # Get job from job manager
                    job = job_manager.get_next_job()
                    if job is None:
                        time.sleep(0.1)  # Brief pause to prevent busy waiting
                        continue

                    # Add job to active jobs
                    job_manager.add_active_job(job)

                    # Update job status
                    job_manager.update_job_status(
                        job.job_id, 
                        "running", 
                        started_at=time.time(),
                        worker_id=worker_id
                    )

                    logger.info(f"Starting training job {job.job_id}")

                    try:
                        # Train model
                        strategy = TrainingStrategyFactory.get_strategy(
                            job.model_type
                        )
                        results = strategy.train(job, self.config)

                        # Update job with results
                        job_manager.update_job_status(
                            job.job_id,
                            "completed",
                            completed_at=time.time(),
                            results=results
                        )
                        
                        job_manager.move_job_to_completed(job.job_id)
                        logger.info(f"Completed training job {job.job_id}")

                    except TrainingError as e:
                        logger.error(
                            f"Training failed for job {job.job_id}: {e}"
                        )
                        
                        # Handle retry logic
                        if job.retry_count < self.config.max_retries:
                            job.retry_count += 1
                            job_manager.update_job_status(
                                job.job_id,
                                "pending",
                                error_message=str(e)
                            )
                            # Re-queue job
                            job_manager.submit_job(job)
                            logger.info(
                                f"Re-queued job {job.job_id} "
                                f"(retry {job.retry_count})"
                            )
                        else:
                            job_manager.update_job_status(
                                job.job_id,
                                "failed",
                                completed_at=time.time(),
                                error_message=str(e)
                            )
                            job_manager.move_job_to_completed(job.job_id)
                            logger.error(
                                f"Job {job.job_id} failed after "
                                f"{job.retry_count} retries"
                            )

                    except Exception as e:
                        logger.error(
                            f"Unexpected error in job {job.job_id}: {e}"
                        )
                        job_manager.update_job_status(
                            job.job_id,
                            "failed",
                            completed_at=time.time(),
                            error_message=str(e)
                        )
                        job_manager.move_job_to_completed(job.job_id)

                    finally:
                        job_manager.task_done()

                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}")
                    time.sleep(1)

        # Start worker threads
        self.worker_pool = []
        for i in range(self.config.num_workers):
            worker_id = f"local_worker_{i}"
            thread = threading.Thread(
                target=worker_thread, 
                args=(worker_id, None),  # job_manager will be passed later
                daemon=True
            )
            thread.start()
            self.worker_pool.append(thread)

    def stop_workers(self) -> None:
        """Stop all workers"""
        self.is_running = False
        for thread in self.worker_pool:
            thread.join(timeout=5.0)  # Wait up to 5 seconds for each thread
        self.worker_pool.clear()
        self.logger.info("All workers stopped")

    def get_worker_status(self) -> Dict[str, Any]:
        """Get worker status information
        
        Returns:
            Worker status dictionary
        """
        alive_workers = sum(
            1 for worker in self.worker_pool if worker.is_alive()
        )
        
        return {
            "total_workers": len(self.worker_pool),
            "alive_workers": alive_workers,
            "dead_workers": len(self.worker_pool) - alive_workers,
            "is_running": self.is_running
        }
