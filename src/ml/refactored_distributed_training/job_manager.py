"""Job management for distributed training system"""

import queue
import threading
from typing import Dict, Optional, Any
from datetime import datetime

from .data_classes import TrainingJob
from .exceptions import JobSubmissionError


class ThreadSafeJobManager:
    """Thread-safe job management with fine-grained locking"""

    def __init__(self):
        """Initialize thread-safe job manager"""
        self._active_jobs: Dict[str, TrainingJob] = {}
        self._completed_jobs: Dict[str, TrainingJob] = {}
        self._job_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.RLock()
        self._job_queue = queue.PriorityQueue()

    def submit_job(self, job: TrainingJob) -> str:
        """Submit a training job to the queue
        
        Args:
            job: Training job to submit
            
        Returns:
            Job ID
            
        Raises:
            JobSubmissionError: If job submission fails
        """
        try:
            # Add to queue (priority queue uses tuple: (priority, job))
            self._job_queue.put((job.priority, job))
            return job.job_id
        except Exception as e:
            raise JobSubmissionError(
                f"Failed to submit job {job.job_id}: {e}",
                job_id=job.job_id
            ) from e

    def get_next_job(self) -> Optional[TrainingJob]:
        """Get the next job from the queue
        
        Returns:
            Next training job or None if queue is empty
        """
        try:
            priority, job = self._job_queue.get_nowait()
            return job
        except queue.Empty:
            return None

    def update_job_status(
        self, 
        job_id: str, 
        status: str, 
        **kwargs
    ) -> None:
        """Thread-safe job status update
        
        Args:
            job_id: ID of job to update
            status: New status
            **kwargs: Additional attributes to update
        """
        with self._global_lock:
            if job_id not in self._job_locks:
                self._job_locks[job_id] = threading.Lock()

        with self._job_locks[job_id]:
            if job_id in self._active_jobs:
                job = self._active_jobs[job_id]
                job.status = status
                for key, value in kwargs.items():
                    setattr(job, key, value)

    def move_job_to_completed(self, job_id: str) -> None:
        """Thread-safe job completion
        
        Args:
            job_id: ID of job to complete
        """
        with self._global_lock:
            if job_id in self._active_jobs:
                job = self._active_jobs.pop(job_id)
                self._completed_jobs[job_id] = job
                # Clean up job-specific lock
                self._job_locks.pop(job_id, None)

    def add_active_job(self, job: TrainingJob) -> None:
        """Add a job to the active jobs list
        
        Args:
            job: Job to add
        """
        with self._global_lock:
            self._active_jobs[job.job_id] = job
            if job.job_id not in self._job_locks:
                self._job_locks[job.job_id] = threading.Lock()

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a training job
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status dictionary or None if not found
        """
        with self._global_lock:
            # Check active jobs
            if job_id in self._active_jobs:
                job = self._active_jobs[job_id]
                return self._job_to_status_dict(job)

            # Check completed jobs
            if job_id in self._completed_jobs:
                job = self._completed_jobs[job_id]
                return self._job_to_status_dict(job)

        return None

    def _job_to_status_dict(self, job: TrainingJob) -> Dict[str, Any]:
        """Convert job to status dictionary
        
        Args:
            job: Job to convert
            
        Returns:
            Status dictionary
        """
        return {
            "job_id": job.job_id,
            "status": job.status,
            "model_type": job.model_type,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat()
            if job.started_at
            else None,
            "completed_at": job.completed_at.isoformat()
            if job.completed_at
            else None,
            "worker_id": job.worker_id,
            "retry_count": job.retry_count,
            "error_message": job.error_message,
            "results": job.results,
        }

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get cluster status information
        
        Returns:
            Cluster status dictionary
        """
        with self._global_lock:
            active_jobs = len(self._active_jobs)
            completed_jobs = len(self._completed_jobs)
            queued_jobs = self._job_queue.qsize()

        return {
            "active_jobs": active_jobs,
            "completed_jobs": completed_jobs,
            "queued_jobs": queued_jobs,
            "total_jobs": active_jobs + completed_jobs + queued_jobs,
        }

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if job was cancelled, False otherwise
        """
        with self._global_lock:
            if job_id in self._active_jobs:
                job = self._active_jobs[job_id]
                job.status = "cancelled"
                job.completed_at = datetime.now()
                self._completed_jobs[job_id] = job
                del self._active_jobs[job_id]
                return True

        return False

    def task_done(self) -> None:
        """Indicate that a formerly enqueued task is complete"""
        self._job_queue.task_done()

    def join(self) -> None:
        """Block until all items in the queue have been processed"""
        self._job_queue.join()
