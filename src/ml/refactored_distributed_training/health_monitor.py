"""Health monitoring for distributed training system"""

import threading
import time
import logging

from .data_classes import DistributedTrainingConfig
from .exceptions import HealthMonitorError
from .job_manager import ThreadSafeJobManager


class HealthMonitor:
    """Monitors system health and job timeouts"""

    def __init__(
        self, 
        job_manager: ThreadSafeJobManager, 
        config: DistributedTrainingConfig
    ):
        """Initialize health monitor
        
        Args:
            job_manager: Job manager to monitor
            config: Distributed training configuration
        """
        self.job_manager = job_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.monitor_thread: threading.Thread = None
        self.is_running = False

    def start_monitoring(self) -> None:
        """Start health monitoring"""
        try:
            self.is_running = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop, 
                daemon=True
            )
            self.monitor_thread.start()
            self.logger.info("Health monitor started")
        except Exception as e:
            raise HealthMonitorError(
                f"Failed to start health monitor: {e}"
            ) from e

    def stop_monitoring(self) -> None:
        """Stop health monitoring"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Health monitor stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_running:
            try:
                self._check_job_timeouts()
                self._log_status()
                time.sleep(self.config.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                time.sleep(10)

    def _check_job_timeouts(self) -> None:
        """Check for timed out jobs"""
        # Note: In a full implementation, we would need access to the active jobs
        # This is a simplified version that demonstrates the concept
        pass

    def _log_status(self) -> None:
        """Log system status"""
        try:
            status = self.job_manager.get_cluster_status()
            self.logger.info(
                "Training status - Active: %d, Completed: %d, Queued: %d",
                status['active_jobs'],
                status['completed_jobs'],
                status['queued_jobs']
            )
        except Exception as e:
            self.logger.warning(f"Failed to get cluster status: {e}")
