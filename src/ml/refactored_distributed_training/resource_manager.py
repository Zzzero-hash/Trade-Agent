"""Resource management for distributed training system"""

import logging
from contextlib import contextmanager
from typing import Generator

# Ray imports with fallback
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None

from .data_classes import DistributedTrainingConfig
from .exceptions import ResourceAllocationError


@contextmanager
def ray_cluster_context(
    config: DistributedTrainingConfig
) -> Generator[bool, None, None]:
    """Context manager for Ray cluster lifecycle
    
    Args:
        config: Distributed training configuration
        
    Yields:
        bool: True if Ray was successfully initialized, False otherwise
        
    Raises:
        ResourceAllocationError: If Ray initialization fails
    """
    ray_initialized = False
    ray_logger = logging.getLogger(__name__)
    
    try:
        if not RAY_AVAILABLE:
            ray_logger.warning("Ray not available, using local training")
            yield False
            return
            
        if not ray.is_initialized():
            ray.init(
                num_cpus=config.num_workers * config.cpus_per_worker,
                num_gpus=int(config.num_workers * config.gpus_per_worker),
                object_store_memory=int(2e9),
                ignore_reinit_error=True,
            )
        ray_initialized = True
        ray_logger.info("Ray cluster initialized successfully")
        yield ray_initialized
        
    except Exception as e:
        ray_logger.error(f"Failed to initialize Ray: {e}")
        yield False
        
    finally:
        if ray_initialized and ray and ray.is_initialized():
            ray.shutdown()
            ray_logger.info("Ray cluster shutdown complete")


class ResourceManager:
    """Manages resources for distributed training"""

    def __init__(self, config: DistributedTrainingConfig):
        """Initialize resource manager
        
        Args:
            config: Distributed training configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.ray_context = None
        self.ray_initialized = False

    def initialize_resources(self) -> None:
        """Initialize training resources"""
        try:
            self.ray_context = ray_cluster_context(self.config)
            # Start the context manager
            self.ray_initialized = next(self.ray_context)
        except Exception as e:
            raise ResourceAllocationError(
                f"Failed to initialize resources: {e}"
            ) from e

    def cleanup_resources(self) -> None:
        """Clean up training resources"""
        if self.ray_context:
            try:
                # Properly close the context manager
                try:
                    next(self.ray_context)
                except StopIteration:
                    pass  # Expected when generator is exhausted
            except Exception as e:
                self.logger.warning(f"Error cleaning up resources: {e}")
            finally:
                self.ray_context = None
                self.ray_initialized = False
