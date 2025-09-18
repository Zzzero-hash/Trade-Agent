"""Test for the refactored distributed training system."""

import sys
import os
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.ml.refactored_distributed_training import (
    DistributedTrainingConfig,
    create_distributed_training_system
)


def test_refactored_system():
    """Test the refactored distributed training system."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = DistributedTrainingConfig(
        num_workers=2,
        cpus_per_worker=1,
        gpus_per_worker=0.0,
        max_retries=1,
        health_check_interval=10.0
    )
    
    # Create system
    orchestrator = create_distributed_training_system(config)
    
    # Test job submission
    try:
        job_id = orchestrator.submit_training_job(
            model_type="cnn",
            config={
                "input_dim": 10,
                "output_dim": 1,
                "features": [],
                "targets": []
            },
            priority=1
        )
        print(f"Successfully submitted job: {job_id}")
    except Exception as e:
        print(f"Error submitting job: {e}")
    
    # Test getting job status
    try:
        status = orchestrator.get_job_status("nonexistent_job")
        print(f"Status for nonexistent job: {status}")
    except Exception as e:
        print(f"Error getting job status: {e}")
    
    # Test cluster status
    try:
        cluster_status = orchestrator.get_cluster_status()
        print(f"Cluster status: {cluster_status}")
    except Exception as e:
        print(f"Error getting cluster status: {e}")
    
    # Shutdown system
    try:
        orchestrator.shutdown()
        print("Successfully shut down system")
    except Exception as e:
        print(f"Error shutting down system: {e}")


if __name__ == "__main__":
    test_refactored_system()
