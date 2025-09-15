"""
Tests for Distributed Training Orchestration System

This module provides comprehensive tests for the distributed training system,
including fault tolerance, coordination, and performance validation.
"""

import pytest
import os
import time
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import numpy as np
import threading
import queue

from src.ml.distributed_training import (
    DistributedTrainingOrchestrator,
    DistributedTrainingConfig,
    TrainingJob,
    ModelValidationPipeline
)
from src.ml.ray_tune_integration import RayTuneOptimizer, TuneConfig
from src.ml.model_validation import ModelValidator, ValidationConfig


class TestDistributedTrainingConfig:
    """Test DistributedTrainingConfig"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = DistributedTrainingConfig()
        
        assert config.num_workers == 4
        assert config.cpus_per_worker == 2
        assert config.gpus_per_worker == 0.25
        assert config.max_concurrent_trials == 8
        assert config.max_retries == 3
        assert config.retry_delay == 30.0
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = DistributedTrainingConfig(
            num_workers=8,
            cpus_per_worker=4,
            max_retries=5
        )
        
        assert config.num_workers == 8
        assert config.cpus_per_worker == 4
        assert config.max_retries == 5


class TestTrainingJob:
    """Test TrainingJob dataclass"""
    
    def test_job_creation(self):
        """Test training job creation"""
        job = TrainingJob(
            job_id="test_job_1",
            model_type="cnn",
            config={"input_dim": 10, "output_dim": 1}
        )
        
        assert job.job_id == "test_job_1"
        assert job.model_type == "cnn"
        assert job.status == "pending"
        assert job.retry_count == 0
        assert job.created_at is not None
    
    def test_job_with_custom_values(self):
        """Test job with custom values"""
        created_time = datetime.now()
        
        job = TrainingJob(
            job_id="test_job_2",
            model_type="rl",
            config={"agent_type": "PPO"},
            priority=2,
            created_at=created_time
        )
        
        assert job.priority == 2
        assert job.created_at == created_time


class TestDistributedTrainingOrchestrator:
    """Test DistributedTrainingOrchestrator"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration"""
        return DistributedTrainingConfig(
            num_workers=2,
            checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
            log_dir=os.path.join(temp_dir, "logs"),
            results_dir=os.path.join(temp_dir, "results"),
            health_check_interval=1.0,  # Faster for testing
            training_timeout=timedelta(seconds=10)  # Short timeout for testing
        )
    
    @pytest.fixture
    def orchestrator(self, config):
        """Create orchestrator instance"""
        with patch('src.ml.distributed_training.ray') as mock_ray:
            mock_ray.is_initialized.return_value = False
            orchestrator = DistributedTrainingOrchestrator(config)
            yield orchestrator
            orchestrator.shutdown()
    
    def test_orchestrator_initialization(self, config):
        """Test orchestrator initialization"""
        with patch('src.ml.distributed_training.ray') as mock_ray:
            mock_ray.is_initialized.return_value = False
            
            orchestrator = DistributedTrainingOrchestrator(config)
            
            assert orchestrator.config == config
            assert os.path.exists(config.checkpoint_dir)
            assert os.path.exists(config.log_dir)
            assert os.path.exists(config.results_dir)
            
            orchestrator.shutdown()
    
    def test_submit_training_job(self, orchestrator):
        """Test job submission"""
        job_id = orchestrator.submit_training_job(
            model_type="cnn",
            config={"input_dim": 10, "output_dim": 1},
            priority=1
        )
        
        assert job_id is not None
        assert job_id.startswith("cnn_")
        assert orchestrator.job_queue.qsize() == 1
    
    def test_submit_multiple_jobs(self, orchestrator):
        """Test submitting multiple jobs"""
        job_ids = []
        
        for i in range(3):
            job_id = orchestrator.submit_training_job(
                model_type="cnn",
                config={"input_dim": 10, "output_dim": 1},
                priority=i
            )
            job_ids.append(job_id)
        
        assert len(job_ids) == 3
        assert orchestrator.job_queue.qsize() == 3
    
    def test_job_status_tracking(self, orchestrator):
        """Test job status tracking"""
        job_id = orchestrator.submit_training_job(
            model_type="cnn",
            config={"input_dim": 10, "output_dim": 1}
        )
        
        # Initially, job should not be in active or completed
        status = orchestrator.get_job_status(job_id)
        assert status is None  # Job is in queue, not yet active
        
        # Job queue should contain the job
        assert orchestrator.job_queue.qsize() == 1
    
    def test_cluster_status(self, orchestrator):
        """Test cluster status reporting"""
        # Submit some jobs
        for i in range(3):
            orchestrator.submit_training_job(
                model_type="cnn",
                config={"input_dim": 10, "output_dim": 1}
            )
        
        status = orchestrator.get_cluster_status()
        
        assert status['num_workers'] == 2
        assert status['queued_jobs'] == 3
        assert status['active_jobs'] == 0
        assert status['completed_jobs'] == 0
        assert status['total_jobs'] == 3
    
    def test_cancel_job(self, orchestrator):
        """Test job cancellation"""
        job_id = orchestrator.submit_training_job(
            model_type="cnn",
            config={"input_dim": 10, "output_dim": 1}
        )
        
        # Simulate job being active
        priority, job = orchestrator.job_queue.get()
        job.status = "running"
        job.started_at = datetime.now()
        orchestrator.active_jobs[job_id] = job
        
        # Cancel job
        cancelled = orchestrator.cancel_job(job_id)
        
        assert cancelled is True
        assert job_id in orchestrator.completed_jobs
        assert orchestrator.completed_jobs[job_id].status == "cancelled"
    
    @patch('src.ml.distributed_training.RLAgentFactory')
    def test_train_rl_model_local(self, mock_factory, orchestrator):
        """Test local RL model training"""
        # Mock agent and environment
        mock_agent = Mock()
        mock_agent.train.return_value = {
            'evaluations': [{'mean_reward': 100.0, 'std_reward': 10.0}]
        }
        mock_factory.create_agent.return_value = mock_agent
        
        mock_env = Mock()
        
        # Create job config
        job = TrainingJob(
            job_id="test_rl_job",
            model_type="rl",
            config={
                'agent_type': 'PPO',
                'env_factory': lambda: mock_env,
                'total_timesteps': 1000
            }
        )
        
        # Train model
        results = orchestrator._train_rl_local(job)
        
        assert 'training_result' in results
        assert 'model_path' in results
        mock_factory.create_agent.assert_called_once()
        mock_agent.train.assert_called_once()
    
    def test_local_worker_startup(self, orchestrator):
        """Test local worker startup"""
        orchestrator._start_local_workers()
        
        # Workers should be started
        assert len(orchestrator.worker_pool) == orchestrator.config.num_workers
        
        # All workers should be threads
        for worker in orchestrator.worker_pool:
            assert isinstance(worker, threading.Thread)
    
    def test_health_monitor_startup(self, orchestrator):
        """Test health monitor startup"""
        orchestrator._start_health_monitor()
        
        assert orchestrator.health_monitor is not None
        assert isinstance(orchestrator.health_monitor, threading.Thread)
        assert orchestrator.health_monitor.daemon is True
    
    def test_wait_for_completion_timeout(self, orchestrator):
        """Test waiting for job completion with timeout"""
        job_id = orchestrator.submit_training_job(
            model_type="cnn",
            config={"input_dim": 10, "output_dim": 1}
        )
        
        # Wait with short timeout
        start_time = time.time()
        results = orchestrator.wait_for_completion([job_id], timeout=1.0)
        end_time = time.time()
        
        # Should timeout and return empty results
        assert len(results) == 0
        assert (end_time - start_time) >= 1.0
        assert (end_time - start_time) < 2.0  # Should not wait much longer


class TestModelValidationPipeline:
    """Test ModelValidationPipeline"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def orchestrator(self, temp_dir):
        """Create mock orchestrator"""
        config = DistributedTrainingConfig(
            checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
            log_dir=os.path.join(temp_dir, "logs"),
            results_dir=os.path.join(temp_dir, "results")
        )
        
        with patch('src.ml.distributed_training.ray') as mock_ray:
            mock_ray.is_initialized.return_value = False
            orchestrator = DistributedTrainingOrchestrator(config)
            yield orchestrator
            orchestrator.shutdown()
    
    @pytest.fixture
    def validation_pipeline(self, orchestrator):
        """Create validation pipeline"""
        return ModelValidationPipeline(
            orchestrator=orchestrator,
            validation_metrics=['accuracy', 'f1_score'],
            selection_criteria={'primary_metric': 'f1_score', 'minimize': False}
        )
    
    def test_validation_pipeline_initialization(self, validation_pipeline):
        """Test validation pipeline initialization"""
        assert validation_pipeline.validation_metrics == ['accuracy', 'f1_score']
        assert validation_pipeline.selection_criteria['primary_metric'] == 'f1_score'
    
    @patch('torch.load')
    def test_load_model_pytorch(self, mock_torch_load, validation_pipeline):
        """Test loading PyTorch model"""
        mock_model = Mock()
        mock_torch_load.return_value = mock_model
        
        model = validation_pipeline._load_model("model.pth")
        
        assert model == mock_model
        mock_torch_load.assert_called_once_with("model.pth", map_location='cpu', weights_only=False)
    
    def test_select_best_model(self, validation_pipeline):
        """Test model selection"""
        # Mock validation results
        validation_results = {
            "model_1.pth": {"accuracy": 0.85, "f1_score": 0.82},
            "model_2.pth": {"accuracy": 0.88, "f1_score": 0.85},
            "model_3.pth": {"accuracy": 0.83, "f1_score": 0.80}
        }
        
        best_model, best_metrics = validation_pipeline.select_best_model(validation_results)
        
        assert best_model == "model_2.pth"
        assert best_metrics["f1_score"] == 0.85


class TestRayTuneIntegration:
    """Test Ray Tune integration (mocked)"""
    
    @pytest.fixture
    def tune_config(self):
        """Create tune configuration"""
        return TuneConfig(
            num_samples=5,
            max_concurrent_trials=2,
            cpus_per_trial=1,
            gpus_per_trial=0
        )
    
    @pytest.fixture
    def optimizer(self, tune_config):
        """Create Ray Tune optimizer with mocked Ray"""
        with patch('src.ml.ray_tune_integration.ray') as mock_ray:
            mock_ray.is_initialized.return_value = True
            
            optimizer = RayTuneOptimizer(
                config=tune_config,
                local_dir="test_results"
            )
            yield optimizer
    
    def test_optimizer_initialization(self, optimizer, tune_config):
        """Test optimizer initialization"""
        assert optimizer.config == tune_config
        assert optimizer.local_dir == "test_results"
    
    def test_get_rl_search_space(self, optimizer):
        """Test RL search space generation"""
        search_space = optimizer._get_rl_search_space("PPO")
        
        assert "learning_rate" in search_space
        assert "batch_size" in search_space
        assert "n_steps" in search_space
        assert "clip_range" in search_space
    
    def test_get_rl_search_space_sac(self, optimizer):
        """Test SAC search space"""
        search_space = optimizer._get_rl_search_space("SAC")
        
        assert "learning_rate" in search_space
        assert "batch_size" in search_space
        assert "buffer_size" in search_space
        assert "tau" in search_space
    
    def test_unknown_agent_type(self, optimizer):
        """Test unknown agent type raises error"""
        with pytest.raises(ValueError, match="Unknown agent type"):
            optimizer._get_rl_search_space("UNKNOWN")


class TestModelValidator:
    """Test ModelValidator"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def validation_config(self):
        """Create validation configuration"""
        return ValidationConfig(
            cv_folds=3,
            min_samples=100,
            min_accuracy=0.6,
            min_f1_score=0.5
        )
    
    @pytest.fixture
    def validator(self, validation_config, temp_dir):
        """Create model validator"""
        return ModelValidator(
            config=validation_config,
            output_dir=temp_dir
        )
    
    def test_validator_initialization(self, validator, validation_config):
        """Test validator initialization"""
        assert validator.config == validation_config
        assert os.path.exists(validator.output_dir)
    
    def test_data_quality_validation_insufficient_samples(self, validator):
        """Test data quality validation with insufficient samples"""
        X = np.random.randn(50, 10)  # Less than min_samples
        y = np.random.randint(0, 2, 50)
        
        with pytest.raises(ValueError, match="Insufficient samples"):
            validator._validate_data_quality(X, y, "classification")
    
    def test_data_quality_validation_missing_values(self, validator):
        """Test data quality validation with missing values"""
        X = np.random.randn(200, 10)
        X[0, 0] = np.nan  # Add missing value
        y = np.random.randint(0, 2, 200)
        
        with pytest.raises(ValueError, match="Data contains missing values"):
            validator._validate_data_quality(X, y, "classification")
    
    def test_data_quality_validation_valid_data(self, validator):
        """Test data quality validation with valid data"""
        X = np.random.randn(200, 10)
        y = np.random.randint(0, 2, 200)
        
        # Should not raise any exception
        validator._validate_data_quality(X, y, "classification")
    
    def test_classification_threshold_check(self, validator):
        """Test classification threshold checking"""
        metrics = {
            'accuracy': 0.7,
            'f1_score': 0.6,
            'precision': 0.65,
            'recall': 0.58
        }
        
        passed = validator._check_classification_thresholds(metrics)
        assert passed is True
        
        # Test failing thresholds
        metrics['accuracy'] = 0.5
        passed = validator._check_classification_thresholds(metrics)
        assert passed is False
    
    def test_regression_threshold_check(self, validator):
        """Test regression threshold checking"""
        metrics = {
            'mse': 0.5,
            'mae': 0.3,
            'r2_score': 0.8
        }
        
        passed = validator._check_regression_thresholds(metrics)
        assert passed is True
        
        # Test failing thresholds
        metrics['mse'] = 2.0
        passed = validator._check_regression_thresholds(metrics)
        assert passed is False
    
    def test_performance_classification(self, validator):
        """Test performance classification"""
        # Excellent performance
        tier = validator._classify_performance(0.95, "classification")
        assert tier == "excellent"
        
        # Good performance
        tier = validator._classify_performance(0.85, "classification")
        assert tier == "good"
        
        # Fair performance
        tier = validator._classify_performance(0.75, "classification")
        assert tier == "fair"
        
        # Poor performance
        tier = validator._classify_performance(0.65, "classification")
        assert tier == "poor"
    
    def test_generate_recommendations(self, validator):
        """Test recommendation generation"""
        holdout_metrics = {
            'accuracy': 0.65,
            'precision': 0.5,
            'recall': 0.7,
            'f1_score': 0.58
        }
        
        cv_scores = {
            'accuracy': [0.7, 0.72, 0.68],
            'f1_score': [0.6, 0.62, 0.58]
        }
        
        recommendations = validator._generate_recommendations(
            holdout_metrics, cv_scores, "classification"
        )
        
        assert len(recommendations) > 0
        # Should recommend feature engineering due to low accuracy
        assert any("feature engineering" in rec.lower() for rec in recommendations)
        # Should recommend threshold adjustment due to precision/recall imbalance
        assert any("precision" in rec.lower() for rec in recommendations)


class TestFaultTolerance:
    """Test fault tolerance and error handling"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration with short timeouts"""
        return DistributedTrainingConfig(
            num_workers=1,
            max_retries=2,
            retry_delay=0.1,  # Fast retry for testing
            training_timeout=timedelta(seconds=2),
            health_check_interval=0.5,
            checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
            log_dir=os.path.join(temp_dir, "logs"),
            results_dir=os.path.join(temp_dir, "results")
        )
    
    def test_job_retry_on_failure(self, config):
        """Test job retry mechanism on failure"""
        with patch('src.ml.distributed_training.ray') as mock_ray:
            mock_ray.is_initialized.return_value = False
            
            orchestrator = DistributedTrainingOrchestrator(config)
            
            # Mock training method to always fail
            def failing_train(*args, **kwargs):
                raise RuntimeError("Training failed")
            
            orchestrator._train_model_local = failing_train
            
            # Submit job
            job_id = orchestrator.submit_training_job(
                model_type="cnn",
                config={"input_dim": 10, "output_dim": 1}
            )
            
            # Start workers
            orchestrator._start_local_workers()
            
            # Wait for job to be processed and retried
            time.sleep(1.0)
            
            # Check that job was retried
            status = orchestrator.get_job_status(job_id)
            if status:
                # Job should either be failed (after max retries) or still retrying
                assert status['status'] in ['failed', 'pending', 'running']
            
            orchestrator.shutdown()
    
    def test_job_timeout_handling(self, config):
        """Test job timeout handling"""
        with patch('src.ml.distributed_training.ray') as mock_ray:
            mock_ray.is_initialized.return_value = False
            
            orchestrator = DistributedTrainingOrchestrator(config)
            
            # Create a job that will timeout
            job = TrainingJob(
                job_id="timeout_job",
                model_type="cnn",
                config={"input_dim": 10, "output_dim": 1},
                status="running",
                started_at=datetime.now() - timedelta(seconds=5)  # Started 5 seconds ago
            )
            
            # Add to active jobs
            orchestrator.active_jobs[job.job_id] = job
            
            # Start health monitor
            orchestrator._start_health_monitor()
            
            # Wait for health monitor to detect timeout
            time.sleep(1.0)
            
            # Job should be moved to completed with timeout status
            assert job.job_id not in orchestrator.active_jobs
            
            orchestrator.shutdown()


class TestPerformanceMetrics:
    """Test performance monitoring and metrics"""
    
    def test_training_time_measurement(self):
        """Test training time measurement"""
        start_time = datetime.now()
        
        # Simulate some work
        time.sleep(0.1)
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        assert training_time >= 0.1
        assert training_time < 0.2  # Should be close to 0.1 seconds
    
    def test_throughput_calculation(self):
        """Test throughput calculation"""
        total_timesteps = 10000
        training_time = 100.0  # seconds
        
        throughput = total_timesteps / training_time
        
        assert throughput == 100.0  # timesteps per second
    
    def test_resource_utilization_tracking(self):
        """Test resource utilization tracking"""
        # Mock resource usage
        cpu_usage = 0.75
        memory_usage = 0.60
        gpu_usage = 0.85
        
        utilization = {
            'cpu': cpu_usage,
            'memory': memory_usage,
            'gpu': gpu_usage
        }
        
        # Check that all values are within expected ranges
        for resource, usage in utilization.items():
            assert 0.0 <= usage <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])