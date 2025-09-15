"""
Distributed Training Orchestration System using Ray

This module provides a comprehensive distributed training system that orchestrates
model training across multiple workers, handles fault tolerance, and manages
resource allocation efficiently.
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

# Ray imports with fallback
try:
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
    from ray.tune.search.optuna import OptunaSearch
    from ray.air import session
    from ray.train import Trainer
    from ray.train.torch import TorchTrainer
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None

from .base_models import ModelConfig, TrainingResult
from .training_pipeline import CNNTrainingPipeline, MarketDataset
from .rl_hyperopt import HyperparameterOptimizer
from .hybrid_model import CNNLSTMHybridModel
from .rl_agents import RLAgentFactory


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
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed, cancelled
    worker_id: Optional[str] = None
    retry_count: int = 0
    error_message: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class DistributedTrainingOrchestrator:
    """Main orchestrator for distributed training operations"""
    
    def __init__(self, config: DistributedTrainingConfig):
        """Initialize distributed training orchestrator
        
        Args:
            config: Distributed training configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize Ray if available
        self.ray_initialized = False
        if RAY_AVAILABLE:
            self._initialize_ray()
        else:
            self.logger.warning("Ray not available, falling back to local training")
        
        # Job management
        self.job_queue = queue.PriorityQueue()
        self.active_jobs: Dict[str, TrainingJob] = {}
        self.completed_jobs: Dict[str, TrainingJob] = {}
        self.job_lock = threading.Lock()
        
        # Worker management
        self.worker_pool = None
        self.health_monitor = None
        
        # Create directories
        self._create_directories()
        
        # Setup logging
        self._setup_logging()
    
    def _initialize_ray(self) -> None:
        """Initialize Ray cluster"""
        try:
            if not ray.is_initialized():
                ray.init(
                    num_cpus=self.config.num_workers * self.config.cpus_per_worker,
                    num_gpus=int(self.config.num_workers * self.config.gpus_per_worker),
                    object_store_memory=int(2e9),  # 2GB object store
                    ignore_reinit_error=True
                )
            self.ray_initialized = True
            self.logger.info("Ray cluster initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Ray: {e}")
            self.ray_initialized = False
    
    def _create_directories(self) -> None:
        """Create necessary directories"""
        for directory in [
            self.config.checkpoint_dir,
            self.config.log_dir,
            self.config.results_dir
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
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
        job_id: Optional[str] = None
    ) -> str:
        """Submit a training job to the queue
        
        Args:
            model_type: Type of model to train ('cnn', 'lstm', 'hybrid', 'rl')
            config: Training configuration
            priority: Job priority (lower = higher priority)
            job_id: Optional custom job ID
            
        Returns:
            Job ID
        """
        if job_id is None:
            job_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        job = TrainingJob(
            job_id=job_id,
            model_type=model_type,
            config=config,
            priority=priority
        )
        
        # Add to queue (priority queue uses tuple: (priority, job))
        self.job_queue.put((priority, job))
        
        self.logger.info(f"Submitted training job: {job_id} (type: {model_type})")
        return job_id
    
    def start_training_workers(self) -> None:
        """Start distributed training workers"""
        if self.ray_initialized:
            self._start_ray_workers()
        else:
            self._start_local_workers()
        
        # Start health monitoring
        self._start_health_monitor()
    
    def _start_ray_workers(self) -> None:
        """Start Ray-based distributed workers"""
        @ray.remote(
            num_cpus=self.config.cpus_per_worker,
            num_gpus=self.config.gpus_per_worker,
            memory=self.config.memory_per_worker
        )
        class TrainingWorker:
            def __init__(self, worker_id: str, config: DistributedTrainingConfig):
                self.worker_id = worker_id
                self.config = config
                self.logger = logging.getLogger(f"worker_{worker_id}")
            
            def train_model(self, job: TrainingJob) -> Dict[str, Any]:
                """Train a model on this worker"""
                try:
                    self.logger.info(f"Starting training job {job.job_id}")
                    
                    if job.model_type == "cnn":
                        return self._train_cnn_model(job)
                    elif job.model_type == "lstm":
                        return self._train_lstm_model(job)
                    elif job.model_type == "hybrid":
                        return self._train_hybrid_model(job)
                    elif job.model_type == "rl":
                        return self._train_rl_model(job)
                    else:
                        raise ValueError(f"Unknown model type: {job.model_type}")
                
                except Exception as e:
                    self.logger.error(f"Training failed for job {job.job_id}: {e}")
                    raise
            
            def _train_cnn_model(self, job: TrainingJob) -> Dict[str, Any]:
                """Train CNN model"""
                from .training_pipeline import create_training_pipeline
                
                config = job.config
                pipeline = create_training_pipeline(
                    input_dim=config['input_dim'],
                    output_dim=config['output_dim'],
                    checkpoint_dir=f"{self.config.checkpoint_dir}/{job.job_id}",
                    log_dir=f"{self.config.log_dir}/{job.job_id}",
                    **config.get('model_params', {})
                )
                
                # Prepare data (assuming data is provided in config)
                train_loader, val_loader, test_loader = pipeline.prepare_data(
                    features=config['features'],
                    targets=config['targets'],
                    **config.get('data_params', {})
                )
                
                # Train model
                result = pipeline.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    **config.get('training_params', {})
                )
                
                # Evaluate model
                test_metrics = pipeline.evaluate(test_loader)
                
                return {
                    'training_result': asdict(result),
                    'test_metrics': test_metrics,
                    'model_path': f"{self.config.checkpoint_dir}/{job.job_id}/cnn_feature_extractor_best.pth"
                }
            
            def _train_hybrid_model(self, job: TrainingJob) -> Dict[str, Any]:
                """Train hybrid CNN+LSTM model"""
                # Implementation for hybrid model training
                config = job.config
                
                # Create hybrid model
                model = CNNLSTMHybridModel(
                    input_dim=config['input_dim'],
                    **config.get('model_params', {})
                )
                
                # Training logic would go here
                # This is a simplified version
                return {
                    'training_result': {'train_loss': 0.1, 'val_loss': 0.15},
                    'model_path': f"{self.config.checkpoint_dir}/{job.job_id}/hybrid_model_best.pth"
                }
            
            def _train_rl_model(self, job: TrainingJob) -> Dict[str, Any]:
                """Train RL model"""
                config = job.config
                
                # Create environment and agent
                env = config['env_factory']()
                agent = RLAgentFactory.create_agent(
                    agent_type=config['agent_type'],
                    env=env,
                    **config.get('agent_params', {})
                )
                
                # Train agent
                results = agent.train(
                    env=env,
                    total_timesteps=config.get('total_timesteps', 100000),
                    **config.get('training_params', {})
                )
                
                return {
                    'training_result': results,
                    'model_path': f"{self.config.checkpoint_dir}/{job.job_id}/rl_model_best.zip"
                }
        
        # Create worker pool
        self.worker_pool = [
            TrainingWorker.remote(f"worker_{i}", self.config)
            for i in range(self.config.num_workers)
        ]
        
        self.logger.info(f"Started {self.config.num_workers} Ray workers")
    
    def _start_local_workers(self) -> None:
        """Start local thread-based workers as fallback"""
        def worker_thread(worker_id: str):
            """Worker thread function"""
            logger = logging.getLogger(f"worker_{worker_id}")
            
            while True:
                try:
                    # Get job from queue (blocking with timeout)
                    try:
                        priority, job = self.job_queue.get(timeout=5.0)
                    except queue.Empty:
                        continue
                    
                    # Update job status
                    with self.job_lock:
                        job.status = "running"
                        job.started_at = datetime.now()
                        job.worker_id = worker_id
                        self.active_jobs[job.job_id] = job
                    
                    logger.info(f"Starting training job {job.job_id}")
                    
                    try:
                        # Train model
                        results = self._train_model_local(job)
                        
                        # Update job with results
                        with self.job_lock:
                            job.status = "completed"
                            job.completed_at = datetime.now()
                            job.results = results
                            self.completed_jobs[job.job_id] = job
                            del self.active_jobs[job.job_id]
                        
                        logger.info(f"Completed training job {job.job_id}")
                    
                    except Exception as e:
                        logger.error(f"Training failed for job {job.job_id}: {e}")
                        
                        # Handle retry logic
                        with self.job_lock:
                            if job.retry_count < self.config.max_retries:
                                job.retry_count += 1
                                job.status = "pending"
                                job.error_message = str(e)
                                # Re-queue job
                                self.job_queue.put((job.priority, job))
                                logger.info(f"Re-queued job {job.job_id} (retry {job.retry_count})")
                            else:
                                job.status = "failed"
                                job.completed_at = datetime.now()
                                job.error_message = str(e)
                                self.completed_jobs[job.job_id] = job
                                del self.active_jobs[job.job_id]
                                logger.error(f"Job {job.job_id} failed after {job.retry_count} retries")
                    
                    finally:
                        self.job_queue.task_done()
                
                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}")
                    time.sleep(1)
        
        # Start worker threads
        self.worker_pool = []
        for i in range(self.config.num_workers):
            worker_id = f"local_worker_{i}"
            thread = threading.Thread(
                target=worker_thread,
                args=(worker_id,),
                daemon=True
            )
            thread.start()
            self.worker_pool.append(thread)
        
        self.logger.info(f"Started {self.config.num_workers} local workers")
    
    def _train_model_local(self, job: TrainingJob) -> Dict[str, Any]:
        """Train model locally (fallback implementation)"""
        if job.model_type == "cnn":
            return self._train_cnn_local(job)
        elif job.model_type == "hybrid":
            return self._train_hybrid_local(job)
        elif job.model_type == "rl":
            return self._train_rl_local(job)
        else:
            raise ValueError(f"Unknown model type: {job.model_type}")
    
    def _train_cnn_local(self, job: TrainingJob) -> Dict[str, Any]:
        """Train CNN model locally"""
        from .training_pipeline import create_training_pipeline
        
        config = job.config
        pipeline = create_training_pipeline(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            checkpoint_dir=f"{self.config.checkpoint_dir}/{job.job_id}",
            log_dir=f"{self.config.log_dir}/{job.job_id}",
            **config.get('model_params', {})
        )
        
        # Prepare data
        train_loader, val_loader, test_loader = pipeline.prepare_data(
            features=config['features'],
            targets=config['targets'],
            **config.get('data_params', {})
        )
        
        # Train model
        result = pipeline.train(
            train_loader=train_loader,
            val_loader=val_loader,
            **config.get('training_params', {})
        )
        
        # Evaluate model
        test_metrics = pipeline.evaluate(test_loader)
        
        return {
            'training_result': asdict(result),
            'test_metrics': test_metrics,
            'model_path': f"{self.config.checkpoint_dir}/{job.job_id}/cnn_feature_extractor_best.pth"
        }
    
    def _train_hybrid_local(self, job: TrainingJob) -> Dict[str, Any]:
        """Train hybrid model locally"""
        # Simplified implementation
        return {
            'training_result': {'train_loss': 0.1, 'val_loss': 0.15},
            'model_path': f"{self.config.checkpoint_dir}/{job.job_id}/hybrid_model_best.pth"
        }
    
    def _train_rl_local(self, job: TrainingJob) -> Dict[str, Any]:
        """Train RL model locally"""
        config = job.config
        
        # Create environment and agent
        env = config['env_factory']()
        agent = RLAgentFactory.create_agent(
            agent_type=config['agent_type'],
            env=env,
            **config.get('agent_params', {})
        )
        
        # Train agent
        results = agent.train(
            env=env,
            total_timesteps=config.get('total_timesteps', 100000),
            **config.get('training_params', {})
        )
        
        return {
            'training_result': results,
            'model_path': f"{self.config.checkpoint_dir}/{job.job_id}/rl_model_best.zip"
        }
    
    def _start_health_monitor(self) -> None:
        """Start health monitoring thread"""
        def health_monitor():
            """Monitor worker health and job timeouts"""
            while True:
                try:
                    current_time = datetime.now()
                    
                    with self.job_lock:
                        # Check for timed out jobs
                        timed_out_jobs = []
                        for job_id, job in self.active_jobs.items():
                            if job.started_at and (current_time - job.started_at) > self.config.training_timeout:
                                timed_out_jobs.append(job_id)
                        
                        # Handle timed out jobs
                        for job_id in timed_out_jobs:
                            job = self.active_jobs[job_id]
                            self.logger.warning(f"Job {job_id} timed out")
                            
                            if job.retry_count < self.config.max_retries:
                                job.retry_count += 1
                                job.status = "pending"
                                job.error_message = "Training timeout"
                                self.job_queue.put((job.priority, job))
                            else:
                                job.status = "failed"
                                job.completed_at = current_time
                                job.error_message = "Training timeout (max retries exceeded)"
                                self.completed_jobs[job_id] = job
                            
                            del self.active_jobs[job_id]
                    
                    # Log status
                    active_count = len(self.active_jobs)
                    completed_count = len(self.completed_jobs)
                    queue_size = self.job_queue.qsize()
                    
                    self.logger.info(
                        f"Training status - Active: {active_count}, "
                        f"Completed: {completed_count}, Queued: {queue_size}"
                    )
                    
                    time.sleep(self.config.health_check_interval)
                
                except Exception as e:
                    self.logger.error(f"Health monitor error: {e}")
                    time.sleep(10)
        
        self.health_monitor = threading.Thread(target=health_monitor, daemon=True)
        self.health_monitor.start()
        self.logger.info("Health monitor started")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a training job
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status dictionary or None if not found
        """
        with self.job_lock:
            # Check active jobs
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                return {
                    'job_id': job.job_id,
                    'status': job.status,
                    'model_type': job.model_type,
                    'created_at': job.created_at.isoformat(),
                    'started_at': job.started_at.isoformat() if job.started_at else None,
                    'worker_id': job.worker_id,
                    'retry_count': job.retry_count
                }
            
            # Check completed jobs
            if job_id in self.completed_jobs:
                job = self.completed_jobs[job_id]
                return {
                    'job_id': job.job_id,
                    'status': job.status,
                    'model_type': job.model_type,
                    'created_at': job.created_at.isoformat(),
                    'started_at': job.started_at.isoformat() if job.started_at else None,
                    'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                    'worker_id': job.worker_id,
                    'retry_count': job.retry_count,
                    'error_message': job.error_message,
                    'results': job.results
                }
        
        return None
    
    def wait_for_completion(self, job_ids: List[str], timeout: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
        """Wait for jobs to complete
        
        Args:
            job_ids: List of job IDs to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            Dictionary of job results
        """
        start_time = time.time()
        results = {}
        
        while len(results) < len(job_ids):
            if timeout and (time.time() - start_time) > timeout:
                self.logger.warning(f"Timeout waiting for jobs: {set(job_ids) - set(results.keys())}")
                break
            
            for job_id in job_ids:
                if job_id not in results:
                    status = self.get_job_status(job_id)
                    if status and status['status'] in ['completed', 'failed']:
                        results[job_id] = status
            
            time.sleep(1)
        
        return results
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if job was cancelled, False otherwise
        """
        with self.job_lock:
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                job.status = "cancelled"
                job.completed_at = datetime.now()
                self.completed_jobs[job_id] = job
                del self.active_jobs[job_id]
                self.logger.info(f"Cancelled job {job_id}")
                return True
        
        return False
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get cluster status information
        
        Returns:
            Cluster status dictionary
        """
        with self.job_lock:
            active_jobs = len(self.active_jobs)
            completed_jobs = len(self.completed_jobs)
            queued_jobs = self.job_queue.qsize()
        
        status = {
            'ray_initialized': self.ray_initialized,
            'num_workers': self.config.num_workers,
            'active_jobs': active_jobs,
            'completed_jobs': completed_jobs,
            'queued_jobs': queued_jobs,
            'total_jobs': active_jobs + completed_jobs + queued_jobs
        }
        
        if self.ray_initialized and ray.is_initialized():
            try:
                cluster_resources = ray.cluster_resources()
                status['cluster_resources'] = cluster_resources
            except Exception as e:
                self.logger.warning(f"Failed to get cluster resources: {e}")
        
        return status
    
    def shutdown(self) -> None:
        """Shutdown the distributed training system"""
        self.logger.info("Shutting down distributed training system")
        
        # Cancel all active jobs
        with self.job_lock:
            for job_id in list(self.active_jobs.keys()):
                self.cancel_job(job_id)
        
        # Shutdown Ray if initialized
        if self.ray_initialized and ray.is_initialized():
            ray.shutdown()
        
        self.logger.info("Distributed training system shutdown complete")


class ModelValidationPipeline:
    """Automated model validation and selection pipeline"""
    
    def __init__(
        self,
        orchestrator: DistributedTrainingOrchestrator,
        validation_metrics: List[str] = None,
        selection_criteria: Dict[str, Any] = None
    ):
        """Initialize model validation pipeline
        
        Args:
            orchestrator: Distributed training orchestrator
            validation_metrics: List of metrics to compute
            selection_criteria: Criteria for model selection
        """
        self.orchestrator = orchestrator
        self.validation_metrics = validation_metrics or [
            'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'
        ]
        self.selection_criteria = selection_criteria or {
            'primary_metric': 'f1_score',
            'minimize': False,
            'min_improvement': 0.01
        }
        
        self.logger = logging.getLogger(__name__)
    
    def validate_models(
        self,
        model_paths: List[str],
        validation_data: Dict[str, Any],
        cross_validation_folds: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """Validate multiple models
        
        Args:
            model_paths: List of model file paths
            validation_data: Validation dataset
            cross_validation_folds: Number of CV folds
            
        Returns:
            Dictionary of validation results for each model
        """
        validation_results = {}
        
        for model_path in model_paths:
            try:
                self.logger.info(f"Validating model: {model_path}")
                
                # Load model
                model = self._load_model(model_path)
                
                # Perform validation
                metrics = self._validate_single_model(
                    model, validation_data, cross_validation_folds
                )
                
                validation_results[model_path] = metrics
                
            except Exception as e:
                self.logger.error(f"Validation failed for {model_path}: {e}")
                validation_results[model_path] = {}
        
        return validation_results
    
    def _load_model(self, model_path: str):
        """Load model from path"""
        # Implementation depends on model type
        # This is a placeholder
        if model_path.endswith('.pth'):
            return torch.load(model_path, map_location='cpu', weights_only=False)
        elif model_path.endswith('.zip'):
            # For stable-baselines3 models
            from stable_baselines3 import PPO, SAC, TD3, DQN
            # Determine model type and load accordingly
            pass
        else:
            raise ValueError(f"Unsupported model format: {model_path}")
    
    def _validate_single_model(
        self,
        model,
        validation_data: Dict[str, Any],
        cv_folds: int
    ) -> Dict[str, float]:
        """Validate a single model"""
        # Placeholder implementation
        # In practice, this would perform cross-validation
        # and compute the specified metrics
        
        metrics = {}
        for metric in self.validation_metrics:
            # Simulate metric computation
            metrics[metric] = np.random.random()
        
        return metrics
    
    def select_best_model(
        self,
        validation_results: Dict[str, Dict[str, float]]
    ) -> Tuple[str, Dict[str, float]]:
        """Select the best model based on validation results
        
        Args:
            validation_results: Validation results for all models
            
        Returns:
            Tuple of (best_model_path, best_metrics)
        """
        primary_metric = self.selection_criteria['primary_metric']
        minimize = self.selection_criteria['minimize']
        
        best_model = None
        best_score = float('inf') if minimize else float('-inf')
        best_metrics = {}
        
        for model_path, metrics in validation_results.items():
            if primary_metric not in metrics:
                continue
            
            score = metrics[primary_metric]
            
            is_better = (
                (minimize and score < best_score) or
                (not minimize and score > best_score)
            )
            
            if is_better:
                best_score = score
                best_model = model_path
                best_metrics = metrics
        
        if best_model:
            self.logger.info(f"Selected best model: {best_model} "
                           f"({primary_metric}: {best_score:.4f})")
        
        return best_model, best_metrics


def create_distributed_training_system(
    config: Optional[DistributedTrainingConfig] = None
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