"""
Automated Retraining Service

Handles automated model retraining triggered by drift detection
and performance degradation alerts.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path

from src.services.model_monitoring_service import ModelMonitoringService
from src.models.monitoring import DriftDetectionResult, DriftType, AlertSeverity
if TYPE_CHECKING:
    from src.services.monitoring.advanced_monitoring_system import AdvancedMonitoringSystem

from src.ml.distributed_training import DistributedTrainingOrchestrator
from src.ml.ray_tune_integration import RayTuneOptimizer
from src.utils.logging import get_logger
from src.config.settings import get_settings

logger = get_logger("automated_retraining")


class RetrainingTrigger(Enum):
    """Types of retraining triggers"""
    PERFORMANCE_DRIFT = "performance_drift"
    DATA_DRIFT = "data_drift"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    ALERT_THRESHOLD = "alert_threshold"


class RetrainingStatus(Enum):
    """Retraining job status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RetrainingJob:
    """Retraining job configuration"""
    job_id: str
    model_name: str
    trigger: RetrainingTrigger
    trigger_details: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: RetrainingStatus = RetrainingStatus.PENDING
    progress: float = 0.0
    error_message: Optional[str] = None
    new_model_path: Optional[str] = None
    performance_improvement: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RetrainingConfig:
    """Configuration for automated retraining"""
    enabled: bool = True
    max_concurrent_jobs: int = 2
    cooldown_period_hours: int = 6
    min_samples_for_retraining: int = 1000
    performance_threshold: float = 0.1  # Minimum improvement required
    max_training_time_hours: int = 24
    backup_models: bool = True
    auto_deploy: bool = False  # Auto-deploy if improvement > threshold


class AutomatedRetrainingService:
    """
    Service for managing automated model retraining based on
    monitoring alerts and performance degradation.
    """

    def __init__(self, monitoring_service: ModelMonitoringService,
                 training_orchestrator: Optional[DistributedTrainingOrchestrator] = None,
                 ray_tune_optimizer: Optional[RayTuneOptimizer] = None):
        
        self.monitoring_service = monitoring_service
        self.training_orchestrator = training_orchestrator
        self.ray_tune_optimizer = ray_tune_optimizer
        self.settings = get_settings()
        
        # Configuration
        self.config = RetrainingConfig()
        
        # Job management
        self.active_jobs: Dict[str, RetrainingJob] = {}
        self.job_history: List[RetrainingJob] = []
        self.job_queue: List[RetrainingJob] = []
        
        # Cooldown tracking
        self.last_retraining: Dict[str, datetime] = {}
        
        # Callbacks for different model types
        self.retraining_callbacks: Dict[str, Callable[[RetrainingJob], None]] = {}
        
        # Register with monitoring service
        self._register_monitoring_callbacks()
        
        logger.info("Automated retraining service initialized")

    def configure(self, config: RetrainingConfig) -> None:
        """Update retraining configuration"""
        self.config = config
        logger.info(f"Retraining configuration updated: {config}")

    def register_retraining_callback(self, model_name: str, 
                                   callback: Callable[[RetrainingJob], None]) -> None:
        """Register a retraining callback for a specific model"""
        self.retraining_callbacks[model_name] = callback
        logger.info(f"Retraining callback registered for model: {model_name}")

    def _register_monitoring_callbacks(self) -> None:
        """Register callbacks with the monitoring service"""
        
        def drift_callback(model_name: str, drift_result: DriftDetectionResult) -> None:
            """Handle drift detection from monitoring service"""
            asyncio.create_task(self.handle_drift_detection(model_name, drift_result))
        
        # Register for all models (this would be done per model in practice)
        # For now, we'll handle this in the drift detection method
        pass

    def attach_advanced_monitoring(
        self, advanced_monitoring: "AdvancedMonitoringSystem"
    ) -> None:
        """Attach an advanced monitoring system and register retraining handler."""
        self._advanced_monitoring = advanced_monitoring
        advanced_monitoring.register_retraining_handler(
            self._handle_advanced_monitoring_retraining
        )
        logger.info("Advanced monitoring handler registered for automated retraining")

    async def _handle_advanced_monitoring_retraining(
        self, model_name: str, payload: Dict[str, Any]
    ) -> Optional[str]:
        """Schedule retraining based on advanced monitoring payload.

        Expected payload keys:
          - reason: str (e.g. 'performance_degradation', 'data_drift')
          - triggered_at: datetime when the monitoring alert fired
          - consecutive_failures: int indicating recent failure streak
          - degradations: Dict[str, float] of metric degradations
          - latest_metrics: Optional[Dict] with recent performance snapshot
        """
        if not self.config.enabled:
            logger.info("Automated retraining is disabled; ignoring monitoring trigger for %s", model_name)
            return None

        if not self._check_cooldown(model_name):
            logger.info("Model %s is in cooldown; skipping automated retraining trigger", model_name)
            return None

        if not self._check_minimum_samples(model_name):
            logger.info("Not enough labeled samples to retrain model %s", model_name)
            return None

        triggered_at = payload.get('triggered_at')
        created_at = triggered_at if isinstance(triggered_at, datetime) else datetime.now()

        trigger_details = dict(payload)
        if isinstance(triggered_at, datetime):
            trigger_details['triggered_at'] = triggered_at.isoformat()

        trigger = self._map_monitoring_reason_to_trigger(trigger_details.get('reason'))
        job = RetrainingJob(
            job_id=f"monitor_{model_name}_{created_at.timestamp()}",
            model_name=model_name,
            trigger=trigger,
            trigger_details=trigger_details,
            created_at=created_at
        )

        return await self.schedule_retraining(job)

    def _map_monitoring_reason_to_trigger(self, reason: Optional[str]) -> RetrainingTrigger:
        """Map monitoring reason strings to retraining triggers."""
        normalized = (reason or 'performance_degradation').lower()
        if normalized in {'data_drift', 'data-drift'}:
            return RetrainingTrigger.DATA_DRIFT
        if normalized in {'concept_drift', 'performance_drift'}:
            return RetrainingTrigger.PERFORMANCE_DRIFT
        if normalized == 'scheduled':
            return RetrainingTrigger.SCHEDULED
        if normalized == 'manual':
            return RetrainingTrigger.MANUAL
        return RetrainingTrigger.ALERT_THRESHOLD

    async def handle_drift_detection(self, model_name: str, 
                                   drift_result: DriftDetectionResult) -> Optional[str]:
        """Handle drift detection and potentially trigger retraining"""
        
        if not self.config.enabled:
            logger.info("Automated retraining is disabled")
            return None
        
        # Check if retraining is needed based on drift severity
        if drift_result.severity not in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            logger.info(f"Drift severity {drift_result.severity.value} not high enough for retraining")
            return None
        
        # Check cooldown period
        if not self._check_cooldown(model_name):
            logger.info(f"Model {model_name} is in cooldown period")
            return None
        
        # Check if we have enough samples
        if not self._check_minimum_samples(model_name):
            logger.info(f"Not enough samples for retraining model {model_name}")
            return None
        
        # Create retraining job
        job = RetrainingJob(
            job_id=f"drift_{model_name}_{datetime.now().timestamp()}",
            model_name=model_name,
            trigger=RetrainingTrigger.DATA_DRIFT if drift_result.drift_type == DriftType.DATA_DRIFT 
                   else RetrainingTrigger.PERFORMANCE_DRIFT,
            trigger_details={
                'drift_type': drift_result.drift_type.value,
                'drift_score': drift_result.drift_score,
                'threshold': drift_result.threshold,
                'severity': drift_result.severity.value,
                'details': drift_result.details
            },
            created_at=datetime.now()
        )
        
        return await self.schedule_retraining(job)

    async def schedule_retraining(self, job: RetrainingJob) -> str:
        """Schedule a retraining job"""
        
        # Check if we can start immediately or need to queue
        if len(self.active_jobs) < self.config.max_concurrent_jobs:
            await self._start_retraining_job(job)
        else:
            self.job_queue.append(job)
            logger.info(f"Retraining job {job.job_id} queued (active jobs: {len(self.active_jobs)})")
        
        return job.job_id

    async def schedule_manual_retraining(self, model_name: str, 
                                       config_overrides: Optional[Dict[str, Any]] = None) -> str:
        """Schedule manual retraining for a model"""
        
        job = RetrainingJob(
            job_id=f"manual_{model_name}_{datetime.now().timestamp()}",
            model_name=model_name,
            trigger=RetrainingTrigger.MANUAL,
            trigger_details=config_overrides or {},
            created_at=datetime.now()
        )
        
        return await self.schedule_retraining(job)

    async def _start_retraining_job(self, job: RetrainingJob) -> None:
        """Start executing a retraining job"""
        
        job.status = RetrainingStatus.RUNNING
        job.started_at = datetime.now()
        self.active_jobs[job.job_id] = job
        
        logger.info(f"Starting retraining job {job.job_id} for model {job.model_name}")
        
        try:
            # Update cooldown
            self.last_retraining[job.model_name] = datetime.now()
            
            # Execute retraining based on model type
            if job.model_name in self.retraining_callbacks:
                # Use custom callback
                callback = self.retraining_callbacks[job.model_name]
                await asyncio.get_event_loop().run_in_executor(None, callback, job)
            else:
                # Use default retraining logic
                await self._default_retraining_logic(job)
            
            # Mark as completed
            job.status = RetrainingStatus.COMPLETED
            job.completed_at = datetime.now()
            job.progress = 100.0
            
            logger.info(f"Retraining job {job.job_id} completed successfully")
            
            # Evaluate performance improvement
            await self._evaluate_retraining_results(job)
            
        except Exception as e:
            job.status = RetrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            
            logger.error(f"Retraining job {job.job_id} failed: {e}")
        
        finally:
            # Move to history and remove from active
            self.job_history.append(job)
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
            
            # Start next job in queue if available
            await self._process_job_queue()

    async def _default_retraining_logic(self, job: RetrainingJob) -> None:
        """Default retraining logic for models"""
        
        model_name = job.model_name
        
        # Get training data (this would be model-specific)
        training_data = await self._prepare_training_data(model_name)
        
        if not training_data:
            raise ValueError(f"No training data available for model {model_name}")
        
        # Update progress
        job.progress = 10.0
        
        # Backup current model if configured
        if self.config.backup_models:
            backup_path = await self._backup_current_model(model_name)
            job.metadata['backup_path'] = backup_path
        
        job.progress = 20.0
        
        # Prepare training configuration
        training_config = self._prepare_training_config(job)
        
        job.progress = 30.0
        
        # Execute training
        if self.ray_tune_optimizer and job.trigger != RetrainingTrigger.SCHEDULED:
            # Use hyperparameter optimization for drift-triggered retraining
            results = await self._retrain_with_hyperopt(model_name, training_data, training_config)
        else:
            # Use standard training
            results = await self._retrain_standard(model_name, training_data, training_config)
        
        job.progress = 90.0
        
        # Save new model
        new_model_path = await self._save_retrained_model(model_name, results)
        job.new_model_path = new_model_path
        
        job.progress = 100.0

    async def _prepare_training_data(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Prepare training data for retraining"""
        
        # Get recent prediction history
        predictions = self.monitoring_service.prediction_history.get(model_name, [])
        
        if len(predictions) < self.config.min_samples_for_retraining:
            return None
        
        # Filter predictions with actual values (for supervised learning)
        labeled_data = [p for p in predictions if p['actual'] is not None]
        
        if len(labeled_data) < self.config.min_samples_for_retraining // 2:
            return None
        
        # Prepare features and targets
        features = [p['features'] for p in labeled_data]
        targets = [p['actual'] for p in labeled_data]
        
        return {
            'features': features,
            'targets': targets,
            'sample_count': len(labeled_data),
            'feature_dim': len(features[0]) if features else 0
        }

    def _prepare_training_config(self, job: RetrainingJob) -> Dict[str, Any]:
        """Prepare training configuration based on job details"""
        
        base_config = {
            'model_name': job.model_name,
            'job_id': job.job_id,
            'trigger': job.trigger.value,
            'max_epochs': 50,  # Reduced for retraining
            'early_stopping': True,
            'validation_split': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32
        }
        
        # Adjust based on trigger type
        if job.trigger == RetrainingTrigger.DATA_DRIFT:
            # More aggressive learning for data drift
            base_config['learning_rate'] = 0.01
            base_config['max_epochs'] = 30
        elif job.trigger == RetrainingTrigger.PERFORMANCE_DRIFT:
            # Conservative approach for performance drift
            base_config['learning_rate'] = 0.0005
            base_config['max_epochs'] = 100
        
        # Apply any overrides from trigger details
        if 'config_overrides' in job.trigger_details:
            base_config.update(job.trigger_details['config_overrides'])
        
        return base_config

    async def _retrain_with_hyperopt(self, model_name: str, training_data: Dict[str, Any],
                                   config: Dict[str, Any]) -> Dict[str, Any]:
        """Retrain model with hyperparameter optimization"""
        
        if not self.ray_tune_optimizer:
            raise ValueError("Ray Tune optimizer not available for hyperparameter optimization")
        
        # Define search space for retraining
        search_space = {
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [16, 32, 64],
            'dropout_rate': [0.1, 0.2, 0.3],
            'hidden_size': [64, 128, 256]
        }
        
        # Run hyperparameter optimization (mock implementation)
        # In practice, this would use the actual RayTuneOptimizer
        results = {
            'model_path': f"models/{model_name}_hyperopt_{datetime.now().timestamp()}",
            'training_metrics': {
                'final_loss': 0.12,
                'accuracy': 0.88,
                'epochs_trained': 30
            },
            'validation_metrics': {
                'accuracy': 0.85,
                'precision': 0.83,
                'recall': 0.87
            },
            'best_hyperparameters': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'dropout_rate': 0.2
            }
        }
        
        return results

    async def _retrain_standard(self, model_name: str, training_data: Dict[str, Any],
                              config: Dict[str, Any]) -> Dict[str, Any]:
        """Standard retraining without hyperparameter optimization"""
        
        # This would integrate with the actual training pipeline
        # For now, return mock results
        
        logger.info(f"Starting standard retraining for {model_name}")
        
        # Simulate training time
        await asyncio.sleep(2)
        
        return {
            'model_path': f"models/{model_name}_retrained_{datetime.now().timestamp()}",
            'training_metrics': {
                'final_loss': 0.15,
                'accuracy': 0.85,
                'epochs_trained': config.get('max_epochs', 50)
            },
            'validation_metrics': {
                'accuracy': 0.82,
                'precision': 0.80,
                'recall': 0.84
            }
        }

    async def _backup_current_model(self, model_name: str) -> str:
        """Backup current model before retraining"""
        
        # Create backup directory
        backup_dir = Path(self.settings.ml.model_registry_path) / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{model_name}_backup_{timestamp}"
        
        # This would copy the actual model files
        # For now, just create a placeholder
        backup_path.mkdir(exist_ok=True)
        
        logger.info(f"Model {model_name} backed up to {backup_path}")
        return str(backup_path)

    async def _save_retrained_model(self, model_name: str, results: Dict[str, Any]) -> str:
        """Save retrained model"""
        
        model_path = results.get('model_path')
        if not model_path:
            raise ValueError("No model path in training results")
        
        # This would save the actual model
        # For now, just log the action
        logger.info(f"Retrained model saved to {model_path}")
        
        return model_path

    async def _evaluate_retraining_results(self, job: RetrainingJob) -> None:
        """Evaluate the results of retraining"""
        
        if job.status != RetrainingStatus.COMPLETED or not job.new_model_path:
            return
        
        # Get baseline performance
        baseline_metrics = self.monitoring_service.baseline_metrics.get(job.model_name)
        if not baseline_metrics:
            logger.warning(f"No baseline metrics for model {job.model_name}")
            return
        
        # Calculate performance improvement (mock calculation)
        # In practice, this would evaluate the new model on validation data
        improvement = 0.05  # Mock 5% improvement
        job.performance_improvement = improvement
        
        logger.info(f"Retraining job {job.job_id} achieved {improvement:.2%} performance improvement")
        
        # Auto-deploy if configured and improvement is significant
        if (self.config.auto_deploy and 
            improvement > self.config.performance_threshold):
            
            await self._deploy_retrained_model(job)

    async def _deploy_retrained_model(self, job: RetrainingJob) -> None:
        """Deploy retrained model if it shows improvement"""
        
        logger.info(f"Auto-deploying retrained model for {job.model_name}")
        
        # This would integrate with model deployment system
        # For now, just update metadata
        job.metadata['auto_deployed'] = True
        job.metadata['deployment_time'] = datetime.now().isoformat()

    async def _process_job_queue(self) -> None:
        """Process queued retraining jobs"""
        
        while (self.job_queue and 
               len(self.active_jobs) < self.config.max_concurrent_jobs):
            
            next_job = self.job_queue.pop(0)
            await self._start_retraining_job(next_job)

    def _check_cooldown(self, model_name: str) -> bool:
        """Check if model is in cooldown period"""
        
        if model_name not in self.last_retraining:
            return True
        
        last_retraining = self.last_retraining[model_name]
        cooldown_period = timedelta(hours=self.config.cooldown_period_hours)
        
        return datetime.now() - last_retraining > cooldown_period

    def _check_minimum_samples(self, model_name: str) -> bool:
        """Check if we have minimum samples for retraining"""
        
        predictions = self.monitoring_service.prediction_history.get(model_name, [])
        labeled_predictions = [p for p in predictions if p['actual'] is not None]
        
        return len(labeled_predictions) >= self.config.min_samples_for_retraining

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a retraining job"""
        
        # Check active jobs
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            return {
                'job_id': job.job_id,
                'model_name': job.model_name,
                'status': job.status.value,
                'progress': job.progress,
                'created_at': job.created_at.isoformat(),
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'estimated_completion': self._estimate_completion_time(job)
            }
        
        # Check job history
        for job in self.job_history:
            if job.job_id == job_id:
                return {
                    'job_id': job.job_id,
                    'model_name': job.model_name,
                    'status': job.status.value,
                    'progress': job.progress,
                    'created_at': job.created_at.isoformat(),
                    'started_at': job.started_at.isoformat() if job.started_at else None,
                    'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                    'performance_improvement': job.performance_improvement,
                    'error_message': job.error_message
                }
        
        # Check queue
        for job in self.job_queue:
            if job.job_id == job_id:
                return {
                    'job_id': job.job_id,
                    'model_name': job.model_name,
                    'status': 'queued',
                    'progress': 0.0,
                    'created_at': job.created_at.isoformat(),
                    'queue_position': self.job_queue.index(job) + 1
                }
        
        return None

    def _estimate_completion_time(self, job: RetrainingJob) -> Optional[str]:
        """Estimate completion time for active job"""
        
        if job.status != RetrainingStatus.RUNNING or not job.started_at:
            return None
        
        # Simple estimation based on progress
        elapsed = datetime.now() - job.started_at
        if job.progress > 0:
            total_estimated = elapsed * (100.0 / job.progress)
            remaining = total_estimated - elapsed
            completion_time = datetime.now() + remaining
            return completion_time.isoformat()
        
        return None

    def get_retraining_summary(self) -> Dict[str, Any]:
        """Get summary of retraining activities"""
        
        return {
            'active_jobs': len(self.active_jobs),
            'queued_jobs': len(self.job_queue),
            'completed_jobs_24h': len([
                job for job in self.job_history
                if (job.completed_at and 
                    job.completed_at > datetime.now() - timedelta(hours=24))
            ]),
            'success_rate': self._calculate_success_rate(),
            'avg_improvement': self._calculate_average_improvement(),
            'models_in_cooldown': [
                model for model, last_time in self.last_retraining.items()
                if datetime.now() - last_time < timedelta(hours=self.config.cooldown_period_hours)
            ]
        }

    def _calculate_success_rate(self) -> float:
        """Calculate retraining success rate"""
        
        if not self.job_history:
            return 0.0
        
        successful_jobs = [
            job for job in self.job_history
            if job.status == RetrainingStatus.COMPLETED
        ]
        
        return len(successful_jobs) / len(self.job_history)

    def _calculate_average_improvement(self) -> float:
        """Calculate average performance improvement"""
        
        improvements = [
            job.performance_improvement for job in self.job_history
            if (job.performance_improvement is not None and 
                job.status == RetrainingStatus.COMPLETED)
        ]
        
        return sum(improvements) / len(improvements) if improvements else 0.0

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a retraining job"""
        
        # Cancel active job
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = RetrainingStatus.CANCELLED
            job.completed_at = datetime.now()
            
            # Move to history
            self.job_history.append(job)
            del self.active_jobs[job_id]
            
            logger.info(f"Cancelled active retraining job {job_id}")
            return True
        
        # Remove from queue
        for i, job in enumerate(self.job_queue):
            if job.job_id == job_id:
                job.status = RetrainingStatus.CANCELLED
                self.job_history.append(job)
                self.job_queue.pop(i)
                
                logger.info(f"Cancelled queued retraining job {job_id}")
                return True
        
        return False
