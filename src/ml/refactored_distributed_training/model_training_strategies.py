"""Model training strategies implementing the Strategy Pattern"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

from .exceptions import TrainingError
from .data_classes import TrainingJob, DistributedTrainingConfig
from ..training_pipeline import create_training_pipeline
from ..rl_agents import RLAgentFactory
from ..hybrid_model import CNNLSTMHybridModel, create_hybrid_model_config
from ..decision_auditor import DecisionAuditor, create_decision_auditor
from ..uncertainty_calibrator import UncertaintyCalibrator, create_uncertainty_calibrator


class ModelTrainingStrategy(ABC):
    """Abstract base class for model training strategies"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def train(
        self, 
        job: 'TrainingJob', 
        config: 'DistributedTrainingConfig'
    ) -> Dict[str, Any]:
        """Train a model using this strategy
        
        Args:
            job: Training job containing model type and configuration
            config: Distributed training configuration
            
        Returns:
            Dictionary containing training results
            
        Raises:
            TrainingError: If training fails
        """
        pass


class CNNTrainingStrategy(ModelTrainingStrategy):
    """CNN model training strategy"""

    def train(
        self, 
        job: 'TrainingJob', 
        config: 'DistributedTrainingConfig'
    ) -> Dict[str, Any]:
        """Train CNN model"""
        try:
            self.logger.info(f"Starting CNN training for job {job.job_id}")
            
            training_config = job.config
            pipeline = create_training_pipeline(
                input_dim=training_config["input_dim"],
                output_dim=training_config["output_dim"],
                checkpoint_dir=f"{config.checkpoint_dir}/{job.job_id}",
                log_dir=f"{config.log_dir}/{job.job_id}",
                **training_config.get("model_params", {}),
            )

            # Prepare data (assuming data is provided in config)
            train_loader, val_loader, test_loader = pipeline.prepare_data(
                features=training_config["features"],
                targets=training_config["targets"],
                **training_config.get("data_params", {}),
            )

            # Train model
            result = pipeline.train(
                train_loader=train_loader,
                val_loader=val_loader,
                **training_config.get("training_params", {}),
            )

            # Evaluate model
            test_metrics = pipeline.evaluate(test_loader)

            model_path = (
                f"{config.checkpoint_dir}/{job.job_id}/"
                "cnn_feature_extractor_best.pth"
            )

            return {
                "training_result": result.__dict__,
                "test_metrics": test_metrics,
                "model_path": model_path,
            }
        except Exception as e:
            self.logger.error(f"CNN training failed for job {job.job_id}: {e}")
            raise TrainingError(f"CNN training failed: {e}") from e


class HybridTrainingStrategy(ModelTrainingStrategy):
    """Hybrid CNN+LSTM model training strategy"""

    def train(
        self, 
        job: 'TrainingJob', 
        config: 'DistributedTrainingConfig'
    ) -> Dict[str, Any]:
        """Train hybrid CNN+LSTM model"""
        try:
            self.logger.info(f"Starting hybrid training for job {job.job_id}")
            
            training_config = job.config
            
            # 1. Create proper HybridModelConfig
            hybrid_config = create_hybrid_model_config(
                input_dim=training_config["input_dim"],
                **training_config.get("model_params", {}),
            )

            # 2. Initialize CNNLSTMHybridModel with config
            model = CNNLSTMHybridModel(hybrid_config)

            # 3. Prepare data loaders
            # This is a simplified data loading. In a real scenario, you'd use a proper dataset class.
            X_train = training_config["features"]
            y_class_train = training_config["targets_class"]
            y_reg_train = training_config["targets_reg"]
            
            X_val = training_config["val_features"]
            y_class_val = training_config["val_targets_class"]
            y_reg_val = training_config["val_targets_reg"]

            # 4. Train the model
            training_result = model.fit(
                X_train, y_class_train, y_reg_train,
                X_val, y_class_val, y_reg_val,
                **training_config.get("training_params", {}),
            )

            # 5. Evaluate the model
            # In a real scenario, you would have a separate test set.
            test_metrics = model.evaluate(X_val, y_class_val, y_reg_val)

            # 6. Register model version
            decision_auditor = create_decision_auditor()
            decision_auditor.register_model_version(
                model=model,
                training_data_hash=training_config.get("data_hash"),
                hyperparameters=training_config.get("model_params"),
                performance_metrics=test_metrics,
            )

            # 7. Calibrate uncertainty
            uncertainty_calibrator = create_uncertainty_calibrator(model)
            uncertainty_calibrator.calibrate_uncertainty_isotonic(
                X_val=X_val,
                y_class_val=y_class_val,
                y_reg_val=y_reg_val
            )

            model_path = (
                f"{config.checkpoint_dir}/{job.job_id}/"
                "hybrid_model_best.pth"
            )
            model.save_model(model_path)

            return {
                "training_result": training_result.__dict__,
                "test_metrics": test_metrics,
                "model_path": model_path,
            }
        except Exception as e:
            self.logger.error(f"Hybrid training failed for job {job.job_id}: {e}")
            raise TrainingError(f"Hybrid training failed: {e}") from e


class RLTrainingStrategy(ModelTrainingStrategy):
    """Reinforcement Learning model training strategy"""

    def train(
        self, 
        job: 'TrainingJob', 
        config: 'DistributedTrainingConfig'
    ) -> Dict[str, Any]:
        """Train RL model"""
        try:
            self.logger.info(f"Starting RL training for job {job.job_id}")
            
            training_config = job.config

            # Create environment and agent
            env = training_config["env_factory"]()
            agent = RLAgentFactory.create_agent(
                agent_type=training_config["agent_type"],
                env=env,
                **training_config.get("agent_params", {}),
            )

            # Train agent
            results = agent.train(
                env=env,
                total_timesteps=training_config.get("total_timesteps", 100000),
                **training_config.get("training_params", {}),
            )

            model_path = (
                f"{config.checkpoint_dir}/{job.job_id}/"
                "rl_model_best.zip"
            )

            return {
                "training_result": results,
                "model_path": model_path,
            }
        except Exception as e:
            self.logger.error(f"RL training failed for job {job.job_id}: {e}")
            raise TrainingError(f"RL training failed: {e}") from e


class TrainingStrategyFactory:
    """Factory for creating model training strategies"""

    _strategies = {
        "cnn": CNNTrainingStrategy(),
        "hybrid": HybridTrainingStrategy(),
        "rl": RLTrainingStrategy(),
    }

    @classmethod
    def get_strategy(cls, model_type: str) -> ModelTrainingStrategy:
        """Get training strategy for model type
        
        Args:
            model_type: Type of model to train
            
        Returns:
            Appropriate training strategy
            
        Raises:
            ValueError: If model type is not supported
        """
        if model_type not in cls._strategies:
            raise ValueError(f"Unknown model type: {model_type}")
        return cls._strategies[model_type]

    @classmethod
    def register_strategy(
        cls, 
        model_type: str, 
        strategy: ModelTrainingStrategy
    ) -> None:
        """Register a new training strategy
        
        Args:
            model_type: Type of model
            strategy: Training strategy implementation
        """
        cls._strategies[model_type] = strategy
