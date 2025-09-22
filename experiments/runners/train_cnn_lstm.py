#!/usr/bin/env python3
"""
CNN+LSTM Training Runner

This script implements the complete training pipeline for CNN+LSTM feature extractors
with support for different model types, hyperparameter optimization, and comprehensive
evaluation.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import yaml
from omegaconf import DictConfig, OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.cnn_lstm.trainer import CNNLSTMTrainer, TrainingConfig
from data.pipeline import create_data_loaders
from utils.gpu_utils import get_device_info, setup_gpu


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('experiments/logs/cnn_lstm_training.log')
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> DictConfig:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return OmegaConf.create(config_dict)


def create_training_config(config: DictConfig, model_type: str = "hybrid") -> TrainingConfig:
    """Create TrainingConfig from loaded configuration."""
    
    # Select model-specific configuration
    if model_type == "hybrid":
        model_config = config.model.hybrid_config
        training_config = config.training
    elif model_type == "cnn":
        model_config = config.cnn_training.model_config
        training_config = config.cnn_training.training
    elif model_type == "lstm":
        model_config = config.lstm_training.model_config
        training_config = config.lstm_training.training
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Convert to TrainingConfig
    training_cfg = TrainingConfig(
        model_type=model_type,
        model_config=OmegaConf.to_container(model_config, resolve=True),
        
        # Training parameters
        num_epochs=training_config.get("num_epochs", 200),
        batch_size=training_config.get("batch_size", 32),
        learning_rate=training_config.get("learning_rate", 1e-4),
        weight_decay=training_config.get("weight_decay", 1e-5),
        gradient_clip_val=training_config.get("gradient_clip_val", 1.0),
        
        # Mixed precision
        use_mixed_precision=training_config.get("use_mixed_precision", True),
        loss_scale=training_config.get("loss_scale", "dynamic"),
        
        # Optimization
        optimizer=training_config.get("optimizer", "adamw"),
        scheduler=training_config.get("scheduler", "cosine"),
        warmup_epochs=training_config.get("warmup_epochs", 10),
        
        # Early stopping
        early_stopping_patience=training_config.early_stopping.get("patience", 20),
        early_stopping_min_delta=training_config.early_stopping.get("min_delta", 1e-4),
        early_stopping_metric=training_config.early_stopping.get("metric", "val_loss"),
        
        # Checkpointing
        save_every_n_epochs=training_config.checkpointing.get("save_every_n_epochs", 10),
        save_best_only=training_config.checkpointing.get("save_best_only", True),
        checkpoint_dir=training_config.checkpointing.get("checkpoint_dir", "models/checkpoints"),
        
        # Validation
        validation_split=training_config.get("validation_split", 0.2),
        validation_frequency=training_config.get("validation_frequency", 1),
        
        # Logging
        log_every_n_steps=training_config.logging.get("log_every_n_steps", 100),
        track_feature_quality=training_config.logging.get("track_feature_quality", True),
        use_wandb=training_config.logging.get("use_wandb", False),
        experiment_name=training_config.logging.get("experiment_name", "cnn_lstm_training"),
        
        # Multi-task learning
        tasks=training_config.get("tasks", ["price_prediction"]),
        task_weights=training_config.get("task_weights", {"price_prediction": 1.0}),
        
        # Data augmentation
        use_data_augmentation=training_config.data_augmentation.get("enabled", True),
        noise_std=training_config.data_augmentation.get("noise_std", 0.01),
        temporal_jitter_prob=training_config.data_augmentation.get("temporal_jitter_prob", 0.1),
        price_scaling_range=tuple(training_config.data_augmentation.get("price_scaling_range", [0.95, 1.05])),
        
        # Device
        device=training_config.get("device", "auto"),
        num_workers=training_config.get("num_workers", 4),
        pin_memory=training_config.get("pin_memory", True)
    )
    
    return training_cfg


def train_cnn_models(config: DictConfig, logger: logging.Logger) -> Dict[str, Any]:
    """Train CNN models for multi-timeframe price pattern recognition."""
    logger.info("Starting CNN model training for multi-timeframe price pattern recognition")
    
    # Create CNN-specific training configuration
    training_config = create_training_config(config, model_type="cnn")
    
    # Override epochs for CNN training (50+ epochs as per requirements)
    training_config.num_epochs = max(50, training_config.num_epochs)
    
    # Create trainer
    trainer = CNNLSTMTrainer(training_config)
    
    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders(
        timeframes=["1min", "5min", "15min"],
        sequence_length=100,
        target_columns=["price_prediction"],
        batch_size=training_config.batch_size,
        validation_split=training_config.validation_split,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory
    )
    
    # Train model
    results = trainer.train(train_loader, val_loader)
    
    logger.info("CNN model training completed")
    logger.info(f"Best metrics: {results['best_metrics']}")
    
    return results


def train_lstm_models(config: DictConfig, logger: logging.Logger) -> Dict[str, Any]:
    """Train LSTM models for temporal sequence modeling."""
    logger.info("Starting LSTM model training for temporal sequence modeling")
    
    # Create LSTM-specific training configuration
    training_config = create_training_config(config, model_type="lstm")
    
    # Override epochs for LSTM training (100+ epochs as per requirements)
    training_config.num_epochs = max(100, training_config.num_epochs)
    
    # Create trainer
    trainer = CNNLSTMTrainer(training_config)
    
    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders(
        timeframes=["sequence"],  # LSTM uses sequential data
        sequence_length=100,
        target_columns=["price_prediction"],
        batch_size=training_config.batch_size,
        validation_split=training_config.validation_split,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory
    )
    
    # Train model
    results = trainer.train(train_loader, val_loader)
    
    logger.info("LSTM model training completed")
    logger.info(f"Best metrics: {results['best_metrics']}")
    
    return results


def train_hybrid_models(config: DictConfig, logger: logging.Logger) -> Dict[str, Any]:
    """Train integrated CNN+LSTM hybrid architecture."""
    logger.info("Starting hybrid CNN+LSTM model training")
    
    # Create hybrid training configuration
    training_config = create_training_config(config, model_type="hybrid")
    
    # Override epochs for hybrid training (200+ epochs as per requirements)
    training_config.num_epochs = max(200, training_config.num_epochs)
    
    # Create trainer
    trainer = CNNLSTMTrainer(training_config)
    
    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders(
        timeframes=["1min", "5min", "15min"],
        sequence_length=100,
        target_columns=["price_prediction", "volatility_estimation", "regime_detection"],
        batch_size=training_config.batch_size,
        validation_split=training_config.validation_split,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory
    )
    
    # Train model
    results = trainer.train(train_loader, val_loader)
    
    logger.info("Hybrid CNN+LSTM model training completed")
    logger.info(f"Best metrics: {results['best_metrics']}")
    
    return results


def create_dummy_data_loaders(config: TrainingConfig):
    """Create dummy data loaders for testing purposes."""
    # This is a placeholder implementation
    # In the actual implementation, this would create proper data loaders
    # from the data pipeline module
    
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create dummy data based on model type
    batch_size = config.batch_size
    seq_len = 100
    input_dim = 5
    num_samples = 1000
    
    if config.model_type == "hybrid":
        # Multi-timeframe data for hybrid model
        data_1min = torch.randn(num_samples, input_dim, seq_len)
        data_5min = torch.randn(num_samples, input_dim, seq_len)
        data_15min = torch.randn(num_samples, input_dim, seq_len)
        sequence_data = torch.randn(num_samples, seq_len, input_dim)
        
        # Create targets for multi-task learning
        price_targets = torch.randn(num_samples, 1)
        volatility_targets = torch.randn(num_samples, 1)
        regime_targets = torch.randint(0, 4, (num_samples,))
        
        # Package data
        data_dict = {
            "1min": data_1min,
            "5min": data_5min,
            "15min": data_15min,
            "sequence_data": sequence_data
        }
        
        targets_dict = {
            "price_prediction": price_targets,
            "volatility_estimation": volatility_targets,
            "regime_detection": regime_targets
        }
        
        # Create custom dataset
        class HybridDataset(torch.utils.data.Dataset):
            def __init__(self, data_dict, targets_dict):
                self.data_dict = data_dict
                self.targets_dict = targets_dict
                self.length = len(data_dict["1min"])
            
            def __len__(self):
                return self.length
            
            def __getitem__(self, idx):
                data = {k: v[idx] for k, v in self.data_dict.items()}
                targets = {k: v[idx] for k, v in self.targets_dict.items()}
                return data, targets
        
        dataset = HybridDataset(data_dict, targets_dict)
        
    elif config.model_type == "cnn":
        # Multi-timeframe data for CNN
        data_1min = torch.randn(num_samples, input_dim, seq_len)
        data_5min = torch.randn(num_samples, input_dim, seq_len)
        data_15min = torch.randn(num_samples, input_dim, seq_len)
        targets = torch.randn(num_samples, 1)
        
        data_dict = {
            "1min": data_1min,
            "5min": data_5min,
            "15min": data_15min
        }
        
        class CNNDataset(torch.utils.data.Dataset):
            def __init__(self, data_dict, targets):
                self.data_dict = data_dict
                self.targets = targets
                self.length = len(targets)
            
            def __len__(self):
                return self.length
            
            def __getitem__(self, idx):
                data = {k: v[idx] for k, v in self.data_dict.items()}
                return data, self.targets[idx]
        
        dataset = CNNDataset(data_dict, targets)
        
    elif config.model_type == "lstm":
        # Sequential data for LSTM
        data = torch.randn(num_samples, seq_len, input_dim)
        targets = torch.randn(num_samples, 1)
        dataset = TensorDataset(data, targets)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    return train_loader, val_loader


def run_hyperparameter_optimization(config: DictConfig, logger: logging.Logger) -> Dict[str, Any]:
    """Run hyperparameter optimization using Optuna."""
    if not config.hyperparameter_optimization.enabled:
        logger.info("Hyperparameter optimization disabled")
        return {}
    
    logger.info("Starting hyperparameter optimization")
    
    try:
        import optuna
        from optuna.pruners import MedianPruner
    except ImportError:
        logger.error("Optuna not installed. Please install with: pip install optuna")
        return {}
    
    def objective(trial):
        """Optuna objective function."""
        # Sample hyperparameters
        learning_rate = trial.suggest_loguniform(
            "learning_rate",
            config.hyperparameter_optimization.optuna.search_space.learning_rate.low,
            config.hyperparameter_optimization.optuna.search_space.learning_rate.high
        )
        
        batch_size = trial.suggest_categorical(
            "batch_size",
            config.hyperparameter_optimization.optuna.search_space.batch_size.choices
        )
        
        fusion_dim = trial.suggest_categorical(
            "fusion_dim",
            config.hyperparameter_optimization.optuna.search_space.fusion_dim.choices
        )
        
        num_attention_heads = trial.suggest_categorical(
            "num_attention_heads",
            config.hyperparameter_optimization.optuna.search_space.num_attention_heads.choices
        )
        
        dropout_rate = trial.suggest_uniform(
            "dropout_rate",
            config.hyperparameter_optimization.optuna.search_space.dropout_rate.low,
            config.hyperparameter_optimization.optuna.search_space.dropout_rate.high
        )
        
        weight_decay = trial.suggest_loguniform(
            "weight_decay",
            config.hyperparameter_optimization.optuna.search_space.weight_decay.low,
            config.hyperparameter_optimization.optuna.search_space.weight_decay.high
        )
        
        # Create modified configuration
        training_config = create_training_config(config, model_type="hybrid")
        training_config.learning_rate = learning_rate
        training_config.batch_size = batch_size
        training_config.weight_decay = weight_decay
        
        # Update model config
        training_config.model_config["fusion_dim"] = fusion_dim
        training_config.model_config["fusion_heads"] = num_attention_heads
        training_config.model_config["fusion_dropout"] = dropout_rate
        
        # Reduce epochs for optimization
        training_config.num_epochs = 50
        training_config.early_stopping_patience = 10
        
        # Create trainer
        trainer = CNNLSTMTrainer(training_config)
        
        # Create data loaders
        train_loader, val_loader = create_dummy_data_loaders(training_config)
        
        # Train model
        try:
            results = trainer.train(train_loader, val_loader)
            
            # Return metric to optimize
            if config.hyperparameter_optimization.optuna.direction == "maximize":
                return results["best_metrics"].get("val_accuracy", 0.0)
            else:
                return results["best_metrics"].get("val_loss", float('inf'))
        
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return float('inf') if config.hyperparameter_optimization.optuna.direction == "minimize" else 0.0
    
    # Create study
    study = optuna.create_study(
        study_name=config.hyperparameter_optimization.optuna.study_name,
        direction=config.hyperparameter_optimization.optuna.direction,
        pruner=MedianPruner() if config.hyperparameter_optimization.optuna.pruning else None
    )
    
    # Optimize
    study.optimize(objective, n_trials=config.hyperparameter_optimization.optuna.n_trials)
    
    logger.info("Hyperparameter optimization completed")
    logger.info(f"Best parameters: {study.best_params}")
    logger.info(f"Best value: {study.best_value}")
    
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study
    }


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="CNN+LSTM Training Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/cnn_lstm_training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["cnn", "lstm", "hybrid", "all"],
        default="hybrid",
        help="Type of model to train"
    )
    parser.add_argument(
        "--optimize-hyperparams",
        action="store_true",
        help="Run hyperparameter optimization"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Setup GPU
    setup_gpu()
    device_info = get_device_info()
    logger.info(f"Device info: {device_info}")
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Create logs directory
    Path("experiments/logs").mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    try:
        # Run hyperparameter optimization if requested
        if args.optimize_hyperparams:
            optimization_results = run_hyperparameter_optimization(config, logger)
            results["hyperparameter_optimization"] = optimization_results
        
        # Train models based on type
        if args.model_type == "cnn" or args.model_type == "all":
            cnn_results = train_cnn_models(config, logger)
            results["cnn_training"] = cnn_results
        
        if args.model_type == "lstm" or args.model_type == "all":
            lstm_results = train_lstm_models(config, logger)
            results["lstm_training"] = lstm_results
        
        if args.model_type == "hybrid" or args.model_type == "all":
            hybrid_results = train_hybrid_models(config, logger)
            results["hybrid_training"] = hybrid_results
        
        logger.info("All training completed successfully!")
        
        # Save results summary
        results_path = Path("experiments/results") / "cnn_lstm_training_results.yaml"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        logger.info(f"Results saved to {results_path}")
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()