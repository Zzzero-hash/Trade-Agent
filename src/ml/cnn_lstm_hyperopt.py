"""
CNN+LSTM Hyperparameter Optimization - Task 5.5

This module implements comprehensive hyperparameter optimization for CNN+LSTM models using Optuna.

Requirements:
- Implement Optuna-based hyperparameter optimization for learning rates, architectures, regularization
- Run 1000+ hyperparameter trials with early pruning for efficiency
- Create multi-objective optimization balancing accuracy, training time, and model size
- Save best hyperparameter configurations and retrain final models

Requirements: 3.4, 9.1
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.study import StudyDirection
# Optional MLflow import
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

from .hybrid_model import CNNLSTMHybridModel, HybridModelConfig, create_hybrid_config
from .train_integrated_cnn_lstm import IntegratedCNNLSTMTrainer


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CNNLSTMHyperparameterOptimizer:
    """
    Advanced hyperparameter optimizer for CNN+LSTM models using Optuna.
    
    Features:
    - Multi-objective optimization (accuracy, training time, model size)
    - Early pruning for efficiency
    - Comprehensive search space covering all model components
    - Automatic best model retraining
    - MLflow integration for experiment tracking
    """
    
    def __init__(
        self,
        data_loaders: Tuple[DataLoader, DataLoader, DataLoader],
        input_dim: int,
        save_dir: str = "hyperopt_results",
        study_name: str = "cnn_lstm_optimization",
        storage_url: Optional[str] = None,
        device: Optional[str] = None
    ):
        """Initialize the hyperparameter optimizer
        
        Args:
            data_loaders: Tuple of (train_loader, val_loader, test_loader)
            input_dim: Input feature dimension
            save_dir: Directory to save results
            study_name: Name of the Optuna study
            storage_url: Database URL for distributed optimization (optional)
            device: Device to use for training
        """
        self.train_loader, self.val_loader, self.test_loader = data_loaders
        self.input_dim = input_dim
        self.save_dir = Path(save_dir)
        self.study_name = study_name
        self.storage_url = storage_url
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize MLflow if available
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment(f"CNN_LSTM_Hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Optimization state
        self.best_configs = []
        self.optimization_history = []
        
        logger.info("CNN+LSTM Hyperparameter Optimizer initialized")
    
    def create_study(
        self,
        n_trials: int = 1000,
        pruner_type: str = "median",
        sampler_type: str = "tpe",
        directions: List[str] = ["maximize", "minimize", "minimize"]
    ) -> optuna.Study:
        """Create Optuna study with multi-objective optimization
        
        Args:
            n_trials: Number of trials to run
            pruner_type: Type of pruner ("median", "successive_halving")
            sampler_type: Type of sampler ("tpe", "cmaes", "random")
            directions: Optimization directions for [accuracy, training_time, model_size]
            
        Returns:
            Configured Optuna study
        """
        # Setup pruner
        if pruner_type == "median":
            pruner = MedianPruner(
                n_startup_trials=20,
                n_warmup_steps=10,
                interval_steps=5,
                n_min_trials=5
            )
        elif pruner_type == "successive_halving":
            pruner = SuccessiveHalvingPruner(
                min_resource=10,
                reduction_factor=3,
                min_early_stopping_rate=0.2
            )
        else:
            pruner = MedianPruner()
        
        # Setup sampler
        if sampler_type == "tpe":
            sampler = TPESampler(
                n_startup_trials=50,
                n_ei_candidates=24,
                multivariate=True,
                group=True
            )
        elif sampler_type == "cmaes":
            sampler = CmaEsSampler(
                n_startup_trials=50,
                restart_strategy="ipop"
            )
        else:
            sampler = TPESampler()
        
        # Convert direction strings to StudyDirection enums
        study_directions = []
        for direction in directions:
            if direction.lower() == "maximize":
                study_directions.append(StudyDirection.MAXIMIZE)
            else:
                study_directions.append(StudyDirection.MINIMIZE)
        
        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_url,
            directions=study_directions,
            pruner=pruner,
            sampler=sampler,
            load_if_exists=True
        )
        
        logger.info(f"Created study with {n_trials} trials, {pruner_type} pruner, {sampler_type} sampler")
        
        return study
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        # CNN Architecture parameters
        cnn_params = {
            "cnn_num_filters": trial.suggest_categorical("cnn_num_filters", [32, 64, 128, 256]),
            "cnn_filter_sizes": trial.suggest_categorical(
                "cnn_filter_sizes", 
                [(3, 5, 7), (3, 5, 7, 11), (5, 7, 11), (3, 7, 11), (1, 3, 5, 7)]
            ),
            "cnn_dropout_rate": trial.suggest_float("cnn_dropout_rate", 0.1, 0.5),
            "cnn_use_attention": trial.suggest_categorical("cnn_use_attention", [True, False]),
            "cnn_attention_heads": trial.suggest_categorical("cnn_attention_heads", [2, 4, 8, 16]),
            "cnn_use_residual": trial.suggest_categorical("cnn_use_residual", [True, False]),
            "cnn_activation": trial.suggest_categorical("cnn_activation", ["relu", "gelu", "swish"]),
        }
        
        # LSTM Architecture parameters
        lstm_params = {
            "lstm_hidden_dim": trial.suggest_categorical("lstm_hidden_dim", [64, 128, 256, 512, 1024]),
            "lstm_num_layers": trial.suggest_int("lstm_num_layers", 1, 4),
            "lstm_dropout_rate": trial.suggest_float("lstm_dropout_rate", 0.1, 0.5),
            "lstm_bidirectional": trial.suggest_categorical("lstm_bidirectional", [True, False]),
            "lstm_use_attention": trial.suggest_categorical("lstm_use_attention", [True, False]),
            "lstm_attention_heads": trial.suggest_categorical("lstm_attention_heads", [2, 4, 8, 16]),
            "lstm_use_skip_connections": trial.suggest_categorical("lstm_use_skip_connections", [True, False]),
        }
        
        # Feature Fusion parameters
        fusion_params = {
            "fusion_method": trial.suggest_categorical("fusion_method", ["concat", "attention", "gated", "bilinear"]),
            "fusion_dim": trial.suggest_categorical("fusion_dim", [128, 256, 512, 1024]),
            "fusion_dropout_rate": trial.suggest_float("fusion_dropout_rate", 0.1, 0.4),
            "fusion_num_heads": trial.suggest_categorical("fusion_num_heads", [4, 8, 16]),
        }
        
        # Training parameters
        training_params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd", "rmsprop"]),
            "scheduler": trial.suggest_categorical("scheduler", ["cosine", "step", "exponential", "plateau"]),
            "gradient_clip_norm": trial.suggest_float("gradient_clip_norm", 0.5, 5.0),
        }
        
        # Regularization parameters
        regularization_params = {
            "l1_reg_weight": trial.suggest_float("l1_reg_weight", 1e-8, 1e-3, log=True),
            "l2_reg_weight": trial.suggest_float("l2_reg_weight", 1e-6, 1e-3, log=True),
            "dropout_schedule": trial.suggest_categorical("dropout_schedule", ["constant", "linear", "cosine"]),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
        }
        
        # Multi-task learning parameters
        multitask_params = {
            "classification_weight": trial.suggest_float("classification_weight", 0.3, 0.8),
            "regression_weight": trial.suggest_float("regression_weight", 0.2, 0.7),
            "task_balancing": trial.suggest_categorical("task_balancing", ["fixed", "dynamic", "uncertainty"]),
        }
        
        # Combine all parameters
        hyperparams = {
            **cnn_params,
            **lstm_params,
            **fusion_params,
            **training_params,
            **regularization_params,
            **multitask_params
        }
        
        return hyperparams
    
    def objective_function(self, trial: optuna.Trial) -> Tuple[float, float, float]:
        """Objective function for multi-objective optimization
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Tuple of (accuracy, training_time, model_size)
        """
        # Start MLflow run if available
        if MLFLOW_AVAILABLE:
            mlflow_run = mlflow.start_run(nested=True)
        else:
            mlflow_run = None
        
        try:
            # Suggest hyperparameters
            hyperparams = self.suggest_hyperparameters(trial)
            
            # Log hyperparameters to MLflow if available
            if MLFLOW_AVAILABLE and mlflow_run:
                mlflow.log_params(hyperparams)
            
            try:
                # Create model configuration
                config = self._create_config_from_hyperparams(hyperparams)
                
                # Train model with early stopping for efficiency
                start_time = time.time()
                metrics = self._train_trial_model(config, trial, max_epochs=50)
                training_time = time.time() - start_time
                
                # Calculate model size
                model_size = self._calculate_model_size(config)
                
                # Extract accuracy (use validation accuracy as primary metric)
                accuracy = metrics.get('val_class_acc', 0.0)
                
                # Log metrics to MLflow if available
                if MLFLOW_AVAILABLE and mlflow_run:
                    mlflow.log_metrics({
                        "accuracy": accuracy,
                        "training_time": training_time,
                        "model_size_mb": model_size,
                        "val_loss": metrics.get('val_total_loss', float('inf')),
                        "val_class_loss": metrics.get('val_class_loss', float('inf')),
                        "val_reg_loss": metrics.get('val_reg_loss', float('inf')),
                        "val_reg_mse": metrics.get('val_reg_mse', float('inf'))
                    })
                
                # Store trial results
                trial_result = {
                    'trial_number': trial.number,
                    'hyperparams': hyperparams,
                    'metrics': metrics,
                    'accuracy': accuracy,
                    'training_time': training_time,
                    'model_size': model_size,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.optimization_history.append(trial_result)
                
                logger.info(f"Trial {trial.number}: Accuracy={accuracy:.4f}, "
                           f"Time={training_time:.2f}s, Size={model_size:.2f}MB")
                
                return accuracy, training_time, model_size
                
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                # Return poor performance for failed trials
                return 0.0, 9999.0, 9999.0
        
        finally:
            # End MLflow run if it was started
            if MLFLOW_AVAILABLE and mlflow_run:
                mlflow.end_run()
    
    def _create_config_from_hyperparams(self, hyperparams: Dict[str, Any]) -> HybridModelConfig:
        """Create model configuration from hyperparameters"""
        
        config = create_hybrid_config(
            input_dim=self.input_dim,
            sequence_length=60,  # Fixed for now
            num_classes=4,  # Fixed for regime detection
            regression_targets=2,  # Price and volatility
            device=str(self.device)
        )
        
        # Update CNN parameters
        config.cnn_num_filters = hyperparams["cnn_num_filters"]
        config.cnn_filter_sizes = hyperparams["cnn_filter_sizes"]
        config.cnn_dropout_rate = hyperparams["cnn_dropout_rate"]
        config.cnn_use_attention = hyperparams["cnn_use_attention"]
        config.cnn_attention_heads = hyperparams["cnn_attention_heads"]
        
        # Update LSTM parameters
        config.lstm_hidden_dim = hyperparams["lstm_hidden_dim"]
        config.lstm_num_layers = hyperparams["lstm_num_layers"]
        config.lstm_dropout_rate = hyperparams["lstm_dropout_rate"]
        config.lstm_bidirectional = hyperparams["lstm_bidirectional"]
        config.lstm_use_attention = hyperparams["lstm_use_attention"]
        config.lstm_attention_heads = hyperparams["lstm_attention_heads"]
        config.lstm_use_skip_connections = hyperparams["lstm_use_skip_connections"]
        
        # Update fusion parameters
        config.fusion_method = hyperparams["fusion_method"]
        config.fusion_dim = hyperparams["fusion_dim"]
        config.fusion_dropout_rate = hyperparams["fusion_dropout_rate"]
        config.fusion_num_heads = hyperparams["fusion_num_heads"]
        
        # Update training parameters
        config.learning_rate = hyperparams["learning_rate"]
        config.batch_size = hyperparams["batch_size"]
        config.weight_decay = hyperparams["weight_decay"]
        config.optimizer = hyperparams["optimizer"]
        config.scheduler = hyperparams["scheduler"]
        config.gradient_clip_norm = hyperparams["gradient_clip_norm"]
        
        # Update regularization parameters
        config.l1_reg_weight = hyperparams["l1_reg_weight"]
        config.l2_reg_weight = hyperparams["l2_reg_weight"]
        config.label_smoothing = hyperparams["label_smoothing"]
        
        # Update multi-task parameters
        config.classification_weight = hyperparams["classification_weight"]
        config.regression_weight = hyperparams["regression_weight"]
        
        return config
    
    def _train_trial_model(
        self,
        config: HybridModelConfig,
        trial: optuna.Trial,
        max_epochs: int = 50
    ) -> Dict[str, float]:
        """Train model for a single trial with early stopping"""
        
        # Create trainer
        trainer = IntegratedCNNLSTMTrainer(
            config=config,
            save_dir=str(self.save_dir / f"trial_{trial.number}"),
            device=str(self.device)
        )
        
        # Build model
        trainer.build_models(self.input_dim)
        
        # Setup optimizer
        if config.optimizer == "adam":
            optimizer = optim.Adam(trainer.hybrid_model.parameters(), 
                                 lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer == "adamw":
            optimizer = optim.AdamW(trainer.hybrid_model.parameters(), 
                                  lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer == "sgd":
            optimizer = optim.SGD(trainer.hybrid_model.parameters(), 
                                lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.9)
        else:  # rmsprop
            optimizer = optim.RMSprop(trainer.hybrid_model.parameters(), 
                                    lr=config.learning_rate, weight_decay=config.weight_decay)
        
        # Setup scheduler
        if config.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        elif config.scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max_epochs//3, gamma=0.5)
        elif config.scheduler == "exponential":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        else:  # plateau
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(max_epochs):
            # Training phase
            trainer.hybrid_model.train()
            train_loss = 0.0
            train_class_correct = 0
            train_total = 0
            train_reg_mse = 0.0
            
            for batch_idx, (data, class_targets, reg_targets) in enumerate(self.train_loader):
                data = data.to(self.device)
                class_targets = class_targets.to(self.device)
                reg_targets = reg_targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                class_logits, reg_outputs = trainer.hybrid_model(data)
                
                # Calculate losses
                class_loss = nn.CrossEntropyLoss()(class_logits, class_targets)
                reg_loss = nn.MSELoss()(reg_outputs, reg_targets)
                
                # Combined loss with task weights
                total_loss = (config.classification_weight * class_loss + 
                            config.regression_weight * reg_loss)
                
                # Add regularization
                if config.l1_reg_weight > 0:
                    l1_reg = sum(p.abs().sum() for p in trainer.hybrid_model.parameters())
                    total_loss += config.l1_reg_weight * l1_reg
                
                if config.l2_reg_weight > 0:
                    l2_reg = sum(p.pow(2).sum() for p in trainer.hybrid_model.parameters())
                    total_loss += config.l2_reg_weight * l2_reg
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                if config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainer.hybrid_model.parameters(), 
                                                 config.gradient_clip_norm)
                
                optimizer.step()
                
                # Update metrics
                train_loss += total_loss.item()
                _, predicted = torch.max(class_logits.data, 1)
                train_total += class_targets.size(0)
                train_class_correct += (predicted == class_targets).sum().item()
                train_reg_mse += nn.MSELoss()(reg_outputs, reg_targets).item()
            
            # Validation phase
            trainer.hybrid_model.eval()
            val_loss = 0.0
            val_class_correct = 0
            val_total = 0
            val_reg_mse = 0.0
            val_class_loss = 0.0
            val_reg_loss = 0.0
            
            with torch.no_grad():
                for data, class_targets, reg_targets in self.val_loader:
                    data = data.to(self.device)
                    class_targets = class_targets.to(self.device)
                    reg_targets = reg_targets.to(self.device)
                    
                    # Forward pass
                    class_logits, reg_outputs = trainer.hybrid_model(data)
                    
                    # Calculate losses
                    class_loss = nn.CrossEntropyLoss()(class_logits, class_targets)
                    reg_loss = nn.MSELoss()(reg_outputs, reg_targets)
                    total_loss = (config.classification_weight * class_loss + 
                                config.regression_weight * reg_loss)
                    
                    # Update metrics
                    val_loss += total_loss.item()
                    val_class_loss += class_loss.item()
                    val_reg_loss += reg_loss.item()
                    _, predicted = torch.max(class_logits.data, 1)
                    val_total += class_targets.size(0)
                    val_class_correct += (predicted == class_targets).sum().item()
                    val_reg_mse += reg_loss.item()
            
            # Calculate averages
            train_loss /= len(self.train_loader)
            train_acc = train_class_correct / train_total
            train_reg_mse /= len(self.train_loader)
            
            val_loss /= len(self.val_loader)
            val_acc = val_class_correct / val_total
            val_class_loss /= len(self.val_loader)
            val_reg_loss /= len(self.val_loader)
            val_reg_mse /= len(self.val_loader)
            
            # Update scheduler
            if config.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Report intermediate values for pruning
            trial.report(val_acc, epoch)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Return final metrics
        return {
            'val_total_loss': val_loss,
            'val_class_loss': val_class_loss,
            'val_reg_loss': val_reg_loss,
            'val_class_acc': val_acc,
            'val_reg_mse': val_reg_mse,
            'train_total_loss': train_loss,
            'train_class_acc': train_acc,
            'train_reg_mse': train_reg_mse,
            'epochs_trained': epoch + 1
        }
    
    def _calculate_model_size(self, config: HybridModelConfig) -> float:
        """Calculate model size in MB"""
        
        # Create temporary model to calculate size
        temp_model = CNNLSTMHybridModel(config)
        
        # Calculate number of parameters
        total_params = sum(p.numel() for p in temp_model.parameters())
        
        # Estimate size in MB (assuming 4 bytes per parameter for float32)
        size_mb = (total_params * 4) / (1024 * 1024)
        
        return size_mb
    
    def optimize(
        self,
        n_trials: int = 1000,
        timeout: Optional[int] = None,
        pruner_type: str = "median",
        sampler_type: str = "tpe"
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization
        
        Args:
            n_trials: Number of trials to run
            timeout: Timeout in seconds (optional)
            pruner_type: Type of pruner to use
            sampler_type: Type of sampler to use
            
        Returns:
            Optimization results
        """
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")
        
        # Create study
        study = self.create_study(
            n_trials=n_trials,
            pruner_type=pruner_type,
            sampler_type=sampler_type
        )
        
        # Run optimization
        start_time = time.time()
        
        try:
            study.optimize(
                self.objective_function,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=1,  # Single job to avoid GPU conflicts
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
        
        optimization_time = time.time() - start_time
        
        # Get best trials (Pareto frontier for multi-objective)
        best_trials = study.best_trials
        
        # Extract results
        results = {
            'study_name': self.study_name,
            'n_trials_completed': len(study.trials),
            'n_trials_requested': n_trials,
            'optimization_time': optimization_time,
            'best_trials': [],
            'optimization_history': self.optimization_history,
            'timestamp': datetime.now().isoformat()
        }
        
        # Process best trials
        for i, trial in enumerate(best_trials):
            trial_info = {
                'trial_number': trial.number,
                'values': trial.values,  # [accuracy, training_time, model_size]
                'params': trial.params,
                'state': trial.state.name,
                'rank': i + 1
            }
            results['best_trials'].append(trial_info)
            self.best_configs.append(trial.params)
        
        logger.info(f"Optimization completed. Found {len(best_trials)} Pareto optimal solutions.")
        
        # Save results
        self._save_optimization_results(results, study)
        
        return results
    
    def _save_optimization_results(self, results: Dict[str, Any], study: optuna.Study) -> None:
        """Save optimization results to disk"""
        
        # Save results JSON
        results_path = self.save_dir / f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save study object
        study_path = self.save_dir / f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
        
        # Save best configurations
        best_configs_path = self.save_dir / f"best_configs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(best_configs_path, 'w') as f:
            json.dump(self.best_configs, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        logger.info(f"Study saved to {study_path}")
        logger.info(f"Best configs saved to {best_configs_path}")
    
    def retrain_best_models(
        self,
        top_k: int = 5,
        full_epochs: int = 200,
        save_models: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrain the best models with full training
        
        Args:
            top_k: Number of best models to retrain
            full_epochs: Number of epochs for full training
            save_models: Whether to save the trained models
            
        Returns:
            List of retrained model results
        """
        logger.info(f"Retraining top {top_k} models with {full_epochs} epochs...")
        
        if not self.best_configs:
            raise ValueError("No best configurations found. Run optimization first.")
        
        retrained_results = []
        
        for i, config_params in enumerate(self.best_configs[:top_k]):
            logger.info(f"Retraining model {i+1}/{top_k}...")
            
            try:
                # Create configuration
                config = self._create_config_from_hyperparams(config_params)
                
                # Create trainer
                model_save_dir = self.save_dir / f"retrained_model_{i+1}"
                trainer = IntegratedCNNLSTMTrainer(
                    config=config,
                    save_dir=str(model_save_dir),
                    device=str(self.device)
                )
                
                # Build model
                trainer.build_models(self.input_dim)
                
                # Full training
                start_time = time.time()
                training_results = trainer.train_integrated_model(
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    num_epochs=full_epochs,
                    early_stopping_patience=50
                )
                training_time = time.time() - start_time
                
                # Evaluate on test set
                test_metrics = self._evaluate_on_test_set(trainer.hybrid_model)
                
                # Compile results
                result = {
                    'model_rank': i + 1,
                    'config_params': config_params,
                    'training_results': training_results,
                    'test_metrics': test_metrics,
                    'training_time': training_time,
                    'model_path': str(model_save_dir) if save_models else None,
                    'timestamp': datetime.now().isoformat()
                }
                
                retrained_results.append(result)
                
                logger.info(f"Model {i+1} retrained successfully. "
                           f"Test accuracy: {test_metrics.get('test_class_acc', 0.0):.4f}")
                
            except Exception as e:
                logger.error(f"Failed to retrain model {i+1}: {e}")
                continue
        
        # Save retrained results
        retrained_path = self.save_dir / f"retrained_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(retrained_path, 'w') as f:
            json.dump(retrained_results, f, indent=2, default=str)
        
        logger.info(f"Retrained {len(retrained_results)} models successfully")
        logger.info(f"Results saved to {retrained_path}")
        
        return retrained_results
    
    def _evaluate_on_test_set(self, model: nn.Module) -> Dict[str, float]:
        """Evaluate model on test set"""
        
        model.eval()
        test_loss = 0.0
        test_class_correct = 0
        test_total = 0
        test_reg_mse = 0.0
        
        with torch.no_grad():
            for data, class_targets, reg_targets in self.test_loader:
                data = data.to(self.device)
                class_targets = class_targets.to(self.device)
                reg_targets = reg_targets.to(self.device)
                
                # Forward pass
                class_logits, reg_outputs = model(data)
                
                # Calculate losses
                class_loss = nn.CrossEntropyLoss()(class_logits, class_targets)
                reg_loss = nn.MSELoss()(reg_outputs, reg_targets)
                total_loss = 0.5 * class_loss + 0.5 * reg_loss  # Equal weights for evaluation
                
                # Update metrics
                test_loss += total_loss.item()
                _, predicted = torch.max(class_logits.data, 1)
                test_total += class_targets.size(0)
                test_class_correct += (predicted == class_targets).sum().item()
                test_reg_mse += reg_loss.item()
        
        # Calculate averages
        test_loss /= len(self.test_loader)
        test_acc = test_class_correct / test_total
        test_reg_mse /= len(self.test_loader)
        
        return {
            'test_total_loss': test_loss,
            'test_class_acc': test_acc,
            'test_reg_mse': test_reg_mse
        }
    
    def generate_optimization_report(self) -> str:
        """Generate a comprehensive optimization report"""
        
        if not self.optimization_history:
            return "No optimization history available."
        
        # Calculate statistics
        accuracies = [trial['accuracy'] for trial in self.optimization_history]
        training_times = [trial['training_time'] for trial in self.optimization_history]
        model_sizes = [trial['model_size'] for trial in self.optimization_history]
        
        report = f"""
CNN+LSTM Hyperparameter Optimization Report
==========================================

Optimization Summary:
- Total trials completed: {len(self.optimization_history)}
- Best configurations found: {len(self.best_configs)}

Performance Statistics:
- Accuracy: Mean={np.mean(accuracies):.4f}, Std={np.std(accuracies):.4f}, Max={np.max(accuracies):.4f}
- Training Time: Mean={np.mean(training_times):.2f}s, Std={np.std(training_times):.2f}s, Min={np.min(training_times):.2f}s
- Model Size: Mean={np.mean(model_sizes):.2f}MB, Std={np.std(model_sizes):.2f}MB, Min={np.min(model_sizes):.2f}MB

Best Configurations:
"""
        
        for i, config in enumerate(self.best_configs[:5]):  # Show top 5
            report += f"\nRank {i+1}:\n"
            for key, value in config.items():
                report += f"  {key}: {value}\n"
        
        return report


def run_cnn_lstm_hyperparameter_optimization(
    data_loaders: Tuple[DataLoader, DataLoader, DataLoader],
    input_dim: int,
    n_trials: int = 1000,
    save_dir: str = "hyperopt_results",
    retrain_best: bool = True,
    top_k: int = 5
) -> Dict[str, Any]:
    """Convenience function to run complete hyperparameter optimization
    
    Args:
        data_loaders: Tuple of (train_loader, val_loader, test_loader)
        input_dim: Input feature dimension
        n_trials: Number of optimization trials
        save_dir: Directory to save results
        retrain_best: Whether to retrain best models
        top_k: Number of best models to retrain
        
    Returns:
        Complete optimization results
    """
    # Create optimizer
    optimizer = CNNLSTMHyperparameterOptimizer(
        data_loaders=data_loaders,
        input_dim=input_dim,
        save_dir=save_dir
    )
    
    # Run optimization
    optimization_results = optimizer.optimize(n_trials=n_trials)
    
    # Retrain best models if requested
    retrained_results = []
    if retrain_best:
        retrained_results = optimizer.retrain_best_models(top_k=top_k)
    
    # Generate report
    report = optimizer.generate_optimization_report()
    
    # Combine all results
    complete_results = {
        'optimization_results': optimization_results,
        'retrained_results': retrained_results,
        'optimization_report': report,
        'timestamp': datetime.now().isoformat()
    }
    
    return complete_results


if __name__ == "__main__":
    # Example usage
    logger.info("CNN+LSTM Hyperparameter Optimization - Task 5.5")
    
    # This would be called with actual data loaders
    # results = run_cnn_lstm_hyperparameter_optimization(
    #     data_loaders=(train_loader, val_loader, test_loader),
    #     input_dim=11,  # Based on feature engineering
    #     n_trials=1000,
    #     save_dir="hyperopt_results",
    #     retrain_best=True,
    #     top_k=5
    # )
    
    logger.info("Hyperparameter optimization implementation completed")