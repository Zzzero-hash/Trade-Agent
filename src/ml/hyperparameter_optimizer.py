"""
Advanced CNN+LSTM Hyperparameter Optimization - Task 5.5

This module implements comprehensive hyperparameter optimization using Optuna
with multi-objective optimization, early pruning, and automated model retraining.

Features:
- Optuna-based hyperparameter optimization for learning rates, architectures, regularization
- 1000+ hyperparameter trials with early pruning for efficiency
- Multi-objective optimization balancing accuracy, training time, and model size
- Best hyperparameter configuration saving and final model retraining

Requirements: 3.4, 9.1
"""

import os
import json
import pickle
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import optuna
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler, NSGAIISampler
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_pareto_front,
    plot_slice
)

from .train_integrated_cnn_lstm import IntegratedCNNLSTMTrainer
from .hybrid_model import create_hybrid_config, HybridModelConfig

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    
    # Optimization settings
    n_trials: int = 1000
    timeout: Optional[int] = None  # seconds
    max_epochs_per_trial: int = 50
    early_stopping_patience: int = 10
    
    # Multi-objective settings
    objectives: List[str] = None  # ['accuracy', 'training_time', 'model_size']
    
    # Pruning settings
    pruner_type: str = "hyperband"  # "median", "hyperband"
    pruning_warmup_steps: int = 5
    
    # Sampling settings
    sampler_type: str = "tpe"  # "tpe", "nsga2"
    
    # Save settings
    save_dir: str = "hyperopt_results"
    save_study: bool = True
    save_plots: bool = True
    
    # Retraining settings
    retrain_top_k: int = 3
    retrain_full_epochs: int = 200
    
    def __post_init__(self):
        if self.objectives is None:
            self.objectives = ['accuracy', 'training_time', 'model_size']


class MultiObjectiveOptimizer:
    """
    Multi-objective hyperparameter optimizer for CNN+LSTM models.
    
    Implements task 5.5 requirements:
    - Optuna-based optimization with 1000+ trials
    - Early pruning for efficiency
    - Multi-objective optimization
    - Best configuration saving and model retraining
    """
    
    def __init__(
        self,
        config: OptimizationConfig,
        data_loaders: Tuple[DataLoader, DataLoader, DataLoader],
        device: Optional[str] = None
    ):
        """Initialize the multi-objective optimizer."""
        self.config = config
        self.train_loader, self.val_loader, self.test_loader = data_loaders
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create save directory
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize study
        self.study = None
        self.pareto_front = []
        
        # Performance tracking
        self.trial_results = []
        self.optimization_start_time = None
        
        logger.info(f"MultiObjectiveOptimizer initialized with {len(config.objectives)} objectives")
        logger.info(f"Device: {self.device}")
        logger.info(f"Save directory: {self.save_dir}")
    
    def create_study(self) -> optuna.Study:
        """Create Optuna study with appropriate sampler and pruner."""
        
        # Create pruner
        if self.config.pruner_type == "median":
            pruner = MedianPruner(
                n_startup_trials=self.config.pruning_warmup_steps,
                n_warmup_steps=self.config.pruning_warmup_steps
            )
        elif self.config.pruner_type == "hyperband":
            pruner = HyperbandPruner(
                min_resource=self.config.pruning_warmup_steps,
                max_resource=self.config.max_epochs_per_trial,
                reduction_factor=3
            )
        else:
            pruner = MedianPruner()
        
        # Create sampler
        if len(self.config.objectives) > 1:
            # Multi-objective optimization
            sampler = NSGAIISampler(population_size=50)
            directions = []
            for obj in self.config.objectives:
                if obj in ['accuracy', 'f1_score', 'precision', 'recall']:
                    directions.append("maximize")
                else:  # training_time, model_size, loss, etc.
                    directions.append("minimize")
        else:
            # Single-objective optimization
            sampler = TPESampler(n_startup_trials=20)
            directions = ["maximize" if self.config.objectives[0] in 
                         ['accuracy', 'f1_score', 'precision', 'recall'] else "minimize"]
        
        # Create study
        study_name = f"cnn_lstm_hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if len(self.config.objectives) > 1:
            study = optuna.create_study(
                study_name=study_name,
                directions=directions,
                sampler=sampler,
                pruner=pruner
            )
        else:
            study = optuna.create_study(
                study_name=study_name,
                direction=directions[0],
                sampler=sampler,
                pruner=pruner
            )
        
        logger.info(f"Created study: {study_name}")
        logger.info(f"Objectives: {self.config.objectives}")
        logger.info(f"Directions: {directions}")
        
        return study
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial."""
        
        params = {}
        
        # Learning rate optimization
        params['learning_rate'] = trial.suggest_float(
            'learning_rate', 1e-5, 1e-2, log=True
        )
        
        # Architecture parameters
        params['cnn_num_filters'] = trial.suggest_categorical(
            'cnn_num_filters', [32, 64, 128, 256]
        )
        
        # Use string representation for categorical choices, then convert back
        filter_size_options = {
            'small': [3, 5, 7],
            'medium': [2, 4, 6], 
            'large': [3, 5],
            'xlarge': [5, 7, 9]
        }
        filter_choice = trial.suggest_categorical('cnn_filter_sizes', tuple(filter_size_options.keys()))
        params['cnn_filter_sizes'] = filter_size_options[filter_choice]
        
        params['lstm_hidden_dim'] = trial.suggest_categorical(
            'lstm_hidden_dim', [64, 128, 256, 512]
        )
        
        params['lstm_num_layers'] = trial.suggest_int(
            'lstm_num_layers', 1, 4
        )
        
        params['lstm_bidirectional'] = trial.suggest_categorical(
            'lstm_bidirectional', [True, False]
        )
        
        # Attention parameters
        params['use_attention'] = trial.suggest_categorical(
            'use_attention', [True, False]
        )
        
        if params['use_attention']:
            params['attention_heads'] = trial.suggest_categorical(
                'attention_heads', [4, 8, 16]
            )
        else:
            params['attention_heads'] = 8  # Default value when attention is disabled
        
        # Regularization parameters
        params['dropout_rate'] = trial.suggest_float(
            'dropout_rate', 0.1, 0.7
        )
        
        params['weight_decay'] = trial.suggest_float(
            'weight_decay', 1e-6, 1e-3, log=True
        )
        
        # Training parameters
        params['batch_size'] = trial.suggest_categorical(
            'batch_size', [16, 32, 64, 128]
        )
        
        params['optimizer_type'] = trial.suggest_categorical(
            'optimizer_type', ['adam', 'adamw', 'sgd']
        )
        
        # Scheduler parameters
        params['scheduler_type'] = trial.suggest_categorical(
            'scheduler_type', ['cosine', 'step', 'exponential', 'none']
        )
        
        if params['scheduler_type'] == 'step':
            params['step_size'] = trial.suggest_int('step_size', 10, 50)
            params['gamma'] = trial.suggest_float('gamma', 0.1, 0.9)
        elif params['scheduler_type'] == 'exponential':
            params['gamma'] = trial.suggest_float('gamma', 0.9, 0.99)
        
        # Feature fusion parameters
        params['fusion_dim'] = trial.suggest_categorical(
            'fusion_dim', [256, 512, 1024]
        )
        
        params['fusion_heads'] = trial.suggest_categorical(
            'fusion_heads', [4, 8, 16]
        )
        
        # Ensemble parameters
        params['num_ensemble_models'] = trial.suggest_int(
            'num_ensemble_models', 3, 7
        )
        
        return params
    
    def create_model_config(self, params: Dict[str, Any]) -> HybridModelConfig:
        """Create model configuration from hyperparameters."""
        
        config = create_hybrid_config(
            input_dim=11,  # OHLCV + derived features
            sequence_length=60,
            device=self.device
        )
        
        # Update with optimized parameters
        config.learning_rate = params['learning_rate']
        config.dropout_rate = params['dropout_rate']
        config.batch_size = params['batch_size']
        config.weight_decay = params['weight_decay']
        
        # CNN parameters
        config.cnn_num_filters = params['cnn_num_filters']
        config.cnn_filter_sizes = params['cnn_filter_sizes']  # This is now a list, not a string
        
        # LSTM parameters
        config.lstm_hidden_dim = params['lstm_hidden_dim']
        config.lstm_num_layers = params['lstm_num_layers']
        config.lstm_bidirectional = params['lstm_bidirectional']
        
        # Attention parameters
        config.cnn_use_attention = params['use_attention']
        config.lstm_use_attention = params['use_attention']
        # Ensure attention_heads exists, default to 8 if missing
        attention_heads = params.get('attention_heads', 8)
        config.cnn_attention_heads = attention_heads
        config.lstm_attention_heads = attention_heads
        
        # Feature fusion parameters
        config.fusion_dim = params['fusion_dim']
        config.fusion_heads = params['fusion_heads']
        
        # Ensemble parameters
        config.num_ensemble_models = params['num_ensemble_models']
        
        return config
    
    def objective(self, trial: optuna.Trial) -> Union[float, List[float]]:
        """Objective function for optimization."""
        
        trial_start_time = time.time()
        
        try:
            # Suggest hyperparameters
            params = self.suggest_hyperparameters(trial)
            
            # Create model configuration
            model_config = self.create_model_config(params)
            
            # Create trainer
            trainer = IntegratedCNNLSTMTrainer(
                config=model_config,
                save_dir=self.save_dir / f"trial_{trial.number}",
                device=self.device
            )
            
            # Build models
            # Get input dimension from data loader
            sample_batch = next(iter(self.train_loader))
            input_dim = sample_batch[0].shape[-1]  # Last dimension is features
            trainer.build_models(input_dim)
            
            # Train model with early stopping and pruning
            best_metrics = self.train_with_pruning(
                trainer, trial, params['batch_size']
            )
            
            # Calculate objectives
            training_time = time.time() - trial_start_time
            model_size = self.calculate_model_size(trainer.hybrid_model)
            
            # Store trial results
            trial_result = {
                'trial_number': trial.number,
                'params': params,
                'metrics': best_metrics,
                'training_time': training_time,
                'model_size': model_size,
                'timestamp': datetime.now().isoformat()
            }
            self.trial_results.append(trial_result)
            
            # Calculate objective values
            objectives = []
            for obj_name in self.config.objectives:
                if obj_name == 'accuracy':
                    objectives.append(best_metrics.get('val_class_acc', 0.0))
                elif obj_name == 'training_time':
                    objectives.append(training_time)
                elif obj_name == 'model_size':
                    objectives.append(model_size)
                elif obj_name == 'loss':
                    objectives.append(best_metrics.get('val_total_loss', float('inf')))
                elif obj_name == 'f1_score':
                    objectives.append(best_metrics.get('val_f1_score', 0.0))
                else:
                    logger.warning(f"Unknown objective: {obj_name}")
                    objectives.append(0.0)
            
            # Log trial results
            logger.info(f"Trial {trial.number} completed:")
            logger.info(f"  Objectives: {dict(zip(self.config.objectives, objectives))}")
            logger.info(f"  Training time: {training_time:.1f}s")
            logger.info(f"  Model size: {model_size:.2f}MB")
            
            return objectives if len(objectives) > 1 else objectives[0]
            
        except optuna.TrialPruned:
            logger.info(f"Trial {trial.number} pruned")
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            # Return worst possible values for failed trials
            if len(self.config.objectives) > 1:
                return [0.0 if obj in ['accuracy', 'f1_score'] else float('inf') 
                       for obj in self.config.objectives]
            else:
                return 0.0 if self.config.objectives[0] in ['accuracy', 'f1_score'] else float('inf')
    
    def train_with_pruning(
        self,
        trainer: IntegratedCNNLSTMTrainer,
        trial: optuna.Trial,
        batch_size: int
    ) -> Dict[str, float]:
        """Train model with pruning support."""
        
        best_val_loss = float('inf')
        best_metrics = {}
        patience_counter = 0
        
        # Update data loaders with new batch size
        if batch_size != self.train_loader.batch_size:
            # Recreate data loaders with new batch size
            train_dataset = self.train_loader.dataset
            val_dataset = self.val_loader.dataset
            
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=torch.cuda.is_available()
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=torch.cuda.is_available()
            )
        else:
            train_loader = self.train_loader
            val_loader = self.val_loader
        
        # Setup optimizer
        optimizer = self.create_optimizer(trainer.hybrid_model, trainer.config)
        scheduler = self.create_scheduler(optimizer, trainer.config)
        
        for epoch in range(self.config.max_epochs_per_trial):
            # Training epoch
            train_metrics = trainer._train_epoch_integrated(train_loader, optimizer)
            
            # Validation epoch
            val_metrics = trainer._validate_epoch_integrated(val_loader)
            
            # Update scheduler
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['total_loss'])
                else:
                    scheduler.step()
            
            # Check for improvement
            val_loss = val_metrics['total_loss']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = {
                    'val_total_loss': val_loss,
                    'val_class_loss': val_metrics['classification_loss'],
                    'val_reg_loss': val_metrics['regression_loss'],
                    'val_class_acc': val_metrics['classification_accuracy'],
                    'val_reg_mse': val_metrics['regression_mse'],
                    'train_total_loss': train_metrics['total_loss'],
                    'train_class_acc': train_metrics['classification_accuracy'],
                    'train_reg_mse': train_metrics['regression_mse']
                }
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Report intermediate value for pruning (only for single-objective)
            if len(self.config.objectives) == 1:
                trial.report(val_metrics['classification_accuracy'], epoch)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        return best_metrics
    
    def create_optimizer(self, model: nn.Module, config: HybridModelConfig) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        
        if config.optimizer_type == 'adam':
            return torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer_type == 'adamw':
            return torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer_type == 'sgd':
            return torch.optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=0.9
            )
        else:
            return torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
    
    def create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        config: HybridModelConfig
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        
        if not hasattr(config, 'scheduler_type') or config.scheduler_type == 'none':
            return None
        
        if config.scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.max_epochs_per_trial
            )
        elif config.scheduler_type == 'step':
            step_size = getattr(config, 'step_size', 20)
            gamma = getattr(config, 'gamma', 0.5)
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        elif config.scheduler_type == 'exponential':
            gamma = getattr(config, 'gamma', 0.95)
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=gamma
            )
        else:
            return None
    
    def calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def optimize(self) -> optuna.Study:
        """Run the hyperparameter optimization."""
        
        logger.info("Starting hyperparameter optimization...")
        logger.info(f"Configuration: {self.config.n_trials} trials, "
                   f"{self.config.max_epochs_per_trial} max epochs per trial")
        
        self.optimization_start_time = time.time()
        
        # Create study
        self.study = self.create_study()
        
        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            show_progress_bar=True
        )
        
        optimization_time = time.time() - self.optimization_start_time
        
        logger.info(f"Optimization completed in {optimization_time:.1f} seconds")
        logger.info(f"Total trials: {len(self.study.trials)}")
        
        # Extract Pareto front for multi-objective optimization
        if len(self.config.objectives) > 1:
            self.pareto_front = self.study.best_trials
        else:
            self.pareto_front = [self.study.best_trial] if self.study.best_trial else []
        
        logger.info(f"Pareto front size: {len(self.pareto_front)}")
        
        # Save results
        self.save_results()
        
        return self.study
    
    def save_results(self):
        """Save optimization results."""
        
        logger.info("Saving optimization results...")
        
        # Save study
        if self.config.save_study:
            study_path = self.save_dir / "optuna_study.pkl"
            with open(study_path, 'wb') as f:
                pickle.dump(self.study, f)
            logger.info(f"Study saved to: {study_path}")
        
        # Save best configurations
        best_configs = []
        for i, trial in enumerate(self.pareto_front[:10]):  # Top 10
            config = {
                'rank': i + 1,
                'trial_number': trial.number,
                'objectives': dict(zip(self.config.objectives, trial.values)),
                'params': trial.params,
                'state': trial.state.name
            }
            best_configs.append(config)
        
        best_configs_path = self.save_dir / "best_configurations.json"
        with open(best_configs_path, 'w') as f:
            json.dump(best_configs, f, indent=2)
        logger.info(f"Best configurations saved to: {best_configs_path}")
        
        # Save detailed analysis
        analysis = {
            'optimization_summary': {
                'total_trials': len(self.study.trials),
                'completed_trials': len([t for t in self.study.trials if t.state.name == 'COMPLETE']),
                'pruned_trials': len([t for t in self.study.trials if t.state.name == 'PRUNED']),
                'failed_trials': len([t for t in self.study.trials if t.state.name == 'FAIL']),
                'optimization_time': time.time() - self.optimization_start_time,
                'pareto_front_size': len(self.pareto_front)
            },
            'objectives': self.config.objectives,
            'best_configurations': best_configs,
            'trial_results': self.trial_results
        }
        
        analysis_path = self.save_dir / "optimization_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        logger.info(f"Analysis saved to: {analysis_path}")
        
        # Save plots
        if self.config.save_plots:
            self.save_plots()
    
    def save_plots(self):
        """Save optimization plots."""
        
        plots_dir = self.save_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        try:
            # Optimization history
            if len(self.config.objectives) == 1:
                fig = plot_optimization_history(self.study)
                fig.write_html(str(plots_dir / "optimization_history.html"))
            
            # Parameter importances
            if len(self.study.trials) > 10:
                for i, objective in enumerate(self.config.objectives):
                    if len(self.config.objectives) > 1:
                        fig = plot_param_importances(self.study, target=lambda t: t.values[i])
                        fig.write_html(str(plots_dir / f"param_importances_{objective}.html"))
                    else:
                        fig = plot_param_importances(self.study)
                        fig.write_html(str(plots_dir / "param_importances.html"))
            
            # Pareto front (for multi-objective)
            if len(self.config.objectives) > 1:
                fig = plot_pareto_front(self.study)
                fig.write_html(str(plots_dir / "pareto_front.html"))
            
            # Parameter slice plots
            if len(self.study.trials) > 10:
                important_params = ['learning_rate', 'cnn_num_filters', 'lstm_hidden_dim', 'dropout_rate']
                for param in important_params:
                    if any(param in trial.params for trial in self.study.trials):
                        fig = plot_slice(self.study, params=[param])
                        fig.write_html(str(plots_dir / f"slice_{param}.html"))
            
            logger.info(f"Plots saved to: {plots_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to save some plots: {e}")
    
    def retrain_best_models(self, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrain the best models with full training."""
        
        logger.info(f"Retraining top {top_k} models with full training...")
        
        retrained_models = []
        
        for i, trial in enumerate(self.pareto_front[:top_k]):
            logger.info(f"Retraining model {i+1}/{top_k} (Trial {trial.number})...")
            
            try:
                # Create model configuration
                model_config = self.create_model_config(trial.params)
                
                # Create trainer
                model_save_dir = self.save_dir / f"retrained_model_{i+1}_trial_{trial.number}"
                trainer = IntegratedCNNLSTMTrainer(
                    config=model_config,
                    save_dir=model_save_dir,
                    device=self.device
                )
                
                # Build models
                sample_batch = next(iter(self.train_loader))
                input_dim = sample_batch[0].shape[-1]
                trainer.build_models(input_dim)
                
                # Train with full epochs
                training_results = trainer.train_integrated_model(
                    self.train_loader,
                    self.val_loader,
                    num_epochs=self.config.retrain_full_epochs,
                    early_stopping_patience=50
                )
                
                # Evaluate on test set
                test_results = trainer._validate_epoch_integrated(self.test_loader)
                
                # Save model
                model_path = model_save_dir / "best_integrated.pth"
                torch.save({
                    'model_state_dict': trainer.hybrid_model.state_dict(),
                    'config': asdict(model_config),
                    'trial_params': trial.params,
                    'training_results': training_results,
                    'test_results': test_results
                }, model_path)
                
                retrained_result = {
                    'rank': i + 1,
                    'trial_number': trial.number,
                    'model_path': str(model_path),
                    'config': asdict(model_config),
                    'trial_params': trial.params,
                    'training_results': training_results,
                    'test_results': test_results,
                    'objectives': dict(zip(self.config.objectives, trial.values))
                }
                
                retrained_models.append(retrained_result)
                
                logger.info(f"Model {i+1} retrained successfully:")
                logger.info(f"  Test accuracy: {test_results.get('classification_accuracy', 'N/A')}")
                logger.info(f"  Test MSE: {test_results.get('regression_mse', 'N/A')}")
                logger.info(f"  Model saved to: {model_path}")
                
            except Exception as e:
                logger.error(f"Failed to retrain model {i+1}: {e}")
                continue
        
        # Save retrained models results
        retrained_results_path = self.save_dir / "retrained_models_results.json"
        with open(retrained_results_path, 'w') as f:
            json.dump(retrained_models, f, indent=2, default=str)
        
        logger.info(f"Retrained {len(retrained_models)} models successfully")
        logger.info(f"Results saved to: {retrained_results_path}")
        
        return retrained_models


def create_optimization_config(
    n_trials: int = 1000,
    max_epochs_per_trial: int = 50,
    objectives: List[str] = None,
    save_dir: str = "hyperopt_results",
    **kwargs
) -> OptimizationConfig:
    """Create optimization configuration."""
    
    if objectives is None:
        objectives = ['accuracy', 'training_time', 'model_size']
    
    return OptimizationConfig(
        n_trials=n_trials,
        max_epochs_per_trial=max_epochs_per_trial,
        objectives=objectives,
        save_dir=save_dir,
        **kwargs
    )


def run_hyperparameter_optimization(
    symbols: List[str],
    start_date: str,
    end_date: str,
    n_trials: int = 1000,
    results_dir: str = "hyperopt_results_task_5_5"
) -> Dict[str, Any]:
    """
    Run complete hyperparameter optimization pipeline.
    
    This is the main entry point for task 5.5 implementation.
    """
    
    logger.info("Starting CNN+LSTM hyperparameter optimization - Task 5.5")
    
    # Create optimization configuration
    config = create_optimization_config(
        n_trials=n_trials,
        max_epochs_per_trial=50,
        objectives=['accuracy', 'training_time', 'model_size'],
        save_dir=results_dir,
        timeout=48 * 3600,  # 48 hours
        retrain_top_k=3
    )
    
    # Prepare data
    from .train_integrated_cnn_lstm import IntegratedCNNLSTMTrainer
    from .hybrid_model import create_hybrid_config
    
    temp_config = create_hybrid_config(input_dim=11, sequence_length=60)
    temp_trainer = IntegratedCNNLSTMTrainer(config=temp_config)
    
    train_loader, val_loader, test_loader = temp_trainer.prepare_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframes=["1min", "5min", "15min"],
        sequence_length=60,
        batch_size=32
    )
    
    # Create and run optimizer
    optimizer = MultiObjectiveOptimizer(
        config=config,
        data_loaders=(train_loader, val_loader, test_loader)
    )
    
    study = optimizer.optimize()
    
    # Retrain best models
    retrained_models = optimizer.retrain_best_models(top_k=config.retrain_top_k)
    
    return {
        'study': study,
        'best_trial': optimizer.pareto_front[0] if optimizer.pareto_front else None,
        'pareto_front': optimizer.pareto_front,
        'retrained_models': retrained_models,
        'optimization_time': time.time() - optimizer.optimization_start_time,
        'results_dir': results_dir
    }