"""
Advanced Hyperparameter Optimization for CNN+LSTM Models - Task 5.5

This module implements comprehensive hyperparameter optimization using Optuna
with multi-objective optimization, early pruning, and automated model retraining.

Requirements: 3.4, 9.1
"""

import os
import sys
import json
import time
import logging
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler, CmaEsSampler

# Optional integration - only import if available
try:
    from optuna.integration import PyTorchLightningPruningCallback
    HAS_PYTORCH_LIGHTNING_INTEGRATION = True
except ImportError:
    PyTorchLightningPruningCallback = None
    HAS_PYTORCH_LIGHTNING_INTEGRATION = False
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.ml.hybrid_model import CNNLSTMHybridModel, HybridModelConfig, create_hybrid_config
from src.ml.train_integrated_cnn_lstm import IntegratedCNNLSTMTrainer
from data.pipeline import create_data_loaders


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hyperparameter_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization"""
    # Optimization settings
    n_trials: int = 1000
    timeout: int = 48 * 3600  # 48 hours in seconds
    n_jobs: int = 1  # Parallel jobs (set to 1 for GPU usage)
    
    # Pruning settings
    pruning_warmup_steps: int = 10
    pruning_interval_steps: int = 5
    
    # Multi-objective settings
    objectives: List[str] = None  # ['accuracy', 'training_time', 'model_size']
    objective_weights: Dict[str, float] = None
    
    # Model configuration
    num_classes: int = 4  # Regime detection classes
    regression_targets: int = 2  # Price and volatility
    
    # Search space settings
    search_space_config: Dict[str, Any] = None
    
    # Training settings for trials
    max_epochs_per_trial: int = 50
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    
    # Resource constraints
    max_model_size_mb: float = 500.0  # Maximum model size in MB
    max_training_time_minutes: float = 120.0  # Maximum training time per trial
    
    # Output settings
    save_dir: str = "hyperopt_results"
    study_name: str = "cnn_lstm_optimization"
    storage_url: str = None  # For distributed optimization
    
    def __post_init__(self):
        if self.objectives is None:
            self.objectives = ['accuracy', 'training_time', 'model_size']
        
        if self.objective_weights is None:
            self.objective_weights = {
                'accuracy': 0.6,
                'training_time': 0.2,
                'model_size': 0.2
            }
        
        if self.search_space_config is None:
            self.search_space_config = self._get_default_search_space()
    
    def _get_default_search_space(self) -> Dict[str, Any]:
        """Get default hyperparameter search space"""
        return {
            # Learning rate optimization
            'learning_rate': {
                'type': 'loguniform',
                'low': 1e-5,
                'high': 1e-2
            },
            
            # CNN architecture
            'cnn_num_filters': {
                'type': 'categorical',
                'choices': [32, 64, 128, 256]
            },
            'cnn_filter_sizes': {
                'type': 'categorical',
                'choices': ['3_5_7', '3_5_7_11', '2_3_5_8', '3_7_11_15']
            },
            'cnn_attention_heads': {
                'type': 'categorical',
                'choices': [4, 8, 12, 16]
            },
            
            # LSTM architecture
            'lstm_hidden_dim': {
                'type': 'categorical',
                'choices': [64, 128, 256, 512]
            },
            'lstm_num_layers': {
                'type': 'int',
                'low': 1,
                'high': 4
            },
            'lstm_bidirectional': {
                'type': 'categorical',
                'choices': [True, False]
            },
            
            # Fusion and ensemble
            'feature_fusion_dim': {
                'type': 'categorical',
                'choices': [128, 256, 512]
            },
            'num_ensemble_models': {
                'type': 'int',
                'low': 3,
                'high': 7
            },
            
            # Regularization
            'dropout_rate': {
                'type': 'uniform',
                'low': 0.1,
                'high': 0.5
            },
            
            # Training dynamics
            'batch_size': {
                'type': 'categorical',
                'choices': [16, 32, 64, 128]
            },
            'optimizer': {
                'type': 'categorical',
                'choices': ['adam', 'adamw', 'rmsprop', 'sgd']
            },
            'scheduler': {
                'type': 'categorical',
                'choices': ['cosine', 'step', 'exponential', 'plateau']
            },
            
            # Multi-task weights
            'classification_weight': {
                'type': 'uniform',
                'low': 0.2,
                'high': 0.8
            },
            'regression_weight': {
                'type': 'uniform',
                'low': 0.2,
                'high': 0.8
            }
        }


class MultiObjectiveOptimizer:
    """
    Multi-objective hyperparameter optimizer for CNN+LSTM models.
    
    Implements task 5.5 requirements:
    - Optuna-based optimization with 1000+ trials
    - Early pruning for efficiency
    - Multi-objective optimization (accuracy, training time, model size)
    - Best configuration saving and model retraining
    """
    
    def __init__(
        self,
        config: OptimizationConfig,
        data_loaders: Tuple[DataLoader, DataLoader, DataLoader],
        device: Optional[str] = None
    ):
        """Initialize the multi-objective optimizer"""
        self.config = config
        self.train_loader, self.val_loader, self.test_loader = data_loaders
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Create output directory
        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize study
        self.study = None
        self.best_trials = []
        self.optimization_history = []
        
        # Performance tracking
        self.trial_results = []
        self.pareto_front = []
        
        logger.info(f"Multi-objective optimizer initialized with {self.config.n_trials} trials")
        logger.info(f"Objectives: {self.config.objectives}")
        logger.info(f"Device: {self.device}")
    
    def create_study(self) -> optuna.Study:
        """Create Optuna study with multi-objective optimization"""
        
        # Setup sampler
        sampler = TPESampler(
            n_startup_trials=50,
            n_ei_candidates=24,
            multivariate=True,
            group=True,
            warn_independent_sampling=False
        )
        
        # Create study based on number of objectives
        if len(self.config.objectives) == 1:
            # Single objective - can use pruning
            pruner = HyperbandPruner(
                min_resource=self.config.pruning_warmup_steps,
                max_resource=self.config.max_epochs_per_trial,
                reduction_factor=3
            )
            
            direction = 'maximize' if self.config.objectives[0] in ['accuracy', 'f1_score', 'precision', 'recall'] else 'minimize'
            
            study = optuna.create_study(
                direction=direction,
                sampler=sampler,
                pruner=pruner,
                study_name=self.config.study_name,
                storage=self.config.storage_url,
                load_if_exists=True
            )
        else:
            # Multi-objective - no pruning supported
            directions = []
            for objective in self.config.objectives:
                if objective in ['accuracy', 'f1_score', 'precision', 'recall']:
                    directions.append('maximize')
                else:  # training_time, model_size, loss
                    directions.append('minimize')
            
            study = optuna.create_study(
                directions=directions,
                sampler=sampler,
                study_name=self.config.study_name,
                storage=self.config.storage_url,
                load_if_exists=True
            )
        
        logger.info(f"Created multi-objective study with directions: {directions}")
        return study
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial"""
        params = {}
        
        for param_name, param_config in self.config.search_space_config.items():
            param_type = param_config['type']
            
            if param_type == 'uniform':
                params[param_name] = trial.suggest_uniform(
                    param_name, param_config['low'], param_config['high']
                )
            elif param_type == 'loguniform':
                params[param_name] = trial.suggest_loguniform(
                    param_name, param_config['low'], param_config['high']
                )
            elif param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, param_config['low'], param_config['high']
                )
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config['choices']
                )
            else:
                logger.warning(f"Unknown parameter type: {param_type} for {param_name}")
        
        return params
    
    def _parse_filter_sizes(self, filter_sizes_str: str) -> List[int]:
        """Convert string representation back to list of integers"""
        return [int(x) for x in filter_sizes_str.split('_')]
    
    def create_model_config(self, params: Dict[str, Any]) -> HybridModelConfig:
        """Create model configuration from hyperparameters"""
        
        # Get input dimension from data loader
        sample_batch = next(iter(self.train_loader))
        input_dim = sample_batch[0].shape[1]  # Assuming (batch, features, sequence)
        
        config = HybridModelConfig(
            # Basic configuration from ModelConfig
            model_type="hybrid_cnn_lstm",
            input_dim=input_dim,
            output_dim=self.config.num_classes + self.config.regression_targets,
            hidden_dims=[params['feature_fusion_dim']],
            sequence_length=60,  # Fixed for now
            device=str(self.device),
            
            # CNN configuration
            cnn_filter_sizes=self._parse_filter_sizes(params['cnn_filter_sizes']),
            cnn_num_filters=params['cnn_num_filters'],
            cnn_use_attention=True,
            cnn_attention_heads=params['cnn_attention_heads'],
            
            # LSTM configuration
            lstm_hidden_dim=params['lstm_hidden_dim'],
            lstm_num_layers=params['lstm_num_layers'],
            lstm_bidirectional=params['lstm_bidirectional'],
            lstm_use_attention=True,
            lstm_use_skip_connections=True,
            
            # Fusion configuration
            feature_fusion_dim=params['feature_fusion_dim'],
            
            # Ensemble configuration
            num_ensemble_models=params['num_ensemble_models'],
            
            # Multi-task configuration
            num_classes=4,  # Updated for regime detection
            regression_targets=2,  # Price and volatility
            classification_weight=params['classification_weight'],
            regression_weight=params['regression_weight'],
            
            # Training configuration
            learning_rate=params['learning_rate'],
            batch_size=params['batch_size'],
            dropout_rate=params['dropout_rate'],
            
            # Regularization
            use_monte_carlo_dropout=True,
            mc_dropout_samples=50  # Reduced for faster optimization
        )
        
        return config
    
    def objective_function(self, trial: optuna.Trial) -> List[float]:
        """
        Multi-objective function for hyperparameter optimization.
        
        Returns:
            List of objective values [accuracy, training_time, model_size]
        """
        trial_start_time = time.time()
        
        try:
            # Suggest hyperparameters
            params = self.suggest_hyperparameters(trial)
            
            # Create model configuration
            model_config = self.create_model_config(params)
            
            # Create trainer
            trainer = IntegratedCNNLSTMTrainer(
                config=model_config,
                save_dir=str(self.save_dir / f"trial_{trial.number}"),
                device=str(self.device)
            )
            
            # Build model
            trainer.build_models(model_config.input_dim)
            
            # Calculate model size
            model_size_mb = self._calculate_model_size(trainer.hybrid_model)
            
            # Check model size constraint
            if model_size_mb > self.config.max_model_size_mb:
                logger.warning(f"Trial {trial.number}: Model too large ({model_size_mb:.2f} MB)")
                raise optuna.TrialPruned()
            
            # Train model with limited epochs
            training_start = time.time()
            
            try:
                training_results = trainer.train_integrated_model(
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    num_epochs=self.config.max_epochs_per_trial,
                    early_stopping_patience=self.config.early_stopping_patience
                )
            except Exception as training_error:
                logger.error(f"Trial {trial.number}: Training failed: {training_error}")
                training_results = None
            
            training_time = (time.time() - training_start) / 60.0  # Convert to minutes
            
            # Check training time constraint
            if training_time > self.config.max_training_time_minutes:
                logger.warning(f"Trial {trial.number}: Training too slow ({training_time:.2f} min)")
                raise optuna.TrialPruned()
            
            # Extract performance metrics - handle different return formats
            if training_results is not None and isinstance(training_results, dict) and 'history' in training_results:
                history = training_results['history']
                if history:
                    best_val_accuracy = max([h.get('val_class_acc', 0.0) for h in history])
                    best_val_loss = min([h.get('val_total_loss', 1.0) for h in history])
                    val_reg_mse = min([h.get('val_reg_mse', 1.0) for h in history])
                    
                    # For multi-objective optimization, we can't use intermediate reporting
                    # Just log progress instead
                    if len(history) > 0:
                        final_accuracy = (
                            model_config.classification_weight * history[-1].get('val_class_acc', 0.0) +
                            model_config.regression_weight * (1.0 / (1.0 + history[-1].get('val_reg_mse', 1.0)))
                        )
                        logger.debug(f"Trial {trial.number}: Final accuracy = {final_accuracy:.4f}")
                else:
                    # No history available
                    best_val_accuracy = 0.0
                    val_reg_mse = 1.0
            else:
                # Training failed or returned unexpected format
                logger.warning(f"Trial {trial.number}: Training failed or returned unexpected format")
                best_val_accuracy = 0.0
                val_reg_mse = 1.0
                
                # For multi-objective optimization, we can't report intermediate values
                # Just continue with poor performance values
            
            # Normalize regression MSE to [0, 1] range for combination
            normalized_reg_performance = 1.0 / (1.0 + val_reg_mse)
            
            # Composite accuracy combining classification and regression
            composite_accuracy = (
                model_config.classification_weight * best_val_accuracy +
                model_config.regression_weight * normalized_reg_performance
            )
            
            # Store trial results
            trial_result = {
                'trial_number': trial.number,
                'params': params,
                'composite_accuracy': composite_accuracy,
                'classification_accuracy': best_val_accuracy,
                'regression_mse': val_reg_mse,
                'training_time_minutes': training_time,
                'model_size_mb': model_size_mb,
                'total_time_minutes': (time.time() - trial_start_time) / 60.0,
                'converged_epoch': len(training_results['history'])
            }
            
            self.trial_results.append(trial_result)
            
            # Save trial results
            self._save_trial_result(trial_result)
            
            logger.info(f"Trial {trial.number} completed: "
                       f"Accuracy={composite_accuracy:.4f}, "
                       f"Time={training_time:.2f}min, "
                       f"Size={model_size_mb:.2f}MB")
            
            # Return objectives (accuracy to maximize, time and size to minimize)
            objectives = []
            for obj_name in self.config.objectives:
                if obj_name == 'accuracy':
                    objectives.append(composite_accuracy)
                elif obj_name == 'training_time':
                    objectives.append(training_time)
                elif obj_name == 'model_size':
                    objectives.append(model_size_mb)
                else:
                    logger.warning(f"Unknown objective: {obj_name}")
                    objectives.append(0.0)
            
            return objectives
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Return poor performance for failed trials
            objectives = []
            for obj_name in self.config.objectives:
                if obj_name == 'accuracy':
                    objectives.append(0.0)
                elif obj_name == 'training_time':
                    objectives.append(self.config.max_training_time_minutes)
                elif obj_name == 'model_size':
                    objectives.append(self.config.max_model_size_mb)
                else:
                    objectives.append(0.0)
            return objectives
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb
    
    def _save_trial_result(self, trial_result: Dict[str, Any]):
        """Save individual trial result"""
        trial_file = self.save_dir / f"trial_{trial_result['trial_number']}_result.json"
        
        # Convert numpy types to Python types for JSON serialization
        serializable_result = {}
        for key, value in trial_result.items():
            if isinstance(value, (np.integer, np.floating)):
                serializable_result[key] = value.item()
            elif isinstance(value, np.ndarray):
                serializable_result[key] = value.tolist()
            else:
                serializable_result[key] = value
        
        with open(trial_file, 'w') as f:
            json.dump(serializable_result, f, indent=2)
    
    def optimize(self) -> optuna.Study:
        """
        Run the complete hyperparameter optimization process.
        
        Returns:
            Completed Optuna study with results
        """
        logger.info(f"Starting hyperparameter optimization with {self.config.n_trials} trials...")
        
        # Create study
        self.study = self.create_study()
        
        # Run optimization
        start_time = time.time()
        
        try:
            self.study.optimize(
                self.objective_function,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout,
                n_jobs=self.config.n_jobs,
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
        
        optimization_time = (time.time() - start_time) / 3600.0  # Convert to hours
        
        logger.info(f"Optimization completed in {optimization_time:.2f} hours")
        logger.info(f"Total trials: {len(self.study.trials)}")
        logger.info(f"Completed trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        logger.info(f"Pruned trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        
        # Analyze results
        self._analyze_results()
        
        # Save study
        self._save_study()
        
        return self.study
    
    def _analyze_results(self):
        """Analyze optimization results and find Pareto front"""
        logger.info("Analyzing optimization results...")
        
        # Extract completed trials
        completed_trials = [
            trial for trial in self.study.trials 
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        
        if not completed_trials:
            logger.warning("No completed trials found")
            return
        
        # Find Pareto front for multi-objective optimization
        if len(self.config.objectives) > 1:
            self.pareto_front = self._find_pareto_front(completed_trials)
            logger.info(f"Found {len(self.pareto_front)} solutions on Pareto front")
        else:
            # Single objective - just get the best trial
            self.pareto_front = [self.study.best_trial]
        
        # Save best configurations
        self._save_best_configurations()
        
        # Generate analysis plots
        self._generate_analysis_plots()
    
    def _find_pareto_front(self, trials: List[optuna.Trial]) -> List[optuna.Trial]:
        """Find Pareto front for multi-objective optimization"""
        pareto_trials = []
        
        for trial in trials:
            is_dominated = False
            
            for other_trial in trials:
                if trial == other_trial:
                    continue
                
                # Check if other_trial dominates trial
                dominates = True
                for i, obj_name in enumerate(self.config.objectives):
                    if obj_name in ['accuracy', 'f1_score', 'precision', 'recall']:
                        # Maximize objective
                        if other_trial.values[i] <= trial.values[i]:
                            dominates = False
                            break
                    else:
                        # Minimize objective
                        if other_trial.values[i] >= trial.values[i]:
                            dominates = False
                            break
                
                if dominates:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_trials.append(trial)
        
        return pareto_trials
    
    def _save_best_configurations(self):
        """Save best hyperparameter configurations"""
        logger.info("Saving best configurations...")
        
        best_configs = []
        
        for i, trial in enumerate(self.pareto_front):
            config_data = {
                'trial_number': trial.number,
                'params': trial.params,
                'values': trial.values,
                'objectives': dict(zip(self.config.objectives, trial.values)),
                'rank': i + 1
            }
            best_configs.append(config_data)
        
        # Save best configurations
        best_configs_file = self.save_dir / "best_configurations.json"
        with open(best_configs_file, 'w') as f:
            json.dump(best_configs, f, indent=2)
        
        # Save detailed analysis
        analysis_file = self.save_dir / "optimization_analysis.json"
        analysis_data = {
            'optimization_config': asdict(self.config),
            'total_trials': len(self.study.trials),
            'completed_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'pruned_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'pareto_front_size': len(self.pareto_front),
            'best_configurations': best_configs,
            'trial_results': self.trial_results
        }
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        logger.info(f"Saved {len(best_configs)} best configurations")
    
    def _generate_analysis_plots(self):
        """Generate analysis plots for optimization results"""
        logger.info("Generating analysis plots...")
        
        try:
            # Create plots directory
            plots_dir = self.save_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Plot optimization history
            self._plot_optimization_history(plots_dir)
            
            # Plot parameter importance
            self._plot_parameter_importance(plots_dir)
            
            # Plot Pareto front (if multi-objective)
            if len(self.config.objectives) > 1:
                self._plot_pareto_front(plots_dir)
            
            # Plot hyperparameter correlations
            self._plot_hyperparameter_correlations(plots_dir)
            
            logger.info("Analysis plots generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")
    
    def _plot_optimization_history(self, plots_dir: Path):
        """Plot optimization history"""
        fig, axes = plt.subplots(1, len(self.config.objectives), figsize=(6*len(self.config.objectives), 5))
        if len(self.config.objectives) == 1:
            axes = [axes]
        
        for i, obj_name in enumerate(self.config.objectives):
            values = []
            trial_numbers = []
            
            for trial in self.study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    trial_numbers.append(trial.number)
                    values.append(trial.values[i])
            
            axes[i].plot(trial_numbers, values, 'b-', alpha=0.6)
            axes[i].set_xlabel('Trial Number')
            axes[i].set_ylabel(f'{obj_name.title()}')
            axes[i].set_title(f'{obj_name.title()} vs Trial Number')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "optimization_history.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_importance(self, plots_dir: Path):
        """Plot parameter importance"""
        try:
            for i, obj_name in enumerate(self.config.objectives):
                importance = optuna.importance.get_param_importances(
                    self.study, target=lambda t: t.values[i]
                )
                
                if importance:
                    params = list(importance.keys())
                    values = list(importance.values())
                    
                    plt.figure(figsize=(10, 6))
                    plt.barh(params, values)
                    plt.xlabel('Importance')
                    plt.title(f'Parameter Importance for {obj_name.title()}')
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"parameter_importance_{obj_name}.png", 
                               dpi=300, bbox_inches='tight')
                    plt.close()
        except Exception as e:
            logger.warning(f"Could not generate parameter importance plots: {e}")
    
    def _plot_pareto_front(self, plots_dir: Path):
        """Plot Pareto front for multi-objective optimization"""
        if len(self.config.objectives) < 2:
            return
        
        # Extract values for all completed trials
        all_values = []
        pareto_values = []
        
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                all_values.append(trial.values)
        
        for trial in self.pareto_front:
            pareto_values.append(trial.values)
        
        all_values = np.array(all_values)
        pareto_values = np.array(pareto_values)
        
        # Plot 2D Pareto front
        if len(self.config.objectives) == 2:
            plt.figure(figsize=(10, 8))
            plt.scatter(all_values[:, 0], all_values[:, 1], alpha=0.6, label='All Trials')
            plt.scatter(pareto_values[:, 0], pareto_values[:, 1], 
                       color='red', s=100, label='Pareto Front')
            plt.xlabel(self.config.objectives[0].title())
            plt.ylabel(self.config.objectives[1].title())
            plt.title('Pareto Front')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / "pareto_front_2d.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot 3D Pareto front
        elif len(self.config.objectives) == 3:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(all_values[:, 0], all_values[:, 1], all_values[:, 2], 
                      alpha=0.6, label='All Trials')
            ax.scatter(pareto_values[:, 0], pareto_values[:, 1], pareto_values[:, 2], 
                      color='red', s=100, label='Pareto Front')
            ax.set_xlabel(self.config.objectives[0].title())
            ax.set_ylabel(self.config.objectives[1].title())
            ax.set_zlabel(self.config.objectives[2].title())
            ax.set_title('3D Pareto Front')
            ax.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "pareto_front_3d.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_hyperparameter_correlations(self, plots_dir: Path):
        """Plot correlations between hyperparameters and objectives"""
        try:
            # Extract data for completed trials
            trial_data = []
            
            for trial in self.study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    row = trial.params.copy()
                    for i, obj_name in enumerate(self.config.objectives):
                        row[obj_name] = trial.values[i]
                    trial_data.append(row)
            
            if not trial_data:
                return
            
            df = pd.DataFrame(trial_data)
            
            # Convert categorical variables to numeric
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.Categorical(df[col]).codes
                    except:
                        pass
            
            # Calculate correlation matrix
            corr_matrix = df.corr()
            
            # Plot correlation heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f')
            plt.title('Hyperparameter-Objective Correlations')
            plt.tight_layout()
            plt.savefig(plots_dir / "hyperparameter_correlations.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not generate correlation plots: {e}")
    
    def _save_study(self):
        """Save the complete study"""
        study_file = self.save_dir / "optuna_study.pkl"
        
        with open(study_file, 'wb') as f:
            pickle.dump(self.study, f)
        
        logger.info(f"Study saved to {study_file}")
    
    def retrain_best_models(self, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrain the top-k best models with full training epochs.
        
        Args:
            top_k: Number of best configurations to retrain
            
        Returns:
            List of retrained model results
        """
        logger.info(f"Retraining top {top_k} models with full training...")
        
        if not self.pareto_front:
            logger.error("No best configurations found. Run optimization first.")
            return []
        
        retrained_models = []
        
        # Select top-k configurations based on composite score
        top_configs = self.pareto_front[:top_k]
        
        for i, trial in enumerate(top_configs):
            logger.info(f"Retraining model {i+1}/{top_k} (Trial {trial.number})...")
            
            try:
                # Create model configuration
                model_config = self.create_model_config(trial.params)
                
                # Create trainer with full training settings
                trainer = IntegratedCNNLSTMTrainer(
                    config=model_config,
                    save_dir=str(self.save_dir / f"retrained_model_{i+1}"),
                    device=str(self.device)
                )
                
                # Build model
                trainer.build_models(model_config.input_dim)
                
                # Train with full epochs (200+)
                training_results = trainer.train_integrated_model(
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    num_epochs=200,  # Full training
                    early_stopping_patience=50
                )
                
                # Evaluate on test set
                test_results = trainer.evaluate_model(self.test_loader)
                
                retrained_result = {
                    'rank': i + 1,
                    'trial_number': trial.number,
                    'hyperparameters': trial.params,
                    'optimization_objectives': dict(zip(self.config.objectives, trial.values)),
                    'training_results': training_results,
                    'test_results': test_results,
                    'model_path': str(self.save_dir / f"retrained_model_{i+1}")
                }
                
                retrained_models.append(retrained_result)
                
                logger.info(f"Model {i+1} retrained successfully")
                
            except Exception as e:
                logger.error(f"Failed to retrain model {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save retrained model results
        retrained_file = self.save_dir / "retrained_models_results.json"
        
        # Convert to serializable format
        serializable_results = []
        for result in retrained_models:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_result[key] = value.item()
                elif isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)
        
        with open(retrained_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Retrained {len(retrained_models)} models successfully")
        
        return retrained_models


def create_optimization_config(
    n_trials: int = 1000,
    max_epochs_per_trial: int = 50,
    objectives: List[str] = None,
    save_dir: str = "hyperopt_results"
) -> OptimizationConfig:
    """Create optimization configuration with sensible defaults"""
    
    if objectives is None:
        objectives = ['accuracy', 'training_time', 'model_size']
    
    return OptimizationConfig(
        n_trials=n_trials,
        max_epochs_per_trial=max_epochs_per_trial,
        objectives=objectives,
        save_dir=save_dir,
        timeout=48 * 3600,  # 48 hours
        n_jobs=1,  # Single job for GPU usage
        pruning_warmup_steps=10,
        pruning_interval_steps=5,
        early_stopping_patience=10,
        max_model_size_mb=500.0,
        max_training_time_minutes=120.0
    )


if __name__ == "__main__":
    # Example usage
    from src.ml.train_integrated_cnn_lstm import IntegratedCNNLSTMTrainer
    
    # Create sample data loaders (replace with actual data)
    logger.info("Creating sample data for optimization...")
    
    # This would be replaced with actual data loading
    sample_data = torch.randn(1000, 11, 60)  # (samples, features, sequence_length)
    sample_targets_class = torch.randint(0, 4, (1000,))  # Classification targets
    sample_targets_reg = torch.randn(1000, 2)  # Regression targets
    
    from torch.utils.data import TensorDataset, DataLoader
    
    dataset = TensorDataset(sample_data, sample_targets_class, sample_targets_reg)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create optimization configuration
    config = create_optimization_config(
        n_trials=100,  # Reduced for testing
        max_epochs_per_trial=20,
        objectives=['accuracy', 'training_time', 'model_size']
    )
    
    # Create optimizer
    optimizer = MultiObjectiveOptimizer(
        config=config,
        data_loaders=(train_loader, val_loader, test_loader)
    )
    
    # Run optimization
    study = optimizer.optimize()
    
    # Retrain best models
    best_models = optimizer.retrain_best_models(top_k=3)
    
    logger.info("Hyperparameter optimization completed!")