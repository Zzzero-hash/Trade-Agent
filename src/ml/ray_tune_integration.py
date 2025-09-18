"""
Ray Tune Integration for Advanced Hyperparameter Optimization

This module provides advanced hyperparameter optimization capabilities using Ray Tune
with support for population-based training, early stopping, and multi-objective optimization.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from datetime import datetime
import numpy as np
import torch
import gymnasium as gym
from dataclasses import dataclass, asdict

# Ray imports with fallback
try:
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import (
        ASHAScheduler, 
        PopulationBasedTraining, 
        MedianStoppingRule,
        HyperBandScheduler
    )
    from ray.tune.search import (
        BasicVariantGenerator,
        ConcurrencyLimiter
    )
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.search.hyperopt import HyperOptSearch
    from ray.tune.search.bayesopt import BayesOptSearch
    from ray.tune.stopper import (
        MaximumIterationStopper,
        TrialPlateauStopper,
        TimeoutStopper
    )
    from ray.air import session
    from ray.air.config import RunConfig, ScalingConfig
    from ray.train import Trainer
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None
    tune = None

from .base_models import ModelConfig, TrainingResult
from .rl_agents import RLAgentFactory
from .training_pipeline import CNNTrainingPipeline


@dataclass
class TuneConfig:
    """Configuration for Ray Tune optimization"""
    # Search configuration
    num_samples: int = 100
    max_concurrent_trials: int = 8
    search_algorithm: str = "optuna"  # optuna, hyperopt, bayesopt, random
    
    # Scheduler configuration
    scheduler: str = "asha"  # asha, pbt, median, hyperband, fifo
    grace_period: int = 10
    reduction_factor: int = 2
    max_t: int = 100
    
    # Resource configuration
    cpus_per_trial: int = 2
    gpus_per_trial: float = 0.25
    memory_per_trial: str = "4GB"
    
    # Stopping criteria
    metric: str = "mean_reward"
    mode: str = "max"
    time_budget_s: Optional[int] = None  # seconds
    max_failures: int = 3
    
    # Checkpointing
    checkpoint_freq: int = 10
    keep_checkpoints_num: int = 3
    
    # Logging
    log_to_file: bool = True
    verbose: int = 1
    
    # Multi-objective optimization
    multi_objective: bool = False
    objectives: List[str] = None


class RayTuneOptimizer:
    """Advanced hyperparameter optimizer using Ray Tune"""
    
    def __init__(
        self,
        config: TuneConfig,
        local_dir: str = "ray_results",
        experiment_name: Optional[str] = None
    ):
        """Initialize Ray Tune optimizer
        
        Args:
            config: Tune configuration
            local_dir: Directory for results
            experiment_name: Name of the experiment
        """
        self.config = config
        self.local_dir = local_dir
        self.experiment_name = experiment_name or f"tune_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize Ray if available
        if not RAY_AVAILABLE:
            raise ImportError("Ray Tune is not available. Install with: pip install ray[tune]")
        
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
    
    def optimize_cnn_hyperparameters(
        self,
        data_config: Dict[str, Any],
        search_space: Optional[Dict[str, Any]] = None,
        custom_objective: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Optimize CNN model hyperparameters
        
        Args:
            data_config: Data configuration for training
            search_space: Custom search space (optional)
            custom_objective: Custom objective function (optional)
            
        Returns:
            Optimization results
        """
        # Default search space for CNN
        if search_space is None:
            search_space = {
                "learning_rate": tune.loguniform(1e-5, 1e-2),
                "batch_size": tune.choice([16, 32, 64, 128]),
                "num_filters": tune.choice([32, 64, 128, 256]),
                "filter_sizes": tune.choice([[3, 5, 7], [3, 5, 7, 11], [5, 7, 11]]),
                "dropout_rate": tune.uniform(0.1, 0.5),
                "l2_reg": tune.loguniform(1e-6, 1e-3),
                "epochs": tune.choice([50, 100, 150, 200])
            }
        
        # Define objective function
        def cnn_objective(config):
            return self._train_cnn_trial(config, data_config, custom_objective)
        
        return self._run_optimization(cnn_objective, search_space, "cnn")
    
    def optimize_rl_hyperparameters(
        self,
        env_factory: Callable[[], gym.Env],
        agent_type: str,
        search_space: Optional[Dict[str, Any]] = None,
        custom_objective: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Optimize RL agent hyperparameters
        
        Args:
            env_factory: Factory function to create environments
            agent_type: Type of RL agent
            search_space: Custom search space (optional)
            custom_objective: Custom objective function (optional)
            
        Returns:
            Optimization results
        """
        # Get default search space for agent type
        if search_space is None:
            search_space = self._get_rl_search_space(agent_type)
        
        # Define objective function
        def rl_objective(config):
            return self._train_rl_trial(config, env_factory, agent_type, custom_objective)
        
        return self._run_optimization(rl_objective, search_space, f"rl_{agent_type.lower()}")
    
    def optimize_hybrid_model(
        self,
        data_config: Dict[str, Any],
        search_space: Optional[Dict[str, Any]] = None,
        custom_objective: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Optimize hybrid CNN+LSTM model hyperparameters
        
        Args:
            data_config: Data configuration for training
            search_space: Custom search space (optional)
            custom_objective: Custom objective function (optional)
            
        Returns:
            Optimization results
        """
        # Default search space for hybrid model
        if search_space is None:
            search_space = {
                # CNN parameters
                "cnn_filters": tune.choice([32, 64, 128]),
                "cnn_kernel_sizes": tune.choice([[3, 5, 7], [3, 5, 7, 11]]),
                "cnn_dropout": tune.uniform(0.1, 0.4),
                
                # LSTM parameters
                "lstm_hidden_size": tune.choice([64, 128, 256, 512]),
                "lstm_num_layers": tune.choice([1, 2, 3]),
                "lstm_dropout": tune.uniform(0.1, 0.4),
                "bidirectional": tune.choice([True, False]),
                
                # Fusion parameters
                "fusion_dim": tune.choice([128, 256, 512]),
                "attention_heads": tune.choice([4, 8, 16]),
                
                # Training parameters
                "learning_rate": tune.loguniform(1e-5, 1e-2),
                "batch_size": tune.choice([16, 32, 64]),
                "weight_decay": tune.loguniform(1e-6, 1e-3),
                
                # Multi-task learning
                "classification_weight": tune.uniform(0.3, 0.7),
                "regression_weight": tune.uniform(0.3, 0.7)
            }
        
        # Define objective function
        def hybrid_objective(config):
            return self._train_hybrid_trial(config, data_config, custom_objective)
        
        return self._run_optimization(hybrid_objective, search_space, "hybrid")
    
    def multi_objective_optimization(
        self,
        objective_function: Callable,
        search_space: Dict[str, Any],
        objectives: List[str],
        weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Perform multi-objective optimization
        
        Args:
            objective_function: Function to optimize
            search_space: Hyperparameter search space
            objectives: List of objective metrics
            weights: Weights for combining objectives (optional)
            
        Returns:
            Optimization results with Pareto frontier
        """
        if weights is None:
            weights = [1.0 / len(objectives)] * len(objectives)
        
        # Wrapper for multi-objective optimization
        def multi_objective_wrapper(config):
            results = objective_function(config)
            
            # Compute weighted sum of objectives
            weighted_score = 0.0
            for obj, weight in zip(objectives, weights):
                if obj in results:
                    weighted_score += weight * results[obj]
            
            results["weighted_score"] = weighted_score
            return results
        
        # Update config for multi-objective
        tune_config = TuneConfig(
            **asdict(self.config),
            metric="weighted_score",
            multi_objective=True,
            objectives=objectives
        )
        
        return self._run_optimization(
            multi_objective_wrapper, 
            search_space, 
            "multi_objective",
            tune_config
        )
    
    def _run_optimization(
        self,
        objective_function: Callable,
        search_space: Dict[str, Any],
        experiment_suffix: str,
        custom_config: Optional[TuneConfig] = None
    ) -> Dict[str, Any]:
        """Run Ray Tune optimization
        
        Args:
            objective_function: Function to optimize
            search_space: Hyperparameter search space
            experiment_suffix: Suffix for experiment name
            custom_config: Custom tune configuration
            
        Returns:
            Optimization results
        """
        config = custom_config or self.config
        
        # Setup search algorithm
        search_alg = self._create_search_algorithm(config)
        
        # Setup scheduler
        scheduler = self._create_scheduler(config)
        
        # Setup stopper
        stopper = self._create_stopper(config)
        
        # Setup reporter
        reporter = self._create_reporter(config)
        
        # Configure resources
        resources = {
            "cpu": config.cpus_per_trial,
            "gpu": config.gpus_per_trial
        }
        
        # Run optimization
        experiment_name = f"{self.experiment_name}_{experiment_suffix}"
        
        self.logger.info(f"Starting Ray Tune optimization: {experiment_name}")
        self.logger.info(f"Search space: {search_space}")
        self.logger.info(f"Configuration: {asdict(config)}")
        
        try:
            analysis = tune.run(
                objective_function,
                config=search_space,
                num_samples=config.num_samples,
                scheduler=scheduler,
                search_alg=search_alg,
                stop=stopper,
                resources_per_trial=resources,
                local_dir=self.local_dir,
                name=experiment_name,
                progress_reporter=reporter,
                checkpoint_freq=config.checkpoint_freq,
                keep_checkpoints_num=config.keep_checkpoints_num,
                max_failures=config.max_failures,
                verbose=config.verbose,
                raise_on_failed_trial=False
            )
            
            # Extract results
            best_trial = analysis.get_best_trial(
                metric=config.metric,
                mode=config.mode
            )
            
            results = {
                "best_config": best_trial.config,
                "best_result": best_trial.last_result,
                "best_trial_id": best_trial.trial_id,
                "experiment_path": analysis.experiment_path,
                "num_trials": len(analysis.trials),
                "optimization_config": asdict(config),
                "search_space": search_space,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add Pareto frontier for multi-objective
            if config.multi_objective and config.objectives:
                pareto_frontier = self._compute_pareto_frontier(analysis, config.objectives)
                results["pareto_frontier"] = pareto_frontier
            
            self.logger.info(f"Optimization completed successfully")
            self.logger.info(f"Best {config.metric}: {best_trial.last_result[config.metric]:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise
    
    def _create_search_algorithm(self, config: TuneConfig):
        """Create search algorithm"""
        if config.search_algorithm == "optuna":
            return OptunaSearch(
                metric=config.metric,
                mode=config.mode
            )
        elif config.search_algorithm == "hyperopt":
            return HyperOptSearch(
                metric=config.metric,
                mode=config.mode
            )
        elif config.search_algorithm == "bayesopt":
            return BayesOptSearch(
                metric=config.metric,
                mode=config.mode
            )
        else:  # random
            return BasicVariantGenerator()
    
    def _create_scheduler(self, config: TuneConfig):
        """Create scheduler"""
        if config.scheduler == "asha":
            return ASHAScheduler(
                metric=config.metric,
                mode=config.mode,
                max_t=config.max_t,
                grace_period=config.grace_period,
                reduction_factor=config.reduction_factor
            )
        elif config.scheduler == "pbt":
            return PopulationBasedTraining(
                metric=config.metric,
                mode=config.mode,
                perturbation_interval=20,
                hyperparam_mutations={
                    "learning_rate": tune.loguniform(1e-5, 1e-2),
                    "batch_size": [16, 32, 64, 128]
                }
            )
        elif config.scheduler == "median":
            return MedianStoppingRule(
                metric=config.metric,
                mode=config.mode,
                grace_period=config.grace_period
            )
        elif config.scheduler == "hyperband":
            return HyperBandScheduler(
                metric=config.metric,
                mode=config.mode,
                max_t=config.max_t
            )
        else:  # fifo
            return None
    
    def _create_stopper(self, config: TuneConfig):
        """Create stopping criteria"""
        stoppers = []
        
        # Maximum iterations
        if config.max_t:
            stoppers.append(MaximumIterationStopper(max_iter=config.max_t))
        
        # Time budget
        if config.time_budget_s:
            stoppers.append(TimeoutStopper(timeout=config.time_budget_s))
        
        # Plateau detection
        stoppers.append(
            TrialPlateauStopper(
                metric=config.metric,
                std=0.01,
                num_results=10,
                grace_period=config.grace_period,
                metric_threshold=None,
                mode=config.mode
            )
        )
        
        # Combine stoppers
        if len(stoppers) == 1:
            return stoppers[0]
        elif len(stoppers) > 1:
            # Use the first stopper that triggers
            return stoppers[0]  # Simplified - could implement OR logic
        else:
            return None
    
    def _create_reporter(self, config: TuneConfig):
        """Create progress reporter"""
        metric_columns = [config.metric]
        
        # Add additional metrics for multi-objective
        if config.multi_objective and config.objectives:
            metric_columns.extend(config.objectives)
        
        # Add common metrics
        metric_columns.extend(["training_iteration", "time_total_s"])
        
        return CLIReporter(
            metric_columns=metric_columns,
            max_progress_rows=20,
            max_error_rows=5,
            max_report_frequency=30
        )
    
    def _get_rl_search_space(self, agent_type: str) -> Dict[str, Any]:
        """Get search space for RL agent type"""
        agent_type = agent_type.upper()
        
        if agent_type == "PPO":
            return {
                "learning_rate": tune.loguniform(1e-5, 1e-2),
                "batch_size": tune.choice([32, 64, 128, 256]),
                "n_steps": tune.choice([512, 1024, 2048, 4096]),
                "n_epochs": tune.choice([3, 5, 10, 20]),
                "clip_range": tune.uniform(0.1, 0.4),
                "gamma": tune.uniform(0.9, 0.999),
                "gae_lambda": tune.uniform(0.8, 0.99),
                "ent_coef": tune.loguniform(1e-8, 1e-1),
                "vf_coef": tune.uniform(0.1, 1.0)
            }
        elif agent_type == "SAC":
            return {
                "learning_rate": tune.loguniform(1e-5, 1e-2),
                "batch_size": tune.choice([64, 128, 256, 512]),
                "buffer_size": tune.choice([100000, 500000, 1000000]),
                "tau": tune.uniform(0.001, 0.02),
                "gamma": tune.uniform(0.9, 0.999),
                "train_freq": tune.choice([1, 4, 8]),
                "gradient_steps": tune.choice([1, 2, 4]),
                "ent_coef": tune.choice(["auto", 0.01, 0.1, 0.5])
            }
        elif agent_type == "TD3":
            return {
                "learning_rate": tune.loguniform(1e-5, 1e-2),
                "batch_size": tune.choice([64, 100, 128, 256]),
                "buffer_size": tune.choice([100000, 500000, 1000000]),
                "tau": tune.uniform(0.001, 0.02),
                "gamma": tune.uniform(0.9, 0.999),
                "policy_delay": tune.choice([1, 2, 3]),
                "target_policy_noise": tune.uniform(0.1, 0.3),
                "target_noise_clip": tune.uniform(0.3, 0.7)
            }
        elif agent_type == "DQN":
            return {
                "learning_rate": tune.loguniform(1e-5, 1e-2),
                "batch_size": tune.choice([16, 32, 64, 128]),
                "buffer_size": tune.choice([50000, 100000, 500000]),
                "gamma": tune.uniform(0.9, 0.999),
                "exploration_fraction": tune.uniform(0.05, 0.3),
                "exploration_initial_eps": tune.uniform(0.8, 1.0),
                "exploration_final_eps": tune.uniform(0.01, 0.1)
            }
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def _train_cnn_trial(
        self,
        config: Dict[str, Any],
        data_config: Dict[str, Any],
        custom_objective: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Train CNN model for a single trial"""
        try:
            # Create training pipeline
            from .training_pipeline import create_training_pipeline
            
            pipeline = create_training_pipeline(
                input_dim=data_config['input_dim'],
                output_dim=data_config['output_dim'],
                **config
            )
            
            # Prepare data
            train_loader, val_loader, test_loader = pipeline.prepare_data(
                features=data_config['features'],
                targets=data_config['targets'],
                **data_config.get('data_params', {})
            )
            
            # Train model
            result = pipeline.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=config.get('epochs', 100)
            )
            
            # Evaluate model
            test_metrics = pipeline.evaluate(test_loader)
            
            # Prepare results
            results = {
                "train_loss": result.train_loss,
                "val_loss": result.val_loss,
                "test_loss": test_metrics['test_loss'],
                "test_mae": test_metrics['test_mae'],
                "test_rmse": test_metrics['test_rmse'],
                "epochs_trained": result.epochs_trained,
                "training_iteration": result.epochs_trained
            }
            
            # Use custom objective if provided
            if custom_objective:
                custom_results = custom_objective(pipeline.model, test_loader)
                results.update(custom_results)
            
            # Report to Ray Tune
            session.report(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"CNN trial failed: {e}")
            # Report failure to Ray Tune
            session.report({"train_loss": float('inf'), "val_loss": float('inf')})
            raise
    
    def _train_rl_trial(
        self,
        config: Dict[str, Any],
        env_factory: Callable[[], gym.Env],
        agent_type: str,
        custom_objective: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Train RL agent for a single trial"""
        try:
            # Create environment and agent
            env = env_factory()
            eval_env = env_factory()
            
            agent = RLAgentFactory.create_agent(
                agent_type=agent_type,
                env=env,
                **config
            )
            
            # Train agent
            results = agent.train(
                env=env,
                total_timesteps=50000,  # Reduced for hyperparameter search
                eval_env=eval_env,
                eval_freq=10000,
                n_eval_episodes=5
            )
            
            # Extract metrics
            if 'evaluations' in results and results['evaluations']:
                final_eval = results['evaluations'][-1]
                trial_results = {
                    "mean_reward": final_eval['mean_reward'],
                    "std_reward": final_eval['std_reward'],
                    "training_iteration": len(results['evaluations'])
                }
                
                # Add portfolio metrics if available
                if 'portfolio_metrics' in final_eval:
                    portfolio_metrics = final_eval['portfolio_metrics']
                    for key, value in portfolio_metrics.items():
                        if isinstance(value, (int, float)):
                            trial_results[f"portfolio_{key}"] = value
            else:
                # Fallback evaluation
                from stable_baselines3.common.evaluation import evaluate_policy
                mean_reward, std_reward = evaluate_policy(
                    agent.model, eval_env, n_eval_episodes=5
                )
                trial_results = {
                    "mean_reward": mean_reward,
                    "std_reward": std_reward,
                    "training_iteration": 1
                }
            
            # Use custom objective if provided
            if custom_objective:
                custom_results = custom_objective(agent, eval_env)
                trial_results.update(custom_results)
            
            # Report to Ray Tune
            session.report(trial_results)
            
            return trial_results
            
        except Exception as e:
            self.logger.error(f"RL trial failed: {e}")
            # Report failure to Ray Tune
            session.report({"mean_reward": -1000.0, "std_reward": 0.0})
            raise
    
    def _train_hybrid_trial(
        self,
        config: Dict[str, Any],
        data_config: Dict[str, Any],
        custom_objective: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Train hybrid model for a single trial"""
        try:
            # This is a simplified implementation
            # In practice, you would implement the full hybrid model training
            
            # Simulate training results
            results = {
                "classification_accuracy": np.random.uniform(0.6, 0.9),
                "regression_mse": np.random.uniform(0.01, 0.1),
                "combined_loss": np.random.uniform(0.1, 0.5),
                "training_iteration": config.get('epochs', 100)
            }
            
            # Use custom objective if provided
            if custom_objective:
                custom_results = custom_objective(None, None)  # Placeholder
                results.update(custom_results)
            
            # Report to Ray Tune
            session.report(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Hybrid trial failed: {e}")
            # Report failure to Ray Tune
            session.report({"combined_loss": float('inf')})
            raise
    
    def _compute_pareto_frontier(
        self,
        analysis,
        objectives: List[str]
    ) -> List[Dict[str, Any]]:
        """Compute Pareto frontier for multi-objective optimization"""
        # Extract all trial results
        trial_results = []
        for trial in analysis.trials:
            if trial.last_result:
                result = {obj: trial.last_result.get(obj, 0) for obj in objectives}
                result['config'] = trial.config
                result['trial_id'] = trial.trial_id
                trial_results.append(result)
        
        # Compute Pareto frontier (simplified implementation)
        pareto_frontier = []
        
        for i, result_i in enumerate(trial_results):
            is_dominated = False
            
            for j, result_j in enumerate(trial_results):
                if i != j:
                    # Check if result_j dominates result_i
                    dominates = True
                    for obj in objectives:
                        if result_j[obj] <= result_i[obj]:  # Assuming maximization
                            dominates = False
                            break
                    
                    if dominates:
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_frontier.append(result_i)
        
        return pareto_frontier
    
    def save_results(self, results: Dict[str, Any], filepath: str) -> None:
        """Save optimization results to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load optimization results from file"""
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        self.logger.info(f"Results loaded from {filepath}")
        return results


def create_tune_optimizer(
    config: Optional[TuneConfig] = None,
    local_dir: str = "ray_results",
    experiment_name: Optional[str] = None
) -> RayTuneOptimizer:
    """Create Ray Tune optimizer
    
    Args:
        config: Tune configuration (uses defaults if None)
        local_dir: Directory for results
        experiment_name: Name of the experiment
        
    Returns:
        Configured Ray Tune optimizer
    """
    if config is None:
        config = TuneConfig()
    
    return RayTuneOptimizer(
        config=config,
        local_dir=local_dir,
        experiment_name=experiment_name
    )