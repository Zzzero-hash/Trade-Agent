"""
Hyperparameter Optimization for RL Agents.

This module provides utilities for optimizing RL agent hyperparameters using
Ray Tune and Optuna for efficient hyperparameter search.
"""

import os
import json
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime
import numpy as np
import gymnasium as gym
import logging

# Optional imports for hyperparameter optimization
try:
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.search.hyperopt import HyperOptSearch
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    tune = None

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

from .rl_agents import RLAgentFactory, StableBaselinesRLAgent, RLAgentConfig
from .trading_environment import TradingEnvironment


class HyperparameterOptimizer:
    """Hyperparameter optimizer for RL agents using Ray Tune"""
    
    def __init__(
        self,
        env_factory: Callable[[], gym.Env],
        agent_type: str,
        optimization_metric: str = "mean_reward",
        optimization_mode: str = "max",
        num_samples: int = 50,
        max_concurrent_trials: int = 4,
        use_gpu: bool = False,
        log_dir: str = "ray_results"
    ):
        """Initialize hyperparameter optimizer
        
        Args:
            env_factory: Factory function to create training environments
            agent_type: Type of RL agent to optimize
            optimization_metric: Metric to optimize
            optimization_mode: 'min' or 'max' for optimization
            num_samples: Number of hyperparameter samples to try
            max_concurrent_trials: Maximum concurrent trials
            use_gpu: Whether to use GPU for training
            log_dir: Directory for Ray Tune logs
        """
        self.env_factory = env_factory
        self.agent_type = agent_type.upper()
        self.optimization_metric = optimization_metric
        self.optimization_mode = optimization_mode
        self.num_samples = num_samples
        self.max_concurrent_trials = max_concurrent_trials
        self.use_gpu = use_gpu
        self.log_dir = log_dir
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Validate agent type
        if self.agent_type not in ['PPO', 'SAC', 'TD3', 'DQN']:
            raise ValueError(f"Unsupported agent type: {agent_type}")
    
    def get_search_space(self) -> Dict[str, Any]:
        """Get hyperparameter search space for the agent type
        
        Returns:
            Dictionary defining the search space
        """
        if self.agent_type == 'PPO':
            return {
                'learning_rate': tune.loguniform(1e-5, 1e-2),
                'batch_size': tune.choice([32, 64, 128, 256]),
                'n_steps': tune.choice([512, 1024, 2048, 4096]),
                'n_epochs': tune.choice([3, 5, 10, 20]),
                'clip_range': tune.uniform(0.1, 0.4),
                'gamma': tune.uniform(0.9, 0.999),
                'gae_lambda': tune.uniform(0.8, 0.99),
                'ent_coef': tune.loguniform(1e-8, 1e-1),
                'vf_coef': tune.uniform(0.1, 1.0),
                'max_grad_norm': tune.uniform(0.3, 2.0)
            }
        
        elif self.agent_type == 'SAC':
            return {
                'learning_rate': tune.loguniform(1e-5, 1e-2),
                'batch_size': tune.choice([64, 128, 256, 512]),
                'buffer_size': tune.choice([100000, 500000, 1000000]),
                'tau': tune.uniform(0.001, 0.02),
                'gamma': tune.uniform(0.9, 0.999),
                'train_freq': tune.choice([1, 4, 8]),
                'gradient_steps': tune.choice([1, 2, 4]),
                'ent_coef': tune.choice(['auto', 0.01, 0.1, 0.5]),
                'target_update_interval': tune.choice([1, 2, 4]),
                'learning_starts': tune.choice([1000, 5000, 10000])
            }
        
        elif self.agent_type == 'TD3':
            return {
                'learning_rate': tune.loguniform(1e-5, 1e-2),
                'batch_size': tune.choice([64, 100, 128, 256]),
                'buffer_size': tune.choice([100000, 500000, 1000000]),
                'tau': tune.uniform(0.001, 0.02),
                'gamma': tune.uniform(0.9, 0.999),
                'train_freq': tune.choice([1, 2, 4]),
                'gradient_steps': tune.choice([1, 2, 4]),
                'policy_delay': tune.choice([1, 2, 3]),
                'target_policy_noise': tune.uniform(0.1, 0.3),
                'target_noise_clip': tune.uniform(0.3, 0.7),
                'learning_starts': tune.choice([1000, 5000, 10000])
            }
        
        elif self.agent_type == 'DQN':
            return {
                'learning_rate': tune.loguniform(1e-5, 1e-2),
                'batch_size': tune.choice([16, 32, 64, 128]),
                'buffer_size': tune.choice([50000, 100000, 500000, 1000000]),
                'tau': tune.uniform(0.8, 1.0),
                'gamma': tune.uniform(0.9, 0.999),
                'train_freq': tune.choice([1, 4, 8, 16]),
                'gradient_steps': tune.choice([1, 2, 4]),
                'target_update_interval': tune.choice([1000, 5000, 10000]),
                'exploration_fraction': tune.uniform(0.05, 0.3),
                'exploration_initial_eps': tune.uniform(0.8, 1.0),
                'exploration_final_eps': tune.uniform(0.01, 0.1),
                'learning_starts': tune.choice([1000, 5000, 10000])
            }
    
    def objective_function(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Objective function for hyperparameter optimization
        
        Args:
            config: Hyperparameter configuration
            
        Returns:
            Dictionary with optimization metrics
        """
        try:
            # Create environment
            env = self.env_factory()
            eval_env = self.env_factory()
            
            # Create agent with hyperparameters
            agent = RLAgentFactory.create_agent(
                agent_type=self.agent_type,
                env=env,
                **config
            )
            
            # Train agent
            training_timesteps = 50000  # Reduced for hyperparameter search
            results = agent.train(
                env=env,
                total_timesteps=training_timesteps,
                eval_env=eval_env,
                eval_freq=10000,
                n_eval_episodes=5
            )
            
            # Extract metrics
            metrics = {}
            
            if 'evaluations' in results and results['evaluations']:
                final_eval = results['evaluations'][-1]
                metrics['mean_reward'] = final_eval['mean_reward']
                metrics['std_reward'] = final_eval['std_reward']
                
                # Add portfolio metrics if available
                if 'portfolio_metrics' in final_eval:
                    portfolio_metrics = final_eval['portfolio_metrics']
                    for key, value in portfolio_metrics.items():
                        if isinstance(value, (int, float)):
                            metrics[f'portfolio_{key}'] = value
            else:
                # Fallback evaluation
                from stable_baselines3.common.evaluation import evaluate_policy
                mean_reward, std_reward = evaluate_policy(
                    agent.model, eval_env, n_eval_episodes=5
                )
                metrics['mean_reward'] = mean_reward
                metrics['std_reward'] = std_reward
            
            # Add training efficiency metrics
            metrics['training_time'] = results.get('training_time', 0)
            metrics['timesteps_per_second'] = training_timesteps / max(metrics['training_time'], 1)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in objective function: {e}")
            # Return poor performance for failed trials
            return {
                'mean_reward': -1000.0,
                'std_reward': 0.0,
                'training_time': 9999.0,
                'timesteps_per_second': 0.0
            }
    
    def optimize(
        self,
        search_algorithm: str = "optuna",
        scheduler: str = "asha",
        custom_search_space: Optional[Dict[str, Any]] = None,
        resume: bool = False
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization
        
        Args:
            search_algorithm: Search algorithm ('optuna', 'hyperopt', 'random')
            scheduler: Scheduler type ('asha', 'pbt', 'fifo')
            custom_search_space: Custom search space (overrides default)
            resume: Whether to resume from previous run
            
        Returns:
            Optimization results
        """
        if not RAY_AVAILABLE:
            self.logger.warning("Ray Tune not available, running simple grid search")
            return self._simple_grid_search(custom_search_space)
        
        # Get search space
        search_space = custom_search_space or self.get_search_space()
        
        # Setup search algorithm
        search_alg = None
        if search_algorithm == "optuna" and RAY_AVAILABLE:
            search_alg = OptunaSearch(
                metric=self.optimization_metric,
                mode=self.optimization_mode
            )
        elif search_algorithm == "hyperopt" and RAY_AVAILABLE:
            search_alg = HyperOptSearch(
                metric=self.optimization_metric,
                mode=self.optimization_mode
            )
        # For 'random', search_alg remains None (uses random search)
        
        # Setup scheduler
        scheduler_obj = None
        if scheduler == "asha":
            scheduler_obj = ASHAScheduler(
                metric=self.optimization_metric,
                mode=self.optimization_mode,
                max_t=100,
                grace_period=10,
                reduction_factor=2
            )
        elif scheduler == "pbt":
            scheduler_obj = PopulationBasedTraining(
                metric=self.optimization_metric,
                mode=self.optimization_mode,
                perturbation_interval=20,
                hyperparam_mutations=search_space
            )
        
        # Setup reporter
        reporter = CLIReporter(
            metric_columns=[
                self.optimization_metric,
                "std_reward",
                "training_time",
                "timesteps_per_second"
            ]
        )
        
        # Configure resources
        resources_per_trial = {"cpu": 1}
        if self.use_gpu:
            resources_per_trial["gpu"] = 0.25  # Share GPU across trials
        
        # Run optimization
        self.logger.info(f"Starting hyperparameter optimization for {self.agent_type}")
        self.logger.info(f"Search space: {search_space}")
        
        analysis = tune.run(
            self.objective_function,
            config=search_space,
            num_samples=self.num_samples,
            scheduler=scheduler_obj,
            search_alg=search_alg,
            progress_reporter=reporter,
            resources_per_trial=resources_per_trial,
            local_dir=self.log_dir,
            name=f"{self.agent_type.lower()}_hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            resume=resume,
            max_concurrent_trials=self.max_concurrent_trials,
            raise_on_failed_trial=False
        )
        
        # Get best results
        best_trial = analysis.get_best_trial(
            metric=self.optimization_metric,
            mode=self.optimization_mode
        )
        
        best_config = best_trial.config
        best_metrics = best_trial.last_result
        
        results = {
            'best_config': best_config,
            'best_metrics': best_metrics,
            'best_trial_id': best_trial.trial_id,
            'optimization_metric': self.optimization_metric,
            'optimization_mode': self.optimization_mode,
            'num_trials': len(analysis.trials),
            'search_algorithm': search_algorithm,
            'scheduler': scheduler,
            'agent_type': self.agent_type
        }
        
        self.logger.info(f"Optimization completed. Best {self.optimization_metric}: "
                        f"{best_metrics[self.optimization_metric]:.4f}")
        self.logger.info(f"Best config: {best_config}")
        
        return results
    
    def _simple_grid_search(self, custom_search_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Simple grid search fallback when Ray is not available"""
        search_space = custom_search_space or self._get_simple_search_space()
        
        best_config = None
        best_metrics = None
        best_score = float('-inf') if self.optimization_mode == 'max' else float('inf')
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(search_space)
        
        self.logger.info(f"Running simple grid search with {len(param_combinations)} combinations")
        
        for i, config in enumerate(param_combinations[:self.num_samples]):
            self.logger.info(f"Testing configuration {i+1}/{min(len(param_combinations), self.num_samples)}")
            
            try:
                metrics = self.objective_function(config)
                score = metrics[self.optimization_metric]
                
                is_better = (
                    (self.optimization_mode == 'max' and score > best_score) or
                    (self.optimization_mode == 'min' and score < best_score)
                )
                
                if is_better:
                    best_score = score
                    best_config = config
                    best_metrics = metrics
                    
            except Exception as e:
                self.logger.warning(f"Configuration {i+1} failed: {e}")
                continue
        
        results = {
            'best_config': best_config or {},
            'best_metrics': best_metrics or {},
            'best_trial_id': 'simple_search_best',
            'optimization_metric': self.optimization_metric,
            'optimization_mode': self.optimization_mode,
            'num_trials': min(len(param_combinations), self.num_samples),
            'search_algorithm': 'simple_grid',
            'scheduler': 'none',
            'agent_type': self.agent_type
        }
        
        return results
    
    def _get_simple_search_space(self) -> Dict[str, List]:
        """Get simplified search space for grid search"""
        if self.agent_type == 'PPO':
            return {
                'learning_rate': [1e-4, 3e-4, 1e-3],
                'batch_size': [32, 64, 128],
                'n_steps': [1024, 2048],
                'clip_range': [0.1, 0.2, 0.3]
            }
        elif self.agent_type == 'SAC':
            return {
                'learning_rate': [1e-4, 3e-4, 1e-3],
                'batch_size': [64, 128, 256],
                'tau': [0.005, 0.01],
                'gamma': [0.99, 0.995]
            }
        elif self.agent_type == 'TD3':
            return {
                'learning_rate': [1e-4, 3e-4, 1e-3],
                'batch_size': [64, 100, 128],
                'tau': [0.005, 0.01],
                'policy_delay': [1, 2]
            }
        elif self.agent_type == 'DQN':
            return {
                'learning_rate': [1e-4, 3e-4, 1e-3],
                'batch_size': [32, 64],
                'gamma': [0.99, 0.995],
                'exploration_fraction': [0.1, 0.2]
            }
        else:
            return {'learning_rate': [3e-4]}
    
    def _generate_param_combinations(self, search_space: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from search space"""
        import itertools
        
        keys = list(search_space.keys())
        values = list(search_space.values())
        
        combinations = []
        for combination in itertools.product(*values):
            config = dict(zip(keys, combination))
            combinations.append(config)
        
        return combinations
    
    def save_results(self, results: Dict[str, Any], filepath: str) -> None:
        """Save optimization results
        
        Args:
            results: Optimization results
            filepath: Path to save results
        """
        # Add metadata
        results['timestamp'] = datetime.now().isoformat()
        results['optimizer_config'] = {
            'agent_type': self.agent_type,
            'optimization_metric': self.optimization_metric,
            'optimization_mode': self.optimization_mode,
            'num_samples': self.num_samples
        }
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load optimization results
        
        Args:
            filepath: Path to load results from
            
        Returns:
            Optimization results
        """
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        self.logger.info(f"Results loaded from {filepath}")
        return results


class MultiAgentHyperparameterOptimizer:
    """Hyperparameter optimizer for multiple agent types"""
    
    def __init__(
        self,
        env_factory: Callable[[], gym.Env],
        agent_types: List[str],
        optimization_metric: str = "mean_reward",
        optimization_mode: str = "max",
        num_samples_per_agent: int = 20,
        max_concurrent_trials: int = 4,
        use_gpu: bool = False,
        log_dir: str = "ray_results"
    ):
        """Initialize multi-agent hyperparameter optimizer
        
        Args:
            env_factory: Factory function to create training environments
            agent_types: List of agent types to optimize
            optimization_metric: Metric to optimize
            optimization_mode: 'min' or 'max' for optimization
            num_samples_per_agent: Number of samples per agent type
            max_concurrent_trials: Maximum concurrent trials
            use_gpu: Whether to use GPU for training
            log_dir: Directory for Ray Tune logs
        """
        self.env_factory = env_factory
        self.agent_types = [t.upper() for t in agent_types]
        self.optimization_metric = optimization_metric
        self.optimization_mode = optimization_mode
        self.num_samples_per_agent = num_samples_per_agent
        self.max_concurrent_trials = max_concurrent_trials
        self.use_gpu = use_gpu
        self.log_dir = log_dir
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Create individual optimizers
        self.optimizers = {}
        for agent_type in self.agent_types:
            self.optimizers[agent_type] = HyperparameterOptimizer(
                env_factory=env_factory,
                agent_type=agent_type,
                optimization_metric=optimization_metric,
                optimization_mode=optimization_mode,
                num_samples=num_samples_per_agent,
                max_concurrent_trials=max_concurrent_trials,
                use_gpu=use_gpu,
                log_dir=log_dir
            )
    
    def optimize_all(
        self,
        search_algorithm: str = "optuna",
        scheduler: str = "asha",
        parallel: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Optimize hyperparameters for all agent types
        
        Args:
            search_algorithm: Search algorithm to use
            scheduler: Scheduler type
            parallel: Whether to run optimizations in parallel
            
        Returns:
            Dictionary of results for each agent type
        """
        results = {}
        
        if parallel:
            # TODO: Implement parallel optimization using Ray
            self.logger.warning("Parallel optimization not yet implemented, running sequentially")
        
        # Sequential optimization
        for agent_type in self.agent_types:
            self.logger.info(f"Optimizing {agent_type} hyperparameters...")
            
            optimizer = self.optimizers[agent_type]
            agent_results = optimizer.optimize(
                search_algorithm=search_algorithm,
                scheduler=scheduler
            )
            
            results[agent_type] = agent_results
            
            # Save individual results
            results_path = os.path.join(
                self.log_dir,
                f"{agent_type.lower()}_optimization_results.json"
            )
            optimizer.save_results(agent_results, results_path)
        
        return results
    
    def get_best_configs(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Extract best configurations for each agent type
        
        Args:
            results: Optimization results for all agents
            
        Returns:
            Dictionary of best configurations
        """
        best_configs = {}
        
        for agent_type, agent_results in results.items():
            best_configs[agent_type] = agent_results['best_config']
        
        return best_configs
    
    def create_optimized_agents(
        self,
        best_configs: Dict[str, Dict[str, Any]],
        env: gym.Env
    ) -> Dict[str, StableBaselinesRLAgent]:
        """Create agents with optimized hyperparameters
        
        Args:
            best_configs: Best configurations for each agent type
            env: Training environment
            
        Returns:
            Dictionary of optimized agents
        """
        agents = {}
        
        for agent_type, config in best_configs.items():
            agent = RLAgentFactory.create_agent(
                agent_type=agent_type,
                env=env,
                **config
            )
            agents[agent_type] = agent
        
        return agents


def optimize_agent_hyperparameters(
    env_factory: Callable[[], gym.Env],
    agent_type: str,
    num_samples: int = 50,
    optimization_metric: str = "mean_reward",
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function for single agent hyperparameter optimization
    
    Args:
        env_factory: Factory function to create environments
        agent_type: Type of agent to optimize
        num_samples: Number of hyperparameter samples
        optimization_metric: Metric to optimize
        save_path: Path to save results (optional)
        
    Returns:
        Optimization results
    """
    optimizer = HyperparameterOptimizer(
        env_factory=env_factory,
        agent_type=agent_type,
        num_samples=num_samples,
        optimization_metric=optimization_metric
    )
    
    results = optimizer.optimize()
    
    if save_path:
        optimizer.save_results(results, save_path)
    
    return results


def optimize_ensemble_hyperparameters(
    env_factory: Callable[[], gym.Env],
    agent_types: List[str] = ['PPO', 'SAC', 'TD3', 'DQN'],
    num_samples_per_agent: int = 20,
    optimization_metric: str = "mean_reward",
    save_dir: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """Convenience function for multi-agent hyperparameter optimization
    
    Args:
        env_factory: Factory function to create environments
        agent_types: List of agent types to optimize
        num_samples_per_agent: Number of samples per agent
        optimization_metric: Metric to optimize
        save_dir: Directory to save results (optional)
        
    Returns:
        Optimization results for all agents
    """
    optimizer = MultiAgentHyperparameterOptimizer(
        env_factory=env_factory,
        agent_types=agent_types,
        num_samples_per_agent=num_samples_per_agent,
        optimization_metric=optimization_metric,
        log_dir=save_dir or "ray_results"
    )
    
    results = optimizer.optimize_all()
    
    return results