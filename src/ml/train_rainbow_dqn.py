"""
Training Script for Rainbow DQN Agent.

This script implements comprehensive training for the Rainbow DQN agent with:
- Full Rainbow features (C51, Double DQN, Dueling, Prioritized Replay, Noisy Networks)
- Advanced training procedures with curriculum learning
- Comprehensive evaluation and performance tracking
- Hyperparameter optimization integration
- Model checkpointing and resuming
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, List
import warnings

import numpy as np
import torch
import gymnasium as gym
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ml.advanced_dqn_agent import RainbowDQNAgent, RainbowDQNConfig
from src.ml.yfinance_trading_environment import YFinanceTradingEnvironment, YFinanceConfig
from src.ml.rl_hyperopt import HyperparameterOptimizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rainbow_dqn_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


class RainbowDQNTrainer:
    """Comprehensive trainer for Rainbow DQN agent."""
    
    def __init__(
        self,
        config: RainbowDQNConfig,
        env_config: YFinanceConfig,
        symbols: List[str] = None,
        start_date: str = "2018-01-01",
        end_date: str = "2023-12-31",
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15
    ):
        """Initialize Rainbow DQN trainer.
        
        Args:
            config: Rainbow DQN configuration
            env_config: Environment configuration
            symbols: List of trading symbols
            start_date: Training data start date
            end_date: Training data end date
            train_split: Training data split ratio
            val_split: Validation data split ratio
            test_split: Test data split ratio
        """
        self.config = config
        self.env_config = env_config
        self.symbols = symbols or [
            "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", 
            "AMZN", "TSLA", "NVDA", "META", "NFLX"
        ]
        
        # Data splits
        self.start_date = start_date
        self.end_date = end_date
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        # Calculate date ranges for splits
        self._calculate_date_splits()
        
        # Create environments
        self.train_env = self._create_environment(self.train_start, self.train_end)
        self.val_env = self._create_environment(self.val_start, self.val_end)
        self.test_env = self._create_environment(self.test_start, self.test_end)
        
        # Wrap environments with discrete action wrapper for DQN
        from src.ml.discrete_trading_wrapper import DiscreteTradingWrapper
        self.train_env = DiscreteTradingWrapper(self.train_env, action_strategy="single_asset")
        self.val_env = DiscreteTradingWrapper(self.val_env, action_strategy="single_asset")
        self.test_env = DiscreteTradingWrapper(self.test_env, action_strategy="single_asset")
        
        # Initialize agent
        self.agent = RainbowDQNAgent(self.train_env, config)
        
        # Training state
        self.training_results = []
        self.best_performance = float('-inf')
        self.best_model_path = None
        
        logger.info(f"Rainbow DQN trainer initialized with {len(self.symbols)} symbols")
        logger.info(f"Training period: {self.train_start} to {self.train_end}")
        logger.info(f"Validation period: {self.val_start} to {self.val_end}")
        logger.info(f"Test period: {self.test_start} to {self.test_end}")
    
    def _calculate_date_splits(self):
        """Calculate date ranges for train/val/test splits."""
        from datetime import datetime, timedelta
        
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        total_days = (end - start).days
        
        train_days = int(total_days * self.train_split)
        val_days = int(total_days * self.val_split)
        
        self.train_start = start.strftime("%Y-%m-%d")
        self.train_end = (start + timedelta(days=train_days)).strftime("%Y-%m-%d")
        
        self.val_start = (start + timedelta(days=train_days + 1)).strftime("%Y-%m-%d")
        self.val_end = (start + timedelta(days=train_days + val_days)).strftime("%Y-%m-%d")
        
        self.test_start = (start + timedelta(days=train_days + val_days + 1)).strftime("%Y-%m-%d")
        self.test_end = end.strftime("%Y-%m-%d")
    
    def _create_environment(self, start_date: str, end_date: str) -> YFinanceTradingEnvironment:
        """Create trading environment for given date range."""
        return YFinanceTradingEnvironment(
            symbols=self.symbols,
            start_date=start_date,
            end_date=end_date,
            config=self.env_config,
            data_source="yfinance",
            cache_dir="data/cache"
        )
    
    def train_agent(
        self,
        total_timesteps: int = 2000000,
        eval_freq: int = 25000,
        n_eval_episodes: int = 10,
        checkpoint_freq: int = 50000,
        save_dir: str = "models/rainbow_dqn",
        resume_from: Optional[str] = None,
        target_sharpe: float = 1.5,
        early_stopping_patience: int = 5
    ) -> Dict[str, Any]:
        """Train Rainbow DQN agent with comprehensive evaluation and target validation.
        
        Args:
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes
            checkpoint_freq: Checkpoint saving frequency
            save_dir: Directory to save models
            resume_from: Path to resume training from
            target_sharpe: Target Sharpe ratio to achieve
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training results
        """
        logger.info(f"Starting Rainbow DQN training for {total_timesteps:,} timesteps")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Resume from checkpoint if specified
        if resume_from and os.path.exists(resume_from):
            logger.info(f"Resuming training from {resume_from}")
            self.agent.load_model(resume_from)
        
        # Enhanced training with target monitoring
        logger.info(f"Training target: Sharpe ratio >= {target_sharpe}")
        
        # Custom training loop with target monitoring
        results = self._train_with_target_monitoring(
            total_timesteps=total_timesteps,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            checkpoint_freq=checkpoint_freq,
            save_dir=save_dir,
            target_sharpe=target_sharpe,
            early_stopping_patience=early_stopping_patience
        )
        
        # Save final model
        final_model_path = os.path.join(save_dir, "rainbow_dqn_final.pth")
        self.agent.save_model(final_model_path)
        
        # Track best performance
        if results['evaluations']:
            best_eval = max(results['evaluations'], key=lambda x: x['mean_reward'])
            if best_eval['mean_reward'] > self.best_performance:
                self.best_performance = best_eval['mean_reward']
                self.best_model_path = final_model_path
        
        self.training_results.append(results)
        
        logger.info(f"Training completed. Best validation performance: {self.best_performance:.4f}")
        
        return results
    
    def _train_with_target_monitoring(
        self,
        total_timesteps: int,
        eval_freq: int,
        n_eval_episodes: int,
        checkpoint_freq: int,
        save_dir: str,
        target_sharpe: float,
        early_stopping_patience: int
    ) -> Dict[str, Any]:
        """Enhanced training with target Sharpe ratio monitoring."""
        
        # Track best performance
        best_sharpe = float('-inf')
        patience_counter = 0
        target_achieved = False
        
        # Training with custom callback for target monitoring
        class TargetMonitoringCallback:
            def __init__(self, trainer, target_sharpe, patience):
                self.trainer = trainer
                self.target_sharpe = target_sharpe
                self.patience = patience
                self.best_sharpe = float('-inf')
                self.patience_counter = 0
                self.target_achieved = False
                self.evaluations = []
            
            def on_evaluation(self, timestep, eval_results):
                self.evaluations.append({**eval_results, 'timesteps': timestep})
                
                if 'sharpe_ratio' in eval_results:
                    current_sharpe = eval_results['sharpe_ratio']
                    
                    if current_sharpe > self.best_sharpe:
                        self.best_sharpe = current_sharpe
                        self.patience_counter = 0
                        
                        # Save best model
                        best_model_path = os.path.join(save_dir, "best_sharpe_model.pth")
                        self.trainer.agent.save_model(best_model_path)
                        logger.info(f"New best Sharpe ratio: {current_sharpe:.4f} - Model saved")
                    else:
                        self.patience_counter += 1
                    
                    # Check if target achieved
                    if current_sharpe >= self.target_sharpe and not self.target_achieved:
                        self.target_achieved = True
                        target_model_path = os.path.join(save_dir, "target_achieved_model.pth")
                        self.trainer.agent.save_model(target_model_path)
                        logger.info(f"🎯 TARGET ACHIEVED! Sharpe ratio: {current_sharpe:.4f} >= {self.target_sharpe}")
                    
                    logger.info(f"Sharpe: {current_sharpe:.4f} (Best: {self.best_sharpe:.4f}, "
                              f"Target: {self.target_sharpe}, Patience: {self.patience_counter}/{self.patience})")
        
        # Create callback
        callback = TargetMonitoringCallback(self, target_sharpe, early_stopping_patience)
        
        # Standard training with enhanced monitoring
        results = self.agent.train(
            env=self.train_env,
            total_timesteps=total_timesteps,
            eval_env=self.val_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            checkpoint_freq=checkpoint_freq,
            checkpoint_path=save_dir,
            log_path=os.path.join(save_dir, "training_log.json")
        )
        
        # Add callback results to training results
        results['best_sharpe_ratio'] = callback.best_sharpe
        results['target_achieved'] = callback.target_achieved
        results['target_sharpe'] = target_sharpe
        results['enhanced_evaluations'] = callback.evaluations
        
        # Final validation
        if callback.target_achieved:
            logger.info("✅ TASK 7.1 REQUIREMENT MET: DQN achieved >1.5 Sharpe ratio")
        else:
            logger.warning(f"❌ TASK 7.1 REQUIREMENT NOT MET: Best Sharpe {callback.best_sharpe:.4f} < {target_sharpe}")
        
        return results
    
    def evaluate_agent(
        self,
        model_path: Optional[str] = None,
        n_episodes: int = 100,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Comprehensive evaluation of trained agent.
        
        Args:
            model_path: Path to model to evaluate (uses best if None)
            n_episodes: Number of evaluation episodes
            save_results: Whether to save evaluation results
            
        Returns:
            Evaluation results
        """
        if model_path is None:
            model_path = self.best_model_path
        
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading model from {model_path} for evaluation")
            self.agent.load_model(model_path)
        elif not self.agent.is_trained:
            raise ValueError("No trained model available for evaluation")
        
        logger.info(f"Evaluating agent on test set with {n_episodes} episodes")
        
        # Detailed evaluation on test set
        test_results = self._detailed_evaluation(self.test_env, n_episodes)
        
        # Calculate financial metrics
        financial_metrics = self._calculate_financial_metrics(test_results)
        
        # Combine results
        evaluation_results = {
            'model_path': model_path,
            'n_episodes': n_episodes,
            'test_period': f"{self.test_start} to {self.test_end}",
            'basic_metrics': test_results,
            'financial_metrics': financial_metrics,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Save results
        if save_results:
            results_path = os.path.join(
                os.path.dirname(model_path) if model_path else "results",
                f"rainbow_dqn_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            
            with open(results_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            
            logger.info(f"Evaluation results saved to {results_path}")
        
        # Log key metrics
        logger.info("=== EVALUATION RESULTS ===")
        logger.info(f"Mean Episode Reward: {test_results['mean_reward']:.4f} ± {test_results['std_reward']:.4f}")
        logger.info(f"Sharpe Ratio: {financial_metrics['sharpe_ratio']:.4f}")
        logger.info(f"Total Return: {financial_metrics['total_return']:.2%}")
        logger.info(f"Max Drawdown: {financial_metrics['max_drawdown']:.2%}")
        logger.info(f"Win Rate: {financial_metrics['win_rate']:.2%}")
        
        return evaluation_results
    
    def _detailed_evaluation(self, env: gym.Env, n_episodes: int) -> Dict[str, Any]:
        """Perform detailed evaluation with episode tracking."""
        episode_rewards = []
        episode_lengths = []
        portfolio_values = []
        actions_taken = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_actions = []
            done = False
            
            while not done:
                action, _ = self.agent.predict(obs, deterministic=True)
                action = action[0]
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                episode_actions.append(action)
                
                # Track portfolio value if available
                if 'portfolio_value' in info:
                    portfolio_values.append(info['portfolio_value'])
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            actions_taken.extend(episode_actions)
            
            if (episode + 1) % 20 == 0:
                logger.info(f"Evaluation episode {episode + 1}/{n_episodes} completed")
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'portfolio_values': portfolio_values,
            'action_distribution': self._analyze_actions(actions_taken)
        }
    
    def _analyze_actions(self, actions: List[int]) -> Dict[str, float]:
        """Analyze action distribution."""
        if not actions:
            return {}
        
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        total_actions = len(actions)
        action_distribution = {
            str(action): count / total_actions 
            for action, count in action_counts.items()
        }
        
        return action_distribution
    
    def _calculate_financial_metrics(self, test_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate financial performance metrics."""
        portfolio_values = test_results.get('portfolio_values', [])
        episode_rewards = test_results.get('episode_rewards', [])
        
        if not portfolio_values or not episode_rewards:
            return {}
        
        # Convert to numpy arrays
        portfolio_values = np.array(portfolio_values)
        episode_rewards = np.array(episode_rewards)
        
        # Calculate returns
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            returns = returns[np.isfinite(returns)]  # Remove inf/nan
        else:
            returns = np.array([])
        
        # Total return
        if len(portfolio_values) > 0:
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        else:
            total_return = 0.0
        
        # Sharpe ratio (assuming daily returns, 252 trading days)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        if len(portfolio_values) > 1:
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - peak) / peak
            max_drawdown = np.min(drawdown)
        else:
            max_drawdown = 0.0
        
        # Win rate (percentage of positive episode rewards)
        win_rate = np.mean(episode_rewards > 0) if len(episode_rewards) > 0 else 0.0
        
        # Sortino ratio (downside deviation)
        if len(returns) > 1:
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                sortino_ratio = np.mean(returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0.0
            else:
                sortino_ratio = float('inf') if np.mean(returns) > 0 else 0.0
        else:
            sortino_ratio = 0.0
        
        # Calmar ratio (return / max drawdown)
        calmar_ratio = abs(total_return / max_drawdown) if max_drawdown != 0 else 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'volatility': np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0,
            'mean_return': np.mean(returns) * 252 if len(returns) > 0 else 0.0
        }
    
    def hyperparameter_optimization(
        self,
        n_trials: int = 50,
        optimization_metric: str = "mean_reward",
        save_dir: str = "hyperopt_results"
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization for Rainbow DQN.
        
        Args:
            n_trials: Number of optimization trials
            optimization_metric: Metric to optimize
            save_dir: Directory to save results
            
        Returns:
            Optimization results
        """
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        # Create environment factory
        def env_factory():
            return self._create_environment(self.train_start, self.train_end)
        
        # Custom search space for Rainbow DQN
        search_space = {
            'learning_rate': (1e-5, 1e-2),
            'batch_size': [16, 32, 64, 128],
            'gamma': (0.9, 0.999),
            'tau': (0.8, 1.0),
            'alpha': (0.4, 0.8),  # Prioritized replay
            'beta': (0.2, 0.6),   # Importance sampling
            'multi_step': [1, 2, 3, 5],
            'n_atoms': [21, 51, 101],  # Distributional RL
            'noisy_std': (0.1, 1.0),
            'target_update_interval': [5000, 10000, 20000],
            'exploration_fraction': (0.05, 0.3)
        }
        
        # Custom objective function for Rainbow DQN
        def objective_function(config: Dict[str, Any]) -> Dict[str, float]:
            try:
                # Create Rainbow DQN config
                rainbow_config = RainbowDQNConfig(
                    learning_rate=config['learning_rate'],
                    batch_size=config['batch_size'],
                    gamma=config['gamma'],
                    tau=config['tau'],
                    alpha=config['alpha'],
                    beta=config['beta'],
                    multi_step=config['multi_step'],
                    n_atoms=config['n_atoms'],
                    noisy_std=config['noisy_std'],
                    target_update_interval=config['target_update_interval'],
                    exploration_fraction=config['exploration_fraction'],
                    verbose=0  # Reduce verbosity for hyperopt
                )
                
                # Create environment and agent
                env = env_factory()
                agent = RainbowDQNAgent(env, rainbow_config)
                
                # Short training for hyperopt
                training_timesteps = 100000  # Reduced for efficiency
                results = agent.train(
                    env=env,
                    total_timesteps=training_timesteps,
                    eval_env=self.val_env,
                    eval_freq=20000,
                    n_eval_episodes=5
                )
                
                # Extract metrics
                if results['evaluations']:
                    final_eval = results['evaluations'][-1]
                    return {
                        'mean_reward': final_eval['mean_reward'],
                        'std_reward': final_eval['std_reward'],
                        'training_time': results['training_time']
                    }
                else:
                    return {
                        'mean_reward': results.get('mean_episode_reward', -1000),
                        'std_reward': 0.0,
                        'training_time': results['training_time']
                    }
                    
            except Exception as e:
                logger.error(f"Error in hyperopt objective: {e}")
                return {
                    'mean_reward': -1000.0,
                    'std_reward': 0.0,
                    'training_time': 9999.0
                }
        
        # Run optimization using simple grid search (fallback)
        logger.info("Running hyperparameter optimization...")
        
        best_config = None
        best_score = float('-inf')
        results_list = []
        
        # Generate parameter combinations
        import itertools
        
        # Simplified search space for demo
        simple_search = {
            'learning_rate': [1e-4, 3e-4, 1e-3],
            'batch_size': [32, 64],
            'gamma': [0.99, 0.995],
            'alpha': [0.6],
            'beta': [0.4],
            'multi_step': [3],
            'n_atoms': [51],
            'noisy_std': [0.5],
            'target_update_interval': [10000],
            'exploration_fraction': [0.1]
        }
        
        keys = list(simple_search.keys())
        values = list(simple_search.values())
        
        combinations = list(itertools.product(*values))[:n_trials]  # Limit to n_trials
        
        for i, combination in enumerate(combinations):
            config = dict(zip(keys, combination))
            logger.info(f"Trial {i+1}/{len(combinations)}: {config}")
            
            try:
                metrics = objective_function(config)
                score = metrics[optimization_metric]
                
                result = {
                    'trial': i + 1,
                    'config': config,
                    'metrics': metrics,
                    'score': score
                }
                results_list.append(result)
                
                if score > best_score:
                    best_score = score
                    best_config = config
                    
                logger.info(f"Trial {i+1} score: {score:.4f}")
                
            except Exception as e:
                logger.error(f"Trial {i+1} failed: {e}")
                continue
        
        # Save results
        os.makedirs(save_dir, exist_ok=True)
        
        optimization_results = {
            'best_config': best_config,
            'best_score': best_score,
            'optimization_metric': optimization_metric,
            'n_trials': len(results_list),
            'all_results': results_list,
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = os.path.join(
            save_dir, 
            f"rainbow_dqn_hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(results_path, 'w') as f:
            json.dump(optimization_results, f, indent=2)
        
        logger.info(f"Hyperparameter optimization completed")
        logger.info(f"Best score: {best_score:.4f}")
        logger.info(f"Best config: {best_config}")
        logger.info(f"Results saved to {results_path}")
        
        return optimization_results
    
    def curriculum_training(
        self,
        stages: List[Dict[str, Any]],
        save_dir: str = "models/rainbow_dqn_curriculum"
    ) -> Dict[str, Any]:
        """Train agent with curriculum learning.
        
        Args:
            stages: List of training stages with different configurations
            save_dir: Directory to save models
            
        Returns:
            Curriculum training results
        """
        logger.info(f"Starting curriculum training with {len(stages)} stages")
        
        os.makedirs(save_dir, exist_ok=True)
        curriculum_results = []
        
        for stage_idx, stage_config in enumerate(stages):
            logger.info(f"Starting curriculum stage {stage_idx + 1}/{len(stages)}")
            logger.info(f"Stage config: {stage_config}")
            
            # Update agent configuration for this stage
            for key, value in stage_config.get('agent_config', {}).items():
                setattr(self.agent.config, key, value)
            
            # Train for this stage
            stage_results = self.agent.train(
                env=self.train_env,
                total_timesteps=stage_config.get('timesteps', 500000),
                eval_env=self.val_env,
                eval_freq=stage_config.get('eval_freq', 25000),
                n_eval_episodes=stage_config.get('n_eval_episodes', 5),
                checkpoint_freq=stage_config.get('checkpoint_freq', 50000),
                checkpoint_path=os.path.join(save_dir, f"stage_{stage_idx + 1}")
            )
            
            # Save stage model
            stage_model_path = os.path.join(save_dir, f"rainbow_dqn_stage_{stage_idx + 1}.pth")
            self.agent.save_model(stage_model_path)
            
            stage_results['stage'] = stage_idx + 1
            stage_results['stage_config'] = stage_config
            curriculum_results.append(stage_results)
            
            logger.info(f"Stage {stage_idx + 1} completed")
        
        # Save curriculum results
        curriculum_summary = {
            'stages': curriculum_results,
            'total_stages': len(stages),
            'curriculum_timestamp': datetime.now().isoformat()
        }
        
        results_path = os.path.join(save_dir, "curriculum_results.json")
        with open(results_path, 'w') as f:
            json.dump(curriculum_summary, f, indent=2)
        
        logger.info(f"Curriculum training completed. Results saved to {results_path}")
        
        return curriculum_summary


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train Rainbow DQN Agent")
    parser.add_argument("--mode", choices=["train", "evaluate", "hyperopt", "curriculum"], 
                       default="train", help="Training mode")
    parser.add_argument("--timesteps", type=int, default=2000000, 
                       help="Total training timesteps")
    parser.add_argument("--symbols", nargs="+", 
                       default=["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"],
                       help="Trading symbols")
    parser.add_argument("--start-date", default="2018-01-01", 
                       help="Start date for data")
    parser.add_argument("--end-date", default="2023-12-31", 
                       help="End date for data")
    parser.add_argument("--save-dir", default="models/rainbow_dqn", 
                       help="Directory to save models")
    parser.add_argument("--resume-from", help="Path to resume training from")
    parser.add_argument("--model-path", help="Path to model for evaluation")
    parser.add_argument("--hyperopt-trials", type=int, default=50, 
                       help="Number of hyperopt trials")
    
    args = parser.parse_args()
    
    # Create configurations
    rainbow_config = RainbowDQNConfig(
        learning_rate=1e-4,
        batch_size=32,
        gamma=0.99,
        distributional=True,
        n_atoms=51,
        prioritized_replay=True,
        noisy=True,
        multi_step=3,
        target_update_interval=10000,
        learning_starts=10000,
        verbose=1
    )
    
    env_config = YFinanceConfig(
        initial_balance=100000.0,
        max_position_size=0.2,
        transaction_cost=0.001,
        lookback_window=60,
        reward_scaling=1000.0
    )
    
    # Create trainer
    trainer = RainbowDQNTrainer(
        config=rainbow_config,
        env_config=env_config,
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Execute based on mode
    if args.mode == "train":
        logger.info("Starting training mode")
        results = trainer.train_agent(
            total_timesteps=args.timesteps,
            save_dir=args.save_dir,
            resume_from=args.resume_from
        )
        
        # Evaluate after training
        eval_results = trainer.evaluate_agent()
        
        logger.info("Training and evaluation completed successfully")
        
    elif args.mode == "evaluate":
        logger.info("Starting evaluation mode")
        eval_results = trainer.evaluate_agent(
            model_path=args.model_path,
            n_episodes=100
        )
        
    elif args.mode == "hyperopt":
        logger.info("Starting hyperparameter optimization mode")
        hyperopt_results = trainer.hyperparameter_optimization(
            n_trials=args.hyperopt_trials
        )
        
    elif args.mode == "curriculum":
        logger.info("Starting curriculum training mode")
        
        # Define curriculum stages
        curriculum_stages = [
            {
                'timesteps': 500000,
                'agent_config': {
                    'learning_rate': 3e-4,
                    'exploration_fraction': 0.3,
                    'target_update_interval': 5000
                }
            },
            {
                'timesteps': 750000,
                'agent_config': {
                    'learning_rate': 1e-4,
                    'exploration_fraction': 0.2,
                    'target_update_interval': 10000
                }
            },
            {
                'timesteps': 750000,
                'agent_config': {
                    'learning_rate': 5e-5,
                    'exploration_fraction': 0.1,
                    'target_update_interval': 15000
                }
            }
        ]
        
        curriculum_results = trainer.curriculum_training(
            stages=curriculum_stages,
            save_dir=os.path.join(args.save_dir, "curriculum")
        )


if __name__ == "__main__":
    main()