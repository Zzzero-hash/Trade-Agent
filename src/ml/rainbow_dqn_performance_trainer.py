"""
Rainbow DQN Performance Trainer for Task 7.1.

This module implements comprehensive training for the Rainbow DQN agent with:
- C51 Distributional DQN for return distribution modeling
- Double DQN for reduced overestimation bias
- Dueling DQN for separate value and advantage estimation
- Prioritized Experience Replay for efficient learning
- Noisy Networks for parameter space exploration
- Multi-step learning for improved sample efficiency

The trainer specifically targets the requirements of Task 7.1:
- Train for 2000+ episodes until convergence
- Achieve >1.5 Sharpe ratio on training environment
- Comprehensive validation and performance tracking
"""

import os
import sys
import json
import logging
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import warnings

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ml.advanced_dqn_agent import RainbowDQNAgent, RainbowDQNConfig
from src.ml.yfinance_trading_environment import YFinanceTradingEnvironment, YFinanceConfig
from src.ml.discrete_trading_wrapper import DiscreteTradingWrapper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rainbow_dqn_task_7_1.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class Task71TrainingConfig:
    """Configuration specifically for Task 7.1 training requirements."""
    
    # Task 7.1 specific requirements
    target_episodes: int = 2000  # Minimum 2000 episodes
    target_sharpe_ratio: float = 1.5  # Must achieve >1.5 Sharpe ratio
    convergence_patience: int = 100  # Episodes to wait for convergence
    
    # Training parameters
    max_timesteps: int = 2000000  # Maximum timesteps (fallback)
    eval_frequency: int = 50  # Evaluate every N episodes
    eval_episodes: int = 10  # Episodes per evaluation
    
    # Environment settings
    symbols: List[str] = None
    start_date: str = "2020-01-01"
    end_date: str = "2023-12-31"
    
    # Model saving
    save_dir: str = "models/rainbow_dqn_task_7_1"
    checkpoint_frequency: int = 100  # Save every N episodes
    
    def __post_init__(self):
        """Initialize default values."""
        if self.symbols is None:
            self.symbols = [
                "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", 
                "AMZN", "TSLA", "NVDA", "META", "NFLX"
            ]


class RainbowDQNTask71Trainer:
    """
    Specialized trainer for Rainbow DQN Task 7.1 requirements.
    
    This trainer implements all Rainbow DQN components and specifically
    targets the performance requirements of Task 7.1.
    """
    
    def __init__(self, config: Task71TrainingConfig):
        """Initialize the Task 7.1 trainer.
        
        Args:
            config: Training configuration for Task 7.1
        """
        self.config = config
        
        # Create directories
        os.makedirs(self.config.save_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Initialize environments
        self._setup_environments()
        
        # Initialize Rainbow DQN agent with all components enabled
        self._setup_rainbow_agent()
        
        # Training state
        self.episode_count = 0
        self.total_timesteps = 0
        self.training_history = []
        self.evaluation_history = []
        self.best_sharpe_ratio = float('-inf')
        self.target_achieved = False
        self.convergence_counter = 0
        
        logger.info("Rainbow DQN Task 7.1 Trainer initialized")
        logger.info(f"Target: {self.config.target_episodes}+ episodes, >{self.config.target_sharpe_ratio} Sharpe ratio")
        logger.info(f"Trading symbols: {self.config.symbols}")
    
    def _setup_environments(self):
        """Setup training and evaluation environments."""
        # Environment configuration optimized for Rainbow DQN training
        env_config = YFinanceConfig(
            initial_balance=100000.0,
            max_position_size=0.15,  # Conservative position sizing
            transaction_cost=0.001,  # Realistic transaction costs
            lookback_window=60,
            reward_scaling=1000.0,
            sharpe_weight=0.4,  # Higher weight on Sharpe ratio
            return_weight=0.3,
            drawdown_penalty=2.0
        )
        
        # Create base trading environment
        base_env = YFinanceTradingEnvironment(
            symbols=self.config.symbols,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            config=env_config,
            data_source="yfinance"
        )
        
        # Wrap with discrete action wrapper for DQN
        self.env = DiscreteTradingWrapper(
            base_env,
            action_strategy="single_asset",
            position_sizes=[0.05, 0.10, 0.15],  # Conservative sizes
            sell_fractions=[0.25, 0.50, 0.75, 1.0]
        )
        
        # Create evaluation environment (same setup)
        eval_base_env = YFinanceTradingEnvironment(
            symbols=self.config.symbols,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            config=env_config,
            data_source="yfinance"
        )
        
        self.eval_env = DiscreteTradingWrapper(
            eval_base_env,
            action_strategy="single_asset",
            position_sizes=[0.05, 0.10, 0.15],
            sell_fractions=[0.25, 0.50, 0.75, 1.0]
        )
        
        logger.info(f"Environment setup complete - Action space: {self.env.action_space}")
    
    def _setup_rainbow_agent(self):
        """Setup Rainbow DQN agent with all components enabled."""
        # Rainbow DQN configuration with all features enabled
        rainbow_config = RainbowDQNConfig(
            # Network architecture
            hidden_dims=[512, 512, 256],
            dueling=True,  # Enable Dueling DQN
            noisy=True,   # Enable Noisy Networks
            
            # Distributional RL (C51)
            distributional=True,  # Enable C51
            n_atoms=51,
            v_min=-10.0,
            v_max=10.0,
            
            # Learning parameters
            learning_rate=1e-4,
            batch_size=32,
            gamma=0.99,
            tau=1.0,  # Hard updates for DQN
            
            # Prioritized Experience Replay
            prioritized_replay=True,  # Enable Prioritized Replay
            alpha=0.6,
            beta=0.4,
            beta_increment=0.001,
            
            # Multi-step learning
            multi_step=3,
            
            # Training schedule
            learning_starts=10000,
            train_freq=4,
            target_update_interval=10000,
            
            # Exploration (handled by noisy networks)
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.01,
            
            # Noisy networks
            noisy_std=0.5,
            
            # Regularization
            max_grad_norm=10.0,
            weight_decay=1e-5,
            
            # Device and logging
            device="auto",
            verbose=1,
            seed=42
        )
        
        # Create Rainbow DQN agent
        self.agent = RainbowDQNAgent(self.env, rainbow_config)
        
        logger.info("Rainbow DQN agent initialized with all components:")
        logger.info("- C51 Distributional DQN: ENABLED")
        logger.info("- Double DQN: ENABLED")
        logger.info("- Dueling Networks: ENABLED")
        logger.info("- Prioritized Experience Replay: ENABLED")
        logger.info("- Noisy Networks: ENABLED")
        logger.info("- Multi-step Learning: ENABLED")
    
    def train_rainbow_dqn(self) -> Dict[str, Any]:
        """
        Train Rainbow DQN agent to meet Task 7.1 requirements.
        
        Returns:
            Training results with validation of requirements
        """
        logger.info("Starting Rainbow DQN training for Task 7.1")
        logger.info(f"Target: {self.config.target_episodes}+ episodes")
        logger.info(f"Performance target: >{self.config.target_sharpe_ratio} Sharpe ratio")
        
        start_time = datetime.now()
        
        # Training loop - episode-based to meet task requirements
        while (self.episode_count < self.config.target_episodes or 
               not self.target_achieved):
            
            # Run training episode
            episode_results = self._run_training_episode()
            self.training_history.append(episode_results)
            
            # Evaluate periodically
            if self.episode_count % self.config.eval_frequency == 0:
                eval_results = self._evaluate_agent()
                self.evaluation_history.append(eval_results)
                
                # Check if target achieved
                if eval_results['sharpe_ratio'] >= self.config.target_sharpe_ratio:
                    if not self.target_achieved:
                        self.target_achieved = True
                        logger.info(f"🎯 TARGET ACHIEVED! Sharpe ratio: {eval_results['sharpe_ratio']:.4f}")
                        self._save_checkpoint("target_achieved")
                
                # Update best performance
                if eval_results['sharpe_ratio'] > self.best_sharpe_ratio:
                    self.best_sharpe_ratio = eval_results['sharpe_ratio']
                    self.convergence_counter = 0
                    self._save_checkpoint("best_performance")
                else:
                    self.convergence_counter += 1
                
                # Log progress
                logger.info(f"Episode {self.episode_count}: "
                          f"Sharpe {eval_results['sharpe_ratio']:.4f} "
                          f"(Best: {self.best_sharpe_ratio:.4f}, "
                          f"Target: {self.config.target_sharpe_ratio})")
            
            # Save checkpoint periodically
            if self.episode_count % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(f"episode_{self.episode_count}")
            
            # Check convergence
            if (self.convergence_counter >= self.config.convergence_patience and 
                self.episode_count >= self.config.target_episodes):
                logger.info(f"Training converged after {self.episode_count} episodes")
                break
            
            # Safety check for maximum timesteps
            if self.total_timesteps >= self.config.max_timesteps:
                logger.warning(f"Reached maximum timesteps: {self.config.max_timesteps}")
                break
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # Final evaluation
        final_eval = self._evaluate_agent(n_episodes=50)  # More thorough final eval
        
        # Compile results
        results = {
            'task_7_1_requirements': {
                'episodes_completed': self.episode_count,
                'target_episodes_met': self.episode_count >= self.config.target_episodes,
                'final_sharpe_ratio': final_eval['sharpe_ratio'],
                'target_sharpe_met': final_eval['sharpe_ratio'] >= self.config.target_sharpe_ratio,
                'target_achieved': self.target_achieved
            },
            'training_summary': {
                'total_episodes': self.episode_count,
                'total_timesteps': self.total_timesteps,
                'training_time_seconds': training_time,
                'best_sharpe_ratio': self.best_sharpe_ratio,
                'convergence_episodes': self.convergence_counter
            },
            'final_evaluation': final_eval,
            'rainbow_components': {
                'c51_distributional': True,
                'double_dqn': True,
                'dueling_networks': True,
                'prioritized_replay': True,
                'noisy_networks': True,
                'multi_step_learning': True
            }
        }
        
        # Save final results
        self._save_results(results)
        
        # Validation summary
        self._log_validation_summary(results)
        
        return results
    
    def _run_training_episode(self) -> Dict[str, Any]:
        """Run a single training episode."""
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            # Get action from agent
            action, _ = self.agent.predict(obs, deterministic=False)
            action = action[0] if isinstance(action, np.ndarray) else action
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store experience in replay buffer
            experience = (obs, action, reward, next_obs, done)
            if hasattr(self.agent, '_add_to_multi_step_buffer'):
                self.agent._add_to_multi_step_buffer(experience)
            
            # Train agent
            if (self.total_timesteps >= self.agent.config.learning_starts and 
                self.total_timesteps % self.agent.config.train_freq == 0):
                
                for _ in range(self.agent.config.gradient_steps):
                    loss_info = self.agent._train_step()
            
            # Update target network
            if (self.total_timesteps % self.agent.config.target_update_interval == 0):
                self.agent.target_network.load_state_dict(
                    self.agent.q_network.state_dict()
                )
            
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            self.total_timesteps += 1
        
        self.episode_count += 1
        
        return {
            'episode': self.episode_count,
            'reward': episode_reward,
            'steps': episode_steps,
            'total_timesteps': self.total_timesteps,
            'portfolio_value': info.get('portfolio_value', 0),
            'timestamp': datetime.now().isoformat()
        }
    
    def _evaluate_agent(self, n_episodes: int = None) -> Dict[str, Any]:
        """Evaluate agent performance."""
        if n_episodes is None:
            n_episodes = self.config.eval_episodes
        
        episode_rewards = []
        episode_returns = []
        portfolio_values = []
        
        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            done = False
            initial_value = None
            
            while not done:
                action, _ = self.agent.predict(obs, deterministic=True)
                action = action[0] if isinstance(action, np.ndarray) else action
                
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                
                if 'portfolio_value' in info:
                    if initial_value is None:
                        initial_value = info['portfolio_value']
                    portfolio_values.append(info['portfolio_value'])
            
            episode_rewards.append(episode_reward)
            
            # Calculate episode return
            if initial_value and portfolio_values:
                final_value = portfolio_values[-1]
                episode_return = (final_value - initial_value) / initial_value
                episode_returns.append(episode_return)
        
        # Calculate metrics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        # Calculate Sharpe ratio
        if len(episode_returns) > 1 and np.std(episode_returns) > 0:
            sharpe_ratio = np.mean(episode_returns) / np.std(episode_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Calculate other financial metrics
        if len(episode_returns) > 0:
            total_return = np.sum(episode_returns)
            win_rate = np.mean(np.array(episode_returns) > 0)
            max_return = np.max(episode_returns)
            min_return = np.min(episode_returns)
        else:
            total_return = 0.0
            win_rate = 0.0
            max_return = 0.0
            min_return = 0.0
        
        return {
            'episode': self.episode_count,
            'n_eval_episodes': n_episodes,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'win_rate': win_rate,
            'max_return': max_return,
            'min_return': min_return,
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            self.config.save_dir, 
            f"rainbow_dqn_{checkpoint_name}.pth"
        )
        self.agent.save_model(checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_name}")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save training results."""
        results_path = os.path.join(
            self.config.save_dir,
            f"task_7_1_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_path}")
    
    def _log_validation_summary(self, results: Dict[str, Any]):
        """Log validation summary for Task 7.1."""
        logger.info("=" * 60)
        logger.info("TASK 7.1 VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        req = results['task_7_1_requirements']
        
        # Episodes requirement
        episodes_status = "PASSED" if req['target_episodes_met'] else "FAILED"
        logger.info(f"Episodes Requirement: {episodes_status}")
        logger.info(f"  Target: {self.config.target_episodes}+ episodes")
        logger.info(f"  Achieved: {req['episodes_completed']} episodes")
        
        # Sharpe ratio requirement
        sharpe_status = "PASSED" if req['target_sharpe_met'] else "FAILED"
        logger.info(f"Sharpe Ratio Requirement: {sharpe_status}")
        logger.info(f"  Target: >{self.config.target_sharpe_ratio}")
        logger.info(f"  Achieved: {req['final_sharpe_ratio']:.4f}")
        
        # Rainbow components
        logger.info("Rainbow DQN Components:")
        components = results['rainbow_components']
        for component, enabled in components.items():
            status = "ENABLED" if enabled else "DISABLED"
            logger.info(f"  {component.replace('_', ' ').title()}: {status}")
        
        # Overall status
        overall_passed = req['target_episodes_met'] and req['target_sharpe_met']
        overall_status = "PASSED" if overall_passed else "FAILED"
        logger.info(f"Overall Task 7.1 Status: {overall_status}")
        
        if overall_passed:
            logger.info("Task 7.1 requirements successfully met!")
        else:
            logger.warning("Task 7.1 requirements not fully met")
        
        logger.info("=" * 60)


def main():
    """Main function to run Task 7.1 training."""
    # Configuration for Task 7.1
    config = Task71TrainingConfig(
        target_episodes=2000,
        target_sharpe_ratio=1.5,
        symbols=["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        start_date="2020-01-01",
        end_date="2023-12-31"
    )
    
    # Create trainer
    trainer = RainbowDQNTask71Trainer(config)
    
    # Run training
    try:
        results = trainer.train_rainbow_dqn()
        
        # Print final summary
        print("\n" + "="*60)
        print("TASK 7.1 TRAINING COMPLETED")
        print("="*60)
        print(f"Episodes: {results['task_7_1_requirements']['episodes_completed']}")
        print(f"Sharpe Ratio: {results['task_7_1_requirements']['final_sharpe_ratio']:.4f}")
        print(f"Target Met: {results['task_7_1_requirements']['target_achieved']}")
        print("="*60)
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()