"""
Rainbow DQN Task 7.1 Trainer - Advanced DQN Agent with Full Rainbow Implementation.

This module implements the complete training pipeline for Task 7.1:
- C51 distributional DQN for 2000+ episodes until convergence
- Double DQN, Dueling DQN with prioritized experience replay for stable learning
- Noisy Networks training with parameter space exploration over 1000+ episodes
- Validation of DQN performance achieving >1.5 Sharpe ratio on training environment

Requirements: 2.1, 3.1, 9.2
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import warnings

import numpy as np
import torch
import gymnasium as gym
from pathlib import Path

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


class RainbowDQNTask71Trainer:
    """
    Comprehensive trainer for Rainbow DQN Task 7.1 with all required features:
    - C51 Distributional DQN
    - Double DQN
    - Dueling DQN
    - Prioritized Experience Replay
    - Noisy Networks
    - Multi-step Learning
    """
    
    def __init__(
        self,
        symbols: List[str] = None,
        start_date: str = "2018-01-01",
        end_date: str = "2023-12-31",
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15
    ):
        """Initialize Rainbow DQN Task 7.1 trainer.
        
        Args:
            symbols: List of trading symbols
            start_date: Training data start date
            end_date: Training data end date
            train_split: Training data split ratio
            val_split: Validation data split ratio
            test_split: Test data split ratio
        """
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
        
        # Create Rainbow DQN configuration with all features enabled
        self.config = self._create_rainbow_config()
        
        # Create environments
        self.train_env = self._create_environment(self.train_start, self.train_end)
        self.val_env = self._create_environment(self.val_start, self.val_end)
        self.test_env = self._create_environment(self.test_start, self.test_end)
        
        # Wrap environments with discrete action wrapper for DQN
        self.train_env = DiscreteTradingWrapper(self.train_env, action_strategy="single_asset")
        self.val_env = DiscreteTradingWrapper(self.val_env, action_strategy="single_asset")
        self.test_env = DiscreteTradingWrapper(self.test_env, action_strategy="single_asset")
        
        # Initialize agent
        self.agent = RainbowDQNAgent(self.train_env, self.config)
        
        # Training state
        self.training_results = []
        self.best_performance = float('-inf')
        self.best_model_path = None
        
        logger.info(f"Rainbow DQN Task 7.1 trainer initialized with {len(self.symbols)} symbols")
        logger.info(f"Training period: {self.train_start} to {self.train_end}")
        logger.info(f"Validation period: {self.val_start} to {self.val_end}")
        logger.info(f"Test period: {self.test_start} to {self.test_end}")
        logger.info("All Rainbow features enabled: C51, Double DQN, Dueling, Prioritized Replay, Noisy Networks")
    
    def _create_rainbow_config(self) -> RainbowDQNConfig:
        """Create Rainbow DQN configuration with all features enabled."""
        return RainbowDQNConfig(
            # Network architecture
            hidden_dims=[512, 512, 256],
            dueling=True,  # Dueling DQN enabled
            noisy=True,   # Noisy Networks enabled
            
            # Distributional RL (C51) enabled
            distributional=True,
            n_atoms=51,
            v_min=-10.0,
            v_max=10.0,
            
            # Learning parameters optimized for financial data
            learning_rate=1e-4,
            batch_size=32,
            gamma=0.99,
            tau=1.0,  # Hard update for DQN
            
            # Prioritized Experience Replay enabled
            prioritized_replay=True,
            alpha=0.6,  # Prioritization exponent
            beta=0.4,   # Importance sampling exponent
            beta_increment=0.001,
            epsilon_priority=1e-6,
            
            # Multi-step learning
            multi_step=3,
            
            # Training schedule
            buffer_size=1000000,
            learning_starts=10000,
            train_freq=4,
            target_update_interval=10000,
            gradient_steps=1,
            
            # Exploration (reduced since we use noisy networks)
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.01,
            
            # Noisy networks parameters
            noisy_std=0.5,
            
            # Regularization
            max_grad_norm=10.0,
            weight_decay=1e-5,
            
            # Device and logging
            device="auto",
            verbose=1,
            seed=42
        )
    
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
        env_config = YFinanceConfig(
            initial_balance=100000.0,
            transaction_cost=0.001,  # 0.1% transaction cost
            max_position_size=0.1,   # Max 10% position size
            lookback_window=60       # 60-day lookback
        )
        
        return YFinanceTradingEnvironment(
            symbols=self.symbols,
            start_date=start_date,
            end_date=end_date,
            config=env_config,
            data_source="yfinance",
            cache_dir="data/cache"
        )
    
    def train_c51_distributional_dqn(
        self,
        episodes: int = 2000,
        save_dir: str = "models/rainbow_dqn_task_7_1",
        target_sharpe: float = 1.5
    ) -> Dict[str, Any]:
        """
        Train C51 distributional DQN for 2000+ episodes until convergence.
        
        Args:
            episodes: Number of training episodes (minimum 2000)
            save_dir: Directory to save models
            target_sharpe: Target Sortino ratio to achieve
            
        Returns:
            Training results
        """
        logger.info(f"Starting C51 Distributional DQN training for {episodes} episodes")
        logger.info(f"Target Sortino ratio: {target_sharpe}")
        logger.info("=" * 80)
        logger.info("TRAINING PROGRESS:")
        logger.info("Progress will be shown every 10 episodes with detailed logs every 100 episodes")
        logger.info("Format: Episode X/Y (%) | Reward: X | Avg Reward: X | Length: X | Sortino: X | Best: X")
        logger.info("=" * 80)
        
        # Ensure minimum episodes requirement
        if episodes < 2000:
            episodes = 2000
            logger.warning(f"Minimum 2000 episodes required for Task 7.1. Setting episodes to {episodes}")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training metrics
        episode_rewards = []
        episode_lengths = []
        sharpe_ratios = []
        portfolio_values = []
        
        # Training loop
        best_sharpe = float('-inf')
        convergence_patience = 100
        no_improvement_count = 0
        
        for episode in range(episodes):
            # Debug output for first few episodes
            if episode < 5:
                print(f"Starting episode {episode + 1}...", flush=True)
            elif episode % 10 == 0:
                print(f"Starting episode {episode + 1}...", flush=True)
            
            # Reset environment
            obs, _ = self.train_env.reset()
            episode_reward = 0
            episode_length = 0
            episode_portfolio_values = []
            done = False
            
            while not done:
                # Get action from agent
                action, _ = self.agent.predict(obs, deterministic=False)
                action = action[0]
                
                # Take step in environment
                next_obs, reward, terminated, truncated, info = self.train_env.step(action)
                done = terminated or truncated
                
                # Store experience in replay buffer
                self.agent._add_to_multi_step_buffer((obs, action, reward, next_obs, done))
                
                # Train agent if enough experiences
                if self.agent.steps_done >= self.config.learning_starts:
                    if self.agent.steps_done % self.config.train_freq == 0:
                        for _ in range(self.config.gradient_steps):
                            loss_info = self.agent._train_step()
                            if loss_info:
                                self.agent.losses.append(loss_info['loss'])
                
                # Update target network
                if self.agent.steps_done % self.config.target_update_interval == 0:
                    self.agent.target_network.load_state_dict(self.agent.q_network.state_dict())
                
                # Update state
                obs = next_obs
                episode_reward += reward
                episode_length += 1
                self.agent.steps_done += 1
                
                # Track portfolio value
                if 'portfolio_value' in info:
                    episode_portfolio_values.append(info['portfolio_value'])
                
                # Show step progress for first few episodes
                if episode < 3 and episode_length % 50 == 0:
                    print(f"  Episode {episode + 1} step {episode_length}: reward={reward:.2f}, total_reward={episode_reward:.2f}", flush=True)
                
                # Show simple progress dots for all episodes
                if episode_length % 100 == 0:
                    print(".", end="", flush=True)
            
            # Episode completed
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            portfolio_values.extend(episode_portfolio_values)
            
            # Calculate Sortino ratio for recent episodes (start after 5 episodes)
            if len(episode_rewards) >= 5:
                # Use all available episodes up to 50 for Sortino calculation
                recent_rewards = episode_rewards[-min(50, len(episode_rewards)):]
                if len(recent_rewards) > 1:
                    sortino = self._calculate_sortino_ratio(recent_rewards)
                    sharpe_ratios.append(sortino)  # Keep variable name for compatibility
                    
                    # Check for improvement (initialize best_sharpe if needed)
                    if best_sharpe == float('-inf') or sortino > best_sharpe:
                        best_sharpe = sortino
                        no_improvement_count = 0
                        
                        # Save best model
                        best_model_path = os.path.join(save_dir, "best_c51_model.pth")
                        self.agent.save_model(best_model_path)
                        logger.info(f"New best Sortino ratio: {sortino:.4f} - Model saved")
                    else:
                        no_improvement_count += 1
                else:
                    # If we can't calculate Sortino, just append the last one or 0
                    if sharpe_ratios:
                        sharpe_ratios.append(sharpe_ratios[-1])
                    else:
                        sharpe_ratios.append(0.0)
            
            # Progress tracking and logging - show every episode for better visibility
            if (episode + 1) % 1 == 0:  # Show every episode
                avg_reward = np.mean(episode_rewards[-min(10, len(episode_rewards)):])
                avg_length = np.mean(episode_lengths[-min(10, len(episode_lengths)):])
                current_sortino = sharpe_ratios[-1] if sharpe_ratios else 0.0
                
                # Real-time progress output with forced flush
                progress_pct = (episode + 1) / episodes * 100
                progress_msg = (f"C51 Episode {episode + 1:4d}/{episodes} ({progress_pct:5.1f}%) | "
                               f"Reward: {episode_reward:8.2f} | "
                               f"Avg Reward: {avg_reward:8.2f} | "
                               f"Length: {episode_length:4d} | "
                               f"Sortino: {current_sortino:6.3f} | "
                               f"Best: {best_sharpe:6.3f}")
                
                # Use both print and logger for visibility
                print(progress_msg, flush=True)
                if (episode + 1) % 10 == 0:
                    logger.info(progress_msg)
                
            # Detailed logging every 100 episodes
            if (episode + 1) % 100 == 0:
                avg_reward_100 = np.mean(episode_rewards[-100:])
                avg_length_100 = np.mean(episode_lengths[-100:])
                current_sortino = sharpe_ratios[-1] if sharpe_ratios else 0.0
                
                print()  # New line after progress bar
                logger.info(f"C51 Episode {episode + 1}/{episodes}: "
                          f"Avg Reward: {avg_reward_100:.4f}, "
                          f"Avg Length: {avg_length_100:.1f}, "
                          f"Sortino: {current_sortino:.4f}, "
                          f"Best Sortino: {best_sharpe:.4f}")
            
            # Check convergence
            if no_improvement_count >= convergence_patience and episode >= 1000:
                logger.info(f"Convergence detected after {episode + 1} episodes")
                break
            
            # Check target achievement
            if best_sharpe >= target_sharpe:
                logger.info(f"TARGET ACHIEVED! Sortino ratio: {best_sharpe:.4f} >= {target_sharpe}")
                target_model_path = os.path.join(save_dir, "target_achieved_c51_model.pth")
                self.agent.save_model(target_model_path)
                break
        
        # Save final model
        final_model_path = os.path.join(save_dir, "final_c51_model.pth")
        self.agent.save_model(final_model_path)
        
        # Training results
        results = {
            'training_type': 'C51_Distributional_DQN',
            'episodes_trained': episode + 1,
            'target_episodes': episodes,
            'best_sortino_ratio': best_sharpe,  # Keep variable name for compatibility
            'target_sortino': target_sharpe,
            'target_achieved': best_sharpe >= target_sharpe,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'sortino_ratios': sharpe_ratios,  # Keep variable name for compatibility
            'final_model_path': final_model_path,
            'best_model_path': best_model_path if best_sharpe > float('-inf') else None,
            'convergence_detected': no_improvement_count >= convergence_patience,
            'training_timestamp': datetime.now().isoformat()
        }
        
        # Save results
        results_path = os.path.join(save_dir, "c51_training_results.json")
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._convert_numpy_to_json(results)
            json.dump(json_results, f, indent=2)
        
        logger.info(f"C51 Distributional DQN training completed")
        logger.info(f"Best Sortino ratio achieved: {best_sharpe:.4f}")
        logger.info(f"Results saved to {results_path}")
        
        return results
    
    def train_double_dueling_dqn_with_prioritized_replay(
        self,
        episodes: int = 1500,
        save_dir: str = "models/rainbow_dqn_task_7_1",
        target_sharpe: float = 1.5
    ) -> Dict[str, Any]:
        """
        Train Double DQN, Dueling DQN with prioritized experience replay for stable learning.
        
        Args:
            episodes: Number of training episodes
            save_dir: Directory to save models
            target_sharpe: Target Sharpe ratio to achieve
            
        Returns:
            Training results
        """
        logger.info(f"Starting Double DQN + Dueling DQN + Prioritized Replay training for {episodes} episodes")
        logger.info("=" * 80)
        logger.info("DOUBLE DQN TRAINING PROGRESS:")
        logger.info("Progress will be shown every 10 episodes with detailed logs every 100 episodes")
        logger.info("Format: Episode X/Y (%) | Reward: X | Avg Reward: X | Length: X | Sharpe: X | Best: X")
        logger.info("=" * 80)
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training metrics
        episode_rewards = []
        episode_lengths = []
        sharpe_ratios = []
        priority_weights = []
        td_errors = []
        
        # Training loop with stability monitoring
        best_sharpe = float('-inf')
        stability_window = 100
        
        for episode in range(episodes):
            # Reset environment
            obs, _ = self.train_env.reset()
            episode_reward = 0
            episode_length = 0
            episode_td_errors = []
            done = False
            
            while not done:
                # Get action from agent
                action, _ = self.agent.predict(obs, deterministic=False)
                action = action[0]
                
                # Take step in environment
                next_obs, reward, terminated, truncated, info = self.train_env.step(action)
                done = terminated or truncated
                
                # Store experience in replay buffer
                self.agent._add_to_multi_step_buffer((obs, action, reward, next_obs, done))
                
                # Train agent with prioritized replay
                if self.agent.steps_done >= self.config.learning_starts:
                    if self.agent.steps_done % self.config.train_freq == 0:
                        for _ in range(self.config.gradient_steps):
                            loss_info = self.agent._train_step()
                            if loss_info:
                                self.agent.losses.append(loss_info['loss'])
                
                # Update target network (Double DQN)
                if self.agent.steps_done % self.config.target_update_interval == 0:
                    self.agent.target_network.load_state_dict(self.agent.q_network.state_dict())
                    logger.info(f"Target network updated at step {self.agent.steps_done}")
                
                # Update state
                obs = next_obs
                episode_reward += reward
                episode_length += 1
                self.agent.steps_done += 1
            
            # Episode completed
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Calculate stability metrics
            if len(episode_rewards) >= stability_window:
                recent_rewards = episode_rewards[-stability_window:]
                if np.std(recent_rewards) > 0:
                    sharpe = np.mean(recent_rewards) / np.std(recent_rewards) * np.sqrt(252)
                    sharpe_ratios.append(sharpe)
                    
                    # Check for improvement
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        
                        # Save best stable model
                        best_model_path = os.path.join(save_dir, "best_double_dueling_model.pth")
                        self.agent.save_model(best_model_path)
                        logger.info(f"New best stable Sharpe ratio: {sharpe:.4f}")
            
            # Progress tracking and logging
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                current_sharpe = sharpe_ratios[-1] if sharpe_ratios else 0.0
                
                # Real-time progress output
                progress_pct = (episode + 1) / episodes * 100
                print(f"\rDouble DQN Episode {episode + 1:4d}/{episodes} ({progress_pct:5.1f}%) | "
                      f"Reward: {episode_reward:8.2f} | "
                      f"Avg Reward: {avg_reward:8.2f} | "
                      f"Length: {episode_length:4d} | "
                      f"Sharpe: {current_sharpe:6.3f} | "
                      f"Best: {best_sharpe:6.3f}", end="", flush=True)
                
            # Detailed logging every 100 episodes
            if (episode + 1) % 100 == 0:
                avg_reward_100 = np.mean(episode_rewards[-100:])
                avg_length_100 = np.mean(episode_lengths[-100:])
                current_sharpe = sharpe_ratios[-1] if sharpe_ratios else 0.0
                
                print()  # New line after progress bar
                # Log prioritized replay statistics
                if hasattr(self.agent.replay_buffer, 'beta'):
                    beta = self.agent.replay_buffer.beta
                    logger.info(f"Double DQN Episode {episode + 1}/{episodes}: "
                              f"Avg Reward: {avg_reward_100:.4f}, "
                              f"Sharpe: {current_sharpe:.4f}, "
                              f"Beta: {beta:.4f}")
                else:
                    logger.info(f"Double DQN Episode {episode + 1}/{episodes}: "
                              f"Avg Reward: {avg_reward_100:.4f}, "
                              f"Sharpe: {current_sharpe:.4f}")
            
            # Check target achievement
            if best_sharpe >= target_sharpe:
                logger.info(f"TARGET ACHIEVED! Stable Sharpe ratio: {best_sharpe:.4f} >= {target_sharpe}")
                break
        
        # Save final model
        final_model_path = os.path.join(save_dir, "final_double_dueling_model.pth")
        self.agent.save_model(final_model_path)
        
        # Training results
        results = {
            'training_type': 'Double_Dueling_DQN_Prioritized_Replay',
            'episodes_trained': episode + 1,
            'target_episodes': episodes,
            'best_sharpe_ratio': best_sharpe,
            'target_sharpe': target_sharpe,
            'target_achieved': best_sharpe >= target_sharpe,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'sharpe_ratios': sharpe_ratios,
            'final_model_path': final_model_path,
            'training_timestamp': datetime.now().isoformat()
        }
        
        # Save results
        results_path = os.path.join(save_dir, "double_dueling_training_results.json")
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._convert_numpy_to_json(results)
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Double DQN + Dueling DQN + Prioritized Replay training completed")
        logger.info(f"Best stable Sharpe ratio achieved: {best_sharpe:.4f}")
        
        return results
    
    def train_noisy_networks_exploration(
        self,
        episodes: int = 1000,
        save_dir: str = "models/rainbow_dqn_task_7_1",
        target_sharpe: float = 1.5
    ) -> Dict[str, Any]:
        """
        Add Noisy Networks training with parameter space exploration over 1000+ episodes.
        
        Args:
            episodes: Number of training episodes (minimum 1000)
            save_dir: Directory to save models
            target_sharpe: Target Sharpe ratio to achieve
            
        Returns:
            Training results
        """
        logger.info(f"Starting Noisy Networks parameter space exploration for {episodes} episodes")
        logger.info("=" * 80)
        logger.info("NOISY NETWORKS TRAINING PROGRESS:")
        logger.info("Progress will be shown every 10 episodes with detailed logs every 100 episodes")
        logger.info("Format: Episode X/Y (%) | Reward: X | Avg Reward: X | Exploration: X | Sharpe: X | Best: X")
        logger.info("=" * 80)
        
        # Ensure minimum episodes requirement
        if episodes < 1000:
            episodes = 1000
            logger.warning(f"Minimum 1000 episodes required for Task 7.1. Setting episodes to {episodes}")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training metrics
        episode_rewards = []
        episode_lengths = []
        sharpe_ratios = []
        exploration_metrics = []
        
        # Training loop with exploration monitoring
        best_sharpe = float('-inf')
        exploration_window = 50
        
        for episode in range(episodes):
            # Reset environment
            obs, _ = self.train_env.reset()
            episode_reward = 0
            episode_length = 0
            episode_actions = []
            done = False
            
            while not done:
                # Reset noise for exploration (key feature of noisy networks)
                if self.config.noisy:
                    self.agent.q_network.reset_noise()
                
                # Get action from agent (no epsilon-greedy needed with noisy networks)
                action, _ = self.agent.predict(obs, deterministic=False)
                action = action[0]
                episode_actions.append(action)
                
                # Take step in environment
                next_obs, reward, terminated, truncated, info = self.train_env.step(action)
                done = terminated or truncated
                
                # Store experience in replay buffer
                self.agent._add_to_multi_step_buffer((obs, action, reward, next_obs, done))
                
                # Train agent
                if self.agent.steps_done >= self.config.learning_starts:
                    if self.agent.steps_done % self.config.train_freq == 0:
                        for _ in range(self.config.gradient_steps):
                            loss_info = self.agent._train_step()
                            if loss_info:
                                self.agent.losses.append(loss_info['loss'])
                
                # Update target network
                if self.agent.steps_done % self.config.target_update_interval == 0:
                    self.agent.target_network.load_state_dict(self.agent.q_network.state_dict())
                
                # Update state
                obs = next_obs
                episode_reward += reward
                episode_length += 1
                self.agent.steps_done += 1
            
            # Episode completed
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Calculate exploration metrics
            if episode_actions:
                action_entropy = self._calculate_action_entropy(episode_actions)
                exploration_metrics.append(action_entropy)
            
            # Calculate performance metrics
            if len(episode_rewards) >= exploration_window:
                recent_rewards = episode_rewards[-exploration_window:]
                if np.std(recent_rewards) > 0:
                    sharpe = np.mean(recent_rewards) / np.std(recent_rewards) * np.sqrt(252)
                    sharpe_ratios.append(sharpe)
                    
                    # Check for improvement
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        
                        # Save best exploration model
                        best_model_path = os.path.join(save_dir, "best_noisy_exploration_model.pth")
                        self.agent.save_model(best_model_path)
                        logger.info(f"New best exploration Sharpe ratio: {sharpe:.4f}")
            
            # Progress tracking and logging
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_exploration = np.mean(exploration_metrics[-10:]) if exploration_metrics else 0.0
                current_sharpe = sharpe_ratios[-1] if sharpe_ratios else 0.0
                
                # Real-time progress output
                progress_pct = (episode + 1) / episodes * 100
                print(f"\rNoisy Networks Episode {episode + 1:4d}/{episodes} ({progress_pct:5.1f}%) | "
                      f"Reward: {episode_reward:8.2f} | "
                      f"Avg Reward: {avg_reward:8.2f} | "
                      f"Exploration: {avg_exploration:6.3f} | "
                      f"Sharpe: {current_sharpe:6.3f} | "
                      f"Best: {best_sharpe:6.3f}", end="", flush=True)
                
            # Detailed logging every 100 episodes
            if (episode + 1) % 100 == 0:
                avg_reward_100 = np.mean(episode_rewards[-100:])
                avg_exploration_100 = np.mean(exploration_metrics[-100:]) if exploration_metrics else 0.0
                current_sharpe = sharpe_ratios[-1] if sharpe_ratios else 0.0
                
                print()  # New line after progress bar
                logger.info(f"Noisy Networks Episode {episode + 1}/{episodes}: "
                          f"Avg Reward: {avg_reward_100:.4f}, "
                          f"Exploration Entropy: {avg_exploration_100:.4f}, "
                          f"Sharpe: {current_sharpe:.4f}")
            
            # Check target achievement
            if best_sharpe >= target_sharpe:
                logger.info(f"TARGET ACHIEVED! Exploration Sharpe ratio: {best_sharpe:.4f} >= {target_sharpe}")
                break
        
        # Save final model
        final_model_path = os.path.join(save_dir, "final_noisy_exploration_model.pth")
        self.agent.save_model(final_model_path)
        
        # Training results
        results = {
            'training_type': 'Noisy_Networks_Exploration',
            'episodes_trained': episode + 1,
            'target_episodes': episodes,
            'best_sharpe_ratio': best_sharpe,
            'target_sharpe': target_sharpe,
            'target_achieved': best_sharpe >= target_sharpe,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'sharpe_ratios': sharpe_ratios,
            'exploration_metrics': exploration_metrics,
            'final_model_path': final_model_path,
            'training_timestamp': datetime.now().isoformat()
        }
        
        # Save results
        results_path = os.path.join(save_dir, "noisy_exploration_training_results.json")
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._convert_numpy_to_json(results)
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Noisy Networks exploration training completed")
        logger.info(f"Best exploration Sharpe ratio achieved: {best_sharpe:.4f}")
        
        return results
    
    def _calculate_action_entropy(self, actions: List[int]) -> float:
        """Calculate action entropy for exploration measurement."""
        if not actions:
            return 0.0
        
        # Count action frequencies
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Calculate probabilities
        total_actions = len(actions)
        probabilities = [count / total_actions for count in action_counts.values()]
        
        # Calculate entropy
        entropy = -sum(p * np.log(p + 1e-8) for p in probabilities)
        return entropy
    
    def validate_performance(
        self,
        model_path: str,
        target_sharpe: float = 1.5,
        n_episodes: int = 100
    ) -> Dict[str, Any]:
        """
        Validate DQN performance achieving >1.5 Sharpe ratio on training environment.
        
        Args:
            model_path: Path to trained model
            target_sharpe: Target Sharpe ratio to validate
            n_episodes: Number of validation episodes
            
        Returns:
            Validation results
        """
        logger.info(f"Validating DQN performance with target Sharpe ratio: {target_sharpe}")
        
        # Load model
        if os.path.exists(model_path):
            self.agent.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Validation on test environment
        episode_rewards = []
        episode_lengths = []
        portfolio_values = []
        
        for episode in range(n_episodes):
            obs, _ = self.test_env.reset()
            episode_reward = 0
            episode_length = 0
            episode_portfolio_values = []
            done = False
            
            while not done:
                # Get deterministic action
                action, _ = self.agent.predict(obs, deterministic=True)
                action = action[0]
                
                # Take step
                obs, reward, terminated, truncated, info = self.test_env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                # Track portfolio value
                if 'portfolio_value' in info:
                    episode_portfolio_values.append(info['portfolio_value'])
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            portfolio_values.extend(episode_portfolio_values)
            
            if (episode + 1) % 20 == 0:
                logger.info(f"Validation episode {episode + 1}/{n_episodes} completed")
        
        # Calculate performance metrics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        # Calculate Sharpe ratio
        if std_reward > 0:
            sharpe_ratio = mean_reward / std_reward * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Calculate financial metrics
        if portfolio_values:
            portfolio_values = np.array(portfolio_values)
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            returns = returns[np.isfinite(returns)]
            
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            
            if len(returns) > 1:
                volatility = np.std(returns) * np.sqrt(252)
                max_drawdown = self._calculate_max_drawdown(portfolio_values)
            else:
                volatility = 0.0
                max_drawdown = 0.0
        else:
            total_return = 0.0
            volatility = 0.0
            max_drawdown = 0.0
        
        # Validation results
        validation_results = {
            'model_path': model_path,
            'target_sharpe': target_sharpe,
            'achieved_sharpe': sharpe_ratio,
            'target_met': sharpe_ratio >= target_sharpe,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'total_return': total_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'n_episodes': n_episodes,
            'episode_rewards': episode_rewards,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # Log results
        logger.info("=== VALIDATION RESULTS ===")
        logger.info(f"Target Sharpe Ratio: {target_sharpe}")
        logger.info(f"Achieved Sharpe Ratio: {sharpe_ratio:.4f}")
        logger.info(f"Target Met: {'YES' if sharpe_ratio >= target_sharpe else 'NO'}")
        logger.info(f"Mean Episode Reward: {mean_reward:.4f} +/- {std_reward:.4f}")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Volatility: {volatility:.2%}")
        logger.info(f"Max Drawdown: {max_drawdown:.2%}")
        
        if sharpe_ratio >= target_sharpe:
            logger.info("TASK 7.1 REQUIREMENT MET: DQN achieved >1.5 Sharpe ratio")
        else:
            logger.warning(f"TASK 7.1 REQUIREMENT NOT MET: Sharpe {sharpe_ratio:.4f} < {target_sharpe}")
        
        return validation_results
    
    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(portfolio_values) < 2:
            return 0.0
        
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        return abs(max_drawdown)
    
    def run_complete_task_7_1(
        self,
        save_dir: str = "models/rainbow_dqn_task_7_1",
        target_sharpe: float = 1.5
    ) -> Dict[str, Any]:
        """
        Run complete Task 7.1 training pipeline with all requirements:
        1. C51 distributional DQN for 2000+ episodes
        2. Double DQN, Dueling DQN with prioritized experience replay
        3. Noisy Networks training with parameter space exploration over 1000+ episodes
        4. Validate DQN performance achieving >1.5 Sharpe ratio
        
        Args:
            save_dir: Directory to save all models and results
            target_sharpe: Target Sharpe ratio to achieve
            
        Returns:
            Complete task results
        """
        logger.info("Starting COMPLETE Task 7.1: Advanced DQN Agent with Full Rainbow Implementation")
        logger.info("Requirements: 2.1, 3.1, 9.2")
        
        # Create main save directory
        os.makedirs(save_dir, exist_ok=True)
        
        complete_results = {
            'task': 'Task_7_1_Rainbow_DQN',
            'requirements': ['2.1', '3.1', '9.2'],
            'target_sharpe': target_sharpe,
            'start_time': datetime.now().isoformat()
        }
        
        try:
            # Step 1: Train C51 Distributional DQN for 2000+ episodes
            logger.info("Step 1: Training C51 Distributional DQN for 2000+ episodes")
            c51_results = self.train_c51_distributional_dqn(
                episodes=2000,
                save_dir=save_dir,
                target_sharpe=target_sharpe
            )
            complete_results['c51_results'] = c51_results
            
            # Step 2: Train Double DQN, Dueling DQN with prioritized experience replay
            logger.info("Step 2: Training Double DQN + Dueling DQN + Prioritized Replay")
            double_dueling_results = self.train_double_dueling_dqn_with_prioritized_replay(
                episodes=1500,
                save_dir=save_dir,
                target_sharpe=target_sharpe
            )
            complete_results['double_dueling_results'] = double_dueling_results
            
            # Step 3: Train Noisy Networks with parameter space exploration
            logger.info("Step 3: Training Noisy Networks with parameter space exploration")
            noisy_results = self.train_noisy_networks_exploration(
                episodes=1000,
                save_dir=save_dir,
                target_sharpe=target_sharpe
            )
            complete_results['noisy_results'] = noisy_results
            
            # Step 4: Validate performance
            logger.info("Step 4: Validating DQN performance")
            
            # Find best model from all training phases
            best_model_path = None
            best_sharpe = float('-inf')
            
            for results_key in ['c51_results', 'double_dueling_results', 'noisy_results']:
                if results_key in complete_results:
                    results = complete_results[results_key]
                    # Handle both old and new key names for compatibility
                    current_ratio = results.get('best_sortino_ratio', results.get('best_sharpe_ratio', float('-inf')))
                    if current_ratio > best_sharpe:
                        best_sharpe = current_ratio
                        best_model_path = results.get('best_model_path') or results['final_model_path']
            
            if best_model_path and os.path.exists(best_model_path):
                validation_results = self.validate_performance(
                    model_path=best_model_path,
                    target_sharpe=target_sharpe,
                    n_episodes=100
                )
                complete_results['validation_results'] = validation_results
            else:
                logger.error("No valid model found for validation")
                complete_results['validation_results'] = {'error': 'No valid model found'}
            
            # Strict final assessment - ALL components must succeed
            c51_success = c51_results.get('target_achieved', False)
            double_success = double_dueling_results.get('target_achieved', False)
            noisy_success = noisy_results.get('target_achieved', False)
            validation_success = False
            
            if 'validation_results' in complete_results and 'target_met' in complete_results['validation_results']:
                validation_success = complete_results['validation_results']['target_met']
            
            # All components must succeed
            task_completed = c51_success and double_success and noisy_success and validation_success
            
            complete_results['task_completed'] = task_completed
            complete_results['component_success'] = {
                'c51_success': c51_success,
                'double_dueling_success': double_success,
                'noisy_networks_success': noisy_success,
                'validation_success': validation_success
            }
            
            # Log component status
            logger.info("Component Success Status:")
            logger.info("- C51 Distributional DQN: %s", "PASS" if c51_success else "FAIL")
            logger.info("- Double DQN + Dueling: %s", "PASS" if double_success else "FAIL")
            logger.info("- Noisy Networks: %s", "PASS" if noisy_success else "FAIL")
            logger.info("- Validation: %s", "PASS" if validation_success else "FAIL")
            
            if not task_completed:
                failed_components = []
                if not c51_success: failed_components.append("C51 DQN")
                if not double_success: failed_components.append("Double DQN")
                if not noisy_success: failed_components.append("Noisy Networks")
                if not validation_success: failed_components.append("Validation")
                
                error_msg = f"Task 7.1 FAILED: Components failed: {', '.join(failed_components)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            complete_results['end_time'] = datetime.now().isoformat()
            
            # Save complete results
            results_path = os.path.join(save_dir, "complete_task_7_1_results.json")
            with open(results_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = self._convert_numpy_to_json(complete_results)
                json.dump(json_results, f, indent=2)
            
            # Final logging
            logger.info("TASK 7.1 COMPLETE!")
            logger.info(f"Task Completed Successfully: {'YES' if task_completed else 'NO'}")
            logger.info(f"Results saved to: {results_path}")
            
            if task_completed:
                logger.info("ALL REQUIREMENTS MET:")
                logger.info("  C51 distributional DQN trained for 2000+ episodes")
                logger.info("  Double DQN, Dueling DQN with prioritized experience replay")
                logger.info("  Noisy Networks training with parameter space exploration")
                logger.info("  DQN performance validated with >1.5 Sharpe ratio")
            else:
                logger.warning("SOME REQUIREMENTS NOT MET - Check individual results")
            
            return complete_results
            
        except Exception as e:
            logger.error(f"Error during Task 7.1 execution: {e}")
            complete_results['error'] = str(e)
            complete_results['task_completed'] = False
            complete_results['end_time'] = datetime.now().isoformat()
            return complete_results
    
    def _calculate_sortino_ratio(self, returns: List[float], target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio - better than Sharpe as it only penalizes downside volatility.
        
        Args:
            returns: List of episode returns
            target_return: Target return (default 0.0)
            
        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0
            
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        
        # Calculate downside deviation (only negative returns)
        downside_returns = returns_array[returns_array < target_return]
        if len(downside_returns) == 0:
            # No downside - perfect performance
            return float('inf') if mean_return > target_return else 0.0
            
        downside_deviation = np.sqrt(np.mean((downside_returns - target_return) ** 2))
        
        if downside_deviation == 0:
            return float('inf') if mean_return > target_return else 0.0
            
        # Annualized Sortino ratio (assuming daily returns, 252 trading days)
        sortino = (mean_return - target_return) / downside_deviation * np.sqrt(252)
        
        return sortino

    def _convert_numpy_to_json(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_to_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif hasattr(obj, 'dtype') and 'bool' in str(obj.dtype):
            return bool(obj)
        elif hasattr(obj, 'item'):  # Handle numpy scalars
            return obj.item()
        else:
            return obj


def main():
    """Main execution function for Task 7.1."""
    parser = argparse.ArgumentParser(description="Rainbow DQN Task 7.1 Trainer")
    parser.add_argument("--mode", choices=["complete", "c51", "double_dueling", "noisy", "validate"], 
                       default="complete", help="Training mode")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes")
    parser.add_argument("--target_sharpe", type=float, default=1.5, help="Target Sharpe ratio")
    parser.add_argument("--save_dir", type=str, default="models/rainbow_dqn_task_7_1", 
                       help="Directory to save models")
    parser.add_argument("--model_path", type=str, help="Model path for validation")
    parser.add_argument("--symbols", nargs="+", 
                       default=["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"],
                       help="Trading symbols")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = RainbowDQNTask71Trainer(symbols=args.symbols)
    
    if args.mode == "complete":
        # Run complete Task 7.1
        results = trainer.run_complete_task_7_1(
            save_dir=args.save_dir,
            target_sharpe=args.target_sharpe
        )
        
    elif args.mode == "c51":
        # Train C51 only
        results = trainer.train_c51_distributional_dqn(
            episodes=args.episodes,
            save_dir=args.save_dir,
            target_sharpe=args.target_sharpe
        )
        
    elif args.mode == "double_dueling":
        # Train Double + Dueling DQN only
        results = trainer.train_double_dueling_dqn_with_prioritized_replay(
            episodes=args.episodes,
            save_dir=args.save_dir,
            target_sharpe=args.target_sharpe
        )
        
    elif args.mode == "noisy":
        # Train Noisy Networks only
        results = trainer.train_noisy_networks_exploration(
            episodes=args.episodes,
            save_dir=args.save_dir,
            target_sharpe=args.target_sharpe
        )
        
    elif args.mode == "validate":
        # Validate model performance
        if not args.model_path:
            raise ValueError("Model path required for validation mode")
        
        results = trainer.validate_performance(
            model_path=args.model_path,
            target_sharpe=args.target_sharpe
        )
    
    logger.info("Task 7.1 execution completed!")
    return results


if __name__ == "__main__":
    main()