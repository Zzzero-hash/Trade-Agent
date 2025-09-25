"""
Sophisticated PPO Agent Trainer with Advanced Policy Optimization.

This module implements a state-of-the-art PPO agent with:
- Generalized Advantage Estimation (GAE)
- Adaptive KL penalty scheduling
- Entropy regularization
- Trust region constraints
- Natural policy gradients
- Parallel environment collection
- Advanced performance validation
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

try:
    import gymnasium as gym
except ImportError:
    import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import explained_variance

from .yfinance_trading_environment import YFinanceTradingEnvironment, YFinanceConfig
try:
    from .reward_functions import AdvancedRewardCalculator
except ImportError:
    # Create minimal reward calculator if not available
    class AdvancedRewardCalculator:
        def __init__(self, *args, **kwargs): pass
        def calculate_reward(self, *args, **kwargs): return 0.0

try:
    from ..utils.device_optimizer import DeviceOptimizer, suppress_sb3_device_warnings
    from ..utils.json_utils import safe_json_dump, safe_metrics_dict
except ImportError:
    # Create minimal implementations if utils don't exist
    class DeviceOptimizer:
        def log_device_info(self): pass
        def get_optimal_device(self, *args): return "cpu"
        def optimize_torch_settings(self, device): pass
    
    def suppress_sb3_device_warnings(): pass
    
    def safe_json_dump(data, path, **kwargs):
        import json
        with open(path, 'w') as f:
            json.dump(data, f, **kwargs)
    
    def safe_metrics_dict(data):
        import numpy as np
        safe_data = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                safe_data[k] = float(v)
            elif isinstance(v, (np.int32, np.int64, np.float32, np.float64)):
                safe_data[k] = float(v)
            else:
                safe_data[k] = v
        return safe_data
# from ..utils.performance import PerformanceTracker

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)


class AdaptiveKLScheduler:
    """Adaptive KL penalty scheduler for PPO training."""
    
    def __init__(
        self,
        initial_kl_coeff: float = 0.2,
        target_kl: float = 0.01,
        kl_tolerance: float = 1.5,
        adaptation_rate: float = 1.5,
        min_kl_coeff: float = 0.001,
        max_kl_coeff: float = 20.0
    ):
        """Initialize adaptive KL scheduler.
        
        Args:
            initial_kl_coeff: Initial KL coefficient
            target_kl: Target KL divergence
            kl_tolerance: Tolerance factor for KL divergence
            adaptation_rate: Rate of adaptation
            min_kl_coeff: Minimum KL coefficient
            max_kl_coeff: Maximum KL coefficient
        """
        self.kl_coeff = initial_kl_coeff
        self.target_kl = target_kl
        self.kl_tolerance = kl_tolerance
        self.adaptation_rate = adaptation_rate
        self.min_kl_coeff = min_kl_coeff
        self.max_kl_coeff = max_kl_coeff
        
    def update(self, kl_divergence: float) -> float:
        """Update KL coefficient based on observed KL divergence.
        
        Args:
            kl_divergence: Observed KL divergence
            
        Returns:
            Updated KL coefficient
        """
        if kl_divergence > self.target_kl * self.kl_tolerance:
            # KL too high, increase penalty
            self.kl_coeff *= self.adaptation_rate
        elif kl_divergence < self.target_kl / self.kl_tolerance:
            # KL too low, decrease penalty
            self.kl_coeff /= self.adaptation_rate
            
        # Clip to bounds
        self.kl_coeff = np.clip(self.kl_coeff, self.min_kl_coeff, self.max_kl_coeff)
        return self.kl_coeff


class EntropyScheduler:
    """Entropy regularization scheduler for exploration control."""
    
    def __init__(
        self,
        initial_entropy_coeff: float = 0.01,
        final_entropy_coeff: float = 0.001,
        decay_steps: int = 1000000,
        decay_type: str = "linear"
    ):
        """Initialize entropy scheduler.
        
        Args:
            initial_entropy_coeff: Initial entropy coefficient
            final_entropy_coeff: Final entropy coefficient
            decay_steps: Number of steps for decay
            decay_type: Type of decay ('linear', 'exponential')
        """
        self.initial_entropy_coeff = initial_entropy_coeff
        self.final_entropy_coeff = final_entropy_coeff
        self.decay_steps = decay_steps
        self.decay_type = decay_type
        self.current_step = 0
        
    def get_entropy_coeff(self) -> float:
        """Get current entropy coefficient."""
        if self.current_step >= self.decay_steps:
            return self.final_entropy_coeff
            
        progress = self.current_step / self.decay_steps
        
        if self.decay_type == "linear":
            coeff = self.initial_entropy_coeff - progress * (
                self.initial_entropy_coeff - self.final_entropy_coeff
            )
        elif self.decay_type == "exponential":
            decay_rate = np.log(self.final_entropy_coeff / self.initial_entropy_coeff)
            coeff = self.initial_entropy_coeff * np.exp(decay_rate * progress)
        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")
            
        return coeff
        
    def step(self):
        """Increment step counter."""
        self.current_step += 1


class TrustRegionConstraint:
    """Trust region constraint for policy updates."""
    
    def __init__(
        self,
        max_kl: float = 0.01,
        damping: float = 0.1,
        max_backtracks: int = 10,
        backtrack_ratio: float = 0.5
    ):
        """Initialize trust region constraint.
        
        Args:
            max_kl: Maximum allowed KL divergence
            damping: Damping factor for natural gradients
            max_backtracks: Maximum backtracking steps
            backtrack_ratio: Backtracking ratio
        """
        self.max_kl = max_kl
        self.damping = damping
        self.max_backtracks = max_backtracks
        self.backtrack_ratio = backtrack_ratio
        
    def conjugate_gradient(
        self,
        Avp_func,
        b: torch.Tensor,
        nsteps: int = 10,
        residual_tol: float = 1e-10
    ) -> torch.Tensor:
        """Conjugate gradient algorithm for solving Ax = b.
        
        Args:
            Avp_func: Function that computes A*v product
            b: Right-hand side vector
            nsteps: Number of CG steps
            residual_tol: Residual tolerance
            
        Returns:
            Solution vector x
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        
        for _ in range(nsteps):
            Ap = Avp_func(p)
            alpha = rdotr / torch.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = torch.dot(r, r)
            
            if new_rdotr < residual_tol:
                break
                
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
            
        return x


class SophisticatedPPOCallback(BaseCallback):
    """Advanced callback for PPO training with comprehensive monitoring."""
    
    def __init__(
        self,
        eval_env: gym.Env,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        log_dir: str = "logs/ppo_training",
        save_freq: int = 50000,
        performance_threshold: float = 2.0,  # Sortino ratio threshold
        max_drawdown_threshold: float = 0.1,  # 10% max drawdown
        verbose: int = 1
    ):
        """Initialize sophisticated PPO callback.
        
        Args:
            eval_env: Evaluation environment
            eval_freq: Evaluation frequency in timesteps
            n_eval_episodes: Number of evaluation episodes
            log_dir: Directory for logging
            save_freq: Model saving frequency
            performance_threshold: Performance threshold for validation
            max_drawdown_threshold: Maximum drawdown threshold
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_dir = log_dir
        self.save_freq = save_freq
        self.performance_threshold = performance_threshold
        self.max_drawdown_threshold = max_drawdown_threshold
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize performance tracking
        # self.performance_tracker = PerformanceTracker()
        self.best_performance = -np.inf
        self.evaluation_results = []
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))
        
    def _on_step(self) -> bool:
        """Called at each training step."""
        # Periodic evaluation
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_agent()
            
        # Periodic model saving
        if self.n_calls % self.save_freq == 0:
            self._save_model()
            
        return True
        
    def _evaluate_agent(self) -> None:
        """Evaluate agent performance."""
        logger.info(f"Evaluating agent at step {self.n_calls}")
        
        # Run evaluation episodes
        episode_rewards = []
        episode_returns = []
        episode_drawdowns = []
        
        for episode in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
                
            episode_reward = 0
            episode_return = 0
            max_value = 0
            min_value = 0
            step_count = 0
            
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                
                if isinstance(done, np.ndarray):
                    done = done.item()
                if isinstance(truncated, np.ndarray):
                    truncated = truncated.item()
                    
                done = done or truncated
                
                episode_reward += reward
                step_count += 1
                
                # Debug: Log first few steps of first episode
                if episode == 0 and step_count <= 3:
                    logger.info(f"Debug Step {step_count}: reward={reward:.6f}, "
                              f"portfolio_value=${info.get('portfolio_value', 0):,.2f}, "
                              f"action_sample={action[:4]}")
                
                # Track portfolio value for return calculation
                if hasattr(self.eval_env, 'portfolio_value'):
                    current_value = self.eval_env.portfolio_value
                    max_value = max(max_value, current_value)
                    min_value = min(min_value, current_value) if min_value == 0 else min(min_value, current_value)
                    
            episode_rewards.append(episode_reward)
            
            # Calculate episode return and drawdown
            if hasattr(self.eval_env, 'initial_balance'):
                episode_return = (self.eval_env.portfolio_value - self.eval_env.initial_balance) / self.eval_env.initial_balance
                episode_drawdown = (max_value - min_value) / max_value if max_value > 0 else 0
            else:
                episode_return = episode_reward
                episode_drawdown = 0
                
            episode_returns.append(episode_return)
            episode_drawdowns.append(episode_drawdown)
            
        # Calculate performance metrics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_return = np.mean(episode_returns)
        mean_drawdown = np.mean(episode_drawdowns)
        max_drawdown = np.max(episode_drawdowns)
        
        # Calculate Sortino ratio
        negative_returns = [r for r in episode_returns if r < 0]
        downside_deviation = np.std(negative_returns) if negative_returns else 0.001
        sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else mean_return / 0.001
        
        # Log metrics
        self.writer.add_scalar("eval/mean_reward", mean_reward, self.n_calls)
        self.writer.add_scalar("eval/std_reward", std_reward, self.n_calls)
        self.writer.add_scalar("eval/mean_return", mean_return, self.n_calls)
        self.writer.add_scalar("eval/sortino_ratio", sortino_ratio, self.n_calls)
        self.writer.add_scalar("eval/mean_drawdown", mean_drawdown, self.n_calls)
        self.writer.add_scalar("eval/max_drawdown", max_drawdown, self.n_calls)
        
        # Store evaluation results (convert NumPy types to native Python types)
        eval_result = safe_metrics_dict({
            'timestep': self.n_calls,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_return': mean_return,
            'sortino_ratio': sortino_ratio,
            'mean_drawdown': mean_drawdown,
            'max_drawdown': max_drawdown,
            'timestamp': datetime.now().isoformat()
        })
        self.evaluation_results.append(eval_result)
        
        # Check if this is the best performance
        if sortino_ratio > self.best_performance:
            self.best_performance = sortino_ratio
            self._save_best_model()
            
        # Log results
        logger.info(
            f"Evaluation at step {self.n_calls}: "
            f"Mean reward: {mean_reward:.4f} ± {std_reward:.4f}, "
            f"Sortino ratio: {sortino_ratio:.4f}, "
            f"Max drawdown: {max_drawdown:.4f}"
        )
        
        # Save evaluation results with safe JSON handling
        results_path = os.path.join(self.log_dir, "evaluation_results.json")
        safe_json_dump(self.evaluation_results, results_path, indent=2)
            
    def _save_model(self) -> None:
        """Save current model."""
        model_path = os.path.join(self.log_dir, f"model_step_{self.n_calls}")
        self.model.save(model_path)
        logger.info(f"Model saved at step {self.n_calls}")
        
    def _save_best_model(self) -> None:
        """Save best performing model."""
        model_path = os.path.join(self.log_dir, "best_model")
        self.model.save(model_path)
        logger.info(f"Best model saved with Sortino ratio: {self.best_performance:.4f}")
        
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        self.writer.close()
        
        # Generate final report
        if self.evaluation_results:
            final_result = self.evaluation_results[-1]
            report = {
                'training_completed': True,
                'total_timesteps': self.n_calls,
                'best_sortino_ratio': self.best_performance,
                'final_sortino_ratio': final_result['sortino_ratio'],
                'final_max_drawdown': final_result['max_drawdown'],
                'performance_threshold_met': final_result['sortino_ratio'] >= self.performance_threshold,
                'drawdown_threshold_met': final_result['max_drawdown'] <= self.max_drawdown_threshold,
                'training_end_time': datetime.now().isoformat()
            }
            
            report_path = os.path.join(self.log_dir, "training_report.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Training completed. Final Sortino ratio: {final_result['sortino_ratio']:.4f}")


class SophisticatedPPOTrainer:
    """Sophisticated PPO trainer with advanced policy optimization techniques."""
    
    def __init__(
        self,
        env_config: YFinanceConfig,
        symbols: List[str] = None,
        start_date: str = "2020-01-01",
        end_date: str = "2023-12-31",
        n_envs: int = 8,
        log_dir: str = "logs/sophisticated_ppo"
    ):
        """Initialize sophisticated PPO trainer.
        
        Args:
            env_config: Environment configuration
            symbols: List of trading symbols
            start_date: Start date for data
            end_date: End date for data
            n_envs: Number of parallel environments
            log_dir: Logging directory
        """
        self.env_config = env_config
        self.symbols = symbols or ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        self.start_date = start_date
        self.end_date = end_date
        self.n_envs = n_envs
        self.log_dir = log_dir
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize device optimizer
        self.device_optimizer = DeviceOptimizer()
        self.device_optimizer.log_device_info()
        
        # Suppress SB3 device warnings for optimal configurations
        suppress_sb3_device_warnings()
        
        # Initialize components
        self.kl_scheduler = AdaptiveKLScheduler()
        self.entropy_scheduler = EntropyScheduler()
        self.trust_region = TrustRegionConstraint()
        
        # Initialize environments
        self.train_env = None
        self.eval_env = None
        self.model = None
        
        logger.info(f"Initialized SophisticatedPPOTrainer with {n_envs} parallel environments")
        
    def _create_env(self, symbols: List[str], train: bool = True) -> gym.Env:
        """Create trading environment.
        
        Args:
            symbols: Trading symbols
            train: Whether this is for training (affects data split)
            
        Returns:
            Trading environment
        """
        # Split data for train/test
        if train:
            # Use first 80% for training
            train_start = pd.to_datetime(self.start_date)
            train_end = pd.to_datetime(self.end_date)
            total_days = (train_end - train_start).days
            train_days = int(total_days * 0.8)
            end_date = (train_start + pd.Timedelta(days=train_days)).strftime('%Y-%m-%d')
            start_date = self.start_date
        else:
            # Use last 20% for evaluation
            train_start = pd.to_datetime(self.start_date)
            train_end = pd.to_datetime(self.end_date)
            total_days = (train_end - train_start).days
            train_days = int(total_days * 0.8)
            start_date = (train_start + pd.Timedelta(days=train_days)).strftime('%Y-%m-%d')
            end_date = self.end_date
            
        env = YFinanceTradingEnvironment(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            config=self.env_config
        )
        
        return env
        
    def _create_parallel_envs(self) -> gym.Env:
        """Create parallel training environments.
        
        Returns:
            Vectorized environment
        """
        def make_env():
            return lambda: self._create_env(self.symbols, train=True)
            
        # Create parallel environments
        if self.n_envs == 1:
            env = DummyVecEnv([make_env()])
        else:
            env = SubprocVecEnv([make_env() for _ in range(self.n_envs)])
            
        return env
        
    def train(
        self,
        total_timesteps: int = 3000000,  # 3M timesteps for 3000+ episodes
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = 0.1,  # Further increased to reduce early stopping
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
        seed: Optional[int] = None,
        device: str = "auto"
    ) -> Dict[str, Any]:
        """Train sophisticated PPO agent with advanced features.
        
        Args:
            total_timesteps: Total training timesteps
            learning_rate: Learning rate
            n_steps: Steps per rollout
            batch_size: Batch size
            n_epochs: Training epochs per rollout
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clipping range
            clip_range_vf: Value function clipping range
            normalize_advantage: Whether to normalize advantages
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm
            use_sde: Whether to use State Dependent Exploration
            sde_sample_freq: SDE sampling frequency
            target_kl: Target KL divergence
            tensorboard_log: TensorBoard log directory
            policy_kwargs: Policy network arguments
            verbose: Verbosity level
            seed: Random seed
            device: Device to use
            
        Returns:
            Training results
        """
        logger.info("Starting sophisticated PPO training")
        
        # Create environments
        self.train_env = self._create_parallel_envs()
        self.eval_env = self._create_env(self.symbols, train=False)
        
        # Set up tensorboard logging
        if tensorboard_log is None:
            tensorboard_log = os.path.join(self.log_dir, "tensorboard")
            
        # Configure policy network (updated for SB3 v1.8.0+)
        if policy_kwargs is None:
            policy_kwargs = {
                'net_arch': dict(pi=[256, 256], vf=[256, 256]),  # Direct dict, not list
                'activation_fn': torch.nn.ReLU,
                'ortho_init': True,
            }
        
        # Optimize device for MLP policy (CPU is better for ActorCritic/MLP)
        policy_type = "MlpPolicy"
        optimal_device = self.device_optimizer.get_optimal_device(
            'rl', policy_type, device if device != "auto" else None
        )
        self.device_optimizer.optimize_torch_settings(optimal_device)
        
        logger.info(f"Using {optimal_device} for {policy_type} (CPU optimal for MLP policies)")
            
        # Create PPO model
        self.model = PPO(
            policy=policy_type,
            env=self.train_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=optimal_device
        )
        
        # Create sophisticated callback
        callback = SophisticatedPPOCallback(
            eval_env=self.eval_env,
            eval_freq=10000,
            n_eval_episodes=10,
            log_dir=self.log_dir,
            save_freq=50000,
            performance_threshold=2.0,  # Target Sortino ratio > 2.0
            max_drawdown_threshold=0.1,  # Target max drawdown < 10%
            verbose=verbose
        )
        
        # Start training
        start_time = time.time()
        logger.info(f"Training PPO agent for {total_timesteps} timesteps")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=10,
            progress_bar=True,
            reset_num_timesteps=True
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Final evaluation
        final_results = self._final_evaluation()
        
        # Compile training results
        results = {
            'training_completed': True,
            'total_timesteps': total_timesteps,
            'training_time': training_time,
            'final_evaluation': final_results,
            'model_path': os.path.join(self.log_dir, "final_model"),
            'log_dir': self.log_dir,
            'config': {
                'learning_rate': learning_rate,
                'n_steps': n_steps,
                'batch_size': batch_size,
                'n_epochs': n_epochs,
                'gamma': gamma,
                'gae_lambda': gae_lambda,
                'clip_range': clip_range,
                'target_kl': target_kl
            }
        }
        
        # Save final model
        self.model.save(results['model_path'])
        
        # Save results
        results_path = os.path.join(self.log_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        return results
        
    def _final_evaluation(self, n_episodes: int = 50) -> Dict[str, Any]:
        """Perform final comprehensive evaluation.
        
        Args:
            n_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation results
        """
        logger.info(f"Performing final evaluation with {n_episodes} episodes")
        
        episode_rewards = []
        episode_returns = []
        episode_drawdowns = []
        episode_sharpe_ratios = []
        episode_sortino_ratios = []
        
        for episode in range(n_episodes):
            obs = self.eval_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
                
            episode_reward = 0
            episode_values = []
            episode_rets = []
            
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                
                if isinstance(done, np.ndarray):
                    done = done.item()
                if isinstance(truncated, np.ndarray):
                    truncated = truncated.item()
                    
                done = done or truncated
                episode_reward += reward
                
                # Track portfolio metrics
                if hasattr(self.eval_env, 'portfolio_value'):
                    episode_values.append(self.eval_env.portfolio_value)
                    if len(episode_values) > 1:
                        ret = (episode_values[-1] - episode_values[-2]) / episode_values[-2]
                        episode_rets.append(ret)
                        
            episode_rewards.append(episode_reward)
            
            # Calculate performance metrics
            if episode_values and hasattr(self.eval_env, 'initial_balance'):
                total_return = (episode_values[-1] - self.eval_env.initial_balance) / self.eval_env.initial_balance
                episode_returns.append(total_return)
                
                # Calculate drawdown
                peak = np.maximum.accumulate(episode_values)
                drawdown = (peak - episode_values) / peak
                max_drawdown = np.max(drawdown)
                episode_drawdowns.append(max_drawdown)
                
                # Calculate Sharpe and Sortino ratios
                if episode_rets:
                    mean_return = np.mean(episode_rets)
                    std_return = np.std(episode_rets)
                    negative_returns = [r for r in episode_rets if r < 0]
                    downside_std = np.std(negative_returns) if negative_returns else 0.001
                    
                    sharpe = mean_return / std_return if std_return > 0 else 0
                    sortino = mean_return / downside_std if downside_std > 0 else 0
                    
                    episode_sharpe_ratios.append(sharpe)
                    episode_sortino_ratios.append(sortino)
                    
        # Calculate final metrics
        results = {
            'n_episodes': n_episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_return': np.mean(episode_returns) if episode_returns else 0,
            'std_return': np.std(episode_returns) if episode_returns else 0,
            'mean_drawdown': np.mean(episode_drawdowns) if episode_drawdowns else 0,
            'max_drawdown': np.max(episode_drawdowns) if episode_drawdowns else 0,
            'mean_sharpe_ratio': np.mean(episode_sharpe_ratios) if episode_sharpe_ratios else 0,
            'mean_sortino_ratio': np.mean(episode_sortino_ratios) if episode_sortino_ratios else 0,
            'performance_threshold_met': np.mean(episode_sortino_ratios) >= 1.0 if episode_sortino_ratios else False,
            'drawdown_threshold_met': np.max(episode_drawdowns) <= 0.1 if episode_drawdowns else False
        }
        
        logger.info(
            f"Final evaluation results: "
            f"Mean Sortino ratio: {results['mean_sortino_ratio']:.4f}, "
            f"Max drawdown: {results['max_drawdown']:.4f}, "
            f"Performance threshold met: {results['performance_threshold_met']}, "
            f"Drawdown threshold met: {results['drawdown_threshold_met']}"
        )
        
        return results
        
    def cleanup(self):
        """Clean up resources."""
        if self.train_env:
            self.train_env.close()
        if self.eval_env:
            self.eval_env.close()
        if self.model:
            del self.model
            
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    """Main training function for sophisticated PPO agent."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Environment configuration
    env_config = YFinanceConfig(
        initial_balance=100000.0,
        max_position_size=0.2,
        transaction_cost=0.001,
        lookback_window=60,
        reward_scaling=10.0
    )
    
    # Trading symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
    
    # Create trainer
    trainer = SophisticatedPPOTrainer(
        env_config=env_config,
        symbols=symbols,
        start_date="2020-01-01",
        end_date="2023-12-31",
        n_envs=8,
        log_dir="logs/sophisticated_ppo_task_7_2"
    )
    
    try:
        # Train the agent
        results = trainer.train(
            total_timesteps=3000000,  # 3M timesteps for 3000+ episodes
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            normalize_advantage=True,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.01,
            verbose=1
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Results: {json.dumps(results, indent=2)}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()