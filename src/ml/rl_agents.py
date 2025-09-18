"""
Reinforcement Learning Agent Implementations.

This module implements PPO, SAC, TD3, and Rainbow DQN agents using Stable-Baselines3
for trading decision making in the AI trading platform.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy

from .base_models import BaseRLAgent


class RLAgentConfig:
    """Configuration for RL agents with comprehensive validation"""
    
    # Supported agent types
    SUPPORTED_AGENTS = {'PPO', 'SAC', 'TD3', 'DQN'}
    
    # Valid ranges for parameters
    PARAM_RANGES = {
        'learning_rate': (1e-6, 1e-1),
        'batch_size': (1, 10000),
        'buffer_size': (1000, 10000000),
        'gamma': (0.0, 1.0),
        'tau': (0.0, 1.0),
        'exploration_fraction': (0.0, 1.0),
        'exploration_initial_eps': (0.0, 1.0),
        'exploration_final_eps': (0.0, 1.0),
    }
    
    def __init__(
        self,
        agent_type: str,
        policy: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        buffer_size: int = 1000000,
        learning_starts: int = 100,
        gamma: float = 0.99,
        tau: float = 0.005,
        train_freq: int = 1,
        gradient_steps: int = 1,
        target_update_interval: int = 1,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10.0,
        tensorboard_log: Optional[str] = None,
        device: str = "auto",
        verbose: int = 1,
        seed: Optional[int] = None,
        **kwargs
    ):
        """Initialize RL agent configuration
        
        Args:
            agent_type: Type of agent ('PPO', 'SAC', 'TD3', 'DQN')
            policy: Policy network type
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            buffer_size: Size of replay buffer
            learning_starts: Number of steps before learning starts
            gamma: Discount factor
            tau: Soft update coefficient for target networks
            train_freq: Training frequency
            gradient_steps: Number of gradient steps per update
            target_update_interval: Target network update interval
            exploration_fraction: Fraction of training for exploration
            exploration_initial_eps: Initial exploration rate
            exploration_final_eps: Final exploration rate
            max_grad_norm: Maximum gradient norm for clipping
            tensorboard_log: TensorBoard log directory
            device: Device to use ('cpu', 'cuda', 'auto')
            verbose: Verbosity level
            seed: Random seed
            **kwargs: Additional agent-specific parameters
        """
        self.agent_type = agent_type.upper()
        self.policy = policy
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.gamma = gamma
        self.tau = tau
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.max_grad_norm = max_grad_norm
        self.tensorboard_log = tensorboard_log
        self.device = device
        self.verbose = verbose
        self.seed = seed
        
        # Store additional parameters
        self.additional_params = kwargs
        
        # Comprehensive validation
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        # Validate agent type
        if self.agent_type not in self.SUPPORTED_AGENTS:
            raise ValueError(
                f"Unsupported agent type: {self.agent_type}. "
                f"Supported types: {self.SUPPORTED_AGENTS}"
            )
        
        # Validate parameter ranges
        for param, (min_val, max_val) in self.PARAM_RANGES.items():
            if hasattr(self, param):
                value = getattr(self, param)
                if not min_val <= value <= max_val:
                    raise ValueError(
                        f"{param} must be between {min_val} and {max_val}, "
                        f"got {value}"
                    )
        
        # Validate device
        if self.device not in ["auto", "cpu", "cuda"]:
            raise ValueError(f"Invalid device: {self.device}")
        
        # Validate verbose level
        if not 0 <= self.verbose <= 2:
            raise ValueError(f"Verbose must be 0, 1, or 2, got {self.verbose}")
        
        # Validate seed
        if self.seed is not None and (self.seed < 0 or self.seed > 2**32 - 1):
            raise ValueError(f"Seed must be between 0 and 2^32-1, got {self.seed}")
        
        # Agent-specific validations
        self._validate_agent_specific_params()
    
    def _validate_agent_specific_params(self) -> None:
        """Validate agent-specific parameters"""
        if self.agent_type == 'PPO':
            # PPO uses n_steps, not buffer_size
            if 'n_steps' in self.additional_params:
                n_steps = self.additional_params['n_steps']
                if not 1 <= n_steps <= 100000:
                    raise ValueError(f"n_steps must be between 1 and 100000, got {n_steps}")
        
        elif self.agent_type in ['SAC', 'TD3']:
            # Off-policy algorithms need reasonable buffer sizes
            if self.buffer_size < 1000:
                raise ValueError(f"Buffer size too small for {self.agent_type}: {self.buffer_size}")
        
        elif self.agent_type == 'DQN':
            # DQN-specific validations
            if self.exploration_initial_eps < self.exploration_final_eps:
                raise ValueError(
                    "exploration_initial_eps must be >= exploration_final_eps"
                )


class TradingCallback(BaseCallback):
    """Custom callback for trading-specific monitoring"""
    
    def __init__(
        self,
        eval_env: gym.Env,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        verbose: int = 1
    ):
        """Initialize trading callback
        
        Args:
            eval_env: Environment for evaluation
            eval_freq: Evaluation frequency (in timesteps)
            n_eval_episodes: Number of episodes for evaluation
            log_path: Path to save evaluation logs
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path = log_path
        self.evaluations = []
        
        if log_path is not None:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    def _on_step(self) -> bool:
        """Called at each step"""
        if self.n_calls % self.eval_freq == 0:
            # Evaluate policy
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True
            )
            
            # Get portfolio metrics if available
            portfolio_metrics = {}
            if hasattr(self.eval_env, 'get_portfolio_metrics'):
                try:
                    portfolio_metrics = self.eval_env.get_portfolio_metrics()
                except:
                    pass
            
            # Log results
            evaluation = {
                'timesteps': self.n_calls,
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'portfolio_metrics': portfolio_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            self.evaluations.append(evaluation)
            
            # Log to TensorBoard if available
            if self.logger is not None:
                self.logger.record("eval/mean_reward", mean_reward)
                self.logger.record("eval/std_reward", std_reward)
                
                # Log portfolio metrics
                for key, value in portfolio_metrics.items():
                    if isinstance(value, (int, float)):
                        self.logger.record(f"eval/{key}", value)
            
            # Save to file if path provided
            if self.log_path is not None:
                with open(self.log_path, 'w') as f:
                    json.dump(self.evaluations, f, indent=2)
            
            if self.verbose > 0:
                print(f"Eval at {self.n_calls} steps: "
                      f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
        return True


class StableBaselinesRLAgent(BaseRLAgent):
    """Stable-Baselines3 RL agent wrapper"""
    
    def __init__(self, config: RLAgentConfig, env: gym.Env):
        """Initialize RL agent
        
        Args:
            config: Agent configuration
            env: Training environment
        """
        super().__init__(config.__dict__)
        self.config = config
        self.env = env
        self.model: Optional[BaseAlgorithm] = None
        self.training_history = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Create model
        self._create_model()
    
    def _create_model(self) -> None:
        """Create the RL model based on configuration"""
        # Prepare model parameters
        model_params = {
            'policy': self.config.policy,
            'env': self.env,
            'learning_rate': self.config.learning_rate,
            'gamma': self.config.gamma,
            'verbose': self.config.verbose,
            'tensorboard_log': self.config.tensorboard_log,
            'device': self.config.device,
            'seed': self.config.seed
        }
        
        # Add agent-specific parameters
        if self.config.agent_type == 'PPO':
            model_params.update({
                'batch_size': self.config.batch_size,
                'max_grad_norm': self.config.max_grad_norm,
                **self.config.additional_params
            })
            self.model = PPO(**model_params)
            
        elif self.config.agent_type == 'SAC':
            model_params.update({
                'buffer_size': self.config.buffer_size,
                'learning_starts': self.config.learning_starts,
                'batch_size': self.config.batch_size,
                'tau': self.config.tau,
                'train_freq': self.config.train_freq,
                'gradient_steps': self.config.gradient_steps,
                **self.config.additional_params
            })
            self.model = SAC(**model_params)
            
        elif self.config.agent_type == 'TD3':
            model_params.update({
                'buffer_size': self.config.buffer_size,
                'learning_starts': self.config.learning_starts,
                'batch_size': self.config.batch_size,
                'tau': self.config.tau,
                'train_freq': self.config.train_freq,
                'gradient_steps': self.config.gradient_steps,
                'target_policy_noise': 0.2,
                'target_noise_clip': 0.5,
                'policy_delay': 2,
                **self.config.additional_params
            })
            self.model = TD3(**model_params)
            
        elif self.config.agent_type == 'DQN':
            # Check if environment supports DQN (discrete action space)
            if not isinstance(self.env.action_space, gym.spaces.Discrete):
                raise ValueError(
                    f"DQN requires discrete action space, but got {type(self.env.action_space)}. "
                    "Use SAC or TD3 for continuous action spaces."
                )
            
            model_params.update({
                'buffer_size': self.config.buffer_size,
                'learning_starts': self.config.learning_starts,
                'batch_size': self.config.batch_size,
                'tau': self.config.tau,
                'train_freq': self.config.train_freq,
                'gradient_steps': self.config.gradient_steps,
                'target_update_interval': self.config.target_update_interval,
                'exploration_fraction': self.config.exploration_fraction,
                'exploration_initial_eps': self.config.exploration_initial_eps,
                'exploration_final_eps': self.config.exploration_final_eps,
                'max_grad_norm': self.config.max_grad_norm,
                **self.config.additional_params
            })
            self.model = DQN(**model_params)
        
        self.logger.info("Created %s agent", self.config.agent_type)
    
    def train(
        self,
        env: gym.Env,
        total_timesteps: int,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        checkpoint_freq: int = 50000,
        checkpoint_path: Optional[str] = None,
        log_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train the RL agent with comprehensive error handling and monitoring.
        
        Args:
            env: Training environment
            total_timesteps: Total training timesteps
            eval_env: Evaluation environment (optional)
            eval_freq: Evaluation frequency in timesteps
            n_eval_episodes: Number of evaluation episodes
            checkpoint_freq: Checkpoint saving frequency in timesteps
            checkpoint_path: Path to save checkpoints (optional)
            log_path: Path to save training logs (optional)
            
        Returns:
            Training results dictionary with metrics and metadata
            
        Raises:
            ValueError: If training parameters are invalid
            RuntimeError: If training fails due to environment or model issues
            MemoryError: If insufficient memory for training
        """
        # Validate training parameters
        if total_timesteps <= 0:
            raise ValueError(f"total_timesteps must be positive, got {total_timesteps}")
        
        if eval_freq <= 0:
            raise ValueError(f"eval_freq must be positive, got {eval_freq}")
        
        if n_eval_episodes <= 0:
            raise ValueError(f"n_eval_episodes must be positive, got {n_eval_episodes}")
        
        if self.model is None:
            raise RuntimeError("Model not initialized. Cannot start training.")
        
        # Import performance monitoring
        from ..utils.performance import performance_monitor, memory_manager
        # Setup callbacks
        callbacks = []
        
        # Evaluation callback
        if eval_env is not None:
            eval_callback = TradingCallback(
                eval_env=eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                log_path=log_path,
                verbose=self.config.verbose
            )
            callbacks.append(eval_callback)
        
        # Checkpoint callback
        if checkpoint_path is not None:
            os.makedirs(checkpoint_path, exist_ok=True)
            checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=checkpoint_path,
                name_prefix=f"{self.config.agent_type.lower()}_model"
            )
            callbacks.append(checkpoint_callback)
        
        # Combine callbacks
        callback = CallbackList(callbacks) if callbacks else None
        
        # Train the model
        start_time = datetime.now()
        self.logger.info("Starting training for %d timesteps", total_timesteps)
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=100,
            progress_bar=True
        )
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        self.is_trained = True
        
        # Collect training results
        results = {
            'agent_type': self.config.agent_type,
            'total_timesteps': total_timesteps,
            'training_time': training_time,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat()
        }
        
        # Add evaluation results if available
        if eval_env is not None and callbacks:
            eval_callback = callbacks[0] if isinstance(callbacks[0], TradingCallback) else None
            if eval_callback and eval_callback.evaluations:
                results['evaluations'] = eval_callback.evaluations
                results['final_mean_reward'] = eval_callback.evaluations[-1]['mean_reward']
        
        self.training_history.append(results)
        self.logger.info("Training completed in %.2f seconds", training_time)
        
        return results
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict action given observation
        
        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, log_prob or None)
        """
        if not self.is_trained:
            raise ValueError("Agent must be trained before making predictions")
        
        action, _states = self.model.predict(
            observation,
            deterministic=deterministic
        )
        
        return action, None  # SB3 doesn't return log probabilities directly
    
    def save_model(self, filepath: str) -> None:
        """Save RL model
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Validate and sanitize filepath
        filepath = self._validate_filepath(filepath)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        self.model.save(filepath)
        
        # Save additional metadata
        metadata = {
            'agent_type': self.config.agent_type,
            'config': self.config.__dict__,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = filepath + '_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info("Model saved to %s", filepath)
    
    def _validate_filepath(self, filepath: str) -> str:
        """Validate and sanitize file path using secure validation
        
        Args:
            filepath: Input file path
            
        Returns:
            Validated file path
            
        Raises:
            ValueError: If path is invalid or unsafe
        """
        from ..utils.security import file_validator, SecurityError
        
        try:
            validated_path = file_validator.validate_file_path(
                filepath, 
                file_type='models',
                create_dirs=True
            )
            return str(validated_path)
        except SecurityError as e:
            raise ValueError(f"Invalid file path: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error validating file path {filepath}: {e}")
            raise ValueError(f"File path validation failed: {e}")
    
    def load_model(self, filepath: str) -> None:
        """Load RL model
        
        Args:
            filepath: Path to load the model from
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model type is incompatible
            RuntimeError: If model loading fails
        """
        # Validate filepath
        filepath = self._validate_filepath(filepath)
        
        if not os.path.exists(filepath + ".zip"):
            raise FileNotFoundError(f"Model file not found: {filepath}.zip")
        
        try:
            # Load the model based on agent type
            if self.config.agent_type == 'PPO':
                self.model = PPO.load(filepath, env=self.env)
            elif self.config.agent_type == 'SAC':
                self.model = SAC.load(filepath, env=self.env)
            elif self.config.agent_type == 'TD3':
                self.model = TD3.load(filepath, env=self.env)
            elif self.config.agent_type == 'DQN':
                self.model = DQN.load(filepath, env=self.env)
            else:
                raise ValueError(f"Unsupported agent type for loading: {self.config.agent_type}")
        except Exception as e:
            self.logger.error("Failed to load model from %s: %s", filepath, str(e))
            raise RuntimeError(f"Model loading failed: {e}") from e
        
        # Load metadata if available
        metadata_path = filepath + '_metadata.json'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.is_trained = metadata.get('is_trained', False)
                self.training_history = metadata.get('training_history', [])
        else:
            self.is_trained = True  # Assume trained if we can load it
        
        self.logger.info("Model loaded from %s", filepath)
    
    def cleanup(self) -> None:
        """Clean up resources used by the agent"""
        if self.model is not None:
            # Clear model from memory
            del self.model
            self.model = None
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Agent resources cleaned up")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()


class RLAgentFactory:
    """Factory for creating RL agents"""
    
    @staticmethod
    def create_agent(
        agent_type: str,
        env: gym.Env,
        **config_kwargs
    ) -> StableBaselinesRLAgent:
        """Create an RL agent
        
        Args:
            agent_type: Type of agent ('PPO', 'SAC', 'TD3', 'DQN')
            env: Training environment
            **config_kwargs: Configuration parameters
            
        Returns:
            Configured RL agent
        """
        config = RLAgentConfig(agent_type=agent_type, **config_kwargs)
        return StableBaselinesRLAgent(config, env)
    
    @staticmethod
    def create_ppo_agent(
        env: gym.Env,
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        n_steps: int = 2048,
        n_epochs: int = 10,
        clip_range: float = 0.2,
        **kwargs
    ) -> StableBaselinesRLAgent:
        """Create PPO agent with optimized parameters for trading
        
        Args:
            env: Trading environment
            learning_rate: Learning rate
            batch_size: Batch size
            n_steps: Steps per rollout
            n_epochs: Training epochs per rollout
            clip_range: PPO clipping range
            **kwargs: Additional parameters
            
        Returns:
            Configured PPO agent
        """
        config_params = {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'n_steps': n_steps,
            'n_epochs': n_epochs,
            'clip_range': clip_range,
            **kwargs
        }
        
        return RLAgentFactory.create_agent('PPO', env, **config_params)
    
    @staticmethod
    def create_sac_agent(
        env: gym.Env,
        learning_rate: float = 3e-4,
        buffer_size: int = 1000000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        **kwargs
    ) -> StableBaselinesRLAgent:
        """Create SAC agent with optimized parameters for trading
        
        Args:
            env: Trading environment
            learning_rate: Learning rate
            buffer_size: Replay buffer size
            batch_size: Batch size
            tau: Soft update coefficient
            gamma: Discount factor
            train_freq: Training frequency
            gradient_steps: Gradient steps per update
            **kwargs: Additional parameters
            
        Returns:
            Configured SAC agent
        """
        config_params = {
            'learning_rate': learning_rate,
            'buffer_size': buffer_size,
            'batch_size': batch_size,
            'tau': tau,
            'gamma': gamma,
            'train_freq': train_freq,
            'gradient_steps': gradient_steps,
            **kwargs
        }
        
        return RLAgentFactory.create_agent('SAC', env, **config_params)
    
    @staticmethod
    def create_td3_agent(
        env: gym.Env,
        learning_rate: float = 1e-3,
        buffer_size: int = 1000000,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        **kwargs
    ) -> StableBaselinesRLAgent:
        """Create TD3 agent with optimized parameters for trading
        
        Args:
            env: Trading environment
            learning_rate: Learning rate
            buffer_size: Replay buffer size
            batch_size: Batch size
            tau: Soft update coefficient
            gamma: Discount factor
            train_freq: Training frequency
            gradient_steps: Gradient steps per update
            policy_delay: Policy update delay
            target_policy_noise: Target policy noise
            target_noise_clip: Target noise clipping
            **kwargs: Additional parameters
            
        Returns:
            Configured TD3 agent
        """
        config_params = {
            'learning_rate': learning_rate,
            'buffer_size': buffer_size,
            'batch_size': batch_size,
            'tau': tau,
            'gamma': gamma,
            'train_freq': train_freq,
            'gradient_steps': gradient_steps,
            'policy_delay': policy_delay,
            'target_policy_noise': target_policy_noise,
            'target_noise_clip': target_noise_clip,
            **kwargs
        }
        
        return RLAgentFactory.create_agent('TD3', env, **config_params)
    
    @staticmethod
    def create_dqn_agent(
        env: gym.Env,
        learning_rate: float = 1e-4,
        buffer_size: int = 1000000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int = 4,
        gradient_steps: int = 1,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        **kwargs
    ) -> StableBaselinesRLAgent:
        """Create DQN agent with optimized parameters for trading
        
        Note: DQN only supports discrete action spaces. For continuous action spaces,
        consider using SAC or TD3 instead.
        
        Args:
            env: Trading environment (must have discrete action space)
            learning_rate: Learning rate
            buffer_size: Replay buffer size
            batch_size: Batch size
            tau: Target network update rate
            gamma: Discount factor
            train_freq: Training frequency
            gradient_steps: Gradient steps per update
            target_update_interval: Target network update interval
            exploration_fraction: Exploration fraction
            exploration_initial_eps: Initial exploration rate
            exploration_final_eps: Final exploration rate
            **kwargs: Additional parameters
            
        Returns:
            Configured DQN agent
            
        Raises:
            ValueError: If environment has continuous action space
        """
        # Check if environment has discrete action space
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError(
                f"DQN only supports discrete action spaces, but got {type(env.action_space)}. "
                "For continuous action spaces, use SAC or TD3 instead."
            )
        
        config_params = {
            'learning_rate': learning_rate,
            'buffer_size': buffer_size,
            'batch_size': batch_size,
            'tau': tau,
            'gamma': gamma,
            'train_freq': train_freq,
            'gradient_steps': gradient_steps,
            'target_update_interval': target_update_interval,
            'exploration_fraction': exploration_fraction,
            'exploration_initial_eps': exploration_initial_eps,
            'exploration_final_eps': exploration_final_eps,
            **kwargs
        }
        
        return RLAgentFactory.create_agent('DQN', env, **config_params)


class RLAgentEnsemble:
    """Ensemble of RL agents with dynamic weighting"""
    
    def __init__(
        self,
        agents: List[StableBaselinesRLAgent],
        weighting_method: str = "performance",
        performance_window: int = 100,
        min_weight: float = 0.1
    ):
        """Initialize RL agent ensemble
        
        Args:
            agents: List of trained RL agents
            weighting_method: Method for weighting agents ('equal', 'performance', 'thompson')
            performance_window: Window size for performance tracking
            min_weight: Minimum weight for any agent
        """
        self.agents = agents
        self.weighting_method = weighting_method
        self.performance_window = performance_window
        self.min_weight = min_weight
        
        # Initialize weights
        self.weights = np.ones(len(agents)) / len(agents)
        
        # Performance tracking
        self.agent_rewards = [[] for _ in agents]
        self.ensemble_history = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Predict action using ensemble of agents
        
        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic policies
            
        Returns:
            Tuple of (ensemble_action, prediction_info)
        """
        if not self.agents:
            raise ValueError("No agents in ensemble")
        
        # Pre-allocate arrays for better performance
        num_agents = len(self.agents)
        action_shape = None
        actions = []
        
        # Get predictions from all agents
        for i, agent in enumerate(self.agents):
            try:
                action, _ = agent.predict(observation, deterministic=deterministic)
                if action_shape is None:
                    action_shape = action.shape
                    # Pre-allocate array now that we know the shape
                    actions_array = np.zeros((num_agents,) + action_shape)
                
                actions_array[i] = action
                actions.append(action)
            except Exception as e:
                self.logger.warning(
                    "Agent %d (%s) prediction failed: %s", 
                    i, agent.config.agent_type, str(e)
                )
                # Use zero action as fallback
                if action_shape is not None:
                    actions_array[i] = np.zeros(action_shape)
                    actions.append(np.zeros(action_shape))
        
        if not actions:
            raise RuntimeError("All agents failed to predict")
        
        # Convert to numpy array if we couldn't pre-allocate
        if action_shape is None:
            actions_array = np.array(actions)
        
        # Weighted ensemble prediction using vectorized operations
        if self.weighting_method == "equal":
            ensemble_action = np.mean(actions_array, axis=0)
        else:
            # Reshape weights for broadcasting
            weights_reshaped = self.weights.reshape(-1, *([1] * len(action_shape)))
            ensemble_action = np.sum(actions_array * weights_reshaped, axis=0)
        
        # Prediction info
        info = {
            'individual_actions': actions_array.tolist(),
            'weights': self.weights.tolist(),
            'weighting_method': self.weighting_method,
            'num_successful_predictions': len(actions)
        }
        
        return ensemble_action, info
    
    def update_performance(self, agent_rewards: List[float]) -> None:
        """Update agent performance tracking
        
        Args:
            agent_rewards: List of rewards for each agent
        """
        # Update reward history
        for i, reward in enumerate(agent_rewards):
            self.agent_rewards[i].append(reward)
            
            # Keep only recent performance
            if len(self.agent_rewards[i]) > self.performance_window:
                self.agent_rewards[i].pop(0)
        
        # Update weights based on performance
        self._update_weights()
    
    def _update_weights(self) -> None:
        """Update ensemble weights based on recent performance"""
        if self.weighting_method == "equal":
            self.weights = np.ones(len(self.agents)) / len(self.agents)
            
        elif self.weighting_method == "performance":
            # Weight based on recent average performance
            avg_rewards = []
            for rewards in self.agent_rewards:
                if rewards:
                    avg_rewards.append(np.mean(rewards))
                else:
                    avg_rewards.append(0.0)
            
            avg_rewards = np.array(avg_rewards)
            
            # Softmax weighting with temperature
            if np.std(avg_rewards) > 0:
                temperature = 0.1
                exp_rewards = np.exp(avg_rewards / temperature)
                self.weights = exp_rewards / np.sum(exp_rewards)
            else:
                self.weights = np.ones(len(self.agents)) / len(self.agents)
            
            # Ensure minimum weight
            self.weights = np.maximum(self.weights, self.min_weight)
            self.weights = self.weights / np.sum(self.weights)
            
        elif self.weighting_method == "thompson":
            # Thompson sampling based weighting
            # Simplified version using Beta distribution
            alpha = 1.0
            beta = 1.0
            
            samples = []
            for rewards in self.agent_rewards:
                if rewards:
                    successes = sum(1 for r in rewards if r > 0)
                    failures = len(rewards) - successes
                    sample = np.random.beta(alpha + successes, beta + failures)
                else:
                    sample = np.random.beta(alpha, beta)
                samples.append(sample)
            
            samples = np.array(samples)
            self.weights = samples / np.sum(samples)
    
    def get_ensemble_metrics(self) -> Dict[str, Any]:
        """Get ensemble performance metrics
        
        Returns:
            Dictionary of ensemble metrics
        """
        metrics = {
            'num_agents': len(self.agents),
            'current_weights': self.weights.tolist(),
            'weighting_method': self.weighting_method,
            'agent_types': [agent.config.agent_type for agent in self.agents]
        }
        
        # Add performance metrics if available
        if any(self.agent_rewards):
            agent_performance = []
            for i, rewards in enumerate(self.agent_rewards):
                if rewards:
                    perf = {
                        'agent_type': self.agents[i].config.agent_type,
                        'mean_reward': np.mean(rewards),
                        'std_reward': np.std(rewards),
                        'num_episodes': len(rewards)
                    }
                else:
                    perf = {
                        'agent_type': self.agents[i].config.agent_type,
                        'mean_reward': 0.0,
                        'std_reward': 0.0,
                        'num_episodes': 0
                    }
                agent_performance.append(perf)
            
            metrics['agent_performance'] = agent_performance
        
        return metrics
    
    def save_ensemble(self, filepath: str) -> None:
        """Save ensemble configuration and weights
        
        Args:
            filepath: Path to save ensemble data
        """
        ensemble_data = {
            'weighting_method': self.weighting_method,
            'performance_window': self.performance_window,
            'min_weight': self.min_weight,
            'weights': self.weights.tolist(),
            'agent_rewards': self.agent_rewards,
            'ensemble_history': self.ensemble_history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(ensemble_data, f, indent=2)
        
        self.logger.info("Ensemble saved to %s", filepath)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for the ensemble
        
        Returns:
            Dictionary containing performance metrics
        """
        metrics = {
            'ensemble_info': {
                'num_agents': len(self.agents),
                'weighting_method': self.weighting_method,
                'performance_window': self.performance_window,
                'min_weight': self.min_weight
            },
            'current_weights': self.weights.tolist(),
            'weight_statistics': {
                'max_weight': float(np.max(self.weights)),
                'min_weight': float(np.min(self.weights)),
                'weight_entropy': float(-np.sum(self.weights * np.log(self.weights + 1e-10))),
                'weight_std': float(np.std(self.weights))
            }
        }
        
        # Add agent-specific performance if available
        if any(self.agent_rewards):
            agent_stats = []
            for i, rewards in enumerate(self.agent_rewards):
                if rewards:
                    stats = {
                        'agent_index': i,
                        'agent_type': self.agents[i].config.agent_type,
                        'num_episodes': len(rewards),
                        'mean_reward': float(np.mean(rewards)),
                        'std_reward': float(np.std(rewards)),
                        'min_reward': float(np.min(rewards)),
                        'max_reward': float(np.max(rewards)),
                        'recent_trend': self._calculate_trend(rewards[-10:]) if len(rewards) >= 10 else 0.0
                    }
                else:
                    stats = {
                        'agent_index': i,
                        'agent_type': self.agents[i].config.agent_type,
                        'num_episodes': 0,
                        'mean_reward': 0.0,
                        'std_reward': 0.0,
                        'min_reward': 0.0,
                        'max_reward': 0.0,
                        'recent_trend': 0.0
                    }
                agent_stats.append(stats)
            
            metrics['agent_performance'] = agent_stats
        
        return metrics
    
    def _calculate_trend(self, rewards: List[float]) -> float:
        """Calculate trend in recent rewards using linear regression
        
        Args:
            rewards: List of recent rewards
            
        Returns:
            Trend slope (positive = improving, negative = declining)
        """
        if len(rewards) < 2:
            return 0.0
        
        x = np.arange(len(rewards))
        y = np.array(rewards)
        
        # Simple linear regression
        n = len(rewards)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        
        return float(slope)
    
    def load_ensemble(self, filepath: str) -> None:
        """Load ensemble configuration and weights
        
        Args:
            filepath: Path to load ensemble data from
        """
        with open(filepath, 'r') as f:
            ensemble_data = json.load(f)
        
        self.weighting_method = ensemble_data['weighting_method']
        self.performance_window = ensemble_data['performance_window']
        self.min_weight = ensemble_data['min_weight']
        self.weights = np.array(ensemble_data['weights'])
        self.agent_rewards = ensemble_data['agent_rewards']
        self.ensemble_history = ensemble_data.get('ensemble_history', [])
        
        self.logger.info("Ensemble loaded from %s", filepath)


def create_rl_ensemble(
    env: gym.Env,
    agent_configs: List[Dict[str, Any]],
    weighting_method: str = "performance"
) -> RLAgentEnsemble:
    """Create an ensemble of RL agents
    
    Args:
        env: Trading environment
        agent_configs: List of agent configuration dictionaries
        weighting_method: Ensemble weighting method
        
    Returns:
        Configured RL agent ensemble
    """
    agents = []
    
    for config in agent_configs:
        agent_type = config.pop('agent_type')
        agent = RLAgentFactory.create_agent(agent_type, env, **config)
        agents.append(agent)
    
    ensemble = RLAgentEnsemble(
        agents=agents,
        weighting_method=weighting_method
    )
    
    return ensemble