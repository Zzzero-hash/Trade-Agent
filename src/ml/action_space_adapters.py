"""
Action Space Adapters for Different RL Algorithms.

This module provides adapters to convert between continuous and discrete
action spaces for compatibility with different RL algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union
import numpy as np
import gymnasium as gym
from enum import IntEnum


class ActionType(IntEnum):
    """Discrete action types for trading."""
    HOLD = 0
    BUY_SMALL = 1
    BUY_MEDIUM = 2
    BUY_LARGE = 3
    SELL_SMALL = 4
    SELL_MEDIUM = 5
    SELL_LARGE = 6


class ActionSpaceAdapter(ABC):
    """Abstract base class for action space adapters."""
    
    @abstractmethod
    def continuous_to_discrete(self, continuous_action: np.ndarray) -> int:
        """Convert continuous action to discrete action."""
        pass
    
    @abstractmethod
    def discrete_to_continuous(self, discrete_action: int) -> np.ndarray:
        """Convert discrete action to continuous action."""
        pass
    
    @abstractmethod
    def get_discrete_action_space(self) -> gym.spaces.Discrete:
        """Get the discrete action space."""
        pass


class TradingActionAdapter(ActionSpaceAdapter):
    """Adapter for trading actions between continuous and discrete spaces."""
    
    def __init__(self, n_symbols: int, max_position_size: float = 0.2):
        """
        Initialize the trading action adapter.
        
        Args:
            n_symbols: Number of symbols to trade
            max_position_size: Maximum position size per symbol
        """
        self.n_symbols = n_symbols
        self.max_position_size = max_position_size
        
        # Define position size levels
        self.position_sizes = {
            ActionType.HOLD: 0.0,
            ActionType.BUY_SMALL: 0.05,
            ActionType.BUY_MEDIUM: 0.1,
            ActionType.BUY_LARGE: max_position_size,
            ActionType.SELL_SMALL: 0.05,
            ActionType.SELL_MEDIUM: 0.1,
            ActionType.SELL_LARGE: 1.0  # Sell all
        }
    
    def continuous_to_discrete(self, continuous_action: np.ndarray) -> List[int]:
        """
        Convert continuous action to discrete actions for each symbol.
        
        Args:
            continuous_action: Array of [action_type, position_size] for each symbol
            
        Returns:
            List of discrete actions for each symbol
        """
        discrete_actions = []
        
        for i in range(self.n_symbols):
            action_idx = i * 2
            action_type = int(np.clip(continuous_action[action_idx], 0, 2))
            position_size = np.clip(continuous_action[action_idx + 1], 0, self.max_position_size)
            
            if action_type == 0:  # HOLD
                discrete_action = ActionType.HOLD
            elif action_type == 1:  # BUY
                if position_size <= 0.05:
                    discrete_action = ActionType.BUY_SMALL
                elif position_size <= 0.1:
                    discrete_action = ActionType.BUY_MEDIUM
                else:
                    discrete_action = ActionType.BUY_LARGE
            else:  # SELL
                if position_size <= 0.05:
                    discrete_action = ActionType.SELL_SMALL
                elif position_size <= 0.1:
                    discrete_action = ActionType.SELL_MEDIUM
                else:
                    discrete_action = ActionType.SELL_LARGE
            
            discrete_actions.append(discrete_action)
        
        return discrete_actions
    
    def discrete_to_continuous(self, discrete_actions: Union[int, List[int]]) -> np.ndarray:
        """
        Convert discrete actions to continuous action array.
        
        Args:
            discrete_actions: Single discrete action or list for each symbol
            
        Returns:
            Continuous action array
        """
        if isinstance(discrete_actions, int):
            discrete_actions = [discrete_actions] * self.n_symbols
        
        continuous_action = []
        
        for discrete_action in discrete_actions:
            action_type = ActionType(discrete_action)
            
            if action_type == ActionType.HOLD:
                continuous_action.extend([0.0, 0.0])
            elif action_type in [ActionType.BUY_SMALL, ActionType.BUY_MEDIUM, ActionType.BUY_LARGE]:
                continuous_action.extend([1.0, self.position_sizes[action_type]])
            else:  # SELL actions
                continuous_action.extend([2.0, self.position_sizes[action_type]])
        
        return np.array(continuous_action, dtype=np.float32)
    
    def get_discrete_action_space(self) -> gym.spaces.MultiDiscrete:
        """Get the discrete action space for multiple symbols."""
        return gym.spaces.MultiDiscrete([len(ActionType)] * self.n_symbols)


class DiscreteActionTradingEnvironment(gym.Wrapper):
    """Wrapper to convert continuous trading environment to discrete actions."""
    
    def __init__(self, env, adapter: TradingActionAdapter):
        """
        Initialize discrete action trading environment.
        
        Args:
            env: Continuous action trading environment
            adapter: Action space adapter
        """
        super().__init__(env)
        self.adapter = adapter
        self.action_space = adapter.get_discrete_action_space()
    
    def step(self, action):
        """Step with discrete action converted to continuous."""
        continuous_action = self.adapter.discrete_to_continuous(action)
        return self.env.step(continuous_action)
    
    def reset(self, **kwargs):
        """Reset environment."""
        return self.env.reset(**kwargs)