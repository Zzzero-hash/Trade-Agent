"""
Discrete Action Wrapper for Trading Environment.

This wrapper converts the continuous action space of the trading environment
to a discrete action space suitable for DQN agents while maintaining
sophisticated trading strategies.
"""

import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)


class DiscreteTradingWrapper(gym.Wrapper):
    """
    Wrapper to convert continuous trading actions to discrete actions.
    
    The wrapper creates a discrete action space with predefined trading strategies:
    - HOLD: Do nothing
    - BUY_SMALL: Buy small position (5% of portfolio)
    - BUY_MEDIUM: Buy medium position (10% of portfolio) 
    - BUY_LARGE: Buy large position (20% of portfolio)
    - SELL_SMALL: Sell small portion (25% of position)
    - SELL_MEDIUM: Sell medium portion (50% of position)
    - SELL_LARGE: Sell large portion (75% of position)
    - SELL_ALL: Sell entire position (100% of position)
    
    For multi-asset environments, actions are applied to the most promising asset
    based on technical indicators or sequentially across assets.
    """
    
    def __init__(
        self,
        env: gym.Env,
        action_strategy: str = "single_asset",
        position_sizes: List[float] = None,
        sell_fractions: List[float] = None
    ):
        """Initialize discrete trading wrapper.
        
        Args:
            env: Base trading environment
            action_strategy: Strategy for multi-asset trading
                - "single_asset": Trade one asset at a time
                - "portfolio": Portfolio-level actions
                - "sequential": Sequential actions across assets
            position_sizes: List of position sizes for buy actions
            sell_fractions: List of sell fractions for sell actions
        """
        super().__init__(env)
        
        self.action_strategy = action_strategy
        self.position_sizes = position_sizes or [0.05, 0.10, 0.20]
        self.sell_fractions = sell_fractions or [0.25, 0.50, 0.75, 1.0]
        
        # Get environment info
        self.n_symbols = getattr(env, 'n_symbols', 1)
        self.symbols = getattr(env, 'symbols', ['ASSET'])
        
        # Ensure action space is compatible
        expected_action_dim = self.n_symbols * 2
        actual_action_dim = env.action_space.shape[0]
        
        if actual_action_dim != expected_action_dim:
            logger.warning(f"Action space mismatch: expected {expected_action_dim}, got {actual_action_dim}")
            # Adjust n_symbols to match actual action space
            self.n_symbols = actual_action_dim // 2
            self.symbols = [f'ASSET_{i}' for i in range(self.n_symbols)]
        
        # Define discrete actions
        self._setup_action_mapping()
        
        # Create discrete action space
        self.action_space = spaces.Discrete(len(self.action_mapping))
        
        logger.info(f"Discrete trading wrapper initialized with {len(self.action_mapping)} actions")
        logger.info(f"Action strategy: {action_strategy}")
        logger.info(f"Trading {self.n_symbols} symbols: {self.symbols}")
    
    def _setup_action_mapping(self):
        """Setup mapping from discrete actions to continuous actions."""
        self.action_mapping = {}
        action_id = 0
        
        if self.action_strategy == "single_asset":
            # Single asset trading with different position sizes
            
            # HOLD action
            self.action_mapping[action_id] = {
                'type': 'HOLD',
                'description': 'Hold all positions',
                'continuous_action': self._create_hold_action()
            }
            action_id += 1
            
            # BUY actions for each symbol and position size
            for symbol_idx, symbol in enumerate(self.symbols):
                for size_idx, position_size in enumerate(self.position_sizes):
                    self.action_mapping[action_id] = {
                        'type': 'BUY',
                        'symbol': symbol,
                        'symbol_idx': symbol_idx,
                        'position_size': position_size,
                        'description': f'Buy {position_size:.1%} of {symbol}',
                        'continuous_action': self._create_buy_action(symbol_idx, position_size)
                    }
                    action_id += 1
            
            # SELL actions for each symbol and sell fraction
            for symbol_idx, symbol in enumerate(self.symbols):
                for frac_idx, sell_fraction in enumerate(self.sell_fractions):
                    self.action_mapping[action_id] = {
                        'type': 'SELL',
                        'symbol': symbol,
                        'symbol_idx': symbol_idx,
                        'sell_fraction': sell_fraction,
                        'description': f'Sell {sell_fraction:.1%} of {symbol} position',
                        'continuous_action': self._create_sell_action(symbol_idx, sell_fraction)
                    }
                    action_id += 1
        
        elif self.action_strategy == "portfolio":
            # Portfolio-level actions
            
            # HOLD
            self.action_mapping[action_id] = {
                'type': 'HOLD',
                'description': 'Hold portfolio',
                'continuous_action': self._create_hold_action()
            }
            action_id += 1
            
            # Portfolio rebalancing actions
            strategies = [
                ('EQUAL_WEIGHT', 'Equal weight all assets'),
                ('MOMENTUM', 'Weight by momentum'),
                ('MEAN_REVERT', 'Mean reversion strategy'),
                ('RISK_PARITY', 'Risk parity weighting'),
                ('CASH', 'Move to cash')
            ]
            
            for strategy_name, description in strategies:
                self.action_mapping[action_id] = {
                    'type': 'PORTFOLIO',
                    'strategy': strategy_name,
                    'description': description,
                    'continuous_action': self._create_portfolio_action(strategy_name)
                }
                action_id += 1
        
        elif self.action_strategy == "sequential":
            # Sequential actions across assets
            
            # HOLD
            self.action_mapping[action_id] = {
                'type': 'HOLD',
                'description': 'Hold all positions',
                'continuous_action': self._create_hold_action()
            }
            action_id += 1
            
            # Combined buy/sell actions
            combined_actions = [
                ('BUY_SELL_ROTATE', 'Buy best, sell worst'),
                ('BUY_TOP_2', 'Buy top 2 performers'),
                ('SELL_BOTTOM_2', 'Sell bottom 2 performers'),
                ('MOMENTUM_TRADE', 'Trade based on momentum'),
                ('CONTRARIAN_TRADE', 'Contrarian trading')
            ]
            
            for action_name, description in combined_actions:
                self.action_mapping[action_id] = {
                    'type': 'COMBINED',
                    'strategy': action_name,
                    'description': description,
                    'continuous_action': self._create_combined_action(action_name)
                }
                action_id += 1
    
    def _create_hold_action(self) -> np.ndarray:
        """Create hold action (all zeros)."""
        action_dim = self.env.action_space.shape[0]
        return np.zeros(action_dim)
    
    def _create_buy_action(self, symbol_idx: int, position_size: float) -> np.ndarray:
        """Create buy action for specific symbol."""
        action_dim = self.env.action_space.shape[0]
        action = np.zeros(action_dim)
        
        # Ensure we don't exceed action dimensions
        if symbol_idx * 2 + 1 < action_dim:
            action[symbol_idx * 2] = 1.0  # BUY action type
            action[symbol_idx * 2 + 1] = position_size  # Position size
        
        return action
    
    def _create_sell_action(self, symbol_idx: int, sell_fraction: float) -> np.ndarray:
        """Create sell action for specific symbol."""
        action_dim = self.env.action_space.shape[0]
        action = np.zeros(action_dim)
        
        # Ensure we don't exceed action dimensions
        if symbol_idx * 2 + 1 < action_dim:
            action[symbol_idx * 2] = 2.0  # SELL action type
            action[symbol_idx * 2 + 1] = sell_fraction  # Sell fraction
        
        return action
    
    def _create_portfolio_action(self, strategy: str) -> np.ndarray:
        """Create portfolio-level action based on strategy."""
        action = np.zeros(self.n_symbols * 2)
        
        if strategy == 'EQUAL_WEIGHT':
            # Equal weight across all assets
            weight_per_asset = 1.0 / self.n_symbols
            for i in range(self.n_symbols):
                action[i * 2] = 1.0  # BUY
                action[i * 2 + 1] = weight_per_asset
        
        elif strategy == 'CASH':
            # Sell all positions
            for i in range(self.n_symbols):
                action[i * 2] = 2.0  # SELL
                action[i * 2 + 1] = 1.0  # Sell all
        
        elif strategy == 'MOMENTUM':
            # Buy top performer, sell bottom performer
            # This would need market data to implement properly
            # For now, use a simple heuristic
            action[0] = 1.0  # BUY first asset
            action[1] = 0.15  # 15% position
            if self.n_symbols > 1:
                action[-2] = 2.0  # SELL last asset
                action[-1] = 0.5  # Sell 50%
        
        elif strategy == 'MEAN_REVERT':
            # Contrarian strategy
            action[-2] = 1.0  # BUY last asset (contrarian)
            action[-1] = 0.10  # 10% position
            if self.n_symbols > 1:
                action[0] = 2.0  # SELL first asset
                action[1] = 0.3  # Sell 30%
        
        elif strategy == 'RISK_PARITY':
            # Risk parity weighting (simplified)
            weight_per_asset = 0.8 / self.n_symbols  # 80% invested
            for i in range(self.n_symbols):
                action[i * 2] = 1.0  # BUY
                action[i * 2 + 1] = weight_per_asset
        
        return action
    
    def _create_combined_action(self, strategy: str) -> np.ndarray:
        """Create combined trading action."""
        action = np.zeros(self.n_symbols * 2)
        
        if strategy == 'BUY_SELL_ROTATE':
            # Buy first asset, sell last asset
            action[0] = 1.0  # BUY first
            action[1] = 0.15  # 15% position
            if self.n_symbols > 1:
                action[-2] = 2.0  # SELL last
                action[-1] = 0.5  # Sell 50%
        
        elif strategy == 'BUY_TOP_2':
            # Buy top 2 assets
            for i in range(min(2, self.n_symbols)):
                action[i * 2] = 1.0  # BUY
                action[i * 2 + 1] = 0.10  # 10% each
        
        elif strategy == 'SELL_BOTTOM_2':
            # Sell bottom 2 assets
            start_idx = max(0, self.n_symbols - 2)
            for i in range(start_idx, self.n_symbols):
                action[i * 2] = 2.0  # SELL
                action[i * 2 + 1] = 0.4  # Sell 40%
        
        elif strategy == 'MOMENTUM_TRADE':
            # Momentum-based trading
            # Buy first half, sell second half
            mid_point = self.n_symbols // 2
            for i in range(mid_point):
                action[i * 2] = 1.0  # BUY
                action[i * 2 + 1] = 0.08  # 8% each
            for i in range(mid_point, self.n_symbols):
                action[i * 2] = 2.0  # SELL
                action[i * 2 + 1] = 0.3  # Sell 30%
        
        elif strategy == 'CONTRARIAN_TRADE':
            # Contrarian trading
            # Sell first half, buy second half
            mid_point = self.n_symbols // 2
            for i in range(mid_point):
                action[i * 2] = 2.0  # SELL
                action[i * 2 + 1] = 0.25  # Sell 25%
            for i in range(mid_point, self.n_symbols):
                action[i * 2] = 1.0  # BUY
                action[i * 2 + 1] = 0.12  # 12% each
        
        return action
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute discrete action by converting to continuous action."""
        # Validate action
        if not 0 <= action < len(self.action_mapping):
            raise ValueError(f"Invalid action {action}. Must be in range [0, {len(self.action_mapping)})")
        
        # Get continuous action
        action_info = self.action_mapping[action]
        continuous_action = action_info['continuous_action']
        
        # Execute in base environment
        obs, reward, terminated, truncated, info = self.env.step(continuous_action)
        
        # Add action info to info dict
        info['discrete_action'] = action
        info['action_type'] = action_info['type']
        info['action_description'] = action_info['description']
        
        return obs, reward, terminated, truncated, info
    
    def get_action_info(self, action: int) -> Dict[str, Any]:
        """Get information about a discrete action."""
        if 0 <= action < len(self.action_mapping):
            return self.action_mapping[action].copy()
        else:
            return {}
    
    def get_action_meanings(self) -> List[str]:
        """Get human-readable meanings for all actions."""
        return [self.action_mapping[i]['description'] for i in range(len(self.action_mapping))]
    
    def sample_action(self) -> int:
        """Sample a random discrete action."""
        return self.action_space.sample()


class AdaptiveDiscreteTradingWrapper(DiscreteTradingWrapper):
    """
    Adaptive discrete trading wrapper that adjusts actions based on market conditions.
    
    This wrapper modifies the action mapping based on:
    - Current portfolio state
    - Market volatility
    - Recent performance
    - Risk metrics
    """
    
    def __init__(
        self,
        env: gym.Env,
        action_strategy: str = "adaptive",
        adaptation_frequency: int = 1000,
        **kwargs
    ):
        """Initialize adaptive discrete trading wrapper.
        
        Args:
            env: Base trading environment
            action_strategy: Action strategy type
            adaptation_frequency: How often to adapt actions (in steps)
            **kwargs: Additional arguments for base wrapper
        """
        super().__init__(env, action_strategy, **kwargs)
        
        self.adaptation_frequency = adaptation_frequency
        self.step_count = 0
        self.adaptation_history = []
        
        # Market state tracking
        self.recent_returns = []
        self.recent_volatility = []
        self.recent_drawdowns = []
        
        logger.info(f"Adaptive discrete wrapper initialized with adaptation frequency {adaptation_frequency}")
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute step with adaptive action modification."""
        self.step_count += 1
        
        # Adapt actions periodically
        if self.step_count % self.adaptation_frequency == 0:
            self._adapt_actions()
        
        # Execute step
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Update market state tracking
        self._update_market_state(obs, reward, info)
        
        return obs, reward, terminated, truncated, info
    
    def _adapt_actions(self):
        """Adapt action mapping based on current market conditions."""
        logger.debug(f"Adapting actions at step {self.step_count}")
        
        # Analyze recent performance
        performance_metrics = self._analyze_recent_performance()
        
        # Adjust position sizes based on volatility
        if performance_metrics['volatility'] > 0.02:  # High volatility
            self.position_sizes = [0.03, 0.06, 0.12]  # Smaller positions
            self.sell_fractions = [0.5, 0.75, 1.0]    # More aggressive selling
        elif performance_metrics['volatility'] < 0.01:  # Low volatility
            self.position_sizes = [0.08, 0.15, 0.25]  # Larger positions
            self.sell_fractions = [0.2, 0.4, 0.6, 1.0]  # More gradual selling
        else:  # Normal volatility
            self.position_sizes = [0.05, 0.10, 0.20]  # Standard positions
            self.sell_fractions = [0.25, 0.50, 0.75, 1.0]  # Standard selling
        
        # Rebuild action mapping with new parameters
        self._setup_action_mapping()
        
        # Record adaptation
        adaptation_record = {
            'step': self.step_count,
            'performance_metrics': performance_metrics,
            'new_position_sizes': self.position_sizes.copy(),
            'new_sell_fractions': self.sell_fractions.copy()
        }
        self.adaptation_history.append(adaptation_record)
        
        logger.debug(f"Actions adapted: position_sizes={self.position_sizes}, sell_fractions={self.sell_fractions}")
    
    def _update_market_state(self, obs: np.ndarray, reward: float, info: Dict[str, Any]):
        """Update market state tracking."""
        # Track returns
        if 'portfolio_return' in info:
            self.recent_returns.append(info['portfolio_return'])
        else:
            # Estimate return from reward
            self.recent_returns.append(reward / 1000.0)  # Assuming reward scaling
        
        # Track volatility (rolling standard deviation of returns)
        if len(self.recent_returns) > 20:
            recent_vol = np.std(self.recent_returns[-20:])
            self.recent_volatility.append(recent_vol)
        
        # Track drawdowns
        if 'drawdown' in info:
            self.recent_drawdowns.append(info['drawdown'])
        
        # Limit history size
        max_history = 1000
        self.recent_returns = self.recent_returns[-max_history:]
        self.recent_volatility = self.recent_volatility[-max_history:]
        self.recent_drawdowns = self.recent_drawdowns[-max_history:]
    
    def _analyze_recent_performance(self) -> Dict[str, float]:
        """Analyze recent performance metrics."""
        metrics = {}
        
        # Volatility
        if len(self.recent_volatility) > 0:
            metrics['volatility'] = np.mean(self.recent_volatility[-50:])
        else:
            metrics['volatility'] = 0.015  # Default
        
        # Returns
        if len(self.recent_returns) > 0:
            metrics['mean_return'] = np.mean(self.recent_returns[-100:])
            metrics['return_volatility'] = np.std(self.recent_returns[-100:])
        else:
            metrics['mean_return'] = 0.0
            metrics['return_volatility'] = 0.01
        
        # Drawdown
        if len(self.recent_drawdowns) > 0:
            metrics['max_drawdown'] = np.max(self.recent_drawdowns[-100:])
            metrics['current_drawdown'] = self.recent_drawdowns[-1] if self.recent_drawdowns else 0.0
        else:
            metrics['max_drawdown'] = 0.0
            metrics['current_drawdown'] = 0.0
        
        # Sharpe ratio estimate
        if metrics['return_volatility'] > 0:
            metrics['sharpe_ratio'] = metrics['mean_return'] / metrics['return_volatility']
        else:
            metrics['sharpe_ratio'] = 0.0
        
        return metrics
    
    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of action adaptations."""
        return self.adaptation_history.copy()


def create_discrete_trading_env(
    base_env: gym.Env,
    wrapper_type: str = "standard",
    **wrapper_kwargs
) -> gym.Env:
    """Factory function to create discrete trading environment.
    
    Args:
        base_env: Base trading environment
        wrapper_type: Type of wrapper ("standard" or "adaptive")
        **wrapper_kwargs: Additional wrapper arguments
        
    Returns:
        Wrapped discrete trading environment
    """
    if wrapper_type == "standard":
        return DiscreteTradingWrapper(base_env, **wrapper_kwargs)
    elif wrapper_type == "adaptive":
        return AdaptiveDiscreteTradingWrapper(base_env, **wrapper_kwargs)
    else:
        raise ValueError(f"Unknown wrapper type: {wrapper_type}")


# Example usage and testing
if __name__ == "__main__":
    # This would be used with the actual trading environment
    # For testing purposes, we'll create a mock environment
    
    class MockTradingEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(100,))
            self.action_space = spaces.Box(low=0, high=2, shape=(10,))  # 5 symbols * 2 actions each
            self.n_symbols = 5
            self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
            
        def reset(self, **kwargs):
            return np.random.randn(100), {}
            
        def step(self, action):
            obs = np.random.randn(100)
            reward = np.random.randn()
            terminated = False
            truncated = False
            info = {'portfolio_value': 100000 + np.random.randn() * 1000}
            return obs, reward, terminated, truncated, info
    
    # Test the wrapper
    base_env = MockTradingEnv()
    
    # Test standard wrapper
    discrete_env = DiscreteTradingWrapper(base_env, action_strategy="single_asset")
    print(f"Discrete action space: {discrete_env.action_space}")
    print(f"Number of actions: {discrete_env.action_space.n}")
    
    # Test some actions
    obs, _ = discrete_env.reset()
    for i in range(5):
        action = discrete_env.sample_action()
        obs, reward, terminated, truncated, info = discrete_env.step(action)
        print(f"Action {action}: {info['action_description']}")
    
    # Test adaptive wrapper
    adaptive_env = AdaptiveDiscreteTradingWrapper(base_env, adaptation_frequency=10)
    print(f"\nAdaptive wrapper created with {adaptive_env.action_space.n} actions")
    
    # Run some steps to test adaptation
    obs, _ = adaptive_env.reset()
    for i in range(25):
        action = adaptive_env.sample_action()
        obs, reward, terminated, truncated, info = adaptive_env.step(action)
        if i % 10 == 0:
            print(f"Step {i}: Action {action}")
    
    print(f"Adaptations made: {len(adaptive_env.get_adaptation_history())}")