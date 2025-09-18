"""
Factory for creating trading environments with different configurations.
"""
from typing import List, Optional
import pandas as pd

from .trading_environment import TradingEnvironment, TradingConfig
from .reward_calculator import (
    RewardCalculator, 
    RiskAdjustedRewardCalculator,
    SimpleReturnRewardCalculator
)


class TradingEnvironmentFactory:
    """Factory for creating configured trading environments."""
    
    @staticmethod
    def create_basic_environment(
        market_data: pd.DataFrame,
        symbols: Optional[List[str]] = None,
        initial_balance: float = 100000.0
    ) -> TradingEnvironment:
        """Create a basic trading environment with default settings."""
        config = TradingConfig(
            initial_balance=initial_balance,
            max_position_size=0.2,
            transaction_cost=0.001,
            slippage=0.0005,
            lookback_window=20,
            max_drawdown_limit=0.25,
            reward_scaling=1000.0
        )
        
        return TradingEnvironment(
            market_data=market_data,
            config=config,
            symbols=symbols
        )
    
    @staticmethod
    def create_high_frequency_environment(
        market_data: pd.DataFrame,
        symbols: Optional[List[str]] = None,
        initial_balance: float = 100000.0
    ) -> TradingEnvironment:
        """Create environment optimized for high-frequency trading."""
        config = TradingConfig(
            initial_balance=initial_balance,
            max_position_size=0.1,  # Smaller positions
            transaction_cost=0.0005,  # Lower costs for HFT
            slippage=0.0002,  # Lower slippage
            lookback_window=10,  # Shorter lookback
            max_drawdown_limit=0.15,  # Stricter risk control
            reward_scaling=10000.0  # Higher scaling for small moves
        )
        
        return TradingEnvironment(
            market_data=market_data,
            config=config,
            symbols=symbols
        )
    
    @staticmethod
    def create_conservative_environment(
        market_data: pd.DataFrame,
        symbols: Optional[List[str]] = None,
        initial_balance: float = 100000.0
    ) -> TradingEnvironment:
        """Create conservative environment with strict risk controls."""
        config = TradingConfig(
            initial_balance=initial_balance,
            max_position_size=0.05,  # Very small positions
            transaction_cost=0.002,  # Higher costs
            slippage=0.001,  # Higher slippage
            lookback_window=60,  # Longer lookback
            max_drawdown_limit=0.1,  # Very strict drawdown limit
            reward_scaling=500.0  # Lower scaling
        )
        
        return TradingEnvironment(
            market_data=market_data,
            config=config,
            symbols=symbols
        )
    
    @staticmethod
    def create_custom_environment(
        market_data: pd.DataFrame,
        config: TradingConfig,
        symbols: Optional[List[str]] = None,
        reward_calculator: Optional[RewardCalculator] = None
    ) -> TradingEnvironment:
        """Create environment with custom configuration."""
        env = TradingEnvironment(
            market_data=market_data,
            config=config,
            symbols=symbols
        )
        
        if reward_calculator:
            # This would require modifying TradingEnvironment to accept
            # a reward calculator - future enhancement
            pass
        
        return env