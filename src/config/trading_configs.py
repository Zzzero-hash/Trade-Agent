"""
Centralized Trading Configuration System

This module provides pre-configured trading setups to eliminate DRY violations
and ensure consistency across different training and testing scenarios.
"""

from dataclasses import dataclass
from typing import List, Optional
from ml.yfinance_trading_environment import YFinanceConfig


@dataclass
class TradingConstants:
    """Centralized constants to eliminate magic numbers"""
    
    # Portfolio defaults
    DEFAULT_INITIAL_BALANCE: float = 100000.0
    DEFAULT_TRANSACTION_COST: float = 0.001  # 0.1%
    DEFAULT_SLIPPAGE_FACTOR: float = 0.0005  # 0.05%
    DEFAULT_LOOKBACK_WINDOW: int = 60
    DEFAULT_MAX_DRAWDOWN_LIMIT: float = 0.2  # 20%
    DEFAULT_MIN_EPISODE_LENGTH: int = 100
    
    # Performance targets
    TARGET_SORTINO_RATIO: float = 2.0
    TARGET_MAX_DRAWDOWN: float = 0.1  # 10%
    
    # Default symbols
    DEFAULT_SYMBOLS: List[str] = None
    
    def __post_init__(self):
        if self.DEFAULT_SYMBOLS is None:
            self.DEFAULT_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']


# Global constants instance
CONSTANTS = TradingConstants()


class TradingConfigFactory:
    """Factory for creating standardized trading configurations"""
    
    @staticmethod
    def create_training_config(
        max_position_size: float = 0.2,
        reward_scaling: float = 10.0,
        **overrides
    ) -> YFinanceConfig:
        """Create configuration optimized for training"""
        config = YFinanceConfig(
            initial_balance=CONSTANTS.DEFAULT_INITIAL_BALANCE,
            max_position_size=max_position_size,
            transaction_cost=CONSTANTS.DEFAULT_TRANSACTION_COST,
            slippage_base=CONSTANTS.DEFAULT_SLIPPAGE_FACTOR,
            lookback_window=CONSTANTS.DEFAULT_LOOKBACK_WINDOW,
            min_episode_length=CONSTANTS.DEFAULT_MIN_EPISODE_LENGTH,
            max_drawdown_limit=CONSTANTS.DEFAULT_MAX_DRAWDOWN_LIMIT,
            reward_scaling=reward_scaling,
            **overrides
        )
        return config
    
    @staticmethod
    def create_debug_config(
        max_position_size: float = 0.1,
        reward_scaling: float = 10.0,
        **overrides
    ) -> YFinanceConfig:
        """Create configuration optimized for debugging"""
        config = YFinanceConfig(
            initial_balance=CONSTANTS.DEFAULT_INITIAL_BALANCE,
            max_position_size=max_position_size,
            transaction_cost=CONSTANTS.DEFAULT_TRANSACTION_COST,
            slippage_base=CONSTANTS.DEFAULT_SLIPPAGE_FACTOR,
            lookback_window=CONSTANTS.DEFAULT_LOOKBACK_WINDOW,
            min_episode_length=50,  # Shorter for debugging
            max_drawdown_limit=CONSTANTS.DEFAULT_MAX_DRAWDOWN_LIMIT,
            reward_scaling=reward_scaling,
            **overrides
        )
        return config
    
    @staticmethod
    def create_production_config(
        max_position_size: float = 0.15,
        reward_scaling: float = 1.0,
        **overrides
    ) -> YFinanceConfig:
        """Create configuration optimized for production"""
        config = YFinanceConfig(
            initial_balance=CONSTANTS.DEFAULT_INITIAL_BALANCE,
            max_position_size=max_position_size,
            transaction_cost=CONSTANTS.DEFAULT_TRANSACTION_COST,
            slippage_base=CONSTANTS.DEFAULT_SLIPPAGE_FACTOR,
            lookback_window=CONSTANTS.DEFAULT_LOOKBACK_WINDOW,
            min_episode_length=CONSTANTS.DEFAULT_MIN_EPISODE_LENGTH,
            max_drawdown_limit=0.1,  # Stricter for production
            reward_scaling=reward_scaling,
            **overrides
        )
        return config


def get_default_symbols() -> List[str]:
    """Get default trading symbols"""
    return CONSTANTS.DEFAULT_SYMBOLS.copy()


def get_performance_targets() -> dict:
    """Get performance targets for validation"""
    return {
        'sortino_ratio': CONSTANTS.TARGET_SORTINO_RATIO,
        'max_drawdown': CONSTANTS.TARGET_MAX_DRAWDOWN
    }


def validate_performance(results: dict) -> dict:
    """Centralized performance validation logic"""
    final_eval = results.get('final_evaluation', {})
    sortino_ratio = final_eval.get('mean_sortino_ratio', 0)
    max_drawdown = final_eval.get('max_drawdown', 1.0)
    
    targets = get_performance_targets()
    
    performance_met = sortino_ratio >= targets['sortino_ratio']
    drawdown_met = max_drawdown <= targets['max_drawdown']
    
    return {
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'targets': targets,
        'performance_met': performance_met,
        'drawdown_met': drawdown_met,
        'all_targets_met': performance_met and drawdown_met
    }


def format_performance_report(validation_results: dict) -> str:
    """Format performance results for consistent reporting"""
    results = validation_results
    targets = results['targets']
    
    report = []
    report.append("="*80)
    report.append("PERFORMANCE VALIDATION REPORT")
    report.append("="*80)
    report.append(f"Sortino Ratio: {results['sortino_ratio']:.4f} (Target: ≥{targets['sortino_ratio']:.1f})")
    report.append(f"Maximum Drawdown: {results['max_drawdown']:.4f} (Target: ≤{targets['max_drawdown']:.1f})")
    report.append(f"Performance Threshold Met: {'✓' if results['performance_met'] else '✗'}")
    report.append(f"Drawdown Threshold Met: {'✓' if results['drawdown_met'] else '✗'}")
    
    if results['all_targets_met']:
        report.append("🎉 ALL PERFORMANCE TARGETS ACHIEVED!")
    else:
        report.append("⚠️  Performance targets not fully met - consider additional training")
    
    report.append("="*80)
    
    return "\n".join(report)