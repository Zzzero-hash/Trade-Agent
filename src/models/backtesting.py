"""
Backtesting models with Pydantic validation.

This module provides data models for comprehensive backtesting with walk-forward analysis,
performance attribution, and statistical significance testing for trading strategies.

Requirements: 2.5, 5.7, 9.6
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from enum import Enum
import numpy as np


class BacktestPeriodType(str, Enum):
    """Types of walk-forward analysis periods."""
    ROLLING = "rolling"
    EXPANDING = "expanding"
    FIXED = "fixed"


class BacktestConfig(BaseModel):
    """
    Configuration for backtesting with walk-forward analysis.
    
    Defines the parameters for comprehensive backtesting including time periods,
    assets, and trading parameters.
    """
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    symbols: List[str] = Field(..., min_length=1, description="List of trading symbols")
    training_period_days: int = Field(252, ge=1, description="Training period in days")
    testing_period_days: int = Field(63, ge=1, description="Testing period in days")
    period_type: BacktestPeriodType = Field(BacktestPeriodType.ROLLING, description="Walk-forward period type")
    overlap_days: int = Field(21, ge=0, description="Overlap between periods in days")
    initial_balance: float = Field(100000.0, gt=0, description="Initial portfolio balance")
    max_position_size: float = Field(0.25, ge=0.01, le=1.0, description="Maximum position size per asset")
    transaction_cost: float = Field(0.001, ge=0, le=0.1, description="Transaction cost rate")
    slippage: float = Field(0.0005, ge=0, le=0.1, description="Slippage rate")
    max_drawdown_limit: float = Field(0.2, ge=0.01, le=0.5, description="Maximum allowable drawdown")
    stop_loss_threshold: float = Field(0.1, ge=0.01, le=0.3, description="Stop loss threshold")
    rebalance_frequency: str = Field("daily", description="Portfolio rebalancing frequency")
    
    @model_validator(mode='after')
    def validate_dates(self):
        """Validate date relationships."""
        if self.end_date <= self.start_date:
            raise ValueError("End date must be after start date")
        
        # Check if date range is sufficient for training/testing
        date_range_days = (self.end_date - self.start_date).days
        min_required_days = self.training_period_days + self.testing_period_days
        if date_range_days < min_required_days:
            raise ValueError(f"Date range is too short. Need at least {min_required_days} days for "
                           f"{self.training_period_days} training + {self.testing_period_days} testing days")
        
        return self
    
    @field_validator('symbols')
    @classmethod
    def validate_symbols(cls, v):
        """Validate trading symbols."""
        if not v:
            raise ValueError("Symbols list cannot be empty")
        return [symbol.strip().upper() for symbol in v]
    
    @field_validator('overlap_days')
    @classmethod
    def validate_overlap(cls, v, info):
        """Validate overlap days don't exceed testing period."""
        if 'testing_period_days' in info.data and v >= info.data['testing_period_days']:
            raise ValueError("Overlap days must be less than testing period days")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "start_date": "2020-01-01T00:00:00",
                "end_date": "2023-12-31T00:00:00",
                "symbols": ["AAPL", "GOOGL", "MSFT"],
                "training_period_days": 252,
                "testing_period_days": 63,
                "period_type": "rolling",
                "overlap_days": 21,
                "initial_balance": 100000.0,
                "max_position_size": 0.25,
                "transaction_cost": 0.001,
                "slippage": 0.0005
            }
        }
    )


class TradeRecord(BaseModel):
    """
    Record of a single trade execution.
    
    Contains detailed information about trade execution including costs and timing.
    """
    timestamp: datetime = Field(..., description="Trade execution timestamp")
    symbol: str = Field(..., min_length=1, max_length=20, description="Trading symbol")
    action: str = Field(..., description="Trade action (BUY/SELL)")
    quantity: float = Field(..., gt=0, description="Quantity traded")
    price: float = Field(..., gt=0, description="Execution price")
    commission: float = Field(0.0, ge=0, description="Commission costs")
    slippage_cost: float = Field(0.0, ge=0, description="Slippage costs")
    
    @property
    def total_cost(self) -> float:
        """Calculate total trade cost."""
        return self.quantity * self.price + self.commission + self.slippage_cost
    
    @property
    def notional_value(self) -> float:
        """Calculate notional value of the trade."""
        return self.quantity * self.price
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v):
        """Validate trading symbol format."""
        if not v or not v.strip():
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()
    
    @field_validator('action')
    @classmethod
    def validate_action(cls, v):
        """Validate trade action."""
        valid_actions = ['BUY', 'SELL']
        if v not in valid_actions:
            raise ValueError(f"Action must be BUY or SELL, got {v}")
        return v
    
    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v):
        """Validate timestamp timezone."""
        if v.tzinfo is None:
            # Assume naive datetime is UTC
            v = v.replace(tzinfo=timezone.utc)
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2023-12-01T15:30:00",
                "symbol": "AAPL",
                "action": "BUY",
                "quantity": 100.0,
                "price": 150.0,
                "commission": 1.0,
                "slippage_cost": 0.75
            }
        }
    )


class PerformanceMetrics(BaseModel):
    """
    Comprehensive performance metrics for backtesting.
    
    Includes risk-adjusted returns, drawdown metrics, and statistical significance testing.
    """
    total_return: float = Field(0.0, description="Total return percentage")
    annualized_return: float = Field(0.0, description="Annualized return percentage")
    volatility: float = Field(0.0, ge=0, description="Annualized volatility percentage")
    sharpe_ratio: float = Field(0.0, description="Sharpe ratio")
    sortino_ratio: float = Field(0.0, description="Sortino ratio")
    calmar_ratio: float = Field(0.0, description="Calmar ratio")
    max_drawdown: float = Field(0.0, ge=0, description="Maximum drawdown percentage")
    max_drawdown_duration: int = Field(0, ge=0, description="Maximum drawdown duration in days")
    win_rate: float = Field(0.0, ge=0.0, le=1.0, description="Win rate (0-1)")
    profit_factor: float = Field(0.0, ge=0, description="Profit factor")
    total_trades: int = Field(0, ge=0, description="Total number of trades")
    avg_trade_return: float = Field(0.0, description="Average trade return percentage")
    
    # Statistical significance testing
    t_statistic: Optional[float] = Field(None, description="T-statistic for return significance")
    p_value: Optional[float] = Field(None, ge=0.0, le=1.0, description="P-value for statistical significance")
    confidence_interval_lower: Optional[float] = Field(None, description="Lower bound of confidence interval")
    confidence_interval_upper: Optional[float] = Field(None, description="Upper bound of confidence interval")
    
    @field_validator('win_rate')
    @classmethod
    def validate_win_rate(cls, v):
        """Validate win rate is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Win rate must be between 0.0 and 1.0, got {v}")
        return v
    
    @field_validator('max_drawdown')
    @classmethod
    def validate_max_drawdown(cls, v):
        """Validate max drawdown is non-negative."""
        if v < 0:
            raise ValueError(f"Max drawdown must be non-negative, got {v}")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_return": 25.5,
                "annualized_return": 12.3,
                "volatility": 18.2,
                "sharpe_ratio": 0.67,
                "sortino_ratio": 0.89,
                "calmar_ratio": 0.61,
                "max_drawdown": 8.5,
                "max_drawdown_duration": 45,
                "win_rate": 0.58,
                "profit_factor": 1.35,
                "total_trades": 156,
                "avg_trade_return": 0.16,
                "t_statistic": 2.34,
                "p_value": 0.021
            }
        }
    )


class BacktestPeriodResult(BaseModel):
    """
    Results from a single backtest period.
    
    Contains performance metrics and portfolio evolution for one walk-forward period.
    """
    period_id: int = Field(..., ge=0, description="Period identifier")
    train_start: datetime = Field(..., description="Training period start")
    train_end: datetime = Field(..., description="Training period end")
    test_start: datetime = Field(..., description="Testing period start")
    test_end: datetime = Field(..., description="Testing period end")
    performance_metrics: PerformanceMetrics = Field(..., description="Performance metrics for this period")
    portfolio_values: List[float] = Field(default_factory=list, description="Portfolio values over time")
    portfolio_dates: List[datetime] = Field(default_factory=list, description="Dates corresponding to portfolio values")
    trades: List[TradeRecord] = Field(default_factory=list, description="Trades executed during period")
    model_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Model prediction accuracy")
    
    @property
    def test_duration_days(self) -> int:
        """Calculate testing period duration in days."""
        return (self.test_end - self.test_start).days
    
    @property
    def final_portfolio_value(self) -> float:
        """Get final portfolio value."""
        return self.portfolio_values[-1] if self.portfolio_values else 0.0
    
    @property
    def period_return(self) -> float:
        """Calculate period return percentage."""
        if not self.portfolio_values or len(self.portfolio_values) < 2:
            return 0.0
        initial = self.portfolio_values[0]
        final = self.portfolio_values[-1]
        return ((final - initial) / initial) * 100 if initial > 0 else 0.0
    
    @model_validator(mode='after')
    def validate_portfolio_data(self):
        """Validate portfolio values and dates have same length."""
        if len(self.portfolio_values) != len(self.portfolio_dates):
            raise ValueError("Portfolio values and dates must have same length")
        return self
    
    @field_validator('train_start', 'train_end', 'test_start', 'test_end')
    @classmethod
    def validate_timestamps(cls, v):
        """Validate timestamp timezones."""
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "period_id": 0,
                "train_start": "2022-01-01T00:00:00",
                "train_end": "2022-12-31T00:00:00",
                "test_start": "2023-01-01T00:00:00",
                "test_end": "2023-03-31T00:00:00",
                "performance_metrics": {
                    "total_return": 10.0,
                    "annualized_return": 8.0,
                    "volatility": 15.0,
                    "sharpe_ratio": 0.5,
                    "sortino_ratio": 0.6,
                    "calmar_ratio": 0.4,
                    "max_drawdown": 5.0,
                    "max_drawdown_duration": 30,
                    "win_rate": 0.6,
                    "profit_factor": 1.2,
                    "total_trades": 50,
                    "avg_trade_return": 0.2
                },
                "portfolio_values": [100000.0, 101000.0, 102000.0],
                "portfolio_dates": ["2023-01-01T00:00:00", "2023-01-02T00:00:00", "2023-01-03T00:00:00"]
            }
        }
    )


class StressTestScenario(BaseModel):
    """
    Stress testing scenario configuration.
    
    Defines market conditions and shocks for stress testing trading strategies.
    """
    name: str = Field(..., min_length=1, max_length=100, description="Scenario name")
    description: str = Field("", max_length=500, description="Detailed scenario description")
    market_shock_magnitude: float = Field(0.0, description="Market shock magnitude (-1.0 to 1.0)")
    shock_duration_days: int = Field(30, ge=1, description="Duration of shock in days")
    recovery_duration_days: int = Field(0, ge=0, description="Recovery period in days")
    volatility_multiplier: float = Field(1.0, ge=0.1, description="Volatility multiplier during stress")
    correlation_increase: float = Field(0.0, ge=0.0, le=1.0, description="Increase in asset correlations")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Validate scenario name."""
        if not v or not v.strip():
            raise ValueError("Scenario name cannot be empty")
        return v.strip()
    
    @field_validator('market_shock_magnitude')
    @classmethod
    def validate_market_shock(cls, v):
        """Validate market shock magnitude."""
        if not -1.0 <= v <= 1.0:
            raise ValueError(f"Market shock magnitude must be between -1.0 and 1.0, got {v}")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Market Crash 2008",
                "description": "Simulates 2008 financial crisis conditions",
                "market_shock_magnitude": -0.35,
                "shock_duration_days": 180,
                "recovery_duration_days": 365,
                "volatility_multiplier": 2.5,
                "correlation_increase": 0.3
            }
        }
    )


class StressTestResult(BaseModel):
    """
    Results from stress testing a trading strategy.
    
    Compares normal performance to stressed performance under adverse market conditions.
    """
    scenario: StressTestScenario = Field(..., description="Stress test scenario")
    stressed_metrics: PerformanceMetrics = Field(..., description="Performance metrics under stress")
    normal_metrics: PerformanceMetrics = Field(..., description="Normal performance metrics")
    performance_degradation: float = Field(0.0, ge=0.0, description="Performance degradation percentage")
    worst_case_loss: float = Field(0.0, ge=0.0, description="Worst case loss percentage")
    time_to_recovery: Optional[int] = Field(None, ge=0, description="Days to recover from stress")
    max_leverage_used: float = Field(1.0, ge=1.0, description="Maximum leverage used during stress")
    positions_liquidated: int = Field(0, ge=0, description="Number of positions liquidated")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "scenario": {
                    "name": "Market Crash 2008",
                    "description": "Simulates 2008 financial crisis conditions",
                    "market_shock_magnitude": -0.35,
                    "shock_duration_days": 180,
                    "recovery_duration_days": 365,
                    "volatility_multiplier": 2.5,
                    "correlation_increase": 0.3
                },
                "stressed_metrics": {
                    "total_return": -15.0,
                    "annualized_return": -7.5,
                    "volatility": 35.0,
                    "sharpe_ratio": -0.21,
                    "sortino_ratio": -0.35,
                    "calmar_ratio": -0.45,
                    "max_drawdown": 25.0,
                    "max_drawdown_duration": 120,
                    "win_rate": 0.42,
                    "profit_factor": 0.85,
                    "total_trades": 120,
                    "avg_trade_return": -0.12
                },
                "normal_metrics": {
                    "total_return": 25.5,
                    "annualized_return": 12.3,
                    "volatility": 18.2,
                    "sharpe_ratio": 0.67,
                    "sortino_ratio": 0.89,
                    "calmar_ratio": 0.61,
                    "max_drawdown": 8.5,
                    "max_drawdown_duration": 45,
                    "win_rate": 0.58,
                    "profit_factor": 1.35,
                    "total_trades": 156,
                    "avg_trade_return": 0.16
                },
                "performance_degradation": 0.45,
                "worst_case_loss": 25.0,
                "time_to_recovery": 200,
                "max_leverage_used": 1.5,
                "positions_liquidated": 3
            }
        }
    )


class BacktestResult(BaseModel):
    """
    Complete backtesting results with comprehensive analysis.
    
    Contains overall performance, period-by-period results, and risk metrics.
    """
    config: BacktestConfig = Field(..., description="Backtest configuration")
    execution_start: datetime = Field(..., description="Execution start timestamp")
    execution_end: datetime = Field(..., description="Execution end timestamp")
    total_periods: int = Field(..., ge=0, description="Total number of periods")
    overall_metrics: PerformanceMetrics = Field(..., description="Overall performance metrics")
    period_results: List[BacktestPeriodResult] = Field(default_factory=list, description="Individual period results")
    cumulative_returns: List[float] = Field(default_factory=list, description="Cumulative returns over time")
    cumulative_dates: List[datetime] = Field(default_factory=list, description="Dates for cumulative returns")
    var_95: float = Field(0.0, description="Value at Risk at 95% confidence")
    cvar_95: float = Field(0.0, description="Conditional Value at Risk at 95% confidence")
    stability_metrics: Dict[str, float] = Field(default_factory=dict, description="Stability metrics across periods")
    performance_consistency: float = Field(0.0, ge=0.0, le=1.0, description="Performance consistency score")
    avg_period_performance: float = Field(0.0, description="Average performance across periods")
    
    @property
    def execution_duration(self) -> float:
        """Calculate execution duration in seconds."""
        return (self.execution_end - self.execution_start).total_seconds()
    
    @property
    def performance_consistency_score(self) -> float:
        """Calculate performance consistency score."""
        if not self.period_results:
            return 0.0
        
        period_returns = [p.performance_metrics.total_return for p in self.period_results]
        if len(period_returns) < 2:
            return 1.0
        
        mean_return = np.mean(period_returns)
        std_return = np.std(period_returns)
        
        if mean_return == 0:
            return 0.0
        
        # Consistency score based on coefficient of variation
        cv = abs(std_return / mean_return) if mean_return != 0 else 0.0
        return max(0.0, min(1.0, 1.0 - cv))
    
    @model_validator(mode='after')
    def validate_cumulative_data(self):
        """Validate cumulative returns and dates have same length."""
        if len(self.cumulative_returns) != len(self.cumulative_dates):
            raise ValueError("Cumulative returns and dates must have same length")
        return self
    
    @field_validator('execution_start', 'execution_end')
    @classmethod
    def validate_timestamps(cls, v):
        """Validate timestamp timezones."""
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "config": {
                    "start_date": "2020-01-01T00:00:00",
                    "end_date": "2023-12-31T00:00:00",
                    "symbols": ["AAPL", "GOOGL"],
                    "training_period_days": 252,
                    "testing_period_days": 63
                },
                "execution_start": "2023-12-01T10:00:00",
                "execution_end": "2023-12-01T10:05:00",
                "total_periods": 12,
                "overall_metrics": {
                    "total_return": 45.2,
                    "annualized_return": 13.8,
                    "volatility": 22.1,
                    "sharpe_ratio": 0.63,
                    "sortino_ratio": 0.82,
                    "calmar_ratio": 0.58,
                    "max_drawdown": 12.3,
                    "max_drawdown_duration": 67,
                    "win_rate": 0.61,
                    "profit_factor": 1.42,
                    "total_trades": 312,
                    "avg_trade_return": 0.18
                },
                "period_results": [],
                "cumulative_returns": [0.0, 2.1, 4.3, 3.8, 6.2],
                "cumulative_dates": [
                    "2020-01-01T00:00:00",
                    "2020-02-01T00:00:00", 
                    "2020-03-01T00:00:00",
                    "2020-04-01T00:00:00",
                    "2020-05-01T00:00:00"
                ],
                "var_95": 2.3,
                "cvar_95": 3.1,
                "stability_metrics": {
                    "return_consistency": 0.78,
                    "sharpe_consistency": 0.82,
                    "drawdown_consistency": 0.65
                },
                "performance_consistency": 0.75
            }
        }
    )