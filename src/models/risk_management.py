"""
Risk management data models.
"""

from enum import Enum
from typing import Dict, Optional
from datetime import datetime
from dataclasses import dataclass


class RiskLimitType(Enum):
    """Types of risk limits."""
    MAX_DRAWDOWN = "max_drawdown"
    PORTFOLIO_VAR = "portfolio_var"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"
    DAILY_LOSS = "daily_loss"
    POSITION_SIZE = "position_size"


class RiskLimitStatus(Enum):
    """Risk limit status."""
    NORMAL = "normal"
    WARNING = "warning"
    BREACH = "breach"


@dataclass
class RiskLimit:
    """Risk limit configuration."""
    limit_type: RiskLimitType
    threshold: float
    warning_threshold: float
    symbol: Optional[str] = None
    enabled: bool = True


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio."""
    portfolio_value: float
    daily_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    max_drawdown: float
    current_drawdown: float
    portfolio_var: float
    portfolio_volatility: float
    concentration_risk: float
    leverage: float
    timestamp: datetime


@dataclass
class RiskAlert:
    """Risk alert notification."""
    alert_id: str
    limit_type: RiskLimitType
    status: RiskLimitStatus
    current_value: float
    threshold: float
    symbol: Optional[str]
    message: str
    timestamp: datetime
    acknowledged: bool = False


@dataclass
class StressTestScenario:
    """Stress test scenario configuration."""
    scenario_name: str
    market_shocks: Dict[str, float]
    correlation_adjustment: float = 1.0
    volatility_multiplier: float = 1.0
    description: str = ""


@dataclass
class StressTestResult:
    """Results of a stress test."""
    scenario_name: str
    portfolio_value_before: float
    portfolio_value_after: float
    total_loss: float
    loss_percentage: float
    position_impacts: Dict[str, float]
    risk_metrics_after: 'RiskMetrics'
    timestamp: datetime


@dataclass
class PositionSizingRule:
    """Position sizing rule configuration."""
    rule_name: str
    kelly_fraction: float = 1.0
    volatility_target: float = 0.15
    correlation_penalty: float = 0.1
    max_position_size: float = 0.1
    enabled: bool = True