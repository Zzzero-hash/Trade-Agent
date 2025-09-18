"""
Risk Management Models

Data models for risk management, alerts, and limits.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional
from enum import Enum
from dataclasses import dataclass

from pydantic import BaseModel


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskAlert(BaseModel):
    alert_id: str
    customer_id: str
    risk_level: RiskLevel
    alert_type: str
    message: str
    timestamp: datetime
    position_id: Optional[str] = None
    current_value: Optional[Decimal] = None
    threshold_value: Optional[Decimal] = None


@dataclass
class RiskLimits:
    max_position_size: Decimal
    max_daily_loss: Decimal
    max_portfolio_risk: Decimal
    max_correlation_exposure: Decimal
    stop_loss_percentage: Decimal
    max_leverage: Decimal


class RiskMetrics(BaseModel):
    customer_id: str
    portfolio_value: Decimal
    daily_pnl: Decimal
    portfolio_risk: float
    max_drawdown: float
    sharpe_ratio: float
    var_95: Decimal
    timestamp: datetime