"""
Trading Models

Pydantic models for trading operations, positions, and risk management.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field


class TradeSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TradeStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class BrokerType(str, Enum):
    ROBINHOOD = "robinhood"
    TD_AMERITRADE = "td_ameritrade"
    INTERACTIVE_BROKERS = "interactive_brokers"
    OANDA = "oanda"
    COINBASE = "coinbase"


class Position(BaseModel):
    """Trading position model."""
    position_id: str
    customer_id: str
    symbol: str
    quantity: Decimal
    cost_basis: Decimal
    current_value: Decimal
    unrealized_pnl: Decimal
    created_at: datetime
    updated_at: Optional[datetime] = None
    broker: Optional[BrokerType] = None
    
    class Config:
        use_enum_values = True


class Trade(BaseModel):
    """Trade execution model."""
    trade_id: str
    customer_id: str
    position_id: Optional[str] = None
    symbol: str
    side: TradeSide
    quantity: Decimal
    price: Decimal
    order_type: OrderType
    status: TradeStatus
    created_at: datetime
    executed_at: Optional[datetime] = None
    broker: Optional[BrokerType] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class RiskMetrics(BaseModel):
    """Risk metrics for positions and portfolios."""
    customer_id: str
    portfolio_value: Decimal
    portfolio_pnl: Decimal
    daily_pnl: Decimal
    max_drawdown: Decimal
    var_95: Decimal  # Value at Risk 95%
    sharpe_ratio: Optional[Decimal] = None
    beta: Optional[Decimal] = None
    calculated_at: datetime
    
    class Config:
        use_enum_values = True