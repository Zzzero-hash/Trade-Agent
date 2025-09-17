"""
Data models for the AI Trading Platform SDK
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


class TradingAction(str, Enum):
    """Trading action types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderType(str, Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class AssetType(str, Enum):
    """Asset types"""
    STOCK = "STOCK"
    ETF = "ETF"
    OPTION = "OPTION"
    FOREX = "FOREX"
    CRYPTO = "CRYPTO"


class TradingSignal(BaseModel):
    """Trading signal model"""
    id: str
    symbol: str
    action: TradingAction
    confidence: float = Field(..., ge=0.0, le=1.0)
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size: Optional[float] = None
    reasoning: Optional[str] = None
    timestamp: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Position(BaseModel):
    """Portfolio position model"""
    symbol: str
    quantity: float
    average_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    asset_type: AssetType
    last_updated: datetime


class Portfolio(BaseModel):
    """Portfolio model"""
    user_id: str
    positions: Dict[str, Position]
    cash_balance: float
    total_value: float
    total_pnl: float
    total_pnl_percent: float
    last_updated: datetime


class MarketData(BaseModel):
    """Market data model"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RiskMetrics(BaseModel):
    """Risk metrics model"""
    portfolio_var: float
    portfolio_cvar: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    beta: Optional[float] = None
    alpha: Optional[float] = None
    volatility: float
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    timestamp: datetime


class Alert(BaseModel):
    """Alert model"""
    id: str
    severity: str
    title: str
    message: str
    timestamp: datetime
    model_name: Optional[str] = None
    metric_name: Optional[str] = None
    acknowledged: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelStatus(BaseModel):
    """Model status model"""
    model_name: str
    status: str
    health_score: float = Field(..., ge=0.0, le=1.0)
    last_prediction: Optional[datetime] = None
    predictions_today: int = 0
    accuracy: Optional[float] = None
    confidence: Optional[float] = None
    version: str
    deployed_at: datetime


class PerformanceMetrics(BaseModel):
    """Performance metrics model"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    timestamp: datetime


class WebSocketMessage(BaseModel):
    """WebSocket message model"""
    type: str
    data: Dict[str, Any]
    timestamp: datetime
    user_id: Optional[str] = None


class PluginConfig(BaseModel):
    """Plugin configuration model"""
    name: str
    version: str
    enabled: bool = True
    config: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)


class StrategyConfig(BaseModel):
    """Trading strategy configuration"""
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    risk_limits: Dict[str, float] = Field(default_factory=dict)
    enabled: bool = True
    created_at: datetime
    updated_at: datetime


class BacktestResult(BaseModel):
    """Backtest result model"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float
    metadata: Dict[str, Any] = Field(default_factory=dict)