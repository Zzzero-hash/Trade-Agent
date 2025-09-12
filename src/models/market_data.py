"""
Market data models with Pydantic validation.
"""
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from enum import Enum


class ExchangeType(str, Enum):
    """Supported exchange types."""
    ROBINHOOD = "robinhood"
    OANDA = "oanda"
    COINBASE = "coinbase"


class MarketData(BaseModel):
    """
    Market data model with comprehensive validation.
    
    Represents OHLCV (Open, High, Low, Close, Volume) data for a financial instrument.
    """
    symbol: str = Field(..., min_length=1, max_length=20, description="Trading symbol")
    timestamp: datetime = Field(..., description="Data timestamp")
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="Highest price")
    low: float = Field(..., gt=0, description="Lowest price")
    close: float = Field(..., gt=0, description="Closing price")
    volume: float = Field(..., ge=0, description="Trading volume")
    exchange: ExchangeType = Field(..., description="Exchange identifier")
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v):
        """Validate trading symbol format."""
        if not v or not v.strip():
            raise ValueError("Symbol cannot be empty")
        # Remove whitespace and convert to uppercase
        return v.strip().upper()
    
    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v):
        """Validate timestamp is not in the future."""
        # Handle both timezone-aware and naive datetimes
        now = datetime.now(timezone.utc)
        if v.tzinfo is None:
            # Assume naive datetime is UTC
            v = v.replace(tzinfo=timezone.utc)
        if v > now:
            raise ValueError("Timestamp cannot be in the future")
        return v
    
    @model_validator(mode='after')
    def validate_price_relationships(self):
        """Validate price relationships (high >= low, etc.)."""
        if self.high < self.low:
            raise ValueError("High price must be >= low price")
        if self.high < self.open:
            raise ValueError("High price must be >= open price")
        if self.high < self.close:
            raise ValueError("High price must be >= close price")
        if self.low > self.open:
            raise ValueError("Low price must be <= open price")
        if self.low > self.close:
            raise ValueError("Low price must be <= close price")
        return self
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "symbol": "AAPL",
                "timestamp": "2023-12-01T15:30:00",
                "open": 150.25,
                "high": 152.10,
                "low": 149.80,
                "close": 151.75,
                "volume": 1000000.0,
                "exchange": "robinhood"
            }
        }
    )