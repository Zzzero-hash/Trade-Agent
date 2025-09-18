"""
Portfolio and position models with Pydantic validation.
"""
from datetime import datetime, timezone
from typing import Dict, Optional
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from decimal import Decimal


class Position(BaseModel):
    """
    Position model representing a holding in a portfolio.
    """
    symbol: str = Field(..., min_length=1, max_length=20, description="Trading symbol")
    quantity: float = Field(..., description="Number of shares/units held")
    avg_cost: float = Field(..., gt=0, description="Average cost per unit")
    current_price: float = Field(..., gt=0, description="Current market price per unit")
    unrealized_pnl: float = Field(..., description="Unrealized profit/loss")
    realized_pnl: float = Field(default=0.0, description="Realized profit/loss from closed trades")
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v):
        """Validate trading symbol format."""
        if not v or not v.strip():
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()
    
    @field_validator('quantity')
    @classmethod
    def validate_quantity(cls, v):
        """Validate quantity is not zero."""
        if v == 0:
            raise ValueError("Position quantity cannot be zero")
        return v
    
    @model_validator(mode='after')
    def validate_unrealized_pnl(self):
        """Validate unrealized P&L calculation."""
        if all([self.quantity, self.avg_cost, self.current_price]):
            expected_pnl = self.quantity * (self.current_price - self.avg_cost)
            # Allow small floating point differences
            if abs(self.unrealized_pnl - expected_pnl) > 0.01:
                raise ValueError(
                    f"Unrealized P&L mismatch. Expected: {expected_pnl:.2f}, "
                    f"Got: {self.unrealized_pnl:.2f}"
                )
        
        return self
    
    @property
    def market_value(self) -> float:
        """Calculate current market value of the position."""
        return abs(self.quantity) * self.current_price
    
    @property
    def total_pnl(self) -> float:
        """Calculate total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def is_long(self) -> bool:
        """Check if position is long (positive quantity)."""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short (negative quantity)."""
        return self.quantity < 0
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "symbol": "AAPL",
                "quantity": 100.0,
                "avg_cost": 150.0,
                "current_price": 155.0,
                "unrealized_pnl": 500.0,
                "realized_pnl": 0.0
            }
        }
    )


class Portfolio(BaseModel):
    """
    Portfolio model representing a collection of positions and cash.
    """
    user_id: str = Field(..., min_length=1, description="User identifier")
    positions: Dict[str, Position] = Field(default_factory=dict, description="Holdings by symbol")
    cash_balance: float = Field(..., ge=0, description="Available cash balance")
    total_value: float = Field(..., ge=0, description="Total portfolio value")
    last_updated: datetime = Field(..., description="Last update timestamp")
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        """Validate user ID format."""
        if not v or not v.strip():
            raise ValueError("User ID cannot be empty")
        return v.strip()
    
    @field_validator('positions')
    @classmethod
    def validate_positions(cls, v):
        """Validate positions dictionary."""
        for symbol, position in v.items():
            if symbol != position.symbol:
                raise ValueError(f"Position symbol {position.symbol} doesn't match key {symbol}")
        return v
    
    @model_validator(mode='after')
    def validate_total_value(self):
        """Validate total portfolio value calculation."""
        # Calculate expected total value
        positions_value = sum(pos.market_value for pos in self.positions.values())
        expected_total = self.cash_balance + positions_value
        
        # Allow small floating point differences
        if abs(self.total_value - expected_total) > 0.01:
            raise ValueError(
                f"Total value mismatch. Expected: {expected_total:.2f}, "
                f"Got: {self.total_value:.2f}"
            )
        
        return self
    
    @property
    def positions_value(self) -> float:
        """Calculate total value of all positions."""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def total_pnl(self) -> float:
        """Calculate total portfolio P&L."""
        return sum(pos.total_pnl for pos in self.positions.values())
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def realized_pnl(self) -> float:
        """Calculate total realized P&L."""
        return sum(pos.realized_pnl for pos in self.positions.values())
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        return self.positions.get(symbol.upper())
    
    def add_position(self, position: Position) -> None:
        """Add or update a position in the portfolio."""
        self.positions[position.symbol] = position
        self._recalculate_total_value()
    
    def remove_position(self, symbol: str) -> Optional[Position]:
        """Remove a position from the portfolio."""
        removed = self.positions.pop(symbol.upper(), None)
        if removed:
            self._recalculate_total_value()
        return removed
    
    def _recalculate_total_value(self) -> None:
        """Recalculate total portfolio value."""
        self.total_value = self.cash_balance + self.positions_value
        self.last_updated = datetime.now(timezone.utc)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "user123",
                "positions": {
                    "AAPL": {
                        "symbol": "AAPL",
                        "quantity": 100.0,
                        "avg_cost": 150.0,
                        "current_price": 155.0,
                        "unrealized_pnl": 500.0,
                        "realized_pnl": 0.0
                    }
                },
                "cash_balance": 10000.0,
                "total_value": 25500.0,
                "last_updated": "2023-12-01T15:30:00"
            }
        }
    )