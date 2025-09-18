"""
Trade Repository

Data access layer for trade records.
"""

from typing import List, Optional
from abc import ABC, abstractmethod
from datetime import datetime

from src.models.trading_models import Trade


class TradeRepository(ABC):
    """Abstract base class for trade data access."""
    
    @abstractmethod
    async def get_trades_by_customer(self, customer_id: str) -> List[Trade]:
        """Get trades for a specific customer."""
        pass
    
    @abstractmethod
    async def get_trade_by_id(self, trade_id: str) -> Optional[Trade]:
        """Get a trade by ID."""
        pass
    
    @abstractmethod
    async def create_trade(self, trade: Trade) -> Trade:
        """Create a new trade record."""
        pass
    
    @abstractmethod
    async def update_trade(self, trade: Trade) -> Trade:
        """Update an existing trade."""
        pass
    
    @abstractmethod
    async def get_trades_by_date_range(
        self, 
        customer_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Trade]:
        """Get trades within a date range."""
        pass


class InMemoryTradeRepository(TradeRepository):
    """In-memory implementation for testing."""
    
    def __init__(self):
        self.trades: List[Trade] = []
    
    async def get_trades_by_customer(self, customer_id: str) -> List[Trade]:
        return [t for t in self.trades if t.customer_id == customer_id]
    
    async def get_trade_by_id(self, trade_id: str) -> Optional[Trade]:
        for trade in self.trades:
            if trade.trade_id == trade_id:
                return trade
        return None
    
    async def create_trade(self, trade: Trade) -> Trade:
        self.trades.append(trade)
        return trade
    
    async def update_trade(self, trade: Trade) -> Trade:
        for i, t in enumerate(self.trades):
            if t.trade_id == trade.trade_id:
                self.trades[i] = trade
                return trade
        raise ValueError(f"Trade {trade.trade_id} not found")
    
    async def get_trades_by_date_range(
        self, 
        customer_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Trade]:
        return [
            t for t in self.trades 
            if (t.customer_id == customer_id and 
                start_date <= t.created_at <= end_date)
        ]