"""Abstract base classes for exchange connectors"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncGenerator
import pandas as pd
from dataclasses import dataclass


@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    exchange: str


@dataclass
class Order:
    """Order structure"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    order_type: str  # 'MARKET', 'LIMIT', 'STOP'
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'GTC'  # Good Till Cancelled


@dataclass
class OrderResult:
    """Order execution result"""
    order_id: str
    status: str  # 'FILLED', 'PARTIAL', 'PENDING', 'REJECTED'
    filled_quantity: float
    avg_fill_price: Optional[float]
    timestamp: datetime
    message: Optional[str] = None


class ExchangeConnector(ABC):
    """Abstract base class for exchange connectors"""

    def __init__(self, api_key: str, api_secret: str, sandbox: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.sandbox = sandbox
        self.is_connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the exchange"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the exchange"""
        pass

    @abstractmethod
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start: datetime, 
        end: datetime
    ) -> pd.DataFrame:
        """Get historical market data"""
        pass

    @abstractmethod
    async def get_real_time_data(
        self, 
        symbols: List[str]
    ) -> AsyncGenerator[MarketData, None]:
        """Stream real-time market data"""
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> OrderResult:
        """Place a trading order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information and balances"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        pass
    
    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols"""
        pass