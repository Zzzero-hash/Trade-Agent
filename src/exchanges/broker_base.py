"""Enhanced broker abstraction layer for real money trading"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from enum import Enum
from dataclasses import dataclass, field
import uuid
from decimal import Decimal


class BrokerType(Enum):
    """Supported broker types"""
    ROBINHOOD = "robinhood"
    TD_AMERITRADE = "td_ameritrade"
    INTERACTIVE_BROKERS = "interactive_brokers"


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class TimeInForce(Enum):
    """Time in force enumeration"""
    GTC = "GTC"  # Good Till Cancelled
    DAY = "DAY"  # Day order
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill


@dataclass
class BrokerCredentials:
    """Broker authentication credentials"""
    broker_type: BrokerType
    client_id: str
    client_secret: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_expires_at: Optional[datetime] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingOrder:
    """Enhanced trading order structure"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    broker_type: Optional[BrokerType] = None
    account_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderExecution:
    """Order execution details"""
    execution_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    commission: Optional[Decimal] = None
    fees: Optional[Decimal] = None
    broker_execution_id: Optional[str] = None
    venue: Optional[str] = None


@dataclass
class OrderResult:
    """Enhanced order execution result"""
    order_id: str
    broker_order_id: Optional[str]
    status: OrderStatus
    filled_quantity: Decimal
    remaining_quantity: Decimal
    avg_fill_price: Optional[Decimal]
    total_commission: Optional[Decimal]
    total_fees: Optional[Decimal]
    timestamp: datetime
    executions: List[OrderExecution] = field(default_factory=list)
    message: Optional[str] = None
    error_code: Optional[str] = None


@dataclass
class Position:
    """Trading position"""
    symbol: str
    quantity: Decimal
    average_cost: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    account_id: str
    broker_type: BrokerType
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class AccountInfo:
    """Account information"""
    account_id: str
    broker_type: BrokerType
    account_type: str
    buying_power: Decimal
    cash_balance: Decimal
    total_value: Decimal
    day_trade_buying_power: Optional[Decimal] = None
    pattern_day_trader: bool = False
    margin_enabled: bool = False
    options_enabled: bool = False
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class TradeConfirmation:
    """Trade confirmation details"""
    confirmation_id: str
    order_id: str
    trade_date: datetime
    settlement_date: datetime
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    gross_amount: Decimal
    commission: Decimal
    fees: Decimal
    net_amount: Decimal
    account_id: str
    broker_type: BrokerType


class BrokerConnector(ABC):
    """Enhanced abstract base class for broker connectors"""

    def __init__(self, credentials: BrokerCredentials, sandbox: bool = True):
        self.credentials = credentials
        self.sandbox = sandbox
        self.is_connected = False
        self.is_authenticated = False
        self.last_heartbeat = None

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the broker"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the broker"""
        pass

    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with the broker using OAuth or other methods"""
        pass

    @abstractmethod
    async def refresh_token(self) -> bool:
        """Refresh authentication token"""
        pass

    @abstractmethod
    async def is_token_valid(self) -> bool:
        """Check if current token is valid"""
        pass

    @abstractmethod
    async def place_order(self, order: TradingOrder) -> OrderResult:
        """Place a trading order"""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, broker_order_id: Optional[str] = None) -> bool:
        """Cancel an existing order"""
        pass

    @abstractmethod
    async def modify_order(self, order_id: str, modifications: Dict[str, Any]) -> OrderResult:
        """Modify an existing order"""
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str, broker_order_id: Optional[str] = None) -> OrderResult:
        """Get current order status"""
        pass

    @abstractmethod
    async def get_order_history(self, start_date: datetime, end_date: datetime) -> List[OrderResult]:
        """Get order history for date range"""
        pass

    @abstractmethod
    async def get_account_info(self, account_id: Optional[str] = None) -> AccountInfo:
        """Get account information and balances"""
        pass

    @abstractmethod
    async def get_positions(self, account_id: Optional[str] = None) -> List[Position]:
        """Get current positions"""
        pass

    @abstractmethod
    async def get_trade_confirmations(self, start_date: datetime, end_date: datetime) -> List[TradeConfirmation]:
        """Get trade confirmations for settlement tracking"""
        pass

    @abstractmethod
    async def stream_order_updates(self) -> AsyncGenerator[OrderResult, None]:
        """Stream real-time order updates"""
        pass

    @abstractmethod
    async def stream_position_updates(self) -> AsyncGenerator[Position, None]:
        """Stream real-time position updates"""
        pass

    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols"""
        pass

    @abstractmethod
    def get_supported_order_types(self) -> List[OrderType]:
        """Get supported order types for this broker"""
        pass

    @abstractmethod
    async def validate_order(self, order: TradingOrder) -> Dict[str, Any]:
        """Validate order before submission"""
        pass

    async def heartbeat(self) -> bool:
        """Send heartbeat to maintain connection"""
        try:
            # Default implementation - can be overridden
            self.last_heartbeat = datetime.now()
            return self.is_connected and self.is_authenticated
        except Exception:
            return False

    def generate_order_id(self) -> str:
        """Generate unique order ID"""
        return str(uuid.uuid4())

    def is_market_hours(self) -> bool:
        """Check if market is currently open (basic implementation)"""
        now = datetime.now()
        # Basic US market hours check (9:30 AM - 4:00 PM ET, weekdays)
        if now.weekday() >= 5:  # Weekend
            return False
        
        # This is a simplified check - real implementation would consider holidays
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close