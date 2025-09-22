"""
Advanced market data models with multi-timeframe support and comprehensive metadata.

This module implements sophisticated data structures for financial market data including:
- MarketData: Multi-timeframe OHLCV data with metadata and technical indicators
- OrderBook: Full depth order book with microstructure data
- Trade: Individual trades with market impact and execution quality metrics
- Data validation with statistical outlier detection
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class AssetClass(Enum):
    """Asset class classification."""
    EQUITY = "equity"
    FOREX = "forex"
    CRYPTO = "crypto"
    FUTURES = "futures"
    OPTIONS = "options"
    BONDS = "bonds"
    COMMODITIES = "commodities"


class TradeSide(Enum):
    """Trade side classification."""
    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"


@dataclass
class MarketMetadata:
    """Comprehensive metadata for market data."""
    symbol: str
    asset_class: AssetClass
    exchange: str
    currency: str
    tick_size: float
    lot_size: float
    trading_hours: Dict[str, Tuple[str, str]]  # day -> (open, close)
    timezone: str
    sector: Optional[str] = None
    market_cap: Optional[float] = None
    average_volume: Optional[float] = None
    volatility: Optional[float] = None
    beta: Optional[float] = None
    
    def __post_init__(self):
        """Validate metadata after initialization."""
        if self.tick_size <= 0:
            raise ValueError("Tick size must be positive")
        if self.lot_size <= 0:
            raise ValueError("Lot size must be positive")


@dataclass
class TechnicalIndicators:
    """Technical indicators for market analysis."""
    # Momentum indicators
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    stochastic_k: Optional[float] = None
    stochastic_d: Optional[float] = None
    williams_r: Optional[float] = None
    
    # Trend indicators
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    
    # Volatility indicators
    atr: Optional[float] = None
    volatility: Optional[float] = None
    
    # Volume indicators
    volume_sma: Optional[float] = None
    volume_ratio: Optional[float] = None
    on_balance_volume: Optional[float] = None
    
    # Custom indicators
    custom_indicators: Dict[str, float] = field(default_factory=dict)


@dataclass
class OrderBookLevel:
    """Individual order book level."""
    price: float
    size: float
    num_orders: int = 1
    
    def __post_init__(self):
        """Validate order book level data."""
        if self.price <= 0:
            raise ValueError("Price must be positive")
        if self.size <= 0:
            raise ValueError("Size must be positive")
        if self.num_orders < 0:
            raise ValueError("Number of orders cannot be negative")


@dataclass
class OrderBook:
    """
    Full depth order book with microstructure data.
    
    Provides comprehensive order book information including:
    - Full depth bids and asks
    - Market microstructure metrics
    - Order flow analysis
    - Liquidity measurements
    """
    timestamp: datetime
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    sequence_number: Optional[int] = None
    
    def __post_init__(self):
        """Validate and sort order book data."""
        # Sort bids in descending order (highest price first)
        self.bids.sort(key=lambda x: x.price, reverse=True)
        # Sort asks in ascending order (lowest price first)
        self.asks.sort(key=lambda x: x.price)
        
        # Validate that bids are below asks
        if self.bids and self.asks:
            best_bid = self.bids[0].price
            best_ask = self.asks[0].price
            if best_bid >= best_ask:
                logger.warning(f"Crossed book detected: bid={best_bid}, ask={best_ask}")
    
    @property
    def best_bid(self) -> Optional[float]:
        """Get the best bid price."""
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        """Get the best ask price."""
        return self.asks[0].price if self.asks else None
    
    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread."""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def spread_bps(self) -> Optional[float]:
        """Calculate spread in basis points."""
        if self.spread is not None and self.mid_price is not None and self.mid_price > 0:
            return (self.spread / self.mid_price) * 10000
        return None
    
    def get_depth(self, side: str, levels: int = 5) -> List[OrderBookLevel]:
        """Get order book depth for specified side."""
        if side.lower() == 'bid':
            return self.bids[:levels]
        elif side.lower() == 'ask':
            return self.asks[:levels]
        else:
            raise ValueError("Side must be 'bid' or 'ask'")
    
    def get_total_size(self, side: str, levels: int = 5) -> float:
        """Get total size for specified side and levels."""
        depth = self.get_depth(side, levels)
        return sum(level.size for level in depth)
    
    def get_weighted_price(self, side: str, levels: int = 5) -> Optional[float]:
        """Get size-weighted average price for specified side."""
        depth = self.get_depth(side, levels)
        if not depth:
            return None
        
        total_value = sum(level.price * level.size for level in depth)
        total_size = sum(level.size for level in depth)
        
        return total_value / total_size if total_size > 0 else None
    
    def calculate_imbalance(self, levels: int = 5) -> Optional[float]:
        """Calculate order book imbalance."""
        bid_size = self.get_total_size('bid', levels)
        ask_size = self.get_total_size('ask', levels)
        
        if bid_size + ask_size > 0:
            return (bid_size - ask_size) / (bid_size + ask_size)
        return None


@dataclass
class Trade:
    """
    Individual trade with market impact and execution quality metrics.
    
    Captures comprehensive trade information including:
    - Basic trade details (price, size, timestamp)
    - Market impact measurements
    - Execution quality metrics
    - Trade classification
    """
    timestamp: datetime
    symbol: str
    price: float
    size: float
    side: TradeSide
    trade_id: Optional[str] = None
    
    # Market microstructure data
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    mid_price: Optional[float] = None
    
    # Execution quality metrics
    effective_spread: Optional[float] = None
    price_improvement: Optional[float] = None
    market_impact: Optional[float] = None
    
    # Trade classification
    is_aggressive: Optional[bool] = None
    liquidity_flag: Optional[str] = None  # 'maker', 'taker', 'unknown'
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.price <= 0:
            raise ValueError("Trade price must be positive")
        if self.size <= 0:
            raise ValueError("Trade size must be positive")
        
        # Calculate mid price if bid/ask available
        if self.bid_price is not None and self.ask_price is not None:
            self.mid_price = (self.bid_price + self.ask_price) / 2
        
        # Calculate effective spread
        if self.mid_price is not None:
            self.effective_spread = 2 * abs(self.price - self.mid_price)
        
        # Determine trade aggressiveness
        if self.bid_price is not None and self.ask_price is not None:
            if self.side == TradeSide.BUY and self.price >= self.ask_price:
                self.is_aggressive = True
                self.liquidity_flag = 'taker'
            elif self.side == TradeSide.SELL and self.price <= self.bid_price:
                self.is_aggressive = True
                self.liquidity_flag = 'taker'
            elif self.side == TradeSide.BUY and self.price > self.bid_price:
                # Buy above bid but below ask - could be aggressive depending on exact price
                self.is_aggressive = self.price >= (self.bid_price + self.ask_price) / 2
                self.liquidity_flag = 'taker' if self.is_aggressive else 'maker'
            elif self.side == TradeSide.SELL and self.price < self.ask_price:
                # Sell below ask but above bid - could be aggressive depending on exact price  
                self.is_aggressive = self.price <= (self.bid_price + self.ask_price) / 2
                self.liquidity_flag = 'taker' if self.is_aggressive else 'maker'
            else:
                self.is_aggressive = False
                self.liquidity_flag = 'maker'
    
    @property
    def notional_value(self) -> float:
        """Calculate notional value of the trade."""
        return self.price * self.size
    
    def calculate_market_impact(self, reference_price: float) -> float:
        """Calculate market impact relative to reference price."""
        if self.side == TradeSide.BUY:
            return (self.price - reference_price) / reference_price
        elif self.side == TradeSide.SELL:
            return (reference_price - self.price) / reference_price
        return 0.0


@dataclass
class MarketData:
    """
    Advanced market data with multi-timeframe support and comprehensive metadata.
    
    Supports multiple timeframes and provides rich market information including:
    - OHLCV data across different timeframes
    - Technical indicators
    - Market microstructure data
    - Statistical validation and outlier detection
    """
    timestamp: datetime
    symbol: str
    timeframe: str  # '1m', '5m', '15m', '1h', '1d', etc.
    
    # OHLCV data
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    # Additional market data
    vwap: Optional[float] = None
    num_trades: Optional[int] = None
    
    # Multi-timeframe data
    multi_timeframe_data: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Metadata and indicators
    metadata: Optional[MarketMetadata] = None
    technical_indicators: Optional[TechnicalIndicators] = None
    
    # Market microstructure
    order_book: Optional[OrderBook] = None
    recent_trades: List[Trade] = field(default_factory=list)
    
    # Data quality metrics
    data_quality_score: Optional[float] = None
    is_outlier: bool = False
    outlier_reasons: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate and process market data after initialization."""
        self._validate_ohlcv()
        self._calculate_derived_metrics()
        self._detect_outliers()
    
    def _validate_ohlcv(self):
        """Validate OHLCV data consistency."""
        if self.open <= 0 or self.high <= 0 or self.low <= 0 or self.close <= 0:
            raise ValueError("All OHLCV prices must be positive")
        
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")
        
        if not (self.low <= self.open <= self.high and 
                self.low <= self.close <= self.high):
            raise ValueError("OHLC prices are inconsistent")
        
        if self.high < self.low:
            raise ValueError("High price cannot be less than low price")
    
    def _calculate_derived_metrics(self):
        """Calculate derived metrics from OHLCV data."""
        # Calculate VWAP if not provided
        if self.vwap is None and self.volume > 0:
            self.vwap = ((self.high + self.low + self.close) / 3)
        
        # Calculate typical price for technical analysis
        self.typical_price = (self.high + self.low + self.close) / 3
        
        # Calculate price range
        self.price_range = self.high - self.low
        self.price_range_pct = (self.price_range / self.close) * 100 if self.close > 0 else 0
        
        # Calculate body and shadow metrics for candlestick analysis
        self.body_size = abs(self.close - self.open)
        self.upper_shadow = self.high - max(self.open, self.close)
        self.lower_shadow = min(self.open, self.close) - self.low
    
    def _detect_outliers(self):
        """Detect statistical outliers in the data."""
        outlier_reasons = []
        
        # Check for extreme price movements
        if self.price_range_pct > 20:  # More than 20% range
            outlier_reasons.append("extreme_price_range")
        
        # Check for zero volume (suspicious)
        if self.volume == 0:
            outlier_reasons.append("zero_volume")
        
        # Check for gaps (if previous close available in multi-timeframe data)
        if '1d' in self.multi_timeframe_data:
            prev_close = self.multi_timeframe_data['1d'].get('prev_close')
            if prev_close and abs(self.open - prev_close) / prev_close > 0.1:  # 10% gap
                outlier_reasons.append("price_gap")
        
        self.outlier_reasons = outlier_reasons
        self.is_outlier = len(outlier_reasons) > 0
    
    def add_timeframe_data(self, timeframe: str, data: Dict[str, float]):
        """Add data for additional timeframe."""
        self.multi_timeframe_data[timeframe] = data
    
    def get_timeframe_data(self, timeframe: str) -> Optional[Dict[str, float]]:
        """Get data for specific timeframe."""
        return self.multi_timeframe_data.get(timeframe)
    
    def calculate_returns(self, periods: int = 1) -> Optional[float]:
        """Calculate returns over specified periods."""
        if f'{periods}d' in self.multi_timeframe_data:
            prev_data = self.multi_timeframe_data[f'{periods}d']
            prev_close = prev_data.get('close')
            if prev_close and prev_close > 0:
                return (self.close - prev_close) / prev_close
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'vwap': self.vwap,
            'num_trades': self.num_trades,
            'multi_timeframe_data': self.multi_timeframe_data,
            'data_quality_score': self.data_quality_score,
            'is_outlier': self.is_outlier,
            'outlier_reasons': self.outlier_reasons
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketData':
        """Create MarketData from dictionary."""
        timestamp = datetime.fromisoformat(data['timestamp'])
        
        return cls(
            timestamp=timestamp,
            symbol=data['symbol'],
            timeframe=data['timeframe'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            volume=data['volume'],
            vwap=data.get('vwap'),
            num_trades=data.get('num_trades'),
            multi_timeframe_data=data.get('multi_timeframe_data', {}),
            data_quality_score=data.get('data_quality_score'),
            is_outlier=data.get('is_outlier', False),
            outlier_reasons=data.get('outlier_reasons', [])
        )


class DataValidator:
    """Statistical data validation and outlier detection."""
    
    @staticmethod
    def validate_price_series(prices: List[float], 
                            z_threshold: float = 3.0) -> Tuple[List[bool], List[str]]:
        """
        Validate price series using statistical methods.
        
        Returns:
            Tuple of (outlier_flags, outlier_reasons)
        """
        if len(prices) < 3:
            return [False] * len(prices), ["insufficient_data"] * len(prices)
        
        prices_array = np.array(prices)
        returns = np.diff(np.log(prices_array))
        
        outlier_flags = []
        outlier_reasons = []
        
        # Calculate z-scores for returns
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Also calculate price level statistics
            mean_price = np.mean(prices_array)
            std_price = np.std(prices_array)
            
            for i, price in enumerate(prices):
                reasons = []
                is_outlier = False
                
                if i > 0:  # Can calculate return
                    return_val = returns[i-1]
                    if std_return > 0:
                        z_score = abs(return_val - mean_return) / std_return
                        if z_score >= z_threshold:
                            is_outlier = True
                            reasons.append(f"extreme_return_zscore_{z_score:.2f}")
                
                # Check for price level outliers
                if std_price > 0:
                    price_z_score = abs(price - mean_price) / std_price
                    if price_z_score >= z_threshold:
                        is_outlier = True
                        reasons.append(f"extreme_price_zscore_{price_z_score:.2f}")
                
                outlier_flags.append(is_outlier)
                outlier_reasons.append(reasons if reasons else ["normal"])
        
        return outlier_flags, outlier_reasons
    
    @staticmethod
    def validate_volume_series(volumes: List[float], 
                             multiplier_threshold: float = 10.0) -> Tuple[List[bool], List[str]]:
        """
        Validate volume series for anomalies.
        
        Returns:
            Tuple of (outlier_flags, outlier_reasons)
        """
        if len(volumes) < 3:
            return [False] * len(volumes), ["insufficient_data"] * len(volumes)
        
        volumes_array = np.array(volumes)
        median_volume = np.median(volumes_array)
        
        outlier_flags = []
        outlier_reasons = []
        
        for volume in volumes:
            reasons = []
            is_outlier = False
            
            # Check for zero volume
            if volume == 0:
                is_outlier = True
                reasons.append("zero_volume")
            
            # Check for extreme volume spikes
            elif median_volume > 0 and volume > median_volume * multiplier_threshold:
                is_outlier = True
                reasons.append(f"volume_spike_{volume/median_volume:.1f}x")
            
            outlier_flags.append(is_outlier)
            outlier_reasons.append(reasons if reasons else ["normal"])
        
        return outlier_flags, outlier_reasons


# Factory functions for creating market data objects
def create_market_data(symbol: str, 
                      timestamp: datetime,
                      timeframe: str,
                      ohlcv: Tuple[float, float, float, float, float],
                      metadata: Optional[MarketMetadata] = None) -> MarketData:
    """Factory function to create MarketData objects."""
    open_price, high, low, close, volume = ohlcv
    
    return MarketData(
        timestamp=timestamp,
        symbol=symbol,
        timeframe=timeframe,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        metadata=metadata
    )


def create_order_book(symbol: str,
                     timestamp: datetime,
                     bids: List[Tuple[float, float]],
                     asks: List[Tuple[float, float]]) -> OrderBook:
    """Factory function to create OrderBook objects."""
    bid_levels = [OrderBookLevel(price=price, size=size) for price, size in bids]
    ask_levels = [OrderBookLevel(price=price, size=size) for price, size in asks]
    
    return OrderBook(
        timestamp=timestamp,
        symbol=symbol,
        bids=bid_levels,
        asks=ask_levels
    )


def create_trade(symbol: str,
                timestamp: datetime,
                price: float,
                size: float,
                side: str,
                bid_price: Optional[float] = None,
                ask_price: Optional[float] = None) -> Trade:
    """Factory function to create Trade objects."""
    trade_side = TradeSide.BUY if side.lower() == 'buy' else TradeSide.SELL
    
    return Trade(
        timestamp=timestamp,
        symbol=symbol,
        price=price,
        size=size,
        side=trade_side,
        bid_price=bid_price,
        ask_price=ask_price
    )