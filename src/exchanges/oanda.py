"""OANDA exchange connector implementation for forex trading"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, AsyncGenerator
import pandas as pd
import httpx
from dataclasses import dataclass, asdict
import json
import time
from urllib.parse import urljoin

from .base import ExchangeConnector, MarketData, Order, OrderResult


@dataclass
class OANDACredentials:
    """OANDA API credentials"""
    api_key: str
    account_id: str
    environment: str = "practice"  # "practice" or "live"


@dataclass
class ForexOrder:
    """Forex-specific order structure"""
    symbol: str  # e.g., "EUR_USD"
    side: str  # "BUY" or "SELL"
    units: int  # Positive for long, negative for short
    order_type: str  # "MARKET", "LIMIT", "STOP", "MARKET_IF_TOUCHED"
    price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    time_in_force: str = "FOK"  # Fill or Kill
    position_fill: str = "DEFAULT"  # Position fill policy


class ForexMarketHours:
    """Forex market hours utility"""
    
    @staticmethod
    def is_market_open(dt: Optional[datetime] = None) -> bool:
        """Check if forex market is open (24/5 market)"""
        if dt is None:
            dt = datetime.now(timezone.utc)
        
        # Forex market is closed from Friday 22:00 UTC to Sunday 22:00 UTC
        weekday = dt.weekday()  # Monday=0, ..., Sunday=6
        hour = dt.hour
        
        # Friday after 22:00 UTC
        if weekday == 4 and hour >= 22:  # Friday after 22:00
            return False
        # Saturday (all day)
        elif weekday == 5:  # Saturday
            return False
        # Sunday before 22:00 UTC
        elif weekday == 6 and hour < 22:  # Sunday before 22:00
            return False
        
        return True
    
    @staticmethod
    def get_next_market_open(dt: Optional[datetime] = None) -> datetime:
        """Get next market open time"""
        if dt is None:
            dt = datetime.now(timezone.utc)
        
        # If market is open, return current time
        if ForexMarketHours.is_market_open(dt):
            return dt
        
        # Find next Sunday 22:00 UTC
        # weekday: Monday=0, Tuesday=1, ..., Sunday=6
        current_weekday = dt.weekday()
        
        if current_weekday == 6:  # Sunday
            if dt.hour < 22:
                # Before 22:00 on Sunday
                return dt.replace(hour=22, minute=0, second=0, microsecond=0)
            else:
                # Past Sunday 22:00, market should be open
                return dt
        else:
            # Calculate days until next Sunday
            days_until_sunday = (6 - current_weekday) % 7
            if days_until_sunday == 0:
                days_until_sunday = 7  # Next Sunday, not today
            
            next_sunday = dt + timedelta(days=days_until_sunday)
            next_open = next_sunday.replace(hour=22, minute=0, second=0, microsecond=0)
            
            return next_open


class OANDAConnector(ExchangeConnector):
    """OANDA exchange connector for forex trading"""
    
    PRACTICE_URL = "https://api-fxpractice.oanda.com"
    LIVE_URL = "https://api-fxtrade.oanda.com"
    STREAM_PRACTICE_URL = "https://stream-fxpractice.oanda.com"
    STREAM_LIVE_URL = "https://stream-fxtrade.oanda.com"
    
    def __init__(self, api_key: str, account_id: str, sandbox: bool = True):
        super().__init__(api_key, "", sandbox)
        self.credentials = OANDACredentials(
            api_key=api_key,
            account_id=account_id,
            environment="practice" if sandbox else "live"
        )
        
        # Set URLs based on environment
        if sandbox:
            self.base_url = self.PRACTICE_URL
            self.stream_url = self.STREAM_PRACTICE_URL
        else:
            self.base_url = self.LIVE_URL
            self.stream_url = self.STREAM_LIVE_URL
        
        # Initialize components
        self.logger = logging.getLogger(__name__)
        self._client: Optional[httpx.AsyncClient] = None
        self._stream_client: Optional[httpx.AsyncClient] = None
        
        # Session headers
        self.headers = {
            "Authorization": f"Bearer {self.credentials.api_key}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339"
        }
    
    async def connect(self) -> bool:
        """Establish connection and validate credentials"""
        try:
            # Create HTTP client
            timeout = httpx.Timeout(30.0)
            self._client = httpx.AsyncClient(timeout=timeout)
            
            # Test connection by getting account info
            response = await self._client.get(
                f"{self.base_url}/v3/accounts/{self.credentials.account_id}",
                headers=self.headers
            )
            
            if response.status_code == 200:
                self.is_connected = True
                self.logger.info(f"Successfully connected to OANDA {self.credentials.environment} environment")
                return True
            else:
                self.logger.error(f"OANDA connection failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"OANDA connection error: {str(e)}")
            return False
    
    async def disconnect(self) -> None:
        """Close connection to OANDA"""
        try:
            if self._client:
                await self._client.aclose()
                self._client = None
            
            if self._stream_client:
                await self._stream_client.aclose()
                self._stream_client = None
            
            self.is_connected = False
            self.logger.info("Disconnected from OANDA API")
            
        except Exception as e:
            self.logger.error(f"Error during OANDA disconnect: {str(e)}")
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start: datetime, 
        end: datetime
    ) -> pd.DataFrame:
        """Get historical forex data"""
        try:
            if not self.is_connected:
                raise RuntimeError("Not connected to OANDA. Call connect() first.")
            
            # Convert timeframe to OANDA granularity
            granularity_map = {
                "1m": "M1",
                "5m": "M5",
                "15m": "M15",
                "30m": "M30",
                "1h": "H1",
                "4h": "H4",
                "1d": "D",
                "1w": "W",
                "1M": "M"
            }
            
            granularity = granularity_map.get(timeframe, "H1")
            
            # Format symbol for OANDA (e.g., EURUSD -> EUR_USD)
            oanda_symbol = self._format_symbol(symbol)
            
            # Prepare parameters
            params = {
                "granularity": granularity,
                "from": start.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "to": end.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "price": "MBA"  # Mid, Bid, Ask prices
            }
            
            response = await self._client.get(
                f"{self.base_url}/v3/instruments/{oanda_symbol}/candles",
                headers=self.headers,
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                candles = data.get("candles", [])
                
                # Convert to DataFrame
                df_data = []
                for candle in candles:
                    if candle.get("complete", False):
                        timestamp = datetime.fromisoformat(candle["time"].replace("Z", "+00:00"))
                        mid_prices = candle["mid"]
                        
                        df_data.append({
                            "timestamp": timestamp,
                            "open": float(mid_prices["o"]),
                            "high": float(mid_prices["h"]),
                            "low": float(mid_prices["l"]),
                            "close": float(mid_prices["c"]),
                            "volume": int(candle.get("volume", 0)),
                            "symbol": symbol.upper()
                        })
                
                df = pd.DataFrame(df_data)
                if not df.empty:
                    df.set_index("timestamp", inplace=True)
                
                return df
            else:
                raise RuntimeError(f"Failed to get historical data: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    async def get_real_time_data(
        self, 
        symbols: List[str]
    ) -> AsyncGenerator[MarketData, None]:
        """Stream real-time forex data"""
        try:
            if not self.is_connected:
                raise RuntimeError("Not connected to OANDA. Call connect() first.")
            
            # Create streaming client
            if not self._stream_client:
                self._stream_client = httpx.AsyncClient(timeout=None)
            
            # Format symbols for OANDA
            oanda_symbols = [self._format_symbol(symbol) for symbol in symbols]
            instruments = ",".join(oanda_symbols)
            
            # Start streaming
            url = f"{self.stream_url}/v3/accounts/{self.credentials.account_id}/pricing/stream"
            params = {"instruments": instruments}
            
            async with self._stream_client.stream(
                "GET", 
                url, 
                headers=self.headers, 
                params=params
            ) as response:
                
                if response.status_code != 200:
                    raise RuntimeError(f"Streaming failed: {response.status_code}")
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            
                            if data.get("type") == "PRICE":
                                # Extract price data
                                instrument = data.get("instrument", "")
                                symbol = self._unformat_symbol(instrument)
                                
                                bids = data.get("bids", [])
                                asks = data.get("asks", [])
                                
                                if bids and asks:
                                    bid_price = float(bids[0]["price"])
                                    ask_price = float(asks[0]["price"])
                                    mid_price = (bid_price + ask_price) / 2
                                    
                                    market_data = MarketData(
                                        symbol=symbol,
                                        timestamp=datetime.now(timezone.utc),
                                        open=mid_price,
                                        high=mid_price,
                                        low=mid_price,
                                        close=mid_price,
                                        volume=0,  # Volume not available in streaming
                                        exchange="OANDA"
                                    )
                                    
                                    yield market_data
                        
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            self.logger.error(f"Error processing streaming data: {str(e)}")
                            continue
                            
        except Exception as e:
            self.logger.error(f"Real-time data stream error: {str(e)}")
    
    async def place_order(self, order: Order) -> OrderResult:
        """Place a forex order"""
        try:
            if not self.is_connected:
                raise RuntimeError("Not connected to OANDA. Call connect() first.")
            
            # Convert to forex order
            forex_order = self._convert_to_forex_order(order)
            
            # Prepare order data
            order_data = {
                "order": {
                    "type": forex_order.order_type,
                    "instrument": self._format_symbol(forex_order.symbol),
                    "units": str(forex_order.units),
                    "timeInForce": forex_order.time_in_force,
                    "positionFill": forex_order.position_fill
                }
            }
            
            # Add price for limit orders
            if forex_order.order_type == "LIMIT" and forex_order.price:
                order_data["order"]["price"] = str(forex_order.price)
            
            # Add stop loss and take profit
            if forex_order.stop_loss_price:
                order_data["order"]["stopLossOnFill"] = {
                    "price": str(forex_order.stop_loss_price)
                }
            
            if forex_order.take_profit_price:
                order_data["order"]["takeProfitOnFill"] = {
                    "price": str(forex_order.take_profit_price)
                }
            
            # Place order
            response = await self._client.post(
                f"{self.base_url}/v3/accounts/{self.credentials.account_id}/orders",
                headers=self.headers,
                json=order_data
            )
            
            if response.status_code == 201:
                result = response.json()
                order_create_transaction = result.get("orderCreateTransaction", {})
                order_fill_transaction = result.get("orderFillTransaction")
                
                # Determine status and fill info
                if order_fill_transaction:
                    status = "FILLED"
                    filled_quantity = abs(float(order_fill_transaction.get("units", 0)))
                    avg_fill_price = float(order_fill_transaction.get("price", 0))
                else:
                    status = "PENDING"
                    filled_quantity = 0
                    avg_fill_price = None
                
                return OrderResult(
                    order_id=order_create_transaction.get("id", ""),
                    status=status,
                    filled_quantity=filled_quantity,
                    avg_fill_price=avg_fill_price,
                    timestamp=datetime.now(timezone.utc),
                    message="Order placed successfully"
                )
            else:
                raise RuntimeError(f"Order placement failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            return OrderResult(
                order_id="",
                status="REJECTED",
                filled_quantity=0,
                avg_fill_price=None,
                timestamp=datetime.now(timezone.utc),
                message=str(e)
            )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        try:
            if not self.is_connected:
                raise RuntimeError("Not connected to OANDA. Call connect() first.")
            
            response = await self._client.put(
                f"{self.base_url}/v3/accounts/{self.credentials.account_id}/orders/{order_id}/cancel",
                headers=self.headers
            )
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Error canceling order {order_id}: {str(e)}")
            return False
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information and balances"""
        try:
            if not self.is_connected:
                raise RuntimeError("Not connected to OANDA. Call connect() first.")
            
            response = await self._client.get(
                f"{self.base_url}/v3/accounts/{self.credentials.account_id}",
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                account = data.get("account", {})
                
                return {
                    "account_id": account.get("id", ""),
                    "currency": account.get("currency", "USD"),
                    "balance": float(account.get("balance", 0)),
                    "nav": float(account.get("NAV", 0)),
                    "unrealized_pl": float(account.get("unrealizedPL", 0)),
                    "realized_pl": float(account.get("realizedPL", 0)),
                    "margin_used": float(account.get("marginUsed", 0)),
                    "margin_available": float(account.get("marginAvailable", 0)),
                    "open_trade_count": int(account.get("openTradeCount", 0)),
                    "open_position_count": int(account.get("openPositionCount", 0))
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {str(e)}")
            return {}
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        try:
            if not self.is_connected:
                raise RuntimeError("Not connected to OANDA. Call connect() first.")
            
            response = await self._client.get(
                f"{self.base_url}/v3/accounts/{self.credentials.account_id}/positions",
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                positions_data = data.get("positions", [])
                positions = []
                
                for pos in positions_data:
                    long_units = float(pos.get("long", {}).get("units", 0))
                    short_units = float(pos.get("short", {}).get("units", 0))
                    net_units = long_units + short_units
                    
                    if net_units != 0:
                        instrument = pos.get("instrument", "")
                        symbol = self._unformat_symbol(instrument)
                        
                        # Calculate average price
                        if long_units != 0:
                            avg_price = float(pos.get("long", {}).get("averagePrice", 0))
                            unrealized_pl = float(pos.get("long", {}).get("unrealizedPL", 0))
                        else:
                            avg_price = float(pos.get("short", {}).get("averagePrice", 0))
                            unrealized_pl = float(pos.get("short", {}).get("unrealizedPL", 0))
                        
                        positions.append({
                            "symbol": symbol,
                            "quantity": net_units,
                            "average_cost": avg_price,
                            "market_value": 0,  # Would need current price to calculate
                            "unrealized_pnl": unrealized_pl
                        })
                
                return positions
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            return []
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported forex pairs"""
        # Major forex pairs
        return [
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
            "EURJPY", "GBPJPY", "EURGBP", "EURAUD", "EURCHF", "AUDCAD", "GBPAUD",
            "GBPCAD", "GBPCHF", "AUDCHF", "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY",
            "GBPNZD", "EURNZD", "AUDNZD", "NZDCAD", "NZDCHF"
        ]
    
    def _format_symbol(self, symbol: str) -> str:
        """Convert symbol to OANDA format (e.g., EURUSD -> EUR_USD)"""
        symbol = symbol.upper().replace("/", "").replace("_", "")
        if len(symbol) == 6:
            return f"{symbol[:3]}_{symbol[3:]}"
        return symbol
    
    def _unformat_symbol(self, oanda_symbol: str) -> str:
        """Convert OANDA symbol back to standard format"""
        return oanda_symbol.replace("_", "")
    
    def _convert_to_forex_order(self, order: Order) -> ForexOrder:
        """Convert generic order to forex-specific order"""
        # Convert quantity to units (positive for buy, negative for sell)
        units = int(order.quantity) if order.side.upper() == "BUY" else -int(order.quantity)
        
        return ForexOrder(
            symbol=order.symbol,
            side=order.side,
            units=units,
            order_type=order.order_type,
            price=order.price,
            stop_loss_price=order.stop_price,
            time_in_force="FOK" if order.time_in_force == "IOC" else order.time_in_force
        )
    
    # Forex-specific methods
    async def get_current_prices(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Get current bid/ask prices for forex pairs"""
        try:
            if not self.is_connected:
                raise RuntimeError("Not connected to OANDA. Call connect() first.")
            
            oanda_symbols = [self._format_symbol(symbol) for symbol in symbols]
            instruments = ",".join(oanda_symbols)
            
            response = await self._client.get(
                f"{self.base_url}/v3/accounts/{self.credentials.account_id}/pricing",
                headers=self.headers,
                params={"instruments": instruments}
            )
            
            if response.status_code == 200:
                data = response.json()
                prices = {}
                
                for price_data in data.get("prices", []):
                    instrument = price_data.get("instrument", "")
                    symbol = self._unformat_symbol(instrument)
                    
                    bids = price_data.get("bids", [])
                    asks = price_data.get("asks", [])
                    
                    if bids and asks:
                        prices[symbol] = {
                            "bid": float(bids[0]["price"]),
                            "ask": float(asks[0]["price"]),
                            "spread": float(asks[0]["price"]) - float(bids[0]["price"]),
                            "timestamp": price_data.get("time", "")
                        }
                
                return prices
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting current prices: {str(e)}")
            return {}
    
    async def close_position(self, symbol: str, units: Optional[int] = None) -> OrderResult:
        """Close a position (or part of it)"""
        try:
            if not self.is_connected:
                raise RuntimeError("Not connected to OANDA. Call connect() first.")
            
            oanda_symbol = self._format_symbol(symbol)
            
            # Prepare close data
            close_data = {}
            if units:
                if units > 0:
                    close_data["longUnits"] = str(units)
                else:
                    close_data["shortUnits"] = str(abs(units))
            else:
                close_data["longUnits"] = "ALL"
                close_data["shortUnits"] = "ALL"
            
            response = await self._client.put(
                f"{self.base_url}/v3/accounts/{self.credentials.account_id}/positions/{oanda_symbol}/close",
                headers=self.headers,
                json=close_data
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract transaction info
                long_fill = result.get("longOrderFillTransaction")
                short_fill = result.get("shortOrderFillTransaction")
                
                fill_transaction = long_fill or short_fill
                
                if fill_transaction:
                    return OrderResult(
                        order_id=fill_transaction.get("id", ""),
                        status="FILLED",
                        filled_quantity=abs(float(fill_transaction.get("units", 0))),
                        avg_fill_price=float(fill_transaction.get("price", 0)),
                        timestamp=datetime.now(timezone.utc),
                        message="Position closed successfully"
                    )
            
            raise RuntimeError(f"Position close failed: {response.status_code}")
            
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {str(e)}")
            return OrderResult(
                order_id="",
                status="REJECTED",
                filled_quantity=0,
                avg_fill_price=None,
                timestamp=datetime.now(timezone.utc),
                message=str(e)
            )