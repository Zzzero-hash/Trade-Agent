"""Coinbase exchange connector implementation for cryptocurrency trading"""

import asyncio
import logging
import hmac
import hashlib
import base64
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, AsyncGenerator
import pandas as pd
import httpx
from dataclasses import dataclass, asdict
import json
from urllib.parse import urljoin

from .base import ExchangeConnector, MarketData, Order, OrderResult


@dataclass
class CoinbaseCredentials:
    """Coinbase API credentials"""
    api_key: str
    api_secret: str
    passphrase: str
    sandbox: bool = True


@dataclass
class CryptoOrder:
    """Cryptocurrency-specific order structure"""
    symbol: str  # e.g., "BTC-USD"
    side: str  # "buy" or "sell"
    size: Optional[str] = None  # Amount in base currency
    funds: Optional[str] = None  # Amount in quote currency
    order_type: str = "market"  # "market", "limit", "stop"
    price: Optional[str] = None
    stop_price: Optional[str] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    post_only: bool = False  # Maker-only orders
    client_oid: Optional[str] = None


class CryptoMarketHours:
    """Cryptocurrency market hours utility (24/7 market)"""
    
    @staticmethod
    def is_market_open(dt: Optional[datetime] = None) -> bool:
        """Crypto markets are always open (24/7)"""
        return True
    
    @staticmethod
    def get_next_market_open(dt: Optional[datetime] = None) -> datetime:
        """Crypto markets are always open"""
        if dt is None:
            dt = datetime.now(timezone.utc)
        return dt


class CoinbaseConnector(ExchangeConnector):
    """Coinbase exchange connector for cryptocurrency trading"""
    
    SANDBOX_URL = "https://api-public.sandbox.pro.coinbase.com"
    LIVE_URL = "https://api.pro.coinbase.com"
    WEBSOCKET_SANDBOX_URL = "wss://ws-feed-public.sandbox.pro.coinbase.com"
    WEBSOCKET_LIVE_URL = "wss://ws-feed.pro.coinbase.com"
    
    def __init__(self, api_key: str, api_secret: str, passphrase: str, sandbox: bool = True):
        super().__init__(api_key, api_secret, sandbox)
        self.credentials = CoinbaseCredentials(
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            sandbox=sandbox
        )
        
        # Set URLs based on environment
        self.base_url = self.SANDBOX_URL if sandbox else self.LIVE_URL
        self.websocket_url = self.WEBSOCKET_SANDBOX_URL if sandbox else self.WEBSOCKET_LIVE_URL
        
        # Initialize components
        self.logger = logging.getLogger(__name__)
        self._client: Optional[httpx.AsyncClient] = None
        self._websocket = None
        
        # Rate limiting
        self._last_request_time = 0
        self._request_count = 0
        self._rate_limit_window = 1  # 1 second
        self._max_requests_per_second = 10
    
    async def connect(self) -> bool:
        """Establish connection and validate credentials"""
        try:
            # Create HTTP client
            timeout = httpx.Timeout(30.0)
            self._client = httpx.AsyncClient(timeout=timeout)
            
            # Test connection by getting accounts
            response = await self._make_authenticated_request("GET", "/accounts")
            
            if response.status_code == 200:
                self.is_connected = True
                self.logger.info(f"Successfully connected to Coinbase {'sandbox' if self.sandbox else 'live'} environment")
                return True
            else:
                self.logger.error(f"Coinbase connection failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Coinbase connection error: {str(e)}")
            return False
    
    async def disconnect(self) -> None:
        """Close connection to Coinbase"""
        try:
            if self._client:
                await self._client.aclose()
                self._client = None
            
            if self._websocket:
                await self._websocket.close()
                self._websocket = None
            
            self.is_connected = False
            self.logger.info("Disconnected from Coinbase API")
            
        except Exception as e:
            self.logger.error(f"Error during Coinbase disconnect: {str(e)}")
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        
        # Reset counter if window has passed
        if current_time - self._last_request_time >= self._rate_limit_window:
            self._request_count = 0
            self._last_request_time = current_time
        
        # Check if we need to wait
        if self._request_count >= self._max_requests_per_second:
            wait_time = self._rate_limit_window - (current_time - self._last_request_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                self._request_count = 0
                self._last_request_time = time.time()
        
        self._request_count += 1
    
    def _generate_signature(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Generate authentication signature"""
        message = timestamp + method + path + body
        signature = hmac.new(
            base64.b64decode(self.credentials.api_secret),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode('utf-8')
    
    async def _make_authenticated_request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Make authenticated API request"""
        if not self._client:
            raise RuntimeError("Not connected to Coinbase. Call connect() first.")
        
        await self._rate_limit()
        
        # Generate authentication headers
        timestamp = str(time.time())
        body = json.dumps(kwargs.get('json', {})) if kwargs.get('json') else ""
        signature = self._generate_signature(timestamp, method, path, body)
        
        headers = {
            'CB-ACCESS-KEY': self.credentials.api_key,
            'CB-ACCESS-SIGN': signature,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-PASSPHRASE': self.credentials.passphrase,
            'Content-Type': 'application/json'
        }
        
        url = urljoin(self.base_url, path)
        response = await self._client.request(method, url, headers=headers, **kwargs)
        
        return response
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start: datetime, 
        end: datetime
    ) -> pd.DataFrame:
        """Get historical cryptocurrency data"""
        try:
            if not self.is_connected:
                raise RuntimeError("Not connected to Coinbase. Call connect() first.")
            
            # Convert timeframe to Coinbase granularity (in seconds)
            granularity_map = {
                "1m": 60,
                "5m": 300,
                "15m": 900,
                "1h": 3600,
                "6h": 21600,
                "1d": 86400
            }
            
            granularity = granularity_map.get(timeframe, 3600)
            
            # Format symbol for Coinbase (e.g., BTCUSD -> BTC-USD)
            coinbase_symbol = self._format_symbol(symbol)
            
            # Prepare parameters
            params = {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "granularity": granularity
            }
            
            response = await self._make_authenticated_request(
                "GET", 
                f"/products/{coinbase_symbol}/candles",
                params=params
            )
            
            if response.status_code == 200:
                candles = response.json()
                
                # Convert to DataFrame
                # Coinbase returns: [timestamp, low, high, open, close, volume]
                df_data = []
                for candle in candles:
                    timestamp = datetime.fromtimestamp(candle[0], tz=timezone.utc)
                    
                    df_data.append({
                        "timestamp": timestamp,
                        "open": float(candle[3]),
                        "high": float(candle[2]),
                        "low": float(candle[1]),
                        "close": float(candle[4]),
                        "volume": float(candle[5]),
                        "symbol": symbol.upper()
                    })
                
                df = pd.DataFrame(df_data)
                if not df.empty:
                    df = df.sort_values("timestamp")
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
        """Stream real-time cryptocurrency data via WebSocket"""
        try:
            if not self.is_connected:
                raise RuntimeError("Not connected to Coinbase. Call connect() first.")
            
            import websockets
            
            # Format symbols for Coinbase
            coinbase_symbols = [self._format_symbol(symbol) for symbol in symbols]
            
            # WebSocket subscription message
            subscribe_message = {
                "type": "subscribe",
                "product_ids": coinbase_symbols,
                "channels": ["ticker"]
            }
            
            async with websockets.connect(self.websocket_url) as websocket:
                self._websocket = websocket
                
                # Send subscription
                await websocket.send(json.dumps(subscribe_message))
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        if data.get("type") == "ticker":
                            product_id = data.get("product_id", "")
                            symbol = self._unformat_symbol(product_id)
                            
                            price = float(data.get("price", 0))
                            
                            market_data = MarketData(
                                symbol=symbol,
                                timestamp=datetime.now(timezone.utc),
                                open=price,
                                high=price,
                                low=price,
                                close=price,
                                volume=float(data.get("last_size", 0)),
                                exchange="COINBASE"
                            )
                            
                            yield market_data
                    
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        self.logger.error(f"Error processing WebSocket data: {str(e)}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Real-time data stream error: {str(e)}")
    
    async def place_order(self, order: Order) -> OrderResult:
        """Place a cryptocurrency order"""
        try:
            if not self.is_connected:
                raise RuntimeError("Not connected to Coinbase. Call connect() first.")
            
            # Convert to crypto order
            crypto_order = self._convert_to_crypto_order(order)
            
            # Prepare order data
            order_data = {
                "side": crypto_order.side,
                "product_id": self._format_symbol(crypto_order.symbol),
                "type": crypto_order.order_type
            }
            
            # Add size or funds
            if crypto_order.size:
                order_data["size"] = crypto_order.size
            elif crypto_order.funds:
                order_data["funds"] = crypto_order.funds
            
            # Add price for limit orders
            if crypto_order.order_type == "limit" and crypto_order.price:
                order_data["price"] = crypto_order.price
            
            # Add stop price for stop orders
            if crypto_order.order_type == "stop" and crypto_order.stop_price:
                order_data["stop_price"] = crypto_order.stop_price
            
            # Add additional parameters
            if crypto_order.time_in_force:
                order_data["time_in_force"] = crypto_order.time_in_force
            
            if crypto_order.post_only:
                order_data["post_only"] = crypto_order.post_only
            
            if crypto_order.client_oid:
                order_data["client_oid"] = crypto_order.client_oid
            
            # Place order
            response = await self._make_authenticated_request("POST", "/orders", json=order_data)
            
            if response.status_code == 200:
                result = response.json()
                
                return OrderResult(
                    order_id=result.get("id", ""),
                    status=result.get("status", "").upper(),
                    filled_quantity=float(result.get("filled_size", 0)),
                    avg_fill_price=float(result.get("executed_value", 0)) / max(float(result.get("filled_size", 1)), 1),
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
                raise RuntimeError("Not connected to Coinbase. Call connect() first.")
            
            response = await self._make_authenticated_request("DELETE", f"/orders/{order_id}")
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Error canceling order {order_id}: {str(e)}")
            return False
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information and balances"""
        try:
            if not self.is_connected:
                raise RuntimeError("Not connected to Coinbase. Call connect() first.")
            
            response = await self._make_authenticated_request("GET", "/accounts")
            
            if response.status_code == 200:
                accounts = response.json()
                
                total_balance = 0
                balances = {}
                
                for account in accounts:
                    currency = account.get("currency", "")
                    balance = float(account.get("balance", 0))
                    available = float(account.get("available", 0))
                    hold = float(account.get("hold", 0))
                    
                    balances[currency] = {
                        "balance": balance,
                        "available": available,
                        "hold": hold
                    }
                    
                    # Convert to USD for total (simplified)
                    if currency == "USD":
                        total_balance += balance
                
                return {
                    "total_balance": total_balance,
                    "balances": balances,
                    "account_count": len(accounts)
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {str(e)}")
            return {}
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions (non-zero balances)"""
        try:
            if not self.is_connected:
                raise RuntimeError("Not connected to Coinbase. Call connect() first.")
            
            response = await self._make_authenticated_request("GET", "/accounts")
            
            if response.status_code == 200:
                accounts = response.json()
                positions = []
                
                for account in accounts:
                    balance = float(account.get("balance", 0))
                    if balance > 0:
                        currency = account.get("currency", "")
                        
                        positions.append({
                            "symbol": currency,
                            "quantity": balance,
                            "average_cost": 0,  # Would need trade history to calculate
                            "market_value": balance,  # Simplified
                            "unrealized_pnl": 0  # Would need cost basis to calculate
                        })
                
                return positions
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            return []
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported cryptocurrency pairs"""
        # Major crypto pairs
        return [
            "BTCUSD", "ETHUSD", "LTCUSD", "BCHUSD", "ADAUSD", "DOTUSD", "LINKUSD",
            "XLMUSD", "XTZUSD", "ATOMUSD", "DASHUSD", "EOSУSD", "ETCUSD", "ZECUSD",
            "BTCEUR", "ETHEUR", "LTCEUR", "BCHEUR", "ADAEUR", "DOTEUR", "LINKEUR",
            "BTCGBP", "ETHGBP", "LTCGBP", "BCHGBP"
        ]
    
    def _format_symbol(self, symbol: str) -> str:
        """Convert symbol to Coinbase format (e.g., BTCUSD -> BTC-USD)"""
        symbol = symbol.upper().replace("-", "")
        
        # Common crypto base currencies
        crypto_bases = ["BTC", "ETH", "LTC", "BCH", "ADA", "DOT", "LINK", "XLM", "XTZ", "ATOM", "DASH", "EOS", "ETC", "ZEC"]
        quote_currencies = ["USD", "EUR", "GBP", "USDC", "DAI"]
        
        for base in crypto_bases:
            for quote in quote_currencies:
                if symbol == base + quote:
                    return f"{base}-{quote}"
        
        # If no match found, assume it's already in correct format or add dash in middle
        if len(symbol) >= 6:
            return f"{symbol[:3]}-{symbol[3:]}"
        
        return symbol
    
    def _unformat_symbol(self, coinbase_symbol: str) -> str:
        """Convert Coinbase symbol back to standard format"""
        return coinbase_symbol.replace("-", "")
    
    def _convert_to_crypto_order(self, order: Order) -> CryptoOrder:
        """Convert generic order to crypto-specific order"""
        return CryptoOrder(
            symbol=order.symbol,
            side=order.side.lower(),
            size=str(order.quantity),
            order_type=order.order_type.lower(),
            price=str(order.price) if order.price else None,
            stop_price=str(order.stop_price) if order.stop_price else None,
            time_in_force=order.time_in_force
        )
    
    # Crypto-specific methods
    async def get_current_prices(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Get current prices for crypto pairs"""
        try:
            if not self.is_connected:
                raise RuntimeError("Not connected to Coinbase. Call connect() first.")
            
            prices = {}
            
            for symbol in symbols:
                coinbase_symbol = self._format_symbol(symbol)
                
                response = await self._make_authenticated_request(
                    "GET", 
                    f"/products/{coinbase_symbol}/ticker"
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    prices[symbol] = {
                        "price": float(data.get("price", 0)),
                        "bid": float(data.get("bid", 0)),
                        "ask": float(data.get("ask", 0)),
                        "volume": float(data.get("volume", 0)),
                        "timestamp": data.get("time", "")
                    }
            
            return prices
            
        except Exception as e:
            self.logger.error(f"Error getting current prices: {str(e)}")
            return {}
    
    async def get_order_book(self, symbol: str, level: int = 1) -> Dict[str, Any]:
        """Get order book for a crypto pair"""
        try:
            if not self.is_connected:
                raise RuntimeError("Not connected to Coinbase. Call connect() first.")
            
            coinbase_symbol = self._format_symbol(symbol)
            
            response = await self._make_authenticated_request(
                "GET", 
                f"/products/{coinbase_symbol}/book",
                params={"level": level}
            )
            
            if response.status_code == 200:
                return response.json()
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting order book for {symbol}: {str(e)}")
            return {}
    
    async def get_trade_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trade history for a crypto pair"""
        try:
            if not self.is_connected:
                raise RuntimeError("Not connected to Coinbase. Call connect() first.")
            
            coinbase_symbol = self._format_symbol(symbol)
            
            response = await self._make_authenticated_request(
                "GET", 
                f"/products/{coinbase_symbol}/trades",
                params={"limit": limit}
            )
            
            if response.status_code == 200:
                return response.json()
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting trade history for {symbol}: {str(e)}")
            return []
    
    async def get_fills(self, order_id: Optional[str] = None, product_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get fills (executed trades) for orders"""
        try:
            if not self.is_connected:
                raise RuntimeError("Not connected to Coinbase. Call connect() first.")
            
            params = {}
            if order_id:
                params["order_id"] = order_id
            if product_id:
                params["product_id"] = product_id
            
            response = await self._make_authenticated_request("GET", "/fills", params=params)
            
            if response.status_code == 200:
                return response.json()
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting fills: {str(e)}")
            return []
    
    async def get_margin_profile(self) -> Dict[str, Any]:
        """Get margin profile information"""
        try:
            if not self.is_connected:
                raise RuntimeError("Not connected to Coinbase. Call connect() first.")
            
            response = await self._make_authenticated_request("GET", "/margin/profile_information")
            
            if response.status_code == 200:
                return response.json()
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting margin profile: {str(e)}")
            return {}
    
    async def place_margin_order(self, order: Order, leverage: float = 1.0) -> OrderResult:
        """Place a margin order with leverage"""
        try:
            if not self.is_connected:
                raise RuntimeError("Not connected to Coinbase. Call connect() first.")
            
            # Convert to crypto order
            crypto_order = self._convert_to_crypto_order(order)
            
            # Prepare margin order data
            order_data = {
                "side": crypto_order.side,
                "product_id": self._format_symbol(crypto_order.symbol),
                "type": crypto_order.order_type,
                "margin_type": "cross",  # or "isolated"
                "leverage": str(leverage)
            }
            
            # Add size or funds
            if crypto_order.size:
                order_data["size"] = crypto_order.size
            elif crypto_order.funds:
                order_data["funds"] = crypto_order.funds
            
            # Add price for limit orders
            if crypto_order.order_type == "limit" and crypto_order.price:
                order_data["price"] = crypto_order.price
            
            # Place margin order
            response = await self._make_authenticated_request("POST", "/margin/orders", json=order_data)
            
            if response.status_code == 200:
                result = response.json()
                
                return OrderResult(
                    order_id=result.get("id", ""),
                    status=result.get("status", "").upper(),
                    filled_quantity=float(result.get("filled_size", 0)),
                    avg_fill_price=float(result.get("executed_value", 0)) / max(float(result.get("filled_size", 1)), 1),
                    timestamp=datetime.now(timezone.utc),
                    message=f"Margin order placed with {leverage}x leverage"
                )
            else:
                raise RuntimeError(f"Margin order placement failed: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error placing margin order: {str(e)}")
            return OrderResult(
                order_id="",
                status="REJECTED",
                filled_quantity=0,
                avg_fill_price=None,
                timestamp=datetime.now(timezone.utc),
                message=str(e)
            )