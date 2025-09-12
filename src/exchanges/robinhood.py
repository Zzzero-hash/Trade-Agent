"""Robinhood exchange connector implementation"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, AsyncGenerator
import pandas as pd
import httpx
from dataclasses import dataclass, asdict
import json
import time
from urllib.parse import urljoin

from .base import ExchangeConnector, MarketData, Order, OrderResult


@dataclass
class RobinhoodCredentials:
    """Robinhood API credentials"""
    username: str
    password: str
    device_token: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None


class RateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self, max_requests: int = 1000, time_window: int = 3600):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make a request"""
        async with self._lock:
            now = time.time()
            # Remove old requests outside the time window
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < self.time_window]
            
            if len(self.requests) >= self.max_requests:
                # Calculate wait time
                oldest_request = min(self.requests)
                wait_time = self.time_window - (now - oldest_request)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            
            self.requests.append(now)


class ConnectionPool:
    """HTTP connection pool manager"""
    
    def __init__(self, max_connections: int = 10, timeout: float = 30.0):
        self.max_connections = max_connections
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()
    
    async def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    limits = httpx.Limits(
                        max_keepalive_connections=self.max_connections,
                        max_connections=self.max_connections
                    )
                    timeout = httpx.Timeout(self.timeout)
                    self._client = httpx.AsyncClient(limits=limits, timeout=timeout)
        return self._client
    
    async def close(self):
        """Close the HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None


class RobinhoodConnector(ExchangeConnector):
    """Robinhood exchange connector for stocks, ETFs, indices, and options"""
    
    BASE_URL = "https://robinhood.com/api/"
    SANDBOX_URL = "https://api.robinhood.com/"  # Placeholder for sandbox
    
    def __init__(self, username: str, password: str, sandbox: bool = True):
        # Initialize with credentials instead of api_key/secret
        super().__init__("", "", sandbox)
        self.credentials = RobinhoodCredentials(username=username, password=password)
        self.base_url = self.SANDBOX_URL if sandbox else self.BASE_URL
        
        # Initialize components
        self.rate_limiter = RateLimiter(max_requests=1000, time_window=3600)
        self.connection_pool = ConnectionPool(max_connections=10)
        self.logger = logging.getLogger(__name__)
        
        # Session state
        self.session_headers = {}
        self._authenticated = False
    
    async def connect(self) -> bool:
        """Establish connection and authenticate with Robinhood"""
        try:
            client = await self.connection_pool.get_client()
            
            # Authenticate
            auth_data = {
                "username": self.credentials.username,
                "password": self.credentials.password,
                "grant_type": "password",
                "client_id": "c82SH0WZOsabOXGP2sxqcj34FxkvfnWRZBKlBjFS"  # Public client ID
            }
            
            await self.rate_limiter.acquire()
            response = await client.post(
                urljoin(self.base_url, "api-token-auth/"),
                data=auth_data
            )
            
            if response.status_code == 200:
                auth_response = response.json()
                self.credentials.access_token = auth_response.get("access_token")
                self.credentials.refresh_token = auth_response.get("refresh_token")
                
                # Set up session headers
                self.session_headers = {
                    "Authorization": f"Token {self.credentials.access_token}",
                    "Content-Type": "application/json"
                }
                
                self._authenticated = True
                self.is_connected = True
                self.logger.info("Successfully connected to Robinhood API")
                return True
            else:
                self.logger.error(f"Authentication failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Connection failed: {str(e)}")
            return False
    
    async def disconnect(self) -> None:
        """Close connection to Robinhood"""
        try:
            if self.credentials.access_token:
                client = await self.connection_pool.get_client()
                await client.post(
                    urljoin(self.base_url, "api-token-logout/"),
                    headers=self.session_headers
                )
            
            await self.connection_pool.close()
            self.is_connected = False
            self._authenticated = False
            self.logger.info("Disconnected from Robinhood API")
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {str(e)}")
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        """Make authenticated API request with rate limiting"""
        if not self._authenticated:
            raise RuntimeError("Not authenticated. Call connect() first.")
        
        await self.rate_limiter.acquire()
        client = await self.connection_pool.get_client()
        
        url = urljoin(self.base_url, endpoint)
        headers = {**self.session_headers, **kwargs.pop("headers", {})}
        
        response = await client.request(method, url, headers=headers, **kwargs)
        
        # Handle token refresh if needed
        if response.status_code == 401 and self.credentials.refresh_token:
            await self._refresh_token()
            headers["Authorization"] = f"Token {self.credentials.access_token}"
            response = await client.request(method, url, headers=headers, **kwargs)
        
        return response
    
    async def _refresh_token(self) -> None:
        """Refresh access token using refresh token"""
        try:
            client = await self.connection_pool.get_client()
            refresh_data = {
                "refresh_token": self.credentials.refresh_token,
                "grant_type": "refresh_token"
            }
            
            response = await client.post(
                urljoin(self.base_url, "api-token-refresh/"),
                data=refresh_data
            )
            
            if response.status_code == 200:
                auth_response = response.json()
                self.credentials.access_token = auth_response.get("access_token")
                self.session_headers["Authorization"] = f"Token {self.credentials.access_token}"
                self.logger.info("Token refreshed successfully")
            else:
                self.logger.error("Token refresh failed")
                
        except Exception as e:
            self.logger.error(f"Token refresh error: {str(e)}")
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start: datetime, 
        end: datetime
    ) -> pd.DataFrame:
        """Get historical market data for stocks, ETFs, indices"""
        try:
            # Map timeframe to Robinhood intervals
            interval_map = {
                "1m": "5minute",
                "5m": "5minute", 
                "15m": "15minute",
                "1h": "hour",
                "1d": "day",
                "1w": "week"
            }
            
            interval = interval_map.get(timeframe, "day")
            
            # Get instrument ID first
            instrument_response = await self._make_request(
                "GET", 
                f"instruments/?symbol={symbol.upper()}"
            )
            
            if instrument_response.status_code != 200:
                raise ValueError(f"Symbol {symbol} not found")
            
            instruments = instrument_response.json()["results"]
            if not instruments:
                raise ValueError(f"No instrument found for symbol {symbol}")
            
            instrument_id = instruments[0]["id"]
            
            # Get historical data
            params = {
                "instrument": instrument_id,
                "interval": interval,
                "span": "year",  # Adjust based on date range
                "bounds": "regular"
            }
            
            response = await self._make_request("GET", "marketdata/historicals/", params=params)
            
            if response.status_code == 200:
                data = response.json()
                historicals = data.get("historicals", [])
                
                # Convert to DataFrame
                df_data = []
                for item in historicals:
                    timestamp = datetime.fromisoformat(item["begins_at"].replace("Z", "+00:00"))
                    if start <= timestamp <= end:
                        df_data.append({
                            "timestamp": timestamp,
                            "open": float(item["open_price"]),
                            "high": float(item["high_price"]),
                            "low": float(item["low_price"]),
                            "close": float(item["close_price"]),
                            "volume": int(item["volume"]),
                            "symbol": symbol.upper()
                        })
                
                df = pd.DataFrame(df_data)
                if not df.empty:
                    df.set_index("timestamp", inplace=True)
                
                return df
            else:
                raise RuntimeError(f"Failed to get historical data: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    async def get_real_time_data(
        self, 
        symbols: List[str]
    ) -> AsyncGenerator[MarketData, None]:
        """Stream real-time market data (polling-based implementation)"""
        try:
            while self.is_connected:
                for symbol in symbols:
                    try:
                        # Get current quote
                        response = await self._make_request(
                            "GET", 
                            f"quotes/?symbols={symbol.upper()}"
                        )
                        
                        if response.status_code == 200:
                            quotes = response.json()["results"]
                            if quotes:
                                quote = quotes[0]
                                
                                # Create MarketData object
                                market_data = MarketData(
                                    symbol=symbol.upper(),
                                    timestamp=datetime.now(),
                                    open=float(quote.get("previous_close", 0)),
                                    high=float(quote.get("last_trade_price", 0)),
                                    low=float(quote.get("last_trade_price", 0)),
                                    close=float(quote.get("last_trade_price", 0)),
                                    volume=0,  # Volume not available in quotes
                                    exchange="ROBINHOOD"
                                )
                                
                                yield market_data
                    
                    except Exception as e:
                        self.logger.error(f"Error getting real-time data for {symbol}: {str(e)}")
                
                # Wait before next poll
                await asyncio.sleep(1)  # 1 second polling interval
                
        except Exception as e:
            self.logger.error(f"Real-time data stream error: {str(e)}")
    
    async def place_order(self, order: Order) -> OrderResult:
        """Place a trading order"""
        try:
            # Get instrument ID
            instrument_response = await self._make_request(
                "GET", 
                f"instruments/?symbol={order.symbol.upper()}"
            )
            
            if instrument_response.status_code != 200:
                raise ValueError(f"Symbol {order.symbol} not found")
            
            instruments = instrument_response.json()["results"]
            if not instruments:
                raise ValueError(f"No instrument found for symbol {order.symbol}")
            
            instrument_url = instruments[0]["url"]
            
            # Prepare order data
            order_data = {
                "account": await self._get_account_url(),
                "instrument": instrument_url,
                "symbol": order.symbol.upper(),
                "side": order.side.lower(),
                "quantity": str(order.quantity),
                "type": order.order_type.lower(),
                "time_in_force": order.time_in_force.lower(),
                "trigger": "immediate"
            }
            
            if order.order_type.upper() == "LIMIT" and order.price:
                order_data["price"] = str(order.price)
            
            # Place order
            response = await self._make_request("POST", "orders/", json=order_data)
            
            if response.status_code == 201:
                order_response = response.json()
                
                return OrderResult(
                    order_id=order_response["id"],
                    status=order_response["state"].upper(),
                    filled_quantity=float(order_response.get("quantity", 0)),
                    avg_fill_price=float(order_response.get("price", 0)) if order_response.get("price") else None,
                    timestamp=datetime.now(),
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
                timestamp=datetime.now(),
                message=str(e)
            )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        try:
            response = await self._make_request("POST", f"orders/{order_id}/cancel/")
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Error canceling order {order_id}: {str(e)}")
            return False
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information and balances"""
        try:
            response = await self._make_request("GET", "accounts/")
            
            if response.status_code == 200:
                accounts = response.json()["results"]
                if accounts:
                    account = accounts[0]
                    
                    # Get portfolio info
                    portfolio_response = await self._make_request("GET", "accounts/{}/portfolio/".format(account["account_number"]))
                    portfolio = portfolio_response.json() if portfolio_response.status_code == 200 else {}
                    
                    return {
                        "account_id": account["account_number"],
                        "buying_power": float(account.get("buying_power", 0)),
                        "cash": float(account.get("cash", 0)),
                        "total_value": float(portfolio.get("total_return_today", 0)),
                        "day_trade_count": account.get("day_trade_buying_power_held", 0),
                        "pattern_day_trader": account.get("is_deactivated", False)
                    }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {str(e)}")
            return {}
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        try:
            response = await self._make_request("GET", "positions/")
            
            if response.status_code == 200:
                positions_data = response.json()["results"]
                positions = []
                
                for pos in positions_data:
                    if float(pos.get("quantity", 0)) > 0:
                        # Get instrument details
                        instrument_response = await self._make_request("GET", pos["instrument"])
                        instrument = instrument_response.json() if instrument_response.status_code == 200 else {}
                        
                        positions.append({
                            "symbol": instrument.get("symbol", "UNKNOWN"),
                            "quantity": float(pos.get("quantity", 0)),
                            "average_cost": float(pos.get("average_buy_price", 0)),
                            "market_value": float(pos.get("market_value", 0)),
                            "unrealized_pnl": float(pos.get("total_return_today", 0))
                        })
                
                return positions
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            return []
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols"""
        # This would typically be a large list or fetched from API
        # For now, return common symbols
        return [
            "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX",
            "SPY", "QQQ", "IWM", "VTI", "VOO", "VEA", "VWO", "BND"
        ]
    
    async def _get_account_url(self) -> str:
        """Get account URL for order placement"""
        try:
            response = await self._make_request("GET", "accounts/")
            if response.status_code == 200:
                accounts = response.json()["results"]
                if accounts:
                    return accounts[0]["url"]
            return ""
        except Exception:
            return ""
    
    # Options-specific methods
    async def get_options_chains(self, symbol: str, expiration_date: Optional[str] = None) -> Dict[str, Any]:
        """Get options chain for a symbol"""
        try:
            params = {"chain_symbol": symbol.upper()}
            if expiration_date:
                params["expiration_dates"] = expiration_date
            
            response = await self._make_request("GET", "options/chains/", params=params)
            
            if response.status_code == 200:
                return response.json()
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting options chain for {symbol}: {str(e)}")
            return {}
    
    async def place_options_order(self, options_order: Dict[str, Any]) -> OrderResult:
        """Place an options order"""
        try:
            response = await self._make_request("POST", "options/orders/", json=options_order)
            
            if response.status_code == 201:
                order_response = response.json()
                
                return OrderResult(
                    order_id=order_response["id"],
                    status=order_response["state"].upper(),
                    filled_quantity=float(order_response.get("quantity", 0)),
                    avg_fill_price=float(order_response.get("price", 0)) if order_response.get("price") else None,
                    timestamp=datetime.now(),
                    message="Options order placed successfully"
                )
            else:
                raise RuntimeError(f"Options order placement failed: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error placing options order: {str(e)}")
            return OrderResult(
                order_id="",
                status="REJECTED", 
                filled_quantity=0,
                avg_fill_price=None,
                timestamp=datetime.now(),
                message=str(e)
            )