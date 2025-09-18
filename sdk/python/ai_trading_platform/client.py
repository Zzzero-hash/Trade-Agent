"""
Main client for the AI Trading Platform SDK
"""

import asyncio
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from datetime import datetime, timedelta
import httpx

from .auth import AuthManager
from .models import *
from .exceptions import *
from .websocket import WebSocketClient


class TradingPlatformClient:
    """Main client for interacting with the AI Trading Platform API"""
    
    def __init__(self, base_url: str, client_id: Optional[str] = None,
                 client_secret: Optional[str] = None, timeout: float = 30.0):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.auth = AuthManager(base_url, client_id, client_secret)
        self.websocket: Optional[WebSocketClient] = None
        self._http_client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_http_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def _ensure_http_client(self):
        """Ensure HTTP client is initialized"""
        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)
    
    async def close(self):
        """Close all connections"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
    
    async def _make_request(self, method: str, endpoint: str, 
                          json_data: Optional[Dict] = None,
                          params: Optional[Dict] = None,
                          require_auth: bool = True) -> Dict[str, Any]:
        """Make HTTP request to API"""
        await self._ensure_http_client()
        
        url = f"{self.base_url}{endpoint}"
        headers = {}
        
        if require_auth:
            auth_headers = await self.auth.get_auth_headers()
            headers.update(auth_headers)
        
        try:
            response = await self._http_client.request(
                method=method,
                url=url,
                json=json_data,
                params=params,
                headers=headers
            )
            
            # Handle different response status codes
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                raise AuthenticationError("Authentication required or token expired")
            elif response.status_code == 403:
                raise AuthorizationError("Insufficient permissions")
            elif response.status_code == 422:
                error_data = response.json()
                raise ValidationError(
                    "Request validation failed",
                    validation_errors=error_data.get("detail", []),
                    status_code=response.status_code,
                    response_data=error_data
                )
            elif response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                raise RateLimitError(
                    "Rate limit exceeded",
                    retry_after=int(retry_after) if retry_after else None,
                    status_code=response.status_code
                )
            else:
                error_data = response.json() if response.content else {}
                raise APIError(
                    f"API request failed: {response.status_code}",
                    status_code=response.status_code,
                    response_data=error_data
                )
                
        except httpx.RequestError as e:
            raise NetworkError(f"Network error: {e}")
    
    # Authentication methods
    async def login(self, username: str, password: str) -> Dict[str, Any]:
        """Login with username and password"""
        return await self.auth.authenticate_with_credentials(username, password)
    
    async def login_with_oauth(self, authorization_code: str, 
                             redirect_uri: str) -> Dict[str, Any]:
        """Login with OAuth2 authorization code"""
        return await self.auth.authenticate_with_oauth(authorization_code, redirect_uri)
    
    async def login_with_api_key(self, api_key: str) -> Dict[str, Any]:
        """Login with API key"""
        return await self.auth.authenticate_with_api_key(api_key)
    
    async def logout(self) -> None:
        """Logout and clear authentication"""
        await self.auth.logout()
    
    def get_oauth_url(self, redirect_uri: str, state: Optional[str] = None,
                     scopes: Optional[List[str]] = None) -> str:
        """Get OAuth2 authorization URL"""
        return self.auth.get_oauth_authorization_url(redirect_uri, state, scopes)
    
    # Trading Signal methods
    async def generate_signal(self, symbol: str) -> TradingSignal:
        """Generate trading signal for a symbol"""
        response = await self._make_request(
            "POST", 
            f"/api/v1/trading/signals/generate?symbol={symbol}"
        )
        return TradingSignal(**response)
    
    async def get_signal_history(self, symbol: Optional[str] = None,
                               action: Optional[TradingAction] = None,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None,
                               limit: int = 100) -> List[TradingSignal]:
        """Get historical trading signals"""
        params = {"limit": limit}
        if symbol:
            params["symbol"] = symbol
        if action:
            params["action"] = action.value
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
        
        response = await self._make_request("GET", "/api/v1/trading/signals/history", params=params)
        return [TradingSignal(**signal) for signal in response]
    
    async def get_signal_performance(self, symbol: Optional[str] = None,
                                   days: int = 30) -> Dict[str, Any]:
        """Get trading signal performance metrics"""
        params = {"days": days}
        if symbol:
            params["symbol"] = symbol
        
        return await self._make_request("GET", "/api/v1/trading/signals/performance", params=params)
    
    # Portfolio methods
    async def get_portfolio(self) -> Portfolio:
        """Get current portfolio"""
        response = await self._make_request("GET", "/api/v1/trading/portfolio")
        return Portfolio(**response)
    
    async def rebalance_portfolio(self, target_allocation: Dict[str, float]) -> Dict[str, Any]:
        """Rebalance portfolio to target allocation"""
        return await self._make_request(
            "POST", 
            "/api/v1/trading/portfolio/rebalance",
            json_data=target_allocation
        )
    
    async def get_portfolio_performance(self, days: int = 30) -> Dict[str, Any]:
        """Get portfolio performance metrics"""
        return await self._make_request(
            "GET", 
            "/api/v1/trading/portfolio/performance",
            params={"days": days}
        )
    
    async def optimize_portfolio(self, optimization_method: str = "mean_variance",
                               risk_tolerance: float = 0.5) -> Dict[str, Any]:
        """Optimize portfolio allocation"""
        params = {
            "optimization_method": optimization_method,
            "risk_tolerance": risk_tolerance
        }
        return await self._make_request(
            "POST", 
            "/api/v1/trading/portfolio/optimize",
            params=params
        )
    
    # Market Data methods
    async def get_market_data(self, symbol: str, timeframe: str = "1h",
                            limit: int = 100) -> Dict[str, Any]:
        """Get historical market data"""
        params = {"timeframe": timeframe, "limit": limit}
        return await self._make_request(
            "GET", 
            f"/api/v1/trading/market-data/{symbol}",
            params=params
        )
    
    # Risk Management methods
    async def get_risk_metrics(self, portfolio: Portfolio,
                             confidence_level: float = 0.05) -> RiskMetrics:
        """Calculate risk metrics for portfolio"""
        response = await self._make_request(
            "POST",
            "/api/risk/metrics",
            json_data={
                "portfolio": portfolio.model_dump(),
                "confidence_level": confidence_level
            }
        )
        return RiskMetrics(**response)
    
    async def check_risk_limits(self, portfolio: Portfolio) -> List[Alert]:
        """Check risk limits and get alerts"""
        response = await self._make_request(
            "POST",
            "/api/risk/check-limits",
            json_data=portfolio.model_dump()
        )
        return [Alert(**alert) for alert in response]
    
    # Monitoring methods
    async def get_system_health(self) -> Dict[str, Any]:
        """Get system health status"""
        return await self._make_request("GET", "/monitoring/health")
    
    async def get_model_status(self, model_name: str) -> ModelStatus:
        """Get status of a specific model"""
        response = await self._make_request("GET", f"/monitoring/models/{model_name}/status")
        return ModelStatus(**response)
    
    async def get_alerts(self, hours: int = 24, severity: Optional[str] = None,
                        model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get recent alerts"""
        params = {"hours": hours}
        if severity:
            params["severity"] = severity
        if model_name:
            params["model_name"] = model_name
        
        return await self._make_request("GET", "/monitoring/alerts", params=params)
    
    # WebSocket methods
    async def connect_websocket(self) -> WebSocketClient:
        """Connect to WebSocket for real-time updates"""
        if not self.websocket:
            user_info = self.auth.get_user_info()
            if not user_info:
                raise AuthenticationError("Must be authenticated to connect WebSocket")
            
            user_id = user_info["user_id"]
            self.websocket = WebSocketClient(self.base_url, user_id, self.auth)
            await self.websocket.connect()
        
        return self.websocket
    
    async def subscribe_to_signals(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribe to real-time trading signals"""
        ws = await self.connect_websocket()
        await ws.subscribe_to_signals()
        
        async for message in ws.listen():
            if message.get("type") in ["signal_generated", "signals_subscribed"]:
                yield message
    
    async def subscribe_to_market_data(self, symbols: List[str]) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribe to real-time market data"""
        ws = await self.connect_websocket()
        await ws.subscribe_to_market_data(symbols)
        
        async for message in ws.listen():
            if message.get("type") in ["market_data", "market_data_subscribed"]:
                yield message
    
    # Utility methods
    def is_authenticated(self) -> bool:
        """Check if client is authenticated"""
        return self.auth.is_authenticated()
    
    def get_user_info(self) -> Optional[Dict[str, Any]]:
        """Get current user information"""
        return self.auth.get_user_info()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return await self._make_request("GET", "/api/v1/trading/health", require_auth=False)