"""TD Ameritrade broker connector for real money trading"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, AsyncGenerator
import httpx
from urllib.parse import urlencode, parse_qs, urlparse
import json
import base64
from decimal import Decimal

from .broker_base import (
    BrokerConnector, BrokerCredentials, BrokerType, TradingOrder, OrderResult, 
    OrderStatus, OrderSide, OrderType, TimeInForce, Position, AccountInfo, 
    TradeConfirmation, OrderExecution
)


class TDAmeritradeBrokerConnector(BrokerConnector):
    """TD Ameritrade broker connector with OAuth 2.0 authentication"""
    
    BASE_URL = "https://api.tdameritrade.com/v1"
    AUTH_URL = "https://auth.tdameritrade.com/auth"
    TOKEN_URL = "https://api.tdameritrade.com/v1/oauth2/token"
    
    def __init__(self, credentials: BrokerCredentials, sandbox: bool = True):
        super().__init__(credentials, sandbox)
        self.logger = logging.getLogger(__name__)
        
        # HTTP client
        self.client: Optional[httpx.AsyncClient] = None
        
        # Account info cache
        self.account_cache: Dict[str, AccountInfo] = {}
        self.positions_cache: Dict[str, List[Position]] = {}
        
        # Rate limiting
        self.rate_limit_remaining = 120  # TD Ameritrade allows 120 requests per minute
        self.rate_limit_reset = datetime.now()
    
    async def connect(self) -> bool:
        """Establish connection to TD Ameritrade"""
        try:
            # Create HTTP client
            timeout = httpx.Timeout(30.0)
            self.client = httpx.AsyncClient(timeout=timeout)
            
            self.is_connected = True
            self.logger.info("Connected to TD Ameritrade API")
            return True
            
        except Exception as e:
            self.logger.error(f"Connection failed: {str(e)}")
            return False
    
    async def disconnect(self) -> None:
        """Close connection to TD Ameritrade"""
        try:
            if self.client:
                await self.client.aclose()
                self.client = None
            
            self.is_connected = False
            self.is_authenticated = False
            self.logger.info("Disconnected from TD Ameritrade API")
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {str(e)}")
    
    async def authenticate(self) -> bool:
        """Authenticate using OAuth 2.0 flow"""
        try:
            if not self.credentials.access_token:
                # Need to perform OAuth flow
                auth_url = await self._get_authorization_url()
                self.logger.info(f"Please visit this URL to authorize: {auth_url}")
                return False
            
            # Validate existing token
            if await self.is_token_valid():
                self.is_authenticated = True
                return True
            
            # Try to refresh token
            if self.credentials.refresh_token:
                return await self.refresh_token()
            
            return False
            
        except Exception as e:
            self.logger.error(f"Authentication failed: {str(e)}")
            return False
    
    async def refresh_token(self) -> bool:
        """Refresh access token using refresh token"""
        try:
            if not self.credentials.refresh_token:
                return False
            
            data = {
                "grant_type": "refresh_token",
                "refresh_token": self.credentials.refresh_token,
                "client_id": self.credentials.client_id
            }
            
            response = await self.client.post(self.TOKEN_URL, data=data)
            
            if response.status_code == 200:
                token_data = response.json()
                
                self.credentials.access_token = token_data["access_token"]
                if "refresh_token" in token_data:
                    self.credentials.refresh_token = token_data["refresh_token"]
                
                expires_in = token_data.get("expires_in", 1800)  # 30 minutes default
                self.credentials.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                
                self.is_authenticated = True
                self.logger.info("Token refreshed successfully")
                return True
            else:
                self.logger.error(f"Token refresh failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Token refresh error: {str(e)}")
            return False
    
    async def is_token_valid(self) -> bool:
        """Check if current token is valid"""
        try:
            if not self.credentials.access_token:
                return False
            
            if self.credentials.token_expires_at:
                # Check if token expires within 5 minutes
                if datetime.now() + timedelta(minutes=5) >= self.credentials.token_expires_at:
                    return False
            
            # Test token with a simple API call
            response = await self._make_request("GET", "/accounts")
            return response.status_code == 200
            
        except Exception:
            return False
    
    async def place_order(self, order: TradingOrder) -> OrderResult:
        """Place a trading order"""
        try:
            # Get account ID
            account_id = order.account_id or await self._get_default_account_id()
            if not account_id:
                raise ValueError("No account ID available")
            
            # Build order payload
            order_payload = await self._build_order_payload(order)
            
            # Submit order
            response = await self._make_request(
                "POST", 
                f"/accounts/{account_id}/orders",
                json=order_payload
            )
            
            if response.status_code == 201:
                # Order placed successfully
                location_header = response.headers.get("Location", "")
                broker_order_id = location_header.split("/")[-1] if location_header else None
                
                return OrderResult(
                    order_id=order.order_id,
                    broker_order_id=broker_order_id,
                    status=OrderStatus.PENDING,
                    filled_quantity=Decimal('0'),
                    remaining_quantity=order.quantity,
                    avg_fill_price=None,
                    total_commission=None,
                    total_fees=None,
                    timestamp=datetime.now(),
                    message="Order submitted successfully"
                )
            else:
                error_msg = response.text
                return OrderResult(
                    order_id=order.order_id,
                    broker_order_id=None,
                    status=OrderStatus.REJECTED,
                    filled_quantity=Decimal('0'),
                    remaining_quantity=order.quantity,
                    avg_fill_price=None,
                    total_commission=None,
                    total_fees=None,
                    timestamp=datetime.now(),
                    message=f"Order rejected: {error_msg}"
                )
                
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            return OrderResult(
                order_id=order.order_id,
                broker_order_id=None,
                status=OrderStatus.REJECTED,
                filled_quantity=Decimal('0'),
                remaining_quantity=order.quantity,
                avg_fill_price=None,
                total_commission=None,
                total_fees=None,
                timestamp=datetime.now(),
                message=str(e)
            )
    
    async def cancel_order(self, order_id: str, broker_order_id: Optional[str] = None) -> bool:
        """Cancel an existing order"""
        try:
            if not broker_order_id:
                return False
            
            account_id = await self._get_default_account_id()
            if not account_id:
                return False
            
            response = await self._make_request(
                "DELETE",
                f"/accounts/{account_id}/orders/{broker_order_id}"
            )
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Error cancelling order: {str(e)}")
            return False
    
    async def modify_order(self, order_id: str, modifications: Dict[str, Any]) -> OrderResult:
        """Modify an existing order"""
        # TD Ameritrade requires cancelling and replacing orders
        # This is a simplified implementation
        raise NotImplementedError("Order modification requires cancel and replace")
    
    async def get_order_status(self, order_id: str, broker_order_id: Optional[str] = None) -> OrderResult:
        """Get current order status"""
        try:
            if not broker_order_id:
                # Search for order by client order ID
                orders = await self.get_order_history(datetime.now() - timedelta(days=1), datetime.now())
                for order_result in orders:
                    if order_result.order_id == order_id:
                        return order_result
                
                # Order not found
                return OrderResult(
                    order_id=order_id,
                    broker_order_id=None,
                    status=OrderStatus.REJECTED,
                    filled_quantity=Decimal('0'),
                    remaining_quantity=Decimal('0'),
                    avg_fill_price=None,
                    total_commission=None,
                    total_fees=None,
                    timestamp=datetime.now(),
                    message="Order not found"
                )
            
            account_id = await self._get_default_account_id()
            if not account_id:
                raise ValueError("No account ID available")
            
            response = await self._make_request(
                "GET",
                f"/accounts/{account_id}/orders/{broker_order_id}"
            )
            
            if response.status_code == 200:
                order_data = response.json()
                return await self._parse_order_response(order_data)
            else:
                return OrderResult(
                    order_id=order_id,
                    broker_order_id=broker_order_id,
                    status=OrderStatus.REJECTED,
                    filled_quantity=Decimal('0'),
                    remaining_quantity=Decimal('0'),
                    avg_fill_price=None,
                    total_commission=None,
                    total_fees=None,
                    timestamp=datetime.now(),
                    message="Failed to get order status"
                )
                
        except Exception as e:
            self.logger.error(f"Error getting order status: {str(e)}")
            return OrderResult(
                order_id=order_id,
                broker_order_id=broker_order_id,
                status=OrderStatus.REJECTED,
                filled_quantity=Decimal('0'),
                remaining_quantity=Decimal('0'),
                avg_fill_price=None,
                total_commission=None,
                total_fees=None,
                timestamp=datetime.now(),
                message=str(e)
            )
    
    async def get_order_history(self, start_date: datetime, end_date: datetime) -> List[OrderResult]:
        """Get order history for date range"""
        try:
            account_id = await self._get_default_account_id()
            if not account_id:
                return []
            
            params = {
                "fromEnteredTime": start_date.strftime("%Y-%m-%d"),
                "toEnteredTime": end_date.strftime("%Y-%m-%d"),
                "status": "FILLED,CANCELED,REJECTED,EXPIRED"
            }
            
            response = await self._make_request(
                "GET",
                f"/accounts/{account_id}/orders",
                params=params
            )
            
            if response.status_code == 200:
                orders_data = response.json()
                order_results = []
                
                for order_data in orders_data:
                    order_result = await self._parse_order_response(order_data)
                    order_results.append(order_result)
                
                return order_results
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting order history: {str(e)}")
            return []
    
    async def get_account_info(self, account_id: Optional[str] = None) -> AccountInfo:
        """Get account information and balances"""
        try:
            if not account_id:
                account_id = await self._get_default_account_id()
            
            if not account_id:
                raise ValueError("No account ID available")
            
            # Check cache
            if account_id in self.account_cache:
                cached_info = self.account_cache[account_id]
                if datetime.now() - cached_info.last_updated < timedelta(minutes=5):
                    return cached_info
            
            response = await self._make_request(
                "GET",
                f"/accounts/{account_id}",
                params={"fields": "positions"}
            )
            
            if response.status_code == 200:
                account_data = response.json()
                account_info = await self._parse_account_response(account_data)
                
                # Cache the result
                self.account_cache[account_id] = account_info
                
                return account_info
            else:
                raise RuntimeError(f"Failed to get account info: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error getting account info: {str(e)}")
            raise
    
    async def get_positions(self, account_id: Optional[str] = None) -> List[Position]:
        """Get current positions"""
        try:
            if not account_id:
                account_id = await self._get_default_account_id()
            
            if not account_id:
                return []
            
            # Check cache
            if account_id in self.positions_cache:
                # Cache positions for 1 minute
                cache_time = getattr(self, f"_positions_cache_time_{account_id}", None)
                if cache_time and datetime.now() - cache_time < timedelta(minutes=1):
                    return self.positions_cache[account_id]
            
            response = await self._make_request(
                "GET",
                f"/accounts/{account_id}",
                params={"fields": "positions"}
            )
            
            if response.status_code == 200:
                account_data = response.json()
                positions = await self._parse_positions_response(account_data, account_id)
                
                # Cache the result
                self.positions_cache[account_id] = positions
                setattr(self, f"_positions_cache_time_{account_id}", datetime.now())
                
                return positions
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            return []
    
    async def get_trade_confirmations(self, start_date: datetime, end_date: datetime) -> List[TradeConfirmation]:
        """Get trade confirmations for settlement tracking"""
        try:
            account_id = await self._get_default_account_id()
            if not account_id:
                return []
            
            params = {
                "type": "TRADE",
                "startDate": start_date.strftime("%Y-%m-%d"),
                "endDate": end_date.strftime("%Y-%m-%d")
            }
            
            response = await self._make_request(
                "GET",
                f"/accounts/{account_id}/transactions",
                params=params
            )
            
            if response.status_code == 200:
                transactions = response.json()
                confirmations = []
                
                for transaction in transactions:
                    if transaction.get("type") == "TRADE":
                        confirmation = await self._parse_trade_confirmation(transaction, account_id)
                        if confirmation:
                            confirmations.append(confirmation)
                
                return confirmations
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting trade confirmations: {str(e)}")
            return []
    
    async def stream_order_updates(self) -> AsyncGenerator[OrderResult, None]:
        """Stream real-time order updates"""
        # TD Ameritrade streaming would require WebSocket connection
        # This is a simplified polling implementation
        while self.is_connected:
            try:
                # Get recent orders
                orders = await self.get_order_history(
                    datetime.now() - timedelta(hours=1),
                    datetime.now()
                )
                
                for order in orders:
                    yield order
                
                await asyncio.sleep(30)  # Poll every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in order stream: {str(e)}")
                await asyncio.sleep(60)
    
    async def stream_position_updates(self) -> AsyncGenerator[Position, None]:
        """Stream real-time position updates"""
        # Simplified polling implementation
        last_positions = {}
        
        while self.is_connected:
            try:
                current_positions = await self.get_positions()
                
                for position in current_positions:
                    position_key = f"{position.account_id}_{position.symbol}"
                    
                    # Check if position changed
                    if (position_key not in last_positions or 
                        last_positions[position_key].quantity != position.quantity or
                        last_positions[position_key].market_value != position.market_value):
                        
                        last_positions[position_key] = position
                        yield position
                
                await asyncio.sleep(60)  # Poll every minute
                
            except Exception as e:
                self.logger.error(f"Error in position stream: {str(e)}")
                await asyncio.sleep(120)
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols"""
        # TD Ameritrade supports most US stocks, ETFs, options, futures
        # This would typically be fetched from their instruments API
        return ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "SPY", "QQQ", "IWM"]
    
    def get_supported_order_types(self) -> List[OrderType]:
        """Get supported order types"""
        return [OrderType.MARKET, OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT]
    
    async def validate_order(self, order: TradingOrder) -> Dict[str, Any]:
        """Validate order before submission"""
        try:
            # Basic validation
            if order.quantity <= 0:
                return {"valid": False, "reason": "Invalid quantity"}
            
            if order.order_type == OrderType.LIMIT and not order.price:
                return {"valid": False, "reason": "Limit order requires price"}
            
            # Check if symbol is supported
            if order.symbol not in self.get_supported_symbols():
                return {"valid": False, "reason": f"Symbol {order.symbol} not supported"}
            
            # Check market hours for market orders
            if order.order_type == OrderType.MARKET and not self.is_market_hours():
                return {"valid": False, "reason": "Market orders only allowed during market hours"}
            
            return {"valid": True, "reason": "Order validation passed"}
            
        except Exception as e:
            return {"valid": False, "reason": f"Validation error: {str(e)}"}
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        """Make authenticated API request with rate limiting"""
        if not self.is_authenticated:
            raise RuntimeError("Not authenticated")
        
        # Check rate limits
        await self._check_rate_limit()
        
        url = f"{self.BASE_URL}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.credentials.access_token}",
            "Content-Type": "application/json"
        }
        
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))
        
        response = await self.client.request(method, url, headers=headers, **kwargs)
        
        # Update rate limit info
        self.rate_limit_remaining = int(response.headers.get("X-RateLimit-Remaining", self.rate_limit_remaining))
        
        return response
    
    async def _check_rate_limit(self):
        """Check and enforce rate limits"""
        if self.rate_limit_remaining <= 1:
            # Wait until rate limit resets
            wait_time = (self.rate_limit_reset - datetime.now()).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            self.rate_limit_remaining = 120
            self.rate_limit_reset = datetime.now() + timedelta(minutes=1)
    
    async def _get_authorization_url(self) -> str:
        """Get OAuth authorization URL"""
        params = {
            "response_type": "code",
            "redirect_uri": self.credentials.additional_params.get("redirect_uri", "https://localhost"),
            "client_id": f"{self.credentials.client_id}@AMER.OAUTHAP"
        }
        
        return f"{self.AUTH_URL}?{urlencode(params)}"
    
    async def _get_default_account_id(self) -> Optional[str]:
        """Get default account ID"""
        try:
            response = await self._make_request("GET", "/accounts")
            
            if response.status_code == 200:
                accounts = response.json()
                if accounts:
                    return accounts[0]["securitiesAccount"]["accountId"]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting default account ID: {str(e)}")
            return None
    
    async def _build_order_payload(self, order: TradingOrder) -> Dict[str, Any]:
        """Build order payload for TD Ameritrade API"""
        payload = {
            "orderType": order.order_type.value,
            "session": "NORMAL",
            "duration": order.time_in_force.value,
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": order.side.value,
                    "quantity": float(order.quantity),
                    "instrument": {
                        "symbol": order.symbol,
                        "assetType": "EQUITY"  # Simplified - would need to determine asset type
                    }
                }
            ]
        }
        
        if order.price:
            payload["price"] = float(order.price)
        
        if order.stop_price:
            payload["stopPrice"] = float(order.stop_price)
        
        return payload
    
    async def _parse_order_response(self, order_data: Dict[str, Any]) -> OrderResult:
        """Parse order response from TD Ameritrade"""
        try:
            order_id = order_data.get("orderId", "")
            status_map = {
                "PENDING_ACTIVATION": OrderStatus.PENDING,
                "QUEUED": OrderStatus.PENDING,
                "WORKING": OrderStatus.PENDING,
                "FILLED": OrderStatus.FILLED,
                "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
                "CANCELED": OrderStatus.CANCELLED,
                "REJECTED": OrderStatus.REJECTED,
                "EXPIRED": OrderStatus.EXPIRED
            }
            
            status = status_map.get(order_data.get("status", ""), OrderStatus.PENDING)
            filled_quantity = Decimal(str(order_data.get("filledQuantity", 0)))
            remaining_quantity = Decimal(str(order_data.get("remainingQuantity", 0)))
            
            # Parse executions
            executions = []
            for execution_data in order_data.get("orderActivityCollection", []):
                if execution_data.get("activityType") == "EXECUTION":
                    for execution in execution_data.get("executionLegs", []):
                        exec_obj = OrderExecution(
                            execution_id=str(execution.get("legId", "")),
                            order_id=order_id,
                            symbol=execution.get("instrument", {}).get("symbol", ""),
                            side=OrderSide.BUY if execution.get("instruction") == "BUY" else OrderSide.SELL,
                            quantity=Decimal(str(execution.get("quantity", 0))),
                            price=Decimal(str(execution.get("price", 0))),
                            timestamp=datetime.now()  # Would parse from actual timestamp
                        )
                        executions.append(exec_obj)
            
            return OrderResult(
                order_id=order_id,
                broker_order_id=order_id,
                status=status,
                filled_quantity=filled_quantity,
                remaining_quantity=remaining_quantity,
                avg_fill_price=Decimal(str(order_data.get("price", 0))) if order_data.get("price") else None,
                total_commission=None,  # Would be calculated from fees
                total_fees=None,
                timestamp=datetime.now(),
                executions=executions
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing order response: {str(e)}")
            raise
    
    async def _parse_account_response(self, account_data: Dict[str, Any]) -> AccountInfo:
        """Parse account response from TD Ameritrade"""
        try:
            securities_account = account_data["securitiesAccount"]
            balances = securities_account.get("currentBalances", {})
            
            return AccountInfo(
                account_id=securities_account["accountId"],
                broker_type=BrokerType.TD_AMERITRADE,
                account_type=securities_account.get("type", "CASH"),
                buying_power=Decimal(str(balances.get("buyingPower", 0))),
                cash_balance=Decimal(str(balances.get("cashBalance", 0))),
                total_value=Decimal(str(balances.get("liquidationValue", 0))),
                day_trade_buying_power=Decimal(str(balances.get("dayTradingBuyingPower", 0))),
                pattern_day_trader=securities_account.get("isDayTrader", False),
                margin_enabled=securities_account.get("isClosingOnlyRestricted", False),
                options_enabled=True  # Would check actual options approval
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing account response: {str(e)}")
            raise
    
    async def _parse_positions_response(self, account_data: Dict[str, Any], account_id: str) -> List[Position]:
        """Parse positions response from TD Ameritrade"""
        try:
            positions = []
            securities_account = account_data["securitiesAccount"]
            
            for position_data in securities_account.get("positions", []):
                instrument = position_data.get("instrument", {})
                
                position = Position(
                    symbol=instrument.get("symbol", ""),
                    quantity=Decimal(str(position_data.get("longQuantity", 0))),
                    average_cost=Decimal(str(position_data.get("averagePrice", 0))),
                    market_value=Decimal(str(position_data.get("marketValue", 0))),
                    unrealized_pnl=Decimal(str(position_data.get("currentDayProfitLoss", 0))),
                    realized_pnl=Decimal('0'),  # Would need separate API call
                    account_id=account_id,
                    broker_type=BrokerType.TD_AMERITRADE
                )
                
                if position.quantity > 0:
                    positions.append(position)
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error parsing positions response: {str(e)}")
            return []
    
    async def _parse_trade_confirmation(self, transaction_data: Dict[str, Any], account_id: str) -> Optional[TradeConfirmation]:
        """Parse trade confirmation from transaction data"""
        try:
            if transaction_data.get("type") != "TRADE":
                return None
            
            trade_date = datetime.fromisoformat(transaction_data["transactionDate"].replace("Z", "+00:00"))
            settlement_date = datetime.fromisoformat(transaction_data["settlementDate"].replace("Z", "+00:00"))
            
            return TradeConfirmation(
                confirmation_id=str(transaction_data["transactionId"]),
                order_id="",  # Would need to map from transaction to order
                trade_date=trade_date,
                settlement_date=settlement_date.date(),
                symbol=transaction_data["transactionItem"]["instrument"]["symbol"],
                side=OrderSide.BUY if transaction_data["transactionItem"]["instruction"] == "BUY" else OrderSide.SELL,
                quantity=Decimal(str(abs(transaction_data["transactionItem"]["amount"]))),
                price=Decimal(str(transaction_data["transactionItem"]["price"])),
                gross_amount=Decimal(str(abs(transaction_data["netAmount"]))),
                commission=Decimal(str(transaction_data.get("fees", {}).get("commission", 0))),
                fees=Decimal(str(transaction_data.get("fees", {}).get("regulatoryFee", 0))),
                net_amount=Decimal(str(transaction_data["netAmount"])),
                account_id=account_id,
                broker_type=BrokerType.TD_AMERITRADE
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing trade confirmation: {str(e)}")
            return None