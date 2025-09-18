"""Interactive Brokers connector for real money trading"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, AsyncGenerator
import httpx
import json
from decimal import Decimal
import hashlib
import hmac
import time

from .broker_base import (
    BrokerConnector, BrokerCredentials, BrokerType, TradingOrder, OrderResult, 
    OrderStatus, OrderSide, OrderType, TimeInForce, Position, AccountInfo, 
    TradeConfirmation, OrderExecution
)


class InteractiveBrokersConnector(BrokerConnector):
    """Interactive Brokers connector using Client Portal Web API"""
    
    BASE_URL = "https://localhost:5000/v1/api"  # IB Gateway/TWS API
    PAPER_BASE_URL = "https://localhost:5000/v1/api"  # Same for paper trading
    
    def __init__(self, credentials: BrokerCredentials, sandbox: bool = True):
        super().__init__(credentials, sandbox)
        self.logger = logging.getLogger(__name__)
        
        # HTTP client with SSL verification disabled for localhost
        self.client: Optional[httpx.AsyncClient] = None
        
        # Session management
        self.session_id: Optional[str] = None
        self.server_info: Dict[str, Any] = {}
        
        # Account info cache
        self.account_cache: Dict[str, AccountInfo] = {}
        self.positions_cache: Dict[str, List[Position]] = {}
        
        # Contract ID cache (symbol -> contract ID mapping)
        self.contract_cache: Dict[str, int] = {}
    
    async def connect(self) -> bool:
        """Establish connection to Interactive Brokers"""
        try:
            # Create HTTP client with SSL verification disabled for localhost
            timeout = httpx.Timeout(30.0)
            self.client = httpx.AsyncClient(
                timeout=timeout,
                verify=False  # IB Gateway uses self-signed certificates
            )
            
            # Check if IB Gateway is running
            response = await self.client.get(f"{self.BASE_URL}/portal/iserver/auth/status")
            
            if response.status_code == 200:
                status_data = response.json()
                if status_data.get("authenticated", False):
                    self.is_connected = True
                    self.is_authenticated = True
                    self.logger.info("Connected to Interactive Brokers Gateway")
                    return True
                else:
                    # Need to authenticate
                    return await self.authenticate()
            else:
                self.logger.error("IB Gateway not accessible. Please ensure IB Gateway or TWS is running.")
                return False
                
        except Exception as e:
            self.logger.error(f"Connection failed: {str(e)}")
            return False
    
    async def disconnect(self) -> None:
        """Close connection to Interactive Brokers"""
        try:
            if self.client and self.is_authenticated:
                # Logout
                await self.client.post(f"{self.BASE_URL}/portal/logout")
            
            if self.client:
                await self.client.aclose()
                self.client = None
            
            self.is_connected = False
            self.is_authenticated = False
            self.session_id = None
            self.logger.info("Disconnected from Interactive Brokers")
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {str(e)}")
    
    async def authenticate(self) -> bool:
        """Authenticate with Interactive Brokers"""
        try:
            # IB Gateway authentication is typically done through the desktop application
            # The web API relies on the Gateway being authenticated
            
            # Check authentication status
            response = await self.client.get(f"{self.BASE_URL}/portal/iserver/auth/status")
            
            if response.status_code == 200:
                status_data = response.json()
                
                if status_data.get("authenticated", False):
                    self.is_authenticated = True
                    
                    # Get server info
                    server_response = await self.client.get(f"{self.BASE_URL}/portal/iserver/accounts")
                    if server_response.status_code == 200:
                        self.server_info = server_response.json()
                    
                    self.logger.info("Authenticated with Interactive Brokers")
                    return True
                else:
                    # Try to re-authenticate
                    auth_response = await self.client.post(f"{self.BASE_URL}/portal/iserver/reauthenticate")
                    
                    if auth_response.status_code == 200:
                        self.is_authenticated = True
                        return True
                    else:
                        self.logger.error("Authentication failed. Please log in through IB Gateway/TWS.")
                        return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"Authentication failed: {str(e)}")
            return False
    
    async def refresh_token(self) -> bool:
        """Refresh authentication (IB uses session-based auth)"""
        try:
            response = await self.client.post(f"{self.BASE_URL}/portal/iserver/reauthenticate")
            
            if response.status_code == 200:
                self.logger.info("Session refreshed successfully")
                return True
            else:
                return await self.authenticate()
                
        except Exception as e:
            self.logger.error(f"Token refresh error: {str(e)}")
            return False
    
    async def is_token_valid(self) -> bool:
        """Check if current session is valid"""
        try:
            response = await self.client.get(f"{self.BASE_URL}/portal/iserver/auth/status")
            
            if response.status_code == 200:
                status_data = response.json()
                return status_data.get("authenticated", False)
            
            return False
            
        except Exception:
            return False
    
    async def place_order(self, order: TradingOrder) -> OrderResult:
        """Place a trading order"""
        try:
            # Get account ID
            account_id = order.account_id or await self._get_default_account_id()
            if not account_id:
                raise ValueError("No account ID available")
            
            # Get contract ID for symbol
            contract_id = await self._get_contract_id(order.symbol)
            if not contract_id:
                raise ValueError(f"Contract not found for symbol {order.symbol}")
            
            # Build order payload
            order_payload = await self._build_order_payload(order, contract_id)
            
            # Submit order
            response = await self.client.post(
                f"{self.BASE_URL}/portal/iserver/account/{account_id}/orders",
                json=order_payload
            )
            
            if response.status_code == 200:
                order_response = response.json()
                
                # IB may return confirmation dialog
                if isinstance(order_response, list) and order_response:
                    order_data = order_response[0]
                    
                    if "id" in order_data:
                        # Order placed successfully
                        return OrderResult(
                            order_id=order.order_id,
                            broker_order_id=str(order_data["id"]),
                            status=OrderStatus.PENDING,
                            filled_quantity=Decimal('0'),
                            remaining_quantity=order.quantity,
                            avg_fill_price=None,
                            total_commission=None,
                            total_fees=None,
                            timestamp=datetime.now(),
                            message="Order submitted successfully"
                        )
                    elif "messageIds" in order_data:
                        # Need to confirm order
                        message_ids = order_data["messageIds"]
                        confirm_response = await self.client.post(
                            f"{self.BASE_URL}/portal/iserver/reply/{message_ids[0]}",
                            json={"confirmed": True}
                        )
                        
                        if confirm_response.status_code == 200:
                            confirm_data = confirm_response.json()
                            if isinstance(confirm_data, list) and confirm_data and "id" in confirm_data[0]:
                                return OrderResult(
                                    order_id=order.order_id,
                                    broker_order_id=str(confirm_data[0]["id"]),
                                    status=OrderStatus.PENDING,
                                    filled_quantity=Decimal('0'),
                                    remaining_quantity=order.quantity,
                                    avg_fill_price=None,
                                    total_commission=None,
                                    total_fees=None,
                                    timestamp=datetime.now(),
                                    message="Order confirmed and submitted"
                                )
                
                # Order rejected or error
                error_msg = str(order_response)
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
                    message=f"Order submission failed: {error_msg}"
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
            
            response = await self.client.delete(
                f"{self.BASE_URL}/portal/iserver/account/{account_id}/order/{broker_order_id}"
            )
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Error cancelling order: {str(e)}")
            return False
    
    async def modify_order(self, order_id: str, modifications: Dict[str, Any]) -> OrderResult:
        """Modify an existing order"""
        try:
            # Get current order details
            current_order = await self.get_order_status(order_id, modifications.get("broker_order_id"))
            
            if current_order.status != OrderStatus.PENDING:
                raise ValueError("Can only modify pending orders")
            
            account_id = await self._get_default_account_id()
            if not account_id:
                raise ValueError("No account ID available")
            
            # Build modification payload
            modify_payload = {}
            
            if "quantity" in modifications:
                modify_payload["quantity"] = float(modifications["quantity"])
            
            if "price" in modifications:
                modify_payload["price"] = float(modifications["price"])
            
            if "stop_price" in modifications:
                modify_payload["auxPrice"] = float(modifications["stop_price"])
            
            response = await self.client.post(
                f"{self.BASE_URL}/portal/iserver/account/{account_id}/order/{current_order.broker_order_id}",
                json=modify_payload
            )
            
            if response.status_code == 200:
                return OrderResult(
                    order_id=order_id,
                    broker_order_id=current_order.broker_order_id,
                    status=OrderStatus.PENDING,
                    filled_quantity=current_order.filled_quantity,
                    remaining_quantity=Decimal(str(modify_payload.get("quantity", current_order.remaining_quantity))),
                    avg_fill_price=current_order.avg_fill_price,
                    total_commission=current_order.total_commission,
                    total_fees=current_order.total_fees,
                    timestamp=datetime.now(),
                    message="Order modified successfully"
                )
            else:
                raise RuntimeError(f"Order modification failed: {response.text}")
                
        except Exception as e:
            self.logger.error(f"Error modifying order: {str(e)}")
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
                message=str(e)
            )
    
    async def get_order_status(self, order_id: str, broker_order_id: Optional[str] = None) -> OrderResult:
        """Get current order status"""
        try:
            account_id = await self._get_default_account_id()
            if not account_id:
                raise ValueError("No account ID available")
            
            # Get all orders and find the matching one
            response = await self.client.get(f"{self.BASE_URL}/portal/iserver/account/orders")
            
            if response.status_code == 200:
                orders_data = response.json()
                
                for order_data in orders_data.get("orders", []):
                    if (broker_order_id and str(order_data.get("orderId")) == broker_order_id) or \
                       (order_data.get("clientOrderId") == order_id):
                        return await self._parse_order_response(order_data, order_id)
                
                # Order not found
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
                    message="Order not found"
                )
            else:
                raise RuntimeError(f"Failed to get order status: {response.status_code}")
                
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
            # IB API doesn't have direct date filtering for orders
            # Get all recent orders and filter
            response = await self.client.get(f"{self.BASE_URL}/portal/iserver/account/orders")
            
            if response.status_code == 200:
                orders_data = response.json()
                order_results = []
                
                for order_data in orders_data.get("orders", []):
                    order_result = await self._parse_order_response(order_data)
                    
                    # Filter by date (simplified - would need proper timestamp parsing)
                    if start_date <= order_result.timestamp <= end_date:
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
            
            # Get account summary
            response = await self.client.get(
                f"{self.BASE_URL}/portal/iserver/account/{account_id}/summary"
            )
            
            if response.status_code == 200:
                summary_data = response.json()
                account_info = await self._parse_account_response(summary_data, account_id)
                
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
                cache_time = getattr(self, f"_positions_cache_time_{account_id}", None)
                if cache_time and datetime.now() - cache_time < timedelta(minutes=1):
                    return self.positions_cache[account_id]
            
            response = await self.client.get(
                f"{self.BASE_URL}/portal/iserver/account/{account_id}/positions/0"
            )
            
            if response.status_code == 200:
                positions_data = response.json()
                positions = await self._parse_positions_response(positions_data, account_id)
                
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
            
            # Get trades from the specified date range
            response = await self.client.get(
                f"{self.BASE_URL}/portal/iserver/account/{account_id}/trades"
            )
            
            if response.status_code == 200:
                trades_data = response.json()
                confirmations = []
                
                for trade_data in trades_data:
                    confirmation = await self._parse_trade_confirmation(trade_data, account_id)
                    if confirmation and start_date <= confirmation.trade_date <= end_date:
                        confirmations.append(confirmation)
                
                return confirmations
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting trade confirmations: {str(e)}")
            return []
    
    async def stream_order_updates(self) -> AsyncGenerator[OrderResult, None]:
        """Stream real-time order updates"""
        # IB has WebSocket streaming, but this is a simplified polling implementation
        last_orders = {}
        
        while self.is_connected:
            try:
                orders = await self.get_order_history(
                    datetime.now() - timedelta(hours=1),
                    datetime.now()
                )
                
                for order in orders:
                    order_key = order.broker_order_id or order.order_id
                    
                    # Check if order status changed
                    if (order_key not in last_orders or 
                        last_orders[order_key].status != order.status or
                        last_orders[order_key].filled_quantity != order.filled_quantity):
                        
                        last_orders[order_key] = order
                        yield order
                
                await asyncio.sleep(10)  # Poll every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in order stream: {str(e)}")
                await asyncio.sleep(30)
    
    async def stream_position_updates(self) -> AsyncGenerator[Position, None]:
        """Stream real-time position updates"""
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
                
                await asyncio.sleep(30)  # Poll every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in position stream: {str(e)}")
                await asyncio.sleep(60)
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols"""
        # IB supports a vast range of instruments globally
        return ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "SPY", "QQQ", "IWM", "EUR.USD", "GBP.USD"]
    
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
            
            # Check if contract exists
            contract_id = await self._get_contract_id(order.symbol)
            if not contract_id:
                return {"valid": False, "reason": f"Contract not found for symbol {order.symbol}"}
            
            return {"valid": True, "reason": "Order validation passed"}
            
        except Exception as e:
            return {"valid": False, "reason": f"Validation error: {str(e)}"}
    
    async def _get_default_account_id(self) -> Optional[str]:
        """Get default account ID"""
        try:
            response = await self.client.get(f"{self.BASE_URL}/portal/iserver/accounts")
            
            if response.status_code == 200:
                accounts_data = response.json()
                if accounts_data and "accounts" in accounts_data:
                    return accounts_data["accounts"][0]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting default account ID: {str(e)}")
            return None
    
    async def _get_contract_id(self, symbol: str) -> Optional[int]:
        """Get contract ID for symbol"""
        try:
            # Check cache first
            if symbol in self.contract_cache:
                return self.contract_cache[symbol]
            
            # Search for contract
            response = await self.client.get(
                f"{self.BASE_URL}/portal/iserver/secdef/search",
                params={"symbol": symbol}
            )
            
            if response.status_code == 200:
                search_results = response.json()
                
                if search_results:
                    # Take the first result (usually the primary exchange)
                    contract_id = search_results[0].get("conid")
                    if contract_id:
                        self.contract_cache[symbol] = contract_id
                        return contract_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting contract ID for {symbol}: {str(e)}")
            return None
    
    async def _build_order_payload(self, order: TradingOrder, contract_id: int) -> Dict[str, Any]:
        """Build order payload for IB API"""
        payload = {
            "conid": contract_id,
            "orderType": order.order_type.value,
            "side": "BUY" if order.side == OrderSide.BUY else "SELL",
            "quantity": float(order.quantity),
            "tif": order.time_in_force.value
        }
        
        if order.price:
            payload["price"] = float(order.price)
        
        if order.stop_price:
            payload["auxPrice"] = float(order.stop_price)
        
        return payload
    
    async def _parse_order_response(self, order_data: Dict[str, Any], order_id: str = None) -> OrderResult:
        """Parse order response from IB"""
        try:
            broker_order_id = str(order_data.get("orderId", ""))
            
            status_map = {
                "PendingSubmit": OrderStatus.PENDING,
                "Submitted": OrderStatus.PENDING,
                "Filled": OrderStatus.FILLED,
                "PartiallyFilled": OrderStatus.PARTIALLY_FILLED,
                "Cancelled": OrderStatus.CANCELLED,
                "Inactive": OrderStatus.REJECTED
            }
            
            status = status_map.get(order_data.get("status", ""), OrderStatus.PENDING)
            filled_quantity = Decimal(str(order_data.get("filledQuantity", 0)))
            total_quantity = Decimal(str(order_data.get("totalSize", 0)))
            remaining_quantity = total_quantity - filled_quantity
            
            return OrderResult(
                order_id=order_id or broker_order_id,
                broker_order_id=broker_order_id,
                status=status,
                filled_quantity=filled_quantity,
                remaining_quantity=remaining_quantity,
                avg_fill_price=Decimal(str(order_data.get("avgPrice", 0))) if order_data.get("avgPrice") else None,
                total_commission=Decimal(str(order_data.get("commission", 0))) if order_data.get("commission") else None,
                total_fees=None,
                timestamp=datetime.now(),  # Would parse from actual timestamp
                executions=[]
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing order response: {str(e)}")
            raise
    
    async def _parse_account_response(self, summary_data: Dict[str, Any], account_id: str) -> AccountInfo:
        """Parse account response from IB"""
        try:
            # IB returns account summary as key-value pairs
            summary_dict = {}
            for item in summary_data:
                summary_dict[item.get("key", "")] = item.get("value", "")
            
            return AccountInfo(
                account_id=account_id,
                broker_type=BrokerType.INTERACTIVE_BROKERS,
                account_type=summary_dict.get("AccountType", "CASH"),
                buying_power=Decimal(str(summary_dict.get("BuyingPower", 0))),
                cash_balance=Decimal(str(summary_dict.get("TotalCashValue", 0))),
                total_value=Decimal(str(summary_dict.get("NetLiquidation", 0))),
                day_trade_buying_power=Decimal(str(summary_dict.get("DayTradesRemaining", 0))),
                pattern_day_trader=False,  # Would need to check specific flag
                margin_enabled=summary_dict.get("AccountType", "") == "MARGIN",
                options_enabled=True  # Would check trading permissions
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing account response: {str(e)}")
            raise
    
    async def _parse_positions_response(self, positions_data: List[Dict[str, Any]], account_id: str) -> List[Position]:
        """Parse positions response from IB"""
        try:
            positions = []
            
            for position_data in positions_data:
                if float(position_data.get("position", 0)) != 0:
                    position = Position(
                        symbol=position_data.get("ticker", ""),
                        quantity=Decimal(str(position_data.get("position", 0))),
                        average_cost=Decimal(str(position_data.get("avgCost", 0))),
                        market_value=Decimal(str(position_data.get("mktValue", 0))),
                        unrealized_pnl=Decimal(str(position_data.get("unrealizedPnl", 0))),
                        realized_pnl=Decimal(str(position_data.get("realizedPnl", 0))),
                        account_id=account_id,
                        broker_type=BrokerType.INTERACTIVE_BROKERS
                    )
                    positions.append(position)
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error parsing positions response: {str(e)}")
            return []
    
    async def _parse_trade_confirmation(self, trade_data: Dict[str, Any], account_id: str) -> Optional[TradeConfirmation]:
        """Parse trade confirmation from IB trade data"""
        try:
            trade_date = datetime.fromtimestamp(trade_data.get("trade_time", 0) / 1000)
            
            return TradeConfirmation(
                confirmation_id=str(trade_data.get("execution_id", "")),
                order_id=str(trade_data.get("order_id", "")),
                trade_date=trade_date,
                settlement_date=(trade_date + timedelta(days=2)).date(),  # T+2 settlement
                symbol=trade_data.get("symbol", ""),
                side=OrderSide.BUY if trade_data.get("side") == "BUY" else OrderSide.SELL,
                quantity=Decimal(str(abs(trade_data.get("size", 0)))),
                price=Decimal(str(trade_data.get("price", 0))),
                gross_amount=Decimal(str(abs(trade_data.get("size", 0)) * trade_data.get("price", 0))),
                commission=Decimal(str(abs(trade_data.get("commission", 0)))),
                fees=Decimal('0'),  # IB includes fees in commission
                net_amount=Decimal(str(trade_data.get("net_amount", 0))),
                account_id=account_id,
                broker_type=BrokerType.INTERACTIVE_BROKERS
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing trade confirmation: {str(e)}")
            return None