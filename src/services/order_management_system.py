"""Order Management System with smart routing and execution tracking"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
from dataclasses import dataclass, field
import uuid
from decimal import Decimal
import json

from ..exchanges.broker_base import (
    BrokerType, BrokerConnector, TradingOrder, OrderResult, OrderStatus,
    OrderSide, OrderType, TimeInForce, OrderExecution
)
from ..services.oauth_token_manager import OAuthTokenManager
from ..models.risk_management import RiskLimits


class RoutingStrategy(Enum):
    """Order routing strategies"""
    BEST_EXECUTION = "best_execution"
    LOWEST_COST = "lowest_cost"
    FASTEST_FILL = "fastest_fill"
    PREFERRED_BROKER = "preferred_broker"


class OrderPriority(Enum):
    """Order priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class RoutingRule:
    """Order routing rule configuration"""
    symbol_pattern: Optional[str] = None
    order_type: Optional[OrderType] = None
    min_quantity: Optional[Decimal] = None
    max_quantity: Optional[Decimal] = None
    preferred_brokers: List[BrokerType] = field(default_factory=list)
    excluded_brokers: List[BrokerType] = field(default_factory=list)
    strategy: RoutingStrategy = RoutingStrategy.BEST_EXECUTION
    priority: int = 0


@dataclass
class ExecutionMetrics:
    """Execution quality metrics"""
    broker_type: BrokerType
    avg_fill_time: float
    fill_rate: float
    avg_slippage: float
    commission_rate: float
    success_rate: float
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class OrderTracker:
    """Order tracking information"""
    order_id: str
    original_order: TradingOrder
    broker_type: BrokerType
    broker_order_id: Optional[str]
    status: OrderStatus
    filled_quantity: Decimal
    remaining_quantity: Decimal
    avg_fill_price: Optional[Decimal]
    executions: List[OrderExecution]
    created_at: datetime
    updated_at: datetime
    completion_time: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


class OrderManagementSystem:
    """Comprehensive order management system with smart routing"""
    
    def __init__(self, token_manager: OAuthTokenManager):
        self.token_manager = token_manager
        self.logger = logging.getLogger(__name__)
        
        # Broker connectors
        self.brokers: Dict[BrokerType, BrokerConnector] = {}
        
        # Order tracking
        self.active_orders: Dict[str, OrderTracker] = {}
        self.completed_orders: Dict[str, OrderTracker] = {}
        
        # Routing configuration
        self.routing_rules: List[RoutingRule] = []
        self.execution_metrics: Dict[BrokerType, ExecutionMetrics] = {}
        
        # Risk management
        self.risk_limits: Optional[RiskLimits] = None
        
        # Event callbacks
        self.order_callbacks: Dict[str, List[Callable]] = {
            "order_filled": [],
            "order_partially_filled": [],
            "order_cancelled": [],
            "order_rejected": [],
            "order_error": []
        }
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.order_timeout = timedelta(minutes=30)
        self.heartbeat_interval = 30  # seconds
        
    def register_broker(self, broker_type: BrokerType, connector: BrokerConnector):
        """Register a broker connector"""
        self.brokers[broker_type] = connector
        
        # Initialize execution metrics
        if broker_type not in self.execution_metrics:
            self.execution_metrics[broker_type] = ExecutionMetrics(
                broker_type=broker_type,
                avg_fill_time=0.0,
                fill_rate=0.0,
                avg_slippage=0.0,
                commission_rate=0.0,
                success_rate=0.0
            )
        
        self.logger.info(f"Registered broker: {broker_type.value}")
    
    def add_routing_rule(self, rule: RoutingRule):
        """Add order routing rule"""
        self.routing_rules.append(rule)
        # Sort by priority (higher priority first)
        self.routing_rules.sort(key=lambda r: r.priority, reverse=True)
        self.logger.info(f"Added routing rule with strategy: {rule.strategy.value}")
    
    def set_risk_limits(self, risk_limits: RiskLimits):
        """Set risk management limits"""
        self.risk_limits = risk_limits
    
    def register_callback(self, event: str, callback: Callable):
        """Register event callback"""
        if event in self.order_callbacks:
            self.order_callbacks[event].append(callback)
    
    async def start(self):
        """Start the order management system"""
        try:
            # Start monitoring tasks
            self._monitoring_task = asyncio.create_task(self._monitor_orders())
            self._metrics_task = asyncio.create_task(self._update_metrics())
            
            self.logger.info("Order Management System started")
            
        except Exception as e:
            self.logger.error(f"Error starting OMS: {str(e)}")
            raise
    
    async def stop(self):
        """Stop the order management system"""
        try:
            # Cancel monitoring tasks
            if self._monitoring_task:
                self._monitoring_task.cancel()
            if self._metrics_task:
                self._metrics_task.cancel()
            
            # Wait for tasks to complete
            tasks = [t for t in [self._monitoring_task, self._metrics_task] if t]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            self.logger.info("Order Management System stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping OMS: {str(e)}")
    
    async def place_order(self, order: TradingOrder, routing_strategy: Optional[RoutingStrategy] = None) -> str:
        """Place order with smart routing"""
        try:
            # Generate order ID if not provided
            if not order.order_id:
                order.order_id = str(uuid.uuid4())
            
            # Validate order
            validation_result = await self._validate_order(order)
            if not validation_result["valid"]:
                raise ValueError(f"Order validation failed: {validation_result['reason']}")
            
            # Determine routing
            broker_type = await self._route_order(order, routing_strategy)
            if not broker_type:
                raise ValueError("No suitable broker found for order routing")
            
            # Get broker connector
            broker = self.brokers.get(broker_type)
            if not broker:
                raise ValueError(f"Broker {broker_type.value} not available")
            
            # Create order tracker
            tracker = OrderTracker(
                order_id=order.order_id,
                original_order=order,
                broker_type=broker_type,
                broker_order_id=None,
                status=OrderStatus.PENDING,
                filled_quantity=Decimal('0'),
                remaining_quantity=order.quantity,
                avg_fill_price=None,
                executions=[],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.active_orders[order.order_id] = tracker
            
            # Submit order to broker
            result = await self._submit_order_to_broker(broker, order, tracker)
            
            if result.status == OrderStatus.REJECTED:
                # Move to completed orders
                tracker.status = OrderStatus.REJECTED
                tracker.error_message = result.message
                tracker.completion_time = datetime.now()
                self.completed_orders[order.order_id] = tracker
                del self.active_orders[order.order_id]
                
                # Trigger callback
                await self._trigger_callback("order_rejected", tracker)
            
            self.logger.info(f"Order {order.order_id} placed with {broker_type.value}")
            return order.order_id
            
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        try:
            if order_id not in self.active_orders:
                self.logger.warning(f"Order {order_id} not found in active orders")
                return False
            
            tracker = self.active_orders[order_id]
            broker = self.brokers.get(tracker.broker_type)
            
            if not broker:
                self.logger.error(f"Broker {tracker.broker_type.value} not available")
                return False
            
            # Cancel with broker
            success = await broker.cancel_order(order_id, tracker.broker_order_id)
            
            if success:
                tracker.status = OrderStatus.CANCELLED
                tracker.completion_time = datetime.now()
                tracker.updated_at = datetime.now()
                
                # Move to completed orders
                self.completed_orders[order_id] = tracker
                del self.active_orders[order_id]
                
                # Trigger callback
                await self._trigger_callback("order_cancelled", tracker)
                
                self.logger.info(f"Order {order_id} cancelled successfully")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[OrderTracker]:
        """Get current order status"""
        # Check active orders first
        if order_id in self.active_orders:
            return self.active_orders[order_id]
        
        # Check completed orders
        if order_id in self.completed_orders:
            return self.completed_orders[order_id]
        
        return None
    
    async def get_active_orders(self) -> List[OrderTracker]:
        """Get all active orders"""
        return list(self.active_orders.values())
    
    async def get_order_history(self, start_date: datetime, end_date: datetime) -> List[OrderTracker]:
        """Get order history for date range"""
        history = []
        
        for tracker in self.completed_orders.values():
            if start_date <= tracker.created_at <= end_date:
                history.append(tracker)
        
        # Sort by creation time
        history.sort(key=lambda t: t.created_at, reverse=True)
        return history
    
    async def _validate_order(self, order: TradingOrder) -> Dict[str, Any]:
        """Validate order before submission"""
        try:
            # Basic validation
            if order.quantity <= 0:
                return {"valid": False, "reason": "Invalid quantity"}
            
            if order.order_type == OrderType.LIMIT and not order.price:
                return {"valid": False, "reason": "Limit order requires price"}
            
            if order.order_type == OrderType.STOP and not order.stop_price:
                return {"valid": False, "reason": "Stop order requires stop price"}
            
            # Risk limits validation
            if self.risk_limits:
                # Check position limits
                if order.quantity > self.risk_limits.max_position_size:
                    return {"valid": False, "reason": "Exceeds maximum position size"}
                
                # Check order value limits
                if order.price and order.quantity * order.price > self.risk_limits.max_order_value:
                    return {"valid": False, "reason": "Exceeds maximum order value"}
            
            # Market hours validation
            if order.order_type == OrderType.MARKET:
                # Check if any broker supports this symbol during market hours
                market_open = any(
                    broker.is_market_hours() for broker in self.brokers.values()
                )
                if not market_open:
                    return {"valid": False, "reason": "Market is closed for market orders"}
            
            return {"valid": True, "reason": "Order validation passed"}
            
        except Exception as e:
            return {"valid": False, "reason": f"Validation error: {str(e)}"}
    
    async def _route_order(self, order: TradingOrder, strategy: Optional[RoutingStrategy] = None) -> Optional[BrokerType]:
        """Determine best broker for order routing"""
        try:
            # Apply routing rules
            applicable_rules = []
            for rule in self.routing_rules:
                if self._rule_matches_order(rule, order):
                    applicable_rules.append(rule)
            
            # Use strategy from rule or parameter
            routing_strategy = strategy
            if applicable_rules:
                routing_strategy = applicable_rules[0].strategy
            
            if not routing_strategy:
                routing_strategy = RoutingStrategy.BEST_EXECUTION
            
            # Get available brokers
            available_brokers = []
            for broker_type, broker in self.brokers.items():
                if broker.is_connected and broker.is_authenticated:
                    # Check if symbol is supported
                    if order.symbol in broker.get_supported_symbols():
                        # Check if order type is supported
                        if order.order_type in broker.get_supported_order_types():
                            available_brokers.append(broker_type)
            
            if not available_brokers:
                return None
            
            # Apply routing strategy
            if routing_strategy == RoutingStrategy.BEST_EXECUTION:
                return self._route_best_execution(available_brokers, order)
            elif routing_strategy == RoutingStrategy.LOWEST_COST:
                return self._route_lowest_cost(available_brokers, order)
            elif routing_strategy == RoutingStrategy.FASTEST_FILL:
                return self._route_fastest_fill(available_brokers, order)
            elif routing_strategy == RoutingStrategy.PREFERRED_BROKER:
                return self._route_preferred_broker(available_brokers, applicable_rules)
            
            # Default to first available broker
            return available_brokers[0]
            
        except Exception as e:
            self.logger.error(f"Error in order routing: {str(e)}")
            return None
    
    def _rule_matches_order(self, rule: RoutingRule, order: TradingOrder) -> bool:
        """Check if routing rule matches order"""
        # Symbol pattern matching
        if rule.symbol_pattern and rule.symbol_pattern not in order.symbol:
            return False
        
        # Order type matching
        if rule.order_type and rule.order_type != order.order_type:
            return False
        
        # Quantity range matching
        if rule.min_quantity and order.quantity < rule.min_quantity:
            return False
        
        if rule.max_quantity and order.quantity > rule.max_quantity:
            return False
        
        return True
    
    def _route_best_execution(self, brokers: List[BrokerType], order: TradingOrder) -> BrokerType:
        """Route based on best execution quality"""
        best_broker = brokers[0]
        best_score = 0
        
        for broker_type in brokers:
            metrics = self.execution_metrics.get(broker_type)
            if metrics:
                # Calculate composite score
                score = (
                    metrics.fill_rate * 0.3 +
                    metrics.success_rate * 0.3 +
                    (1 - metrics.avg_slippage) * 0.2 +
                    (1 - metrics.commission_rate) * 0.2
                )
                
                if score > best_score:
                    best_score = score
                    best_broker = broker_type
        
        return best_broker
    
    def _route_lowest_cost(self, brokers: List[BrokerType], order: TradingOrder) -> BrokerType:
        """Route based on lowest commission cost"""
        best_broker = brokers[0]
        lowest_cost = float('inf')
        
        for broker_type in brokers:
            metrics = self.execution_metrics.get(broker_type)
            if metrics and metrics.commission_rate < lowest_cost:
                lowest_cost = metrics.commission_rate
                best_broker = broker_type
        
        return best_broker
    
    def _route_fastest_fill(self, brokers: List[BrokerType], order: TradingOrder) -> BrokerType:
        """Route based on fastest fill time"""
        best_broker = brokers[0]
        fastest_time = float('inf')
        
        for broker_type in brokers:
            metrics = self.execution_metrics.get(broker_type)
            if metrics and metrics.avg_fill_time < fastest_time:
                fastest_time = metrics.avg_fill_time
                best_broker = broker_type
        
        return best_broker
    
    def _route_preferred_broker(self, brokers: List[BrokerType], rules: List[RoutingRule]) -> BrokerType:
        """Route based on preferred broker from rules"""
        for rule in rules:
            for preferred in rule.preferred_brokers:
                if preferred in brokers:
                    return preferred
        
        return brokers[0]
    
    async def _submit_order_to_broker(self, broker: BrokerConnector, order: TradingOrder, tracker: OrderTracker) -> OrderResult:
        """Submit order to specific broker"""
        try:
            result = await broker.place_order(order)
            
            # Update tracker
            tracker.broker_order_id = result.broker_order_id
            tracker.status = result.status
            tracker.filled_quantity = result.filled_quantity
            tracker.remaining_quantity = result.remaining_quantity
            tracker.avg_fill_price = result.avg_fill_price
            tracker.executions.extend(result.executions)
            tracker.updated_at = datetime.now()
            
            if result.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                tracker.completion_time = datetime.now()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error submitting order to broker: {str(e)}")
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
    
    async def _monitor_orders(self):
        """Background task to monitor active orders"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Check active orders
                orders_to_update = list(self.active_orders.keys())
                
                for order_id in orders_to_update:
                    if order_id not in self.active_orders:
                        continue
                    
                    tracker = self.active_orders[order_id]
                    broker = self.brokers.get(tracker.broker_type)
                    
                    if not broker:
                        continue
                    
                    try:
                        # Get updated status
                        result = await broker.get_order_status(order_id, tracker.broker_order_id)
                        
                        # Update tracker
                        old_status = tracker.status
                        tracker.status = result.status
                        tracker.filled_quantity = result.filled_quantity
                        tracker.remaining_quantity = result.remaining_quantity
                        tracker.avg_fill_price = result.avg_fill_price
                        tracker.executions.extend(result.executions)
                        tracker.updated_at = datetime.now()
                        
                        # Check for status changes
                        if old_status != result.status:
                            await self._handle_status_change(tracker, old_status)
                        
                        # Check for timeout
                        if datetime.now() - tracker.created_at > self.order_timeout:
                            await self.cancel_order(order_id)
                    
                    except Exception as e:
                        self.logger.error(f"Error monitoring order {order_id}: {str(e)}")
                        tracker.retry_count += 1
                        
                        if tracker.retry_count >= tracker.max_retries:
                            tracker.status = OrderStatus.REJECTED
                            tracker.error_message = f"Monitoring failed: {str(e)}"
                            tracker.completion_time = datetime.now()
                            
                            self.completed_orders[order_id] = tracker
                            del self.active_orders[order_id]
                            
                            await self._trigger_callback("order_error", tracker)
                
            except Exception as e:
                self.logger.error(f"Error in order monitoring: {str(e)}")
    
    async def _handle_status_change(self, tracker: OrderTracker, old_status: OrderStatus):
        """Handle order status changes"""
        try:
            if tracker.status == OrderStatus.FILLED:
                tracker.completion_time = datetime.now()
                self.completed_orders[tracker.order_id] = tracker
                del self.active_orders[tracker.order_id]
                await self._trigger_callback("order_filled", tracker)
                
            elif tracker.status == OrderStatus.PARTIALLY_FILLED:
                await self._trigger_callback("order_partially_filled", tracker)
                
            elif tracker.status == OrderStatus.CANCELLED:
                tracker.completion_time = datetime.now()
                self.completed_orders[tracker.order_id] = tracker
                del self.active_orders[tracker.order_id]
                await self._trigger_callback("order_cancelled", tracker)
                
            elif tracker.status == OrderStatus.REJECTED:
                tracker.completion_time = datetime.now()
                self.completed_orders[tracker.order_id] = tracker
                del self.active_orders[tracker.order_id]
                await self._trigger_callback("order_rejected", tracker)
            
        except Exception as e:
            self.logger.error(f"Error handling status change: {str(e)}")
    
    async def _trigger_callback(self, event: str, tracker: OrderTracker):
        """Trigger event callbacks"""
        try:
            callbacks = self.order_callbacks.get(event, [])
            for callback in callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(tracker)
                    else:
                        callback(tracker)
                except Exception as e:
                    self.logger.error(f"Error in callback for {event}: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error triggering callbacks: {str(e)}")
    
    async def _update_metrics(self):
        """Background task to update execution metrics"""
        while True:
            try:
                await asyncio.sleep(300)  # Update every 5 minutes
                
                # Calculate metrics for each broker
                for broker_type in self.brokers.keys():
                    await self._calculate_broker_metrics(broker_type)
                
            except Exception as e:
                self.logger.error(f"Error updating metrics: {str(e)}")
    
    async def _calculate_broker_metrics(self, broker_type: BrokerType):
        """Calculate execution metrics for a broker"""
        try:
            # Get recent completed orders for this broker
            recent_orders = [
                tracker for tracker in self.completed_orders.values()
                if tracker.broker_type == broker_type and
                tracker.completion_time and
                datetime.now() - tracker.completion_time <= timedelta(hours=24)
            ]
            
            if not recent_orders:
                return
            
            # Calculate metrics
            total_orders = len(recent_orders)
            filled_orders = [t for t in recent_orders if t.status == OrderStatus.FILLED]
            successful_orders = [t for t in recent_orders if t.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]]
            
            fill_rate = len(filled_orders) / total_orders if total_orders > 0 else 0
            success_rate = len(successful_orders) / total_orders if total_orders > 0 else 0
            
            # Calculate average fill time
            fill_times = []
            for tracker in filled_orders:
                if tracker.completion_time:
                    fill_time = (tracker.completion_time - tracker.created_at).total_seconds()
                    fill_times.append(fill_time)
            
            avg_fill_time = sum(fill_times) / len(fill_times) if fill_times else 0
            
            # Update metrics
            metrics = self.execution_metrics[broker_type]
            metrics.fill_rate = fill_rate
            metrics.success_rate = success_rate
            metrics.avg_fill_time = avg_fill_time
            metrics.last_updated = datetime.now()
            
            self.logger.debug(f"Updated metrics for {broker_type.value}: fill_rate={fill_rate:.2f}, success_rate={success_rate:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics for {broker_type.value}: {str(e)}")