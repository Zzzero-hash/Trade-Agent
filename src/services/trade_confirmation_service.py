"""Trade confirmation and settlement tracking service"""

import asyncio
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
import json

from ..exchanges.broker_base import (
    BrokerType, BrokerConnector, TradeConfirmation, OrderResult, OrderStatus
)
from ..services.oauth_token_manager import OAuthTokenManager


class SettlementStatus(Enum):
    """Settlement status enumeration"""
    PENDING = "pending"
    SETTLED = "settled"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ConfirmationStatus(Enum):
    """Trade confirmation status"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    DISPUTED = "disputed"
    CORRECTED = "corrected"


@dataclass
class SettlementInfo:
    """Settlement information"""
    trade_date: date
    settlement_date: date
    expected_settlement_date: date
    status: SettlementStatus
    settlement_amount: Decimal
    fees_paid: Decimal
    commission_paid: Decimal
    net_amount: Decimal
    settlement_reference: Optional[str] = None
    failure_reason: Optional[str] = None


@dataclass
class TradeRecord:
    """Comprehensive trade record"""
    trade_id: str
    order_id: str
    confirmation_id: Optional[str]
    broker_type: BrokerType
    account_id: str
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    trade_date: datetime
    settlement_info: SettlementInfo
    confirmation_status: ConfirmationStatus
    broker_confirmation: Optional[TradeConfirmation] = None
    discrepancies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class SettlementMetrics:
    """Settlement tracking metrics"""
    broker_type: BrokerType
    total_trades: int
    settled_trades: int
    pending_trades: int
    failed_trades: int
    avg_settlement_time: float
    settlement_success_rate: float
    total_settlement_value: Decimal
    last_updated: datetime = field(default_factory=datetime.now)


class TradeConfirmationService:
    """Trade confirmation and settlement tracking service"""
    
    def __init__(self, token_manager: OAuthTokenManager):
        self.token_manager = token_manager
        self.logger = logging.getLogger(__name__)
        
        # Broker connectors
        self.brokers: Dict[BrokerType, BrokerConnector] = {}
        
        # Trade tracking
        self.trade_records: Dict[str, TradeRecord] = {}  # trade_id -> TradeRecord
        self.pending_confirmations: Dict[str, TradeRecord] = {}  # confirmation_id -> TradeRecord
        self.settlement_queue: List[TradeRecord] = []
        
        # Metrics
        self.settlement_metrics: Dict[BrokerType, SettlementMetrics] = {}
        
        # Configuration
        self.confirmation_timeout = timedelta(hours=24)
        self.settlement_check_interval = 3600  # 1 hour
        self.max_settlement_days = 3  # T+3 settlement
        
        # Background tasks
        self._confirmation_tasks: Dict[BrokerType, asyncio.Task] = {}
        self._settlement_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        
        # Event callbacks
        self.callbacks = {
            "trade_confirmed": [],
            "settlement_completed": [],
            "settlement_failed": [],
            "discrepancy_detected": []
        }
    
    def register_broker(self, broker_type: BrokerType, connector: BrokerConnector):
        """Register a broker connector"""
        self.brokers[broker_type] = connector
        
        # Initialize metrics
        self.settlement_metrics[broker_type] = SettlementMetrics(
            broker_type=broker_type,
            total_trades=0,
            settled_trades=0,
            pending_trades=0,
            failed_trades=0,
            avg_settlement_time=0.0,
            settlement_success_rate=1.0,
            total_settlement_value=Decimal('0')
        )
        
        self.logger.info(f"Registered broker for trade confirmation: {broker_type.value}")
    
    def register_callback(self, event: str, callback):
        """Register event callback"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    async def start(self):
        """Start trade confirmation service"""
        try:
            # Start confirmation tasks for each broker
            for broker_type in self.brokers.keys():
                task = asyncio.create_task(self._confirmation_loop(broker_type))
                self._confirmation_tasks[broker_type] = task
            
            # Start settlement tracking task
            self._settlement_task = asyncio.create_task(self._settlement_loop())
            
            # Start metrics update task
            self._metrics_task = asyncio.create_task(self._metrics_loop())
            
            self.logger.info("Trade confirmation service started")
            
        except Exception as e:
            self.logger.error(f"Error starting trade confirmation service: {str(e)}")
            raise
    
    async def stop(self):
        """Stop trade confirmation service"""
        try:
            # Cancel all tasks
            for task in self._confirmation_tasks.values():
                task.cancel()
            
            if self._settlement_task:
                self._settlement_task.cancel()
            
            if self._metrics_task:
                self._metrics_task.cancel()
            
            # Wait for tasks to complete
            all_tasks = list(self._confirmation_tasks.values())
            if self._settlement_task:
                all_tasks.append(self._settlement_task)
            if self._metrics_task:
                all_tasks.append(self._metrics_task)
            
            if all_tasks:
                await asyncio.gather(*all_tasks, return_exceptions=True)
            
            self.logger.info("Trade confirmation service stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping trade confirmation service: {str(e)}")
    
    async def record_trade(self, order_result: OrderResult, account_id: str, broker_type: BrokerType) -> str:
        """Record a trade for confirmation tracking"""
        try:
            # Generate trade ID
            trade_id = f"{order_result.order_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Calculate settlement date (T+2 for stocks, T+1 for options, etc.)
            trade_date = order_result.timestamp.date()
            settlement_days = self._get_settlement_days(broker_type)
            expected_settlement_date = self._calculate_settlement_date(trade_date, settlement_days)
            
            # Create settlement info
            settlement_info = SettlementInfo(
                trade_date=trade_date,
                settlement_date=expected_settlement_date,
                expected_settlement_date=expected_settlement_date,
                status=SettlementStatus.PENDING,
                settlement_amount=order_result.filled_quantity * (order_result.avg_fill_price or Decimal('0')),
                fees_paid=order_result.total_fees or Decimal('0'),
                commission_paid=order_result.total_commission or Decimal('0'),
                net_amount=Decimal('0')  # Will be calculated
            )
            
            # Calculate net amount
            gross_amount = settlement_info.settlement_amount
            if order_result.order_id.startswith('SELL'):  # Simplified check
                settlement_info.net_amount = gross_amount - settlement_info.fees_paid - settlement_info.commission_paid
            else:
                settlement_info.net_amount = -(gross_amount + settlement_info.fees_paid + settlement_info.commission_paid)
            
            # Create trade record
            trade_record = TradeRecord(
                trade_id=trade_id,
                order_id=order_result.order_id,
                confirmation_id=None,
                broker_type=broker_type,
                account_id=account_id,
                symbol="",  # Will be filled from order details
                side="",    # Will be filled from order details
                quantity=order_result.filled_quantity,
                price=order_result.avg_fill_price or Decimal('0'),
                trade_date=order_result.timestamp,
                settlement_info=settlement_info,
                confirmation_status=ConfirmationStatus.PENDING
            )
            
            # Store trade record
            self.trade_records[trade_id] = trade_record
            self.settlement_queue.append(trade_record)
            
            # Update metrics
            metrics = self.settlement_metrics[broker_type]
            metrics.total_trades += 1
            metrics.pending_trades += 1
            metrics.total_settlement_value += abs(settlement_info.settlement_amount)
            
            self.logger.info(f"Recorded trade {trade_id} for confirmation tracking")
            return trade_id
            
        except Exception as e:
            self.logger.error(f"Error recording trade: {str(e)}")
            raise
    
    async def get_trade_record(self, trade_id: str) -> Optional[TradeRecord]:
        """Get trade record by ID"""
        return self.trade_records.get(trade_id)
    
    async def get_pending_settlements(self, broker_type: Optional[BrokerType] = None) -> List[TradeRecord]:
        """Get trades pending settlement"""
        pending = [
            record for record in self.trade_records.values()
            if record.settlement_info.status == SettlementStatus.PENDING
        ]
        
        if broker_type:
            pending = [record for record in pending if record.broker_type == broker_type]
        
        return pending
    
    async def get_settlement_metrics(self, broker_type: Optional[BrokerType] = None) -> Dict[BrokerType, SettlementMetrics]:
        """Get settlement metrics"""
        if broker_type:
            return {broker_type: self.settlement_metrics.get(broker_type)}
        return self.settlement_metrics.copy()
    
    async def force_confirmation_check(self, broker_type: BrokerType) -> int:
        """Force immediate confirmation check for a broker"""
        return await self._check_confirmations(broker_type)
    
    async def _confirmation_loop(self, broker_type: BrokerType):
        """Background loop for checking trade confirmations"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                await self._check_confirmations(broker_type)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in confirmation loop for {broker_type.value}: {str(e)}")
    
    async def _check_confirmations(self, broker_type: BrokerType) -> int:
        """Check for trade confirmations from broker"""
        try:
            broker = self.brokers.get(broker_type)
            if not broker or not broker.is_connected:
                return 0
            
            # Get pending trades for this broker
            pending_trades = [
                record for record in self.trade_records.values()
                if (record.broker_type == broker_type and 
                    record.confirmation_status == ConfirmationStatus.PENDING)
            ]
            
            if not pending_trades:
                return 0
            
            # Get date range for confirmation check
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)  # Check last 7 days
            
            # Get confirmations from broker
            confirmations = await broker.get_trade_confirmations(start_date, end_date)
            
            matched_count = 0
            
            for confirmation in confirmations:
                # Try to match confirmation with pending trade
                matched_trade = await self._match_confirmation(confirmation, pending_trades)
                
                if matched_trade:
                    await self._process_confirmation(matched_trade, confirmation)
                    matched_count += 1
            
            self.logger.debug(f"Processed {matched_count} confirmations for {broker_type.value}")
            return matched_count
            
        except Exception as e:
            self.logger.error(f"Error checking confirmations for {broker_type.value}: {str(e)}")
            return 0
    
    async def _match_confirmation(self, confirmation: TradeConfirmation, pending_trades: List[TradeRecord]) -> Optional[TradeRecord]:
        """Match trade confirmation with pending trade"""
        try:
            for trade in pending_trades:
                # Match by order ID
                if trade.order_id == confirmation.order_id:
                    return trade
                
                # Match by trade details (symbol, quantity, price, date)
                if (trade.symbol == confirmation.symbol and
                    trade.quantity == confirmation.quantity and
                    abs(trade.price - confirmation.price) < Decimal('0.01') and
                    trade.trade_date.date() == confirmation.trade_date.date()):
                    return trade
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error matching confirmation: {str(e)}")
            return None
    
    async def _process_confirmation(self, trade_record: TradeRecord, confirmation: TradeConfirmation):
        """Process matched trade confirmation"""
        try:
            # Update trade record
            trade_record.confirmation_id = confirmation.confirmation_id
            trade_record.broker_confirmation = confirmation
            trade_record.confirmation_status = ConfirmationStatus.CONFIRMED
            trade_record.updated_at = datetime.now()
            
            # Update settlement info
            settlement_info = trade_record.settlement_info
            settlement_info.settlement_date = confirmation.settlement_date
            settlement_info.settlement_amount = confirmation.gross_amount
            settlement_info.fees_paid = confirmation.fees
            settlement_info.commission_paid = confirmation.commission
            settlement_info.net_amount = confirmation.net_amount
            
            # Check for discrepancies
            discrepancies = await self._check_discrepancies(trade_record, confirmation)
            if discrepancies:
                trade_record.discrepancies = discrepancies
                trade_record.confirmation_status = ConfirmationStatus.DISPUTED
                
                await self._trigger_callback("discrepancy_detected", {
                    "trade_record": trade_record,
                    "discrepancies": discrepancies
                })
            
            # Trigger confirmation callback
            await self._trigger_callback("trade_confirmed", trade_record)
            
            self.logger.info(f"Processed confirmation for trade {trade_record.trade_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing confirmation: {str(e)}")
    
    async def _check_discrepancies(self, trade_record: TradeRecord, confirmation: TradeConfirmation) -> List[str]:
        """Check for discrepancies between trade record and confirmation"""
        discrepancies = []
        
        try:
            # Check quantity
            if trade_record.quantity != confirmation.quantity:
                discrepancies.append(f"Quantity mismatch: expected {trade_record.quantity}, got {confirmation.quantity}")
            
            # Check price (allow small tolerance)
            price_diff = abs(trade_record.price - confirmation.price)
            if price_diff > Decimal('0.01'):
                discrepancies.append(f"Price mismatch: expected {trade_record.price}, got {confirmation.price}")
            
            # Check settlement date
            expected_date = trade_record.settlement_info.expected_settlement_date
            if confirmation.settlement_date != expected_date:
                discrepancies.append(f"Settlement date mismatch: expected {expected_date}, got {confirmation.settlement_date}")
            
            # Check fees and commissions (allow 10% tolerance)
            if trade_record.settlement_info.commission_paid > 0:
                commission_diff = abs(trade_record.settlement_info.commission_paid - confirmation.commission)
                commission_tolerance = trade_record.settlement_info.commission_paid * Decimal('0.1')
                if commission_diff > commission_tolerance:
                    discrepancies.append(f"Commission mismatch: expected {trade_record.settlement_info.commission_paid}, got {confirmation.commission}")
            
            return discrepancies
            
        except Exception as e:
            self.logger.error(f"Error checking discrepancies: {str(e)}")
            return [f"Error checking discrepancies: {str(e)}"]
    
    async def _settlement_loop(self):
        """Background loop for settlement tracking"""
        while True:
            try:
                await asyncio.sleep(self.settlement_check_interval)
                await self._check_settlements()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in settlement loop: {str(e)}")
    
    async def _check_settlements(self):
        """Check settlement status for pending trades"""
        try:
            today = date.today()
            settled_count = 0
            
            # Check trades that should have settled
            for trade_record in list(self.settlement_queue):
                settlement_info = trade_record.settlement_info
                
                if settlement_info.status != SettlementStatus.PENDING:
                    continue
                
                # Check if settlement date has passed
                if today >= settlement_info.settlement_date:
                    # Mark as settled (in real implementation, would verify with broker)
                    settlement_info.status = SettlementStatus.SETTLED
                    trade_record.updated_at = datetime.now()
                    
                    # Remove from settlement queue
                    self.settlement_queue.remove(trade_record)
                    
                    # Update metrics
                    metrics = self.settlement_metrics[trade_record.broker_type]
                    metrics.settled_trades += 1
                    metrics.pending_trades -= 1
                    
                    # Calculate settlement time
                    settlement_time = (today - settlement_info.trade_date).days
                    if metrics.avg_settlement_time == 0:
                        metrics.avg_settlement_time = settlement_time
                    else:
                        metrics.avg_settlement_time = (metrics.avg_settlement_time * 0.9) + (settlement_time * 0.1)
                    
                    settled_count += 1
                    
                    # Trigger callback
                    await self._trigger_callback("settlement_completed", trade_record)
                    
                    self.logger.info(f"Trade {trade_record.trade_id} settled")
                
                # Check for overdue settlements
                elif today > settlement_info.settlement_date + timedelta(days=2):
                    settlement_info.status = SettlementStatus.FAILED
                    settlement_info.failure_reason = "Settlement overdue"
                    trade_record.updated_at = datetime.now()
                    
                    # Remove from settlement queue
                    self.settlement_queue.remove(trade_record)
                    
                    # Update metrics
                    metrics = self.settlement_metrics[trade_record.broker_type]
                    metrics.failed_trades += 1
                    metrics.pending_trades -= 1
                    
                    # Trigger callback
                    await self._trigger_callback("settlement_failed", trade_record)
                    
                    self.logger.warning(f"Trade {trade_record.trade_id} settlement failed - overdue")
            
            if settled_count > 0:
                self.logger.info(f"Processed {settled_count} settlements")
                
        except Exception as e:
            self.logger.error(f"Error checking settlements: {str(e)}")
    
    async def _metrics_loop(self):
        """Background loop for updating metrics"""
        while True:
            try:
                await asyncio.sleep(3600)  # Update every hour
                await self._update_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics loop: {str(e)}")
    
    async def _update_metrics(self):
        """Update settlement metrics"""
        try:
            for broker_type, metrics in self.settlement_metrics.items():
                # Calculate success rate
                total_completed = metrics.settled_trades + metrics.failed_trades
                if total_completed > 0:
                    metrics.settlement_success_rate = metrics.settled_trades / total_completed
                
                metrics.last_updated = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")
    
    def _get_settlement_days(self, broker_type: BrokerType) -> int:
        """Get settlement days for broker type"""
        # Standard settlement periods
        settlement_days = {
            BrokerType.ROBINHOOD: 2,  # T+2 for stocks
            BrokerType.TD_AMERITRADE: 2,
            BrokerType.INTERACTIVE_BROKERS: 2
        }
        
        return settlement_days.get(broker_type, 2)
    
    def _calculate_settlement_date(self, trade_date: date, settlement_days: int) -> date:
        """Calculate settlement date excluding weekends"""
        settlement_date = trade_date
        days_added = 0
        
        while days_added < settlement_days:
            settlement_date += timedelta(days=1)
            # Skip weekends
            if settlement_date.weekday() < 5:  # Monday = 0, Friday = 4
                days_added += 1
        
        return settlement_date
    
    async def _trigger_callback(self, event: str, data: Any):
        """Trigger event callbacks"""
        try:
            callbacks = self.callbacks.get(event, [])
            for callback in callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    self.logger.error(f"Error in callback for {event}: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error triggering callbacks: {str(e)}")
    
    async def export_trade_records(self, start_date: date, end_date: date, format: str = "json") -> str:
        """Export trade records for reporting"""
        try:
            records = []
            
            for trade_record in self.trade_records.values():
                if start_date <= trade_record.trade_date.date() <= end_date:
                    record_data = {
                        "trade_id": trade_record.trade_id,
                        "order_id": trade_record.order_id,
                        "confirmation_id": trade_record.confirmation_id,
                        "broker_type": trade_record.broker_type.value,
                        "account_id": trade_record.account_id,
                        "symbol": trade_record.symbol,
                        "side": trade_record.side,
                        "quantity": str(trade_record.quantity),
                        "price": str(trade_record.price),
                        "trade_date": trade_record.trade_date.isoformat(),
                        "settlement_status": trade_record.settlement_info.status.value,
                        "settlement_date": trade_record.settlement_info.settlement_date.isoformat(),
                        "net_amount": str(trade_record.settlement_info.net_amount),
                        "confirmation_status": trade_record.confirmation_status.value,
                        "discrepancies": trade_record.discrepancies
                    }
                    records.append(record_data)
            
            if format == "json":
                return json.dumps(records, indent=2)
            else:
                raise ValueError("Only JSON format is currently supported")
                
        except Exception as e:
            self.logger.error(f"Error exporting trade records: {str(e)}")
            raise