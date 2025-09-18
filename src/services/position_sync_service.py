"""Real-time position synchronization and reconciliation service"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
import json

from ..exchanges.broker_base import BrokerType, BrokerConnector, Position
from ..services.oauth_token_manager import OAuthTokenManager


class ReconciliationStatus(Enum):
    """Position reconciliation status"""
    MATCHED = "matched"
    DISCREPANCY = "discrepancy"
    MISSING_LOCAL = "missing_local"
    MISSING_BROKER = "missing_broker"
    ERROR = "error"


@dataclass
class PositionDiscrepancy:
    """Position discrepancy details"""
    symbol: str
    account_id: str
    broker_type: BrokerType
    local_position: Optional[Position]
    broker_position: Optional[Position]
    quantity_diff: Decimal
    value_diff: Decimal
    status: ReconciliationStatus
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None


@dataclass
class SyncMetrics:
    """Position synchronization metrics"""
    broker_type: BrokerType
    last_sync_time: datetime
    sync_duration: float
    positions_synced: int
    discrepancies_found: int
    sync_success_rate: float
    avg_sync_time: float
    error_count: int = 0


class PositionSyncService:
    """Real-time position synchronization and reconciliation service"""
    
    def __init__(self, token_manager: OAuthTokenManager):
        self.token_manager = token_manager
        self.logger = logging.getLogger(__name__)
        
        # Broker connectors
        self.brokers: Dict[BrokerType, BrokerConnector] = {}
        
        # Position storage
        self.local_positions: Dict[str, Dict[str, Position]] = {}  # account_id -> symbol -> position
        self.broker_positions: Dict[str, Dict[str, Position]] = {}  # account_id -> symbol -> position
        
        # Reconciliation tracking
        self.discrepancies: Dict[str, PositionDiscrepancy] = {}
        self.sync_metrics: Dict[BrokerType, SyncMetrics] = {}
        
        # Configuration
        self.sync_interval = 60  # seconds
        self.reconciliation_interval = 300  # seconds (5 minutes)
        self.discrepancy_threshold = Decimal('0.01')  # Minimum discrepancy to report
        
        # Background tasks
        self._sync_tasks: Dict[BrokerType, asyncio.Task] = {}
        self._reconciliation_task: Optional[asyncio.Task] = None
        
        # Event callbacks
        self.position_callbacks = {
            "position_updated": [],
            "discrepancy_detected": [],
            "discrepancy_resolved": [],
            "sync_error": []
        }
        
        # Account mappings
        self.account_mappings: Dict[str, Dict[BrokerType, str]] = {}  # local_account -> broker_type -> broker_account
    
    def register_broker(self, broker_type: BrokerType, connector: BrokerConnector):
        """Register a broker connector"""
        self.brokers[broker_type] = connector
        
        # Initialize sync metrics
        self.sync_metrics[broker_type] = SyncMetrics(
            broker_type=broker_type,
            last_sync_time=datetime.now(),
            sync_duration=0.0,
            positions_synced=0,
            discrepancies_found=0,
            sync_success_rate=1.0,
            avg_sync_time=0.0
        )
        
        self.logger.info(f"Registered broker for position sync: {broker_type.value}")
    
    def map_account(self, local_account_id: str, broker_type: BrokerType, broker_account_id: str):
        """Map local account ID to broker account ID"""
        if local_account_id not in self.account_mappings:
            self.account_mappings[local_account_id] = {}
        
        self.account_mappings[local_account_id][broker_type] = broker_account_id
        self.logger.info(f"Mapped account {local_account_id} to {broker_type.value} account {broker_account_id}")
    
    def register_callback(self, event: str, callback):
        """Register event callback"""
        if event in self.position_callbacks:
            self.position_callbacks[event].append(callback)
    
    async def start(self):
        """Start position synchronization service"""
        try:
            # Start sync tasks for each broker
            for broker_type in self.brokers.keys():
                task = asyncio.create_task(self._sync_positions_loop(broker_type))
                self._sync_tasks[broker_type] = task
            
            # Start reconciliation task
            self._reconciliation_task = asyncio.create_task(self._reconciliation_loop())
            
            self.logger.info("Position synchronization service started")
            
        except Exception as e:
            self.logger.error(f"Error starting position sync service: {str(e)}")
            raise
    
    async def stop(self):
        """Stop position synchronization service"""
        try:
            # Cancel sync tasks
            for task in self._sync_tasks.values():
                task.cancel()
            
            if self._reconciliation_task:
                self._reconciliation_task.cancel()
            
            # Wait for tasks to complete
            all_tasks = list(self._sync_tasks.values())
            if self._reconciliation_task:
                all_tasks.append(self._reconciliation_task)
            
            if all_tasks:
                await asyncio.gather(*all_tasks, return_exceptions=True)
            
            self.logger.info("Position synchronization service stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping position sync service: {str(e)}")
    
    async def sync_positions_now(self, broker_type: Optional[BrokerType] = None) -> Dict[BrokerType, bool]:
        """Trigger immediate position sync"""
        results = {}
        
        brokers_to_sync = [broker_type] if broker_type else list(self.brokers.keys())
        
        for bt in brokers_to_sync:
            try:
                success = await self._sync_positions(bt)
                results[bt] = success
            except Exception as e:
                self.logger.error(f"Error syncing positions for {bt.value}: {str(e)}")
                results[bt] = False
        
        return results
    
    async def get_positions(self, account_id: str, source: str = "local") -> Dict[str, Position]:
        """Get positions for an account"""
        if source == "local":
            return self.local_positions.get(account_id, {})
        elif source == "broker":
            return self.broker_positions.get(account_id, {})
        else:
            raise ValueError("Source must be 'local' or 'broker'")
    
    async def get_position(self, account_id: str, symbol: str, source: str = "local") -> Optional[Position]:
        """Get specific position"""
        positions = await self.get_positions(account_id, source)
        return positions.get(symbol)
    
    async def update_local_position(self, account_id: str, position: Position):
        """Update local position"""
        if account_id not in self.local_positions:
            self.local_positions[account_id] = {}
        
        self.local_positions[account_id][position.symbol] = position
        
        # Trigger callback
        await self._trigger_callback("position_updated", {
            "account_id": account_id,
            "position": position,
            "source": "local"
        })
        
        self.logger.debug(f"Updated local position: {account_id} {position.symbol} {position.quantity}")
    
    async def get_discrepancies(self, account_id: Optional[str] = None) -> List[PositionDiscrepancy]:
        """Get position discrepancies"""
        discrepancies = list(self.discrepancies.values())
        
        if account_id:
            discrepancies = [d for d in discrepancies if d.account_id == account_id]
        
        return discrepancies
    
    async def resolve_discrepancy(self, discrepancy_id: str, resolution_notes: str) -> bool:
        """Mark discrepancy as resolved"""
        if discrepancy_id in self.discrepancies:
            discrepancy = self.discrepancies[discrepancy_id]
            discrepancy.resolved_at = datetime.now()
            discrepancy.resolution_notes = resolution_notes
            
            await self._trigger_callback("discrepancy_resolved", discrepancy)
            
            self.logger.info(f"Resolved discrepancy {discrepancy_id}: {resolution_notes}")
            return True
        
        return False
    
    async def _sync_positions_loop(self, broker_type: BrokerType):
        """Background loop for position synchronization"""
        while True:
            try:
                await asyncio.sleep(self.sync_interval)
                await self._sync_positions(broker_type)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in position sync loop for {broker_type.value}: {str(e)}")
                self.sync_metrics[broker_type].error_count += 1
                await self._trigger_callback("sync_error", {
                    "broker_type": broker_type,
                    "error": str(e)
                })
    
    async def _sync_positions(self, broker_type: BrokerType) -> bool:
        """Sync positions from broker"""
        start_time = datetime.now()
        
        try:
            broker = self.brokers.get(broker_type)
            if not broker or not broker.is_connected:
                return False
            
            # Get all accounts for this broker
            accounts_to_sync = []
            for local_account, broker_mappings in self.account_mappings.items():
                if broker_type in broker_mappings:
                    accounts_to_sync.append((local_account, broker_mappings[broker_type]))
            
            total_positions = 0
            
            for local_account, broker_account in accounts_to_sync:
                try:
                    # Get positions from broker
                    broker_positions = await broker.get_positions(broker_account)
                    
                    # Update broker position cache
                    if local_account not in self.broker_positions:
                        self.broker_positions[local_account] = {}
                    
                    # Clear old positions for this account
                    self.broker_positions[local_account].clear()
                    
                    # Store new positions
                    for pos in broker_positions:
                        # Convert to our Position format if needed
                        position = Position(
                            symbol=pos.symbol,
                            quantity=pos.quantity,
                            average_cost=pos.average_cost,
                            market_value=pos.market_value,
                            unrealized_pnl=pos.unrealized_pnl,
                            realized_pnl=getattr(pos, 'realized_pnl', Decimal('0')),
                            account_id=local_account,
                            broker_type=broker_type,
                            last_updated=datetime.now()
                        )
                        
                        self.broker_positions[local_account][pos.symbol] = position
                        total_positions += 1
                    
                    self.logger.debug(f"Synced {len(broker_positions)} positions for account {local_account}")
                    
                except Exception as e:
                    self.logger.error(f"Error syncing positions for account {local_account}: {str(e)}")
                    continue
            
            # Update metrics
            sync_duration = (datetime.now() - start_time).total_seconds()
            metrics = self.sync_metrics[broker_type]
            
            metrics.last_sync_time = datetime.now()
            metrics.sync_duration = sync_duration
            metrics.positions_synced = total_positions
            
            # Update average sync time
            if metrics.avg_sync_time == 0:
                metrics.avg_sync_time = sync_duration
            else:
                metrics.avg_sync_time = (metrics.avg_sync_time * 0.9) + (sync_duration * 0.1)
            
            self.logger.debug(f"Position sync completed for {broker_type.value}: {total_positions} positions in {sync_duration:.2f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Error syncing positions for {broker_type.value}: {str(e)}")
            self.sync_metrics[broker_type].error_count += 1
            return False
    
    async def _reconciliation_loop(self):
        """Background loop for position reconciliation"""
        while True:
            try:
                await asyncio.sleep(self.reconciliation_interval)
                await self._reconcile_positions()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in reconciliation loop: {str(e)}")
    
    async def _reconcile_positions(self):
        """Reconcile local and broker positions"""
        try:
            total_discrepancies = 0
            
            # Get all accounts
            all_accounts = set(self.local_positions.keys()) | set(self.broker_positions.keys())
            
            for account_id in all_accounts:
                local_positions = self.local_positions.get(account_id, {})
                broker_positions = self.broker_positions.get(account_id, {})
                
                # Get all symbols
                all_symbols = set(local_positions.keys()) | set(broker_positions.keys())
                
                for symbol in all_symbols:
                    local_pos = local_positions.get(symbol)
                    broker_pos = broker_positions.get(symbol)
                    
                    discrepancy = await self._check_position_discrepancy(
                        account_id, symbol, local_pos, broker_pos
                    )
                    
                    if discrepancy:
                        discrepancy_id = f"{account_id}_{symbol}_{discrepancy.broker_type.value}"
                        
                        # Check if this is a new discrepancy
                        if discrepancy_id not in self.discrepancies:
                            self.discrepancies[discrepancy_id] = discrepancy
                            total_discrepancies += 1
                            
                            await self._trigger_callback("discrepancy_detected", discrepancy)
                            
                            self.logger.warning(
                                f"Position discrepancy detected: {account_id} {symbol} "
                                f"Local: {local_pos.quantity if local_pos else 0} "
                                f"Broker: {broker_pos.quantity if broker_pos else 0}"
                            )
                        else:
                            # Update existing discrepancy
                            existing = self.discrepancies[discrepancy_id]
                            existing.local_position = local_pos
                            existing.broker_position = broker_pos
                            existing.quantity_diff = discrepancy.quantity_diff
                            existing.value_diff = discrepancy.value_diff
            
            # Update metrics
            for broker_type in self.brokers.keys():
                if broker_type in self.sync_metrics:
                    self.sync_metrics[broker_type].discrepancies_found = total_discrepancies
            
            if total_discrepancies > 0:
                self.logger.info(f"Reconciliation completed: {total_discrepancies} discrepancies found")
            else:
                self.logger.debug("Reconciliation completed: no discrepancies found")
                
        except Exception as e:
            self.logger.error(f"Error in position reconciliation: {str(e)}")
    
    async def _check_position_discrepancy(
        self, 
        account_id: str, 
        symbol: str, 
        local_pos: Optional[Position], 
        broker_pos: Optional[Position]
    ) -> Optional[PositionDiscrepancy]:
        """Check for position discrepancy"""
        try:
            # Determine broker type
            broker_type = None
            if broker_pos:
                broker_type = broker_pos.broker_type
            elif local_pos:
                # Try to determine broker type from account mapping
                for bt in self.brokers.keys():
                    if account_id in self.account_mappings and bt in self.account_mappings[account_id]:
                        broker_type = bt
                        break
            
            if not broker_type:
                return None
            
            # Check for missing positions
            if local_pos and not broker_pos:
                return PositionDiscrepancy(
                    symbol=symbol,
                    account_id=account_id,
                    broker_type=broker_type,
                    local_position=local_pos,
                    broker_position=None,
                    quantity_diff=local_pos.quantity,
                    value_diff=local_pos.market_value,
                    status=ReconciliationStatus.MISSING_BROKER,
                    detected_at=datetime.now()
                )
            
            if broker_pos and not local_pos:
                return PositionDiscrepancy(
                    symbol=symbol,
                    account_id=account_id,
                    broker_type=broker_type,
                    local_position=None,
                    broker_position=broker_pos,
                    quantity_diff=-broker_pos.quantity,
                    value_diff=-broker_pos.market_value,
                    status=ReconciliationStatus.MISSING_LOCAL,
                    detected_at=datetime.now()
                )
            
            # Check for quantity discrepancies
            if local_pos and broker_pos:
                quantity_diff = local_pos.quantity - broker_pos.quantity
                value_diff = local_pos.market_value - broker_pos.market_value
                
                if abs(quantity_diff) >= self.discrepancy_threshold:
                    return PositionDiscrepancy(
                        symbol=symbol,
                        account_id=account_id,
                        broker_type=broker_type,
                        local_position=local_pos,
                        broker_position=broker_pos,
                        quantity_diff=quantity_diff,
                        value_diff=value_diff,
                        status=ReconciliationStatus.DISCREPANCY,
                        detected_at=datetime.now()
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking position discrepancy: {str(e)}")
            return PositionDiscrepancy(
                symbol=symbol,
                account_id=account_id,
                broker_type=broker_type or BrokerType.ROBINHOOD,
                local_position=local_pos,
                broker_position=broker_pos,
                quantity_diff=Decimal('0'),
                value_diff=Decimal('0'),
                status=ReconciliationStatus.ERROR,
                detected_at=datetime.now()
            )
    
    async def _trigger_callback(self, event: str, data: Any):
        """Trigger event callbacks"""
        try:
            callbacks = self.position_callbacks.get(event, [])
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
    
    async def get_sync_metrics(self) -> Dict[BrokerType, SyncMetrics]:
        """Get synchronization metrics"""
        return self.sync_metrics.copy()
    
    async def export_positions(self, account_id: str, format: str = "json") -> str:
        """Export positions for backup or analysis"""
        try:
            data = {
                "account_id": account_id,
                "export_time": datetime.now().isoformat(),
                "local_positions": {},
                "broker_positions": {}
            }
            
            # Export local positions
            if account_id in self.local_positions:
                for symbol, pos in self.local_positions[account_id].items():
                    data["local_positions"][symbol] = {
                        "symbol": pos.symbol,
                        "quantity": str(pos.quantity),
                        "average_cost": str(pos.average_cost),
                        "market_value": str(pos.market_value),
                        "unrealized_pnl": str(pos.unrealized_pnl),
                        "realized_pnl": str(pos.realized_pnl),
                        "last_updated": pos.last_updated.isoformat()
                    }
            
            # Export broker positions
            if account_id in self.broker_positions:
                for symbol, pos in self.broker_positions[account_id].items():
                    data["broker_positions"][symbol] = {
                        "symbol": pos.symbol,
                        "quantity": str(pos.quantity),
                        "average_cost": str(pos.average_cost),
                        "market_value": str(pos.market_value),
                        "unrealized_pnl": str(pos.unrealized_pnl),
                        "realized_pnl": str(pos.realized_pnl),
                        "broker_type": pos.broker_type.value,
                        "last_updated": pos.last_updated.isoformat()
                    }
            
            if format == "json":
                return json.dumps(data, indent=2)
            else:
                raise ValueError("Only JSON format is currently supported")
                
        except Exception as e:
            self.logger.error(f"Error exporting positions: {str(e)}")
            raise