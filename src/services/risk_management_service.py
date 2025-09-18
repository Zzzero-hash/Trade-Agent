"""
Production Risk Management Service

Provides real-time risk monitoring, position limits, stop-loss automation,
and risk calculations for production trading operations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

import pandas as pd
from pydantic import BaseModel

from src.models.trading_models import Position, Trade
from src.models.risk_models import RiskAlert, RiskLevel, RiskLimits, RiskMetrics
from src.services.monitoring_service import MonitoringService
from src.services.alert_service import ProductionAlertService
from src.repositories.position_repository import PositionRepository
from src.repositories.trade_repository import TradeRepository


class ProductionRiskManager:
    """Production-grade risk management with real-time monitoring and automation."""
    
    def __init__(
        self,
        position_repo: PositionRepository,
        trade_repo: TradeRepository,
        monitoring_service: MonitoringService,
        alert_service: ProductionAlertService
    ):
        self.position_repo = position_repo
        self.trade_repo = trade_repo
        self.monitoring_service = monitoring_service
        self.alert_service = alert_service
        self.logger = logging.getLogger(__name__)
        
        # Risk monitoring state
        self.active_positions: Dict[str, Position] = {}
        self.risk_limits: Dict[str, RiskLimits] = {}
        self.stop_loss_orders: Dict[str, str] = {}  # position_id -> order_id
        self.circuit_breaker_active: Set[str] = set()
        
        # Monitoring intervals
        self.risk_check_interval = 5  # seconds
        self.position_sync_interval = 30  # seconds
        
    async def start_risk_monitoring(self):
        """Start real-time risk monitoring tasks."""
        self.logger.info("Starting production risk monitoring")
        
        # Start concurrent monitoring tasks
        await asyncio.gather(
            self._risk_monitoring_loop(),
            self._position_sync_loop(),
            self._stop_loss_monitoring_loop(),
            self._portfolio_risk_monitoring_loop()
        )
    
    async def _risk_monitoring_loop(self):
        """Main risk monitoring loop - runs every 5 seconds."""
        while True:
            try:
                start_time = datetime.utcnow()
                
                # Check all active positions for risk violations
                for customer_id, positions in self.active_positions.items():
                    await self._check_position_risks(customer_id, positions)
                
                # Record monitoring latency
                latency = (datetime.utcnow() - start_time).total_seconds()
                await self.monitoring_service.record_metric(
                    "risk_monitoring_latency_seconds", latency
                )
                
                await asyncio.sleep(self.risk_check_interval)
                
            except Exception as e:
                self.logger.error(f"Risk monitoring loop error: {e}")
                await self.alert_service.send_critical_alert(
                    "Risk monitoring loop failure", str(e)
                )
                await asyncio.sleep(self.risk_check_interval)
    
    async def _check_position_risks(self, customer_id: str, positions: List[Position]):
        """Check individual position risks and trigger alerts/actions."""
        customer_limits = self.risk_limits.get(customer_id)
        if not customer_limits:
            return
        
        total_portfolio_value = sum(pos.current_value for pos in positions)
        daily_pnl = sum(pos.unrealized_pnl for pos in positions)
        
        for position in positions:
            # Check position size limits
            if position.current_value > customer_limits.max_position_size:
                await self._handle_position_size_violation(position, customer_limits)
            
            # Check stop-loss levels
            if await self._should_trigger_stop_loss(position, customer_limits):
                await self._execute_stop_loss(position)
            
            # Check individual position risk
            position_risk = abs(position.unrealized_pnl) / position.cost_basis
            if position_risk > 0.1:  # 10% loss threshold
                await self._send_risk_alert(
                    customer_id, position, RiskLevel.HIGH,
                    f"Position loss exceeds 10%: {position_risk:.2%}"
                )
        
        # Check portfolio-level risks
        if abs(daily_pnl) > customer_limits.max_daily_loss:
            await self._handle_daily_loss_violation(customer_id, daily_pnl, customer_limits)
        
        # Check portfolio risk concentration
        portfolio_risk = await self._calculate_portfolio_risk(positions)
        if portfolio_risk > customer_limits.max_portfolio_risk:
            await self._handle_portfolio_risk_violation(customer_id, portfolio_risk, customer_limits)
    
    async def _should_trigger_stop_loss(self, position: Position, limits: RiskLimits) -> bool:
        """Determine if stop-loss should be triggered for a position."""
        if position.position_id in self.stop_loss_orders:
            return False  # Stop-loss already placed
        
        loss_percentage = (position.cost_basis - position.current_value) / position.cost_basis
        return loss_percentage >= limits.stop_loss_percentage
    
    async def _execute_stop_loss(self, position: Position):
        """Execute automated stop-loss order."""
        try:
            self.logger.warning(f"Executing stop-loss for position {position.position_id}")
            
            # Create stop-loss order (implementation depends on broker integration)
            order_id = await self._place_stop_loss_order(position)
            self.stop_loss_orders[position.position_id] = order_id
            
            # Send alert
            await self._send_risk_alert(
                position.customer_id, position, RiskLevel.CRITICAL,
                f"Stop-loss executed for position {position.symbol}"
            )
            
            # Record metric
            await self.monitoring_service.record_metric(
                "stop_loss_executions_total", 1,
                labels={"symbol": position.symbol, "customer_id": position.customer_id}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to execute stop-loss for {position.position_id}: {e}")
            await self.alert_service.send_critical_alert(
                "Stop-loss execution failed", 
                f"Position: {position.position_id}, Error: {str(e)}"
            )
    
    async def _place_stop_loss_order(self, position: Position) -> str:
        """Place stop-loss order with broker (placeholder for broker integration)."""
        # This would integrate with the actual broker APIs
        # For now, return a mock order ID
        return f"stop_loss_{position.position_id}_{datetime.utcnow().timestamp()}"
    
    async def _handle_position_size_violation(self, position: Position, limits: RiskLimits):
        """Handle position size limit violations."""
        await self._send_risk_alert(
            position.customer_id, position, RiskLevel.HIGH,
            f"Position size exceeds limit: {position.current_value} > {limits.max_position_size}"
        )
        
        # Record violation metric
        await self.monitoring_service.record_metric(
            "position_size_violations_total", 1,
            labels={"customer_id": position.customer_id}
        )
    
    async def _handle_daily_loss_violation(self, customer_id: str, daily_pnl: Decimal, limits: RiskLimits):
        """Handle daily loss limit violations."""
        await self._send_risk_alert(
            customer_id, None, RiskLevel.CRITICAL,
            f"Daily loss exceeds limit: {daily_pnl} > {limits.max_daily_loss}"
        )
        
        # Activate circuit breaker for this customer
        self.circuit_breaker_active.add(customer_id)
        
        # Record violation metric
        await self.monitoring_service.record_metric(
            "daily_loss_violations_total", 1,
            labels={"customer_id": customer_id}
        )
    
    async def _handle_portfolio_risk_violation(self, customer_id: str, portfolio_risk: Decimal, limits: RiskLimits):
        """Handle portfolio risk limit violations."""
        await self._send_risk_alert(
            customer_id, None, RiskLevel.HIGH,
            f"Portfolio risk exceeds limit: {portfolio_risk} > {limits.max_portfolio_risk}"
        )
        
        # Record violation metric
        await self.monitoring_service.record_metric(
            "portfolio_risk_violations_total", 1,
            labels={"customer_id": customer_id}
        )
    
    async def _calculate_portfolio_risk(self, positions: List[Position]) -> Decimal:
        """Calculate portfolio-level risk metrics."""
        if not positions:
            return Decimal('0')
        
        # Calculate Value at Risk (VaR) using historical simulation
        returns = []
        for position in positions:
            # Get historical returns for the position (simplified)
            position_returns = await self._get_position_returns(position)
            returns.extend(position_returns)
        
        if not returns:
            return Decimal('0')
        
        # Calculate 95% VaR
        returns_series = pd.Series(returns)
        var_95 = returns_series.quantile(0.05)
        
        return abs(Decimal(str(var_95)))
    
    async def _get_position_returns(self, position: Position) -> List[float]:
        """Get historical returns for a position (placeholder)."""
        # This would fetch actual historical data
        # For now, return mock data
        return [0.01, -0.02, 0.015, -0.005, 0.008]
    
    async def _send_risk_alert(
        self, 
        customer_id: str, 
        position: Optional[Position], 
        risk_level: RiskLevel, 
        message: str
    ):
        """Send risk alert through the alert service."""
        alert = RiskAlert(
            alert_id=f"risk_{datetime.utcnow().timestamp()}",
            customer_id=customer_id,
            risk_level=risk_level,
            alert_type="risk_violation",
            message=message,
            timestamp=datetime.utcnow(),
            position_id=position.position_id if position else None,
            current_value=position.current_value if position else None
        )
        
        await self.alert_service.send_risk_alert(alert)
    
    async def _position_sync_loop(self):
        """Sync positions from database every 30 seconds."""
        while True:
            try:
                # Fetch all active positions
                active_positions = await self.position_repo.get_active_positions()
                
                # Group by customer
                customer_positions = {}
                for position in active_positions:
                    if position.customer_id not in customer_positions:
                        customer_positions[position.customer_id] = []
                    customer_positions[position.customer_id].append(position)
                
                self.active_positions = customer_positions
                
                # Record sync metric
                await self.monitoring_service.record_metric(
                    "position_sync_count", len(active_positions)
                )
                
                await asyncio.sleep(self.position_sync_interval)
                
            except Exception as e:
                self.logger.error(f"Position sync error: {e}")
                await asyncio.sleep(self.position_sync_interval)
    
    async def _stop_loss_monitoring_loop(self):
        """Monitor stop-loss order status."""
        while True:
            try:
                # Check status of all active stop-loss orders
                for position_id, order_id in list(self.stop_loss_orders.items()):
                    order_status = await self._check_order_status(order_id)
                    
                    if order_status in ['filled', 'cancelled']:
                        # Remove from tracking
                        del self.stop_loss_orders[position_id]
                        
                        # Record metric
                        await self.monitoring_service.record_metric(
                            "stop_loss_orders_completed_total", 1,
                            labels={"status": order_status}
                        )
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Stop-loss monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _check_order_status(self, order_id: str) -> str:
        """Check order status with broker (placeholder)."""
        # This would integrate with broker APIs
        # For now, return mock status
        return "pending"
    
    async def _portfolio_risk_monitoring_loop(self):
        """Monitor portfolio-level risk metrics."""
        while True:
            try:
                for customer_id, positions in self.active_positions.items():
                    if customer_id in self.circuit_breaker_active:
                        continue  # Skip customers with active circuit breakers
                    
                    # Calculate and record portfolio metrics
                    portfolio_value = sum(pos.current_value for pos in positions)
                    portfolio_pnl = sum(pos.unrealized_pnl for pos in positions)
                    portfolio_risk = await self._calculate_portfolio_risk(positions)
                    
                    # Record metrics
                    await self.monitoring_service.record_metric(
                        "portfolio_value", float(portfolio_value),
                        labels={"customer_id": customer_id}
                    )
                    await self.monitoring_service.record_metric(
                        "portfolio_pnl", float(portfolio_pnl),
                        labels={"customer_id": customer_id}
                    )
                    await self.monitoring_service.record_metric(
                        "portfolio_risk", float(portfolio_risk),
                        labels={"customer_id": customer_id}
                    )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Portfolio risk monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def set_risk_limits(self, customer_id: str, limits: RiskLimits):
        """Set risk limits for a customer."""
        self.risk_limits[customer_id] = limits
        self.logger.info(f"Updated risk limits for customer {customer_id}")
    
    async def is_trading_allowed(self, customer_id: str) -> bool:
        """Check if trading is allowed for a customer (circuit breaker check)."""
        return customer_id not in self.circuit_breaker_active
    
    async def reset_circuit_breaker(self, customer_id: str):
        """Reset circuit breaker for a customer (manual intervention)."""
        if customer_id in self.circuit_breaker_active:
            self.circuit_breaker_active.remove(customer_id)
            self.logger.info(f"Circuit breaker reset for customer {customer_id}")
            
            await self.monitoring_service.record_metric(
                "circuit_breaker_resets_total", 1,
                labels={"customer_id": customer_id}
            )