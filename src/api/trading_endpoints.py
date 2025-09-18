"""
Trading-specific API endpoints for signals and portfolio management.

This module provides REST API endpoints for trading signals, portfolio
management, and real-time market data streaming.

Requirements: 3.1, 3.2, 11.1, 11.2
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from functools import lru_cache
from fastapi import APIRouter, HTTPException, Depends, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import asyncio
import json

from src.models.trading_signal import TradingSignal, TradingAction
from src.models.portfolio import Portfolio, Position
from src.services.trading_decision_engine import TradingDecisionEngine
from src.services.portfolio_management_service import PortfolioManagementService
from src.services.data_aggregator import DataAggregator
from src.utils.logging import get_logger
from src.utils.monitoring import get_metrics_collector
from .auth import get_current_user, User

logger = get_logger(__name__)
metrics = get_metrics_collector()

# Create router
router = APIRouter(prefix="/api/v1/trading", tags=["trading"])


# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time data streaming."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept WebSocket connection and register user."""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        if user_id not in self.user_connections:
            self.user_connections[user_id] = []
        self.user_connections[user_id].append(websocket)
        
        logger.info(f"WebSocket connected for user {user_id}")
    
    def disconnect(self, websocket: WebSocket, user_id: str):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        if user_id in self.user_connections:
            if websocket in self.user_connections[user_id]:
                self.user_connections[user_id].remove(websocket)
            
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        logger.info(f"WebSocket disconnected for user {user_id}")
    
    async def send_personal_message(self, message: str, user_id: str):
        """Send message to specific user's connections."""
        if user_id in self.user_connections:
            for connection in self.user_connections[user_id]:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Failed to send message to user {user_id}: {e}")
    
    async def broadcast(self, message: str):
        """Broadcast message to all active connections."""
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Failed to broadcast message: {e}")


# Global connection manager
manager = ConnectionManager()


@lru_cache()
def _decision_engine_singleton() -> TradingDecisionEngine:
    """Provide a cached trading decision engine instance."""

    return TradingDecisionEngine()


def get_trading_decision_engine() -> TradingDecisionEngine:
    """Dependency provider for the trading decision engine."""

    return _decision_engine_singleton()


@lru_cache()
def _portfolio_service_singleton() -> PortfolioManagementService:
    """Provide a cached portfolio management service."""

    return PortfolioManagementService()


def get_portfolio_management_service() -> PortfolioManagementService:
    """Dependency provider for portfolio operations."""

    return _portfolio_service_singleton()


@lru_cache()
def _data_aggregator_singleton() -> DataAggregator:
    """Provide a cached data aggregator instance."""

    return DataAggregator()


def get_data_aggregator() -> DataAggregator:
    """Dependency provider for market data aggregation."""

    return _data_aggregator_singleton()


# Trading Signal Endpoints

@router.post("/signals/generate", response_model=TradingSignal)
async def generate_trading_signal(
    symbol: str,
    user: User = Depends(get_current_user),
    decision_engine: TradingDecisionEngine = Depends(get_trading_decision_engine)
) -> TradingSignal:
    """
    Generate a trading signal for a specific symbol.
    
    Uses the CNN+LSTM enhanced RL ensemble to generate trading recommendations.
    """
    try:
        # Generate signal using decision engine
        signal = await decision_engine.generate_signal(symbol, user.id)
        
        # Record metrics
        metrics.increment_counter("trading_signals_generated", 1)
        
        # Send real-time update to user
        await manager.send_personal_message(
            json.dumps({
                "type": "signal_generated",
                "data": signal.model_dump(),
                "timestamp": datetime.now().isoformat()
            }),
            user.id
        )
        
        return signal
        
    except Exception as e:
        logger.error(f"Failed to generate signal for {symbol}: {e}")
        metrics.increment_counter("trading_signal_errors", 1)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signals/history", response_model=List[TradingSignal])
async def get_signal_history(
    symbol: Optional[str] = None,
    action: Optional[TradingAction] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = Query(100, le=1000),
    user: User = Depends(get_current_user)
) -> List[TradingSignal]:
    """
    Get historical trading signals with optional filtering.
    
    - **symbol**: Filter by trading symbol
    - **action**: Filter by trading action (BUY/SELL/HOLD)
    - **start_date**: Filter signals after this date
    - **end_date**: Filter signals before this date
    - **limit**: Maximum number of signals to return
    """
    try:
        # TODO: Implement database query for signal history
        # For now, return empty list as placeholder
        signals = []
        
        # Apply filters
        if symbol:
            signals = [s for s in signals if s.symbol == symbol.upper()]
        if action:
            signals = [s for s in signals if s.action == action]
        if start_date:
            signals = [s for s in signals if s.timestamp >= start_date]
        if end_date:
            signals = [s for s in signals if s.timestamp <= end_date]
        
        # Apply limit
        signals = signals[:limit]
        
        return signals
        
    except Exception as e:
        logger.error(f"Failed to get signal history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signals/performance")
async def get_signal_performance(
    symbol: Optional[str] = None,
    days: int = Query(30, ge=1, le=365),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get trading signal performance metrics.
    
    Returns accuracy, profit/loss, and other performance statistics.
    """
    try:
        # TODO: Implement signal performance calculation
        # For now, return mock data
        performance = {
            "symbol": symbol or "ALL",
            "period_days": days,
            "total_signals": 0,
            "accuracy": 0.0,
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "avg_return_per_signal": 0.0,
            "breakdown_by_action": {
                "BUY": {"count": 0, "accuracy": 0.0, "avg_return": 0.0},
                "SELL": {"count": 0, "accuracy": 0.0, "avg_return": 0.0},
                "HOLD": {"count": 0, "accuracy": 0.0, "avg_return": 0.0}
            }
        }
        
        return performance
        
    except Exception as e:
        logger.error(f"Failed to get signal performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Portfolio Management Endpoints

@router.get("/portfolio", response_model=Portfolio)
async def get_portfolio(
    user: User = Depends(get_current_user),
    portfolio_service: PortfolioManagementService = Depends(get_portfolio_management_service)
) -> Portfolio:
    """
    Get current portfolio for the authenticated user.
    """
    try:
        portfolio = await portfolio_service.get_portfolio(user.id)
        
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        return portfolio
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get portfolio for user {user.id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/portfolio/rebalance")
async def rebalance_portfolio(
    target_allocation: Dict[str, float],
    user: User = Depends(get_current_user),
    portfolio_service: PortfolioManagementService = Depends(get_portfolio_management_service)
) -> JSONResponse:
    """
    Rebalance portfolio to target allocation.
    
    - **target_allocation**: Dictionary mapping symbols to target weights (0.0-1.0)
    
    Example:
    ```json
    {
        "AAPL": 0.3,
        "GOOGL": 0.2,
        "MSFT": 0.2,
        "TSLA": 0.1,
        "CASH": 0.2
    }
    ```
    """
    try:
        # Validate allocation weights sum to 1.0
        total_weight = sum(target_allocation.values())
        if abs(total_weight - 1.0) > 0.01:
            raise HTTPException(
                status_code=400,
                detail=f"Target allocation weights must sum to 1.0, got {total_weight}"
            )
        
        # Execute rebalancing
        rebalance_result = await portfolio_service.rebalance_portfolio(
            user.id, target_allocation
        )
        
        # Send real-time update
        await manager.send_personal_message(
            json.dumps({
                "type": "portfolio_rebalanced",
                "data": rebalance_result,
                "timestamp": datetime.now().isoformat()
            }),
            user.id
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Portfolio rebalancing initiated",
                "rebalance_id": rebalance_result.get("rebalance_id"),
                "estimated_trades": rebalance_result.get("trades", []),
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to rebalance portfolio for user {user.id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio/performance")
async def get_portfolio_performance(
    days: int = Query(30, ge=1, le=365),
    user: User = Depends(get_current_user),
    portfolio_service: PortfolioManagementService = Depends(get_portfolio_management_service)
) -> Dict[str, Any]:
    """
    Get portfolio performance metrics over specified period.
    """
    try:
        performance = await portfolio_service.get_performance_metrics(user.id, days)
        
        return performance
        
    except Exception as e:
        logger.error(f"Failed to get portfolio performance for user {user.id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/portfolio/optimize")
async def optimize_portfolio(
    optimization_method: str = Query("mean_variance", regex="^(mean_variance|risk_parity|black_litterman)$"),
    risk_tolerance: float = Query(0.5, ge=0.0, le=1.0),
    user: User = Depends(get_current_user),
    portfolio_service: PortfolioManagementService = Depends(get_portfolio_management_service)
) -> Dict[str, Any]:
    """
    Optimize portfolio allocation using specified method.
    
    - **optimization_method**: Optimization algorithm to use
    - **risk_tolerance**: Risk tolerance level (0.0 = conservative, 1.0 = aggressive)
    """
    try:
        optimization_result = await portfolio_service.optimize_portfolio(
            user.id, optimization_method, risk_tolerance
        )
        
        return optimization_result
        
    except Exception as e:
        logger.error(f"Failed to optimize portfolio for user {user.id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Market Data Endpoints

@router.get("/market-data/{symbol}")
async def get_market_data(
    symbol: str,
    timeframe: str = Query("1h", regex="^(1m|5m|15m|1h|4h|1d)$"),
    limit: int = Query(100, ge=1, le=1000),
    user: User = Depends(get_current_user),
    data_aggregator: DataAggregator = Depends(get_data_aggregator)
) -> Dict[str, Any]:
    """
    Get historical market data for a symbol.
    
    - **symbol**: Trading symbol
    - **timeframe**: Data timeframe (1m, 5m, 15m, 1h, 4h, 1d)
    - **limit**: Number of data points to return
    """
    try:
        # Get market data from aggregator
        market_data = await data_aggregator.get_historical_data(
            symbol.upper(), timeframe, limit
        )
        
        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "data": market_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get market data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket Endpoints

@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for real-time data streaming.
    
    Provides real-time updates for:
    - Trading signals
    - Portfolio changes
    - Market data
    - System notifications
    """
    await manager.connect(websocket, user_id)
    
    try:
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "message": "WebSocket connection established"
        }))
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "subscribe_market_data":
                    symbols = message.get("symbols", [])
                    await handle_market_data_subscription(websocket, user_id, symbols)
                
                elif message.get("type") == "subscribe_signals":
                    await handle_signal_subscription(websocket, user_id)
                
                elif message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.now().isoformat()
                }))
            except Exception as e:
                logger.error(f"WebSocket message handling error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Message processing error",
                    "timestamp": datetime.now().isoformat()
                }))
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        manager.disconnect(websocket, user_id)


async def handle_market_data_subscription(websocket: WebSocket, user_id: str, symbols: List[str]):
    """Handle market data subscription for WebSocket client."""
    try:
        # TODO: Implement real-time market data streaming
        # For now, send mock data
        for symbol in symbols:
            await websocket.send_text(json.dumps({
                "type": "market_data_subscribed",
                "symbol": symbol,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }))
        
        logger.info(f"User {user_id} subscribed to market data for symbols: {symbols}")
        
    except Exception as e:
        logger.error(f"Failed to handle market data subscription: {e}")


async def handle_signal_subscription(websocket: WebSocket, user_id: str):
    """Handle trading signal subscription for WebSocket client."""
    try:
        await websocket.send_text(json.dumps({
            "type": "signals_subscribed",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "message": "Subscribed to trading signals"
        }))
        
        logger.info(f"User {user_id} subscribed to trading signals")
        
    except Exception as e:
        logger.error(f"Failed to handle signal subscription: {e}")


# Background task for real-time data streaming
async def start_real_time_streaming():
    """Start background task for real-time data streaming."""
    while True:
        try:
            # TODO: Implement real-time data fetching and broadcasting
            # For now, just sleep
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Real-time streaming error: {e}")
            await asyncio.sleep(5)


# Health check endpoint
@router.get("/health")
async def trading_health_check() -> Dict[str, Any]:
    """Health check for trading endpoints."""
    return {
        "status": "healthy",
        "active_websocket_connections": len(manager.active_connections),
        "connected_users": len(manager.user_connections),
        "timestamp": datetime.now().isoformat()
    }
