"""
Helper classes and methods for trading workflow tests.
Extracted to improve maintainability and reduce code duplication.
"""

from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock
import pandas as pd
from datetime import datetime, timezone, timedelta

from src.models.market_data import MarketData, ExchangeType
from src.models.trading_signal import TradingSignal, TradingAction


class TradingWorkflowTestHelper:
    """Helper class for setting up and executing trading workflow tests."""
    
    @staticmethod
    def setup_mock_exchange_data(exchange_type: ExchangeType, symbol: str, 
                                sample_data: List[MarketData]) -> pd.DataFrame:
        """Setup mock exchange data for testing."""
        filtered_data = [md for md in sample_data 
                        if md.exchange == exchange_type and md.symbol == symbol]
        
        return pd.DataFrame([{
            'timestamp': md.timestamp,
            'symbol': md.symbol,
            'open': md.open,
            'high': md.high,
            'low': md.low,
            'close': md.close,
            'volume': md.volume
        } for md in filtered_data])
    
    @staticmethod
    def create_mock_trading_signal(symbol: str, action: TradingAction = TradingAction.BUY,
                                  confidence: float = 0.85) -> TradingSignal:
        """Create a mock trading signal for testing."""
        return TradingSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            position_size=0.1,
            target_price=155.0 if symbol == 'AAPL' else 1.1100,
            stop_loss=145.0 if symbol == 'AAPL' else 1.1000,
            timestamp=datetime.now(timezone.utc),
            model_version='test-v1.0'
        )
    
    @staticmethod
    def setup_mock_account_info(exchange_type: ExchangeType) -> Dict[str, Any]:
        """Setup mock account information based on exchange type."""
        if exchange_type == ExchangeType.ROBINHOOD:
            return {
                'buying_power': 50000.0,
                'total_equity': 100000.0
            }
        elif exchange_type == ExchangeType.OANDA:
            return {
                'balance': 25000.0,
                'margin_available': 20000.0,
                'currency': 'USD'
            }
        else:  # COINBASE
            return {
                'available_balance': {'USD': 15000.0, 'BTC': 0.0},
                'total_balance': {'USD': 15000.0, 'BTC': 0.0}
            }


class WorkflowStepExecutor:
    """Executes individual steps of trading workflows."""
    
    def __init__(self, data_aggregator, decision_engine, portfolio_service, risk_service):
        self.data_aggregator = data_aggregator
        self.decision_engine = decision_engine
        self.portfolio_service = portfolio_service
        self.risk_service = risk_service
    
    async def execute_data_ingestion_step(self, symbol: str, exchange: str) -> pd.DataFrame:
        """Execute data ingestion step."""
        return await self.data_aggregator.get_historical_data(
            symbol=symbol,
            exchange=exchange,
            start_date=datetime.now(timezone.utc) - timedelta(days=7),
            end_date=datetime.now(timezone.utc)
        )
    
    async def execute_signal_generation_step(self, symbol: str, data: pd.DataFrame) -> TradingSignal:
        """Execute signal generation step."""
        return await self.decision_engine.generate_signal(symbol, data)
    
    async def execute_risk_validation_step(self, signal: TradingSignal) -> bool:
        """Execute risk validation step."""
        return await self.risk_service.validate_trade(
            signal, self.portfolio_service.get_portfolio()
        )
    
    async def execute_position_sizing_step(self, signal: TradingSignal) -> int:
        """Execute position sizing step."""
        return await self.portfolio_service.calculate_optimal_position_size(signal)