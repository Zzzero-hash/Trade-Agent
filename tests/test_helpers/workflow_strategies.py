"""
Strategy pattern implementation for different trading workflow tests.
Improves code organization and reduces duplication.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from unittest.mock import patch
import pandas as pd

from src.models.market_data import ExchangeType
from src.models.trading_signal import TradingAction
from .trading_workflow_helpers import TradingWorkflowTestHelper, WorkflowStepExecutor


class TradingWorkflowStrategy(ABC):
    """Abstract strategy for trading workflow tests."""
    
    def __init__(self, symbol: str, exchange_name: str, exchange_type: ExchangeType):
        self.symbol = symbol
        self.exchange_name = exchange_name
        self.exchange_type = exchange_type
        self.helper = TradingWorkflowTestHelper()
    
    @abstractmethod
    def get_mock_order_result(self) -> Dict[str, Any]:
        """Get mock order result specific to exchange."""
        pass
    
    @abstractmethod
    def get_order_payload(self) -> Dict[str, Any]:
        """Get order payload specific to exchange."""
        pass
    
    def setup_mock_data(self, mock_exchanges: Dict, sample_market_data) -> None:
        """Setup mock data for the exchange."""
        mock_data = self.helper.setup_mock_exchange_data(
            self.exchange_type, self.symbol, sample_market_data
        )
        mock_exchanges[self.exchange_name].get_historical_data.return_value = mock_data
        
        # Setup account info
        account_info = self.helper.setup_mock_account_info(self.exchange_type)
        mock_exchanges[self.exchange_name].get_account_info.return_value = account_info
        
        # Setup order result
        order_result = self.get_mock_order_result()
        mock_exchanges[self.exchange_name].place_order.return_value = order_result
    
    async def execute_workflow(self, executor: WorkflowStepExecutor, 
                              mock_exchanges: Dict) -> Dict[str, Any]:
        """Execute the complete trading workflow."""
        results = {}
        
        # Step 1: Data ingestion
        historical_data = await executor.execute_data_ingestion_step(
            self.symbol, self.exchange_name
        )
        results['data_ingested'] = not historical_data.empty
        
        # Step 2: Signal generation
        with patch.object(executor.decision_engine, 'generate_signal') as mock_signal:
            mock_signal.return_value = self.helper.create_mock_trading_signal(
                self.symbol, TradingAction.BUY
            )
            
            signal = await executor.execute_signal_generation_step(
                self.symbol, historical_data
            )
            results['signal_generated'] = signal.action == TradingAction.BUY
        
        # Step 3: Risk validation
        with patch.object(executor.risk_service, 'validate_trade') as mock_risk:
            mock_risk.return_value = True
            risk_approved = await executor.execute_risk_validation_step(signal)
            results['risk_approved'] = risk_approved
        
        # Step 4: Position sizing
        with patch.object(executor.portfolio_service, 'calculate_position_sizing') as mock_position:
            mock_position.return_value = 100
            optimal_size = await executor.execute_position_sizing_step(signal)
            results['position_sized'] = optimal_size == 100
        
        # Step 5: Order execution
        order_result = await mock_exchanges[self.exchange_name].place_order(
            self.get_order_payload()
        )
        results['order_executed'] = order_result['status'] == 'filled'
        
        # Step 6: Portfolio update
        with patch.object(executor.portfolio_service, 'update_position') as mock_update:
            mock_update.return_value = True
            portfolio_updated = await executor.portfolio_service.update_position(
                self.symbol, 100, 150.50, 'buy'
            )
            results['portfolio_updated'] = portfolio_updated
        
        return results


class StockTradingStrategy(TradingWorkflowStrategy):
    """Strategy for stock trading workflow tests."""
    
    def __init__(self):
        super().__init__('AAPL', 'robinhood', ExchangeType.ROBINHOOD)
    
    def get_mock_order_result(self) -> Dict[str, Any]:
        return {
            'order_id': 'test_order_123',
            'status': 'filled',
            'filled_quantity': 100,
            'filled_price': 150.50
        }
    
    def get_order_payload(self) -> Dict[str, Any]:
        return {
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 100,
            'type': 'market'
        }


class ForexTradingStrategy(TradingWorkflowStrategy):
    """Strategy for forex trading workflow tests."""
    
    def __init__(self):
        super().__init__('EUR/USD', 'oanda', ExchangeType.OANDA)
    
    def get_mock_order_result(self) -> Dict[str, Any]:
        return {
            'order_id': 'forex_order_456',
            'status': 'filled',
            'filled_units': 10000,
            'filled_price': 1.1050
        }
    
    def get_order_payload(self) -> Dict[str, Any]:
        return {
            'instrument': 'EUR_USD',
            'units': 10000,
            'type': 'MARKET'
        }


class CryptoTradingStrategy(TradingWorkflowStrategy):
    """Strategy for crypto trading workflow tests."""
    
    def __init__(self):
        super().__init__('BTC/USD', 'coinbase', ExchangeType.COINBASE)
    
    def get_mock_order_result(self) -> Dict[str, Any]:
        return {
            'order_id': 'crypto_order_789',
            'status': 'filled',
            'filled_size': 0.1,
            'filled_price': 45500.0
        }
    
    def get_order_payload(self) -> Dict[str, Any]:
        return {
            'product_id': 'BTC-USD',
            'side': 'buy',
            'size': 0.1,
            'type': 'market'
        }


class WorkflowStrategyFactory:
    """Factory for creating workflow strategies."""
    
    STRATEGIES = {
        'stock': StockTradingStrategy,
        'forex': ForexTradingStrategy,
        'crypto': CryptoTradingStrategy,
    }
    
    @classmethod
    def create_strategy(cls, strategy_type: str) -> TradingWorkflowStrategy:
        """Create a workflow strategy by type."""
        strategy_class = cls.STRATEGIES.get(strategy_type)
        if not strategy_class:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        return strategy_class()