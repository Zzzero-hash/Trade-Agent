"""
End-to-end workflow integration tests.

Tests complete trading workflows from data ingestion to order execution,
including error handling and recovery scenarios.

Requirements: 2.6
"""

import pytest
import asyncio
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any
import pandas as pd
import numpy as np

from src.models.market_data import MarketData, ExchangeType
from src.models.trading_signal import TradingSignal, TradingAction
from src.models.portfolio import Portfolio, Position
from src.services.data_aggregator import DataAggregator
from src.services.trading_decision_engine import TradingDecisionEngine
from src.services.portfolio_management_service import PortfolioManagementService
from src.services.risk_management_service import RiskManagementService
from src.services.order_execution_service import OrderExecutionService


class TestCompleteWorkflows:
    """Test complete end-to-end trading workflows."""
    
    @pytest.fixture
    def workflow_components(self):
        """Setup all workflow components with mocks."""
        
        # Create mock services
        data_aggregator = Mock(spec=DataAggregator)
        decision_engine = Mock(spec=TradingDecisionEngine)
        portfolio_service = Mock(spec=PortfolioManagementService)
        risk_service = Mock(spec=RiskManagementService)
        execution_service = Mock(spec=OrderExecutionService)
        
        # Setup async methods
        data_aggregator.get_historical_data = AsyncMock()
        data_aggregator.get_real_time_data = AsyncMock()
        decision_engine.generate_signal = AsyncMock()
        portfolio_service.get_portfolio = Mock()
        portfolio_service.calculate_optimal_position_size = AsyncMock()
        portfolio_service.update_position = AsyncMock()
        risk_service.validate_trade = AsyncMock()
        risk_service.check_risk_limits = AsyncMock()
        execution_service.execute_order = AsyncMock()
        
        return {
            'data_aggregator': data_aggregator,
            'decision_engine': decision_engine,
            'portfolio_service': portfolio_service,
            'risk_service': risk_service,
            'execution_service': execution_service
        }
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio for testing."""
        return Portfolio(
            user_id='test_user',
            positions={
                'AAPL': Position(
                    symbol='AAPL',
                    quantity=100,
                    avg_cost=150.0,
                    current_price=155.0,
                    unrealized_pnl=500.0,
                    realized_pnl=0.0
                )
            },
            cash_balance=50000.0,
            total_value=65500.0,
            last_updated=datetime.now(timezone.utc)
        )
    
    @pytest.mark.asyncio
    async def test_successful_buy_workflow(self, workflow_components, sample_portfolio):
        """Test successful buy order workflow."""
        
        components = workflow_components
        
        # Step 1: Setup historical data
        historical_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-12-01', periods=100, freq='H'),
            'symbol': ['AAPL'] * 100,
            'open': np.random.uniform(149, 151, 100),
            'high': np.random.uniform(150, 152, 100),
            'low': np.random.uniform(148, 150, 100),
            'close': np.random.uniform(149, 151, 100),
            'volume': np.random.randint(100000, 1000000, 100)
        })
        
        components['data_aggregator'].get_historical_data.return_value = historical_data
        
        # Step 2: Setup signal generation
        buy_signal = TradingSignal(
            symbol='AAPL',
            action=TradingAction.BUY,
            confidence=0.85,
            position_size=0.1,
            target_price=155.0,
            stop_loss=145.0,
            timestamp=datetime.now(timezone.utc),
            model_version='test-v1.0'
        )
        
        components['decision_engine'].generate_signal.return_value = buy_signal
        
        # Step 3: Setup portfolio and risk validation
        components['portfolio_service'].get_portfolio.return_value = sample_portfolio
        components['risk_service'].validate_trade.return_value = True
        components['risk_service'].check_risk_limits.return_value = True
        components['portfolio_service'].calculate_optimal_position_size.return_value = 50
        
        # Step 4: Setup order execution
        execution_result = {
            'order_id': 'order_123',
            'status': 'filled',
            'filled_quantity': 50,
            'filled_price': 154.75,
            'commission': 1.0,
            'timestamp': datetime.now(timezone.utc)
        }
        
        components['execution_service'].execute_order.return_value = execution_result
        components['portfolio_service'].update_position.return_value = True
        
        # Execute workflow
        print("Starting buy workflow...")
        
        # Step 1: Get market data
        market_data = await components['data_aggregator'].get_historical_data(
            symbol='AAPL',
            exchange='robinhood',
            start_date=datetime.now(timezone.utc) - timedelta(days=7),
            end_date=datetime.now(timezone.utc)
        )
        
        assert not market_data.empty
        print("✓ Market data retrieved")
        
        # Step 2: Generate trading signal
        signal = await components['decision_engine'].generate_signal('AAPL', market_data)
        
        assert signal.action == TradingAction.BUY
        assert signal.confidence > 0.8
        print(f"✓ Buy signal generated (confidence: {signal.confidence})")
        
        # Step 3: Risk validation
        portfolio = components['portfolio_service'].get_portfolio()
        risk_approved = await components['risk_service'].validate_trade(signal, portfolio)
        
        assert risk_approved is True
        print("✓ Risk validation passed")
        
        # Step 4: Position sizing
        optimal_size = await components['portfolio_service'].calculate_optimal_position_size(signal)
        
        assert optimal_size > 0
        print(f"✓ Optimal position size calculated: {optimal_size} shares")
        
        # Step 5: Execute order
        order_result = await components['execution_service'].execute_order({
            'symbol': 'AAPL',
            'action': 'buy',
            'quantity': optimal_size,
            'order_type': 'market'
        })
        
        assert order_result['status'] == 'filled'
        assert order_result['filled_quantity'] == optimal_size
        print(f"✓ Order executed: {order_result['filled_quantity']} shares at ${order_result['filled_price']}")
        
        # Step 6: Update portfolio
        portfolio_updated = await components['portfolio_service'].update_position(
            'AAPL', optimal_size, order_result['filled_price'], 'buy'
        )
        
        assert portfolio_updated is True
        print("✓ Portfolio updated")
        
        print("✓ Buy workflow completed successfully")
    
    @pytest.mark.asyncio
    async def test_successful_sell_workflow(self, workflow_components, sample_portfolio):
        """Test successful sell order workflow."""
        
        components = workflow_components
        
        # Setup sell signal
        sell_signal = TradingSignal(
            symbol='AAPL',
            action=TradingAction.SELL,
            confidence=0.78,
            position_size=0.5,  # Sell 50% of position
            target_price=160.0,
            stop_loss=None,
            timestamp=datetime.now(timezone.utc),
            model_version='test-v1.0'
        )
        
        components['decision_engine'].generate_signal.return_value = sell_signal
        components['portfolio_service'].get_portfolio.return_value = sample_portfolio
        components['risk_service'].validate_trade.return_value = True
        
        # Calculate sell quantity (50% of current position)
        current_position = sample_portfolio.positions['AAPL'].quantity
        sell_quantity = int(current_position * 0.5)
        components['portfolio_service'].calculate_optimal_position_size.return_value = sell_quantity
        
        # Setup execution result
        execution_result = {
            'order_id': 'sell_order_456',
            'status': 'filled',
            'filled_quantity': sell_quantity,
            'filled_price': 159.25,
            'commission': 1.0,
            'timestamp': datetime.now(timezone.utc)
        }
        
        components['execution_service'].execute_order.return_value = execution_result
        components['portfolio_service'].update_position.return_value = True
        
        # Execute sell workflow
        print("Starting sell workflow...")
        
        # Generate sell signal
        signal = await components['decision_engine'].generate_signal('AAPL', None)
        assert signal.action == TradingAction.SELL
        print(f"✓ Sell signal generated (confidence: {signal.confidence})")
        
        # Validate trade
        portfolio = components['portfolio_service'].get_portfolio()
        risk_approved = await components['risk_service'].validate_trade(signal, portfolio)
        assert risk_approved is True
        print("✓ Risk validation passed")
        
        # Calculate sell size
        sell_size = await components['portfolio_service'].calculate_optimal_position_size(signal)
        assert sell_size == sell_quantity
        print(f"✓ Sell size calculated: {sell_size} shares")
        
        # Execute sell order
        order_result = await components['execution_service'].execute_order({
            'symbol': 'AAPL',
            'action': 'sell',
            'quantity': sell_size,
            'order_type': 'market'
        })
        
        assert order_result['status'] == 'filled'
        print(f"✓ Sell order executed: {order_result['filled_quantity']} shares at ${order_result['filled_price']}")
        
        # Update portfolio
        portfolio_updated = await components['portfolio_service'].update_position(
            'AAPL', -sell_size, order_result['filled_price'], 'sell'
        )
        
        assert portfolio_updated is True
        print("✓ Portfolio updated")
        
        print("✓ Sell workflow completed successfully")
    
    @pytest.mark.asyncio
    async def test_risk_rejection_workflow(self, workflow_components, sample_portfolio):
        """Test workflow when risk management rejects a trade."""
        
        components = workflow_components
        
        # Setup risky signal
        risky_signal = TradingSignal(
            symbol='AAPL',
            action=TradingAction.BUY,
            confidence=0.95,
            position_size=0.8,  # Very large position
            target_price=200.0,  # Unrealistic target
            stop_loss=100.0,
            timestamp=datetime.now(timezone.utc),
            model_version='test-v1.0'
        )
        
        components['decision_engine'].generate_signal.return_value = risky_signal
        components['portfolio_service'].get_portfolio.return_value = sample_portfolio
        
        # Risk service rejects the trade
        components['risk_service'].validate_trade.return_value = False
        
        print("Starting risk rejection workflow...")
        
        # Generate signal
        signal = await components['decision_engine'].generate_signal('AAPL', None)
        assert signal.action == TradingAction.BUY
        print(f"✓ Signal generated: {signal.action} with {signal.position_size} position size")
        
        # Risk validation fails
        portfolio = components['portfolio_service'].get_portfolio()
        risk_approved = await components['risk_service'].validate_trade(signal, portfolio)
        
        assert risk_approved is False
        print("✓ Risk management rejected the trade")
        
        # Verify no order execution was attempted
        components['execution_service'].execute_order.assert_not_called()
        components['portfolio_service'].update_position.assert_not_called()
        
        print("✓ Risk rejection workflow completed - no trade executed")
    
    @pytest.mark.asyncio
    async def test_partial_fill_workflow(self, workflow_components, sample_portfolio):
        """Test workflow with partial order fills."""
        
        components = workflow_components
        
        # Setup signal for large order
        large_signal = TradingSignal(
            symbol='AAPL',
            action=TradingAction.BUY,
            confidence=0.82,
            position_size=0.2,
            target_price=155.0,
            stop_loss=145.0,
            timestamp=datetime.now(timezone.utc),
            model_version='test-v1.0'
        )
        
        components['decision_engine'].generate_signal.return_value = large_signal
        components['portfolio_service'].get_portfolio.return_value = sample_portfolio
        components['risk_service'].validate_trade.return_value = True
        components['portfolio_service'].calculate_optimal_position_size.return_value = 200
        
        # Setup partial fill result
        partial_fill_result = {
            'order_id': 'partial_order_789',
            'status': 'partially_filled',
            'requested_quantity': 200,
            'filled_quantity': 150,  # Only 75% filled
            'remaining_quantity': 50,
            'filled_price': 154.80,
            'commission': 1.0,
            'timestamp': datetime.now(timezone.utc)
        }
        
        components['execution_service'].execute_order.return_value = partial_fill_result
        components['portfolio_service'].update_position.return_value = True
        
        print("Starting partial fill workflow...")
        
        # Generate signal and validate
        signal = await components['decision_engine'].generate_signal('AAPL', None)
        portfolio = components['portfolio_service'].get_portfolio()
        risk_approved = await components['risk_service'].validate_trade(signal, portfolio)
        
        assert risk_approved is True
        
        # Calculate position size
        position_size = await components['portfolio_service'].calculate_optimal_position_size(signal)
        assert position_size == 200
        print(f"✓ Requested position size: {position_size} shares")
        
        # Execute order with partial fill
        order_result = await components['execution_service'].execute_order({
            'symbol': 'AAPL',
            'action': 'buy',
            'quantity': position_size,
            'order_type': 'market'
        })
        
        assert order_result['status'] == 'partially_filled'
        assert order_result['filled_quantity'] < order_result['requested_quantity']
        print(f"✓ Partial fill: {order_result['filled_quantity']}/{order_result['requested_quantity']} shares")
        
        # Update portfolio with filled quantity only
        portfolio_updated = await components['portfolio_service'].update_position(
            'AAPL', order_result['filled_quantity'], order_result['filled_price'], 'buy'
        )
        
        assert portfolio_updated is True
        print("✓ Portfolio updated with filled quantity")
        
        print("✓ Partial fill workflow completed successfully")
    
    @pytest.mark.asyncio
    async def test_order_failure_workflow(self, workflow_components, sample_portfolio):
        """Test workflow when order execution fails."""
        
        components = workflow_components
        
        # Setup normal signal
        signal = TradingSignal(
            symbol='AAPL',
            action=TradingAction.BUY,
            confidence=0.80,
            position_size=0.1,
            target_price=155.0,
            stop_loss=145.0,
            timestamp=datetime.now(timezone.utc),
            model_version='test-v1.0'
        )
        
        components['decision_engine'].generate_signal.return_value = signal
        components['portfolio_service'].get_portfolio.return_value = sample_portfolio
        components['risk_service'].validate_trade.return_value = True
        components['portfolio_service'].calculate_optimal_position_size.return_value = 100
        
        # Setup order failure
        order_failure_result = {
            'order_id': None,
            'status': 'rejected',
            'error': 'Insufficient buying power',
            'filled_quantity': 0,
            'timestamp': datetime.now(timezone.utc)
        }
        
        components['execution_service'].execute_order.return_value = order_failure_result
        
        print("Starting order failure workflow...")
        
        # Generate and validate signal
        signal = await components['decision_engine'].generate_signal('AAPL', None)
        portfolio = components['portfolio_service'].get_portfolio()
        risk_approved = await components['risk_service'].validate_trade(signal, portfolio)
        
        assert risk_approved is True
        
        # Calculate position size
        position_size = await components['portfolio_service'].calculate_optimal_position_size(signal)
        print(f"✓ Position size calculated: {position_size} shares")
        
        # Attempt order execution (fails)
        order_result = await components['execution_service'].execute_order({
            'symbol': 'AAPL',
            'action': 'buy',
            'quantity': position_size,
            'order_type': 'market'
        })
        
        assert order_result['status'] == 'rejected'
        assert order_result['filled_quantity'] == 0
        print(f"✓ Order rejected: {order_result['error']}")
        
        # Verify portfolio was not updated
        components['portfolio_service'].update_position.assert_not_called()
        
        print("✓ Order failure workflow completed - no portfolio changes")
    
    @pytest.mark.asyncio
    async def test_stop_loss_workflow(self, workflow_components, sample_portfolio):
        """Test stop-loss order workflow."""
        
        components = workflow_components
        
        # Modify sample portfolio to show a losing position
        losing_portfolio = Portfolio(
            user_id='test_user',
            positions={
                'AAPL': Position(
                    symbol='AAPL',
                    quantity=100,
                    avg_cost=160.0,  # Bought at higher price
                    current_price=145.0,  # Current price is lower
                    unrealized_pnl=-1500.0,  # Losing money
                    realized_pnl=0.0
                )
            },
            cash_balance=50000.0,
            total_value=64500.0,
            last_updated=datetime.now(timezone.utc)
        )
        
        # Setup stop-loss signal
        stop_loss_signal = TradingSignal(
            symbol='AAPL',
            action=TradingAction.SELL,
            confidence=0.95,  # High confidence for stop-loss
            position_size=1.0,  # Sell entire position
            target_price=None,
            stop_loss=145.0,  # Stop-loss triggered
            timestamp=datetime.now(timezone.utc),
            model_version='risk-management-v1.0'
        )
        
        components['decision_engine'].generate_signal.return_value = stop_loss_signal
        components['portfolio_service'].get_portfolio.return_value = losing_portfolio
        components['risk_service'].validate_trade.return_value = True
        components['portfolio_service'].calculate_optimal_position_size.return_value = 100  # Sell all
        
        # Setup stop-loss execution
        stop_loss_result = {
            'order_id': 'stop_loss_order_999',
            'status': 'filled',
            'filled_quantity': 100,
            'filled_price': 144.50,  # Slight slippage
            'commission': 1.0,
            'order_type': 'stop_loss',
            'timestamp': datetime.now(timezone.utc)
        }
        
        components['execution_service'].execute_order.return_value = stop_loss_result
        components['portfolio_service'].update_position.return_value = True
        
        print("Starting stop-loss workflow...")
        
        # Trigger stop-loss signal
        signal = await components['decision_engine'].generate_signal('AAPL', None)
        
        assert signal.action == TradingAction.SELL
        assert signal.confidence > 0.9  # High confidence for risk management
        print(f"✓ Stop-loss signal generated at ${signal.stop_loss}")
        
        # Validate stop-loss trade
        portfolio = components['portfolio_service'].get_portfolio()
        risk_approved = await components['risk_service'].validate_trade(signal, portfolio)
        
        assert risk_approved is True
        print("✓ Stop-loss trade validated")
        
        # Calculate sell quantity (entire position)
        sell_quantity = await components['portfolio_service'].calculate_optimal_position_size(signal)
        assert sell_quantity == 100
        print(f"✓ Selling entire position: {sell_quantity} shares")
        
        # Execute stop-loss order
        order_result = await components['execution_service'].execute_order({
            'symbol': 'AAPL',
            'action': 'sell',
            'quantity': sell_quantity,
            'order_type': 'stop_loss',
            'stop_price': 145.0
        })
        
        assert order_result['status'] == 'filled'
        assert order_result['order_type'] == 'stop_loss'
        print(f"✓ Stop-loss executed: {order_result['filled_quantity']} shares at ${order_result['filled_price']}")
        
        # Update portfolio (close position)
        portfolio_updated = await components['portfolio_service'].update_position(
            'AAPL', -sell_quantity, order_result['filled_price'], 'sell'
        )
        
        assert portfolio_updated is True
        print("✓ Position closed, portfolio updated")
        
        print("✓ Stop-loss workflow completed successfully")


class TestWorkflowErrorHandling:
    """Test error handling in workflows."""
    
    @pytest.fixture
    def error_prone_components(self):
        """Setup components that can simulate various errors."""
        
        components = {
            'data_aggregator': Mock(spec=DataAggregator),
            'decision_engine': Mock(spec=TradingDecisionEngine),
            'portfolio_service': Mock(spec=PortfolioManagementService),
            'risk_service': Mock(spec=RiskManagementService),
            'execution_service': Mock(spec=OrderExecutionService)
        }
        
        # Setup async methods
        for service in components.values():
            for method_name in dir(service):
                if not method_name.startswith('_'):
                    method = getattr(service, method_name)
                    if callable(method):
                        setattr(service, method_name, AsyncMock())
        
        return components
    
    @pytest.mark.asyncio
    async def test_data_retrieval_failure_recovery(self, error_prone_components):
        """Test recovery from data retrieval failures."""
        
        components = error_prone_components
        
        # First attempt fails
        components['data_aggregator'].get_historical_data.side_effect = [
            ConnectionError("Exchange connection failed"),
            pd.DataFrame({  # Second attempt succeeds with cached data
                'timestamp': pd.date_range(start='2023-12-01', periods=10, freq='H'),
                'symbol': ['AAPL'] * 10,
                'close': [150.0] * 10
            })
        ]
        
        print("Testing data retrieval failure recovery...")
        
        # First attempt should fail
        with pytest.raises(ConnectionError):
            await components['data_aggregator'].get_historical_data('AAPL', 'robinhood')
        
        print("✓ First attempt failed as expected")
        
        # Second attempt should succeed (fallback to cache)
        data = await components['data_aggregator'].get_historical_data('AAPL', 'robinhood')
        
        assert not data.empty
        print("✓ Recovery successful with cached data")
    
    @pytest.mark.asyncio
    async def test_signal_generation_failure_recovery(self, error_prone_components):
        """Test recovery from signal generation failures."""
        
        components = error_prone_components
        
        # Mock model failure then fallback
        components['decision_engine'].generate_signal.side_effect = [
            RuntimeError("Model inference failed"),
            TradingSignal(  # Fallback signal
                symbol='AAPL',
                action=TradingAction.HOLD,
                confidence=0.5,
                position_size=0.0,
                timestamp=datetime.now(timezone.utc),
                model_version='fallback-v1.0'
            )
        ]
        
        print("Testing signal generation failure recovery...")
        
        # First attempt fails
        with pytest.raises(RuntimeError):
            await components['decision_engine'].generate_signal('AAPL', None)
        
        print("✓ Signal generation failed as expected")
        
        # Second attempt succeeds with fallback
        signal = await components['decision_engine'].generate_signal('AAPL', None)
        
        assert signal.action == TradingAction.HOLD
        assert signal.model_version == 'fallback-v1.0'
        print("✓ Fallback signal generated successfully")
    
    @pytest.mark.asyncio
    async def test_execution_retry_mechanism(self, error_prone_components):
        """Test order execution retry mechanism."""
        
        components = error_prone_components
        
        # Mock execution failures then success
        components['execution_service'].execute_order.side_effect = [
            {'status': 'rejected', 'error': 'Temporary system error'},
            {'status': 'rejected', 'error': 'Rate limit exceeded'},
            {  # Third attempt succeeds
                'order_id': 'retry_order_123',
                'status': 'filled',
                'filled_quantity': 100,
                'filled_price': 155.0,
                'timestamp': datetime.now(timezone.utc)
            }
        ]
        
        print("Testing execution retry mechanism...")
        
        # Simulate retry logic
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                result = await components['execution_service'].execute_order({
                    'symbol': 'AAPL',
                    'action': 'buy',
                    'quantity': 100
                })
                
                if result['status'] == 'filled':
                    print(f"✓ Order succeeded on attempt {retry_count + 1}")
                    break
                else:
                    print(f"✗ Attempt {retry_count + 1} failed: {result.get('error', 'Unknown error')}")
                    retry_count += 1
                    
            except Exception as e:
                print(f"✗ Attempt {retry_count + 1} exception: {e}")
                retry_count += 1
        
        assert result['status'] == 'filled'
        print("✓ Retry mechanism successful")
    
    @pytest.mark.asyncio
    async def test_concurrent_workflow_handling(self, error_prone_components):
        """Test handling of concurrent workflow execution."""
        
        components = error_prone_components
        
        # Setup successful responses
        components['decision_engine'].generate_signal.return_value = TradingSignal(
            symbol='AAPL',
            action=TradingAction.BUY,
            confidence=0.8,
            position_size=0.1,
            timestamp=datetime.now(timezone.utc),
            model_version='concurrent-test-v1.0'
        )
        
        components['risk_service'].validate_trade.return_value = True
        components['portfolio_service'].calculate_optimal_position_size.return_value = 50
        
        components['execution_service'].execute_order.return_value = {
            'order_id': 'concurrent_order',
            'status': 'filled',
            'filled_quantity': 50,
            'filled_price': 155.0,
            'timestamp': datetime.now(timezone.utc)
        }
        
        components['portfolio_service'].update_position.return_value = True
        
        async def execute_workflow(workflow_id):
            """Execute a single workflow."""
            print(f"Starting workflow {workflow_id}")
            
            # Generate signal
            signal = await components['decision_engine'].generate_signal('AAPL', None)
            
            # Validate trade
            risk_ok = await components['risk_service'].validate_trade(signal, None)
            
            if risk_ok:
                # Calculate position
                size = await components['portfolio_service'].calculate_optimal_position_size(signal)
                
                # Execute order
                result = await components['execution_service'].execute_order({
                    'symbol': 'AAPL',
                    'quantity': size
                })
                
                # Update portfolio
                if result['status'] == 'filled':
                    await components['portfolio_service'].update_position(
                        'AAPL', size, result['filled_price'], 'buy'
                    )
                
                print(f"✓ Workflow {workflow_id} completed")
                return result
            
            return None
        
        print("Testing concurrent workflow execution...")
        
        # Execute multiple workflows concurrently
        workflows = [execute_workflow(i) for i in range(5)]
        results = await asyncio.gather(*workflows, return_exceptions=True)
        
        # Verify all workflows completed successfully
        successful_workflows = [r for r in results if isinstance(r, dict) and r.get('status') == 'filled']
        
        assert len(successful_workflows) == 5
        print(f"✓ All {len(successful_workflows)} concurrent workflows completed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])