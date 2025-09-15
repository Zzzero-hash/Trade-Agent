"""
Shared pytest fixtures for comprehensive integration tests.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from tests.test_helpers.test_data_factory import MarketDataFactory, MockExchangeFactory
from tests.test_helpers.test_config import TEST_ENV
from tests.test_helpers.workflow_strategies import WorkflowStepExecutor
from src.services.data_aggregator import DataAggregator
from src.services.trading_decision_engine import TradingDecisionEngine
from src.services.portfolio_management_service import PortfolioManagementService
from src.services.risk_management_service import RiskManagementService


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="class")
def mock_exchanges():
    """Create mock exchange connectors."""
    return MockExchangeFactory.create_mock_exchanges()


@pytest.fixture(scope="class") 
def sample_market_data():
    """Generate sample market data."""
    return MarketDataFactory.create_multi_symbol_data()


@pytest.fixture
def workflow_executor(mock_exchanges):
    """Create workflow executor."""
    aggregator = DataAggregator()
    aggregator.robinhood_connector = mock_exchanges['robinhood']
    aggregator.oanda_connector = mock_exchanges['oanda'] 
    aggregator.coinbase_connector = mock_exchanges['coinbase']
    
    return WorkflowStepExecutor(
        data_aggregator=aggregator,
        decision_engine=TradingDecisionEngine(),
        portfolio_service=PortfolioManagementService(),
        risk_service=RiskManagementService()
    )