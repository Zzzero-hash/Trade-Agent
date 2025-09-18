"""Integration tests for real money trading system"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
import json

from src.exchanges.broker_base import (
    BrokerType, BrokerCredentials, TradingOrder, OrderResult, OrderStatus,
    OrderSide, OrderType, TimeInForce, Position, AccountInfo
)
from src.exchanges.robinhood import RobinhoodConnector
from src.exchanges.td_ameritrade_connector import TDAmeritradeBrokerConnector
from src.exchanges.interactive_brokers_connector import InteractiveBrokersConnector
from src.services.oauth_token_manager import OAuthTokenManager
from src.services.order_management_system import OrderManagementSystem, RoutingStrategy
from src.services.position_sync_service import PositionSyncService
from src.services.trade_confirmation_service import TradeConfirmationService
from src.services.encryption_service import EncryptionService


@pytest.fixture
async def encryption_service():
    """Create encryption service for testing"""
    service = EncryptionService()
    await service.initialize()
    return service


@pytest.fixture
async def token_manager(encryption_service):
    """Create OAuth token manager for testing"""
    return OAuthTokenManager(encryption_service, storage_path="test_tokens")


@pytest.fixture
def robinhood_credentials():
    """Create Robinhood credentials for testing"""
    return BrokerCredentials(
        broker_type=BrokerType.ROBINHOOD,
        client_id="test_client_id",
        username="test_user",
        password="test_password"
    )


@pytest.fixture
def td_ameritrade_credentials():
    """Create TD Ameritrade credentials for testing"""
    return BrokerCredentials(
        broker_type=BrokerType.TD_AMERITRADE,
        client_id="test_client_id",
        client_secret="test_client_secret"
    )


@pytest.fixture
def interactive_brokers_credentials():
    """Create Interactive Brokers credentials for testing"""
    return BrokerCredentials(
        broker_type=BrokerType.INTERACTIVE_BROKERS,
        client_id="test_client_id",
        username="test_user"
    )


@pytest.fixture
async def robinhood_connector(robinhood_credentials):
    """Create Robinhood connector for testing"""
    connector = RobinhoodConnector(
        username=robinhood_credentials.username,
        password=robinhood_credentials.password,
        sandbox=True
    )
    return connector


@pytest.fixture
async def td_ameritrade_connector(td_ameritrade_credentials):
    """Create TD Ameritrade connector for testing"""
    return TDAmeritradeBrokerConnector(td_ameritrade_credentials, sandbox=True)


@pytest.fixture
async def interactive_brokers_connector(interactive_brokers_credentials):
    """Create Interactive Brokers connector for testing"""
    return InteractiveBrokersConnector(interactive_brokers_credentials, sandbox=True)


@pytest.fixture
async def order_management_system(token_manager):
    """Create order management system for testing"""
    oms = OrderManagementSystem(token_manager)
    await oms.start()
    yield oms
    await oms.stop()


@pytest.fixture
async def position_sync_service(token_manager):
    """Create position sync service for testing"""
    service = PositionSyncService(token_manager)
    await service.start()
    yield service
    await service.stop()


@pytest.fixture
async def trade_confirmation_service(token_manager):
    """Create trade confirmation service for testing"""
    service = TradeConfirmationService(token_manager)
    await service.start()
    yield service
    await service.stop()


class TestBrokerConnectors:
    """Test broker connector implementations"""
    
    @pytest.mark.asyncio
    async def test_robinhood_connection(self, robinhood_connector):
        """Test Robinhood connection and authentication"""
        with patch.object(robinhood_connector, 'connection_pool') as mock_pool:
            mock_client = AsyncMock()
            mock_pool.get_client.return_value = mock_client
            
            # Mock successful authentication response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "access_token": "test_token",
                "refresh_token": "test_refresh_token"
            }
            mock_client.post.return_value = mock_response
            
            # Test connection
            result = await robinhood_connector.connect()
            assert result is True
            assert robinhood_connector.is_connected is True
            assert robinhood_connector.is_authenticated is True
    
    @pytest.mark.asyncio
    async def test_td_ameritrade_authentication(self, td_ameritrade_connector):
        """Test TD Ameritrade OAuth authentication"""
        with patch.object(td_ameritrade_connector, 'client') as mock_client:
            # Mock token validation response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response
            
            # Set up credentials
            td_ameritrade_connector.credentials.access_token = "test_token"
            td_ameritrade_connector.credentials.token_expires_at = datetime.now() + timedelta(hours=1)
            
            # Test authentication
            result = await td_ameritrade_connector.authenticate()
            assert result is True
            assert td_ameritrade_connector.is_authenticated is True
    
    @pytest.mark.asyncio
    async def test_interactive_brokers_connection(self, interactive_brokers_connector):
        """Test Interactive Brokers connection"""
        with patch.object(interactive_brokers_connector, 'client') as mock_client:
            # Mock authentication status response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"authenticated": True}
            mock_client.get.return_value = mock_response
            
            # Test connection
            result = await interactive_brokers_connector.connect()
            assert result is True
            assert interactive_brokers_connector.is_connected is True
            assert interactive_brokers_connector.is_authenticated is True


class TestOrderManagementSystem:
    """Test order management system functionality"""
    
    @pytest.mark.asyncio
    async def test_order_placement_with_routing(self, order_management_system, robinhood_connector):
        """Test order placement with smart routing"""
        # Register broker
        order_management_system.register_broker(BrokerType.ROBINHOOD, robinhood_connector)
        
        # Mock broker methods
        robinhood_connector.is_connected = True
        robinhood_connector.is_authenticated = True
        robinhood_connector.get_supported_symbols = Mock(return_value=["AAPL"])
        robinhood_connector.get_supported_order_types = Mock(return_value=[OrderType.MARKET, OrderType.LIMIT])
        
        # Mock order placement
        expected_result = OrderResult(
            order_id="test_order_123",
            broker_order_id="broker_123",
            status=OrderStatus.PENDING,
            filled_quantity=Decimal('0'),
            remaining_quantity=Decimal('100'),
            avg_fill_price=None,
            total_commission=None,
            total_fees=None,
            timestamp=datetime.now()
        )
        robinhood_connector.place_order = AsyncMock(return_value=expected_result)
        
        # Create test order
        order = TradingOrder(
            order_id="test_order_123",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal('100'),
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY
        )
        
        # Place order
        order_id = await order_management_system.place_order(order, RoutingStrategy.BEST_EXECUTION)
        
        assert order_id == "test_order_123"
        robinhood_connector.place_order.assert_called_once()
        
        # Check order tracking
        tracker = await order_management_system.get_order_status(order_id)
        assert tracker is not None
        assert tracker.broker_type == BrokerType.ROBINHOOD
        assert tracker.status == OrderStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_order_cancellation(self, order_management_system, robinhood_connector):
        """Test order cancellation"""
        # Register broker and place order first
        order_management_system.register_broker(BrokerType.ROBINHOOD, robinhood_connector)
        
        # Mock broker methods
        robinhood_connector.is_connected = True
        robinhood_connector.is_authenticated = True
        robinhood_connector.get_supported_symbols = Mock(return_value=["AAPL"])
        robinhood_connector.get_supported_order_types = Mock(return_value=[OrderType.MARKET])
        robinhood_connector.place_order = AsyncMock(return_value=OrderResult(
            order_id="test_order_123",
            broker_order_id="broker_123",
            status=OrderStatus.PENDING,
            filled_quantity=Decimal('0'),
            remaining_quantity=Decimal('100'),
            avg_fill_price=None,
            total_commission=None,
            total_fees=None,
            timestamp=datetime.now()
        ))
        robinhood_connector.cancel_order = AsyncMock(return_value=True)
        
        # Place order
        order = TradingOrder(
            order_id="test_order_123",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal('100'),
            order_type=OrderType.MARKET
        )
        order_id = await order_management_system.place_order(order)
        
        # Cancel order
        result = await order_management_system.cancel_order(order_id)
        
        assert result is True
        robinhood_connector.cancel_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_multi_broker_routing(self, order_management_system, robinhood_connector, td_ameritrade_connector):
        """Test routing between multiple brokers"""
        # Register multiple brokers
        order_management_system.register_broker(BrokerType.ROBINHOOD, robinhood_connector)
        order_management_system.register_broker(BrokerType.TD_AMERITRADE, td_ameritrade_connector)
        
        # Mock broker availability
        robinhood_connector.is_connected = True
        robinhood_connector.is_authenticated = True
        robinhood_connector.get_supported_symbols = Mock(return_value=["AAPL"])
        robinhood_connector.get_supported_order_types = Mock(return_value=[OrderType.MARKET])
        
        td_ameritrade_connector.is_connected = True
        td_ameritrade_connector.is_authenticated = True
        td_ameritrade_connector.get_supported_symbols = Mock(return_value=["AAPL"])
        td_ameritrade_connector.get_supported_order_types = Mock(return_value=[OrderType.MARKET])
        
        # Mock order placement (TD Ameritrade should be selected for best execution)
        td_ameritrade_connector.place_order = AsyncMock(return_value=OrderResult(
            order_id="test_order_123",
            broker_order_id="td_broker_123",
            status=OrderStatus.PENDING,
            filled_quantity=Decimal('0'),
            remaining_quantity=Decimal('100'),
            avg_fill_price=None,
            total_commission=None,
            total_fees=None,
            timestamp=datetime.now()
        ))
        
        # Set up execution metrics to favor TD Ameritrade
        order_management_system.execution_metrics[BrokerType.TD_AMERITRADE].fill_rate = 0.95
        order_management_system.execution_metrics[BrokerType.TD_AMERITRADE].success_rate = 0.98
        order_management_system.execution_metrics[BrokerType.ROBINHOOD].fill_rate = 0.85
        order_management_system.execution_metrics[BrokerType.ROBINHOOD].success_rate = 0.90
        
        # Place order with best execution routing
        order = TradingOrder(
            order_id="test_order_123",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal('100'),
            order_type=OrderType.MARKET
        )
        
        order_id = await order_management_system.place_order(order, RoutingStrategy.BEST_EXECUTION)
        
        # Verify TD Ameritrade was selected
        tracker = await order_management_system.get_order_status(order_id)
        assert tracker.broker_type == BrokerType.TD_AMERITRADE
        td_ameritrade_connector.place_order.assert_called_once()


class TestPositionSyncService:
    """Test position synchronization service"""
    
    @pytest.mark.asyncio
    async def test_position_synchronization(self, position_sync_service, robinhood_connector):
        """Test position synchronization from broker"""
        # Register broker
        position_sync_service.register_broker(BrokerType.ROBINHOOD, robinhood_connector)
        position_sync_service.map_account("test_account", BrokerType.ROBINHOOD, "rh_account_123")
        
        # Mock broker connection
        robinhood_connector.is_connected = True
        
        # Mock positions from broker
        mock_positions = [
            Position(
                symbol="AAPL",
                quantity=Decimal('100'),
                average_cost=Decimal('150.00'),
                market_value=Decimal('15500.00'),
                unrealized_pnl=Decimal('500.00'),
                realized_pnl=Decimal('0'),
                account_id="test_account",
                broker_type=BrokerType.ROBINHOOD
            ),
            Position(
                symbol="GOOGL",
                quantity=Decimal('50'),
                average_cost=Decimal('2800.00'),
                market_value=Decimal('142000.00'),
                unrealized_pnl=Decimal('2000.00'),
                realized_pnl=Decimal('0'),
                account_id="test_account",
                broker_type=BrokerType.ROBINHOOD
            )
        ]
        robinhood_connector.get_positions = AsyncMock(return_value=mock_positions)
        
        # Sync positions
        result = await position_sync_service.sync_positions_now(BrokerType.ROBINHOOD)
        
        assert result[BrokerType.ROBINHOOD] is True
        
        # Verify positions were synced
        synced_positions = await position_sync_service.get_positions("test_account", "broker")
        assert len(synced_positions) == 2
        assert "AAPL" in synced_positions
        assert "GOOGL" in synced_positions
        assert synced_positions["AAPL"].quantity == Decimal('100')
    
    @pytest.mark.asyncio
    async def test_position_discrepancy_detection(self, position_sync_service, robinhood_connector):
        """Test position discrepancy detection"""
        # Register broker and account
        position_sync_service.register_broker(BrokerType.ROBINHOOD, robinhood_connector)
        position_sync_service.map_account("test_account", BrokerType.ROBINHOOD, "rh_account_123")
        
        # Set up local position
        local_position = Position(
            symbol="AAPL",
            quantity=Decimal('100'),
            average_cost=Decimal('150.00'),
            market_value=Decimal('15000.00'),
            unrealized_pnl=Decimal('0'),
            realized_pnl=Decimal('0'),
            account_id="test_account",
            broker_type=BrokerType.ROBINHOOD
        )
        await position_sync_service.update_local_position("test_account", local_position)
        
        # Mock broker position with discrepancy
        broker_position = Position(
            symbol="AAPL",
            quantity=Decimal('95'),  # 5 shares difference
            average_cost=Decimal('150.00'),
            market_value=Decimal('14725.00'),
            unrealized_pnl=Decimal('225.00'),
            realized_pnl=Decimal('0'),
            account_id="test_account",
            broker_type=BrokerType.ROBINHOOD
        )
        
        robinhood_connector.is_connected = True
        robinhood_connector.get_positions = AsyncMock(return_value=[broker_position])
        
        # Sync and trigger reconciliation
        await position_sync_service.sync_positions_now(BrokerType.ROBINHOOD)
        await position_sync_service._reconcile_positions()
        
        # Check for discrepancies
        discrepancies = await position_sync_service.get_discrepancies("test_account")
        assert len(discrepancies) == 1
        assert discrepancies[0].symbol == "AAPL"
        assert discrepancies[0].quantity_diff == Decimal('5')  # 100 - 95


class TestTradeConfirmationService:
    """Test trade confirmation and settlement tracking"""
    
    @pytest.mark.asyncio
    async def test_trade_recording(self, trade_confirmation_service, robinhood_connector):
        """Test trade recording for confirmation tracking"""
        # Register broker
        trade_confirmation_service.register_broker(BrokerType.ROBINHOOD, robinhood_connector)
        
        # Create order result
        order_result = OrderResult(
            order_id="test_order_123",
            broker_order_id="broker_123",
            status=OrderStatus.FILLED,
            filled_quantity=Decimal('100'),
            remaining_quantity=Decimal('0'),
            avg_fill_price=Decimal('155.00'),
            total_commission=Decimal('1.00'),
            total_fees=Decimal('0.50'),
            timestamp=datetime.now()
        )
        
        # Record trade
        trade_id = await trade_confirmation_service.record_trade(
            order_result, "test_account", BrokerType.ROBINHOOD
        )
        
        assert trade_id is not None
        
        # Verify trade record
        trade_record = await trade_confirmation_service.get_trade_record(trade_id)
        assert trade_record is not None
        assert trade_record.order_id == "test_order_123"
        assert trade_record.quantity == Decimal('100')
        assert trade_record.settlement_info.status.value == "pending"
    
    @pytest.mark.asyncio
    async def test_confirmation_matching(self, trade_confirmation_service, robinhood_connector):
        """Test trade confirmation matching"""
        # Register broker
        trade_confirmation_service.register_broker(BrokerType.ROBINHOOD, robinhood_connector)
        
        # Record a trade
        order_result = OrderResult(
            order_id="test_order_123",
            broker_order_id="broker_123",
            status=OrderStatus.FILLED,
            filled_quantity=Decimal('100'),
            remaining_quantity=Decimal('0'),
            avg_fill_price=Decimal('155.00'),
            total_commission=Decimal('1.00'),
            total_fees=Decimal('0.50'),
            timestamp=datetime.now()
        )
        
        trade_id = await trade_confirmation_service.record_trade(
            order_result, "test_account", BrokerType.ROBINHOOD
        )
        
        # Mock broker confirmation
        from src.exchanges.broker_base import TradeConfirmation
        mock_confirmation = TradeConfirmation(
            confirmation_id="conf_123",
            order_id="test_order_123",
            trade_date=datetime.now(),
            settlement_date=(datetime.now() + timedelta(days=2)).date(),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal('100'),
            price=Decimal('155.00'),
            gross_amount=Decimal('15500.00'),
            commission=Decimal('1.00'),
            fees=Decimal('0.50'),
            net_amount=Decimal('-15501.50'),
            account_id="test_account",
            broker_type=BrokerType.ROBINHOOD
        )
        
        robinhood_connector.is_connected = True
        robinhood_connector.get_trade_confirmations = AsyncMock(return_value=[mock_confirmation])
        
        # Force confirmation check
        matched_count = await trade_confirmation_service.force_confirmation_check(BrokerType.ROBINHOOD)
        
        assert matched_count == 1
        
        # Verify trade was confirmed
        trade_record = await trade_confirmation_service.get_trade_record(trade_id)
        assert trade_record.confirmation_status.value == "confirmed"
        assert trade_record.confirmation_id == "conf_123"


class TestOAuthTokenManager:
    """Test OAuth token management"""
    
    @pytest.mark.asyncio
    async def test_token_storage_and_retrieval(self, token_manager):
        """Test token storage and retrieval"""
        from src.services.oauth_token_manager import TokenInfo
        
        # Create test token
        token_info = TokenInfo(
            access_token="test_access_token",
            refresh_token="test_refresh_token",
            expires_at=datetime.now() + timedelta(hours=1),
            token_type="Bearer",
            scope="trading"
        )
        
        # Store token
        result = await token_manager.store_token("test_account", BrokerType.ROBINHOOD, token_info)
        assert result is True
        
        # Retrieve token
        retrieved_token = await token_manager.get_token("test_account", BrokerType.ROBINHOOD)
        assert retrieved_token is not None
        assert retrieved_token.access_token == "test_access_token"
        assert retrieved_token.refresh_token == "test_refresh_token"
    
    @pytest.mark.asyncio
    async def test_token_refresh(self, token_manager):
        """Test automatic token refresh"""
        from src.services.oauth_token_manager import TokenInfo
        
        # Register refresh callback
        async def mock_refresh_callback(refresh_token):
            return {
                "access_token": "new_access_token",
                "refresh_token": "new_refresh_token",
                "expires_in": 3600,
                "token_type": "Bearer"
            }
        
        token_manager.register_refresh_callback(BrokerType.ROBINHOOD, mock_refresh_callback)
        
        # Create expired token
        expired_token = TokenInfo(
            access_token="old_access_token",
            refresh_token="old_refresh_token",
            expires_at=datetime.now() - timedelta(minutes=1),  # Expired
            token_type="Bearer"
        )
        
        # Store expired token
        await token_manager.store_token("test_account", BrokerType.ROBINHOOD, expired_token)
        
        # Retrieve token (should trigger refresh)
        refreshed_token = await token_manager.get_token("test_account", BrokerType.ROBINHOOD)
        
        assert refreshed_token is not None
        assert refreshed_token.access_token == "new_access_token"
        assert refreshed_token.refresh_token == "new_refresh_token"


class TestIntegrationWorkflows:
    """Test complete integration workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_trading_workflow(
        self, 
        order_management_system, 
        position_sync_service, 
        trade_confirmation_service,
        robinhood_connector
    ):
        """Test complete trading workflow from order to settlement"""
        # Register broker in all services
        order_management_system.register_broker(BrokerType.ROBINHOOD, robinhood_connector)
        position_sync_service.register_broker(BrokerType.ROBINHOOD, robinhood_connector)
        position_sync_service.map_account("test_account", BrokerType.ROBINHOOD, "rh_account_123")
        trade_confirmation_service.register_broker(BrokerType.ROBINHOOD, robinhood_connector)
        
        # Mock broker methods
        robinhood_connector.is_connected = True
        robinhood_connector.is_authenticated = True
        robinhood_connector.get_supported_symbols = Mock(return_value=["AAPL"])
        robinhood_connector.get_supported_order_types = Mock(return_value=[OrderType.MARKET])
        
        # Mock order placement and execution
        order_result = OrderResult(
            order_id="test_order_123",
            broker_order_id="broker_123",
            status=OrderStatus.FILLED,
            filled_quantity=Decimal('100'),
            remaining_quantity=Decimal('0'),
            avg_fill_price=Decimal('155.00'),
            total_commission=Decimal('1.00'),
            total_fees=Decimal('0.50'),
            timestamp=datetime.now()
        )
        robinhood_connector.place_order = AsyncMock(return_value=order_result)
        
        # Step 1: Place order
        order = TradingOrder(
            order_id="test_order_123",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal('100'),
            order_type=OrderType.MARKET,
            account_id="test_account"
        )
        
        order_id = await order_management_system.place_order(order)
        assert order_id == "test_order_123"
        
        # Step 2: Record trade for confirmation
        trade_id = await trade_confirmation_service.record_trade(
            order_result, "test_account", BrokerType.ROBINHOOD
        )
        assert trade_id is not None
        
        # Step 3: Update positions
        new_position = Position(
            symbol="AAPL",
            quantity=Decimal('100'),
            average_cost=Decimal('155.00'),
            market_value=Decimal('15500.00'),
            unrealized_pnl=Decimal('0'),
            realized_pnl=Decimal('0'),
            account_id="test_account",
            broker_type=BrokerType.ROBINHOOD
        )
        
        robinhood_connector.get_positions = AsyncMock(return_value=[new_position])
        await position_sync_service.sync_positions_now(BrokerType.ROBINHOOD)
        
        # Verify position was synced
        positions = await position_sync_service.get_positions("test_account", "broker")
        assert len(positions) == 1
        assert positions["AAPL"].quantity == Decimal('100')
        
        # Step 4: Verify trade confirmation tracking
        trade_record = await trade_confirmation_service.get_trade_record(trade_id)
        assert trade_record is not None
        assert trade_record.quantity == Decimal('100')
        assert trade_record.settlement_info.status.value == "pending"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])