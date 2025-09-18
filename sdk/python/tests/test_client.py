"""
Tests for the TradingPlatformClient
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from ai_trading_platform import TradingPlatformClient
from ai_trading_platform.models import TradingSignal, Portfolio, TradingAction
from ai_trading_platform.exceptions import (
    AuthenticationError,
    RateLimitError,
    ValidationError,
    NetworkError
)


@pytest.fixture
async def client():
    """Create a test client"""
    client = TradingPlatformClient("https://api.test.com")
    yield client
    await client.close()


@pytest.fixture
def mock_auth_manager():
    """Mock auth manager"""
    mock = MagicMock()
    mock.get_auth_headers = AsyncMock(return_value={"Authorization": "Bearer test-token"})
    mock.is_authenticated = MagicMock(return_value=True)
    mock.get_user_info = MagicMock(return_value={"user_id": "test-user"})
    return mock


@pytest.mark.asyncio
async def test_client_initialization():
    """Test client initialization"""
    client = TradingPlatformClient("https://api.test.com")
    assert client.base_url == "https://api.test.com"
    assert client.timeout == 30.0
    await client.close()


@pytest.mark.asyncio
async def test_context_manager():
    """Test client as context manager"""
    async with TradingPlatformClient("https://api.test.com") as client:
        assert client._http_client is not None


@pytest.mark.asyncio
async def test_login_with_api_key(client):
    """Test API key authentication"""
    with patch.object(client.auth, 'authenticate_with_api_key') as mock_auth:
        mock_auth.return_value = {"access_token": "test-token"}
        
        result = await client.login_with_api_key("test-api-key")
        
        mock_auth.assert_called_once_with("test-api-key")
        assert result["access_token"] == "test-token"


@pytest.mark.asyncio
async def test_login_with_credentials(client):
    """Test username/password authentication"""
    with patch.object(client.auth, 'authenticate_with_credentials') as mock_auth:
        mock_auth.return_value = {"access_token": "test-token"}
        
        result = await client.login("username", "password")
        
        mock_auth.assert_called_once_with("username", "password")
        assert result["access_token"] == "test-token"


@pytest.mark.asyncio
async def test_generate_signal(client):
    """Test signal generation"""
    expected_signal = TradingSignal(
        id="test-signal",
        symbol="AAPL",
        action=TradingAction.BUY,
        confidence=0.85,
        timestamp=datetime.now()
    )
    
    with patch.object(client, '_make_request') as mock_request:
        mock_request.return_value = expected_signal.model_dump()
        
        signal = await client.generate_signal("AAPL")
        
        mock_request.assert_called_once_with(
            "POST", 
            "/api/v1/trading/signals/generate?symbol=AAPL"
        )
        assert signal.symbol == "AAPL"
        assert signal.action == TradingAction.BUY
        assert signal.confidence == 0.85


@pytest.mark.asyncio
async def test_get_portfolio(client):
    """Test portfolio retrieval"""
    expected_portfolio = Portfolio(
        user_id="test-user",
        positions={},
        cash_balance=10000.0,
        total_value=10000.0,
        total_pnl=0.0,
        total_pnl_percent=0.0,
        last_updated=datetime.now()
    )
    
    with patch.object(client, '_make_request') as mock_request:
        mock_request.return_value = expected_portfolio.model_dump()
        
        portfolio = await client.get_portfolio()
        
        mock_request.assert_called_once_with("GET", "/api/v1/trading/portfolio")
        assert portfolio.user_id == "test-user"
        assert portfolio.total_value == 10000.0


@pytest.mark.asyncio
async def test_rebalance_portfolio(client):
    """Test portfolio rebalancing"""
    target_allocation = {"AAPL": 0.5, "GOOGL": 0.3, "CASH": 0.2}
    expected_result = {"rebalance_id": "test-rebalance", "status": "initiated"}
    
    with patch.object(client, '_make_request') as mock_request:
        mock_request.return_value = expected_result
        
        result = await client.rebalance_portfolio(target_allocation)
        
        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/trading/portfolio/rebalance",
            json_data=target_allocation
        )
        assert result["rebalance_id"] == "test-rebalance"


@pytest.mark.asyncio
async def test_get_market_data(client):
    """Test market data retrieval"""
    expected_data = {
        "symbol": "AAPL",
        "timeframe": "1h",
        "data": [
            {"timestamp": "2024-01-01T10:00:00", "close": 150.0},
            {"timestamp": "2024-01-01T11:00:00", "close": 151.0}
        ]
    }
    
    with patch.object(client, '_make_request') as mock_request:
        mock_request.return_value = expected_data
        
        data = await client.get_market_data("AAPL", "1h", 100)
        
        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/trading/market-data/AAPL",
            params={"timeframe": "1h", "limit": 100}
        )
        assert data["symbol"] == "AAPL"
        assert len(data["data"]) == 2


@pytest.mark.asyncio
async def test_authentication_error(client):
    """Test authentication error handling"""
    with patch.object(client, '_make_request') as mock_request:
        mock_request.side_effect = AuthenticationError("Invalid token")
        
        with pytest.raises(AuthenticationError):
            await client.generate_signal("AAPL")


@pytest.mark.asyncio
async def test_rate_limit_error(client):
    """Test rate limit error handling"""
    with patch.object(client, '_make_request') as mock_request:
        mock_request.side_effect = RateLimitError("Rate limited", retry_after=60)
        
        with pytest.raises(RateLimitError) as exc_info:
            await client.generate_signal("AAPL")
        
        assert exc_info.value.retry_after == 60


@pytest.mark.asyncio
async def test_validation_error(client):
    """Test validation error handling"""
    validation_errors = [{"field": "symbol", "message": "Required field"}]
    
    with patch.object(client, '_make_request') as mock_request:
        mock_request.side_effect = ValidationError(
            "Validation failed", 
            validation_errors=validation_errors
        )
        
        with pytest.raises(ValidationError) as exc_info:
            await client.generate_signal("")
        
        assert exc_info.value.validation_errors == validation_errors


@pytest.mark.asyncio
async def test_network_error(client):
    """Test network error handling"""
    with patch.object(client, '_make_request') as mock_request:
        mock_request.side_effect = NetworkError("Connection failed")
        
        with pytest.raises(NetworkError):
            await client.generate_signal("AAPL")


@pytest.mark.asyncio
async def test_websocket_connection(client):
    """Test WebSocket connection"""
    with patch.object(client, 'connect_websocket') as mock_connect:
        mock_ws = AsyncMock()
        mock_connect.return_value = mock_ws
        
        ws = await client.connect_websocket()
        
        assert ws == mock_ws
        mock_connect.assert_called_once()


@pytest.mark.asyncio
async def test_signal_subscription(client):
    """Test signal subscription"""
    mock_signals = [
        {"type": "signal_generated", "data": {"symbol": "AAPL", "action": "BUY"}},
        {"type": "signal_generated", "data": {"symbol": "GOOGL", "action": "SELL"}}
    ]
    
    async def mock_subscribe():
        for signal in mock_signals:
            yield signal
    
    with patch.object(client, 'subscribe_to_signals', side_effect=mock_subscribe):
        signals = []
        async for signal in client.subscribe_to_signals():
            signals.append(signal)
            if len(signals) >= 2:
                break
        
        assert len(signals) == 2
        assert signals[0]["data"]["symbol"] == "AAPL"
        assert signals[1]["data"]["symbol"] == "GOOGL"


@pytest.mark.asyncio
async def test_market_data_subscription(client):
    """Test market data subscription"""
    symbols = ["AAPL", "GOOGL"]
    mock_data = [
        {"type": "market_data", "data": {"symbol": "AAPL", "price": 150.0}},
        {"type": "market_data", "data": {"symbol": "GOOGL", "price": 2800.0}}
    ]
    
    async def mock_subscribe(symbols_list):
        for data in mock_data:
            yield data
    
    with patch.object(client, 'subscribe_to_market_data', side_effect=mock_subscribe):
        data_points = []
        async for data in client.subscribe_to_market_data(symbols):
            data_points.append(data)
            if len(data_points) >= 2:
                break
        
        assert len(data_points) == 2
        assert data_points[0]["data"]["symbol"] == "AAPL"
        assert data_points[1]["data"]["symbol"] == "GOOGL"


@pytest.mark.asyncio
async def test_health_check(client):
    """Test health check"""
    expected_health = {
        "status": "healthy",
        "timestamp": "2024-01-01T12:00:00Z",
        "version": "1.0.0"
    }
    
    with patch.object(client, '_make_request') as mock_request:
        mock_request.return_value = expected_health
        
        health = await client.health_check()
        
        mock_request.assert_called_once_with(
            "GET", 
            "/api/v1/trading/health", 
            require_auth=False
        )
        assert health["status"] == "healthy"


@pytest.mark.asyncio
async def test_user_info(client):
    """Test user info retrieval"""
    with patch.object(client.auth, 'get_user_info') as mock_get_user_info:
        mock_get_user_info.return_value = {
            "user_id": "test-user",
            "username": "testuser",
            "email": "test@example.com",
            "tier": "premium"
        }
        
        user_info = client.get_user_info()
        
        assert user_info["user_id"] == "test-user"
        assert user_info["tier"] == "premium"


@pytest.mark.asyncio
async def test_is_authenticated(client):
    """Test authentication status check"""
    with patch.object(client.auth, 'is_authenticated') as mock_is_auth:
        mock_is_auth.return_value = True
        
        assert client.is_authenticated() is True
        
        mock_is_auth.return_value = False
        assert client.is_authenticated() is False


@pytest.mark.asyncio
async def test_oauth_url_generation(client):
    """Test OAuth URL generation"""
    with patch.object(client.auth, 'get_oauth_authorization_url') as mock_get_url:
        mock_get_url.return_value = "https://api.test.com/oauth/authorize?client_id=test"
        
        url = client.get_oauth_url("http://localhost:3000/callback")
        
        mock_get_url.assert_called_once_with("http://localhost:3000/callback", None, None)
        assert "oauth/authorize" in url


@pytest.mark.asyncio
async def test_logout(client):
    """Test logout"""
    with patch.object(client.auth, 'logout') as mock_logout:
        await client.logout()
        mock_logout.assert_called_once()


# Integration tests (these would require a test server)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_workflow():
    """Test complete workflow (requires test server)"""
    # This test would require a running test server
    # and would test the complete workflow from authentication
    # to signal generation and portfolio management
    pass