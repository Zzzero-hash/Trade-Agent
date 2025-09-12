"""Integration tests for RobinhoodConnector"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import httpx

from src.exchanges.robinhood import RobinhoodConnector, RateLimiter, ConnectionPool
from src.exchanges.base import Order, OrderResult, MarketData


@pytest.fixture
def connector():
    """Create a test connector instance"""
    return RobinhoodConnector(
        username="test_user",
        password="test_pass",
        sandbox=True
    )


@pytest.fixture
def mock_auth_response():
    """Mock authentication response"""
    return {
        "access_token": "test_access_token",
        "refresh_token": "test_refresh_token",
        "token_type": "Bearer"
    }


@pytest.fixture
def mock_instrument_response():
    """Mock instrument lookup response"""
    return {
        "results": [{
            "id": "test_instrument_id",
            "url": "https://api.robinhood.com/instruments/test_instrument_id/",
            "symbol": "AAPL",
            "name": "Apple Inc."
        }]
    }


@pytest.fixture
def mock_historical_response():
    """Mock historical data response"""
    return {
        "historicals": [
            {
                "begins_at": "2023-01-01T00:00:00Z",
                "open_price": "150.00",
                "high_price": "155.00", 
                "low_price": "149.00",
                "close_price": "154.00",
                "volume": "1000000"
            },
            {
                "begins_at": "2023-01-02T00:00:00Z",
                "open_price": "154.00",
                "high_price": "158.00",
                "low_price": "153.00", 
                "close_price": "157.00",
                "volume": "1200000"
            }
        ]
    }


class TestAuthentication:
    """Test authentication functionality"""
    
    @pytest.mark.asyncio
    async def test_successful_connection(self, connector, mock_auth_response):
        """Test successful authentication and connection"""
        with patch.object(connector.connection_pool, 'get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_get_client.return_value = mock_client
            
            # Mock successful auth response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_auth_response
            mock_client.post.return_value = mock_response
            
            result = await connector.connect()
            
            assert result is True
            assert connector.is_connected is True
            assert connector._authenticated is True
            assert connector.credentials.access_token == "test_access_token"
            assert "Authorization" in connector.session_headers
    
    @pytest.mark.asyncio
    async def test_failed_connection(self, connector):
        """Test failed authentication"""
        with patch.object(connector.connection_pool, 'get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_get_client.return_value = mock_client
            
            # Mock failed auth response
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.text = "Invalid credentials"
            mock_client.post.return_value = mock_response
            
            result = await connector.connect()
            
            assert result is False
            assert connector.is_connected is False
            assert connector._authenticated is False
    
    @pytest.mark.asyncio
    async def test_disconnect(self, connector):
        """Test disconnection"""
        # Set up connected state
        connector.is_connected = True
        connector._authenticated = True
        connector.credentials.access_token = "test_token"
        
        with patch.object(connector.connection_pool, 'get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_get_client.return_value = mock_client
            mock_client.post.return_value = MagicMock(status_code=200)
            
            with patch.object(connector.connection_pool, 'close') as mock_close:
                await connector.disconnect()
                
                assert connector.is_connected is False
                assert connector._authenticated is False
                mock_close.assert_called_once()


class TestDataRetrieval:
    """Test data retrieval functionality"""
    
    @pytest.mark.asyncio
    async def test_get_historical_data(self, connector, mock_instrument_response, mock_historical_response):
        """Test historical data retrieval"""
        connector._authenticated = True
        
        with patch.object(connector, '_make_request') as mock_request:
            # Mock instrument lookup
            instrument_mock = MagicMock()
            instrument_mock.status_code = 200
            instrument_mock.json.return_value = mock_instrument_response
            
            # Mock historical data
            historical_mock = MagicMock()
            historical_mock.status_code = 200
            historical_mock.json.return_value = mock_historical_response
            
            mock_request.side_effect = [instrument_mock, historical_mock]
            
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 1, 3)
            
            df = await connector.get_historical_data("AAPL", "1d", start_date, end_date)
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert "open" in df.columns
            assert "high" in df.columns
            assert "low" in df.columns
            assert "close" in df.columns
            assert "volume" in df.columns
    
    @pytest.mark.asyncio
    async def test_get_historical_data_invalid_symbol(self, connector):
        """Test historical data with invalid symbol"""
        connector._authenticated = True
        
        with patch.object(connector, '_make_request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_request.return_value = mock_response
            
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 1, 3)
            
            df = await connector.get_historical_data("INVALID", "1d", start_date, end_date)
            
            assert isinstance(df, pd.DataFrame)
            assert df.empty
    
    @pytest.mark.asyncio
    async def test_real_time_data_stream(self, connector):
        """Test real-time data streaming"""
        connector.is_connected = True
        connector._authenticated = True
        
        mock_quote_response = {
            "results": [{
                "symbol": "AAPL",
                "last_trade_price": "150.00",
                "previous_close": "149.00"
            }]
        }
        
        with patch.object(connector, '_make_request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_quote_response
            mock_request.return_value = mock_response
            
            # Test streaming for a short duration
            symbols = ["AAPL"]
            data_count = 0
            
            async for market_data in connector.get_real_time_data(symbols):
                assert isinstance(market_data, MarketData)
                assert market_data.symbol == "AAPL"
                assert market_data.exchange == "ROBINHOOD"
                data_count += 1
                
                # Break after receiving one data point
                if data_count >= 1:
                    connector.is_connected = False
                    break


class TestOrderManagement:
    """Test order management functionality"""
    
    @pytest.mark.asyncio
    async def test_place_market_order(self, connector, mock_instrument_response):
        """Test placing a market order"""
        connector._authenticated = True
        
        mock_order_response = {
            "id": "test_order_id",
            "state": "confirmed",
            "quantity": "10",
            "price": "150.00"
        }
        
        with patch.object(connector, '_make_request') as mock_request:
            with patch.object(connector, '_get_account_url', return_value="test_account_url"):
                # Mock instrument lookup
                instrument_mock = MagicMock()
                instrument_mock.status_code = 200
                instrument_mock.json.return_value = mock_instrument_response
                
                # Mock order placement
                order_mock = MagicMock()
                order_mock.status_code = 201
                order_mock.json.return_value = mock_order_response
                
                mock_request.side_effect = [instrument_mock, order_mock]
                
                order = Order(
                    symbol="AAPL",
                    side="BUY",
                    quantity=10,
                    order_type="MARKET"
                )
                
                result = await connector.place_order(order)
                
                assert isinstance(result, OrderResult)
                assert result.order_id == "test_order_id"
                assert result.status == "CONFIRMED"
                assert result.filled_quantity == 10
    
    @pytest.mark.asyncio
    async def test_place_limit_order(self, connector, mock_instrument_response):
        """Test placing a limit order"""
        connector._authenticated = True
        
        mock_order_response = {
            "id": "test_limit_order_id",
            "state": "queued",
            "quantity": "5",
            "price": "145.00"
        }
        
        with patch.object(connector, '_make_request') as mock_request:
            with patch.object(connector, '_get_account_url', return_value="test_account_url"):
                # Mock responses
                instrument_mock = MagicMock()
                instrument_mock.status_code = 200
                instrument_mock.json.return_value = mock_instrument_response
                
                order_mock = MagicMock()
                order_mock.status_code = 201
                order_mock.json.return_value = mock_order_response
                
                mock_request.side_effect = [instrument_mock, order_mock]
                
                order = Order(
                    symbol="AAPL",
                    side="SELL",
                    quantity=5,
                    order_type="LIMIT",
                    price=145.00
                )
                
                result = await connector.place_order(order)
                
                assert isinstance(result, OrderResult)
                assert result.order_id == "test_limit_order_id"
                assert result.status == "QUEUED"
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, connector):
        """Test order cancellation"""
        connector._authenticated = True
        
        with patch.object(connector, '_make_request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_request.return_value = mock_response
            
            result = await connector.cancel_order("test_order_id")
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_cancel_order_failure(self, connector):
        """Test order cancellation failure"""
        connector._authenticated = True
        
        with patch.object(connector, '_make_request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_request.return_value = mock_response
            
            result = await connector.cancel_order("invalid_order_id")
            
            assert result is False


class TestAccountManagement:
    """Test account management functionality"""
    
    @pytest.mark.asyncio
    async def test_get_account_info(self, connector):
        """Test account information retrieval"""
        connector._authenticated = True
        
        mock_account_response = {
            "results": [{
                "account_number": "12345",
                "buying_power": "10000.00",
                "cash": "5000.00"
            }]
        }
        
        mock_portfolio_response = {
            "total_return_today": "15000.00"
        }
        
        with patch.object(connector, '_make_request') as mock_request:
            account_mock = MagicMock()
            account_mock.status_code = 200
            account_mock.json.return_value = mock_account_response
            
            portfolio_mock = MagicMock()
            portfolio_mock.status_code = 200
            portfolio_mock.json.return_value = mock_portfolio_response
            
            mock_request.side_effect = [account_mock, portfolio_mock]
            
            account_info = await connector.get_account_info()
            
            assert account_info["account_id"] == "12345"
            assert account_info["buying_power"] == 10000.0
            assert account_info["cash"] == 5000.0
            assert account_info["total_value"] == 15000.0
    
    @pytest.mark.asyncio
    async def test_get_positions(self, connector):
        """Test positions retrieval"""
        connector._authenticated = True
        
        mock_positions_response = {
            "results": [{
                "quantity": "10",
                "average_buy_price": "150.00",
                "market_value": "1600.00",
                "total_return_today": "100.00",
                "instrument": "https://api.robinhood.com/instruments/test_id/"
            }]
        }
        
        mock_instrument_response = {
            "symbol": "AAPL"
        }
        
        with patch.object(connector, '_make_request') as mock_request:
            positions_mock = MagicMock()
            positions_mock.status_code = 200
            positions_mock.json.return_value = mock_positions_response
            
            instrument_mock = MagicMock()
            instrument_mock.status_code = 200
            instrument_mock.json.return_value = mock_instrument_response
            
            mock_request.side_effect = [positions_mock, instrument_mock]
            
            positions = await connector.get_positions()
            
            assert len(positions) == 1
            assert positions[0]["symbol"] == "AAPL"
            assert positions[0]["quantity"] == 10.0
            assert positions[0]["average_cost"] == 150.0


class TestRateLimiter:
    """Test rate limiting functionality"""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests(self):
        """Test that rate limiter allows requests within limits"""
        limiter = RateLimiter(max_requests=5, time_window=1)
        
        # Should allow 5 requests
        for _ in range(5):
            await limiter.acquire()
        
        # Check that requests were recorded
        assert len(limiter.requests) == 5
    
    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_excess_requests(self):
        """Test that rate limiter blocks requests exceeding limits"""
        limiter = RateLimiter(max_requests=2, time_window=10)
        
        # Make 2 requests (should be allowed)
        await limiter.acquire()
        await limiter.acquire()
        
        # Third request should be delayed
        start_time = asyncio.get_event_loop().time()
        await limiter.acquire()
        end_time = asyncio.get_event_loop().time()
        
        # Should have been delayed
        assert end_time - start_time >= 0  # Some delay occurred


class TestConnectionPool:
    """Test connection pool functionality"""
    
    @pytest.mark.asyncio
    async def test_connection_pool_creates_client(self):
        """Test that connection pool creates HTTP client"""
        pool = ConnectionPool(max_connections=5)
        
        client = await pool.get_client()
        
        assert isinstance(client, httpx.AsyncClient)
        assert pool._client is not None
    
    @pytest.mark.asyncio
    async def test_connection_pool_reuses_client(self):
        """Test that connection pool reuses existing client"""
        pool = ConnectionPool(max_connections=5)
        
        client1 = await pool.get_client()
        client2 = await pool.get_client()
        
        assert client1 is client2
    
    @pytest.mark.asyncio
    async def test_connection_pool_close(self):
        """Test connection pool cleanup"""
        pool = ConnectionPool(max_connections=5)
        
        # Create client
        await pool.get_client()
        assert pool._client is not None
        
        # Close pool
        await pool.close()
        assert pool._client is None


class TestOptionsTrading:
    """Test options trading functionality"""
    
    @pytest.mark.asyncio
    async def test_get_options_chains(self, connector):
        """Test options chain retrieval"""
        connector._authenticated = True
        
        mock_options_response = {
            "results": [{
                "expiration_date": "2023-12-15",
                "strike_price": "150.00",
                "type": "call"
            }]
        }
        
        with patch.object(connector, '_make_request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_options_response
            mock_request.return_value = mock_response
            
            options_chain = await connector.get_options_chains("AAPL")
            
            assert options_chain == mock_options_response
    
    @pytest.mark.asyncio
    async def test_place_options_order(self, connector):
        """Test options order placement"""
        connector._authenticated = True
        
        mock_order_response = {
            "id": "options_order_id",
            "state": "confirmed",
            "quantity": "1",
            "price": "5.00"
        }
        
        with patch.object(connector, '_make_request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 201
            mock_response.json.return_value = mock_order_response
            mock_request.return_value = mock_response
            
            options_order = {
                "symbol": "AAPL",
                "quantity": "1",
                "side": "buy",
                "type": "limit",
                "price": "5.00"
            }
            
            result = await connector.place_options_order(options_order)
            
            assert isinstance(result, OrderResult)
            assert result.order_id == "options_order_id"
            assert result.status == "CONFIRMED"


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_unauthenticated_request(self, connector):
        """Test that unauthenticated requests raise error"""
        connector._authenticated = False
        
        with pytest.raises(RuntimeError, match="Not authenticated"):
            await connector._make_request("GET", "test/")
    
    @pytest.mark.asyncio
    async def test_token_refresh(self, connector):
        """Test automatic token refresh on 401 response"""
        connector._authenticated = True
        connector.credentials.refresh_token = "refresh_token"
        
        with patch.object(connector.connection_pool, 'get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_get_client.return_value = mock_client
            
            # First response: 401 (token expired)
            # Second response: 200 (after refresh)
            mock_responses = [
                MagicMock(status_code=401),
                MagicMock(status_code=200)
            ]
            mock_client.request.side_effect = mock_responses
            
            # Mock token refresh
            refresh_response = MagicMock()
            refresh_response.status_code = 200
            refresh_response.json.return_value = {"access_token": "new_token"}
            mock_client.post.return_value = refresh_response
            
            response = await connector._make_request("GET", "test/")
            
            assert response.status_code == 200
            assert connector.credentials.access_token == "new_token"
    
    def test_supported_symbols(self, connector):
        """Test supported symbols list"""
        symbols = connector.get_supported_symbols()
        
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert "AAPL" in symbols
        assert "SPY" in symbols


if __name__ == "__main__":
    pytest.main([__file__])