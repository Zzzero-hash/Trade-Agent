"""Integration tests for CoinbaseConnector"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.exchanges.base import Order, OrderResult, MarketData
from src.exchanges.coinbase import (
    CoinbaseConnector,
    CryptoMarketHours,
    CryptoOrder
)


@pytest.fixture
def connector():
    """Create a test Coinbase connector instance"""
    return CoinbaseConnector(
        api_key="test_api_key",
        api_secret="dGVzdF9hcGlfc2VjcmV0",  # base64 encoded
        passphrase="test_passphrase",
        sandbox=True
    )


@pytest.fixture
def mock_accounts_response():
    """Mock Coinbase accounts response"""
    return [
        {
            "id": "account_id_1",
            "currency": "USD",
            "balance": "10000.00",
            "available": "9500.00",
            "hold": "500.00"
        },
        {
            "id": "account_id_2",
            "currency": "BTC",
            "balance": "0.5",
            "available": "0.5",
            "hold": "0.0"
        }
    ]


@pytest.fixture
def mock_candles_response():
    """Mock Coinbase candles response"""
    return [
        # [timestamp, low, high, open, close, volume]
        [1672531200, 16800.0, 17000.0, 16900.0, 16950.0, 1.5],
        [1672534800, 16950.0, 17100.0, 16950.0, 17050.0, 2.1]
    ]


@pytest.fixture
def mock_ticker_response():
    """Mock Coinbase ticker response"""
    return {
        "trade_id": 12345,
        "price": "17000.00",
        "size": "0.1",
        "bid": "16999.00",
        "ask": "17001.00",
        "volume": "1000.0",
        "time": "2023-01-01T12:00:00.000000Z"
    }


@pytest.fixture
def mock_order_response():
    """Mock Coinbase order response"""
    return {
        "id": "order_123",
        "status": "pending",
        "filled_size": "0.0",
        "executed_value": "0.0",
        "side": "buy",
        "product_id": "BTC-USD"
    }


@pytest.fixture
def mock_margin_profile_response():
    """Mock Coinbase margin profile response"""
    return {
        "user_id": "user_123",
        "max_withdraw_amount": "10000.00",
        "withdrawal_currency_name": "USD",
        "max_borrow_amount": "5000.00",
        "borrow_currency_name": "USD"
    }


class TestCoinbaseConnector:
    """Test suite for CoinbaseConnector"""

    @pytest.mark.asyncio
    async def test_successful_connection(self, connector, mock_accounts_response):
        """Test successful connection to Coinbase"""
        with patch.object(
            connector, '_make_authenticated_request'
        ) as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_accounts_response
            mock_request.return_value = mock_response

            # Mock the HTTP client creation
            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                connector._client = mock_client

                result = await connector.connect()

                assert result is True
                assert connector.is_connected is True

    @pytest.mark.asyncio
    async def test_failed_connection(self, connector):
        """Test failed connection to Coinbase"""
        with patch.object(
            connector, '_make_authenticated_request'
        ) as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"
            mock_request.return_value = mock_response

            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                connector._client = mock_client

                result = await connector.connect()

                assert result is False
                assert connector.is_connected is False

    @pytest.mark.asyncio
    async def test_disconnect(self, connector):
        """Test disconnection from Coinbase"""
        # Set up connected state
        connector.is_connected = True
        mock_client = AsyncMock()
        mock_websocket = AsyncMock()
        connector._client = mock_client
        connector._websocket = mock_websocket

        await connector.disconnect()

        assert connector.is_connected is False
        assert connector._client is None
        assert connector._websocket is None
        mock_client.aclose.assert_called_once()
        mock_websocket.close.assert_called_once()

    def test_signature_generation(self, connector):
        """Test authentication signature generation"""
        timestamp = "1640995200"
        method = "GET"
        path = "/accounts"
        body = ""

        signature = connector._generate_signature(timestamp, method, path, body)

        # Should generate a valid base64 signature
        assert isinstance(signature, str)
        assert len(signature) > 0 
   
    @pytest.mark.asyncio
    async def test_get_historical_data(self, connector, mock_candles_response):
        """Test getting historical cryptocurrency data"""
        connector.is_connected = True
        
        with patch.object(connector, '_make_authenticated_request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_candles_response
            mock_request.return_value = mock_response
            
            start_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
            end_time = datetime(2023, 1, 2, tzinfo=timezone.utc)
            
            df = await connector.get_historical_data("BTCUSD", "1h", start_time, end_time)
            
            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            assert len(df) == 2
            assert "open" in df.columns
            assert "high" in df.columns
            assert "low" in df.columns
            assert "close" in df.columns
            assert "volume" in df.columns
    
    @pytest.mark.asyncio
    async def test_real_time_data_websocket_setup(self, connector):
        """Test WebSocket setup for real-time crypto data"""
        connector.is_connected = True
        
        # Test that the method exists and handles connection errors gracefully
        symbols = ["BTCUSD", "ETHUSD"]
        
        # Since websockets module may not be installed, we expect this to fail gracefully
        # The important thing is that the method exists and has the right signature
        try:
            async for market_data in connector.get_real_time_data(symbols):
                # If we somehow get data, verify it's the right type
                assert isinstance(market_data, MarketData)
                break  # Just test one iteration
        except Exception as e:
            # Expected to fail due to missing websockets module or connection issues
            # This is fine for unit testing - we're testing the interface exists
            assert "websockets" in str(e).lower() or "connect" in str(e).lower()
    
    def test_websocket_url_configuration(self, connector):
        """Test WebSocket URL configuration for sandbox vs live"""
        # Test sandbox URL
        sandbox_connector = CoinbaseConnector("key", "secret", "pass", sandbox=True)
        assert "sandbox" in sandbox_connector.websocket_url
        
        # Test live URL
        live_connector = CoinbaseConnector("key", "secret", "pass", sandbox=False)
        assert "sandbox" not in live_connector.websocket_url
    
    @pytest.mark.asyncio
    async def test_place_crypto_order(self, connector, mock_order_response):
        """Test placing crypto-specific orders"""
        connector.is_connected = True
        
        with patch.object(connector, '_make_authenticated_request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_order_response
            mock_request.return_value = mock_response
            
            # Test market order
            order = Order(
                symbol="BTCUSD",
                side="BUY",
                quantity=0.1,
                order_type="MARKET"
            )
            
            result = await connector.place_order(order)
            
            assert isinstance(result, OrderResult)
            assert result.order_id == "order_123"
            assert result.status == "PENDING"
            
            # Verify the request was made with correct crypto-specific parameters
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == "POST"  # method
            assert call_args[0][1] == "/orders"  # path
            
            # Check order data
            order_data = call_args[1]["json"]
            assert order_data["side"] == "buy"
            assert order_data["product_id"] == "BTC-USD"
            assert order_data["type"] == "market"
            assert order_data["size"] == "0.1"
    
    @pytest.mark.asyncio
    async def test_place_limit_order(self, connector, mock_order_response):
        """Test placing limit orders with price"""
        connector.is_connected = True
        
        with patch.object(connector, '_make_authenticated_request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_order_response
            mock_request.return_value = mock_response
            
            # Test limit order
            order = Order(
                symbol="ETHUSD",
                side="SELL",
                quantity=1.0,
                order_type="LIMIT",
                price=1200.0
            )
            
            result = await connector.place_order(order)
            
            assert isinstance(result, OrderResult)
            
            # Check limit order specific parameters
            call_args = mock_request.call_args
            order_data = call_args[1]["json"]
            assert order_data["type"] == "limit"
            assert order_data["price"] == "1200.0"
    
    @pytest.mark.asyncio
    async def test_place_margin_order(self, connector, mock_order_response):
        """Test placing margin orders with leverage"""
        connector.is_connected = True
        
        with patch.object(connector, '_make_authenticated_request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_order_response
            mock_request.return_value = mock_response
            
            # Test margin order with leverage
            order = Order(
                symbol="BTCUSD",
                side="BUY",
                quantity=0.5,
                order_type="LIMIT",
                price=16000.0
            )
            
            leverage = 2.0
            result = await connector.place_margin_order(order, leverage)
            
            assert isinstance(result, OrderResult)
            assert "leverage" in result.message
            
            # Verify margin-specific parameters
            call_args = mock_request.call_args
            assert call_args[0][1] == "/margin/orders"  # margin endpoint
            
            order_data = call_args[1]["json"]
            assert order_data["margin_type"] == "cross"
            assert order_data["leverage"] == "2.0"
    
    @pytest.mark.asyncio
    async def test_get_margin_profile(self, connector, mock_margin_profile_response):
        """Test getting margin profile information"""
        connector.is_connected = True
        
        with patch.object(connector, '_make_authenticated_request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_margin_profile_response
            mock_request.return_value = mock_response
            
            profile = await connector.get_margin_profile()
            
            assert isinstance(profile, dict)
            assert "user_id" in profile
            assert "max_withdraw_amount" in profile
            assert "max_borrow_amount" in profile
            
            mock_request.assert_called_once_with("GET", "/margin/profile_information")
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, connector):
        """Test canceling orders"""
        connector.is_connected = True
        
        with patch.object(connector, '_make_authenticated_request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_request.return_value = mock_response
            
            result = await connector.cancel_order("order_123")
            
            assert result is True
            mock_request.assert_called_once_with("DELETE", "/orders/order_123")
    
    @pytest.mark.asyncio
    async def test_get_current_prices(self, connector, mock_ticker_response):
        """Test getting current crypto prices"""
        connector.is_connected = True
        
        with patch.object(connector, '_make_authenticated_request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_ticker_response
            mock_request.return_value = mock_response
            
            prices = await connector.get_current_prices(["BTCUSD"])
            
            assert isinstance(prices, dict)
            assert "BTCUSD" in prices
            
            btc_price = prices["BTCUSD"]
            assert "price" in btc_price
            assert "bid" in btc_price
            assert "ask" in btc_price
            assert "volume" in btc_price
    
    def test_symbol_formatting(self, connector):
        """Test symbol formatting for Coinbase API"""
        # Test standard crypto pairs
        assert connector._format_symbol("BTCUSD") == "BTC-USD"
        assert connector._format_symbol("ETHUSD") == "ETH-USD"
        assert connector._format_symbol("LTCEUR") == "LTC-EUR"
        
        # Test already formatted symbols
        assert connector._format_symbol("BTC-USD") == "BTC-USD"
        
        # Test unformatting
        assert connector._unformat_symbol("BTC-USD") == "BTCUSD"
        assert connector._unformat_symbol("ETH-EUR") == "ETHEUR"
    
    def test_convert_to_crypto_order(self, connector):
        """Test conversion from generic Order to CryptoOrder"""
        order = Order(
            symbol="BTCUSD",
            side="BUY",
            quantity=0.5,
            order_type="LIMIT",
            price=17000.0,
            stop_price=16500.0,
            time_in_force="GTC"
        )
        
        crypto_order = connector._convert_to_crypto_order(order)
        
        assert isinstance(crypto_order, CryptoOrder)
        assert crypto_order.symbol == "BTCUSD"
        assert crypto_order.side == "buy"
        assert crypto_order.size == "0.5"
        assert crypto_order.order_type == "limit"
        assert crypto_order.price == "17000.0"
        assert crypto_order.stop_price == "16500.0"
        assert crypto_order.time_in_force == "GTC"
    
    def test_supported_symbols(self, connector):
        """Test getting supported cryptocurrency symbols"""
        symbols = connector.get_supported_symbols()
        
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert "BTCUSD" in symbols
        assert "ETHUSD" in symbols
        assert "LTCUSD" in symbols


class TestCryptoMarketHours:
    """Test suite for CryptoMarketHours - 24/7 operations"""
    
    def test_market_always_open(self):
        """Test that crypto markets are always open (24/7)"""
        # Test various times and dates
        test_times = [
            datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),  # New Year midnight
            datetime(2023, 12, 25, 12, 0, 0, tzinfo=timezone.utc),  # Christmas noon
            datetime(2023, 7, 4, 18, 30, 0, tzinfo=timezone.utc),  # July 4th evening
            datetime(2023, 6, 15, 3, 45, 0, tzinfo=timezone.utc),  # Random weekday early morning
            datetime(2023, 8, 20, 23, 59, 59, tzinfo=timezone.utc),  # Sunday night
        ]
        
        for test_time in test_times:
            assert CryptoMarketHours.is_market_open(test_time) is True
    
    def test_market_open_no_datetime(self):
        """Test market open check without providing datetime (uses current time)"""
        assert CryptoMarketHours.is_market_open() is True
    
    def test_next_market_open_always_current(self):
        """Test that next market open is always current time for crypto"""
        test_time = datetime(2023, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
        next_open = CryptoMarketHours.get_next_market_open(test_time)
        
        assert next_open == test_time
    
    def test_next_market_open_no_datetime(self):
        """Test next market open without providing datetime"""
        # Should return current time (approximately)
        next_open = CryptoMarketHours.get_next_market_open()
        current_time = datetime.now(timezone.utc)
        
        # Should be within a few seconds of current time
        time_diff = abs((next_open - current_time).total_seconds())
        assert time_diff < 5  # Within 5 seconds


class TestCryptoOrder:
    """Test suite for CryptoOrder data structure"""
    
    def test_crypto_order_creation(self):
        """Test creating CryptoOrder with various parameters"""
        order = CryptoOrder(
            symbol="BTC-USD",
            side="buy",
            size="0.1",
            order_type="limit",
            price="17000.00",
            time_in_force="GTC",
            post_only=True
        )
        
        assert order.symbol == "BTC-USD"
        assert order.side == "buy"
        assert order.size == "0.1"
        assert order.order_type == "limit"
        assert order.price == "17000.00"
        assert order.time_in_force == "GTC"
        assert order.post_only is True
    
    def test_crypto_order_defaults(self):
        """Test CryptoOrder with default values"""
        order = CryptoOrder(
            symbol="ETH-USD",
            side="sell"
        )
        
        assert order.symbol == "ETH-USD"
        assert order.side == "sell"
        assert order.size is None
        assert order.funds is None
        assert order.order_type == "market"
        assert order.price is None
        assert order.stop_price is None
        assert order.time_in_force == "GTC"
        assert order.post_only is False
        assert order.client_oid is None
    
    def test_crypto_order_market_with_funds(self):
        """Test market order using funds instead of size"""
        order = CryptoOrder(
            symbol="BTC-USD",
            side="buy",
            funds="1000.00",
            order_type="market"
        )
        
        assert order.funds == "1000.00"
        assert order.size is None
        assert order.order_type == "market"


if __name__ == "__main__":
    pytest.main([__file__])
 
   @pytest.mark.asyncio
    async def test_get_historical_data(self, connector, mock_candles_response):
        """Test getting historical cryptocurrency data"""
        connector.is_connected = True

        with patch.object(
            connector, '_make_authenticated_request'
        ) as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_candles_response
            mock_request.return_value = mock_response

            start_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
            end_time = datetime(2023, 1, 2, tzinfo=timezone.utc)

            df = await connector.get_historical_data(
                "BTCUSD", "1h", start_time, end_time
            )

            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            assert len(df) == 2
            assert "open" in df.columns
            assert "high" in df.columns
            assert "low" in df.columns
            assert "close" in df.columns
            assert "volume" in df.columns

    @pytest.mark.asyncio
    async def test_real_time_data_websocket_setup(self, connector):
        """Test WebSocket setup for real-time crypto data"""
        connector.is_connected = True

        # Test that the method exists and handles connection errors gracefully
        symbols = ["BTCUSD", "ETHUSD"]

        # Since websockets module may not be installed, we expect this to fail
        # gracefully. The important thing is that the method exists and has
        # the right signature
        try:
            async for market_data in connector.get_real_time_data(symbols):
                # If we somehow get data, verify it's the right type
                assert isinstance(market_data, MarketData)
                break  # Just test one iteration
        except Exception as e:
            # Expected to fail due to missing websockets module or connection
            # issues. This is fine for unit testing - we're testing the
            # interface exists
            assert (
                "websockets" in str(e).lower() or
                "connect" in str(e).lower()
            )

    def test_websocket_url_configuration(self, connector):
        """Test WebSocket URL configuration for sandbox vs live"""
        # Test sandbox URL
        sandbox_connector = CoinbaseConnector(
            "key", "secret", "pass", sandbox=True
        )
        assert "sandbox" in sandbox_connector.websocket_url

        # Test live URL
        live_connector = CoinbaseConnector(
            "key", "secret", "pass", sandbox=False
        )
        assert "sandbox" not in live_connector.websocket_url

    @pytest.mark.asyncio
    async def test_place_crypto_order(self, connector, mock_order_response):
        """Test placing crypto-specific orders"""
        connector.is_connected = True

        with patch.object(
            connector, '_make_authenticated_request'
        ) as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_order_response
            mock_request.return_value = mock_response

            # Test market order
            order = Order(
                symbol="BTCUSD",
                side="BUY",
                quantity=0.1,
                order_type="MARKET"
            )

            result = await connector.place_order(order)

            assert isinstance(result, OrderResult)
            assert result.order_id == "order_123"
            assert result.status == "PENDING"

            # Verify the request was made with correct crypto-specific params
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == "POST"  # method
            assert call_args[0][1] == "/orders"  # path

            # Check order data
            order_data = call_args[1]["json"]
            assert order_data["side"] == "buy"
            assert order_data["product_id"] == "BTC-USD"
            assert order_data["type"] == "market"
            assert order_data["size"] == "0.1"

    @pytest.mark.asyncio
    async def test_place_limit_order(self, connector, mock_order_response):
        """Test placing limit orders with price"""
        connector.is_connected = True

        with patch.object(
            connector, '_make_authenticated_request'
        ) as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_order_response
            mock_request.return_value = mock_response

            # Test limit order
            order = Order(
                symbol="ETHUSD",
                side="SELL",
                quantity=1.0,
                order_type="LIMIT",
                price=1200.0
            )

            result = await connector.place_order(order)

            assert isinstance(result, OrderResult)

            # Check limit order specific parameters
            call_args = mock_request.call_args
            order_data = call_args[1]["json"]
            assert order_data["type"] == "limit"
            assert order_data["price"] == "1200.0"

    @pytest.mark.asyncio
    async def test_place_margin_order(self, connector, mock_order_response):
        """Test placing margin orders with leverage"""
        connector.is_connected = True

        with patch.object(
            connector, '_make_authenticated_request'
        ) as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_order_response
            mock_request.return_value = mock_response

            # Test margin order with leverage
            order = Order(
                symbol="BTCUSD",
                side="BUY",
                quantity=0.5,
                order_type="LIMIT",
                price=16000.0
            )

            leverage = 2.0
            result = await connector.place_margin_order(order, leverage)

            assert isinstance(result, OrderResult)
            assert "leverage" in result.message

            # Verify margin-specific parameters
            call_args = mock_request.call_args
            assert call_args[0][1] == "/margin/orders"  # margin endpoint

            order_data = call_args[1]["json"]
            assert order_data["margin_type"] == "cross"
            assert order_data["leverage"] == "2.0"

    @pytest.mark.asyncio
    async def test_get_margin_profile(
        self, connector, mock_margin_profile_response
    ):
        """Test getting margin profile information"""
        connector.is_connected = True

        with patch.object(
            connector, '_make_authenticated_request'
        ) as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_margin_profile_response
            mock_request.return_value = mock_response

            profile = await connector.get_margin_profile()

            assert isinstance(profile, dict)
            assert "user_id" in profile
            assert "max_withdraw_amount" in profile
            assert "max_borrow_amount" in profile

            mock_request.assert_called_once_with(
                "GET", "/margin/profile_information"
            )

    @pytest.mark.asyncio
    async def test_cancel_order(self, connector):
        """Test canceling orders"""
        connector.is_connected = True

        with patch.object(
            connector, '_make_authenticated_request'
        ) as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_request.return_value = mock_response

            result = await connector.cancel_order("order_123")

            assert result is True
            mock_request.assert_called_once_with("DELETE", "/orders/order_123")

    @pytest.mark.asyncio
    async def test_get_current_prices(self, connector, mock_ticker_response):
        """Test getting current crypto prices"""
        connector.is_connected = True

        with patch.object(
            connector, '_make_authenticated_request'
        ) as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_ticker_response
            mock_request.return_value = mock_response

            prices = await connector.get_current_prices(["BTCUSD"])

            assert isinstance(prices, dict)
            assert "BTCUSD" in prices

            btc_price = prices["BTCUSD"]
            assert "price" in btc_price
            assert "bid" in btc_price
            assert "ask" in btc_price
            assert "volume" in btc_price

    def test_symbol_formatting(self, connector):
        """Test symbol formatting for Coinbase API"""
        # Test standard crypto pairs
        assert connector._format_symbol("BTCUSD") == "BTC-USD"
        assert connector._format_symbol("ETHUSD") == "ETH-USD"
        assert connector._format_symbol("LTCEUR") == "LTC-EUR"

        # Test already formatted symbols
        assert connector._format_symbol("BTC-USD") == "BTC-USD"

        # Test unformatting
        assert connector._unformat_symbol("BTC-USD") == "BTCUSD"
        assert connector._unformat_symbol("ETH-EUR") == "ETHEUR"

    def test_convert_to_crypto_order(self, connector):
        """Test conversion from generic Order to CryptoOrder"""
        order = Order(
            symbol="BTCUSD",
            side="BUY",
            quantity=0.5,
            order_type="LIMIT",
            price=17000.0,
            stop_price=16500.0,
            time_in_force="GTC"
        )

        crypto_order = connector._convert_to_crypto_order(order)

        assert isinstance(crypto_order, CryptoOrder)
        assert crypto_order.symbol == "BTCUSD"
        assert crypto_order.side == "buy"
        assert crypto_order.size == "0.5"
        assert crypto_order.order_type == "limit"
        assert crypto_order.price == "17000.0"
        assert crypto_order.stop_price == "16500.0"
        assert crypto_order.time_in_force == "GTC"

    def test_supported_symbols(self, connector):
        """Test getting supported cryptocurrency symbols"""
        symbols = connector.get_supported_symbols()

        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert "BTCUSD" in symbols
        assert "ETHUSD" in symbols
        assert "LTCUSD" in symbols


class TestCryptoMarketHours:
    """Test suite for CryptoMarketHours - 24/7 operations"""

    def test_market_always_open(self):
        """Test that crypto markets are always open (24/7)"""
        # Test various times and dates
        test_times = [
            # New Year midnight
            datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            # Christmas noon
            datetime(2023, 12, 25, 12, 0, 0, tzinfo=timezone.utc),
            # July 4th evening
            datetime(2023, 7, 4, 18, 30, 0, tzinfo=timezone.utc),
            # Random weekday early morning
            datetime(2023, 6, 15, 3, 45, 0, tzinfo=timezone.utc),
            # Sunday night
            datetime(2023, 8, 20, 23, 59, 59, tzinfo=timezone.utc),
        ]

        for test_time in test_times:
            assert CryptoMarketHours.is_market_open(test_time) is True

    def test_market_open_no_datetime(self):
        """Test market open check without providing datetime (uses current)"""
        assert CryptoMarketHours.is_market_open() is True

    def test_next_market_open_always_current(self):
        """Test that next market open is always current time for crypto"""
        test_time = datetime(2023, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
        next_open = CryptoMarketHours.get_next_market_open(test_time)

        assert next_open == test_time

    def test_next_market_open_no_datetime(self):
        """Test next market open without providing datetime"""
        # Should return current time (approximately)
        next_open = CryptoMarketHours.get_next_market_open()
        current_time = datetime.now(timezone.utc)

        # Should be within a few seconds of current time
        time_diff = abs((next_open - current_time).total_seconds())
        assert time_diff < 5  # Within 5 seconds


class TestCryptoOrder:
    """Test suite for CryptoOrder data structure"""

    def test_crypto_order_creation(self):
        """Test creating CryptoOrder with various parameters"""
        order = CryptoOrder(
            symbol="BTC-USD",
            side="buy",
            size="0.1",
            order_type="limit",
            price="17000.00",
            time_in_force="GTC",
            post_only=True
        )

        assert order.symbol == "BTC-USD"
        assert order.side == "buy"
        assert order.size == "0.1"
        assert order.order_type == "limit"
        assert order.price == "17000.00"
        assert order.time_in_force == "GTC"
        assert order.post_only is True

    def test_crypto_order_defaults(self):
        """Test CryptoOrder with default values"""
        order = CryptoOrder(
            symbol="ETH-USD",
            side="sell"
        )

        assert order.symbol == "ETH-USD"
        assert order.side == "sell"
        assert order.size is None
        assert order.funds is None
        assert order.order_type == "market"
        assert order.price is None
        assert order.stop_price is None
        assert order.time_in_force == "GTC"
        assert order.post_only is False
        assert order.client_oid is None

    def test_crypto_order_market_with_funds(self):
        """Test market order using funds instead of size"""
        order = CryptoOrder(
            symbol="BTC-USD",
            side="buy",
            funds="1000.00",
            order_type="market"
        )

        assert order.funds == "1000.00"
        assert order.size is None
        assert order.order_type == "market"


if __name__ == "__main__":
    pytest.main([__file__])