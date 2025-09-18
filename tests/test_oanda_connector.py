"""Integration tests for OANDAConnector"""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import httpx

from src.exchanges.oanda import OANDAConnector, ForexMarketHours, ForexOrder
from src.exchanges.base import Order, OrderResult, MarketData


@pytest.fixture
def connector():
    """Create a test OANDA connector instance"""
    return OANDAConnector(
        api_key="test_api_key",
        account_id="test_account_id",
        sandbox=True
    )


@pytest.fixture
def mock_account_response():
    """Mock OANDA account response"""
    return {
        "account": {
            "id": "test_account_id",
            "currency": "USD",
            "balance": "10000.0000",
            "NAV": "10500.0000",
            "unrealizedPL": "500.0000",
            "realizedPL": "0.0000",
            "marginUsed": "1000.0000",
            "marginAvailable": "9000.0000",
            "openTradeCount": 2,
            "openPositionCount": 1
        }
    }


@pytest.fixture
def mock_candles_response():
    """Mock OANDA candles response"""
    return {
        "instrument": "EUR_USD",
        "granularity": "H1",
        "candles": [
            {
                "complete": True,
                "volume": 1000,
                "time": "2023-01-01T00:00:00.000000000Z",
                "mid": {
                    "o": "1.0500",
                    "h": "1.0550",
                    "l": "1.0480",
                    "c": "1.0520"
                }
            },
            {
                "complete": True,
                "volume": 1200,
                "time": "2023-01-01T01:00:00.000000000Z",
                "mid": {
                    "o": "1.0520",
                    "h": "1.0580",
                    "l": "1.0510",
                    "c": "1.0560"
                }
            }
        ]
    }


@pytest.fixture
def mock_pricing_response():
    """Mock OANDA pricing response"""
    return {
        "prices": [
            {
                "instrument": "EUR_USD",
                "time": "2023-01-01T12:00:00.000000000Z",
                "bids": [{"price": "1.0500", "liquidity": 10000000}],
                "asks": [{"price": "1.0502", "liquidity": 10000000}]
            }
        ]
    }


class TestOANDAConnector:
    """Test suite for OANDAConnector"""
    
    @pytest.mark.asyncio
    async def test_successful_connection(self, connector, mock_account_response):
        """Test successful connection to OANDA"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock successful account response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_account_response
            mock_client.get.return_value = mock_response
            
            result = await connector.connect()
            
            assert result is True
            assert connector.is_connected is True
            assert connector._client is not None
    
    @pytest.mark.asyncio
    async def test_failed_connection(self, connector):
        """Test failed connection to OANDA"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock failed response
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"
            mock_client.get.return_value = mock_response
            
            result = await connector.connect()
            
            assert result is False
            assert connector.is_connected is False
    
    @pytest.mark.asyncio
    async def test_disconnect(self, connector):
        """Test disconnection from OANDA"""
        # Set up connected state
        connector.is_connected = True
        connector._client = AsyncMock()
        connector._stream_client = AsyncMock()
        
        await connector.disconnect()
        
        assert connector.is_connected is False
        connector._client.aclose.assert_called_once()
        connector._stream_client.aclose.assert_called_once()


class TestHistoricalData:
    """Test historical data retrieval"""
    
    @pytest.mark.asyncio
    async def test_get_historical_data(self, connector, mock_candles_response):
        """Test historical data retrieval"""
        connector.is_connected = True
        connector._client = AsyncMock()
        
        # Mock candles response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_candles_response
        connector._client.get.return_value = mock_response
        
        start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2023, 1, 2, tzinfo=timezone.utc)
        
        df = await connector.get_historical_data("EURUSD", "1h", start_date, end_date)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        
        # Check data values
        assert df.iloc[0]["open"] == 1.0500
        assert df.iloc[0]["close"] == 1.0520
        assert df.iloc[1]["open"] == 1.0520
        assert df.iloc[1]["close"] == 1.0560
    
    @pytest.mark.asyncio
    async def test_get_historical_data_error(self, connector):
        """Test historical data retrieval with error"""
        connector.is_connected = True
        connector._client = AsyncMock()
        
        # Mock error response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        connector._client.get.return_value = mock_response
        
        start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2023, 1, 2, tzinfo=timezone.utc)
        
        df = await connector.get_historical_data("INVALID", "1h", start_date, end_date)
        
        assert isinstance(df, pd.DataFrame)
        assert df.empty


class TestRealTimeData:
    """Test real-time data streaming"""
    
    @pytest.mark.asyncio
    async def test_real_time_data_stream(self, connector):
        """Test real-time data streaming"""
        connector.is_connected = True
        
        # Mock streaming response
        mock_stream_data = [
            '{"type":"PRICE","instrument":"EUR_USD","time":"2023-01-01T12:00:00.000000000Z","bids":[{"price":"1.0500"}],"asks":[{"price":"1.0502"}]}',
            '{"type":"HEARTBEAT","time":"2023-01-01T12:00:05.000000000Z"}'
        ]
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_stream_client = AsyncMock()
            mock_client_class.return_value = mock_stream_client
            
            # Mock stream response
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.aiter_lines.return_value = mock_stream_data.__aiter__()
            
            mock_stream_client.stream.return_value.__aenter__.return_value = mock_response
            
            # Test streaming
            symbols = ["EURUSD"]
            data_count = 0
            
            async for market_data in connector.get_real_time_data(symbols):
                assert isinstance(market_data, MarketData)
                assert market_data.symbol == "EURUSD"
                assert market_data.exchange == "OANDA"
                data_count += 1
                
                # Break after receiving one data point
                if data_count >= 1:
                    break


class TestOrderManagement:
    """Test order management functionality"""
    
    @pytest.mark.asyncio
    async def test_place_market_order(self, connector):
        """Test placing a market order"""
        connector.is_connected = True
        connector._client = AsyncMock()
        
        mock_order_response = {
            "orderCreateTransaction": {
                "id": "test_order_id",
                "type": "MARKET_ORDER",
                "instrument": "EUR_USD",
                "units": "10000"
            },
            "orderFillTransaction": {
                "id": "test_fill_id",
                "units": "10000",
                "price": "1.0500"
            }
        }
        
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = mock_order_response
        connector._client.post.return_value = mock_response
        
        order = Order(
            symbol="EURUSD",
            side="BUY",
            quantity=10000,
            order_type="MARKET"
        )
        
        result = await connector.place_order(order)
        
        assert isinstance(result, OrderResult)
        assert result.order_id == "test_order_id"
        assert result.status == "FILLED"
        assert result.filled_quantity == 10000
        assert result.avg_fill_price == 1.0500
    
    @pytest.mark.asyncio
    async def test_place_limit_order(self, connector):
        """Test placing a limit order"""
        connector.is_connected = True
        connector._client = AsyncMock()
        
        mock_order_response = {
            "orderCreateTransaction": {
                "id": "test_limit_order_id",
                "type": "LIMIT_ORDER",
                "instrument": "EUR_USD",
                "units": "5000",
                "price": "1.0450"
            }
        }
        
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = mock_order_response
        connector._client.post.return_value = mock_response
        
        order = Order(
            symbol="EURUSD",
            side="BUY",
            quantity=5000,
            order_type="LIMIT",
            price=1.0450
        )
        
        result = await connector.place_order(order)
        
        assert isinstance(result, OrderResult)
        assert result.order_id == "test_limit_order_id"
        assert result.status == "PENDING"
        assert result.filled_quantity == 0
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, connector):
        """Test order cancellation"""
        connector.is_connected = True
        connector._client = AsyncMock()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        connector._client.put.return_value = mock_response
        
        result = await connector.cancel_order("test_order_id")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_cancel_order_failure(self, connector):
        """Test order cancellation failure"""
        connector.is_connected = True
        connector._client = AsyncMock()
        
        mock_response = MagicMock()
        mock_response.status_code = 404
        connector._client.put.return_value = mock_response
        
        result = await connector.cancel_order("invalid_order_id")
        
        assert result is False


class TestAccountManagement:
    """Test account management functionality"""
    
    @pytest.mark.asyncio
    async def test_get_account_info(self, connector, mock_account_response):
        """Test account information retrieval"""
        connector.is_connected = True
        connector._client = AsyncMock()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_account_response
        connector._client.get.return_value = mock_response
        
        account_info = await connector.get_account_info()
        
        assert account_info["account_id"] == "test_account_id"
        assert account_info["currency"] == "USD"
        assert account_info["balance"] == 10000.0
        assert account_info["nav"] == 10500.0
        assert account_info["unrealized_pl"] == 500.0
        assert account_info["margin_used"] == 1000.0
        assert account_info["open_trade_count"] == 2
    
    @pytest.mark.asyncio
    async def test_get_positions(self, connector):
        """Test positions retrieval"""
        connector.is_connected = True
        connector._client = AsyncMock()
        
        mock_positions_response = {
            "positions": [
                {
                    "instrument": "EUR_USD",
                    "long": {
                        "units": "10000",
                        "averagePrice": "1.0500",
                        "unrealizedPL": "100.0000"
                    },
                    "short": {
                        "units": "0",
                        "averagePrice": "0.0000",
                        "unrealizedPL": "0.0000"
                    }
                }
            ]
        }
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_positions_response
        connector._client.get.return_value = mock_response
        
        positions = await connector.get_positions()
        
        assert len(positions) == 1
        assert positions[0]["symbol"] == "EURUSD"
        assert positions[0]["quantity"] == 10000.0
        assert positions[0]["average_cost"] == 1.0500
        assert positions[0]["unrealized_pnl"] == 100.0


class TestForexSpecificFeatures:
    """Test forex-specific features"""
    
    @pytest.mark.asyncio
    async def test_get_current_prices(self, connector, mock_pricing_response):
        """Test current price retrieval"""
        connector.is_connected = True
        connector._client = AsyncMock()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_pricing_response
        connector._client.get.return_value = mock_response
        
        prices = await connector.get_current_prices(["EURUSD"])
        
        assert "EURUSD" in prices
        assert prices["EURUSD"]["bid"] == 1.0500
        assert prices["EURUSD"]["ask"] == 1.0502
        assert prices["EURUSD"]["spread"] == 0.0002
    
    @pytest.mark.asyncio
    async def test_close_position(self, connector):
        """Test position closing"""
        connector.is_connected = True
        connector._client = AsyncMock()
        
        mock_close_response = {
            "longOrderFillTransaction": {
                "id": "close_transaction_id",
                "units": "-10000",
                "price": "1.0520"
            }
        }
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_close_response
        connector._client.put.return_value = mock_response
        
        result = await connector.close_position("EURUSD")
        
        assert isinstance(result, OrderResult)
        assert result.order_id == "close_transaction_id"
        assert result.status == "FILLED"
        assert result.filled_quantity == 10000
        assert result.avg_fill_price == 1.0520
    
    def test_symbol_formatting(self, connector):
        """Test symbol formatting for OANDA"""
        # Test format symbol
        assert connector._format_symbol("EURUSD") == "EUR_USD"
        assert connector._format_symbol("GBPJPY") == "GBP_JPY"
        assert connector._format_symbol("EUR_USD") == "EUR_USD"  # Already formatted
        
        # Test unformat symbol
        assert connector._unformat_symbol("EUR_USD") == "EURUSD"
        assert connector._unformat_symbol("GBP_JPY") == "GBPJPY"
    
    def test_forex_order_conversion(self, connector):
        """Test conversion from generic order to forex order"""
        # Test buy order
        buy_order = Order(
            symbol="EURUSD",
            side="BUY",
            quantity=10000,
            order_type="MARKET"
        )
        
        forex_order = connector._convert_to_forex_order(buy_order)
        assert forex_order.units == 10000  # Positive for buy
        assert forex_order.side == "BUY"
        
        # Test sell order
        sell_order = Order(
            symbol="EURUSD",
            side="SELL",
            quantity=5000,
            order_type="LIMIT",
            price=1.0500
        )
        
        forex_order = connector._convert_to_forex_order(sell_order)
        assert forex_order.units == -5000  # Negative for sell
        assert forex_order.side == "SELL"
        assert forex_order.price == 1.0500
    
    def test_supported_symbols(self, connector):
        """Test supported symbols list"""
        symbols = connector.get_supported_symbols()
        
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert "EURUSD" in symbols
        assert "GBPUSD" in symbols
        assert "USDJPY" in symbols


class TestForexMarketHours:
    """Test forex market hours functionality"""
    
    def test_market_open_weekday(self):
        """Test market is open on weekdays"""
        # Tuesday 15:00 UTC
        tuesday = datetime(2023, 1, 3, 15, 0, tzinfo=timezone.utc)
        assert ForexMarketHours.is_market_open(tuesday) is True
    
    def test_market_closed_saturday_night(self):
        """Test market is closed Saturday night"""
        # Saturday 23:00 UTC
        saturday_night = datetime(2023, 1, 7, 23, 0, tzinfo=timezone.utc)
        assert ForexMarketHours.is_market_open(saturday_night) is False
    
    def test_market_closed_sunday_morning(self):
        """Test market is closed Sunday morning"""
        # Sunday 10:00 UTC
        sunday_morning = datetime(2023, 1, 8, 10, 0, tzinfo=timezone.utc)
        assert ForexMarketHours.is_market_open(sunday_morning) is False
    
    def test_market_open_sunday_night(self):
        """Test market opens Sunday night"""
        # Sunday 22:00 UTC
        sunday_night = datetime(2023, 1, 8, 22, 0, tzinfo=timezone.utc)
        assert ForexMarketHours.is_market_open(sunday_night) is True
    
    def test_next_market_open(self):
        """Test next market open calculation"""
        # Saturday afternoon
        saturday = datetime(2023, 1, 7, 15, 0, tzinfo=timezone.utc)
        next_open = ForexMarketHours.get_next_market_open(saturday)
        
        # Should be next Sunday 22:00
        expected = datetime(2023, 1, 8, 22, 0, tzinfo=timezone.utc)
        assert next_open == expected


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_not_connected_error(self, connector):
        """Test that operations fail when not connected"""
        connector.is_connected = False
        
        with pytest.raises(RuntimeError, match="Not connected to OANDA"):
            await connector.get_historical_data("EURUSD", "1h", datetime.now(), datetime.now())
        
        with pytest.raises(RuntimeError, match="Not connected to OANDA"):
            await connector.place_order(Order("EURUSD", "BUY", 1000, "MARKET"))
        
        with pytest.raises(RuntimeError, match="Not connected to OANDA"):
            await connector.get_account_info()
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, connector):
        """Test API error handling"""
        connector.is_connected = True
        connector._client = AsyncMock()
        
        # Mock API error
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        connector._client.get.return_value = mock_response
        
        # Should return empty results instead of raising
        account_info = await connector.get_account_info()
        assert account_info == {}
        
        positions = await connector.get_positions()
        assert positions == []


if __name__ == "__main__":
    pytest.main([__file__])