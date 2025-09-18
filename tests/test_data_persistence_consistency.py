"""
Tests for data persistence and cache consistency.
"""
import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from src.services.data_storage_service import DataStorageService
from src.models.market_data import MarketData, ExchangeType
from src.models.time_series import MarketDataPoint, PredictionPoint, PerformanceMetric


class TestDataPersistence:
    """Test data persistence across different storage backends."""
    
    @pytest.fixture
    def mock_repositories(self):
        """Mock repositories for testing."""
        with patch('src.services.data_storage_service.TimescaleDBRepository') as mock_timescale, \
             patch('src.services.data_storage_service.InfluxDBRepository') as mock_influx, \
             patch('src.services.data_storage_service.RedisCache') as mock_redis:
            
            # Setup mock instances
            mock_timescale_instance = AsyncMock()
            mock_influx_instance = AsyncMock()
            mock_redis_instance = AsyncMock()
            
            mock_timescale.return_value = mock_timescale_instance
            mock_influx.return_value = mock_influx_instance
            mock_redis.return_value = mock_redis_instance
            
            # Mock health checks
            mock_timescale_instance.health_check.return_value = True
            mock_influx_instance.health_check.return_value = True
            mock_redis_instance.health_check.return_value = True
            
            yield {
                'timescale': mock_timescale_instance,
                'influx': mock_influx_instance,
                'redis': mock_redis_instance
            }
    
    @pytest.mark.asyncio
    async def test_market_data_persistence_consistency(self, mock_repositories):
        """Test that market data is consistently stored across all backends."""
        service = DataStorageService()
        await service.initialize()
        
        # Create test market data
        market_data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000.0,
            exchange=ExchangeType.ROBINHOOD
        )
        
        # Store market data
        result = await service.store_market_data(market_data)
        assert result is True
        
        # Verify data was stored in TimescaleDB
        mock_repositories['timescale'].insert_market_data.assert_called_once()
        timescale_call_args = mock_repositories['timescale'].insert_market_data.call_args[0][0]
        assert isinstance(timescale_call_args, MarketDataPoint)
        assert timescale_call_args.symbol == "AAPL"
        assert timescale_call_args.close == 151.0
        
        # Verify data was stored in InfluxDB
        mock_repositories['influx'].write_market_data.assert_called_once()
        influx_call_args = mock_repositories['influx'].write_market_data.call_args[0][0]
        assert isinstance(influx_call_args, MarketDataPoint)
        assert influx_call_args.symbol == "AAPL"
        assert influx_call_args.close == 151.0
        
        # Verify data was cached in Redis
        mock_repositories['redis'].cache_market_data.assert_called_once()
        redis_call_args = mock_repositories['redis'].cache_market_data.call_args[0][0]
        assert isinstance(redis_call_args, MarketData)
        assert redis_call_args.symbol == "AAPL"
        assert redis_call_args.close == 151.0
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_batch_persistence_consistency(self, mock_repositories):
        """Test batch operations maintain consistency across backends."""
        service = DataStorageService()
        await service.initialize()
        
        # Create batch of market data
        market_data_list = []
        for i in range(5):
            market_data = MarketData(
                symbol=f"STOCK{i}",
                timestamp=datetime.now(timezone.utc) + timedelta(minutes=i),
                open=100.0 + i,
                high=102.0 + i,
                low=99.0 + i,
                close=101.0 + i,
                volume=1000000.0 + i * 1000,
                exchange=ExchangeType.ROBINHOOD
            )
            market_data_list.append(market_data)
        
        # Store batch
        result = await service.store_market_data_batch(market_data_list)
        assert result is True
        
        # Verify batch was stored in TimescaleDB
        mock_repositories['timescale'].insert_market_data_batch.assert_called_once()
        timescale_batch = mock_repositories['timescale'].insert_market_data_batch.call_args[0][0]
        assert len(timescale_batch) == 5
        assert all(isinstance(item, MarketDataPoint) for item in timescale_batch)
        
        # Verify batch was stored in InfluxDB
        mock_repositories['influx'].write_market_data_batch.assert_called_once()
        influx_batch = mock_repositories['influx'].write_market_data_batch.call_args[0][0]
        assert len(influx_batch) == 5
        assert all(isinstance(item, MarketDataPoint) for item in influx_batch)
        
        # Verify individual items were cached in Redis
        assert mock_repositories['redis'].cache_market_data.call_count == 5
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_prediction_persistence_consistency(self, mock_repositories):
        """Test prediction data persistence across backends."""
        service = DataStorageService()
        await service.initialize()
        
        # Create test prediction
        prediction = PredictionPoint(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            model_name="cnn_lstm_hybrid",
            model_version="v1.0",
            predicted_price=155.0,
            predicted_direction="BUY",
            confidence_score=0.85,
            uncertainty=0.1,
            feature_importance={"rsi": 0.3, "macd": 0.2}
        )
        
        # Store prediction
        result = await service.store_prediction(prediction)
        assert result is True
        
        # Verify prediction was stored in all backends
        mock_repositories['timescale'].insert_prediction.assert_called_once()
        mock_repositories['influx'].write_prediction.assert_called_once()
        mock_repositories['redis'].cache_prediction.assert_called_once()
        
        # Verify data consistency
        for repo_name in ['timescale', 'influx', 'redis']:
            if repo_name == 'timescale':
                call_args = mock_repositories[repo_name].insert_prediction.call_args[0][0]
            elif repo_name == 'influx':
                call_args = mock_repositories[repo_name].write_prediction.call_args[0][0]
            else:  # redis
                call_args = mock_repositories[repo_name].cache_prediction.call_args[0][0]
            
            assert call_args.symbol == "AAPL"
            assert call_args.predicted_price == 155.0
            assert call_args.confidence_score == 0.85
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_partial_failure_handling(self, mock_repositories):
        """Test handling of partial failures across backends."""
        service = DataStorageService()
        await service.initialize()
        
        # Mock TimescaleDB failure
        mock_repositories['timescale'].insert_market_data.side_effect = Exception("TimescaleDB error")
        
        # Create test market data
        market_data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000.0,
            exchange=ExchangeType.ROBINHOOD
        )
        
        # Store should fail due to TimescaleDB error
        result = await service.store_market_data(market_data)
        assert result is False
        
        # Verify TimescaleDB was attempted
        mock_repositories['timescale'].insert_market_data.assert_called_once()
        
        # Other backends should not be called due to early failure
        mock_repositories['influx'].write_market_data.assert_not_called()
        mock_repositories['redis'].cache_market_data.assert_not_called()
        
        await service.shutdown()


class TestCacheConsistency:
    """Test cache consistency and invalidation."""
    
    @pytest.fixture
    def mock_repositories(self):
        """Mock repositories for testing."""
        with patch('src.services.data_storage_service.TimescaleDBRepository') as mock_timescale, \
             patch('src.services.data_storage_service.InfluxDBRepository') as mock_influx, \
             patch('src.services.data_storage_service.RedisCache') as mock_redis:
            
            # Setup mock instances
            mock_timescale_instance = AsyncMock()
            mock_influx_instance = AsyncMock()
            mock_redis_instance = AsyncMock()
            
            mock_timescale.return_value = mock_timescale_instance
            mock_influx.return_value = mock_influx_instance
            mock_redis.return_value = mock_redis_instance
            
            # Mock health checks
            mock_timescale_instance.health_check.return_value = True
            mock_influx_instance.health_check.return_value = True
            mock_redis_instance.health_check.return_value = True
            
            yield {
                'timescale': mock_timescale_instance,
                'influx': mock_influx_instance,
                'redis': mock_redis_instance
            }
    
    @pytest.mark.asyncio
    async def test_cache_hit_consistency(self, mock_repositories):
        """Test that cache hits return consistent data."""
        service = DataStorageService()
        await service.initialize()
        
        # Mock cached data
        now = datetime.now(timezone.utc)
        cached_data = {
            'symbol': 'AAPL',
            'timestamp': now.isoformat(),
            'open': 150.0,
            'high': 152.0,
            'low': 149.0,
            'close': 151.0,
            'volume': 1000000.0,
            'exchange': 'robinhood'
        }
        mock_repositories['redis'].get_market_data.return_value = cached_data
        
        # Query data within cache timestamp range
        start_time = now - timedelta(minutes=5)
        end_time = now + timedelta(minutes=5)
        
        result = await service.get_market_data('AAPL', 'robinhood', start_time, end_time)
        
        # Should return cached data
        assert len(result) == 1
        assert result[0] == cached_data
        
        # Database should not be queried
        mock_repositories['timescale'].query_market_data.assert_not_called()
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_cache_miss_fallback(self, mock_repositories):
        """Test fallback to database when cache misses."""
        service = DataStorageService()
        await service.initialize()
        
        # Mock cache miss
        mock_repositories['redis'].get_market_data.return_value = None
        
        # Mock database data
        db_data = [
            {
                'time': datetime.now(timezone.utc).isoformat(),
                'symbol': 'AAPL',
                'close': 151.0
            }
        ]
        mock_repositories['timescale'].query_market_data.return_value = db_data
        
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)
        
        result = await service.get_market_data('AAPL', 'robinhood', start_time, end_time)
        
        # Should return database data
        assert result == db_data
        
        # Verify cache was checked first
        mock_repositories['redis'].get_market_data.assert_called_once()
        
        # Verify database was queried
        mock_repositories['timescale'].query_market_data.assert_called_once()
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_cache_invalidation_on_new_data(self, mock_repositories):
        """Test that new data properly updates cache."""
        service = DataStorageService()
        await service.initialize()
        
        # Store new market data
        market_data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000.0,
            exchange=ExchangeType.ROBINHOOD
        )
        
        await service.store_market_data(market_data)
        
        # Verify cache was updated with new data
        mock_repositories['redis'].cache_market_data.assert_called_once()
        cached_data = mock_repositories['redis'].cache_market_data.call_args[0][0]
        assert cached_data.symbol == "AAPL"
        assert cached_data.close == 151.0
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_cache_ttl_consistency(self, mock_repositories):
        """Test that cache TTL is consistently applied."""
        service = DataStorageService()
        await service.initialize()
        
        # Store market data with custom TTL
        market_data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000.0,
            exchange=ExchangeType.ROBINHOOD
        )
        
        custom_ttl = 600  # 10 minutes
        await service.store_market_data(market_data, cache_ttl=custom_ttl)
        
        # Verify TTL was passed to cache
        mock_repositories['redis'].cache_market_data.assert_called_once()
        call_args = mock_repositories['redis'].cache_market_data.call_args
        assert call_args[1] == custom_ttl  # TTL should be second argument
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_cache_error_fallback(self, mock_repositories):
        """Test fallback when cache operations fail."""
        service = DataStorageService()
        await service.initialize()
        
        # Mock cache error
        mock_repositories['redis'].get_market_data.side_effect = Exception("Redis error")
        
        # Mock database data
        db_data = [{'symbol': 'AAPL', 'close': 151.0}]
        mock_repositories['timescale'].query_market_data.return_value = db_data
        
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)
        
        result = await service.get_market_data('AAPL', 'robinhood', start_time, end_time)
        
        # Should still return data from database
        assert result == db_data
        
        # Verify database was queried despite cache error
        mock_repositories['timescale'].query_market_data.assert_called_once()
        
        await service.shutdown()


class TestDataIntegrity:
    """Test data integrity across storage backends."""
    
    @pytest.fixture
    def mock_repositories(self):
        """Mock repositories for testing."""
        with patch('src.services.data_storage_service.TimescaleDBRepository') as mock_timescale, \
             patch('src.services.data_storage_service.InfluxDBRepository') as mock_influx, \
             patch('src.services.data_storage_service.RedisCache') as mock_redis:
            
            # Setup mock instances
            mock_timescale_instance = AsyncMock()
            mock_influx_instance = AsyncMock()
            mock_redis_instance = AsyncMock()
            
            mock_timescale.return_value = mock_timescale_instance
            mock_influx.return_value = mock_influx_instance
            mock_redis.return_value = mock_redis_instance
            
            # Mock health checks
            mock_timescale_instance.health_check.return_value = True
            mock_influx_instance.health_check.return_value = True
            mock_redis_instance.health_check.return_value = True
            
            yield {
                'timescale': mock_timescale_instance,
                'influx': mock_influx_instance,
                'redis': mock_redis_instance
            }
    
    @pytest.mark.asyncio
    async def test_data_validation_before_storage(self, mock_repositories):
        """Test that data is validated before storage."""
        service = DataStorageService()
        await service.initialize()
        
        # Create invalid market data (high < low)
        try:
            invalid_market_data = MarketData(
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc),
                open=150.0,
                high=149.0,  # Invalid: high < low
                low=151.0,
                close=150.5,
                volume=1000000.0,
                exchange=ExchangeType.ROBINHOOD
            )
            # This should raise a validation error
            assert False, "Should have raised validation error"
        except ValueError:
            # Expected validation error
            pass
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_timestamp_consistency(self, mock_repositories):
        """Test that timestamps are consistently handled across backends."""
        service = DataStorageService()
        await service.initialize()
        
        # Create market data with timezone-naive timestamp
        naive_timestamp = datetime(2023, 12, 1, 15, 30, 0)  # No timezone
        
        market_data = MarketData(
            symbol="AAPL",
            timestamp=naive_timestamp,
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000.0,
            exchange=ExchangeType.ROBINHOOD
        )
        
        await service.store_market_data(market_data)
        
        # Verify that all backends received timezone-aware timestamps
        timescale_call = mock_repositories['timescale'].insert_market_data.call_args[0][0]
        assert timescale_call.timestamp.tzinfo is not None
        
        influx_call = mock_repositories['influx'].write_market_data.call_args[0][0]
        assert influx_call.timestamp.tzinfo is not None
        
        redis_call = mock_repositories['redis'].cache_market_data.call_args[0][0]
        assert redis_call.timestamp.tzinfo is not None
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_concurrent_access_consistency(self, mock_repositories):
        """Test data consistency under concurrent access."""
        service = DataStorageService()
        await service.initialize()
        
        # Create multiple market data points
        market_data_list = []
        for i in range(10):
            market_data = MarketData(
                symbol=f"STOCK{i}",
                timestamp=datetime.now(timezone.utc) + timedelta(seconds=i),
                open=100.0 + i,
                high=102.0 + i,
                low=99.0 + i,
                close=101.0 + i,
                volume=1000000.0,
                exchange=ExchangeType.ROBINHOOD
            )
            market_data_list.append(market_data)
        
        # Store concurrently
        tasks = [service.store_market_data(md) for md in market_data_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All operations should succeed
        assert all(result is True for result in results if not isinstance(result, Exception))
        
        # Verify all data was stored
        assert mock_repositories['timescale'].insert_market_data.call_count == 10
        assert mock_repositories['influx'].write_market_data.call_count == 10
        assert mock_repositories['redis'].cache_market_data.call_count == 10
        
        await service.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])