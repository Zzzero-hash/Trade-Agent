"""
Tests for data storage service.
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from src.services.data_storage_service import (
    DataStorageService,
    get_data_storage_service,
)
from src.models.market_data import MarketData, ExchangeType
from src.models.trading_signal import TradingSignal
from src.models.time_series import (
    PredictionPoint,
    PerformanceMetric,
    TimeSeriesDatabase,
)


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return MarketData(
        symbol="AAPL",
        timestamp=datetime.now(timezone.utc),
        open=150.0,
        high=152.0,
        low=149.0,
        close=151.0,
        volume=1000000.0,
        exchange=ExchangeType.ROBINHOOD,
    )


@pytest.fixture
def sample_trading_signal():
    """Sample trading signal for testing."""
    return TradingSignal(
        symbol="AAPL",
        action="BUY",
        confidence=0.85,
        position_size=0.1,
        target_price=155.0,
        stop_loss=145.0,
        timestamp=datetime.now(timezone.utc),
        model_version="v1.0",
    )


@pytest.fixture
def sample_prediction():
    """Sample prediction for testing."""
    return PredictionPoint(
        symbol="AAPL",
        timestamp=datetime.now(timezone.utc),
        model_name="cnn_lstm_hybrid",
        model_version="v1.0",
        predicted_price=155.0,
        predicted_direction="BUY",
        confidence_score=0.85,
        uncertainty=0.1,
        feature_importance={"rsi": 0.3, "macd": 0.2},
    )


@pytest.fixture
def sample_performance_metric():
    """Sample performance metric for testing."""
    return PerformanceMetric(
        timestamp=datetime.now(timezone.utc),
        metric_name="sharpe_ratio",
        metric_value=1.5,
        symbol="AAPL",
        strategy="momentum",
        model_name="cnn_lstm_hybrid",
    )


class TestDataStorageService:
    """Test cases for DataStorageService."""

    @pytest.fixture
    def mock_repositories(self):
        """Mock repositories for testing."""
        with (
            patch(
                "src.services.data_storage_service.TimescaleDBRepository"
            ) as mock_timescale,
            patch(
                "src.services.data_storage_service.InfluxDBRepository"
            ) as mock_influx,
            patch(
                "src.services.data_storage_service.RedisCache"
            ) as mock_redis,
        ):
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
                "timescale": mock_timescale_instance,
                "influx": mock_influx_instance,
                "redis": mock_redis_instance,
            }

    @pytest.mark.asyncio
    async def test_initialization(self, mock_repositories):
        """Test service initialization."""
        service = DataStorageService()
        await service.initialize()

        assert service._initialized is True
        assert service.timescaledb is not None
        assert service.influxdb is not None
        assert service.redis is not None

        # Verify connections were established
        mock_repositories["timescale"].connect.assert_called_once()
        mock_repositories["influx"].connect.assert_called_once()
        mock_repositories["redis"].connect.assert_called_once()

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown(self, mock_repositories):
        """Test service shutdown."""
        service = DataStorageService()
        await service.initialize()
        await service.shutdown()

        assert service._initialized is False

        # Verify disconnections were called
        mock_repositories["timescale"].disconnect.assert_called_once()
        mock_repositories["influx"].disconnect.assert_called_once()
        mock_repositories["redis"].disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check(self, mock_repositories):
        """Test health check functionality."""
        service = DataStorageService()
        await service.initialize()

        health_status = await service.health_check()

        assert "timescaledb" in health_status
        assert "influxdb" in health_status
        assert "redis" in health_status
        assert all(health_status.values())

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_store_market_data(
        self, mock_repositories, sample_market_data
    ):
        """Test storing market data."""
        service = DataStorageService()
        await service.initialize()

        result = await service.store_market_data(sample_market_data)

        assert result is True

        # Verify data was stored in all backends
        mock_repositories["timescale"].insert_market_data.assert_called_once()
        mock_repositories["influx"].write_market_data.assert_called_once()
        mock_repositories["redis"].cache_market_data.assert_called_once()

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_store_market_data_batch(
        self, mock_repositories, sample_market_data
    ):
        """Test storing market data in batch."""
        service = DataStorageService()
        await service.initialize()

        market_data_list = [sample_market_data] * 5
        result = await service.store_market_data_batch(market_data_list)

        assert result is True

        # Verify batch operations were called
        mock_repositories[
            "timescale"
        ].insert_market_data_batch.assert_called_once()
        mock_repositories[
            "influx"
        ].write_market_data_batch.assert_called_once()

        # Verify individual cache operations
        assert mock_repositories["redis"].cache_market_data.call_count == 5

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_get_market_data_from_cache(self, mock_repositories):
        """Test retrieving market data from cache."""
        service = DataStorageService()
        await service.initialize()

        # Mock cached data
        cached_data = {
            "symbol": "AAPL",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "close": 151.0,
        }
        mock_repositories["redis"].get_market_data.return_value = cached_data

        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc) + timedelta(hours=1)

        result = await service.get_market_data(
            "AAPL", "robinhood", start_time, end_time
        )

        assert len(result) == 1
        assert result[0] == cached_data

        # Verify cache was checked
        mock_repositories["redis"].get_market_data.assert_called_once_with(
            "AAPL", "robinhood"
        )

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_get_market_data_from_database(self, mock_repositories):
        """Test retrieving market data from database when cache misses."""
        service = DataStorageService()
        await service.initialize()

        # Mock cache miss
        mock_repositories["redis"].get_market_data.return_value = None

        # Mock database data
        db_data = [{"symbol": "AAPL", "close": 151.0}]
        mock_repositories["timescale"].query_market_data.return_value = db_data

        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)

        result = await service.get_market_data(
            "AAPL", "robinhood", start_time, end_time
        )

        assert result == db_data

        # Verify database was queried
        mock_repositories["timescale"].query_market_data.assert_called_once()

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_store_prediction(
        self, mock_repositories, sample_prediction
    ):
        """Test storing prediction."""
        service = DataStorageService()
        await service.initialize()

        result = await service.store_prediction(sample_prediction)

        assert result is True

        # Verify prediction was stored in all backends
        mock_repositories["timescale"].insert_prediction.assert_called_once()
        mock_repositories["influx"].write_prediction.assert_called_once()
        mock_repositories["redis"].cache_prediction.assert_called_once()

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_store_performance_metric(
        self, mock_repositories, sample_performance_metric
    ):
        """Test storing performance metric."""
        service = DataStorageService()
        await service.initialize()

        result = await service.store_performance_metric(
            sample_performance_metric
        )

        assert result is True

        # Verify metric was stored
        mock_repositories[
            "timescale"
        ].insert_performance_metric.assert_called_once()
        mock_repositories[
            "influx"
        ].write_performance_metric.assert_called_once()

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_store_trading_signal(
        self, mock_repositories, sample_trading_signal
    ):
        """Test storing trading signal."""
        service = DataStorageService()
        await service.initialize()

        result = await service.store_trading_signal(sample_trading_signal)

        assert result is True

        # Verify signal was cached and stored as metric
        mock_repositories["redis"].cache_trading_signal.assert_called_once()
        mock_repositories[
            "timescale"
        ].insert_performance_metric.assert_called_once()

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_get_trading_signal(self, mock_repositories):
        """Test retrieving trading signal."""
        service = DataStorageService()
        await service.initialize()

        # Mock cached signal
        cached_signal = {"symbol": "AAPL", "action": "BUY", "confidence": 0.85}
        mock_repositories["redis"].get_trading_signal.return_value = (
            cached_signal
        )

        result = await service.get_trading_signal("AAPL")

        assert result == cached_signal
        mock_repositories["redis"].get_trading_signal.assert_called_once_with(
            "AAPL"
        )

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, mock_repositories):
        """Test data cleanup functionality."""
        service = DataStorageService()
        await service.initialize()

        # Mock cleanup results
        mock_repositories["timescale"].delete_old_data.return_value = 100
        mock_repositories["redis"].get_keys_by_pattern.return_value = [
            "key1",
            "key2",
        ]

        result = await service.cleanup_old_data(retention_days=30)

        assert "timescaledb" in result
        assert "influxdb" in result
        assert "redis" in result

        # Verify cleanup operations were called
        assert (
            mock_repositories["timescale"].delete_old_data.call_count == 3
        )  # 3 tables
        assert (
            mock_repositories["influx"].delete_old_data.call_count == 3
        )  # 3 measurements

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_get_storage_stats(self, mock_repositories):
        """Test getting storage statistics."""
        service = DataStorageService()
        await service.initialize()

        # Mock stats
        db_stats = {"total_size": "100MB", "row_counts": {"market_data": 1000}}
        cache_stats = {"used_memory": 1024, "keyspace_hits": 500}

        mock_repositories["timescale"].get_database_stats.return_value = (
            db_stats
        )
        mock_repositories["redis"].get_cache_stats.return_value = cache_stats

        result = await service.get_storage_stats()

        assert "timescaledb" in result
        assert "redis" in result
        assert "health" in result
        assert result["timescaledb"] == db_stats
        assert result["redis"] == cache_stats

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_data_quality_report(self, mock_repositories):
        """Test data quality report generation."""
        service = DataStorageService()
        await service.initialize()

        # Mock market data with some gaps and anomalies
        now = datetime.now(timezone.utc)
        mock_data = [
            {"time": now.isoformat(), "close": 100.0},
            {"time": (now + timedelta(minutes=1)).isoformat(), "close": 101.0},
            {
                "time": (now + timedelta(minutes=10)).isoformat(),
                "close": 120.0,
            },  # Large gap and price jump
            {
                "time": (now + timedelta(minutes=11)).isoformat(),
                "close": 121.0,
            },
        ]

        mock_repositories["timescale"].query_market_data.return_value = (
            mock_data
        )

        start_time = now - timedelta(hours=1)
        end_time = now + timedelta(hours=1)

        result = await service.get_data_quality_report(
            "AAPL", "robinhood", start_time, end_time
        )

        assert "symbol" in result
        assert "total_records" in result
        assert "data_gaps" in result
        assert "price_anomalies" in result
        assert "quality_score" in result

        assert result["symbol"] == "AAPL"
        assert result["total_records"] == 4
        assert result["data_gaps"]["count"] > 0  # Should detect the gap
        assert (
            result["price_anomalies"]["large_movements_count"] > 0
        )  # Should detect price jump

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_backup_data(self, mock_repositories):
        """Test data backup functionality."""
        service = DataStorageService(backup_enabled=True)
        await service.initialize()

        # Mock database stats
        mock_repositories["timescale"].get_database_stats.return_value = {
            "size": "100MB"
        }

        start_time = datetime.now(timezone.utc) - timedelta(days=1)
        end_time = datetime.now(timezone.utc)

        result = await service.backup_data("/tmp/backup", start_time, end_time)

        assert result is True
        mock_repositories["timescale"].get_database_stats.assert_called_once()

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_backup_disabled(self, mock_repositories):
        """Test backup when disabled."""
        service = DataStorageService(backup_enabled=False)
        await service.initialize()

        start_time = datetime.now(timezone.utc) - timedelta(days=1)
        end_time = datetime.now(timezone.utc)

        result = await service.backup_data("/tmp/backup", start_time, end_time)

        assert result is False

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_error_handling_store_market_data(
        self, mock_repositories, sample_market_data
    ):
        """Test error handling in store_market_data."""
        service = DataStorageService()
        await service.initialize()

        # Mock database error
        mock_repositories["timescale"].insert_market_data.side_effect = (
            Exception("Database error")
        )

        result = await service.store_market_data(sample_market_data)

        assert result is False

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_error_handling_get_market_data(self, mock_repositories):
        """Test error handling in get_market_data."""
        service = DataStorageService()
        await service.initialize()

        # Mock database error
        mock_repositories["redis"].get_market_data.side_effect = Exception(
            "Cache error"
        )
        mock_repositories["timescale"].query_market_data.side_effect = (
            Exception("Database error")
        )

        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)

        result = await service.get_market_data(
            "AAPL", "robinhood", start_time, end_time
        )

        assert result == []

        await service.shutdown()


class TestDataStorageServiceIntegration:
    """Integration tests for DataStorageService."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using service as context manager."""
        service = DataStorageService()

        with (
            patch("src.services.data_storage_service.TimescaleDBRepository"),
            patch("src.services.data_storage_service.InfluxDBRepository"),
            patch("src.services.data_storage_service.RedisCache"),
        ):
            async with service.get_storage_context() as storage:
                assert storage._initialized is True
                assert isinstance(storage, DataStorageService)

    @pytest.mark.asyncio
    async def test_global_service_instance(self):
        """Test global service instance management."""
        with (
            patch("src.services.data_storage_service.TimescaleDBRepository"),
            patch("src.services.data_storage_service.InfluxDBRepository"),
            patch("src.services.data_storage_service.RedisCache"),
        ):
            # Get service instance
            service1 = await get_data_storage_service()
            service2 = await get_data_storage_service()

            # Should be the same instance
            assert service1 is service2
            assert service1._initialized is True

            # Cleanup
            from src.services.data_storage_service import (
                shutdown_data_storage_service,
            )

            await shutdown_data_storage_service()


class TestDataStorageServiceConfiguration:
    """Test configuration-related functionality."""

    @pytest.mark.asyncio
    async def test_different_primary_database(self):
        """Test using different primary database."""
        with (
            patch("src.services.data_storage_service.TimescaleDBRepository"),
            patch("src.services.data_storage_service.InfluxDBRepository"),
            patch("src.services.data_storage_service.RedisCache"),
        ):
            service = DataStorageService(
                primary_db=TimeSeriesDatabase.INFLUXDB
            )
            await service.initialize()

            assert service.primary_db == TimeSeriesDatabase.INFLUXDB

            await service.shutdown()

    @pytest.mark.asyncio
    async def test_disabled_components(self):
        """Test service with disabled components."""
        with patch(
            "src.services.data_storage_service.TimescaleDBRepository"
        ) as mock_timescale:
            mock_timescale_instance = AsyncMock()
            mock_timescale.return_value = mock_timescale_instance

            service = DataStorageService(
                enable_influxdb=False, enable_redis=False
            )
            await service.initialize()

            assert service.influxdb is None
            assert service.redis is None
            assert service.timescaledb is not None

            await service.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])
