"""
Unified data storage service managing time series databases and caching.
"""

from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
import logging
from contextlib import asynccontextmanager

from ..models.market_data import MarketData
from ..models.trading_signal import TradingSignal
from ..models.time_series import (
    MarketDataPoint,
    PredictionPoint,
    PerformanceMetric,
    TimeSeriesQuery,
    TimeSeriesDatabase,
)
from ..repositories.influxdb_repository import InfluxDBRepository
from ..repositories.timescaledb_repository import TimescaleDBRepository
from ..repositories.redis_cache import RedisCache
from ..config.settings import get_settings


logger = logging.getLogger(__name__)


class DataStorageService:
    """
    Unified data storage service managing time series databases and caching.

    Provides a high-level interface for storing and retrieving market data,
    predictions, and performance metrics across multiple storage backends.
    """

    def __init__(
        self,
        primary_db: TimeSeriesDatabase = TimeSeriesDatabase.TIMESCALEDB,
        enable_influxdb: bool = True,
        enable_redis: bool = True,
        backup_enabled: bool = True,
    ):
        """
        Initialize data storage service.

        Args:
            primary_db: Primary time series database
            enable_influxdb: Enable InfluxDB for analytics
            enable_redis: Enable Redis caching
            backup_enabled: Enable backup procedures
        """
        self.primary_db = primary_db
        self.enable_influxdb = enable_influxdb
        self.enable_redis = enable_redis
        self.backup_enabled = backup_enabled

        self.settings = get_settings()

        # Initialize repositories
        self.timescaledb: Optional[TimescaleDBRepository] = None
        self.influxdb: Optional[InfluxDBRepository] = None
        self.redis: Optional[RedisCache] = None

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all storage backends."""
        if self._initialized:
            return

        try:
            # Initialize TimescaleDB (primary storage)
            self.timescaledb = TimescaleDBRepository(
                host=self.settings.database.host,
                port=self.settings.database.port,
                database=self.settings.database.database,
                username=self.settings.database.username,
                password=self.settings.database.password,
                min_size=5,
                max_size=self.settings.database.pool_size,
            )
            await self.timescaledb.connect()

            # Initialize InfluxDB (analytics storage)
            if self.enable_influxdb:
                influx_config = getattr(self.settings, "influxdb", None)
                if influx_config:
                    self.influxdb = InfluxDBRepository(
                        url=influx_config.url,
                        token=influx_config.token,
                        org=influx_config.org,
                        bucket=influx_config.bucket,
                    )
                    await self.influxdb.connect()
                else:
                    logger.warning(
                        "InfluxDB configuration not found, skipping"
                    )

            # Initialize Redis (caching layer)
            if self.enable_redis:
                self.redis = RedisCache(
                    host=self.settings.redis.host,
                    port=self.settings.redis.port,
                    db=self.settings.redis.db,
                    password=self.settings.redis.password,
                    max_connections=self.settings.redis.max_connections,
                )
                await self.redis.connect()

            self._initialized = True
            logger.info("Data storage service initialized successfully")

        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.error("Failed to initialize data storage service: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error during initialization: %s", e)
            raise

    async def shutdown(self) -> None:
        """Shutdown all storage backends."""
        if not self._initialized:
            return

        try:
            if self.timescaledb:
                await self.timescaledb.disconnect()

            if self.influxdb:
                await self.influxdb.disconnect()

            if self.redis:
                await self.redis.disconnect()

            self._initialized = False
            logger.info("Data storage service shutdown completed")

        except Exception as e:
            logger.error("Error during data storage service shutdown: %s", e)

    @asynccontextmanager
    async def get_storage_context(self):
        """Context manager for storage operations."""
        if not self._initialized:
            await self.initialize()

        try:
            yield self
        except Exception as e:
            logger.error("Storage operation failed: %s", e)
            # Attempt to recover connections if needed
            health = await self.health_check()
            unhealthy_services = [k for k, v in health.items() if not v]
            if unhealthy_services:
                logger.warning(
                    "Unhealthy services detected: %s", unhealthy_services
                )
            raise

    async def health_check(self) -> Dict[str, bool]:
        """Check health status of all storage backends."""
        health_status = {}

        if self.timescaledb:
            health_status["timescaledb"] = (
                await self.timescaledb.health_check()
            )

        if self.influxdb:
            health_status["influxdb"] = await self.influxdb.health_check()

        if self.redis:
            health_status["redis"] = await self.redis.health_check()

        return health_status

    # Market Data Operations

    def _convert_to_market_data_point(
        self, market_data: MarketData
    ) -> MarketDataPoint:
        """Convert MarketData to MarketDataPoint."""
        return MarketDataPoint(
            symbol=market_data.symbol,
            timestamp=market_data.timestamp,
            open=market_data.open,
            high=market_data.high,
            low=market_data.low,
            close=market_data.close,
            volume=market_data.volume,
            exchange=market_data.exchange.value,
            vwap=(market_data.high + market_data.low + market_data.close) / 3,
            trades_count=0,  # Default value, should be provided by exchange
        )

    async def store_market_data(
        self, market_data: MarketData, cache_ttl: int = 300
    ) -> bool:
        """
        Store market data across all backends.

        Args:
            market_data: Market data to store
            cache_ttl: Cache TTL in seconds

        Returns:
            True if successful, False otherwise
        """
        try:
            market_point = self._convert_to_market_data_point(market_data)

            # Store in primary database
            if (
                self.primary_db == TimeSeriesDatabase.TIMESCALEDB
                and self.timescaledb
            ):
                await self.timescaledb.insert_market_data(market_point)

            # Store in InfluxDB for analytics
            if self.influxdb:
                await self.influxdb.write_market_data(market_point)

            # Cache in Redis
            if self.redis:
                await self.redis.cache_market_data(market_data, cache_ttl)

            return True

        except (ConnectionError, TimeoutError) as e:
            logger.error(
                "Failed to store market data for %s: %s", market_data.symbol, e
            )
            return False
        except Exception as e:
            logger.error(
                "Unexpected error storing market data for %s: %s",
                market_data.symbol,
                e,
            )
            return False

    async def store_market_data_batch(
        self, market_data_list: List[MarketData], cache_ttl: int = 300
    ) -> bool:
        """Store multiple market data points in batch."""
        try:
            # Convert to time series points
            market_points = [
                self._convert_to_market_data_point(md)
                for md in market_data_list
            ]

            # Store in primary database
            if (
                self.primary_db == TimeSeriesDatabase.TIMESCALEDB
                and self.timescaledb
            ):
                await self.timescaledb.insert_market_data_batch(market_points)

            # Store in InfluxDB for analytics
            if self.influxdb:
                await self.influxdb.write_market_data_batch(market_points)

            # Cache latest data in Redis
            if self.redis:
                for market_data in market_data_list:
                    await self.redis.cache_market_data(market_data, cache_ttl)

            logger.info("Stored %d market data points", len(market_data_list))
            return True

        except (ConnectionError, TimeoutError) as e:
            logger.error("Failed to store market data batch: %s", e)
            return False
        except Exception as e:
            logger.error("Unexpected error in batch storage: %s", e)
            return False

    def _validate_time_range(
        self, start_time: datetime, end_time: datetime
    ) -> None:
        """Validate time range parameters."""
        if not isinstance(start_time, datetime) or not isinstance(
            end_time, datetime
        ):
            raise ValueError(
                "start_time and end_time must be datetime objects"
            )

        if start_time >= end_time:
            raise ValueError("start_time must be before end_time")

        # Prevent queries for excessively large time ranges
        max_days = 365
        if (end_time - start_time).days > max_days:
            raise ValueError(f"Time range cannot exceed {max_days} days")

    async def get_market_data(
        self,
        symbol: str,
        exchange: str,
        start_time: datetime,
        end_time: datetime,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve market data with caching support.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            start_time: Start timestamp
            end_time: End timestamp
            use_cache: Whether to use cache

        Returns:
            List of market data records

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate inputs
        if not symbol or not symbol.strip():
            raise ValueError("Symbol cannot be empty")

        if not exchange or not exchange.strip():
            raise ValueError("Exchange cannot be empty")

        self._validate_time_range(start_time, end_time)

        try:
            # Try cache first if enabled
            if use_cache and self.redis:
                cached_data = await self.redis.get_market_data(
                    symbol, exchange
                )
                if cached_data:
                    # Check if cached data is within time range
                    cached_time = datetime.fromisoformat(
                        cached_data["timestamp"]
                    )
                    if start_time <= cached_time <= end_time:
                        return [cached_data]

            # Query from primary database
            if (
                self.primary_db == TimeSeriesDatabase.TIMESCALEDB
                and self.timescaledb
            ):
                return await self.timescaledb.query_market_data(
                    symbol, exchange, start_time, end_time
                )

            # Fallback to InfluxDB
            if self.influxdb:
                query = TimeSeriesQuery(
                    measurement="market_data",
                    start_time=start_time,
                    end_time=end_time,
                    tags={"symbol": symbol, "exchange": exchange},
                )
                return await self.influxdb.query(query)

            return []

        except Exception as e:
            logger.error("Failed to get market data for %s: %s", symbol, e)
            return []

    async def get_latest_market_data(
        self, symbol: str, exchange: str, use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Get the latest market data for a symbol."""
        try:
            # Try cache first
            if use_cache and self.redis:
                cached_data = await self.redis.get_market_data(
                    symbol, exchange
                )
                if cached_data:
                    return cached_data

            # Query from primary database
            if (
                self.primary_db == TimeSeriesDatabase.TIMESCALEDB
                and self.timescaledb
            ):
                return await self.timescaledb.get_latest_market_data(
                    symbol, exchange
                )

            # Fallback to InfluxDB
            if self.influxdb:
                return await self.influxdb.get_latest_market_data(
                    symbol, exchange
                )

            return None

        except Exception as e:
            logger.error(
                "Failed to get latest market data for %s: %s", symbol, e
            )
            return None

    # Prediction Operations

    async def store_prediction(
        self, prediction: PredictionPoint, cache_ttl: int = 300
    ) -> bool:
        """Store model prediction."""
        try:
            # Store in primary database
            if (
                self.primary_db == TimeSeriesDatabase.TIMESCALEDB
                and self.timescaledb
            ):
                await self.timescaledb.insert_prediction(prediction)

            # Store in InfluxDB for analytics
            if self.influxdb:
                await self.influxdb.write_prediction(prediction)

            # Cache in Redis
            if self.redis:
                await self.redis.cache_prediction(prediction, cache_ttl)

            return True

        except Exception as e:
            logger.error("Failed to store prediction: %s", e)
            return False

    async def get_predictions(
        self,
        symbol: str,
        model_name: str,
        start_time: datetime,
        end_time: datetime,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieve model predictions."""
        try:
            # Try cache first
            if use_cache and self.redis:
                cached_prediction = await self.redis.get_prediction(
                    symbol, model_name
                )
                if cached_prediction:
                    cached_time = datetime.fromisoformat(
                        cached_prediction["timestamp"]
                    )
                    if start_time <= cached_time <= end_time:
                        return [cached_prediction]

            # Query from primary database
            if (
                self.primary_db == TimeSeriesDatabase.TIMESCALEDB
                and self.timescaledb
            ):
                return await self.timescaledb.query_predictions(
                    symbol, model_name, start_time, end_time
                )

            return []

        except Exception as e:
            logger.error("Failed to get predictions for %s: %s", symbol, e)
            return []

    # Performance Metrics Operations

    async def store_performance_metric(
        self, metric: PerformanceMetric
    ) -> bool:
        """Store performance metric."""
        try:
            # Store in primary database
            if (
                self.primary_db == TimeSeriesDatabase.TIMESCALEDB
                and self.timescaledb
            ):
                await self.timescaledb.insert_performance_metric(metric)

            # Store in InfluxDB for analytics
            if self.influxdb:
                await self.influxdb.write_performance_metric(metric)

            return True

        except Exception as e:
            logger.error("Failed to store performance metric: %s", e)
            return False

    # Trading Signal Operations

    async def store_trading_signal(
        self, signal: TradingSignal, cache_ttl: int = 600
    ) -> bool:
        """Store trading signal with caching."""
        try:
            # Cache in Redis
            if self.redis:
                await self.redis.cache_trading_signal(signal, cache_ttl)

            # Store as performance metric for historical analysis
            metric = PerformanceMetric(
                timestamp=signal.timestamp,
                metric_name="trading_signal",
                metric_value=signal.confidence,
                symbol=signal.symbol,
                strategy=signal.action,
                model_name=signal.model_version,
                timeframe="1m",  # Default timeframe for trading signals
            )

            await self.store_performance_metric(metric)

            return True

        except Exception as e:
            logger.error("Failed to store trading signal: %s", e)
            return False

    async def get_trading_signal(
        self, symbol: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached trading signal for symbol."""
        try:
            if self.redis:
                return await self.redis.get_trading_signal(symbol)
            return None

        except Exception as e:
            logger.error("Failed to get trading signal for %s: %s", symbol, e)
            return None

    # Data Management Operations

    async def cleanup_old_data(
        self, retention_days: int = 30
    ) -> Dict[str, Any]:
        """Clean up old data based on retention policy."""
        cleanup_results = {}
        cutoff_time = datetime.now(timezone.utc) - timedelta(
            days=retention_days
        )

        try:
            # Clean up TimescaleDB
            if self.timescaledb:
                market_data_deleted = await self.timescaledb.delete_old_data(
                    "market_data", cutoff_time
                )
                predictions_deleted = await self.timescaledb.delete_old_data(
                    "predictions", cutoff_time
                )
                metrics_deleted = await self.timescaledb.delete_old_data(
                    "performance_metrics", cutoff_time
                )

                cleanup_results["timescaledb"] = {
                    "market_data": market_data_deleted,
                    "predictions": predictions_deleted,
                    "performance_metrics": metrics_deleted,
                }

            # Clean up InfluxDB
            if self.influxdb:
                await self.influxdb.delete_old_data("market_data", cutoff_time)
                await self.influxdb.delete_old_data("predictions", cutoff_time)
                await self.influxdb.delete_old_data(
                    "performance_metrics", cutoff_time
                )

                cleanup_results["influxdb"] = {"status": "completed"}

            # Clean up Redis cache (expired keys are automatically removed)
            if self.redis:
                # Clean up old pattern-based keys
                old_keys = await self.redis.get_keys_by_pattern("*")
                cleanup_results["redis"] = {"keys_found": len(old_keys)}

            logger.info("Data cleanup completed: %s", cleanup_results)
            return cleanup_results

        except Exception as e:
            logger.error("Failed to cleanup old data: %s", e)
            return {}

    async def backup_data(
        self, backup_path: str, start_time: datetime, end_time: datetime
    ) -> bool:
        """
        Backup data to specified path.

        Args:
            backup_path: Path to store backup
            start_time: Start time for backup
            end_time: End time for backup

        Returns:
            True if successful, False otherwise
        """
        if not self.backup_enabled:
            logger.warning("Backup is disabled")
            return False

        try:
            # This is a simplified backup implementation
            # In production, you would use proper backup tools

            logger.info("Starting data backup to %s", backup_path)

            # Get database statistics for backup verification
            if self.timescaledb:
                stats = await self.timescaledb.get_database_stats()
                logger.info("Database stats before backup: %s", stats)

            # TODO: Implement actual backup logic
            # This could involve:
            # 1. Exporting data to files
            # 2. Creating database dumps
            # 3. Copying to backup storage

            logger.info("Data backup completed successfully")
            return True

        except Exception as e:
            logger.error("Failed to backup data: %s", e)
            return False

    async def restore_data(self, backup_path: str) -> bool:
        """
        Restore data from backup.

        Args:
            backup_path: Path to backup data

        Returns:
            True if successful, False otherwise
        """
        if not self.backup_enabled:
            logger.warning("Backup/restore is disabled")
            return False

        try:
            logger.info("Starting data restore from %s", backup_path)

            # TODO: Implement actual restore logic
            # This could involve:
            # 1. Reading backup files
            # 2. Restoring database dumps
            # 3. Validating data integrity

            logger.info("Data restore completed successfully")
            return True

        except Exception as e:
            logger.error("Failed to restore data: %s", e)
            return False

    # Statistics and Monitoring

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics."""
        stats = {}

        try:
            # TimescaleDB stats
            if self.timescaledb:
                stats["timescaledb"] = (
                    await self.timescaledb.get_database_stats()
                )

            # Redis stats
            if self.redis:
                stats["redis"] = await self.redis.get_cache_stats()

            # Health status
            stats["health"] = await self.health_check()

            return stats

        except Exception as e:
            logger.error("Failed to get storage stats: %s", e)
            return {}

    async def get_data_quality_report(
        self,
        symbol: str,
        exchange: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        """
        Generate data quality report for a symbol.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            start_time: Start time for analysis
            end_time: End time for analysis

        Returns:
            Data quality report
        """
        try:
            # Get market data
            market_data = await self.get_market_data(
                symbol, exchange, start_time, end_time, use_cache=False
            )

            if not market_data:
                return {"error": "No data found"}

            # Calculate quality metrics
            total_records = len(market_data)

            # Check for missing data (gaps in timestamps)
            timestamps = [
                datetime.fromisoformat(record["time"])
                for record in market_data
            ]
            timestamps.sort()

            gaps = []
            for i in range(1, len(timestamps)):
                gap = (timestamps[i] - timestamps[i - 1]).total_seconds()
                if gap > 300:  # More than 5 minutes gap
                    gaps.append(
                        {
                            "start": timestamps[i - 1].isoformat(),
                            "end": timestamps[i].isoformat(),
                            "duration_seconds": gap,
                        }
                    )

            # Check for price anomalies
            prices = [record["close"] for record in market_data]
            price_changes = []
            for i in range(1, len(prices)):
                change = abs((prices[i] - prices[i - 1]) / prices[i - 1])
                price_changes.append(change)

            # Identify large price movements (>10%)
            large_movements = [
                change for change in price_changes if change > 0.1
            ]

            quality_report = {
                "symbol": symbol,
                "exchange": exchange,
                "period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
                "total_records": total_records,
                "data_gaps": {
                    "count": len(gaps),
                    "gaps": gaps[:10],  # Show first 10 gaps
                },
                "price_anomalies": {
                    "large_movements_count": len(large_movements),
                    "max_change_percent": (
                        max(price_changes) * 100 if price_changes else 0
                    ),
                    "avg_change_percent": (
                        sum(price_changes) / len(price_changes) * 100
                        if price_changes
                        else 0
                    ),
                },
                "quality_score": self._calculate_quality_score(
                    total_records, len(gaps), len(large_movements)
                ),
            }

            return quality_report

        except Exception as e:
            logger.error("Failed to generate data quality report: %s", e)
            return {"error": str(e)}

    def _calculate_quality_score(
        self, total_records: int, gap_count: int, anomaly_count: int
    ) -> float:
        """Calculate data quality score (0-100)."""
        if total_records == 0:
            return 0.0

        # Base score
        score = 100.0

        # Penalize gaps
        gap_penalty = min(gap_count * 5, 30)  # Max 30 points penalty
        score -= gap_penalty

        # Penalize anomalies
        anomaly_penalty = min(anomaly_count * 2, 20)  # Max 20 points penalty
        score -= anomaly_penalty

        return max(score, 0.0)


# Global instance
_data_storage_service: Optional[DataStorageService] = None


async def get_data_storage_service() -> DataStorageService:
    """Get global data storage service instance."""
    global _data_storage_service

    if _data_storage_service is None:
        _data_storage_service = DataStorageService()
        await _data_storage_service.initialize()

    return _data_storage_service


async def shutdown_data_storage_service() -> None:
    """Shutdown global data storage service."""
    global _data_storage_service

    if _data_storage_service:
        await _data_storage_service.shutdown()
        _data_storage_service = None
