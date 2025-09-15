"""
TimescaleDB repository for time series data storage and retrieval.
"""
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import logging
from contextlib import asynccontextmanager

import asyncpg
from asyncpg import Connection, Pool
from asyncpg.exceptions import PostgresError

from ..models.time_series import (
    TimeSeriesPoint, 
    TimeSeriesQuery, 
    MarketDataPoint,
    PredictionPoint,
    PerformanceMetric
)


logger = logging.getLogger(__name__)


class TimescaleDBRepository:
    """
    Repository for TimescaleDB time series operations.
    
    Provides async interface for storing and querying time series data
    using PostgreSQL with TimescaleDB extension.
    """
    
    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        username: str,
        password: str,
        min_size: int = 5,
        max_size: int = 20,
        command_timeout: float = 60.0,
        server_settings: Optional[Dict[str, str]] = None
    ):
        """
        Initialize TimescaleDB repository.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            username: Database username
            password: Database password
            min_size: Minimum connection pool size
            max_size: Maximum connection pool size
            command_timeout: Command timeout in seconds
            server_settings: Additional server settings
        """
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.min_size = min_size
        self.max_size = max_size
        self.command_timeout = command_timeout
        self.server_settings = server_settings or {}
        
        self._pool: Optional[Pool] = None
    
    async def connect(self) -> None:
        """Establish connection pool to TimescaleDB."""
        try:
            self._pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password,
                min_size=self.min_size,
                max_size=self.max_size,
                command_timeout=self.command_timeout,
                server_settings=self.server_settings
            )
            
            # Test connection and create tables
            await self._initialize_schema()
            logger.info("Connected to TimescaleDB successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to TimescaleDB: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close TimescaleDB connection pool."""
        if self._pool:
            try:
                await self._pool.close()
                logger.info("Disconnected from TimescaleDB")
            except Exception as e:
                logger.error(f"Error disconnecting from TimescaleDB: {e}")
            finally:
                self._pool = None
    
    @asynccontextmanager
    async def get_connection(self):
        """Context manager for database connection."""
        if not self._pool:
            await self.connect()
        
        async with self._pool.acquire() as connection:
            try:
                yield connection
            except Exception as e:
                logger.error(f"TimescaleDB operation failed: {e}")
                raise
    
    async def health_check(self) -> bool:
        """Check TimescaleDB health status."""
        try:
            if not self._pool:
                return False
            
            async with self.get_connection() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
                
        except Exception as e:
            logger.error(f"TimescaleDB health check failed: {e}")
            return False
    
    async def _initialize_schema(self) -> None:
        """Initialize database schema and hypertables."""
        async with self.get_connection() as conn:
            # Enable TimescaleDB extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
            
            # Create market_data table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    open DOUBLE PRECISION NOT NULL,
                    high DOUBLE PRECISION NOT NULL,
                    low DOUBLE PRECISION NOT NULL,
                    close DOUBLE PRECISION NOT NULL,
                    volume DOUBLE PRECISION NOT NULL,
                    vwap DOUBLE PRECISION,
                    trades_count INTEGER,
                    PRIMARY KEY (time, symbol, exchange)
                );
            """)
            
            # Create hypertable for market_data
            try:
                await conn.execute("""
                    SELECT create_hypertable('market_data', 'time', 
                                           if_not_exists => TRUE);
                """)
            except PostgresError as e:
                if "already a hypertable" not in str(e):
                    raise
            
            # Create predictions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    predicted_price DOUBLE PRECISION,
                    predicted_direction TEXT,
                    confidence_score DOUBLE PRECISION,
                    uncertainty DOUBLE PRECISION,
                    feature_importance JSONB,
                    PRIMARY KEY (time, symbol, model_name)
                );
            """)
            
            # Create hypertable for predictions
            try:
                await conn.execute("""
                    SELECT create_hypertable('predictions', 'time', 
                                           if_not_exists => TRUE);
                """)
            except PostgresError as e:
                if "already a hypertable" not in str(e):
                    raise
            
            # Create performance_metrics table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    time TIMESTAMPTZ NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value DOUBLE PRECISION NOT NULL,
                    symbol TEXT,
                    strategy TEXT,
                    model_name TEXT,
                    timeframe TEXT,
                    PRIMARY KEY (time, metric_name, symbol, strategy, model_name)
                );
            """)
            
            # Create hypertable for performance_metrics
            try:
                await conn.execute("""
                    SELECT create_hypertable('performance_metrics', 'time', 
                                           if_not_exists => TRUE);
                """)
            except PostgresError as e:
                if "already a hypertable" not in str(e):
                    raise
            
            # Create indexes for better query performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time 
                ON market_data (symbol, time DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_symbol_model_time 
                ON predictions (symbol, model_name, time DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_performance_metrics_name_time 
                ON performance_metrics (metric_name, time DESC);
            """)
    
    async def insert_market_data(
        self, 
        market_data: MarketDataPoint
    ) -> None:
        """Insert single market data point."""
        async with self.get_connection() as conn:
            await conn.execute("""
                INSERT INTO market_data 
                (time, symbol, exchange, open, high, low, close, volume, vwap, trades_count)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (time, symbol, exchange) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    vwap = EXCLUDED.vwap,
                    trades_count = EXCLUDED.trades_count;
            """, 
                market_data.timestamp,
                market_data.symbol,
                market_data.exchange,
                market_data.open,
                market_data.high,
                market_data.low,
                market_data.close,
                market_data.volume,
                market_data.vwap,
                market_data.trades_count
            )
    
    async def insert_market_data_batch(
        self, 
        market_data_list: List[MarketDataPoint]
    ) -> None:
        """Insert multiple market data points in batch."""
        if not market_data_list:
            return
        
        async with self.get_connection() as conn:
            # Prepare data for batch insert
            data = [
                (
                    md.timestamp,
                    md.symbol,
                    md.exchange,
                    md.open,
                    md.high,
                    md.low,
                    md.close,
                    md.volume,
                    md.vwap,
                    md.trades_count
                )
                for md in market_data_list
            ]
            
            await conn.executemany("""
                INSERT INTO market_data 
                (time, symbol, exchange, open, high, low, close, volume, vwap, trades_count)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (time, symbol, exchange) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    vwap = EXCLUDED.vwap,
                    trades_count = EXCLUDED.trades_count;
            """, data)
            
            logger.debug(f"Inserted {len(market_data_list)} market data points")
    
    async def insert_prediction(
        self, 
        prediction: PredictionPoint
    ) -> None:
        """Insert single prediction point."""
        async with self.get_connection() as conn:
            await conn.execute("""
                INSERT INTO predictions 
                (time, symbol, model_name, model_version, predicted_price, 
                 predicted_direction, confidence_score, uncertainty, feature_importance)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (time, symbol, model_name) DO UPDATE SET
                    model_version = EXCLUDED.model_version,
                    predicted_price = EXCLUDED.predicted_price,
                    predicted_direction = EXCLUDED.predicted_direction,
                    confidence_score = EXCLUDED.confidence_score,
                    uncertainty = EXCLUDED.uncertainty,
                    feature_importance = EXCLUDED.feature_importance;
            """, 
                prediction.timestamp,
                prediction.symbol,
                prediction.model_name,
                prediction.model_version,
                prediction.predicted_price,
                prediction.predicted_direction,
                prediction.confidence_score,
                prediction.uncertainty,
                prediction.feature_importance
            )
    
    async def insert_performance_metric(
        self, 
        metric: PerformanceMetric
    ) -> None:
        """Insert single performance metric."""
        async with self.get_connection() as conn:
            await conn.execute("""
                INSERT INTO performance_metrics 
                (time, metric_name, metric_value, symbol, strategy, model_name, timeframe)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (time, metric_name, symbol, strategy, model_name) DO UPDATE SET
                    metric_value = EXCLUDED.metric_value,
                    timeframe = EXCLUDED.timeframe;
            """, 
                metric.timestamp,
                metric.metric_name,
                metric.metric_value,
                metric.symbol,
                metric.strategy,
                metric.model_name,
                metric.timeframe
            )
    
    async def query_market_data(
        self,
        symbol: str,
        exchange: str,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Query market data for symbol and time range."""
        async with self.get_connection() as conn:
            query = """
                SELECT time, symbol, exchange, open, high, low, close, volume, vwap, trades_count
                FROM market_data
                WHERE symbol = $1 AND exchange = $2 
                  AND time >= $3 AND time <= $4
                ORDER BY time DESC
            """
            
            params = [symbol, exchange, start_time, end_time]
            
            if limit:
                query += " LIMIT $5"
                params.append(limit)
            
            rows = await conn.fetch(query, *params)
            
            return [
                {
                    'time': row['time'],
                    'symbol': row['symbol'],
                    'exchange': row['exchange'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume'],
                    'vwap': row['vwap'],
                    'trades_count': row['trades_count']
                }
                for row in rows
            ]
    
    async def query_predictions(
        self,
        symbol: str,
        model_name: str,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Query predictions for symbol, model and time range."""
        async with self.get_connection() as conn:
            query = """
                SELECT time, symbol, model_name, model_version, predicted_price,
                       predicted_direction, confidence_score, uncertainty, feature_importance
                FROM predictions
                WHERE symbol = $1 AND model_name = $2 
                  AND time >= $3 AND time <= $4
                ORDER BY time DESC
            """
            
            params = [symbol, model_name, start_time, end_time]
            
            if limit:
                query += " LIMIT $5"
                params.append(limit)
            
            rows = await conn.fetch(query, *params)
            
            return [
                {
                    'time': row['time'],
                    'symbol': row['symbol'],
                    'model_name': row['model_name'],
                    'model_version': row['model_version'],
                    'predicted_price': row['predicted_price'],
                    'predicted_direction': row['predicted_direction'],
                    'confidence_score': row['confidence_score'],
                    'uncertainty': row['uncertainty'],
                    'feature_importance': row['feature_importance']
                }
                for row in rows
            ]
    
    async def get_latest_market_data(
        self, 
        symbol: str, 
        exchange: str
    ) -> Optional[Dict[str, Any]]:
        """Get the latest market data for a symbol."""
        results = await self.query_market_data(
            symbol, exchange, 
            datetime.now(timezone.utc).replace(hour=0, minute=0, second=0),
            datetime.now(timezone.utc),
            limit=1
        )
        return results[0] if results else None
    
    async def get_ohlcv_aggregates(
        self,
        symbol: str,
        exchange: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = '1h'  # '1m', '5m', '1h', '1d'
    ) -> List[Dict[str, Any]]:
        """Get OHLCV aggregates for specified interval."""
        async with self.get_connection() as conn:
            query = f"""
                SELECT 
                    time_bucket('{interval}', time) AS bucket,
                    symbol,
                    exchange,
                    first(open, time) AS open,
                    max(high) AS high,
                    min(low) AS low,
                    last(close, time) AS close,
                    sum(volume) AS volume,
                    avg(vwap) AS vwap,
                    sum(trades_count) AS trades_count
                FROM market_data
                WHERE symbol = $1 AND exchange = $2 
                  AND time >= $3 AND time <= $4
                GROUP BY bucket, symbol, exchange
                ORDER BY bucket DESC;
            """
            
            rows = await conn.fetch(query, symbol, exchange, start_time, end_time)
            
            return [
                {
                    'time': row['bucket'],
                    'symbol': row['symbol'],
                    'exchange': row['exchange'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume'],
                    'vwap': row['vwap'],
                    'trades_count': row['trades_count']
                }
                for row in rows
            ]
    
    async def delete_old_data(
        self, 
        table_name: str, 
        older_than: datetime
    ) -> int:
        """Delete data older than specified timestamp."""
        async with self.get_connection() as conn:
            result = await conn.execute(
                f"DELETE FROM {table_name} WHERE time < $1",
                older_than
            )
            
            # Extract number of deleted rows from result
            deleted_count = int(result.split()[-1]) if result else 0
            logger.info(f"Deleted {deleted_count} rows from {table_name} older than {older_than}")
            
            return deleted_count
    
    async def create_continuous_aggregate(
        self,
        view_name: str,
        source_table: str,
        time_column: str,
        interval: str,
        select_clause: str
    ) -> None:
        """Create continuous aggregate view for better query performance."""
        async with self.get_connection() as conn:
            query = f"""
                CREATE MATERIALIZED VIEW IF NOT EXISTS {view_name}
                WITH (timescaledb.continuous) AS
                SELECT 
                    time_bucket('{interval}', {time_column}) AS bucket,
                    {select_clause}
                FROM {source_table}
                GROUP BY bucket;
            """
            
            await conn.execute(query)
            logger.info(f"Created continuous aggregate view: {view_name}")
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics and health metrics."""
        async with self.get_connection() as conn:
            # Get table sizes
            table_stats = await conn.fetch("""
                SELECT 
                    schemaname,
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                    pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                FROM pg_tables 
                WHERE tablename IN ('market_data', 'predictions', 'performance_metrics')
                ORDER BY size_bytes DESC;
            """)
            
            # Get row counts
            market_data_count = await conn.fetchval("SELECT COUNT(*) FROM market_data")
            predictions_count = await conn.fetchval("SELECT COUNT(*) FROM predictions")
            metrics_count = await conn.fetchval("SELECT COUNT(*) FROM performance_metrics")
            
            # Get database size
            db_size = await conn.fetchval("""
                SELECT pg_size_pretty(pg_database_size(current_database()))
            """)
            
            return {
                'database_size': db_size,
                'table_stats': [dict(row) for row in table_stats],
                'row_counts': {
                    'market_data': market_data_count,
                    'predictions': predictions_count,
                    'performance_metrics': metrics_count
                }
            }