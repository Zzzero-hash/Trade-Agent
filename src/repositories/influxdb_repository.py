"""
InfluxDB repository for time series data storage and retrieval.
"""
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, AsyncGenerator
import logging
from contextlib import asynccontextmanager

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
from influxdb_client.client.exceptions import InfluxDBError

from ..models.time_series import (
    TimeSeriesPoint, 
    TimeSeriesQuery, 
    MarketDataPoint,
    PredictionPoint,
    PerformanceMetric
)


logger = logging.getLogger(__name__)


class InfluxDBRepository:
    """
    Repository for InfluxDB time series operations.
    
    Provides async interface for storing and querying time series data.
    """
    
    def __init__(
        self,
        url: str,
        token: str,
        org: str,
        bucket: str,
        timeout: int = 30000,
        enable_gzip: bool = True
    ):
        """
        Initialize InfluxDB repository.
        
        Args:
            url: InfluxDB server URL
            token: Authentication token
            org: Organization name
            bucket: Default bucket name
            timeout: Request timeout in milliseconds
            enable_gzip: Enable gzip compression
        """
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.timeout = timeout
        self.enable_gzip = enable_gzip
        
        self._client: Optional[InfluxDBClient] = None
        self._write_api = None
        self._query_api = None
    
    async def connect(self) -> None:
        """Establish connection to InfluxDB."""
        try:
            self._client = InfluxDBClient(
                url=self.url,
                token=self.token,
                org=self.org,
                timeout=self.timeout,
                enable_gzip=self.enable_gzip
            )
            
            # Initialize APIs
            self._write_api = self._client.write_api(write_options=ASYNCHRONOUS)
            self._query_api = self._client.query_api()
            
            # Test connection
            await self.health_check()
            logger.info("Connected to InfluxDB successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to InfluxDB: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close InfluxDB connection."""
        if self._client:
            try:
                self._client.close()
                logger.info("Disconnected from InfluxDB")
            except Exception as e:
                logger.error(f"Error disconnecting from InfluxDB: {e}")
            finally:
                self._client = None
                self._write_api = None
                self._query_api = None
    
    @asynccontextmanager
    async def get_client(self):
        """Context manager for InfluxDB client."""
        if not self._client:
            await self.connect()
        
        try:
            yield self._client
        except Exception as e:
            logger.error(f"InfluxDB operation failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check InfluxDB health status."""
        try:
            if not self._client:
                return False
            
            health = self._client.health()
            return health.status == "pass"
            
        except Exception as e:
            logger.error(f"InfluxDB health check failed: {e}")
            return False
    
    def _convert_to_influx_point(self, ts_point: TimeSeriesPoint) -> Point:
        """Convert TimeSeriesPoint to InfluxDB Point."""
        point = Point(ts_point.measurement)
        
        # Add timestamp
        point.time(ts_point.timestamp)
        
        # Add tags
        for key, value in ts_point.tags.items():
            point.tag(key, str(value))
        
        # Add fields
        for key, value in ts_point.fields.items():
            if isinstance(value, (int, float)):
                point.field(key, value)
            elif isinstance(value, bool):
                point.field(key, value)
            else:
                point.field(key, str(value))
        
        return point
    
    async def write_point(
        self, 
        point: TimeSeriesPoint, 
        bucket: Optional[str] = None
    ) -> None:
        """
        Write a single time series point.
        
        Args:
            point: Time series point to write
            bucket: Target bucket (uses default if None)
        """
        target_bucket = bucket or self.bucket
        
        try:
            async with self.get_client():
                influx_point = self._convert_to_influx_point(point)
                self._write_api.write(
                    bucket=target_bucket,
                    org=self.org,
                    record=influx_point
                )
                
        except InfluxDBError as e:
            logger.error(f"Failed to write point to InfluxDB: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error writing to InfluxDB: {e}")
            raise
    
    async def write_points(
        self, 
        points: List[TimeSeriesPoint], 
        bucket: Optional[str] = None
    ) -> None:
        """
        Write multiple time series points in batch.
        
        Args:
            points: List of time series points to write
            bucket: Target bucket (uses default if None)
        """
        if not points:
            return
        
        target_bucket = bucket or self.bucket
        
        try:
            async with self.get_client():
                influx_points = [
                    self._convert_to_influx_point(point) 
                    for point in points
                ]
                
                self._write_api.write(
                    bucket=target_bucket,
                    org=self.org,
                    record=influx_points
                )
                
                logger.debug(f"Wrote {len(points)} points to InfluxDB")
                
        except InfluxDBError as e:
            logger.error(f"Failed to write {len(points)} points to InfluxDB: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error writing batch to InfluxDB: {e}")
            raise
    
    async def query(
        self, 
        query: TimeSeriesQuery, 
        bucket: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute time series query.
        
        Args:
            query: Query parameters
            bucket: Source bucket (uses default if None)
            
        Returns:
            List of query results as dictionaries
        """
        target_bucket = bucket or self.bucket
        
        try:
            async with self.get_client():
                flux_query = self._build_flux_query(query, target_bucket)
                
                # Execute query
                result = self._query_api.query(flux_query, org=self.org)
                
                # Convert to list of dictionaries
                records = []
                for table in result:
                    for record in table.records:
                        record_dict = {
                            'time': record.get_time(),
                            'measurement': record.get_measurement(),
                            'field': record.get_field(),
                            'value': record.get_value()
                        }
                        
                        # Add tags
                        for key, value in record.values.items():
                            if key.startswith('_') or key in ['result', 'table']:
                                continue
                            record_dict[key] = value
                        
                        records.append(record_dict)
                
                return records
                
        except InfluxDBError as e:
            logger.error(f"InfluxDB query failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during query: {e}")
            raise
    
    def _build_flux_query(
        self, 
        query: TimeSeriesQuery, 
        bucket: str
    ) -> str:
        """Build Flux query string from TimeSeriesQuery."""
        flux_parts = [
            f'from(bucket: "{bucket}")'
        ]
        
        # Time range
        start_time = query.start_time.isoformat()
        end_time = query.end_time.isoformat()
        flux_parts.append(f'|> range(start: {start_time}, stop: {end_time})')
        
        # Measurement filter
        flux_parts.append(f'|> filter(fn: (r) => r._measurement == "{query.measurement}")')
        
        # Tag filters
        for tag_key, tag_value in query.tags.items():
            flux_parts.append(f'|> filter(fn: (r) => r.{tag_key} == "{tag_value}")')
        
        # Field filters
        if query.fields:
            field_conditions = ' or '.join([
                f'r._field == "{field}"' for field in query.fields
            ])
            flux_parts.append(f'|> filter(fn: (r) => {field_conditions})')
        
        # Aggregation
        if query.aggregation and query.window:
            flux_parts.append(f'|> aggregateWindow(every: {query.window}, fn: {query.aggregation})')
        elif query.aggregation:
            flux_parts.append(f'|> {query.aggregation}()')
        
        # Group by
        if query.group_by:
            group_columns = ', '.join([f'"{col}"' for col in query.group_by])
            flux_parts.append(f'|> group(columns: [{group_columns}])')
        
        # Limit
        if query.limit:
            flux_parts.append(f'|> limit(n: {query.limit})')
        
        return '\n  '.join(flux_parts)
    
    async def write_market_data(
        self, 
        market_data: MarketDataPoint, 
        bucket: Optional[str] = None
    ) -> None:
        """Write market data point."""
        ts_point = market_data.to_time_series_point()
        await self.write_point(ts_point, bucket)
    
    async def write_market_data_batch(
        self, 
        market_data_list: List[MarketDataPoint], 
        bucket: Optional[str] = None
    ) -> None:
        """Write multiple market data points."""
        ts_points = [data.to_time_series_point() for data in market_data_list]
        await self.write_points(ts_points, bucket)
    
    async def write_prediction(
        self, 
        prediction: PredictionPoint, 
        bucket: Optional[str] = None
    ) -> None:
        """Write prediction point."""
        ts_point = prediction.to_time_series_point()
        await self.write_point(ts_point, bucket)
    
    async def write_performance_metric(
        self, 
        metric: PerformanceMetric, 
        bucket: Optional[str] = None
    ) -> None:
        """Write performance metric point."""
        ts_point = metric.to_time_series_point()
        await self.write_point(ts_point, bucket)
    
    async def get_latest_market_data(
        self, 
        symbol: str, 
        exchange: str,
        bucket: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get the latest market data for a symbol."""
        query = TimeSeriesQuery(
            measurement='market_data',
            start_time=datetime.now(timezone.utc).replace(hour=0, minute=0, second=0),
            end_time=datetime.now(timezone.utc),
            tags={'symbol': symbol, 'exchange': exchange},
            limit=1
        )
        
        results = await self.query(query, bucket)
        return results[0] if results else None
    
    async def delete_old_data(
        self, 
        measurement: str, 
        older_than: datetime,
        bucket: Optional[str] = None
    ) -> None:
        """Delete data older than specified timestamp."""
        target_bucket = bucket or self.bucket
        
        try:
            async with self.get_client():
                delete_api = self._client.delete_api()
                
                await delete_api.delete(
                    start=datetime(1970, 1, 1, tzinfo=timezone.utc),
                    stop=older_than,
                    predicate=f'_measurement="{measurement}"',
                    bucket=target_bucket,
                    org=self.org
                )
                
                logger.info(f"Deleted {measurement} data older than {older_than}")
                
        except Exception as e:
            logger.error(f"Failed to delete old data: {e}")
            raise