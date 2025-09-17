"""
Unified data aggregation system for normalizing data from multiple exchanges.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import statistics

from ..models.market_data import MarketData, ExchangeType
from ..exchanges.base import ExchangeConnector


class DataQualityIssue(str, Enum):
    """Types of data quality issues."""
    MISSING_DATA = "missing_data"
    DUPLICATE_DATA = "duplicate_data"
    PRICE_ANOMALY = "price_anomaly"
    VOLUME_ANOMALY = "volume_anomaly"
    TIMESTAMP_GAP = "timestamp_gap"
    STALE_DATA = "stale_data"


@dataclass
class DataQualityReport:
    """Data quality validation report."""
    symbol: str
    exchange: str
    timestamp: datetime
    issue_type: DataQualityIssue
    severity: str  # "low", "medium", "high"
    description: str
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class AggregatedData:
    """Aggregated market data from multiple exchanges."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    exchanges: List[str] = field(default_factory=list)
    source_count: int = 0
    confidence_score: float = 1.0
    quality_issues: List[DataQualityReport] = field(default_factory=list)


class TimestampSynchronizer:
    """Handles timestamp synchronization and alignment."""
    
    def __init__(self, tolerance_seconds: int = 5):
        self.tolerance_seconds = tolerance_seconds
        self.logger = logging.getLogger(__name__)
    
    def normalize_timestamp(self, timestamp: datetime) -> datetime:
        """Normalize timestamp to UTC and round to nearest second."""
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        elif timestamp.tzinfo != timezone.utc:
            timestamp = timestamp.astimezone(timezone.utc)
        
        # Round to nearest second for alignment
        if timestamp.microsecond >= 500000:
            timestamp = timestamp.replace(microsecond=0) + timedelta(seconds=1)
        else:
            timestamp = timestamp.replace(microsecond=0)
        
        return timestamp
    
    def align_timestamps(
        self, 
        data_points: List[MarketData]
    ) -> Dict[datetime, List[MarketData]]:
        """Group data points by aligned timestamps."""
        aligned_data = defaultdict(list)
        
        for data_point in data_points:
            normalized_ts = self.normalize_timestamp(data_point.timestamp)
            aligned_data[normalized_ts].append(data_point)
        
        return dict(aligned_data)
    
    def find_timestamp_gaps(
        self, 
        timestamps: List[datetime], 
        expected_interval: timedelta
    ) -> List[tuple]:
        """Find gaps in timestamp sequence."""
        if len(timestamps) < 2:
            return []
        
        sorted_timestamps = sorted(timestamps)
        gaps = []
        
        for i in range(1, len(sorted_timestamps)):
            gap = sorted_timestamps[i] - sorted_timestamps[i-1]
            if gap > expected_interval * 1.5:  # Allow 50% tolerance
                gaps.append((sorted_timestamps[i-1], sorted_timestamps[i], gap))
        
        return gaps


class DataQualityValidator:
    """Validates data quality and detects anomalies."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.price_history = defaultdict(lambda: deque(maxlen=100))
        self.volume_history = defaultdict(lambda: deque(maxlen=100))
    
    def validate_market_data(self, data: MarketData) -> List[DataQualityReport]:
        """Validate a single market data point."""
        issues = []
        
        # Basic validation
        issues.extend(self._validate_basic_constraints(data))
        
        # Price anomaly detection
        issues.extend(self._detect_price_anomalies(data))
        
        # Volume anomaly detection
        issues.extend(self._detect_volume_anomalies(data))
        
        # Timestamp validation
        issues.extend(self._validate_timestamp(data))
        
        # Update history for future validations
        self._update_history(data)
        
        return issues
    
    def _validate_basic_constraints(self, data: MarketData) -> List[DataQualityReport]:
        """Validate basic price relationships and constraints."""
        issues = []
        
        # Check for negative or zero prices
        if any(price <= 0 for price in [data.open, data.high, data.low, data.close]):
            issues.append(DataQualityReport(
                symbol=data.symbol,
                exchange=data.exchange.value,
                timestamp=data.timestamp,
                issue_type=DataQualityIssue.PRICE_ANOMALY,
                severity="high",
                description="Negative or zero price detected",
                raw_data={"open": data.open, "high": data.high, 
                         "low": data.low, "close": data.close}
            ))
        
        # Check price relationships
        if data.high < data.low:
            issues.append(DataQualityReport(
                symbol=data.symbol,
                exchange=data.exchange.value,
                timestamp=data.timestamp,
                issue_type=DataQualityIssue.PRICE_ANOMALY,
                severity="high",
                description=f"High price ({data.high}) < Low price ({data.low})",
                raw_data={"high": data.high, "low": data.low}
            ))
        
        if data.high < max(data.open, data.close):
            issues.append(DataQualityReport(
                symbol=data.symbol,
                exchange=data.exchange.value,
                timestamp=data.timestamp,
                issue_type=DataQualityIssue.PRICE_ANOMALY,
                severity="medium",
                description="High price less than open/close price",
                raw_data={"high": data.high, "open": data.open, "close": data.close}
            ))
        
        if data.low > min(data.open, data.close):
            issues.append(DataQualityReport(
                symbol=data.symbol,
                exchange=data.exchange.value,
                timestamp=data.timestamp,
                issue_type=DataQualityIssue.PRICE_ANOMALY,
                severity="medium",
                description="Low price greater than open/close price",
                raw_data={"low": data.low, "open": data.open, "close": data.close}
            ))
        
        # Check for negative volume
        if data.volume < 0:
            issues.append(DataQualityReport(
                symbol=data.symbol,
                exchange=data.exchange.value,
                timestamp=data.timestamp,
                issue_type=DataQualityIssue.VOLUME_ANOMALY,
                severity="medium",
                description=f"Negative volume: {data.volume}",
                raw_data={"volume": data.volume}
            ))
        
        return issues
    
    def _detect_price_anomalies(self, data: MarketData) -> List[DataQualityReport]:
        """Detect price anomalies using statistical methods."""
        issues = []
        key = f"{data.symbol}_{data.exchange.value}"
        history = self.price_history[key]
        
        if len(history) < 10:  # Need sufficient history
            return issues
        
        # Calculate statistics from historical data
        historical_prices = [d.close for d in history]
        mean_price = statistics.mean(historical_prices)
        std_price = statistics.stdev(historical_prices) if len(historical_prices) > 1 else 0
        
        if std_price == 0:
            return issues
        
        # Z-score anomaly detection
        z_score = abs(data.close - mean_price) / std_price
        
        if z_score > 5:  # More than 5 standard deviations
            issues.append(DataQualityReport(
                symbol=data.symbol,
                exchange=data.exchange.value,
                timestamp=data.timestamp,
                issue_type=DataQualityIssue.PRICE_ANOMALY,
                severity="high",
                description=f"Price anomaly detected: z-score = {z_score:.2f}",
                raw_data={"price": data.close, "mean": mean_price, 
                         "std": std_price, "z_score": z_score}
            ))
        elif z_score > 3:  # More than 3 standard deviations
            issues.append(DataQualityReport(
                symbol=data.symbol,
                exchange=data.exchange.value,
                timestamp=data.timestamp,
                issue_type=DataQualityIssue.PRICE_ANOMALY,
                severity="medium",
                description=f"Potential price anomaly: z-score = {z_score:.2f}",
                raw_data={"price": data.close, "mean": mean_price, 
                         "std": std_price, "z_score": z_score}
            ))
        
        return issues
    
    def _detect_volume_anomalies(self, data: MarketData) -> List[DataQualityReport]:
        """Detect volume anomalies."""
        issues = []
        key = f"{data.symbol}_{data.exchange.value}"
        history = self.volume_history[key]
        
        if len(history) < 10:
            return issues
        
        historical_volumes = [d.volume for d in history if d.volume > 0]
        if not historical_volumes:
            return issues
        
        mean_volume = statistics.mean(historical_volumes)
        
        # Check for extremely high volume (more than 10x average)
        if data.volume > mean_volume * 10:
            issues.append(DataQualityReport(
                symbol=data.symbol,
                exchange=data.exchange.value,
                timestamp=data.timestamp,
                issue_type=DataQualityIssue.VOLUME_ANOMALY,
                severity="medium",
                description=f"Unusually high volume: {data.volume} vs avg {mean_volume:.2f}",
                raw_data={"volume": data.volume, "avg_volume": mean_volume}
            ))
        
        return issues
    
    def _validate_timestamp(self, data: MarketData) -> List[DataQualityReport]:
        """Validate timestamp for staleness and future dates."""
        issues = []
        now = datetime.now(timezone.utc)
        
        # Check for future timestamps
        if data.timestamp > now + timedelta(minutes=5):
            issues.append(DataQualityReport(
                symbol=data.symbol,
                exchange=data.exchange.value,
                timestamp=data.timestamp,
                issue_type=DataQualityIssue.TIMESTAMP_GAP,
                severity="high",
                description="Future timestamp detected",
                raw_data={"timestamp": data.timestamp.isoformat(), "now": now.isoformat()}
            ))
        
        # Check for stale data (older than 1 hour for real-time data)
        if now - data.timestamp > timedelta(hours=1):
            issues.append(DataQualityReport(
                symbol=data.symbol,
                exchange=data.exchange.value,
                timestamp=data.timestamp,
                issue_type=DataQualityIssue.STALE_DATA,
                severity="low",
                description=f"Stale data: {now - data.timestamp} old",
                raw_data={"timestamp": data.timestamp.isoformat(), "age": str(now - data.timestamp)}
            ))
        
        return issues
    
    def _update_history(self, data: MarketData):
        """Update historical data for future validations."""
        key = f"{data.symbol}_{data.exchange.value}"
        self.price_history[key].append(data)
        self.volume_history[key].append(data)


class DataAggregator:
    """Main data aggregation class that normalizes data from all exchanges."""
    
    def __init__(self, exchanges: List[ExchangeConnector] = None):
        self.exchanges = exchanges or []
        self.logger = logging.getLogger(__name__)
        self.synchronizer = TimestampSynchronizer()
        self.validator = DataQualityValidator()
        
        # Configuration
        self.aggregation_window = timedelta(seconds=5)
        self.min_sources_for_aggregation = 1
        self.quality_threshold = 0.7
        
        # Data storage
        self.raw_data_buffer = deque(maxlen=10000)
        self.aggregated_data_cache = {}
        
    async def start_aggregation(self, symbols: List[str]) -> AsyncGenerator[AggregatedData, None]:
        """Start real-time data aggregation from all exchanges."""
        try:
            # Create tasks for each exchange
            tasks = []
            for exchange in self.exchanges:
                if exchange.is_connected:
                    task = asyncio.create_task(
                        self._collect_exchange_data(exchange, symbols)
                    )
                    tasks.append(task)
            
            if not tasks:
                self.logger.warning("No connected exchanges available for aggregation")
                return
            
            # Process aggregated data
            async for aggregated_data in self._process_aggregated_data():
                yield aggregated_data
                
        except Exception as e:
            self.logger.error(f"Error in data aggregation: {str(e)}")
    
    async def _collect_exchange_data(
        self, 
        exchange: ExchangeConnector, 
        symbols: List[str]
    ):
        """Collect data from a single exchange."""
        try:
            async for market_data in exchange.get_real_time_data(symbols):
                # Validate data quality
                quality_issues = self.validator.validate_market_data(market_data)
                
                # Add to buffer with quality information
                self.raw_data_buffer.append({
                    'data': market_data,
                    'quality_issues': quality_issues,
                    'received_at': datetime.now(timezone.utc)
                })
                
        except Exception as e:
            self.logger.error(f"Error collecting data from {exchange.__class__.__name__}: {str(e)}")
    
    async def _process_aggregated_data(self) -> AsyncGenerator[AggregatedData, None]:
        """Process and aggregate buffered data."""
        last_processed = datetime.now(timezone.utc)
        
        while True:
            try:
                await asyncio.sleep(1)  # Process every second
                
                current_time = datetime.now(timezone.utc)
                if current_time - last_processed < self.aggregation_window:
                    continue
                
                # Get data from buffer for processing
                buffer_data = list(self.raw_data_buffer)
                if not buffer_data:
                    continue
                
                # Group by symbol and timestamp
                symbol_groups = defaultdict(list)
                for item in buffer_data:
                    market_data = item['data']
                    key = market_data.symbol
                    symbol_groups[key].append(item)
                
                # Process each symbol group
                for symbol, data_items in symbol_groups.items():
                    aggregated = await self._aggregate_symbol_data(symbol, data_items)
                    if aggregated:
                        yield aggregated
                
                last_processed = current_time
                
            except Exception as e:
                self.logger.error(f"Error processing aggregated data: {str(e)}")
    
    async def _aggregate_symbol_data(
        self, 
        symbol: str, 
        data_items: List[Dict[str, Any]]
    ) -> Optional[AggregatedData]:
        """Aggregate data for a single symbol."""
        if not data_items:
            return None
        
        # Extract market data and align timestamps
        market_data_list = [item['data'] for item in data_items]
        aligned_data = self.synchronizer.align_timestamps(market_data_list)
        
        # Find the most recent timestamp with sufficient data
        recent_timestamps = sorted(aligned_data.keys(), reverse=True)[:5]
        
        for timestamp in recent_timestamps:
            data_points = aligned_data[timestamp]
            
            if len(data_points) >= self.min_sources_for_aggregation:
                return await self._create_aggregated_data(symbol, timestamp, data_points, data_items)
        
        return None
    
    async def _create_aggregated_data(
        self, 
        symbol: str, 
        timestamp: datetime, 
        data_points: List[MarketData],
        all_items: List[Dict[str, Any]]
    ) -> AggregatedData:
        """Create aggregated data from multiple data points."""
        
        # Calculate weighted averages (simple average for now)
        opens = [dp.open for dp in data_points]
        highs = [dp.high for dp in data_points]
        lows = [dp.low for dp in data_points]
        closes = [dp.close for dp in data_points]
        volumes = [dp.volume for dp in data_points]
        
        # Use median for price aggregation (more robust to outliers)
        aggregated_open = statistics.median(opens)
        aggregated_high = max(highs)  # Use max for high
        aggregated_low = min(lows)    # Use min for low
        aggregated_close = statistics.median(closes)
        aggregated_volume = sum(volumes)  # Sum volumes
        
        # Collect exchanges and quality issues
        exchanges = [dp.exchange.value for dp in data_points]
        all_quality_issues = []
        
        for item in all_items:
            if item['data'].timestamp == timestamp:
                all_quality_issues.extend(item['quality_issues'])
        
        # Calculate confidence score based on data quality
        confidence_score = self._calculate_confidence_score(data_points, all_quality_issues)
        
        return AggregatedData(
            symbol=symbol,
            timestamp=timestamp,
            open=aggregated_open,
            high=aggregated_high,
            low=aggregated_low,
            close=aggregated_close,
            volume=aggregated_volume,
            exchanges=exchanges,
            source_count=len(data_points),
            confidence_score=confidence_score,
            quality_issues=all_quality_issues
        )
    
    def _calculate_confidence_score(
        self, 
        data_points: List[MarketData], 
        quality_issues: List[DataQualityReport]
    ) -> float:
        """Calculate confidence score for aggregated data."""
        base_score = 1.0
        
        # Reduce score based on number of sources
        if len(data_points) == 1:
            base_score *= 0.7
        elif len(data_points) == 2:
            base_score *= 0.85
        
        # Reduce score based on quality issues
        high_severity_issues = sum(1 for issue in quality_issues if issue.severity == "high")
        medium_severity_issues = sum(1 for issue in quality_issues if issue.severity == "medium")
        
        base_score -= (high_severity_issues * 0.3)
        base_score -= (medium_severity_issues * 0.1)
        
        # Check price consistency across exchanges
        if len(data_points) > 1:
            closes = [dp.close for dp in data_points]
            price_std = statistics.stdev(closes) if len(closes) > 1 else 0
            mean_price = statistics.mean(closes)
            
            if mean_price > 0:
                cv = price_std / mean_price  # Coefficient of variation
                if cv > 0.05:  # More than 5% variation
                    base_score *= 0.8
        
        return max(0.0, min(1.0, base_score))
    
    async def get_historical_aggregated_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start: datetime, 
        end: datetime
    ) -> pd.DataFrame:
        """Get historical aggregated data from all exchanges."""
        try:
            all_dataframes = []
            
            # Collect data from all connected exchanges
            for exchange in self.exchanges:
                if exchange.is_connected:
                    try:
                        df = await exchange.get_historical_data(symbol, timeframe, start, end)
                        if not df.empty:
                            df['exchange'] = exchange.__class__.__name__
                            all_dataframes.append(df)
                    except Exception as e:
                        self.logger.error(f"Error getting historical data from {exchange.__class__.__name__}: {str(e)}")
            
            if not all_dataframes:
                return pd.DataFrame()
            
            # Combine and aggregate data
            combined_df = pd.concat(all_dataframes, ignore_index=False)
            
            # Group by timestamp and aggregate
            aggregated_data = []
            for timestamp, group in combined_df.groupby(combined_df.index):
                if len(group) >= self.min_sources_for_aggregation:
                    agg_row = {
                        'timestamp': timestamp,
                        'open': group['open'].median(),
                        'high': group['high'].max(),
                        'low': group['low'].min(),
                        'close': group['close'].median(),
                        'volume': group['volume'].sum(),
                        'source_count': len(group),
                        'exchanges': ','.join(group['exchange'].unique())
                    }
                    aggregated_data.append(agg_row)
            
            result_df = pd.DataFrame(aggregated_data)
            if not result_df.empty:
                result_df.set_index('timestamp', inplace=True)
                result_df.sort_index(inplace=True)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error aggregating historical data: {str(e)}")
            return pd.DataFrame()
    
    async def get_historical_data(
        self,
        symbol: str,
        exchange: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get historical data for a specific symbol and exchange.
        
        This method provides a simplified interface for getting historical data
        from a specific exchange, adapting the more general aggregation method.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with historical data
        """
        try:
            # Find the specific exchange
            target_exchange = None
            for exch in self.exchanges:
                if exch.__class__.__name__.lower() == exchange.lower():
                    target_exchange = exch
                    break
            
            # If we can't find the specific exchange, try to get data from any connected exchange
            if target_exchange is None or not target_exchange.is_connected:
                for exch in self.exchanges:
                    if exch.is_connected:
                        target_exchange = exch
                        break
            
            # If no connected exchanges, return empty DataFrame
            if target_exchange is None:
                return pd.DataFrame()
            
            # Get historical data from the exchange
            # Using a default timeframe of '1d' (daily) for simplicity
            df = await target_exchange.get_historical_data(
                symbol=symbol,
                timeframe='1d',
                start=start_date,
                end=end_date
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol} from {exchange}: {str(e)}")
            return pd.DataFrame()
    
    def get_data_quality_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get data quality summary for the last N hours."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        # Analyze recent data from buffer
        recent_items = [
            item for item in self.raw_data_buffer 
            if item['received_at'] >= cutoff_time
        ]
        
        if not recent_items:
            return {"message": "No recent data available"}
        
        # Collect statistics
        total_data_points = len(recent_items)
        total_issues = sum(len(item['quality_issues']) for item in recent_items)
        
        issue_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        exchange_stats = defaultdict(lambda: {'count': 0, 'issues': 0})
        
        for item in recent_items:
            exchange = item['data'].exchange.value
            exchange_stats[exchange]['count'] += 1
            exchange_stats[exchange]['issues'] += len(item['quality_issues'])
            
            for issue in item['quality_issues']:
                issue_counts[issue.issue_type.value] += 1
                severity_counts[issue.severity] += 1
        
        return {
            "period_hours": hours,
            "total_data_points": total_data_points,
            "total_quality_issues": total_issues,
            "issue_rate": total_issues / total_data_points if total_data_points > 0 else 0,
            "issue_types": dict(issue_counts),
            "severity_distribution": dict(severity_counts),
            "exchange_statistics": dict(exchange_stats)
        }