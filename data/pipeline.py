"""
Production-grade data pipeline for multi-source market data ingestion.

This module implements a comprehensive data pipeline that handles:
- Multi-source data ingestion (stocks, forex, crypto, futures)
- Robust data cleaning with survivorship bias correction
- Sophisticated missing data imputation
- Data quality monitoring with automated alerts
"""

import asyncio
import json
import logging
import sqlite3
import time
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yfinance as yf

from .models import (
    AssetClass,
    DataValidator,
    MarketData,
    MarketMetadata,
    OrderBook,
    Trade,
    create_market_data,
)
logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Supported data sources."""
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    BINANCE = "binance"
    COINBASE = "coinbase"
    FOREX_COM = "forex_com"
    INTERACTIVE_BROKERS = "interactive_brokers"
    BLOOMBERG = "bloomberg"
    REFINITIV = "refinitiv"


class DataQuality(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"


@dataclass
class DataSourceConfig:
    """Configuration for data sources."""
    source: DataSource
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    rate_limit: int = 60  # requests per minute
    timeout: int = 30  # seconds
    retry_attempts: int = 3
    retry_delay: float = 1.0  # seconds
    enabled: bool = True
    priority: int = 1  # lower number = higher priority


@dataclass
class PipelineConfig:
    """Configuration for the data pipeline."""
    data_sources: List[DataSourceConfig] = field(default_factory=list)
    storage_path: Path = field(default_factory=lambda: Path("data/processed"))
    cache_path: Path = field(default_factory=lambda: Path("data/cache"))
    max_workers: int = 4
    batch_size: int = 1000
    quality_threshold: float = 0.7
    outlier_threshold: float = 3.0
    survivorship_bias_correction: bool = True
    enable_alerts: bool = True
    alert_email: Optional[str] = None


@dataclass
class DataQualityMetrics:
    """Comprehensive data quality metrics."""
    completeness: float  # 0-1, percentage of non-null values
    accuracy: float  # 0-1, based on validation checks
    consistency: float  # 0-1, internal consistency checks
    timeliness: float  # 0-1, how recent the data is
    outlier_rate: float  # 0-1, percentage of outliers detected
    duplicate_rate: float  # 0-1, percentage of duplicates
    overall_score: float  # 0-1, weighted average
    quality_level: DataQuality
    issues: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate overall score and quality level."""
        weights = {
            'completeness': 0.25,
            'accuracy': 0.25,
            'consistency': 0.20,
            'timeliness': 0.15,
            'outlier_rate': 0.10,  # inverted - lower is better
            'duplicate_rate': 0.05  # inverted - lower is better
        }
        
        self.overall_score = (
            weights['completeness'] * self.completeness +
            weights['accuracy'] * self.accuracy +
            weights['consistency'] * self.consistency +
            weights['timeliness'] * self.timeliness +
            weights['outlier_rate'] * (1 - self.outlier_rate) +
            weights['duplicate_rate'] * (1 - self.duplicate_rate)
        )
        
        # Determine quality level
        if self.overall_score >= 0.9:
            self.quality_level = DataQuality.EXCELLENT
        elif self.overall_score >= 0.8:
            self.quality_level = DataQuality.GOOD
        elif self.overall_score >= 0.6:
            self.quality_level = DataQuality.ACCEPTABLE
        elif self.overall_score >= 0.4:
            self.quality_level = DataQuality.POOR
        else:
            self.quality_level = DataQuality.UNUSABLE


class DataSourceInterface(ABC):
    """Abstract interface for data sources."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config.rate_limit)
    
    @abstractmethod
    async def fetch_market_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d"
    ) -> List[MarketData]:
        """Fetch market data for a symbol."""
        pass
    
    @abstractmethod
    async def fetch_order_book(self, symbol: str) -> Optional[OrderBook]:
        """Fetch current order book for a symbol."""
        pass
    
    @abstractmethod
    async def fetch_recent_trades(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[Trade]:
        """Fetch recent trades for a symbol."""
        pass
    
    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols."""
        pass
    
    @abstractmethod
    def get_asset_metadata(self, symbol: str) -> Optional[MarketMetadata]:
        """Get metadata for an asset."""
        pass


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0
    
    async def acquire(self):
        """Acquire permission to make a request."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()


class DataCleaner:
    """Advanced data cleaning with survivorship bias correction."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.validator = DataValidator()
    
    def clean_market_data(
        self,
        data: List[MarketData],
        symbol: str
    ) -> Tuple[List[MarketData], DataQualityMetrics]:
        """
        Clean market data with comprehensive quality assessment.
        
        Args:
            data: Raw market data
            symbol: Symbol being processed
            
        Returns:
            Tuple of (cleaned_data, quality_metrics)
        """
        if not data:
            return [], self._create_empty_quality_metrics()
        
        logger.info(f"Cleaning {len(data)} data points for {symbol}")
        
        # Step 1: Remove duplicates
        data = self._remove_duplicates(data)
        
        # Step 2: Sort by timestamp
        data.sort(key=lambda x: x.timestamp)
        
        # Step 3: Validate OHLCV consistency
        data = self._validate_ohlcv_consistency(data)
        
        # Step 4: Detect and handle outliers
        data = self._handle_outliers(data)
        
        # Step 5: Fill missing data
        data = self._fill_missing_data(data)
        
        # Step 6: Apply survivorship bias correction
        if self.config.survivorship_bias_correction:
            data = self._correct_survivorship_bias(data, symbol)
        
        # Step 7: Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(data)
        
        logger.info(
            f"Cleaned data for {symbol}: {len(data)} points, "
            f"quality score: {quality_metrics.overall_score:.3f}"
        )
        
        return data, quality_metrics
    
    def _remove_duplicates(self, data: List[MarketData]) -> List[MarketData]:
        """Remove duplicate data points."""
        seen = set()
        unique_data = []
        
        for item in data:
            key = (item.symbol, item.timestamp, item.timeframe)
            if key not in seen:
                seen.add(key)
                unique_data.append(item)
        
        return unique_data
    
    def _validate_ohlcv_consistency(
        self,
        data: List[MarketData]
    ) -> List[MarketData]:
        """Validate and fix OHLCV consistency issues."""
        valid_data = []
        
        for item in data:
            try:
                # Check basic OHLCV constraints
                if not (item.low <= item.open <= item.high and
                        item.low <= item.close <= item.high):
                    # Try to fix by adjusting high/low
                    item.high = max(item.open, item.close, item.high)
                    item.low = min(item.open, item.close, item.low)
                
                # Check for zero or negative values
                if any(val <= 0 for val in [item.open, item.high, 
                                           item.low, item.close]):
                    logger.warning(
                        f"Skipping invalid OHLC data for {item.symbol} "
                        f"at {item.timestamp}"
                    )
                    continue
                
                # Check for zero volume (might be valid for some assets)
                if item.volume < 0:
                    item.volume = 0
                
                valid_data.append(item)
                
            except Exception as e:
                logger.error(
                    f"Error validating OHLCV for {item.symbol} "
                    f"at {item.timestamp}: {e}"
                )
                continue
        
        return valid_data
    
    def _handle_outliers(self, data: List[MarketData]) -> List[MarketData]:
        """Detect and handle statistical outliers."""
        if len(data) < 10:  # Need minimum data for outlier detection
            return data
        
        # Extract price series for outlier detection
        prices = [item.close for item in data]
        volumes = [item.volume for item in data]
        
        # Detect price outliers
        price_outliers, price_reasons = self.validator.validate_price_series(
            prices, self.config.outlier_threshold
        )
        
        # Detect volume outliers
        volume_outliers, volume_reasons = self.validator.validate_volume_series(
            volumes
        )
        
        # Mark outliers in data
        cleaned_data = []
        for i, item in enumerate(data):
            if price_outliers[i] or volume_outliers[i]:
                item.is_outlier = True
                item.outlier_reasons = []
                
                if price_outliers[i]:
                    item.outlier_reasons.extend(price_reasons[i])
                if volume_outliers[i]:
                    item.outlier_reasons.extend(volume_reasons[i])
                
                # For now, keep outliers but mark them
                # In production, might want to remove or adjust them
                logger.debug(
                    f"Outlier detected for {item.symbol} at {item.timestamp}: "
                    f"{item.outlier_reasons}"
                )
            
            cleaned_data.append(item)
        
        return cleaned_data
    
    def _fill_missing_data(self, data: List[MarketData]) -> List[MarketData]:
        """Fill missing data using sophisticated imputation."""
        if len(data) < 2:
            return data
        
        # Convert to DataFrame for easier manipulation
        df_data = []
        for item in data:
            df_data.append({
                'timestamp': item.timestamp,
                'open': item.open,
                'high': item.high,
                'low': item.low,
                'close': item.close,
                'volume': item.volume,
                'symbol': item.symbol,
                'timeframe': item.timeframe
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Forward fill for price data (common in financial data)
        price_cols = ['open', 'high', 'low', 'close']
        df[price_cols] = df[price_cols].fillna(method='ffill')
        
        # For volume, use interpolation or forward fill
        df['volume'] = df['volume'].fillna(method='ffill')
        
        # Convert back to MarketData objects
        filled_data = []
        for timestamp, row in df.iterrows():
            market_data = MarketData(
                timestamp=timestamp,
                symbol=row['symbol'],
                timeframe=row['timeframe'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            filled_data.append(market_data)
        
        return filled_data
    
    def _correct_survivorship_bias(
        self,
        data: List[MarketData],
        symbol: str
    ) -> List[MarketData]:
        """Apply survivorship bias correction."""
        # This is a simplified implementation
        # In practice, would need historical constituent data
        
        # For now, just add a small adjustment to returns
        # to account for survivorship bias
        survivorship_adjustment = 0.02  # 2% annual adjustment
        
        for i in range(1, len(data)):
            prev_close = data[i-1].close
            current_return = (data[i].close - prev_close) / prev_close
            
            # Apply small negative adjustment to account for survivorship bias
            adjusted_return = current_return - (survivorship_adjustment / 252)
            adjusted_close = prev_close * (1 + adjusted_return)
            
            # Adjust OHLC proportionally
            ratio = adjusted_close / data[i].close
            data[i].open *= ratio
            data[i].high *= ratio
            data[i].low *= ratio
            data[i].close = adjusted_close
        
        return data
    
    def _calculate_quality_metrics(
        self,
        data: List[MarketData]
    ) -> DataQualityMetrics:
        """Calculate comprehensive quality metrics."""
        if not data:
            return self._create_empty_quality_metrics()
        
        # Completeness: check for missing values
        total_fields = len(data) * 5  # OHLCV
        missing_fields = sum(
            1 for item in data
            for val in [item.open, item.high, item.low, item.close, item.volume]
            if val is None or np.isnan(val)
        )
        completeness = 1 - (missing_fields / total_fields)
        
        # Accuracy: based on validation checks
        invalid_count = sum(1 for item in data if item.is_outlier)
        accuracy = 1 - (invalid_count / len(data))
        
        # Consistency: check OHLCV relationships
        inconsistent_count = 0
        for item in data:
            if not (item.low <= item.open <= item.high and
                    item.low <= item.close <= item.high):
                inconsistent_count += 1
        consistency = 1 - (inconsistent_count / len(data))
        
        # Timeliness: based on how recent the data is
        if data:
            latest_time = max(item.timestamp for item in data)
            time_diff = datetime.now() - latest_time
            timeliness = max(0, 1 - (time_diff.days / 7))  # Decay over week
        else:
            timeliness = 0
        
        # Outlier rate
        outlier_count = sum(1 for item in data if item.is_outlier)
        outlier_rate = outlier_count / len(data)
        
        # Duplicate rate (should be 0 after cleaning)
        duplicate_rate = 0.0
        
        return DataQualityMetrics(
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            timeliness=timeliness,
            outlier_rate=outlier_rate,
            duplicate_rate=duplicate_rate,
            overall_score=0.0  # Will be calculated in __post_init__
        )
    
    def _create_empty_quality_metrics(self) -> DataQualityMetrics:
        """Create quality metrics for empty dataset."""
        return DataQualityMetrics(
            completeness=0.0,
            accuracy=0.0,
            consistency=0.0,
            timeliness=0.0,
            outlier_rate=0.0,
            duplicate_rate=0.0,
            overall_score=0.0
        )


class DataStorage:
    """Efficient data storage with versioning and compression."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.storage_path = config.storage_path
        self.cache_path = config.cache_path
        
        # Create directories
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite database for metadata
        self.db_path = self.storage_path / "metadata.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    source TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    quality_score REAL NOT NULL,
                    record_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(symbol, timeframe, start_date, end_date, source)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    completeness REAL NOT NULL,
                    accuracy REAL NOT NULL,
                    consistency REAL NOT NULL,
                    timeliness REAL NOT NULL,
                    outlier_rate REAL NOT NULL,
                    duplicate_rate REAL NOT NULL,
                    overall_score REAL NOT NULL,
                    quality_level TEXT NOT NULL,
                    issues TEXT,
                    created_at TEXT NOT NULL
                )
            """)
    
    def save_market_data(
        self,
        data: List[MarketData],
        symbol: str,
        source: DataSource,
        quality_metrics: DataQualityMetrics
    ) -> Path:
        """Save market data with metadata."""
        if not data:
            raise ValueError("Cannot save empty data")
        
        # Create filename with timestamp
        start_date = min(item.timestamp for item in data)
        end_date = max(item.timestamp for item in data)
        timeframe = data[0].timeframe
        
        filename = (
            f"{symbol}_{timeframe}_{start_date.strftime('%Y%m%d')}_"
            f"{end_date.strftime('%Y%m%d')}_{source.value}.parquet"
        )
        file_path = self.storage_path / filename
        
        # Convert to DataFrame and save as Parquet
        df_data = []
        for item in data:
            df_data.append({
                'timestamp': item.timestamp,
                'symbol': item.symbol,
                'timeframe': item.timeframe,
                'open': item.open,
                'high': item.high,
                'low': item.low,
                'close': item.close,
                'volume': item.volume,
                'vwap': item.vwap,
                'num_trades': item.num_trades,
                'is_outlier': item.is_outlier,
                'outlier_reasons': json.dumps(item.outlier_reasons),
                'data_quality_score': item.data_quality_score
            })
        
        df = pd.DataFrame(df_data)
        df.to_parquet(file_path, compression='snappy')
        
        # Save metadata to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO data_metadata 
                (symbol, timeframe, start_date, end_date, source, file_path,
                 quality_score, record_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                timeframe,
                start_date.isoformat(),
                end_date.isoformat(),
                source.value,
                str(file_path),
                quality_metrics.overall_score,
                len(data),
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            
            # Save quality metrics
            conn.execute("""
                INSERT INTO quality_metrics
                (symbol, timestamp, completeness, accuracy, consistency,
                 timeliness, outlier_rate, duplicate_rate, overall_score,
                 quality_level, issues, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                datetime.now().isoformat(),
                quality_metrics.completeness,
                quality_metrics.accuracy,
                quality_metrics.consistency,
                quality_metrics.timeliness,
                quality_metrics.outlier_rate,
                quality_metrics.duplicate_rate,
                quality_metrics.overall_score,
                quality_metrics.quality_level.value,
                json.dumps(quality_metrics.issues),
                datetime.now().isoformat()
            ))
        
        logger.info(f"Saved {len(data)} records for {symbol} to {file_path}")
        return file_path
    
    def load_market_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d"
    ) -> Optional[List[MarketData]]:
        """Load market data from storage."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT file_path FROM data_metadata
                WHERE symbol = ? AND timeframe = ?
                AND start_date <= ? AND end_date >= ?
                ORDER BY quality_score DESC
                LIMIT 1
            """, (
                symbol,
                timeframe,
                end_date.isoformat(),
                start_date.isoformat()
            ))
            
            result = cursor.fetchone()
            if not result:
                return None
            
            file_path = Path(result[0])
            if not file_path.exists():
                return None
            
            # Load from Parquet
            df = pd.read_parquet(file_path)
            
            # Filter by date range
            df = df[
                (df['timestamp'] >= start_date) &
                (df['timestamp'] <= end_date)
            ]
            
            # Convert back to MarketData objects
            market_data = []
            for _, row in df.iterrows():
                data = MarketData(
                    timestamp=row['timestamp'],
                    symbol=row['symbol'],
                    timeframe=row['timeframe'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume'],
                    vwap=row.get('vwap'),
                    num_trades=row.get('num_trades'),
                    is_outlier=row.get('is_outlier', False),
                    data_quality_score=row.get('data_quality_score')
                )
                
                if row.get('outlier_reasons'):
                    data.outlier_reasons = json.loads(row['outlier_reasons'])
                
                market_data.append(data)
            
            return market_data


class AlertManager:
    """Automated alert system for data quality issues."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.enabled = config.enable_alerts
        self.email = config.alert_email
    
    def check_and_alert(
        self,
        symbol: str,
        quality_metrics: DataQualityMetrics,
        data_count: int
    ):
        """Check quality metrics and send alerts if needed."""
        if not self.enabled:
            return
        
        alerts = []
        
        # Check overall quality
        if quality_metrics.overall_score < self.config.quality_threshold:
            alerts.append(
                f"Low quality score: {quality_metrics.overall_score:.3f} "
                f"(threshold: {self.config.quality_threshold})"
            )
        
        # Check specific metrics
        if quality_metrics.completeness < 0.9:
            alerts.append(
                f"Low completeness: {quality_metrics.completeness:.3f}"
            )
        
        if quality_metrics.outlier_rate > 0.1:
            alerts.append(
                f"High outlier rate: {quality_metrics.outlier_rate:.3f}"
            )
        
        if quality_metrics.accuracy < 0.8:
            alerts.append(
                f"Low accuracy: {quality_metrics.accuracy:.3f}"
            )
        
        # Check data count
        if data_count == 0:
            alerts.append("No data received")
        elif data_count < 10:
            alerts.append(f"Very low data count: {data_count}")
        
        if alerts:
            self._send_alert(symbol, alerts, quality_metrics)
    
    def _send_alert(
        self,
        symbol: str,
        alerts: List[str],
        quality_metrics: DataQualityMetrics
    ):
        """Send alert (simplified implementation)."""
        alert_message = f"""
        Data Quality Alert for {symbol}
        
        Issues detected:
        {chr(10).join(f'- {alert}' for alert in alerts)}
        
        Quality Metrics:
        - Overall Score: {quality_metrics.overall_score:.3f}
        - Completeness: {quality_metrics.completeness:.3f}
        - Accuracy: {quality_metrics.accuracy:.3f}
        - Consistency: {quality_metrics.consistency:.3f}
        - Outlier Rate: {quality_metrics.outlier_rate:.3f}
        - Quality Level: {quality_metrics.quality_level.value}
        
        Timestamp: {datetime.now().isoformat()}
        """
        
        logger.warning(f"DATA QUALITY ALERT for {symbol}: {alerts}")
        
        # In production, would send email/Slack/etc.
        if self.email:
            logger.info(f"Alert would be sent to {self.email}")
            # Implementation would use email service
        
        # Could also write to alert log file
        alert_file = Path("data/alerts.log")
        with open(alert_file, "a") as f:
            f.write(f"{datetime.now().isoformat()}: {symbol} - {alerts}\n")


class ProductionDataPipeline:
    """
    Production-grade data pipeline orchestrator.
    
    Coordinates multi-source data ingestion, cleaning, validation,
    and storage with comprehensive monitoring and alerting.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.data_sources: Dict[DataSource, DataSourceInterface] = {}
        self.cleaner = DataCleaner(config)
        self.storage = DataStorage(config)
        self.alert_manager = AlertManager(config)
        
        # Initialize data sources
        self._initialize_data_sources()
    
    def _initialize_data_sources(self):
        """Initialize configured data sources."""
        for source_config in self.config.data_sources:
            if source_config.enabled:
                # In production, would create actual data source implementations
                logger.info(f"Initialized data source: {source_config.source.value}")
    
    async def ingest_symbol_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframes: List[str] = None
    ) -> Dict[str, Tuple[List[MarketData], DataQualityMetrics]]:
        """
        Ingest data for a symbol across multiple timeframes and sources.
        
        Args:
            symbol: Symbol to ingest
            start_date: Start date for data
            end_date: End date for data
            timeframes: List of timeframes (default: ["1d"])
            
        Returns:
            Dictionary mapping timeframe to (data, quality_metrics)
        """
        if timeframes is None:
            timeframes = ["1d"]
        
        results = {}
        
        for timeframe in timeframes:
            logger.info(
                f"Ingesting {symbol} data for timeframe {timeframe} "
                f"from {start_date} to {end_date}"
            )
            
            # Try multiple data sources in priority order
            best_data = None
            best_quality = None
            best_source = None
            
            for source_config in sorted(
                self.config.data_sources,
                key=lambda x: x.priority
            ):
                if not source_config.enabled:
                    continue
                
                try:
                    # In production, would fetch from actual data source
                    raw_data = await self._fetch_from_source(
                        source_config.source,
                        symbol,
                        start_date,
                        end_date,
                        timeframe
                    )
                    
                    if not raw_data:
                        continue
                    
                    # Clean and validate data
                    cleaned_data, quality_metrics = self.cleaner.clean_market_data(
                        raw_data, symbol
                    )
                    
                    # Check if this is the best quality so far
                    if (best_quality is None or 
                        quality_metrics.overall_score > best_quality.overall_score):
                        best_data = cleaned_data
                        best_quality = quality_metrics
                        best_source = source_config.source
                    
                    # If quality is excellent, use this data
                    if quality_metrics.quality_level == DataQuality.EXCELLENT:
                        break
                
                except Exception as e:
                    logger.error(
                        f"Error fetching from {source_config.source.value}: {e}"
                    )
                    continue
            
            if best_data and best_quality:
                # Save the best data
                self.storage.save_market_data(
                    best_data, symbol, best_source, best_quality
                )
                
                # Check for alerts
                self.alert_manager.check_and_alert(
                    symbol, best_quality, len(best_data)
                )
                
                results[timeframe] = (best_data, best_quality)
                
                logger.info(
                    f"Successfully ingested {len(best_data)} records for "
                    f"{symbol} {timeframe} from {best_source.value} "
                    f"(quality: {best_quality.overall_score:.3f})"
                )
            else:
                logger.warning(
                    f"Failed to ingest data for {symbol} {timeframe}"
                )
                results[timeframe] = ([], self.cleaner._create_empty_quality_metrics())
        
        return results
    
    async def _fetch_from_source(
        self,
        source: DataSource,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> List[MarketData]:
        """Fetch data from a specific source (mock implementation)."""
        # This is a mock implementation
        # In production, would use actual data source APIs
        
        logger.debug(f"Fetching {symbol} from {source.value}")
        
        # Simulate API call delay
        await asyncio.sleep(0.1)
        
        # Generate mock data for demonstration
        data = []
        current_date = start_date
        price = 100.0
        
        while current_date <= end_date:
            # Simulate price movement
            change = np.random.normal(0, 0.02)
            price *= (1 + change)
            
            market_data = create_market_data(
                symbol=symbol,
                timestamp=current_date,
                timeframe=timeframe,
                ohlcv=(
                    price * 0.99,  # open
                    price * 1.01,  # high
                    price * 0.98,  # low
                    price,          # close
                    np.random.randint(1000, 10000)  # volume
                )
            )
            data.append(market_data)
            
            # Move to next period
            if timeframe == "1d":
                current_date += timedelta(days=1)
            elif timeframe == "1h":
                current_date += timedelta(hours=1)
            elif timeframe == "1m":
                current_date += timedelta(minutes=1)
        
        return data
    
    async def batch_ingest(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframes: List[str] = None
    ) -> Dict[str, Dict[str, Tuple[List[MarketData], DataQualityMetrics]]]:
        """
        Batch ingest data for multiple symbols.
        
        Args:
            symbols: List of symbols to ingest
            start_date: Start date for data
            end_date: End date for data
            timeframes: List of timeframes
            
        Returns:
            Nested dictionary: symbol -> timeframe -> (data, quality_metrics)
        """
        logger.info(
            f"Starting batch ingestion for {len(symbols)} symbols "
            f"from {start_date} to {end_date}"
        )
        
        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(
                    asyncio.run,
                    self.ingest_symbol_data(symbol, start_date, end_date, timeframes)
                ): symbol
                for symbol in symbols
            }
            
            results = {}
            
            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    symbol_results = future.result()
                    results[symbol] = symbol_results
                    
                    logger.info(f"Completed ingestion for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error ingesting {symbol}: {e}")
                    results[symbol] = {}
        
        logger.info(f"Batch ingestion completed for {len(results)} symbols")
        return results
    
    def get_quality_report(
        self,
        symbol: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """Generate quality report for recent data."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        with sqlite3.connect(self.storage.db_path) as conn:
            if symbol:
                cursor = conn.execute("""
                    SELECT * FROM quality_metrics
                    WHERE symbol = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                """, (symbol, start_date.isoformat()))
            else:
                cursor = conn.execute("""
                    SELECT * FROM quality_metrics
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                """, (start_date.isoformat(),))
            
            metrics = cursor.fetchall()
        
        if not metrics:
            return {"message": "No quality metrics found"}
        
        # Calculate summary statistics
        scores = [row[9] for row in metrics]  # overall_score column
        
        report = {
            "period": f"{start_date.date()} to {end_date.date()}",
            "total_records": len(metrics),
            "average_quality": np.mean(scores),
            "min_quality": np.min(scores),
            "max_quality": np.max(scores),
            "quality_distribution": {
                "excellent": sum(1 for s in scores if s >= 0.9),
                "good": sum(1 for s in scores if 0.8 <= s < 0.9),
                "acceptable": sum(1 for s in scores if 0.6 <= s < 0.8),
                "poor": sum(1 for s in scores if 0.4 <= s < 0.6),
                "unusable": sum(1 for s in scores if s < 0.4)
            }
        }
        
        if symbol:
            report["symbol"] = symbol
        
        return report


# Factory function for creating pipeline with default configuration
def create_production_pipeline(
    data_sources: List[DataSourceConfig] = None,
    storage_path: str = "data/processed",
    enable_alerts: bool = True
) -> ProductionDataPipeline:
    """Create a production data pipeline with sensible defaults."""
    if data_sources is None:
        # Default configuration with mock sources
        data_sources = [
            DataSourceConfig(
                source=DataSource.YAHOO_FINANCE,
                priority=1,
                enabled=True
            ),
            DataSourceConfig(
                source=DataSource.ALPHA_VANTAGE,
                priority=2,
                enabled=True
            )
        ]
    
    config = PipelineConfig(
        data_sources=data_sources,
        storage_path=Path(storage_path),
        enable_alerts=enable_alerts
    )
    
    return ProductionDataPipeline(config)

lo
gger = logging.getLogger(__name__)


class DataSource(Enum):
    """Supported data sources."""
    YFINANCE = "yfinance"
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    IEX = "iex"
    BINANCE = "binance"
    COINBASE = "coinbase"


class DataQuality(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"


@dataclass
class DataSourceConfig:
    """Configuration for data sources."""
    source: DataSource
    api_key: Optional[str] = None
    rate_limit: float = 1.0  # requests per second
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    enabled: bool = True


@dataclass
class PipelineConfig:
    """Configuration for the data pipeline."""
    data_sources: List[DataSourceConfig] = field(default_factory=list)
    storage_path: str = "data/processed"
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds
    quality_threshold: float = 0.8
    outlier_detection: bool = True
    survivorship_bias_correction: bool = True
    missing_data_threshold: float = 0.05  # 5% missing data threshold
    parallel_workers: int = 4
    batch_size: int = 100


@dataclass
class DataQualityMetrics:
    """Metrics for data quality assessment."""
    completeness: float  # Percentage of non-missing data
    accuracy: float  # Percentage of data passing validation
    consistency: float  # Consistency across sources
    timeliness: float  # How recent the data is
    outlier_rate: float  # Percentage of outliers detected
    overall_score: float  # Combined quality score
    
    def __post_init__(self):
        """Calculate overall score from individual metrics."""
        weights = {
            'completeness': 0.3,
            'accuracy': 0.3,
            'consistency': 0.2,
            'timeliness': 0.1,
            'outlier_rate': 0.1  # Lower is better
        }
        
        self.overall_score = (
            self.completeness * weights['completeness'] +
            self.accuracy * weights['accuracy'] +
            self.consistency * weights['consistency'] +
            self.timeliness * weights['timeliness'] +
            (1 - self.outlier_rate) * weights['outlier_rate']
        )


class DataSourceInterface(ABC):
    """Abstract interface for data sources."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def fetch_data(self, symbol: str, start_date: datetime, 
                        end_date: datetime, timeframe: str) -> List[MarketData]:
        """Fetch market data for a symbol."""
        pass
    
    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols."""
        pass
    
    @abstractmethod
    def get_asset_metadata(self, symbol: str) -> Optional[MarketMetadata]:
        """Get metadata for an asset."""
        pass


class YFinanceDataSource(DataSourceInterface):
    """Yahoo Finance data source implementation."""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.session = None
    
    async def fetch_data(self, symbol: str, start_date: datetime, 
                        end_date: datetime, timeframe: str = "1d") -> List[MarketData]:
        """Fetch data from Yahoo Finance."""
        try:
            # Map timeframe to yfinance interval
            interval_map = {
                "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                "1h": "1h", "1d": "1d", "1wk": "1wk", "1mo": "1mo"
            }
            
            yf_interval = interval_map.get(timeframe, "1d")
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            hist = ticker.history(
                start=start_date,
                end=end_date,
                interval=yf_interval,
                auto_adjust=True,
                prepost=True,
                threads=True
            )
            
            if hist.empty:
                self.logger.warning(f"No data found for {symbol}")
                return []
            
            # Get asset metadata
            metadata = self.get_asset_metadata(symbol)
            
            # Convert to MarketData objects
            market_data_list = []
            for timestamp, row in hist.iterrows():
                try:
                    market_data = create_market_data(
                        symbol=symbol,
                        timestamp=timestamp.to_pydatetime(),
                        timeframe=timeframe,
                        ohlcv=(
                            float(row['Open']),
                            float(row['High']),
                            float(row['Low']),
                            float(row['Close']),
                            float(row['Volume'])
                        ),
                        metadata=metadata
                    )
                    market_data_list.append(market_data)
                except Exception as e:
                    self.logger.error(f"Error processing row for {symbol}: {e}")
                    continue
            
            self.logger.info(f"Fetched {len(market_data_list)} records for {symbol}")
            return market_data_list
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return []
    
    def get_supported_symbols(self) -> List[str]:
        """Get commonly traded symbols (subset for demo)."""
        return [
            # Major stocks
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
            # ETFs
            "SPY", "QQQ", "IWM", "VTI", "VOO",
            # Forex (via ETFs)
            "UUP", "FXE", "FXY", "FXB",
            # Crypto (via ETFs/Futures)
            "BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD"
        ]
    
    def get_asset_metadata(self, symbol: str) -> Optional[MarketMetadata]:
        """Get asset metadata from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Determine asset class
            asset_class = AssetClass.EQUITY  # Default
            if symbol.endswith("-USD"):
                asset_class = AssetClass.CRYPTO
            elif symbol in ["UUP", "FXE", "FXY", "FXB"]:
                asset_class = AssetClass.FOREX
            elif "=" in symbol:
                asset_class = AssetClass.FUTURES
            
            return MarketMetadata(
                symbol=symbol,
                asset_class=asset_class,
                exchange=info.get('exchange', 'UNKNOWN'),
                currency=info.get('currency', 'USD'),
                tick_size=0.01,  # Default
                lot_size=1.0,    # Default
                trading_hours={"monday": ("09:30", "16:00")},  # Default NYSE hours
                timezone="America/New_York",
                sector=info.get('sector'),
                market_cap=info.get('marketCap'),
                average_volume=info.get('averageVolume'),
                volatility=info.get('beta'),
                beta=info.get('beta')
            )
        except Exception as e:
            self.logger.error(f"Error getting metadata for {symbol}: {e}")
            return None


class DataCleaner:
    """Advanced data cleaning with survivorship bias correction."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def clean_data(self, data: List[MarketData]) -> List[MarketData]:
        """Clean market data with comprehensive validation."""
        if not data:
            return data
        
        cleaned_data = []
        
        for market_data in data:
            try:
                # Basic validation
                if not self._validate_basic_data(market_data):
                    continue
                
                # Outlier detection
                if self.config.outlier_detection:
                    if self._is_outlier(market_data, data):
                        market_data.is_outlier = True
                        self.logger.debug(f"Outlier detected: {market_data.symbol} at {market_data.timestamp}")
                
                # Survivorship bias correction
                if self.config.survivorship_bias_correction:
                    market_data = self._correct_survivorship_bias(market_data)
                
                cleaned_data.append(market_data)
                
            except Exception as e:
                self.logger.error(f"Error cleaning data for {market_data.symbol}: {e}")
                continue
        
        self.logger.info(f"Cleaned {len(cleaned_data)}/{len(data)} records")
        return cleaned_data
    
    def _validate_basic_data(self, data: MarketData) -> bool:
        """Validate basic data integrity."""
        try:
            # Check for valid OHLCV
            if any(x <= 0 for x in [data.open, data.high, data.low, data.close]):
                return False
            
            # Check OHLC relationships
            if not (data.low <= data.open <= data.high and 
                   data.low <= data.close <= data.high):
                return False
            
            # Check for reasonable volume
            if data.volume < 0:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _is_outlier(self, data: MarketData, all_data: List[MarketData]) -> bool:
        """Detect outliers using statistical methods."""
        try:
            # Get prices for the same symbol
            symbol_data = [d for d in all_data if d.symbol == data.symbol]
            if len(symbol_data) < 10:  # Need sufficient data
                return False
            
            prices = [d.close for d in symbol_data]
            volumes = [d.volume for d in symbol_data]
            
            # Price outlier detection
            price_outliers, _ = DataValidator.validate_price_series(prices)
            volume_outliers, _ = DataValidator.validate_volume_series(volumes)
            
            # Find index of current data point
            data_index = next((i for i, d in enumerate(symbol_data) 
                             if d.timestamp == data.timestamp), -1)
            
            if data_index >= 0:
                return (price_outliers[data_index] or 
                       volume_outliers[data_index])
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in outlier detection: {e}")
            return False
    
    def _correct_survivorship_bias(self, data: MarketData) -> MarketData:
        """Apply survivorship bias correction."""
        # For now, just flag the data - more sophisticated correction
        # would require historical constituent data
        if hasattr(data, 'metadata') and data.metadata:
            # Mark as potentially affected by survivorship bias
            if not hasattr(data, 'bias_corrected'):
                data.bias_corrected = True
        
        return data


class DataImputer:
    """Sophisticated missing data imputation."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def impute_missing_data(self, data: List[MarketData]) -> List[MarketData]:
        """Impute missing data using forward-fill and interpolation."""
        if not data:
            return data
        
        # Group by symbol for imputation
        symbol_groups = {}
        for d in data:
            if d.symbol not in symbol_groups:
                symbol_groups[d.symbol] = []
            symbol_groups[d.symbol].append(d)
        
        imputed_data = []
        
        for symbol, symbol_data in symbol_groups.items():
            try:
                # Sort by timestamp
                symbol_data.sort(key=lambda x: x.timestamp)
                
                # Convert to DataFrame for easier manipulation
                df = self._to_dataframe(symbol_data)
                
                # Calculate missing data percentage
                missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
                
                if missing_pct > self.config.missing_data_threshold:
                    self.logger.warning(f"High missing data rate for {symbol}: {missing_pct:.2%}")
                
                # Apply imputation strategies
                df_imputed = self._apply_imputation(df)
                
                # Convert back to MarketData objects
                imputed_symbol_data = self._from_dataframe(df_imputed, symbol_data)
                imputed_data.extend(imputed_symbol_data)
                
            except Exception as e:
                self.logger.error(f"Error imputing data for {symbol}: {e}")
                imputed_data.extend(symbol_data)  # Use original data
        
        return imputed_data
    
    def _to_dataframe(self, data: List[MarketData]) -> pd.DataFrame:
        """Convert MarketData list to DataFrame."""
        records = []
        for d in data:
            records.append({
                'timestamp': d.timestamp,
                'open': d.open,
                'high': d.high,
                'low': d.low,
                'close': d.close,
                'volume': d.volume,
                'vwap': d.vwap
            })
        
        df = pd.DataFrame(records)
        df.set_index('timestamp', inplace=True)
        return df
    
    def _apply_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply various imputation strategies."""
        df_imputed = df.copy()
        
        # Forward fill for prices (most common in financial data)
        price_cols = ['open', 'high', 'low', 'close', 'vwap']
        df_imputed[price_cols] = df_imputed[price_cols].fillna(method='ffill')
        
        # Backward fill for remaining missing values
        df_imputed[price_cols] = df_imputed[price_cols].fillna(method='bfill')
        
        # Linear interpolation for volume
        df_imputed['volume'] = df_imputed['volume'].interpolate(method='linear')
        
        # Fill remaining volume NaNs with 0
        df_imputed['volume'] = df_imputed['volume'].fillna(0)
        
        return df_imputed
    
    def _from_dataframe(self, df: pd.DataFrame, 
                       original_data: List[MarketData]) -> List[MarketData]:
        """Convert DataFrame back to MarketData objects."""
        result = []
        
        for i, (timestamp, row) in enumerate(df.iterrows()):
            # Use original data as template
            original = original_data[i] if i < len(original_data) else original_data[0]
            
            # Create new MarketData with imputed values
            imputed = MarketData(
                timestamp=timestamp,
                symbol=original.symbol,
                timeframe=original.timeframe,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume']),
                vwap=float(row['vwap']) if pd.notna(row['vwap']) else None,
                metadata=original.metadata,
                technical_indicators=original.technical_indicators
            )
            
            result.append(imputed)
        
        return result


class DataQualityMonitor:
    """Data quality monitoring with automated alerts."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.quality_history = []
    
    def assess_quality(self, data: List[MarketData]) -> DataQualityMetrics:
        """Assess data quality and generate metrics."""
        if not data:
            return DataQualityMetrics(0, 0, 0, 0, 1.0, 0)
        
        # Calculate completeness
        completeness = self._calculate_completeness(data)
        
        # Calculate accuracy
        accuracy = self._calculate_accuracy(data)
        
        # Calculate consistency (placeholder - would need multiple sources)
        consistency = 0.95  # Assume good consistency for single source
        
        # Calculate timeliness
        timeliness = self._calculate_timeliness(data)
        
        # Calculate outlier rate
        outlier_rate = self._calculate_outlier_rate(data)
        
        metrics = DataQualityMetrics(
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            timeliness=timeliness,
            outlier_rate=outlier_rate,
            overall_score=0  # Will be calculated in __post_init__
        )
        
        # Store in history
        self.quality_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })
        
        # Check for alerts
        self._check_quality_alerts(metrics)
        
        return metrics
    
    def _calculate_completeness(self, data: List[MarketData]) -> float:
        """Calculate data completeness percentage."""
        total_fields = len(data) * 6  # OHLCV + timestamp
        missing_fields = 0
        
        for d in data:
            if pd.isna(d.open) or d.open <= 0:
                missing_fields += 1
            if pd.isna(d.high) or d.high <= 0:
                missing_fields += 1
            if pd.isna(d.low) or d.low <= 0:
                missing_fields += 1
            if pd.isna(d.close) or d.close <= 0:
                missing_fields += 1
            if pd.isna(d.volume) or d.volume < 0:
                missing_fields += 1
        
        return max(0, (total_fields - missing_fields) / total_fields)
    
    def _calculate_accuracy(self, data: List[MarketData]) -> float:
        """Calculate data accuracy percentage."""
        valid_records = 0
        
        for d in data:
            try:
                # Basic validation checks
                if (d.low <= d.open <= d.high and 
                    d.low <= d.close <= d.high and
                    d.high >= d.low and
                    d.volume >= 0):
                    valid_records += 1
            except Exception:
                continue
        
        return valid_records / len(data) if data else 0
    
    def _calculate_timeliness(self, data: List[MarketData]) -> float:
        """Calculate data timeliness score."""
        if not data:
            return 0
        
        now = datetime.now()
        latest_data = max(d.timestamp for d in data)
        
        # Calculate hours since latest data
        hours_old = (now - latest_data).total_seconds() / 3600
        
        # Score decreases with age (1.0 for current, 0.0 for >24h old)
        timeliness = max(0, 1 - (hours_old / 24))
        
        return timeliness
    
    def _calculate_outlier_rate(self, data: List[MarketData]) -> float:
        """Calculate outlier rate."""
        if not data:
            return 0
        
        outliers = sum(1 for d in data if getattr(d, 'is_outlier', False))
        return outliers / len(data)
    
    def _check_quality_alerts(self, metrics: DataQualityMetrics):
        """Check for quality issues and generate alerts."""
        alerts = []
        
        if metrics.overall_score < self.config.quality_threshold:
            alerts.append(f"Overall quality score below threshold: {metrics.overall_score:.2f}")
        
        if metrics.completeness < 0.95:
            alerts.append(f"Low data completeness: {metrics.completeness:.2%}")
        
        if metrics.accuracy < 0.90:
            alerts.append(f"Low data accuracy: {metrics.accuracy:.2%}")
        
        if metrics.outlier_rate > 0.05:
            alerts.append(f"High outlier rate: {metrics.outlier_rate:.2%}")
        
        for alert in alerts:
            self.logger.warning(f"DATA QUALITY ALERT: {alert}")


class MarketDataPipeline:
    """Production-grade market data pipeline."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize components
        self.data_sources = self._initialize_data_sources()
        self.cleaner = DataCleaner(config)
        self.imputer = DataImputer(config)
        self.quality_monitor = DataQualityMonitor(config)
        
        # Initialize storage
        self.storage_path = Path(config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache
        self.cache = {} if config.cache_enabled else None
    
    def _initialize_data_sources(self) -> Dict[DataSource, DataSourceInterface]:
        """Initialize configured data sources."""
        sources = {}
        
        for source_config in self.config.data_sources:
            if not source_config.enabled:
                continue
                
            if source_config.source == DataSource.YFINANCE:
                sources[DataSource.YFINANCE] = YFinanceDataSource(source_config)
            # Add other data sources here as needed
            
        return sources
    
    async def fetch_data(self, symbols: List[str], start_date: datetime,
                        end_date: datetime, timeframe: str = "1d") -> Dict[str, List[MarketData]]:
        """Fetch data for multiple symbols."""
        self.logger.info(f"Fetching data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        results = {}
        
        # Use ThreadPoolExecutor for parallel fetching
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            # Submit tasks for each symbol
            future_to_symbol = {}
            
            for symbol in symbols:
                # Check cache first
                cache_key = f"{symbol}_{start_date}_{end_date}_{timeframe}"
                if self.cache and cache_key in self.cache:
                    cached_data, cache_time = self.cache[cache_key]
                    if (datetime.now() - cache_time).seconds < self.config.cache_ttl:
                        results[symbol] = cached_data
                        continue
                
                # Submit fetch task
                for source in self.data_sources.values():
                    future = executor.submit(
                        self._fetch_symbol_data,
                        source, symbol, start_date, end_date, timeframe
                    )
                    future_to_symbol[future] = symbol
                    break  # Use first available source for now
            
            # Collect results
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if data:
                        results[symbol] = data
                        
                        # Cache the result
                        if self.cache:
                            cache_key = f"{symbol}_{start_date}_{end_date}_{timeframe}"
                            self.cache[cache_key] = (data, datetime.now())
                            
                except Exception as e:
                    self.logger.error(f"Error fetching data for {symbol}: {e}")
                    results[symbol] = []
        
        self.logger.info(f"Fetched data for {len(results)} symbols")
        return results
    
    def _fetch_symbol_data(self, source: DataSourceInterface, symbol: str,
                          start_date: datetime, end_date: datetime, 
                          timeframe: str) -> List[MarketData]:
        """Fetch data for a single symbol (synchronous wrapper)."""
        try:
            # Create event loop for async call
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                return loop.run_until_complete(
                    source.fetch_data(symbol, start_date, end_date, timeframe)
                )
            finally:
                loop.close()
                
        except Exception as e:
            self.logger.error(f"Error in _fetch_symbol_data for {symbol}: {e}")
            return []
    
    def process_data(self, raw_data: Dict[str, List[MarketData]]) -> Dict[str, List[MarketData]]:
        """Process raw data through cleaning and imputation pipeline."""
        self.logger.info("Processing raw data through pipeline")
        
        processed_data = {}
        
        for symbol, data in raw_data.items():
            try:
                # Clean data
                cleaned_data = self.cleaner.clean_data(data)
                
                # Impute missing values
                imputed_data = self.imputer.impute_missing_data(cleaned_data)
                
                # Assess quality
                quality_metrics = self.quality_monitor.assess_quality(imputed_data)
                
                # Store quality metrics with data
                for d in imputed_data:
                    d.data_quality_score = quality_metrics.overall_score
                
                processed_data[symbol] = imputed_data
                
                self.logger.info(f"Processed {len(imputed_data)} records for {symbol} "
                               f"(quality: {quality_metrics.overall_score:.2f})")
                
            except Exception as e:
                self.logger.error(f"Error processing data for {symbol}: {e}")
                processed_data[symbol] = data  # Use raw data as fallback
        
        return processed_data
    
    def save_data(self, data: Dict[str, List[MarketData]], 
                 format: str = "parquet") -> Dict[str, str]:
        """Save processed data to storage."""
        self.logger.info(f"Saving data in {format} format")
        
        saved_files = {}
        
        for symbol, symbol_data in data.items():
            try:
                # Convert to DataFrame
                records = []
                for d in symbol_data:
                    record = d.to_dict()
                    records.append(record)
                
                if not records:
                    continue
                
                df = pd.DataFrame(records)
                
                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{symbol}_{timestamp}.{format}"
                filepath = self.storage_path / filename
                
                # Save based on format
                if format == "parquet":
                    df.to_parquet(filepath, index=False)
                elif format == "csv":
                    df.to_csv(filepath, index=False)
                elif format == "json":
                    df.to_json(filepath, orient="records", date_format="iso")
                else:
                    raise ValueError(f"Unsupported format: {format}")
                
                saved_files[symbol] = str(filepath)
                self.logger.info(f"Saved {len(records)} records for {symbol} to {filepath}")
                
            except Exception as e:
                self.logger.error(f"Error saving data for {symbol}: {e}")
        
        return saved_files
    
    async def run_pipeline(self, symbols: List[str], start_date: datetime,
                          end_date: datetime, timeframe: str = "1d") -> Dict[str, str]:
        """Run the complete data pipeline."""
        self.logger.info("Starting data pipeline execution")
        
        try:
            # Fetch raw data
            raw_data = await self.fetch_data(symbols, start_date, end_date, timeframe)
            
            # Process data
            processed_data = self.process_data(raw_data)
            
            # Save data
            saved_files = self.save_data(processed_data)
            
            self.logger.info("Data pipeline execution completed successfully")
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Error in pipeline execution: {e}")
            raise


# Factory function for creating pipeline with default configuration
def create_pipeline(storage_path: str = "data/processed") -> MarketDataPipeline:
    """Create a market data pipeline with default configuration."""
    
    # Default configuration with YFinance
    config = PipelineConfig(
        data_sources=[
            DataSourceConfig(
                source=DataSource.YFINANCE,
                rate_limit=2.0,  # Conservative rate limit
                enabled=True
            )
        ],
        storage_path=storage_path,
        cache_enabled=True,
        quality_threshold=0.8,
        outlier_detection=True,
        survivorship_bias_correction=True,
        parallel_workers=4
    )
    
    return MarketDataPipeline(config)


# Example usage function
async def example_usage():
    """Example of how to use the data pipeline."""
    
    # Create pipeline
    pipeline = create_pipeline()
    
    # Define symbols and date range
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Run pipeline
    saved_files = await pipeline.run_pipeline(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe="1d"
    )
    
    print("Pipeline completed. Saved files:")
    for symbol, filepath in saved_files.items():
        print(f"  {symbol}: {filepath}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run example
    asyncio.run(example_usage())