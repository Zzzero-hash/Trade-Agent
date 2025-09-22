"""
YFinance Data Manager - Comprehensive data ingestion system for real market data.

This module implements a production-ready data ingestion system that:
- Downloads OHLCV data from 100+ liquid stocks using yfinance
- Supports multi-timeframe data collection (1m, 5m, 15m, 1h, 1d)
- Implements comprehensive data validation and quality checks
- Provides efficient data storage using Parquet/HDF5 with compression
- Handles missing data, outliers, and data integrity issues
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Set
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import time
import hashlib
import json

import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow as pa
import pyarrow.parquet as pq

from data.models import MarketData, DataValidator, create_market_data

# Suppress yfinance warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')

logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """Data quality metrics for validation."""
    total_records: int = 0
    missing_records: int = 0
    outlier_records: int = 0
    data_completeness: float = 0.0
    outlier_percentage: float = 0.0
    quality_score: float = 0.0
    validation_errors: List[str] = field(default_factory=list)
    
    def calculate_quality_score(self):
        """Calculate overall data quality score (0-100)."""
        completeness_score = self.data_completeness * 0.6
        outlier_penalty = min(self.outlier_percentage * 0.4, 40)  # Max 40% penalty
        error_penalty = min(len(self.validation_errors) * 5, 30)  # Max 30% penalty
        
        self.quality_score = max(0, completeness_score - outlier_penalty - error_penalty)


@dataclass
class IngestionConfig:
    """Configuration for data ingestion."""
    symbols: List[str] = field(default_factory=lambda: [
        # Large Cap Tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
        # Large Cap Traditional
        "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "ADBE",
        # ETFs for market exposure
        "SPY", "QQQ", "IWM", "VTI", "VOO", "VEA", "VWO", "GLD", "TLT", "HYG",
        # Financial sector
        "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW",
        # Healthcare
        "PFE", "ABBV", "TMO", "ABT", "MRK", "LLY", "BMY", "AMGN",
        # Consumer
        "KO", "PEP", "WMT", "TGT", "NKE", "SBUX", "MCD", "COST",
        # Industrial
        "BA", "CAT", "GE", "MMM", "HON", "UPS", "LMT", "RTX",
        # Energy
        "XOM", "CVX", "COP", "EOG", "SLB", "KMI", "OXY", "MPC",
        # Utilities
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "PEG",
        # Real Estate
        "AMT", "PLD", "CCI", "EQIX", "PSA", "WELL", "DLR", "O",
        # Materials
        "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "DOW", "DD",
        # Communication Services
        "T", "VZ", "CMCSA", "CHTR", "TMUS", "ATVI", "EA", "TTWO"
    ])
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h", "1d"])
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    cache_dir: str = "data/cache"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    max_workers: int = 4
    rate_limit_delay: float = 0.5
    retry_attempts: int = 3
    chunk_size_days: int = 7  # For 1m data
    compression: str = "snappy"  # Parquet compression
    validate_data: bool = True
    outlier_z_threshold: float = 3.0
    volume_spike_threshold: float = 10.0


class YFinanceDataManager:
    """
    Comprehensive yfinance data ingestion system.
    
    Features:
    - Multi-symbol, multi-timeframe data collection
    - Robust error handling and retry logic
    - Data validation and quality checks
    - Efficient storage with compression
    - Progress tracking and logging
    - Concurrent downloads with rate limiting
    """
    
    def __init__(self, config: Optional[IngestionConfig] = None):
        """Initialize the data manager."""
        self.config = config or IngestionConfig()
        self._setup_directories()
        self._setup_logging()
        
        # Track download statistics
        self.download_stats = {
            'successful': 0,
            'failed': 0,
            'cached': 0,
            'total_records': 0,
            'quality_issues': 0
        }
        
        # Timeframe configurations
        self.timeframe_configs = {
            "1m": {"max_days": 7, "chunk_days": 7},
            "5m": {"max_days": 60, "chunk_days": 30},
            "15m": {"max_days": 60, "chunk_days": 30},
            "1h": {"max_days": 730, "chunk_days": 90},
            "1d": {"max_days": None, "chunk_days": None}
        }
    
    def _setup_directories(self):
        """Create necessary directories."""
        for dir_path in [self.config.cache_dir, self.config.raw_data_dir, 
                        self.config.processed_data_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _get_cache_key(self, symbol: str, timeframe: str, 
                      start_date: str, end_date: str) -> str:
        """Generate cache key for data."""
        key_string = f"{symbol}_{timeframe}_{start_date}_{end_date}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, symbol: str, timeframe: str, 
                       start_date: str, end_date: str) -> Path:
        """Get cache file path."""
        cache_key = self._get_cache_key(symbol, timeframe, start_date, end_date)
        return Path(self.config.cache_dir) / f"{cache_key}.parquet"
    
    def _is_data_cached(self, symbol: str, timeframe: str, 
                       start_date: str, end_date: str) -> bool:
        """Check if data is already cached."""
        cache_path = self._get_cache_path(symbol, timeframe, start_date, end_date)
        return cache_path.exists()
    
    def _load_cached_data(self, symbol: str, timeframe: str, 
                         start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load data from cache."""
        try:
            cache_path = self._get_cache_path(symbol, timeframe, start_date, end_date)
            if cache_path.exists():
                data = pd.read_parquet(cache_path)
                logger.info(f"Loaded cached data for {symbol} {timeframe}: {len(data)} records")
                self.download_stats['cached'] += 1
                return data
        except Exception as e:
            logger.warning(f"Failed to load cached data for {symbol} {timeframe}: {e}")
        return None
    
    def _save_to_cache(self, data: pd.DataFrame, symbol: str, timeframe: str,
                      start_date: str, end_date: str):
        """Save data to cache."""
        try:
            cache_path = self._get_cache_path(symbol, timeframe, start_date, end_date)
            data.to_parquet(cache_path, compression=self.config.compression)
            logger.debug(f"Cached data for {symbol} {timeframe}: {len(data)} records")
        except Exception as e:
            logger.warning(f"Failed to cache data for {symbol} {timeframe}: {e}")
    
    def _fetch_data_chunk(self, symbol: str, timeframe: str, 
                         start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch a single chunk of data from yfinance."""
        for attempt in range(self.config.retry_attempts):
            try:
                ticker = yf.Ticker(symbol)
                
                # Add delay for rate limiting
                if attempt > 0:
                    time.sleep(self.config.rate_limit_delay * (2 ** attempt))
                
                logger.debug(f"Fetching {symbol} {timeframe} data: {start_date} to {end_date} (attempt {attempt + 1})")
                
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=timeframe,
                    auto_adjust=False,
                    back_adjust=False,
                    prepost=False,
                    actions=False
                )
                
                if data is None or data.empty:
                    logger.warning(f"No data returned for {symbol} {timeframe}")
                    return None
                
                # Normalize column names (handle special cases first)
                data.columns = data.columns.str.lower().str.replace(' ', '_')
                data['symbol'] = symbol
                data['timeframe'] = timeframe
                
                # Reset index to make timestamp a column
                data = data.reset_index()
                if 'Datetime' in data.columns:
                    data = data.rename(columns={'Datetime': 'timestamp'})
                elif 'Date' in data.columns:
                    data = data.rename(columns={'Date': 'timestamp'})
                elif 'datetime' in data.columns:
                    data = data.rename(columns={'datetime': 'timestamp'})
                elif 'date' in data.columns:
                    data = data.rename(columns={'date': 'timestamp'})
                
                logger.info(f"Successfully fetched {symbol} {timeframe}: {len(data)} records")
                return data
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol} {timeframe}: {e}")
                if attempt == self.config.retry_attempts - 1:
                    logger.error(f"All attempts failed for {symbol} {timeframe}")
                    self.download_stats['failed'] += 1
                    return None
        
        return None
    
    def _fetch_data_in_chunks(self, symbol: str, timeframe: str, 
                             start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch data in chunks for timeframes with limitations."""
        config = self.timeframe_configs.get(timeframe, {})
        max_days = config.get('max_days')
        chunk_days = config.get('chunk_days', self.config.chunk_size_days)
        
        # If no chunking needed, fetch all at once
        if max_days is None:
            return self._fetch_data_chunk(symbol, timeframe, start_date, end_date)
        
        # Calculate date range
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # For 1m data, limit to recent data due to yfinance restrictions
        if timeframe == "1m":
            # yfinance only provides 1m data for the last 30 days
            max_start = datetime.now() - timedelta(days=30)
            if start_dt < max_start:
                start_dt = max_start
                logger.warning(f"Adjusted start date for {symbol} 1m data to {start_dt.date()}")
        
        all_chunks = []
        current_start = start_dt
        
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=chunk_days), end_dt)
            
            chunk_start_str = current_start.strftime("%Y-%m-%d")
            chunk_end_str = current_end.strftime("%Y-%m-%d")
            
            chunk_data = self._fetch_data_chunk(symbol, timeframe, chunk_start_str, chunk_end_str)
            
            if chunk_data is not None and not chunk_data.empty:
                all_chunks.append(chunk_data)
            
            # Rate limiting between chunks
            time.sleep(self.config.rate_limit_delay)
            current_start = current_end
        
        if all_chunks:
            combined_data = pd.concat(all_chunks, ignore_index=True)
            # Remove duplicates that might occur at chunk boundaries
            combined_data = combined_data.drop_duplicates(subset=['timestamp', 'symbol'])
            combined_data = combined_data.sort_values('timestamp')
            return combined_data
        
        return None
    
    def fetch_symbol_data(self, symbol: str, timeframe: str, 
                         start_date: Optional[str] = None, 
                         end_date: Optional[str] = None,
                         use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single symbol and timeframe.
        
        Args:
            symbol: Stock symbol to fetch
            timeframe: Data timeframe ('1m', '5m', '15m', '1h', '1d')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        start_date = start_date or self.config.start_date
        end_date = end_date or self.config.end_date
        
        # Check cache first
        if use_cache and self._is_data_cached(symbol, timeframe, start_date, end_date):
            cached_data = self._load_cached_data(symbol, timeframe, start_date, end_date)
            if cached_data is not None:
                return cached_data
        
        # Fetch fresh data
        data = self._fetch_data_in_chunks(symbol, timeframe, start_date, end_date)
        
        if data is not None:
            # Save to cache
            if use_cache:
                self._save_to_cache(data, symbol, timeframe, start_date, end_date)
            
            self.download_stats['successful'] += 1
            self.download_stats['total_records'] += len(data)
            
            return data
        
        return None
    
    def validate_data(self, data: pd.DataFrame) -> DataQualityMetrics:
        """
        Validate data quality and detect outliers.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            DataQualityMetrics with validation results
        """
        metrics = DataQualityMetrics()
        metrics.total_records = len(data)
        
        if data.empty:
            metrics.validation_errors.append("empty_dataset")
            return metrics
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            metrics.validation_errors.append(f"missing_columns_{missing_cols}")
        
        # Check for missing values
        missing_count = data[required_cols].isnull().sum().sum()
        metrics.missing_records = missing_count
        metrics.data_completeness = (1 - missing_count / (len(data) * len(required_cols))) * 100
        
        # Validate OHLCV consistency
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close']) |
            (data['open'] <= 0) |
            (data['high'] <= 0) |
            (data['low'] <= 0) |
            (data['close'] <= 0) |
            (data['volume'] < 0)
        ).sum()
        
        if invalid_ohlc > 0:
            metrics.validation_errors.append(f"invalid_ohlcv_{invalid_ohlc}_records")
        
        # Detect outliers using statistical methods
        if self.config.validate_data and len(data) > 2:
            try:
                # Price outliers
                price_outliers, _ = DataValidator.validate_price_series(
                    data['close'].tolist(), 
                    self.config.outlier_z_threshold
                )
                
                # Volume outliers
                volume_outliers, _ = DataValidator.validate_volume_series(
                    data['volume'].tolist(),
                    self.config.volume_spike_threshold
                )
                
                outlier_count = sum(price_outliers) + sum(volume_outliers)
                metrics.outlier_records = outlier_count
                metrics.outlier_percentage = (outlier_count / len(data)) * 100
                
                if metrics.outlier_percentage > 10:  # More than 10% outliers
                    metrics.validation_errors.append(f"high_outlier_rate_{metrics.outlier_percentage:.1f}%")
                
            except Exception as e:
                logger.warning(f"Outlier detection failed: {e}")
                metrics.validation_errors.append("outlier_detection_failed")
        
        # Calculate overall quality score
        metrics.calculate_quality_score()
        
        if metrics.quality_score < 70:
            self.download_stats['quality_issues'] += 1
        
        return metrics
    
    def fetch_multiple_symbols(self, symbols: Optional[List[str]] = None,
                              timeframes: Optional[List[str]] = None,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              use_cache: bool = True,
                              max_workers: Optional[int] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch data for multiple symbols and timeframes concurrently.
        
        Args:
            symbols: List of symbols to fetch
            timeframes: List of timeframes to fetch
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            use_cache: Whether to use cached data
            max_workers: Maximum number of concurrent workers
            
        Returns:
            Nested dictionary: {symbol: {timeframe: DataFrame}}
        """
        symbols = symbols or self.config.symbols
        timeframes = timeframes or self.config.timeframes
        max_workers = max_workers or self.config.max_workers
        
        results = {}
        total_tasks = len(symbols) * len(timeframes)
        completed_tasks = 0
        
        logger.info(f"Starting data ingestion for {len(symbols)} symbols, {len(timeframes)} timeframes")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_params = {}
            for symbol in symbols:
                for timeframe in timeframes:
                    future = executor.submit(
                        self.fetch_symbol_data, 
                        symbol, timeframe, start_date, end_date, use_cache
                    )
                    future_to_params[future] = (symbol, timeframe)
            
            # Collect results
            for future in as_completed(future_to_params):
                symbol, timeframe = future_to_params[future]
                completed_tasks += 1
                
                try:
                    data = future.result()
                    
                    if symbol not in results:
                        results[symbol] = {}
                    
                    if data is not None:
                        # Validate data quality
                        quality_metrics = self.validate_data(data)
                        
                        logger.info(
                            f"[{completed_tasks}/{total_tasks}] {symbol} {timeframe}: "
                            f"{len(data)} records, quality: {quality_metrics.quality_score:.1f}%"
                        )
                        
                        if quality_metrics.validation_errors:
                            logger.warning(f"Quality issues for {symbol} {timeframe}: {quality_metrics.validation_errors}")
                        
                        results[symbol][timeframe] = data
                    else:
                        logger.error(f"[{completed_tasks}/{total_tasks}] Failed to fetch {symbol} {timeframe}")
                        results[symbol][timeframe] = None
                
                except Exception as e:
                    logger.error(f"Error processing {symbol} {timeframe}: {e}")
                    if symbol not in results:
                        results[symbol] = {}
                    results[symbol][timeframe] = None
        
        self._log_download_summary()
        return results
    
    def save_raw_data(self, data_dict: Dict[str, Dict[str, pd.DataFrame]], 
                     format: str = "parquet") -> Dict[str, str]:
        """
        Save raw data to storage with efficient compression.
        
        Args:
            data_dict: Nested dictionary of data
            format: Storage format ('parquet' or 'hdf5')
            
        Returns:
            Dictionary mapping symbols to file paths
        """
        saved_files = {}
        raw_data_path = Path(self.config.raw_data_dir)
        
        for symbol, timeframe_data in data_dict.items():
            try:
                if format.lower() == "parquet":
                    # Save each timeframe separately for efficient querying
                    symbol_dir = raw_data_path / symbol
                    symbol_dir.mkdir(exist_ok=True)
                    
                    for timeframe, data in timeframe_data.items():
                        if data is not None and not data.empty:
                            file_path = symbol_dir / f"{timeframe}.parquet"
                            data.to_parquet(file_path, compression=self.config.compression)
                            logger.debug(f"Saved {symbol} {timeframe} to {file_path}")
                    
                    saved_files[symbol] = str(symbol_dir)
                
                elif format.lower() == "hdf5":
                    # Save all timeframes for a symbol in one HDF5 file
                    file_path = raw_data_path / f"{symbol}.h5"
                    
                    with pd.HDFStore(file_path, mode='w', complevel=9, complib='blosc') as store:
                        for timeframe, data in timeframe_data.items():
                            if data is not None and not data.empty:
                                store[f"{timeframe}"] = data
                    
                    saved_files[symbol] = str(file_path)
                    logger.debug(f"Saved {symbol} all timeframes to {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to save data for {symbol}: {e}")
        
        logger.info(f"Saved raw data for {len(saved_files)} symbols to {raw_data_path}")
        return saved_files
    
    def load_raw_data(self, symbol: str, timeframe: Optional[str] = None,
                     format: str = "parquet") -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load raw data from storage.
        
        Args:
            symbol: Symbol to load
            timeframe: Specific timeframe to load (None for all)
            format: Storage format ('parquet' or 'hdf5')
            
        Returns:
            DataFrame or dictionary of DataFrames
        """
        raw_data_path = Path(self.config.raw_data_dir)
        
        try:
            if format.lower() == "parquet":
                symbol_dir = raw_data_path / symbol
                
                if timeframe:
                    file_path = symbol_dir / f"{timeframe}.parquet"
                    if file_path.exists():
                        return pd.read_parquet(file_path)
                    else:
                        logger.warning(f"File not found: {file_path}")
                        return None
                else:
                    # Load all timeframes
                    data_dict = {}
                    for tf_file in symbol_dir.glob("*.parquet"):
                        tf_name = tf_file.stem
                        data_dict[tf_name] = pd.read_parquet(tf_file)
                    return data_dict
            
            elif format.lower() == "hdf5":
                file_path = raw_data_path / f"{symbol}.h5"
                
                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    return None
                
                with pd.HDFStore(file_path, mode='r') as store:
                    if timeframe:
                        if f"/{timeframe}" in store:
                            return store[timeframe]
                        else:
                            logger.warning(f"Timeframe {timeframe} not found in {file_path}")
                            return None
                    else:
                        # Load all timeframes
                        data_dict = {}
                        for key in store.keys():
                            tf_name = key.strip('/')
                            data_dict[tf_name] = store[key]
                        return data_dict
        
        except Exception as e:
            logger.error(f"Failed to load data for {symbol}: {e}")
            return None
    
    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with available data."""
        raw_data_path = Path(self.config.raw_data_dir)
        symbols = []
        
        for item in raw_data_path.iterdir():
            if item.is_dir():
                symbols.append(item.name)
            elif item.suffix == '.h5':
                symbols.append(item.stem)
        
        return sorted(symbols)
    
    def get_data_info(self, symbol: str) -> Dict[str, any]:
        """Get information about available data for a symbol."""
        info = {
            'symbol': symbol,
            'timeframes': [],
            'date_range': {},
            'record_counts': {},
            'file_sizes': {},
            'last_updated': None
        }
        
        raw_data_path = Path(self.config.raw_data_dir)
        symbol_dir = raw_data_path / symbol
        
        if symbol_dir.exists():
            for tf_file in symbol_dir.glob("*.parquet"):
                timeframe = tf_file.stem
                info['timeframes'].append(timeframe)
                
                try:
                    # Get file stats
                    stat = tf_file.stat()
                    info['file_sizes'][timeframe] = stat.st_size
                    
                    if info['last_updated'] is None or stat.st_mtime > info['last_updated']:
                        info['last_updated'] = datetime.fromtimestamp(stat.st_mtime)
                    
                    # Get data info without loading full dataset
                    parquet_file = pq.ParquetFile(tf_file)
                    info['record_counts'][timeframe] = parquet_file.metadata.num_rows
                    
                    # Get date range from metadata if available
                    if parquet_file.schema_arrow.metadata:
                        metadata = parquet_file.schema_arrow.metadata
                        if b'date_range' in metadata:
                            date_range_str = metadata[b'date_range'].decode()
                            info['date_range'][timeframe] = json.loads(date_range_str)
                
                except Exception as e:
                    logger.warning(f"Failed to get info for {symbol} {timeframe}: {e}")
        
        return info
    
    def _log_download_summary(self):
        """Log download statistics summary."""
        stats = self.download_stats
        total_attempts = stats['successful'] + stats['failed']
        success_rate = (stats['successful'] / total_attempts * 100) if total_attempts > 0 else 0
        
        logger.info("=== Download Summary ===")
        logger.info(f"Successful downloads: {stats['successful']}")
        logger.info(f"Failed downloads: {stats['failed']}")
        logger.info(f"Cached downloads: {stats['cached']}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"Total records: {stats['total_records']:,}")
        logger.info(f"Quality issues: {stats['quality_issues']}")
        logger.info("========================")
    
    def cleanup_cache(self, older_than_days: int = 7):
        """Clean up old cache files."""
        cache_path = Path(self.config.cache_dir)
        cutoff_time = datetime.now() - timedelta(days=older_than_days)
        
        removed_count = 0
        for cache_file in cache_path.glob("*.parquet"):
            if datetime.fromtimestamp(cache_file.stat().st_mtime) < cutoff_time:
                cache_file.unlink()
                removed_count += 1
        
        logger.info(f"Cleaned up {removed_count} old cache files")


def main():
    """Example usage of YFinanceDataManager."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration
    config = IngestionConfig(
        symbols=["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"],
        timeframes=["1d", "1h", "15m"],
        start_date="2023-01-01",
        end_date="2024-01-01",
        max_workers=2
    )
    
    # Initialize data manager
    manager = YFinanceDataManager(config)
    
    # Fetch data for multiple symbols
    print("Fetching data for multiple symbols...")
    data_dict = manager.fetch_multiple_symbols()
    
    # Save raw data
    print("Saving raw data...")
    saved_files = manager.save_raw_data(data_dict)
    
    # Display summary
    print(f"\nData ingestion completed!")
    print(f"Symbols processed: {len(data_dict)}")
    print(f"Files saved: {len(saved_files)}")
    
    # Show data info for first symbol
    if data_dict:
        first_symbol = list(data_dict.keys())[0]
        info = manager.get_data_info(first_symbol)
        print(f"\nData info for {first_symbol}:")
        print(f"  Timeframes: {info['timeframes']}")
        print(f"  Record counts: {info['record_counts']}")
        print(f"  Last updated: {info['last_updated']}")


if __name__ == "__main__":
    main()