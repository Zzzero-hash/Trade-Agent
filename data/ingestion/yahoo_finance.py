"""
Yahoo Finance Data Ingestion Module

This module provides functionality to fetch real market data from Yahoo Finance
and preprocess it for training CNN+LSTM models.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

from data.models import create_market_data

logger = logging.getLogger(__name__)


class YahooFinanceIngestor:
    """Ingestor for Yahoo Finance market data."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize Yahoo Finance ingestor.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Common stock symbols to fetch
        self.default_symbols = [
            "SPY",    # S&P 500 ETF
            "QQQ",    # Nasdaq 100 ETF
            "AAPL",   # Apple
            "MSFT",   # Microsoft
            "GOOGL",  # Google
            "AMZN",   # Amazon
            "TSLA",   # Tesla
            "NVDA",   # NVIDIA
            "META",   # Meta/Facebook
            "NFLX"    # Netflix
        ]
    
    def fetch_symbol_data(
        self,
        symbol: str,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        interval: str = "1m",
        cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single symbol from Yahoo Finance.
        
        Args:
            symbol: Stock symbol to fetch
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            cache: Whether to cache the data
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            # Adjust date ranges based on Yahoo Finance limitations
            adjusted_start, adjusted_end = self._adjust_dates_for_interval(start_date, end_date, interval)
            
            # Check cache first
            cache_file = self.cache_dir / f"{symbol}_{adjusted_start}_{adjusted_end}_{interval}.parquet"
            if cache and cache_file.exists():
                logger.info(f"Loading {symbol} data from cache: {cache_file}")
                return pd.read_parquet(cache_file)
            
            # Fetch data from Yahoo Finance
            logger.info(f"Fetching {symbol} data from Yahoo Finance...")
            ticker = yf.Ticker(symbol)
            
            # For 1m interval, we can only get 7 days at a time and only recent data
            if interval == "1m":
                data = self._fetch_1m_data_in_chunks(ticker, adjusted_start, adjusted_end)
            elif interval in ["5m", "15m"]:
                # 5m and 15m data limited to last 60 days
                data = self._fetch_intraday_data_in_chunks(ticker, adjusted_start, adjusted_end, interval)
            else:
                data = ticker.history(
                    start=adjusted_start,
                    end=adjusted_end,
                    interval=interval,
                    auto_adjust=False,
                    back_adjust=False
                )
            
            if data is None or data.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Add symbol column
            data['symbol'] = symbol
            
            # Cache the data
            if cache:
                logger.info(f"Caching {symbol} data to: {cache_file}")
                data.to_parquet(cache_file)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
        """
        Fetch data for a single symbol from Yahoo Finance.
        
        Args:
            symbol: Stock symbol to fetch
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            cache: Whether to cache the data
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            # Check cache first
            cache_file = self.cache_dir / f"{symbol}_{start_date}_{end_date}_{interval}.parquet"
            if cache and cache_file.exists():
                logger.info(f"Loading {symbol} data from cache: {cache_file}")
                return pd.read_parquet(cache_file)
            
            # Fetch data from Yahoo Finance
            logger.info(f"Fetching {symbol} data from Yahoo Finance...")
            ticker = yf.Ticker(symbol)
            
            # For 1m interval, we can only get 7 days at a time
            if interval == "1m":
                data = self._fetch_1m_data_in_chunks(ticker, start_date, end_date)
            else:
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=False,
                    back_adjust=False
                )
            
            if data is None or data.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Add symbol column
            data['symbol'] = symbol
            
            # Cache the data
            if cache:
                logger.info(f"Caching {symbol} data to: {cache_file}")
                data.to_parquet(cache_file)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _adjust_dates_for_interval(self, start_date: str, end_date: str, interval: str) -> Tuple[str, str]:
        """
        Adjust date ranges based on Yahoo Finance limitations for different intervals.
        
        Args:
            start_date: Original start date
            end_date: Original end date  
            interval: Data interval
            
        Returns:
            Tuple of (adjusted_start_date, adjusted_end_date)
        """
        end_dt = datetime.now()
        
        if interval == "1m":
            # 1m data only available for last 30 days
            start_dt = end_dt - timedelta(days=29)
        elif interval in ["5m", "15m"]:
            # 5m and 15m data only available for last 60 days
            start_dt = end_dt - timedelta(days=59)
        else:
            # Daily and longer intervals can use original dates
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            except:
                # Fallback to recent data if date parsing fails
                end_dt = datetime.now()
                start_dt = end_dt - timedelta(days=365)
        
        return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
    
    def _fetch_intraday_data_in_chunks(
        self,
        ticker: yf.Ticker,
        start_date: str,
        end_date: str,
        interval: str
    ) -> pd.DataFrame:
        """
        Fetch intraday data (5m, 15m) in chunks due to Yahoo Finance limitations.
        
        Args:
            ticker: Yahoo Finance ticker object
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval
            
        Returns:
            Combined DataFrame with all data
        """
        try:
            # For 5m and 15m, we can get all data in one request within 60 days
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False,
                back_adjust=False
            )
            return data
        except Exception as e:
            logger.error(f"Error fetching {interval} data: {e}")
            return pd.DataFrame()

    def _fetch_1m_data_in_chunks(
        self,
        ticker: yf.Ticker,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch 1-minute data in chunks due to Yahoo Finance limitations.
        
        Args:
            ticker: Yahoo Finance ticker object
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Combined DataFrame with all data
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_data = []
        current_start = start_dt
        
        while current_start < end_dt:
            # Yahoo Finance only allows 7 days of 1m data at a time
            current_end = min(current_start + timedelta(days=7), end_dt)
            
            logger.info(f"Fetching 1m data for period: {current_start.date()} to {current_end.date()}")
            
            try:
                chunk = ticker.history(
                    start=current_start.strftime("%Y-%m-%d"),
                    end=current_end.strftime("%Y-%m-%d"),
                    interval="1m",
                    auto_adjust=False,
                    back_adjust=False
                )
                
                if chunk is not None and not chunk.empty:
                    all_data.append(chunk)
                
                # Add delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"Error fetching chunk {current_start.date()}: {e}")
            
            current_start = current_end + timedelta(days=1)
        
        if all_data:
            return pd.concat(all_data).drop_duplicates()
        else:
            return pd.DataFrame()
    
    def fetch_multiple_symbols(
        self,
        symbols: List[str] = None,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        interval: str = "1m",
        cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols: List of stock symbols to fetch (uses default if None)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval
            cache: Whether to cache the data
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        if symbols is None:
            symbols = self.default_symbols
        
        data_dict = {}
        
        for symbol in symbols:
            data = self.fetch_symbol_data(symbol, start_date, end_date, interval, cache)
            if data is not None:
                data_dict[symbol] = data
                logger.info(f"Successfully fetched {len(data)} rows for {symbol}")
            else:
                logger.warning(f"Failed to fetch data for {symbol}")
            
            # Add delay to avoid rate limiting
            time.sleep(1)
        
        return data_dict
    
    def create_multi_timeframe_dataset(
        self,
        symbol: str,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Create a multi-timeframe dataset for a single symbol.
        
        Args:
            symbol: Stock symbol to fetch
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            cache: Whether to cache the data
            
        Returns:
            Dictionary with different timeframes as keys
        """
        timeframes = {
            "1min": "1m",
            "5min": "5m", 
            "15min": "15m",
            "1hour": "1h",
            "1day": "1d"
        }
        
        dataset = {}
        
        for timeframe_key, interval in timeframes.items():
            logger.info(f"Fetching {timeframe_key} data for {symbol}")
            data = self.fetch_symbol_data(symbol, start_date, end_date, interval, cache)
            
            if data is not None:
                # Resample to ensure consistent OHLCV columns
                if 'Open' in data.columns and 'High' in data.columns:
                    # Rename columns to lowercase for consistency
                    data = data.rename(columns={
                        'Open': 'open',
                        'High': 'high', 
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    })
                    
                    # Ensure all required columns exist
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    for col in required_cols:
                        if col not in data.columns:
                            data[col] = 0.0
                    
                    dataset[timeframe_key] = data[required_cols + ['symbol']]
                    logger.info(f"Added {timeframe_key} data: {len(data)} rows")
                else:
                    logger.warning(f"Missing required columns in {timeframe_key} data for {symbol}")
            else:
                logger.warning(f"Failed to fetch {timeframe_key} data for {symbol}")
        
        return dataset
    
    def save_dataset(
        self,
        dataset: Dict[str, pd.DataFrame],
        symbol: str,
        output_dir: str = "data/processed"
    ):
        """
        Save dataset to processed directory.
        
        Args:
            dataset: Dictionary of timeframes to DataFrames
            symbol: Stock symbol
            output_dir: Output directory
        """
        output_path = Path(output_dir) / symbol
        output_path.mkdir(parents=True, exist_ok=True)
        
        for timeframe, data in dataset.items():
            file_path = output_path / f"{timeframe}.parquet"
            data.to_parquet(file_path)
            logger.info(f"Saved {timeframe} data to {file_path}")


def main():
    """Example usage of the Yahoo Finance ingestor."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create ingestor
    ingestor = YahooFinanceIngestor()
    
    # Fetch data for a single symbol
    print("Fetching SPY data...")
    spy_data = ingestor.fetch_symbol_data("SPY", "2023-01-01", "2023-12-31", "1d")
    if spy_data is not None:
        print(f"SPY data shape: {spy_data.shape}")
        print(spy_data.head())
    
    # Create multi-timeframe dataset
    print("\nCreating multi-timeframe dataset for AAPL...")
    aapl_dataset = ingestor.create_multi_timeframe_dataset("AAPL", "2023-01-01", "2023-12-31")
    print(f"Created dataset with {len(aapl_dataset)} timeframes")
    
    for timeframe, data in aapl_dataset.items():
        print(f"  {timeframe}: {data.shape}")


if __name__ == "__main__":
    main()
