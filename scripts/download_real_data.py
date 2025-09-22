#!/usr/bin/env python3
"""
Script to download and process real market data from Yahoo Finance.

This script downloads real stock data and prepares it for CNN+LSTM training.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import logging
from data.ingestion.yahoo_finance import YahooFinanceIngestor
from data.datasets.real_market_dataset import RealMarketDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('experiments/logs/data_download.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main function to download and process real market data."""
    logger.info("Starting real market data download and processing")
    
    # Create directories
    data_dirs = [
        "data/raw",
        "data/processed", 
        "data/cache",
        "experiments/logs"
    ]
    
    for dir_path in data_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Initialize ingestor
    ingestor = YahooFinanceIngestor(cache_dir="data/cache")
    
    # Symbols to download (focus on highly liquid stocks/ETFs)
    symbols = [
        "SPY",    # S&P 500 ETF - benchmark
        "QQQ",    # Nasdaq 100 ETF - tech focus
        "AAPL",   # Apple - large tech
        "MSFT",   # Microsoft - large tech
        "GOOGL",  # Google - large tech
        "AMZN",   # Amazon - large tech
        "NVDA",   # NVIDIA - AI/semiconductors
        "TSLA"    # Tesla - volatile growth
    ]
    
    # Download data for each symbol
    for symbol in symbols:
        try:
            logger.info(f"Processing {symbol}...")
            
            # Create multi-timeframe dataset
            dataset = ingestor.create_multi_timeframe_dataset(
                symbol=symbol,
                start_date="2020-01-01",
                end_date="2024-06-30",
                cache=True
            )
            
            if dataset:
                # Save the dataset
                ingestor.save_dataset(dataset, symbol, "data/processed")
                logger.info(f"Successfully processed {symbol}")
            else:
                logger.warning(f"No data available for {symbol}")
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue
    
    # Test loading one dataset
    logger.info("Testing dataset loading...")
    try:
        test_dataset = RealMarketDataset(
            symbol="SPY",
            data_dir="data/processed",
            timeframes=["1min", "5min", "15min"],
            sequence_length=100
        )
        logger.info(f"Test dataset loaded successfully: {len(test_dataset)} samples")
    except Exception as e:
        logger.error(f"Error loading test dataset: {e}")
    
    logger.info("Data download and processing completed")


if __name__ == "__main__":
    main()
