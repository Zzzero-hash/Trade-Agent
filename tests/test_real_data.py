#!/usr/bin/env python3
"""
Test script to verify real market data download and processing.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from data.ingestion.yahoo_finance import YahooFinanceIngestor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_yahoo_finance_download():
    """Test downloading data from Yahoo Finance."""
    logger.info("Testing Yahoo Finance data download...")
    
    # Create ingestor
    ingestor = YahooFinanceIngestor(cache_dir="data/cache")
    
    # Test with a single symbol
    symbol = "SPY"
    logger.info(f"Downloading data for {symbol}...")
    
    try:
        # Create multi-timeframe dataset
        dataset = ingestor.create_multi_timeframe_dataset(
            symbol=symbol,
            start_date="2023-01-01",
            end_date="2023-12-31",
            cache=True
        )
        
        if dataset:
            logger.info(f"Successfully downloaded data for {symbol}")
            for timeframe, data in dataset.items():
                logger.info(f"  {timeframe}: {len(data)} rows")
                
            # Save the dataset
            ingestor.save_dataset(dataset, symbol, "data/processed")
            logger.info(f"Saved dataset for {symbol}")
            return True
        else:
            logger.error(f"Failed to download data for {symbol}")
            return False
            
    except Exception as e:
        logger.error(f"Error downloading data for {symbol}: {e}")
        return False


def main():
    """Main test function."""
    logger.info("Starting real data test...")
    
    # Create required directories
    dirs = ["data/cache", "data/processed", "data/raw"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Test Yahoo Finance download
    success = test_yahoo_finance_download()
    
    if success:
        logger.info("Real data test completed successfully!")
        print("\n✓ Yahoo Finance data download test passed")
        print("✓ Data processing pipeline is ready for real market data")
        print("\nTo train with real data, run:")
        print("  python experiments/runners/train_cnn_multi_timeframe_real.py")
    else:
        logger.error("Real data test failed!")
        print("\n✗ Yahoo Finance data download test failed")
        print("Please check your internet connection and try again")


if __name__ == "__main__":
    main()
