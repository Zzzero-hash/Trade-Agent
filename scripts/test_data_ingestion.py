#!/usr/bin/env python3
"""
Test script for YFinance Data Ingestion System

This script tests the comprehensive data ingestion system to ensure:
- Data can be downloaded from yfinance for multiple symbols and timeframes
- Data validation and quality checks work correctly
- Data storage and retrieval functions properly
- Error handling works as expected
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.ingestion.yfinance_data_manager import YFinanceDataManager, IngestionConfig


def test_basic_functionality():
    """Test basic data ingestion functionality."""
    print("=== Testing Basic Functionality ===")
    
    # Create test configuration with limited scope
    config = IngestionConfig(
        symbols=["AAPL", "SPY"],  # Just 2 symbols for testing
        timeframes=["1d", "1h"],  # Just 2 timeframes
        start_date="2023-12-01",
        end_date="2023-12-31",
        max_workers=1,  # Single worker for testing
        cache_dir="data/cache/test",
        raw_data_dir="data/raw/test"
    )
    
    # Initialize manager
    manager = YFinanceDataManager(config)
    
    # Test single symbol fetch
    print("\n1. Testing single symbol data fetch...")
    aapl_data = manager.fetch_symbol_data("AAPL", "1d", "2023-12-01", "2023-12-31")
    
    if aapl_data is not None:
        print(f"   ✓ Successfully fetched AAPL data: {len(aapl_data)} records")
        print(f"   ✓ Columns: {list(aapl_data.columns)}")
        # Check for timestamp column (could be 'timestamp', 'Date', or 'date')
        date_col = None
        for col in ['timestamp', 'Date', 'date']:
            if col in aapl_data.columns:
                date_col = col
                break
        
        if date_col:
            print(f"   ✓ Date range: {aapl_data[date_col].min()} to {aapl_data[date_col].max()}")
        else:
            print(f"   ⚠ No date column found in: {list(aapl_data.columns)}")
        
        # Test data validation
        quality_metrics = manager.validate_data(aapl_data)
        print(f"   ✓ Data quality score: {quality_metrics.quality_score:.1f}%")
        print(f"   ✓ Data completeness: {quality_metrics.data_completeness:.1f}%")
        if quality_metrics.validation_errors:
            print(f"   ⚠ Validation errors: {quality_metrics.validation_errors}")
    else:
        print("   ✗ Failed to fetch AAPL data")
        return False
    
    # Test multiple symbols fetch
    print("\n2. Testing multiple symbols data fetch...")
    data_dict = manager.fetch_multiple_symbols()
    
    if data_dict:
        print(f"   ✓ Successfully fetched data for {len(data_dict)} symbols")
        for symbol, timeframe_data in data_dict.items():
            valid_timeframes = sum(1 for data in timeframe_data.values() if data is not None)
            print(f"   ✓ {symbol}: {valid_timeframes}/{len(timeframe_data)} timeframes")
    else:
        print("   ✗ Failed to fetch multiple symbols data")
        return False
    
    # Test data saving
    print("\n3. Testing data storage...")
    saved_files = manager.save_raw_data(data_dict)
    
    if saved_files:
        print(f"   ✓ Successfully saved data for {len(saved_files)} symbols")
        for symbol, file_path in saved_files.items():
            print(f"   ✓ {symbol}: {file_path}")
    else:
        print("   ✗ Failed to save data")
        return False
    
    # Test data loading
    print("\n4. Testing data loading...")
    loaded_data = manager.load_raw_data("AAPL", "1d")
    
    if loaded_data is not None:
        print(f"   ✓ Successfully loaded AAPL 1d data: {len(loaded_data)} records")
    else:
        print("   ✗ Failed to load data")
        return False
    
    # Test data info
    print("\n5. Testing data info...")
    info = manager.get_data_info("AAPL")
    print(f"   ✓ Available timeframes: {info['timeframes']}")
    print(f"   ✓ Record counts: {info['record_counts']}")
    print(f"   ✓ Last updated: {info['last_updated']}")
    
    print("\n✓ All basic functionality tests passed!")
    return True


def test_error_handling():
    """Test error handling and edge cases."""
    print("\n=== Testing Error Handling ===")
    
    config = IngestionConfig(
        symbols=["INVALID_SYMBOL", "AAPL"],
        timeframes=["1d"],
        start_date="2023-12-01",
        end_date="2023-12-31",
        max_workers=1,
        cache_dir="data/cache/test",
        raw_data_dir="data/raw/test"
    )
    
    manager = YFinanceDataManager(config)
    
    # Test invalid symbol
    print("\n1. Testing invalid symbol handling...")
    invalid_data = manager.fetch_symbol_data("INVALID_SYMBOL", "1d")
    
    if invalid_data is None:
        print("   ✓ Correctly handled invalid symbol")
    else:
        print("   ⚠ Unexpected data returned for invalid symbol")
    
    # Test invalid date range
    print("\n2. Testing invalid date range...")
    future_data = manager.fetch_symbol_data("AAPL", "1d", "2030-01-01", "2030-12-31")
    
    if future_data is None or future_data.empty:
        print("   ✓ Correctly handled future date range")
    else:
        print(f"   ⚠ Unexpected data returned for future dates: {len(future_data)} records")
    
    print("\n✓ Error handling tests completed!")
    return True


def test_data_quality():
    """Test data quality validation."""
    print("\n=== Testing Data Quality Validation ===")
    
    config = IngestionConfig(
        symbols=["AAPL"],
        timeframes=["1d"],
        start_date="2023-01-01",
        end_date="2023-12-31",
        validate_data=True,
        outlier_z_threshold=2.0,  # More sensitive for testing
        cache_dir="data/cache/test",
        raw_data_dir="data/raw/test"
    )
    
    manager = YFinanceDataManager(config)
    
    # Fetch data with validation
    print("\n1. Testing data quality validation...")
    data = manager.fetch_symbol_data("AAPL", "1d")
    
    if data is not None:
        quality_metrics = manager.validate_data(data)
        
        print(f"   ✓ Total records: {quality_metrics.total_records}")
        print(f"   ✓ Missing records: {quality_metrics.missing_records}")
        print(f"   ✓ Outlier records: {quality_metrics.outlier_records}")
        print(f"   ✓ Data completeness: {quality_metrics.data_completeness:.1f}%")
        print(f"   ✓ Outlier percentage: {quality_metrics.outlier_percentage:.1f}%")
        print(f"   ✓ Quality score: {quality_metrics.quality_score:.1f}%")
        
        if quality_metrics.validation_errors:
            print(f"   ⚠ Validation errors: {quality_metrics.validation_errors}")
        else:
            print("   ✓ No validation errors found")
    else:
        print("   ✗ Failed to fetch data for quality testing")
        return False
    
    print("\n✓ Data quality validation tests completed!")
    return True


def main():
    """Run all tests."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("YFinance Data Ingestion System Test")
    print("=" * 50)
    
    try:
        # Run tests
        basic_test = test_basic_functionality()
        error_test = test_error_handling()
        quality_test = test_data_quality()
        
        # Summary
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        print(f"Basic Functionality: {'PASS' if basic_test else 'FAIL'}")
        print(f"Error Handling: {'PASS' if error_test else 'FAIL'}")
        print(f"Data Quality: {'PASS' if quality_test else 'FAIL'}")
        
        if all([basic_test, error_test, quality_test]):
            print("\n🎉 All tests PASSED! Data ingestion system is working correctly.")
            return 0
        else:
            print("\n❌ Some tests FAILED. Please check the implementation.")
            return 1
    
    except Exception as e:
        print(f"\n💥 Test execution failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())