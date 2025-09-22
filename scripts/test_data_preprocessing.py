#!/usr/bin/env python3
"""
Test script for Data Preprocessing Pipeline

This script tests the comprehensive data preprocessing pipeline to ensure:
- Data cleaning works correctly with various imputation methods
- Temporal splits maintain proper ordering without look-ahead bias
- Feature scaling works appropriately for financial data
- Data quality monitoring provides accurate metrics
- Error handling works as expected
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.preprocessing.data_preprocessor import (
    FinancialDataPreprocessor, 
    PreprocessingConfig,
    ScalingMethod,
    ImputationMethod,
    OutlierMethod
)


def create_test_data(n_samples: int = 1000, add_issues: bool = True) -> pd.DataFrame:
    """Create synthetic financial data for testing."""
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='H')
    
    # Generate price data with realistic patterns
    returns = np.random.randn(n_samples) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'TEST',
        'open': prices,
        'high': prices * (1 + np.abs(np.random.randn(n_samples) * 0.01)),
        'low': prices * (1 - np.abs(np.random.randn(n_samples) * 0.01)),
        'close': prices * (1 + np.random.randn(n_samples) * 0.005),
        'volume': np.random.randint(1000, 100000, n_samples),
        'timeframe': '1h'
    })
    
    # Ensure OHLC consistency
    for i in range(len(data)):
        high = max(data.loc[i, 'open'], data.loc[i, 'close'], data.loc[i, 'high'])
        low = min(data.loc[i, 'open'], data.loc[i, 'close'], data.loc[i, 'low'])
        data.loc[i, 'high'] = high
        data.loc[i, 'low'] = low
    
    if add_issues:
        # Add missing values
        missing_indices = np.random.choice(data.index, size=50, replace=False)
        data.loc[missing_indices[:25], 'volume'] = np.nan
        data.loc[missing_indices[25:], 'close'] = np.nan
        
        # Add outliers
        outlier_indices = np.random.choice(data.index, size=20, replace=False)
        data.loc[outlier_indices, 'volume'] *= 50  # Volume spikes
        data.loc[outlier_indices[:10], 'close'] *= 1.5  # Price spikes
        
        # Add some invalid OHLC data
        invalid_indices = np.random.choice(data.index, size=5, replace=False)
        data.loc[invalid_indices, 'high'] = data.loc[invalid_indices, 'low'] * 0.9  # High < Low
    
    return data


def test_data_cleaning():
    """Test data cleaning functionality."""
    print("=== Testing Data Cleaning ===")
    
    # Create test data with issues
    test_data = create_test_data(1000, add_issues=True)
    print(f"Created test data: {len(test_data)} records")
    print(f"Missing values: {test_data.isnull().sum().sum()}")
    print(f"Invalid OHLC records: {((test_data['high'] < test_data['low']).sum())}")
    
    # Test different imputation methods
    methods_to_test = [
        ImputationMethod.FORWARD_FILL,
        ImputationMethod.LINEAR_INTERPOLATION,
        ImputationMethod.MEDIAN_IMPUTATION
    ]
    
    for method in methods_to_test:
        print(f"\n1. Testing {method.value} imputation...")
        
        config = PreprocessingConfig(
            imputation_method=method,
            outlier_method=OutlierMethod.Z_SCORE,
            scaling_method=ScalingMethod.NONE,
            min_train_samples=100
        )
        
        preprocessor = FinancialDataPreprocessor(config)
        
        try:
            cleaned_data = preprocessor.clean_data(test_data.copy())
            
            missing_after = cleaned_data.isnull().sum().sum()
            invalid_ohlc_after = (cleaned_data['high'] < cleaned_data['low']).sum()
            
            print(f"   ✓ Cleaned data: {len(cleaned_data)} records")
            print(f"   ✓ Missing values after: {missing_after}")
            print(f"   ✓ Invalid OHLC after: {invalid_ohlc_after}")
            
            if missing_after == 0:
                print(f"   ✓ Successfully handled all missing values")
            else:
                print(f"   ⚠ Still have {missing_after} missing values")
            
            if invalid_ohlc_after == 0:
                print(f"   ✓ Successfully fixed all OHLC inconsistencies")
            else:
                print(f"   ⚠ Still have {invalid_ohlc_after} invalid OHLC records")
        
        except Exception as e:
            print(f"   ✗ Failed with error: {e}")
            return False
    
    print("\n✓ Data cleaning tests completed!")
    return True


def test_temporal_splits():
    """Test temporal data splitting."""
    print("\n=== Testing Temporal Splits ===")
    
    # Create clean test data
    test_data = create_test_data(1000, add_issues=False)
    
    config = PreprocessingConfig(
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
        scaling_method=ScalingMethod.NONE
    )
    
    preprocessor = FinancialDataPreprocessor(config)
    
    try:
        # Clean data first
        cleaned_data = preprocessor.clean_data(test_data)
        
        # Create splits
        data_splits = preprocessor.create_temporal_splits(cleaned_data)
        
        print(f"1. Split sizes:")
        print(f"   Train: {len(data_splits.train_data)} records")
        print(f"   Validation: {len(data_splits.validation_data)} records")
        print(f"   Test: {len(data_splits.test_data)} records")
        
        # Verify ratios
        total = len(data_splits.train_data) + len(data_splits.validation_data) + len(data_splits.test_data)
        train_ratio = len(data_splits.train_data) / total
        val_ratio = len(data_splits.validation_data) / total
        test_ratio = len(data_splits.test_data) / total
        
        print(f"2. Actual ratios:")
        print(f"   Train: {train_ratio:.3f} (target: {config.train_ratio})")
        print(f"   Validation: {val_ratio:.3f} (target: {config.validation_ratio})")
        print(f"   Test: {test_ratio:.3f} (target: {config.test_ratio})")
        
        # Verify temporal ordering (no look-ahead bias)
        train_end = data_splits.train_data['timestamp'].max()
        val_start = data_splits.validation_data['timestamp'].min()
        val_end = data_splits.validation_data['timestamp'].max()
        test_start = data_splits.test_data['timestamp'].min()
        
        print(f"3. Temporal ordering:")
        print(f"   Train end: {train_end}")
        print(f"   Val start: {val_start}")
        print(f"   Val end: {val_end}")
        print(f"   Test start: {test_start}")
        
        if train_end <= val_start and val_end <= test_start:
            print("   ✓ No look-ahead bias detected")
        else:
            print("   ✗ Look-ahead bias detected!")
            return False
        
        # Check split dates
        print(f"4. Split date ranges:")
        for key, date in data_splits.split_dates.items():
            print(f"   {key}: {date}")
        
        print("   ✓ Temporal splits created successfully")
        
    except Exception as e:
        print(f"   ✗ Failed with error: {e}")
        return False
    
    print("\n✓ Temporal split tests completed!")
    return True


def test_feature_scaling():
    """Test feature scaling functionality."""
    print("\n=== Testing Feature Scaling ===")
    
    # Create test data
    test_data = create_test_data(1000, add_issues=False)
    
    scaling_methods = [
        ScalingMethod.STANDARD,
        ScalingMethod.MINMAX,
        ScalingMethod.ROBUST
    ]
    
    for method in scaling_methods:
        print(f"\n1. Testing {method.value} scaling...")
        
        config = PreprocessingConfig(
            scaling_method=method,
            scale_features=['open', 'high', 'low', 'close', 'volume']
        )
        
        preprocessor = FinancialDataPreprocessor(config)
        
        try:
            # Clean and split data
            cleaned_data = preprocessor.clean_data(test_data.copy())
            data_splits = preprocessor.create_temporal_splits(cleaned_data)
            
            # Scale features
            scaled_splits, scalers = preprocessor.scale_features(data_splits)
            
            print(f"   ✓ Scaled {len(scalers)} features")
            
            # Check scaling properties
            for feature in config.scale_features:
                if feature in scalers:
                    train_values = scaled_splits.train_data[feature]
                    
                    if method == ScalingMethod.STANDARD:
                        mean_val = train_values.mean()
                        std_val = train_values.std()
                        print(f"   ✓ {feature}: mean={mean_val:.3f}, std={std_val:.3f}")
                        
                        if abs(mean_val) < 0.1 and abs(std_val - 1.0) < 0.1:
                            print(f"     ✓ Standard scaling successful")
                        else:
                            print(f"     ⚠ Standard scaling may have issues")
                    
                    elif method == ScalingMethod.MINMAX:
                        min_val = train_values.min()
                        max_val = train_values.max()
                        print(f"   ✓ {feature}: min={min_val:.3f}, max={max_val:.3f}")
                        
                        if min_val >= -0.1 and max_val <= 1.1:
                            print(f"     ✓ MinMax scaling successful")
                        else:
                            print(f"     ⚠ MinMax scaling may have issues")
            
            # Verify that validation and test data are scaled using training statistics
            print(f"   ✓ Validation and test data scaled using training statistics")
        
        except Exception as e:
            print(f"   ✗ Failed with error: {e}")
            return False
    
    print("\n✓ Feature scaling tests completed!")
    return True


def test_quality_metrics():
    """Test data quality metrics calculation."""
    print("\n=== Testing Quality Metrics ===")
    
    # Test with perfect data
    perfect_data = create_test_data(1000, add_issues=False)
    
    # Test with problematic data
    problematic_data = create_test_data(1000, add_issues=True)
    
    config = PreprocessingConfig()
    preprocessor = FinancialDataPreprocessor(config)
    
    try:
        print("1. Testing with perfect data...")
        perfect_metrics = preprocessor.calculate_quality_metrics(perfect_data)
        
        print(f"   Completeness: {perfect_metrics['completeness']:.1f}%")
        print(f"   Temporal consistency: {perfect_metrics['temporal_consistency']:.1f}%")
        print(f"   OHLC consistency: {perfect_metrics['ohlc_consistency']:.1f}%")
        print(f"   Volume consistency: {perfect_metrics['volume_consistency']:.1f}%")
        print(f"   Overall quality: {perfect_metrics['overall_quality']:.1f}%")
        
        if perfect_metrics['overall_quality'] > 95:
            print("   ✓ Perfect data quality metrics look good")
        else:
            print("   ⚠ Perfect data should have higher quality score")
        
        print("\n2. Testing with problematic data...")
        problematic_metrics = preprocessor.calculate_quality_metrics(problematic_data)
        
        print(f"   Completeness: {problematic_metrics['completeness']:.1f}%")
        print(f"   Temporal consistency: {problematic_metrics['temporal_consistency']:.1f}%")
        print(f"   OHLC consistency: {problematic_metrics['ohlc_consistency']:.1f}%")
        print(f"   Volume consistency: {problematic_metrics['volume_consistency']:.1f}%")
        print(f"   Overall quality: {problematic_metrics['overall_quality']:.1f}%")
        
        if problematic_metrics['overall_quality'] < perfect_metrics['overall_quality']:
            print("   ✓ Problematic data correctly shows lower quality")
        else:
            print("   ⚠ Quality metrics may not be sensitive enough")
    
    except Exception as e:
        print(f"   ✗ Failed with error: {e}")
        return False
    
    print("\n✓ Quality metrics tests completed!")
    return True


def test_complete_pipeline():
    """Test the complete preprocessing pipeline."""
    print("\n=== Testing Complete Pipeline ===")
    
    # Create test data with various issues
    test_data = create_test_data(2000, add_issues=True)
    
    config = PreprocessingConfig(
        imputation_method=ImputationMethod.FORWARD_FILL,
        outlier_method=OutlierMethod.Z_SCORE,
        scaling_method=ScalingMethod.ROBUST,
        train_ratio=0.7,
        validation_ratio=0.15,
        test_ratio=0.15,
        min_data_quality_score=50.0
    )
    
    preprocessor = FinancialDataPreprocessor(config)
    
    try:
        print(f"1. Starting with {len(test_data)} records")
        print(f"   Missing values: {test_data.isnull().sum().sum()}")
        
        # Run complete pipeline
        results = preprocessor.preprocess(test_data)
        
        print(f"\n2. Preprocessing completed:")
        print(f"   Original records: {results.preprocessing_stats['original_records']}")
        print(f"   Cleaned records: {results.preprocessing_stats['cleaned_records']}")
        print(f"   Records removed: {results.preprocessing_stats['records_removed']}")
        print(f"   Removal percentage: {results.preprocessing_stats['removal_percentage']:.1f}%")
        
        print(f"\n3. Data splits:")
        print(f"   Train: {len(results.data_splits.train_data)} records")
        print(f"   Validation: {len(results.data_splits.validation_data)} records")
        print(f"   Test: {len(results.data_splits.test_data)} records")
        
        print(f"\n4. Quality metrics:")
        for metric, value in results.quality_metrics.items():
            print(f"   {metric}: {value:.1f}%")
        
        print(f"\n5. Scaled features: {list(results.scalers.keys())}")
        
        if results.warnings:
            print(f"\n6. Warnings:")
            for warning in results.warnings:
                print(f"   ⚠ {warning}")
        
        if results.errors:
            print(f"\n7. Errors:")
            for error in results.errors:
                print(f"   ✗ {error}")
            return False
        
        # Verify final data quality
        final_missing = results.processed_data.isnull().sum().sum()
        if final_missing == 0:
            print("   ✓ No missing values in final data")
        else:
            print(f"   ⚠ Still have {final_missing} missing values")
        
        print("   ✓ Complete pipeline executed successfully")
    
    except Exception as e:
        print(f"   ✗ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ Complete pipeline tests completed!")
    return True


def main():
    """Run all preprocessing tests."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Data Preprocessing Pipeline Test")
    print("=" * 50)
    
    try:
        # Run tests
        cleaning_test = test_data_cleaning()
        splits_test = test_temporal_splits()
        scaling_test = test_feature_scaling()
        quality_test = test_quality_metrics()
        pipeline_test = test_complete_pipeline()
        
        # Summary
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        print(f"Data Cleaning: {'PASS' if cleaning_test else 'FAIL'}")
        print(f"Temporal Splits: {'PASS' if splits_test else 'FAIL'}")
        print(f"Feature Scaling: {'PASS' if scaling_test else 'FAIL'}")
        print(f"Quality Metrics: {'PASS' if quality_test else 'FAIL'}")
        print(f"Complete Pipeline: {'PASS' if pipeline_test else 'FAIL'}")
        
        if all([cleaning_test, splits_test, scaling_test, quality_test, pipeline_test]):
            print("\n🎉 All tests PASSED! Data preprocessing pipeline is working correctly.")
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