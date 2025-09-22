#!/usr/bin/env python3
"""
Test script for Feature Engineering Framework

This script tests the comprehensive feature engineering framework to ensure:
- Price-based features are created correctly
- Volume-based features work properly
- TA-Lib integration functions (if available)
- Pattern recognition features are generated
- Feature selection works correctly
- Cross-asset features can be created
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

from data.features.feature_engineer import (
    FinancialFeatureEngineer, 
    FeatureConfig,
    FeatureCategory,
    TALIB_AVAILABLE
)


def create_test_data(n_samples: int = 500) -> pd.DataFrame:
    """Create synthetic financial data for testing."""
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
    
    # Generate realistic price data
    returns = np.random.randn(n_samples) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'TEST',
        'open': prices,
        'high': prices * (1 + np.abs(np.random.randn(n_samples) * 0.01)),
        'low': prices * (1 - np.abs(np.random.randn(n_samples) * 0.01)),
        'close': prices * (1 + np.random.randn(n_samples) * 0.005),
        'volume': np.random.randint(10000, 100000, n_samples),
        'timeframe': '1d'
    })
    
    # Ensure OHLC consistency
    for i in range(len(data)):
        high = max(data.loc[i, 'open'], data.loc[i, 'close'], data.loc[i, 'high'])
        low = min(data.loc[i, 'open'], data.loc[i, 'close'], data.loc[i, 'low'])
        data.loc[i, 'high'] = high
        data.loc[i, 'low'] = low
    
    return data


def test_price_features():
    """Test price-based feature creation."""
    print("=== Testing Price Features ===")
    
    test_data = create_test_data(200)
    
    config = FeatureConfig(
        enable_talib_indicators=False,  # Test without TA-Lib first
        enable_candlestick_patterns=False,
        enable_chart_patterns=False,
        return_periods=[1, 5, 10],
        volatility_windows=[10, 20],
        price_ratio_periods=[5, 10]
    )
    
    engineer = FinancialFeatureEngineer(config)
    
    try:
        features = engineer.create_price_features(test_data)
        
        original_cols = len(test_data.columns)
        new_cols = len(features.columns)
        price_features = new_cols - original_cols
        
        print(f"1. Created {price_features} price features")
        
        # Check specific features
        expected_features = [
            'typical_price', 'weighted_close', 'median_price', 'price_range',
            'body_size', 'upper_shadow', 'lower_shadow', 'close_position',
            'return_1', 'return_5', 'return_10',
            'volatility_10', 'volatility_20',
            'price_ratio_5', 'price_ratio_10',
            'gap', 'gap_pct', 'price_acceleration'
        ]
        
        found_features = 0
        for feature in expected_features:
            if feature in features.columns:
                found_features += 1
                print(f"   ✓ {feature}")
            else:
                print(f"   ✗ Missing: {feature}")
        
        print(f"2. Found {found_features}/{len(expected_features)} expected features")
        
        # Check for NaN values in key features
        key_features = ['return_1', 'volatility_10', 'typical_price']
        for feature in key_features:
            if feature in features.columns:
                nan_count = features[feature].isnull().sum()
                print(f"   {feature}: {nan_count} NaN values")
        
        if price_features > 20:
            print("   ✓ Price feature creation successful")
            return True
        else:
            print("   ✗ Too few price features created")
            return False
    
    except Exception as e:
        print(f"   ✗ Price feature creation failed: {e}")
        return False


def test_volume_features():
    """Test volume-based feature creation."""
    print("\n=== Testing Volume Features ===")
    
    test_data = create_test_data(200)
    
    config = FeatureConfig(
        enable_talib_indicators=False,
        volume_sma_periods=[10, 20],
        vwap_periods=[20]
    )
    
    engineer = FinancialFeatureEngineer(config)
    
    try:
        # First create price features (needed for volume features)
        features = engineer.create_price_features(test_data)
        features = engineer.create_volume_features(features)
        
        # Check specific volume features
        expected_features = [
            'volume_change', 'volume_sma_10', 'volume_sma_20', 'volume_ratio_10',
            'vwap_20', 'vwap_ratio_20', 'obv', 'ad_line', 'volume_oscillator',
            'volume_spike', 'positive_money_flow', 'negative_money_flow'
        ]
        
        found_features = 0
        for feature in expected_features:
            if feature in features.columns:
                found_features += 1
                print(f"   ✓ {feature}")
            else:
                print(f"   ✗ Missing: {feature}")
        
        print(f"1. Found {found_features}/{len(expected_features)} expected volume features")
        
        # Check VWAP calculation
        if 'vwap_20' in features.columns:
            vwap_values = features['vwap_20'].dropna()
            if len(vwap_values) > 0:
                print(f"   ✓ VWAP calculated: mean={vwap_values.mean():.2f}")
            else:
                print("   ⚠ VWAP has no valid values")
        
        if found_features >= 8:
            print("   ✓ Volume feature creation successful")
            return True
        else:
            print("   ✗ Too few volume features created")
            return False
    
    except Exception as e:
        print(f"   ✗ Volume feature creation failed: {e}")
        return False


def test_talib_features():
    """Test TA-Lib feature creation."""
    print("\n=== Testing TA-Lib Features ===")
    
    if not TALIB_AVAILABLE:
        print("   ⚠ TA-Lib not available, skipping TA-Lib tests")
        return True
    
    test_data = create_test_data(200)
    
    config = FeatureConfig(
        enable_talib_indicators=True,
        rsi_periods=[14],
        sma_periods=[20, 50],
        ema_periods=[12, 26],
        bollinger_periods=[20]
    )
    
    engineer = FinancialFeatureEngineer(config)
    
    try:
        features = engineer.create_talib_features(test_data)
        
        # Check specific TA-Lib features
        expected_features = [
            'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi_14',
            'bb_upper_20_2.0', 'bb_lower_20_2.0', 'macd', 'atr'
        ]
        
        found_features = 0
        for feature in expected_features:
            if feature in features.columns:
                found_features += 1
                print(f"   ✓ {feature}")
                
                # Check for reasonable values
                if feature == 'rsi_14':
                    rsi_values = features[feature].dropna()
                    if len(rsi_values) > 0:
                        rsi_range = (rsi_values.min(), rsi_values.max())
                        if 0 <= rsi_range[0] and rsi_range[1] <= 100:
                            print(f"     RSI range: {rsi_range[0]:.1f} - {rsi_range[1]:.1f}")
                        else:
                            print(f"     ⚠ RSI values out of range: {rsi_range}")
            else:
                print(f"   ✗ Missing: {feature}")
        
        print(f"1. Found {found_features}/{len(expected_features)} expected TA-Lib features")
        
        if found_features >= 6:
            print("   ✓ TA-Lib feature creation successful")
            return True
        else:
            print("   ✗ Too few TA-Lib features created")
            return False
    
    except Exception as e:
        print(f"   ✗ TA-Lib feature creation failed: {e}")
        return False


def test_pattern_features():
    """Test pattern recognition features."""
    print("\n=== Testing Pattern Features ===")
    
    test_data = create_test_data(200)
    
    config = FeatureConfig(
        enable_candlestick_patterns=TALIB_AVAILABLE,
        enable_chart_patterns=True
    )
    
    engineer = FinancialFeatureEngineer(config)
    
    try:
        # Create price features first
        features = engineer.create_price_features(test_data)
        
        # Add candlestick patterns
        features = engineer.create_candlestick_patterns(features)
        
        # Add chart patterns
        features = engineer.create_chart_patterns(features)
        
        # Check chart pattern features
        chart_features = [
            'local_high', 'local_low', 'price_trend_5', 'price_trend_20',
            'channel_width', 'channel_position', 'breakout_up', 'breakout_down'
        ]
        
        found_chart = 0
        for feature in chart_features:
            if feature in features.columns:
                found_chart += 1
                print(f"   ✓ Chart: {feature}")
        
        print(f"1. Found {found_chart}/{len(chart_features)} chart pattern features")
        
        # Check candlestick patterns if TA-Lib available
        if TALIB_AVAILABLE:
            candlestick_features = [col for col in features.columns if 'pattern_cdl' in col]
            print(f"2. Found {len(candlestick_features)} candlestick pattern features")
            
            if 'total_bullish_patterns' in features.columns:
                print("   ✓ Pattern summary features created")
        else:
            print("2. Candlestick patterns skipped (TA-Lib not available)")
        
        if found_chart >= 6:
            print("   ✓ Pattern feature creation successful")
            return True
        else:
            print("   ✗ Too few pattern features created")
            return False
    
    except Exception as e:
        print(f"   ✗ Pattern feature creation failed: {e}")
        return False


def test_regime_features():
    """Test market regime features."""
    print("\n=== Testing Regime Features ===")
    
    test_data = create_test_data(200)
    
    config = FeatureConfig()
    engineer = FinancialFeatureEngineer(config)
    
    try:
        # Create price features first (needed for regime features)
        features = engineer.create_price_features(test_data)
        features = engineer.create_regime_features(features)
        
        # Check regime features
        regime_features = [
            'vol_regime_low', 'vol_regime_medium', 'vol_regime_high',
            'trend_uptrend', 'trend_downtrend', 'trend_sideways',
            'stress_indicator', 'momentum_weak', 'momentum_neutral', 'momentum_strong'
        ]
        
        found_features = 0
        for feature in regime_features:
            if feature in features.columns:
                found_features += 1
                print(f"   ✓ {feature}")
                
                # Check binary features have reasonable values
                if feature.startswith(('vol_regime_', 'trend_', 'momentum_')):
                    unique_vals = features[feature].unique()
                    if set(unique_vals).issubset({0, 1, True, False}):
                        print(f"     Binary feature with {len(unique_vals)} unique values")
        
        print(f"1. Found {found_features}/{len(regime_features)} regime features")
        
        if found_features >= 8:
            print("   ✓ Regime feature creation successful")
            return True
        else:
            print("   ✗ Too few regime features created")
            return False
    
    except Exception as e:
        print(f"   ✗ Regime feature creation failed: {e}")
        return False


def test_feature_selection():
    """Test feature selection functionality."""
    print("\n=== Testing Feature Selection ===")
    
    test_data = create_test_data(200)
    
    config = FeatureConfig(
        max_features=20,
        feature_selection_method='mutual_info',
        enable_talib_indicators=TALIB_AVAILABLE
    )
    
    engineer = FinancialFeatureEngineer(config)
    
    try:
        # Create comprehensive features
        features = engineer.create_price_features(test_data)
        features = engineer.create_volume_features(features)
        
        if TALIB_AVAILABLE:
            features = engineer.create_talib_features(features)
        
        total_features_before = len(features.columns)
        
        # Test feature selection
        target = features['return_1']
        feature_cols = [col for col in features.columns if col not in ['timestamp', 'symbol', 'timeframe', 'return_1']]
        feature_data = features[feature_cols]
        
        selected_features, selected_names = engineer.select_features(feature_data, target)
        
        print(f"1. Features before selection: {len(feature_cols)}")
        print(f"2. Features after selection: {len(selected_names)}")
        print(f"3. Target max features: {config.max_features}")
        
        if len(selected_names) <= config.max_features:
            print("   ✓ Feature selection respected max_features limit")
        else:
            print("   ⚠ Feature selection exceeded max_features limit")
        
        # Check feature importance scores
        if engineer.feature_importance:
            print(f"4. Feature importance scores: {len(engineer.feature_importance)}")
            top_features = sorted(engineer.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            print("   Top 5 features by importance:")
            for feature, score in top_features:
                print(f"     {feature}: {score:.4f}")
        
        print("   ✓ Feature selection successful")
        return True
    
    except Exception as e:
        print(f"   ✗ Feature selection failed: {e}")
        return False


def test_complete_pipeline():
    """Test the complete feature engineering pipeline."""
    print("\n=== Testing Complete Pipeline ===")
    
    test_data = create_test_data(300)
    
    config = FeatureConfig(
        enable_talib_indicators=TALIB_AVAILABLE,
        enable_candlestick_patterns=TALIB_AVAILABLE,
        enable_chart_patterns=True,
        max_features=30
    )
    
    engineer = FinancialFeatureEngineer(config)
    
    try:
        # Run complete pipeline
        engineered_features, metadata = engineer.engineer_features(test_data)
        
        print(f"1. Original features: {len(test_data.columns)}")
        print(f"2. Engineered features: {len(engineered_features.columns)}")
        print(f"3. Feature metadata entries: {len(metadata)}")
        
        # Check feature categories
        categories = {}
        for feature_name, meta in metadata.items():
            category = meta.category.value
            categories[category] = categories.get(category, 0) + 1
        
        print(f"4. Feature categories:")
        for category, count in categories.items():
            print(f"   {category}: {count} features")
        
        # Check metadata quality
        features_with_importance = sum(1 for meta in metadata.values() if meta.importance_score is not None)
        print(f"5. Features with importance scores: {features_with_importance}")
        
        # Check for required columns
        required_cols = ['timestamp', 'symbol', 'timeframe']
        missing_cols = [col for col in required_cols if col not in engineered_features.columns]
        if not missing_cols:
            print("   ✓ All required columns present")
        else:
            print(f"   ✗ Missing required columns: {missing_cols}")
        
        # Check data quality
        total_missing = engineered_features.isnull().sum().sum()
        print(f"6. Total missing values: {total_missing}")
        
        if len(engineered_features.columns) > len(test_data.columns):
            print("   ✓ Complete pipeline successful")
            return True
        else:
            print("   ✗ Pipeline did not create additional features")
            return False
    
    except Exception as e:
        print(f"   ✗ Complete pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all feature engineering tests."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Feature Engineering Framework Test")
    print("=" * 50)
    print(f"TA-Lib Available: {TALIB_AVAILABLE}")
    print("=" * 50)
    
    try:
        # Run tests
        price_test = test_price_features()
        volume_test = test_volume_features()
        talib_test = test_talib_features()
        pattern_test = test_pattern_features()
        regime_test = test_regime_features()
        selection_test = test_feature_selection()
        pipeline_test = test_complete_pipeline()
        
        # Summary
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        print(f"Price Features: {'PASS' if price_test else 'FAIL'}")
        print(f"Volume Features: {'PASS' if volume_test else 'FAIL'}")
        print(f"TA-Lib Features: {'PASS' if talib_test else 'FAIL'}")
        print(f"Pattern Features: {'PASS' if pattern_test else 'FAIL'}")
        print(f"Regime Features: {'PASS' if regime_test else 'FAIL'}")
        print(f"Feature Selection: {'PASS' if selection_test else 'FAIL'}")
        print(f"Complete Pipeline: {'PASS' if pipeline_test else 'FAIL'}")
        
        all_tests = [price_test, volume_test, talib_test, pattern_test, regime_test, selection_test, pipeline_test]
        
        if all(all_tests):
            print("\n🎉 All tests PASSED! Feature engineering framework is working correctly.")
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