"""
Market feature calculation for trading environments.
"""
import numpy as np
import pandas as pd
from typing import List


class MarketFeatureCalculator:
    """Calculates technical indicators and market features."""
    
    @staticmethod
    def calculate_features(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for market data."""
        features_data = []
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            symbol_data = MarketFeatureCalculator._add_price_features(
                symbol_data
            )
            symbol_data = MarketFeatureCalculator._add_technical_indicators(
                symbol_data
            )
            symbol_data = MarketFeatureCalculator._add_volume_indicators(
                symbol_data
            )
            features_data.append(symbol_data)
        
        return pd.concat(features_data, ignore_index=True)
    
    @staticmethod
    def _add_price_features(data: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(
            data['close'] / data['close'].shift(1)
        )
        
        # Moving averages
        data['sma_5'] = data['close'].rolling(5).mean()
        data['sma_20'] = data['close'].rolling(20).mean()
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_26'] = data['close'].ewm(span=26).mean()
        
        # Volatility
        data['volatility'] = data['returns'].rolling(20).std()
        
        return data
    
    @staticmethod
    def _add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators."""
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        data['bb_middle'] = data['close'].rolling(bb_period).mean()
        bb_std_dev = data['close'].rolling(bb_period).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std_dev * bb_std)
        data['bb_lower'] = data['bb_middle'] - (bb_std_dev * bb_std)
        data['bb_position'] = (
            (data['close'] - data['bb_lower']) / 
            (data['bb_upper'] - data['bb_lower'])
        )
        
        return data
    
    @staticmethod
    def _add_volume_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        data['volume_sma'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        return data
    
    @staticmethod
    def get_feature_columns() -> List[str]:
        """Get list of feature column names."""
        return [
            'open', 'high', 'low', 'close', 'volume',
            'returns', 'volatility', 'rsi', 'macd', 'macd_signal',
            'bb_position', 'volume_ratio', 'sma_5', 'sma_20', 'ema_12'
        ]