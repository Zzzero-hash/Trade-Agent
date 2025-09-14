"""
Data caching for efficient market data access.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class MarketDataCache:
    """Efficient caching for market data access patterns."""
    
    def __init__(self, data: pd.DataFrame, symbols: List[str]):
        self.data = data
        self.symbols = symbols
        self._price_cache: Dict[int, Dict[str, float]] = {}
        self._feature_cache: Dict[int, np.ndarray] = {}
        self._build_indices()
    
    def _build_indices(self) -> None:
        """Build efficient indices for data access."""
        # Group data by timestamp for faster access
        self.data_by_step = {}
        for idx, row in self.data.iterrows():
            step = idx
            if step not in self.data_by_step:
                self.data_by_step[step] = {}
            self.data_by_step[step][row['symbol']] = row
    
    def get_prices_at_step(self, step: int) -> Dict[str, float]:
        """Get prices for all symbols at a specific step."""
        if step in self._price_cache:
            return self._price_cache[step]
        
        prices = {}
        step_data = self.data_by_step.get(step, {})
        
        for symbol in self.symbols:
            if symbol in step_data:
                prices[symbol] = step_data[symbol]['close']
            else:
                # Find last known price
                for prev_step in range(step - 1, -1, -1):
                    prev_data = self.data_by_step.get(prev_step, {})
                    if symbol in prev_data:
                        prices[symbol] = prev_data[symbol]['close']
                        break
                else:
                    prices[symbol] = 100.0  # Default price
        
        self._price_cache[step] = prices
        return prices
    
    def get_features_window(
        self, 
        end_step: int, 
        window_size: int,
        feature_columns: List[str]
    ) -> np.ndarray:
        """Get feature window for observation."""
        cache_key = (end_step, window_size)
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        start_step = max(0, end_step - window_size)
        
        features = []
        for symbol in self.symbols:
            symbol_features = []
            
            for step in range(start_step, end_step):
                step_data = self.data_by_step.get(step, {})
                if symbol in step_data:
                    row_features = [
                        step_data[symbol].get(col, 0.0) 
                        for col in feature_columns
                    ]
                else:
                    row_features = [0.0] * len(feature_columns)
                
                symbol_features.extend(row_features)
            
            # Pad if necessary
            expected_length = window_size * len(feature_columns)
            while len(symbol_features) < expected_length:
                symbol_features = [0.0] * len(feature_columns) + symbol_features
            
            features.extend(symbol_features)
        
        result = np.array(features, dtype=np.float32)
        self._feature_cache[cache_key] = result
        return result
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self._price_cache.clear()
        self._feature_cache.clear()