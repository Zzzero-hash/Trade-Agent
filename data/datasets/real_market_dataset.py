"""
Real Market Dataset for CNN+LSTM Training

This module provides a dataset implementation that works with real market data
from Yahoo Finance, supporting multi-timeframe analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, RobustScaler

logger = logging.getLogger(__name__)


class RealMarketDataset(Dataset):
    """Dataset for real market data with multi-timeframe support."""
    
    def __init__(
        self,
        symbol: str,
        data_dir: str = "data/processed",
        timeframes: List[str] = ["1min", "5min", "15min"],
        sequence_length: int = 100,
        target_columns: List[str] = ["price_prediction"],
        transform: Optional[callable] = None,
        cache_processed: bool = True
    ):
        """
        Initialize real market dataset.
        
        Args:
            symbol: Stock symbol to load data for
            data_dir: Directory containing processed data
            timeframes: List of timeframes to include
            sequence_length: Length of input sequences
            target_columns: Target columns for prediction
            transform: Data augmentation transform
            cache_processed: Whether to cache processed data
        """
        self.symbol = symbol
        self.data_dir = Path(data_dir)
        self.timeframes = timeframes
        self.sequence_length = sequence_length
        self.target_columns = target_columns
        self.transform = transform
        self.cache_processed = cache_processed
        
        # Load and preprocess data
        self.data = self._load_data()
        if not self.data:
            raise ValueError(f"No data found for symbol {symbol}")
        
        self.scalers = self._fit_scalers()
        self.processed_data = self._preprocess_data()
        
        logger.info(f"Loaded real market dataset for {symbol} with {len(self)} samples")
    
    def _load_data(self) -> Dict[str, pd.DataFrame]:
        """Load real market data from processed files."""
        data = {}
        symbol_dir = self.data_dir / self.symbol
        
        if not symbol_dir.exists():
            logger.warning(f"Data directory {symbol_dir} does not exist")
            return data
        
        for timeframe in self.timeframes:
            file_path = symbol_dir / f"{timeframe}.parquet"
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    if not df.empty:
                        # Ensure required columns exist
                        required_cols = ['open', 'high', 'low', 'close', 'volume']
                        for col in required_cols:
                            if col not in df.columns:
                                logger.warning(f"Missing column {col} in {timeframe} data")
                                df[col] = 0.0
                        
                        data[timeframe] = df
                        logger.info(f"Loaded {timeframe} data: {len(df)} rows")
                    else:
                        logger.warning(f"Empty data for {timeframe}")
                except Exception as e:
                    logger.error(f"Error loading {timeframe} data: {e}")
            else:
                logger.warning(f"File not found: {file_path}")
        
        return data
    
    def _fit_scalers(self) -> Dict[str, RobustScaler]:
        """Fit scalers for data normalization."""
        scalers = {}
        
        for timeframe in self.timeframes:
            if timeframe in self.data:
                scaler = RobustScaler()
                feature_columns = ['open', 'high', 'low', 'close', 'volume']
                scaler.fit(self.data[timeframe][feature_columns])
                scalers[timeframe] = scaler
        
        return scalers
    
    def _preprocess_data(self) -> Dict[str, Dict]:
        """Preprocess and normalize data."""
        processed = {}
        
        for timeframe in self.timeframes:
            if timeframe not in self.data:
                continue
                
            df = self.data[timeframe].copy()
            
            # Scale features
            feature_columns = ['open', 'high', 'low', 'close', 'volume']
            scaled_features = self.scalers[timeframe].transform(df[feature_columns])
            
            # Add target columns
            df = self._add_targets(df)
            
            # Convert to sequences
            sequences = []
            targets = []
            
            for i in range(len(scaled_features) - self.sequence_length):
                seq = scaled_features[i:i + self.sequence_length]
                target = df[self.target_columns].iloc[i + self.sequence_length].values
                
                sequences.append(seq)
                targets.append(target)
            
            processed[timeframe] = {
                'sequences': np.array(sequences),
                'targets': np.array(targets)
            }
        
        return processed
    
    def _add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add target columns to dataframe."""
        df = df.copy()
        
        # Price prediction target: next period return
        returns = df['close'].pct_change()
        df['price_prediction'] = returns.shift(-1).fillna(0)
        
        # Volatility estimation target: rolling standard deviation
        df['volatility_estimation'] = df['close'].rolling(20).std().fillna(0.01)
        
        # Regime detection target: volatility quantiles (simplified)
        volatility = df['volatility_estimation']
        df['regime_detection'] = pd.qcut(volatility, 4, labels=False, duplicates='drop')
        df['regime_detection'] = df['regime_detection'].fillna(0).astype(int)
        
        return df
    
    def __len__(self) -> int:
        """Return dataset length."""
        if not self.processed_data:
            return 0
            
        # Use the shortest timeframe as reference
        lengths = [len(self.processed_data[tf]['sequences']) for tf in self.timeframes 
                  if tf in self.processed_data]
        if not lengths:
            return 0
            
        return min(lengths)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get a single sample."""
        if not self.processed_data:
            raise IndexError("No processed data available")
        
        data_dict = {}
        
        # Get multi-timeframe data
        for timeframe in self.timeframes:
            if timeframe in self.processed_data:
                sequence = self.processed_data[timeframe]['sequences'][idx]
                data_dict[timeframe] = torch.FloatTensor(sequence).transpose(0, 1)  # (features, time)
        
        # Add sequence data for LSTM
        # Use the first timeframe as the main sequence
        first_timeframe = self.timeframes[0]
        if first_timeframe in self.processed_data:
            main_sequence = self.processed_data[first_timeframe]['sequences'][idx]
            data_dict['sequence_data'] = torch.FloatTensor(main_sequence)  # (time, features)
        
        # Get targets
        if first_timeframe in self.processed_data:
            targets = self.processed_data[first_timeframe]['targets'][idx]
            
            # Convert to multi-task targets if needed
            if len(self.target_columns) > 1:
                target_dict = {}
                for i, col in enumerate(self.target_columns):
                    if col == 'regime_detection':
                        target_dict[col] = torch.LongTensor([int(targets[i])])
                    else:
                        target_dict[col] = torch.FloatTensor([targets[i]])
                targets = target_dict
            else:
                targets = torch.FloatTensor([targets[0]])
        else:
            targets = torch.FloatTensor([0.0])
        
        # Apply transforms if provided
        if self.transform:
            data_dict, targets = self.transform(data_dict, targets)
        
        return data_dict, targets


def create_real_data_loaders(
    symbol: str,
    data_dir: str = "data/processed",
    timeframes: List[str] = ["1min", "5min", "15min"],
    sequence_length: int = 100,
    target_columns: List[str] = ["price_prediction"],
    batch_size: int = 32,
    validation_split: float = 0.2,
    test_split: float = 0.1,
    num_workers: int = 0,  # Set to 0 for Windows compatibility
    pin_memory: bool = True,
    shuffle: bool = True
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test data loaders for real market data.
    
    Args:
        symbol: Stock symbol to load data for
        data_dir: Directory containing processed data
        timeframes: List of timeframes to include
        sequence_length: Length of input sequences
        target_columns: Target columns for prediction
        batch_size: Batch size for data loaders
        validation_split: Fraction of data for validation
        test_split: Fraction of data for testing
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        shuffle: Whether to shuffle training data
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    try:
        # Create dataset
        dataset = RealMarketDataset(
            symbol=symbol,
            data_dir=data_dir,
            timeframes=timeframes,
            sequence_length=sequence_length,
            target_columns=target_columns
        )
        
        if len(dataset) == 0:
            raise ValueError(f"Dataset for {symbol} is empty")
        
        # Calculate split sizes
        total_size = len(dataset)
        test_size = int(test_split * total_size)
        val_size = int(validation_split * total_size)
        train_size = total_size - val_size - test_size
        
        if train_size <= 0:
            raise ValueError(f"Not enough data for training: {total_size} samples")
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        logger.info(f"Created real data loaders for {symbol}: "
                   f"train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        logger.error(f"Error creating data loaders for {symbol}: {e}")
        raise


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # This would be used after data has been downloaded and processed
    # For now, we'll just show the interface
    print("RealMarketDataset module ready for use")
    print("To use with real data, first download with YahooFinanceIngestor")
