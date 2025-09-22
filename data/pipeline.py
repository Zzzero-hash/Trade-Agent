"""
Data Pipeline for CNN+LSTM Training

This module provides data loading and preprocessing utilities for training
CNN+LSTM feature extractors with multi-timeframe market data.
Supports both synthetic data and real market data from Yahoo Finance.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler

from data.ingestion.yahoo_finance import YahooFinanceIngestor


logger = logging.getLogger(__name__)


class MarketDataset(Dataset):
    """Dataset for market data with multi-timeframe support."""
    
    def __init__(
        self,
        data_path: str,
        timeframes: List[str] = ["1min", "5min", "15min"],
        sequence_length: int = 100,
        target_columns: List[str] = ["price_prediction"],
        transform: Optional[callable] = None
    ):
        self.data_path = Path(data_path)
        self.timeframes = timeframes
        self.sequence_length = sequence_length
        self.target_columns = target_columns
        self.transform = transform
        
        # Load and preprocess data
        self.data = self._load_data()
        self.scalers = self._fit_scalers()
        self.processed_data = self._preprocess_data()
        
        logger.info(f"Loaded dataset with {len(self)} samples")
    
    def _load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from files."""
        # Placeholder implementation
        # In actual implementation, this would load real market data
        data = {}
        
        for timeframe in self.timeframes:
            # Create dummy data for now
            dates = pd.date_range('2020-01-01', periods=10000, freq='1min')
            dummy_data = pd.DataFrame({
                'timestamp': dates,
                'open': np.random.randn(10000).cumsum() + 100,
                'high': np.random.randn(10000).cumsum() + 102,
                'low': np.random.randn(10000).cumsum() + 98,
                'close': np.random.randn(10000).cumsum() + 100,
                'volume': np.random.exponential(1000, 10000)
            })
            
            # Add target columns with more realistic values
            # Price prediction: future returns with some trend and noise
            returns = dummy_data['close'].pct_change()
            trend = np.sin(np.arange(10000) * 0.01) * 0.002  # Small trend component
            noise = np.random.normal(0, 0.01, 10000)  # Realistic noise level
            dummy_data['price_prediction'] = (returns.shift(-1) + trend + noise).fillna(0)
            
            # Ensure non-zero targets for direction accuracy
            dummy_data['price_prediction'] = np.where(
                np.abs(dummy_data['price_prediction']) < 1e-6,
                np.random.choice([-0.001, 0.001], 10000),
                dummy_data['price_prediction']
            )
            
            dummy_data['volatility_estimation'] = dummy_data['close'].rolling(20).std().fillna(0.01)
            dummy_data['regime_detection'] = np.random.randint(0, 4, 10000)
            
            data[timeframe] = dummy_data.dropna()
        
        return data
    
    def _fit_scalers(self) -> Dict[str, StandardScaler]:
        """Fit scalers for data normalization."""
        scalers = {}
        
        for timeframe in self.timeframes:
            scaler = RobustScaler()
            feature_columns = ['open', 'high', 'low', 'close', 'volume']
            scaler.fit(self.data[timeframe][feature_columns])
            scalers[timeframe] = scaler
        
        return scalers
    
    def _preprocess_data(self) -> Dict[str, np.ndarray]:
        """Preprocess and normalize data."""
        processed = {}
        
        for timeframe in self.timeframes:
            df = self.data[timeframe].copy()
            
            # Scale features
            feature_columns = ['open', 'high', 'low', 'close', 'volume']
            scaled_features = self.scalers[timeframe].transform(df[feature_columns])
            
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
    
    def __len__(self) -> int:
        """Return dataset length."""
        # Use the shortest timeframe as reference
        min_length = min(len(self.processed_data[tf]['sequences']) for tf in self.timeframes)
        return min_length
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get a single sample."""
        data_dict = {}
        
        # Get multi-timeframe data
        for timeframe in self.timeframes:
            sequence = self.processed_data[timeframe]['sequences'][idx]
            data_dict[timeframe] = torch.FloatTensor(sequence).transpose(0, 1)  # (features, time)
        
        # Add sequence data for LSTM
        # Use the first timeframe as the main sequence
        main_sequence = self.processed_data[self.timeframes[0]]['sequences'][idx]
        data_dict['sequence_data'] = torch.FloatTensor(main_sequence)  # (time, features)
        
        # Get targets
        targets = self.processed_data[self.timeframes[0]]['targets'][idx]
        
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
        
        # Apply transforms if provided
        if self.transform:
            data_dict, targets = self.transform(data_dict, targets)
        
        return data_dict, targets


def create_data_loaders(
    data_path: str = "data/processed",
    timeframes: List[str] = ["1min", "5min", "15min"],
    sequence_length: int = 100,
    target_columns: List[str] = ["price_prediction"],
    batch_size: int = 32,
    validation_split: float = 0.2,
    test_split: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_path: Path to processed data
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
    # Create dataset
    dataset = MarketDataset(
        data_path=data_path,
        timeframes=timeframes,
        sequence_length=sequence_length,
        target_columns=target_columns
    )
    
    # Calculate split sizes
    total_size = len(dataset)
    test_size = int(test_split * total_size)
    val_size = int(validation_split * total_size)
    train_size = total_size - val_size - test_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    logger.info(f"Created data loaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
    
    return train_loader, val_loader, test_loader


# Data augmentation transforms
class NoiseInjection:
    """Add Gaussian noise to input data."""
    
    def __init__(self, noise_std: float = 0.01):
        self.noise_std = noise_std
    
    def __call__(self, data: Dict[str, torch.Tensor], targets: torch.Tensor):
        """Apply noise injection."""
        augmented_data = {}
        
        for key, tensor in data.items():
            if key != 'sequence_data':  # Don't augment sequence data
                noise = torch.randn_like(tensor) * self.noise_std
                augmented_data[key] = tensor + noise
            else:
                augmented_data[key] = tensor
        
        return augmented_data, targets


class TemporalJittering:
    """Apply temporal jittering to sequences."""
    
    def __init__(self, jitter_prob: float = 0.1, max_shift: int = 2):
        self.jitter_prob = jitter_prob
        self.max_shift = max_shift
    
    def __call__(self, data: Dict[str, torch.Tensor], targets: torch.Tensor):
        """Apply temporal jittering."""
        if torch.rand(1).item() < self.jitter_prob:
            shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
            
            augmented_data = {}
            for key, tensor in data.items():
                if tensor.dim() >= 2:  # Has time dimension
                    augmented_data[key] = torch.roll(tensor, shift, dims=-1)
                else:
                    augmented_data[key] = tensor
            
            return augmented_data, targets
        
        return data, targets


class PriceScaling:
    """Apply random price scaling."""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.95, 1.05)):
        self.scale_range = scale_range
    
    def __call__(self, data: Dict[str, torch.Tensor], targets: torch.Tensor):
        """Apply price scaling."""
        scale_factor = torch.rand(1).item() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        
        augmented_data = {}
        for key, tensor in data.items():
            # Apply scaling to price-related features (first 4 features: OHLC)
            if tensor.dim() >= 2:
                scaled_tensor = tensor.clone()
                scaled_tensor[:4] *= scale_factor  # Scale OHLC
                augmented_data[key] = scaled_tensor
            else:
                augmented_data[key] = tensor
        
        return augmented_data, targets


def create_augmentation_transform(
    noise_std: float = 0.01,
    jitter_prob: float = 0.1,
    scale_range: Tuple[float, float] = (0.95, 1.05)
):
    """Create composed data augmentation transform."""
    
    class ComposedTransform:
        def __init__(self):
            self.transforms = [
                NoiseInjection(noise_std),
                TemporalJittering(jitter_prob),
                PriceScaling(scale_range)
            ]
        
        def __call__(self, data, targets):
            for transform in self.transforms:
                data, targets = transform(data, targets)
            return data, targets
    
    return ComposedTransform()


if __name__ == "__main__":
    # Test data pipeline
    train_loader, val_loader, test_loader = create_data_loaders(
        data_path="data/processed",
        batch_size=32,
        validation_split=0.2,
        test_split=0.1
    )
    
    # Test a batch
    for batch_idx, (data, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"Data keys: {data.keys()}")
        for key, tensor in data.items():
            print(f"  {key}: {tensor.shape}")
        
        if isinstance(targets, dict):
            print(f"Target keys: {targets.keys()}")
            for key, tensor in targets.items():
                print(f"  {key}: {tensor.shape}")
        else:
            print(f"Targets: {targets.shape}")
        
        break
