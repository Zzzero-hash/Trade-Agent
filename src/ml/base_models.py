"""Abstract base classes for ML models"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for ML models"""
    model_type: str
    input_dim: int
    output_dim: int
    hidden_dims: List[int]
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    device: str = "cpu"
    random_seed: int = 42


@dataclass
class TrainingResult:
    """Training result metrics"""
    train_loss: float
    val_loss: float
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    epochs_trained: int = 0
    best_epoch: int = 0


class BaseMLModel(ABC):
    """Abstract base class for ML models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.is_trained = False
        self.training_history = []
    
    @abstractmethod
    def build_model(self) -> None:
        """Build the model architecture"""
        pass
    
    def fit(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> TrainingResult:
        """Train the model - default implementation"""
        raise NotImplementedError("Subclasses should implement fit method")
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """Save model to file"""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """Load model from file"""
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(X)
        
        # Calculate basic metrics
        mse = np.mean((predictions - y) ** 2)
        mae = np.mean(np.abs(predictions - y))
        
        return {
            "mse": mse,
            "mae": mae,
            "rmse": np.sqrt(mse)
        }


class BasePyTorchModel(BaseMLModel, nn.Module):
    """Base class for PyTorch models"""
    
    def __init__(self, config: ModelConfig):
        BaseMLModel.__init__(self, config)
        nn.Module.__init__(self)
        
        # Set device
        self.device = torch.device(config.device)
        
        # Set random seed for reproducibility
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.random_seed)
    
    def train_model(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> TrainingResult:
        """Train PyTorch model"""
        if self.model is None:
            self.build_model()
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(self.config.epochs):
            # Training phase
            super().train(True)  # Use PyTorch's train method
            optimizer.zero_grad()
            
            outputs = self.forward(X_train_tensor)
            train_loss = criterion(outputs, y_train_tensor)
            train_loss.backward()
            optimizer.step()
            
            # Validation phase
            val_loss = 0.0
            if X_val is not None:
                self.eval()
                with torch.no_grad():
                    val_outputs = self.forward(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
        
        self.is_trained = True
        
        return TrainingResult(
            train_loss=train_loss.item(),
            val_loss=val_loss,
            epochs_trained=self.config.epochs,
            best_epoch=best_epoch
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with PyTorch model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.forward(X_tensor)
            return predictions.cpu().numpy()
    
    def save_model(self, filepath: str) -> None:
        """Save PyTorch model"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'is_trained': self.is_trained
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load PyTorch model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
        self.is_trained = checkpoint['is_trained']


class BaseRLAgent(ABC):
    """Abstract base class for Reinforcement Learning agents"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_trained = False
        self.policy = None
    
    @abstractmethod
    def train(self, env, total_timesteps: int) -> Dict[str, Any]:
        """Train the RL agent"""
        pass
    
    @abstractmethod
    def predict(self, observation: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict action given observation"""
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """Save RL model"""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """Load RL model"""
        pass
    
    def evaluate(self, env, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate agent performance"""
        if not self.is_trained:
            raise ValueError("Agent must be trained before evaluation")
        
        episode_rewards = []
        
        for _ in range(n_episodes):
            obs, _ = env.reset()  # Handle new gymnasium API
            episode_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                action, _ = self.predict(obs)
                obs, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
        
        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards)
        }