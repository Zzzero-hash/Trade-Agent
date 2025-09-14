"""Machine learning models and training pipelines"""

from .base_models import (
    BaseMLModel, BasePyTorchModel, BaseRLAgent, ModelConfig, TrainingResult
)
from .cnn_model import (
    CNNFeatureExtractor, MultiHeadAttention, create_cnn_config,
    create_cnn_data_loader
)
from .lstm_model import (
    LSTMTemporalProcessor, LSTMAttention, create_lstm_config,
    create_lstm_data_loader, create_sequence_data
)
from .feature_engineering import FeatureEngineer
from .trading_environment import TradingEnvironment, TradingConfig
from .rl_agents import (
    RLAgentConfig, StableBaselinesRLAgent, RLAgentFactory,
    RLAgentEnsemble, TradingCallback, create_rl_ensemble
)
from .rl_hyperopt import (
    HyperparameterOptimizer, MultiAgentHyperparameterOptimizer,
    optimize_agent_hyperparameters, optimize_ensemble_hyperparameters
)

__all__ = [
    # Base classes
    'BaseMLModel',
    'BasePyTorchModel',
    'BaseRLAgent',
    'ModelConfig',
    'TrainingResult',

    # CNN Model
    'CNNFeatureExtractor',
    'MultiHeadAttention',
    'create_cnn_config',
    'create_cnn_data_loader',

    # LSTM Model
    'LSTMTemporalProcessor',
    'LSTMAttention',
    'create_lstm_config',
    'create_lstm_data_loader',
    'create_sequence_data',

    # Feature Engineering
    'FeatureEngineer',

    # Trading Environment
    'TradingEnvironment',
    'TradingConfig',

    # RL Agents
    'RLAgentConfig',
    'StableBaselinesRLAgent',
    'RLAgentFactory',
    'RLAgentEnsemble',
    'TradingCallback',
    'create_rl_ensemble',

    # Hyperparameter Optimization
    'HyperparameterOptimizer',
    'MultiAgentHyperparameterOptimizer',
    'optimize_agent_hyperparameters',
    'optimize_ensemble_hyperparameters'
]