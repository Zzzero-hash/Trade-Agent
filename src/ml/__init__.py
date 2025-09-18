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
# Interpretability modules
from .shap_explainer import SHAPExplainer, create_shap_explainer
from .attention_visualizer import (
    AttentionVisualizer, create_attention_visualizer
)
from .feature_importance_analyzer import (
    FeatureImportanceAnalyzer, create_feature_importance_analyzer
)
from .decision_auditor import DecisionAuditor, create_decision_auditor
from .uncertainty_calibrator import (
    UncertaintyCalibrator, create_uncertainty_calibrator
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
    'optimize_ensemble_hyperparameters',
    
    # Interpretability modules
    'SHAPExplainer',
    'create_shap_explainer',
    'AttentionVisualizer',
    'create_attention_visualizer',
    'FeatureImportanceAnalyzer',
    'create_feature_importance_analyzer',
    'DecisionAuditor',
    'create_decision_auditor',
    'UncertaintyCalibrator',
    'create_uncertainty_calibrator'
]