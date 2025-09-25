"""
Common Imports Module

Centralized imports to eliminate duplicate import patterns across modules.
"""

# Standard library imports
import sys
import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass

# Scientific computing
import numpy as np
import pandas as pd

# Machine learning
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Reinforcement learning
try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.evaluation import evaluate_policy
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

# Financial data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


def setup_path(additional_paths: List[str] = None):
    """Setup Python path for imports"""
    if additional_paths is None:
        additional_paths = ['src']
    
    for path in additional_paths:
        if path not in sys.path:
            sys.path.append(path)


def setup_logging(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Setup standardized logging configuration"""
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)
    
    return logger


def check_dependencies():
    """Check availability of optional dependencies"""
    return {
        'torch': TORCH_AVAILABLE,
        'reinforcement_learning': RL_AVAILABLE,
        'yfinance': YFINANCE_AVAILABLE
    }