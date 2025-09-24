"""Device Optimization Utility for ML Models

This module provides intelligent device selection based on model architecture:
- CNN models: Use GPU for optimal performance
- MLP/ActorCritic models: Use CPU for better efficiency
- Hybrid models: Use GPU if CNN components are present
"""

import logging
import torch
from typing import Dict, Any, Optional, Union
import warnings

logger = logging.getLogger(__name__)


class DeviceOptimizer:
    """Intelligent device selection for different model architectures"""
    
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.device_info = self._get_device_info()
        
    def _get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information"""
        info = {
            'cuda_available': self.cuda_available,
            'cpu_count': torch.get_num_threads(),
        }
        
        if self.cuda_available:
            info.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
                'gpu_count': torch.cuda.device_count(),
                'cuda_version': torch.version.cuda,
            })
            
        return info
    
    def get_optimal_device(
        self, 
        model_type: str, 
        policy_type: Optional[str] = None,
        force_device: Optional[str] = None
    ) -> str:
        """Get optimal device based on model architecture
        
        Args:
            model_type: Type of model ('cnn', 'lstm', 'hybrid', 'rl', 'mlp')
            policy_type: RL policy type ('CnnPolicy', 'MlpPolicy', 'ActorCriticPolicy')
            force_device: Force specific device ('cpu', 'cuda', 'auto')
            
        Returns:
            Optimal device string ('cpu' or 'cuda')
        """
        if force_device and force_device != 'auto':
            if force_device == 'cuda' and not self.cuda_available:
                logger.warning("CUDA requested but not available, falling back to CPU")
                return 'cpu'
            return force_device
        
        # Device selection logic based on model architecture
        device = self._select_device_by_architecture(model_type, policy_type)
        
        logger.info(f"Selected device '{device}' for {model_type} model" + 
                   (f" with {policy_type} policy" if policy_type else ""))
        
        return device
    
    def _select_device_by_architecture(self, model_type: str, policy_type: Optional[str]) -> str:
        """Select device based on model architecture"""
        if not self.cuda_available:
            return 'cpu'
        
        model_type = model_type.lower()
        
        # CNN models benefit significantly from GPU
        if 'cnn' in model_type:
            return 'cuda'
        
        # Hybrid models with CNN components should use GPU
        if 'hybrid' in model_type:
            return 'cuda'
        
        # LSTM models can benefit from GPU for large sequences
        if 'lstm' in model_type:
            return 'cuda'
        
        # RL policy-specific optimization
        if policy_type:
            policy_type = policy_type.lower()
            
            # CNN policies should use GPU
            if 'cnn' in policy_type:
                return 'cuda'
            
            # MLP/ActorCritic policies are more efficient on CPU
            if any(p in policy_type for p in ['mlp', 'actorcritic']):
                logger.info("Using CPU for MLP/ActorCritic policy (more efficient than GPU)")
                return 'cpu'
        
        # MLP models are generally more efficient on CPU
        if 'mlp' in model_type:
            return 'cpu'
        
        # Default to CPU for unknown architectures
        return 'cpu'
    
    def optimize_torch_settings(self, device: str) -> None:
        """Optimize PyTorch settings for the selected device"""
        if device == 'cuda' and self.cuda_available:
            # GPU optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Set memory fraction to avoid OOM
            if hasattr(torch.cuda, 'set_memory_fraction'):
                torch.cuda.set_memory_fraction(0.9)
                
            logger.info("Applied CUDA optimizations")
            
        else:
            # CPU optimizations
            torch.set_num_threads(min(torch.get_num_threads(), 8))  # Limit threads
            torch.set_num_interop_threads(2)
            
            # Disable CUDA if explicitly using CPU
            if device == 'cpu':
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                
            logger.info("Applied CPU optimizations")
    
    def get_model_config_with_device(
        self, 
        base_config: Dict[str, Any], 
        model_type: str,
        policy_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get model configuration with optimal device selection
        
        Args:
            base_config: Base model configuration
            model_type: Type of model
            policy_type: RL policy type (optional)
            
        Returns:
            Configuration with optimal device
        """
        config = base_config.copy()
        optimal_device = self.get_optimal_device(model_type, policy_type)
        config['device'] = optimal_device
        
        # Apply device-specific optimizations
        self.optimize_torch_settings(optimal_device)
        
        return config
    
    def log_device_info(self) -> None:
        """Log comprehensive device information"""
        logger.info("=== Device Information ===")
        logger.info(f"CUDA Available: {self.device_info['cuda_available']}")
        logger.info(f"CPU Threads: {self.device_info['cpu_count']}")
        
        if self.cuda_available:
            logger.info(f"GPU: {self.device_info['gpu_name']}")
            logger.info(f"GPU Memory: {self.device_info['gpu_memory_gb']:.1f} GB")
            logger.info(f"GPU Count: {self.device_info['gpu_count']}")
            logger.info(f"CUDA Version: {self.device_info['cuda_version']}")
        
        logger.info("=== Optimization Recommendations ===")
        logger.info("• CNN models: GPU recommended")
        logger.info("• MLP/ActorCritic policies: CPU recommended")
        logger.info("• Hybrid CNN+LSTM: GPU recommended")
        logger.info("• Pure LSTM: GPU for large sequences, CPU for small")


def suppress_sb3_device_warnings():
    """Suppress Stable Baselines3 device warnings for optimal configurations"""
    warnings.filterwarnings(
        'ignore', 
        message='.*primarily intended to run on the CPU when not using a CNN policy.*',
        category=UserWarning
    )
    # Suppress SB3 v1.8.0+ net_arch warning (we've already fixed the format)
    warnings.filterwarnings(
        'ignore',
        message='.*shared layers in the mlp_extractor are removed since SB3 v1.8.0.*',
        category=UserWarning
    )


# Global device optimizer instance
device_optimizer = DeviceOptimizer()