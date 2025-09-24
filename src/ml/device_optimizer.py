"""Device Optimization for ML Models

This module provides intelligent device selection based on model architecture
and policy types to optimize performance and resource usage.

CNN models benefit from GPU acceleration due to parallel convolution operations,
while simple MLP policies can run efficiently on CPU without GPU overhead.
"""

import torch
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Model architecture types for device optimization"""
    CNN = "cnn"
    LSTM = "lstm" 
    CNN_LSTM_HYBRID = "cnn_lstm_hybrid"
    MLP = "mlp"
    ACTOR_CRITIC_MLP = "actor_critic_mlp"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"


class PolicyType(Enum):
    """RL Policy types for device optimization"""
    CNN_POLICY = "CnnPolicy"
    MLP_POLICY = "MlpPolicy" 
    ACTOR_CRITIC_POLICY = "ActorCriticPolicy"
    MULTI_INPUT_POLICY = "MultiInputPolicy"


@dataclass
class DeviceInfo:
    """Device information and capabilities"""
    cuda_available: bool
    cuda_device_count: int
    cuda_memory_gb: float
    cpu_cores: int
    device_name: str
    compute_capability: Optional[tuple] = None


@dataclass
class ModelComplexity:
    """Model complexity metrics for device selection"""
    parameter_count: int
    has_convolutions: bool
    has_attention: bool
    sequence_length: int
    batch_size: int
    estimated_memory_mb: float


class DeviceOptimizer:
    """Intelligent device selection for ML models"""
    
    def __init__(self):
        self.device_info = self._get_device_info()
        self.gpu_memory_threshold_gb = 2.0  # Minimum GPU memory for CNN models
        self.cpu_preference_threshold = 1e6  # Parameter count threshold for CPU preference
        
    def _get_device_info(self) -> DeviceInfo:
        """Get comprehensive device information"""
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0
        
        cuda_memory_gb = 0.0
        device_name = "CPU"
        compute_capability = None
        
        if cuda_available and cuda_device_count > 0:
            # Get GPU memory in GB
            cuda_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            device_name = torch.cuda.get_device_name(0)
            
            # Get compute capability
            props = torch.cuda.get_device_properties(0)
            compute_capability = (props.major, props.minor)
        
        # Get CPU core count
        try:
            import os
            cpu_cores = os.cpu_count() or 4
        except:
            cpu_cores = 4
            
        return DeviceInfo(
            cuda_available=cuda_available,
            cuda_device_count=cuda_device_count,
            cuda_memory_gb=cuda_memory_gb,
            cpu_cores=cpu_cores,
            device_name=device_name,
            compute_capability=compute_capability
        )
    
    def estimate_model_complexity(
        self,
        model: Optional[torch.nn.Module] = None,
        model_type: Optional[ModelType] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> ModelComplexity:
        """Estimate model complexity for device selection"""
        
        if model is not None:
            # Count parameters from actual model
            param_count = sum(p.numel() for p in model.parameters())
            
            # Detect convolutions
            has_convolutions = any(
                isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d))
                for module in model.modules()
            )
            
            # Detect attention mechanisms
            has_attention = any(
                'attention' in str(type(module)).lower() or 'multihead' in str(type(module)).lower()
                for module in model.modules()
            )
            
        else:
            # Estimate from config
            param_count = self._estimate_parameters_from_config(config or {}, model_type)
            has_convolutions = model_type in [ModelType.CNN, ModelType.CNN_LSTM_HYBRID]
            has_attention = config.get('use_attention', False) if config else False
        
        # Get sequence and batch info from config
        sequence_length = config.get('sequence_length', 60) if config else 60
        batch_size = config.get('batch_size', 32) if config else 32
        
        # Estimate memory usage (rough approximation)
        estimated_memory_mb = self._estimate_memory_usage(
            param_count, sequence_length, batch_size, has_convolutions
        )
        
        return ModelComplexity(
            parameter_count=param_count,
            has_convolutions=has_convolutions,
            has_attention=has_attention,
            sequence_length=sequence_length,
            batch_size=batch_size,
            estimated_memory_mb=estimated_memory_mb
        )
    
    def _estimate_parameters_from_config(
        self, 
        config: Dict[str, Any], 
        model_type: Optional[ModelType]
    ) -> int:
        """Estimate parameter count from configuration"""
        
        if model_type == ModelType.CNN:
            # CNN parameter estimation
            input_dim = config.get('input_dim', 100)
            num_filters = config.get('num_filters', 64)
            filter_sizes = config.get('filter_sizes', [3, 5, 7, 11])
            output_dim = config.get('output_dim', 256)
            
            # Rough CNN parameter count
            conv_params = sum(input_dim * num_filters * fs for fs in filter_sizes)
            fc_params = len(filter_sizes) * num_filters * output_dim
            return conv_params + fc_params
            
        elif model_type == ModelType.LSTM:
            # LSTM parameter estimation
            input_dim = config.get('input_dim', 100)
            hidden_dim = config.get('hidden_dim', 128)
            num_layers = config.get('num_layers', 3)
            bidirectional = config.get('bidirectional', True)
            
            # LSTM parameters: 4 * (input_size + hidden_size + 1) * hidden_size per layer
            params_per_layer = 4 * (input_dim + hidden_dim + 1) * hidden_dim
            if bidirectional:
                params_per_layer *= 2
            
            return params_per_layer * num_layers
            
        elif model_type == ModelType.CNN_LSTM_HYBRID:
            # Combine CNN and LSTM estimates
            cnn_params = self._estimate_parameters_from_config(config, ModelType.CNN)
            lstm_params = self._estimate_parameters_from_config(config, ModelType.LSTM)
            fusion_params = config.get('feature_fusion_dim', 256) ** 2
            
            return cnn_params + lstm_params + fusion_params
            
        elif model_type == ModelType.MLP:
            # Simple MLP estimation
            input_dim = config.get('input_dim', 100)
            hidden_dims = config.get('hidden_dims', [256, 128])
            output_dim = config.get('output_dim', 10)
            
            params = input_dim * hidden_dims[0]
            for i in range(1, len(hidden_dims)):
                params += hidden_dims[i-1] * hidden_dims[i]
            params += hidden_dims[-1] * output_dim
            
            return params
            
        else:
            # Default estimation
            return config.get('estimated_parameters', 1000000)
    
    def _estimate_memory_usage(
        self,
        param_count: int,
        sequence_length: int,
        batch_size: int,
        has_convolutions: bool
    ) -> float:
        """Estimate memory usage in MB"""
        
        # Base memory for parameters (4 bytes per float32 parameter)
        param_memory_mb = param_count * 4 / (1024 * 1024)
        
        # Activation memory (rough estimate)
        if has_convolutions:
            # CNNs typically have higher memory usage due to feature maps
            activation_memory_mb = batch_size * sequence_length * 256 * 4 / (1024 * 1024)
        else:
            # MLPs have lower activation memory
            activation_memory_mb = batch_size * sequence_length * 128 * 4 / (1024 * 1024)
        
        # Gradient memory (same as parameters)
        gradient_memory_mb = param_memory_mb
        
        # Total with some overhead
        total_memory_mb = (param_memory_mb + activation_memory_mb + gradient_memory_mb) * 1.5
        
        return total_memory_mb
    
    def select_optimal_device(
        self,
        model: Optional[torch.nn.Module] = None,
        model_type: Optional[ModelType] = None,
        policy_type: Optional[PolicyType] = None,
        config: Optional[Dict[str, Any]] = None,
        force_device: Optional[str] = None
    ) -> torch.device:
        """Select optimal device based on model characteristics
        
        Args:
            model: PyTorch model (optional)
            model_type: Type of model architecture
            policy_type: Type of RL policy (if applicable)
            config: Model configuration dictionary
            force_device: Force specific device ('cpu', 'cuda', 'auto')
            
        Returns:
            torch.device: Optimal device for the model
        """
        
        if force_device and force_device != 'auto':
            device = torch.device(force_device)
            logger.info(f"Using forced device: {device}")
            return device
        
        # If no CUDA available, use CPU
        if not self.device_info.cuda_available:
            logger.info("CUDA not available, using CPU")
            return torch.device('cpu')
        
        # Get model complexity
        complexity = self.estimate_model_complexity(model, model_type, config)
        
        # Decision logic based on model characteristics
        use_gpu = self._should_use_gpu(complexity, model_type, policy_type)
        
        if use_gpu:
            # Check if GPU has enough memory
            required_memory_gb = complexity.estimated_memory_mb / 1024
            if required_memory_gb > self.device_info.cuda_memory_gb * 0.8:  # 80% threshold
                logger.warning(
                    f"Model requires ~{required_memory_gb:.1f}GB but GPU only has "
                    f"{self.device_info.cuda_memory_gb:.1f}GB. Using CPU instead."
                )
                return torch.device('cpu')
            
            device = torch.device('cuda')
            logger.info(
                f"Using GPU for {model_type.value if model_type else 'model'}: "
                f"{complexity.parameter_count:,} parameters, "
                f"~{complexity.estimated_memory_mb:.0f}MB memory"
            )
        else:
            device = torch.device('cpu')
            logger.info(
                f"Using CPU for {model_type.value if model_type else 'model'}: "
                f"Optimal for this architecture type"
            )
        
        return device
    
    def _should_use_gpu(
        self,
        complexity: ModelComplexity,
        model_type: Optional[ModelType],
        policy_type: Optional[PolicyType]
    ) -> bool:
        """Determine if GPU should be used based on model characteristics"""
        
        # Policy-based decisions (your optimization strategy)
        if policy_type == PolicyType.CNN_POLICY:
            logger.debug("CNN Policy detected - GPU recommended for convolutions")
            return True
            
        elif policy_type in [PolicyType.MLP_POLICY, PolicyType.ACTOR_CRITIC_POLICY]:
            logger.debug("MLP/ActorCritic Policy detected - CPU optimal for simple MLPs")
            return False
        
        # Model type-based decisions
        if model_type == ModelType.CNN or complexity.has_convolutions:
            logger.debug("CNN operations detected - GPU beneficial for parallel convolutions")
            return True
            
        elif model_type == ModelType.CNN_LSTM_HYBRID:
            logger.debug("CNN+LSTM Hybrid detected - GPU beneficial for CNN component")
            return True
            
        elif model_type == ModelType.TRANSFORMER or complexity.has_attention:
            logger.debug("Attention mechanisms detected - GPU beneficial for matrix operations")
            return True
            
        elif model_type == ModelType.MLP:
            # For MLPs, consider size and complexity
            if complexity.parameter_count < self.cpu_preference_threshold:
                logger.debug("Small MLP detected - CPU sufficient")
                return False
            else:
                logger.debug("Large MLP detected - GPU may be beneficial")
                return True
                
        elif model_type == ModelType.LSTM:
            # LSTMs can benefit from GPU for large sequences
            if complexity.sequence_length > 100 and complexity.parameter_count > 500000:
                logger.debug("Large LSTM detected - GPU beneficial for long sequences")
                return True
            else:
                logger.debug("Small/Medium LSTM detected - CPU sufficient")
                return False
        
        # Default: use GPU for large models, CPU for small ones
        if complexity.parameter_count > self.cpu_preference_threshold:
            return True
        else:
            return False
    
    def get_device_recommendations(self) -> Dict[str, Any]:
        """Get device recommendations for different model types"""
        
        recommendations = {
            "device_info": {
                "cuda_available": self.device_info.cuda_available,
                "cuda_memory_gb": self.device_info.cuda_memory_gb,
                "device_name": self.device_info.device_name,
                "cpu_cores": self.device_info.cpu_cores
            },
            "recommendations": {
                "CNN models": "GPU (parallel convolutions)",
                "CNN+LSTM Hybrid": "GPU (CNN component benefits)",
                "Large LSTM (>100 seq_len)": "GPU (matrix operations)",
                "Small LSTM (<100 seq_len)": "CPU (sufficient performance)",
                "MLP/ActorCritic Policy": "CPU (optimal for simple MLPs)",
                "CNN Policy": "GPU (convolution operations)",
                "Transformer/Attention": "GPU (attention matrix ops)",
                "Small models (<1M params)": "CPU (lower overhead)",
                "Large models (>1M params)": "GPU (if memory allows)"
            },
            "memory_considerations": {
                "gpu_memory_gb": self.device_info.cuda_memory_gb,
                "recommended_model_size_gb": self.device_info.cuda_memory_gb * 0.8,
                "batch_size_recommendations": {
                    "small_gpu_2gb": "batch_size <= 16",
                    "medium_gpu_4gb": "batch_size <= 32", 
                    "large_gpu_8gb": "batch_size <= 64"
                }
            }
        }
        
        return recommendations
    
    def optimize_model_config(
        self,
        config: Dict[str, Any],
        model_type: ModelType,
        target_device: Optional[str] = None
    ) -> Dict[str, Any]:
        """Optimize model configuration for target device"""
        
        optimized_config = config.copy()
        
        # Select optimal device
        device = self.select_optimal_device(
            model_type=model_type,
            config=config,
            force_device=target_device
        )
        
        optimized_config['device'] = str(device)
        
        # Adjust batch size based on device and memory
        if device.type == 'cuda':
            # GPU optimizations
            if self.device_info.cuda_memory_gb >= 8:
                optimized_config['batch_size'] = min(config.get('batch_size', 32), 64)
            elif self.device_info.cuda_memory_gb >= 4:
                optimized_config['batch_size'] = min(config.get('batch_size', 32), 32)
            else:
                optimized_config['batch_size'] = min(config.get('batch_size', 32), 16)
                
            # Enable mixed precision for supported GPUs
            if (self.device_info.compute_capability and 
                self.device_info.compute_capability >= (7, 0)):
                optimized_config['use_mixed_precision'] = True
                
        else:
            # CPU optimizations
            optimized_config['batch_size'] = min(config.get('batch_size', 32), 16)
            optimized_config['num_workers'] = min(self.device_info.cpu_cores, 4)
            optimized_config['use_mixed_precision'] = False
        
        return optimized_config


# Global device optimizer instance
device_optimizer = DeviceOptimizer()


def get_optimal_device(
    model: Optional[torch.nn.Module] = None,
    model_type: Optional[Union[str, ModelType]] = None,
    policy_type: Optional[Union[str, PolicyType]] = None,
    config: Optional[Dict[str, Any]] = None,
    force_device: Optional[str] = None
) -> torch.device:
    """Convenience function to get optimal device
    
    Args:
        model: PyTorch model (optional)
        model_type: Model architecture type
        policy_type: RL policy type (if applicable)
        config: Model configuration
        force_device: Force specific device
        
    Returns:
        torch.device: Optimal device
    """
    
    # Convert string types to enums
    if isinstance(model_type, str):
        model_type = ModelType(model_type.lower())
    if isinstance(policy_type, str):
        policy_type = PolicyType(policy_type)
    
    return device_optimizer.select_optimal_device(
        model=model,
        model_type=model_type,
        policy_type=policy_type,
        config=config,
        force_device=force_device
    )


def optimize_config_for_device(
    config: Dict[str, Any],
    model_type: Union[str, ModelType],
    target_device: Optional[str] = None
) -> Dict[str, Any]:
    """Optimize configuration for optimal device
    
    Args:
        config: Model configuration
        model_type: Model architecture type
        target_device: Target device ('cpu', 'cuda', 'auto')
        
    Returns:
        Optimized configuration dictionary
    """
    
    if isinstance(model_type, str):
        model_type = ModelType(model_type.lower())
    
    return device_optimizer.optimize_model_config(
        config=config,
        model_type=model_type,
        target_device=target_device
    )


def print_device_recommendations():
    """Print device recommendations for different model types"""
    
    recommendations = device_optimizer.get_device_recommendations()
    
    print("\n=== Device Optimization Recommendations ===")
    print(f"\nDevice Info:")
    for key, value in recommendations["device_info"].items():
        print(f"  {key}: {value}")
    
    print(f"\nModel Type Recommendations:")
    for model_type, recommendation in recommendations["recommendations"].items():
        print(f"  {model_type}: {recommendation}")
    
    print(f"\nMemory Considerations:")
    memory_info = recommendations["memory_considerations"]
    print(f"  GPU Memory: {memory_info['gpu_memory_gb']:.1f}GB")
    print(f"  Recommended Model Size: {memory_info['recommended_model_size_gb']:.1f}GB")
    
    print(f"\nBatch Size Recommendations:")
    for gpu_type, batch_rec in memory_info["batch_size_recommendations"].items():
        print(f"  {gpu_type}: {batch_rec}")


if __name__ == "__main__":
    # Example usage and testing
    print_device_recommendations()
    
    # Test different model configurations
    test_configs = [
        {
            "name": "CNN Policy",
            "model_type": ModelType.CNN,
            "policy_type": PolicyType.CNN_POLICY,
            "config": {"input_dim": 100, "num_filters": 64, "filter_sizes": [3, 5, 7]}
        },
        {
            "name": "MLP Policy", 
            "model_type": ModelType.MLP,
            "policy_type": PolicyType.MLP_POLICY,
            "config": {"input_dim": 100, "hidden_dims": [256, 128], "output_dim": 3}
        },
        {
            "name": "CNN+LSTM Hybrid",
            "model_type": ModelType.CNN_LSTM_HYBRID,
            "config": {"input_dim": 100, "sequence_length": 60, "feature_fusion_dim": 256}
        }
    ]
    
    print("\n=== Device Selection Tests ===")
    for test in test_configs:
        device = get_optimal_device(
            model_type=test["model_type"],
            policy_type=test.get("policy_type"),
            config=test["config"]
        )
        print(f"{test['name']}: {device}")