"""
GPU Utilities and Mixed Precision Training Setup

This module provides utilities for GPU detection, memory management,
and mixed precision training configuration.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class GPUManager:
    """GPU detection and management utilities"""
    
    def __init__(self):
        self.device = self._detect_device()
        self.device_count = self._get_device_count()
        self.device_properties = self._get_device_properties()
    
    def _detect_device(self) -> torch.device:
        """Automatically detect the best available device"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("MPS (Apple Silicon) available")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        
        return device
    
    def _get_device_count(self) -> int:
        """Get number of available devices"""
        if self.device.type == "cuda":
            return torch.cuda.device_count()
        return 1
    
    def _get_device_properties(self) -> Dict[str, Any]:
        """Get device properties and capabilities"""
        properties = {}
        
        if self.device.type == "cuda":
            for i in range(self.device_count):
                props = torch.cuda.get_device_properties(i)
                properties[f"cuda:{i}"] = {
                    "name": props.name,
                    "total_memory": props.total_memory,
                    "major": props.major,
                    "minor": props.minor,
                    "multi_processor_count": props.multi_processor_count
                }
        
        return properties
    
    def get_memory_info(self, device_id: Optional[int] = None) -> Dict[str, int]:
        """Get GPU memory information"""
        if self.device.type != "cuda":
            return {"allocated": 0, "reserved": 0, "free": 0}
        
        if device_id is None:
            device_id = torch.cuda.current_device()
        
        allocated = torch.cuda.memory_allocated(device_id)
        reserved = torch.cuda.memory_reserved(device_id)
        total = torch.cuda.get_device_properties(device_id).total_memory
        free = total - reserved
        
        return {
            "allocated": allocated,
            "reserved": reserved,
            "free": free,
            "total": total
        }
    
    def clear_cache(self):
        """Clear GPU memory cache"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
    
    def set_memory_fraction(self, fraction: float, device_id: Optional[int] = None):
        """Set maximum memory fraction for GPU"""
        if self.device.type == "cuda":
            if device_id is None:
                device_id = torch.cuda.current_device()
            torch.cuda.set_per_process_memory_fraction(fraction, device_id)


class MixedPrecisionTrainer:
    """Mixed precision training utilities"""
    
    def __init__(self, enabled: bool = True, dtype: str = "float16"):
        self.enabled = enabled and torch.cuda.is_available()
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.scaler = GradScaler() if self.enabled else None
        
        logger.info(f"Mixed precision training: {'enabled' if self.enabled else 'disabled'}")
        if self.enabled:
            logger.info(f"Using dtype: {self.dtype}")
    
    def autocast_context(self):
        """Get autocast context manager"""
        if self.enabled:
            return autocast(dtype=self.dtype)
        else:
            return torch.no_grad()  # Dummy context
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training"""
        if self.enabled and self.scaler:
            return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer):
        """Step optimizer with gradient scaling"""
        if self.enabled and self.scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def backward(self, loss: torch.Tensor):
        """Backward pass with gradient scaling"""
        if self.enabled and self.scaler:
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()
        else:
            loss.backward()


class ModelOptimizer:
    """Model optimization utilities"""
    
    @staticmethod
    def compile_model(model: nn.Module, mode: str = "default") -> nn.Module:
        """Compile model for optimization (PyTorch 2.0+)"""
        if hasattr(torch, 'compile'):
            try:
                compiled_model = torch.compile(model, mode=mode)
                logger.info(f"Model compiled with mode: {mode}")
                return compiled_model
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
                return model
        else:
            logger.warning("torch.compile not available")
            return model
    
    @staticmethod
    def to_channels_last(model: nn.Module) -> nn.Module:
        """Convert model to channels_last memory format"""
        try:
            model = model.to(memory_format=torch.channels_last)
            logger.info("Model converted to channels_last format")
        except Exception as e:
            logger.warning(f"channels_last conversion failed: {e}")
        return model
    
    @staticmethod
    def enable_deterministic(seed: int = 42):
        """Enable deterministic training"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Deterministic training enabled with seed: {seed}")


def setup_distributed_training(backend: str = "nccl", 
                             world_size: int = 1, 
                             rank: int = 0) -> bool:
    """Setup distributed training"""
    if world_size > 1:
        try:
            torch.distributed.init_process_group(
                backend=backend,
                world_size=world_size,
                rank=rank
            )
            logger.info(f"Distributed training initialized: {backend}")
            return True
        except Exception as e:
            logger.error(f"Distributed training setup failed: {e}")
            return False
    return False


def get_device_info() -> Dict[str, Any]:
    """Get comprehensive device information"""
    gpu_manager = GPUManager()
    
    info = {
        "device": str(gpu_manager.device),
        "device_count": gpu_manager.device_count,
        "device_properties": gpu_manager.device_properties,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "memory_info": gpu_manager.get_memory_info()
        })
    
    return info