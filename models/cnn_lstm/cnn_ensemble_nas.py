"""
CNN Ensemble with Neural Architecture Search

This module implements a sophisticated CNN ensemble system with automated
neural architecture search (DARTS-based), progressive growing, adaptive depth,
and ensemble distillation for efficient inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import math
import numpy as np
from abc import ABC, abstractmethod
import copy


@dataclass
class NASConfig:
    """Configuration for Neural Architecture Search."""
    
    # Search space
    max_layers: int = 8
    min_layers: int = 2
    filter_choices: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    kernel_choices: List[int] = field(default_factory=lambda: [3, 5, 7])
    dilation_choices: List[int] = field(default_factory=lambda: [1, 2, 4])
    
    # DARTS parameters
    learning_rate_arch: float = 3e-4
    learning_rate_model: float = 1e-3
    arch_weight_decay: float = 1e-3
    model_weight_decay: float = 3e-4
    
    # Progressive growing
    enable_progressive_growing: bool = True
    growth_epochs: int = 50
    max_depth: int = 12
    
    # Adaptive depth
    enable_adaptive_depth: bool = True
    depth_threshold: float = 0.1
    
    # Ensemble parameters
    num_architectures: int = 5
    diversity_weight: float = 0.1
    
    # Distillation
    enable_distillation: bool = True
    temperature: float = 4.0
    distillation_weight: float = 0.7


@dataclass
class CNNEnsembleConfig:
    """Configuration for CNN Ensemble."""
    
    # Input dimensions
    input_channels: int = 5  # OHLCV
    sequence_length: int = 100
    
    # NAS configuration
    nas_config: NASConfig = field(default_factory=NASConfig)
    
    # Ensemble configuration
    ensemble_size: int = 5
    diversity_regularization: float = 0.1
    
    # Output
    output_dim: int = 512
    
    # Training
    dropout_rate: float = 0.1
    batch_norm: bool = True


class ArchitectureOperation(nn.Module, ABC):
    """Abstract base class for architecture operations."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_flops(self, input_shape: Tuple[int, ...]) -> int:
        """Get FLOPs for this operation."""
        pass


class ConvOperation(ArchitectureOperation):
    """Convolution operation for NAS."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x
    
    def get_flops(self, input_shape: Tuple[int, ...]) -> int:
        """Calculate FLOPs for convolution operation."""
        batch_size, in_channels, seq_len = input_shape
        return (
            batch_size * seq_len * self.out_channels * 
            self.in_channels * self.kernel_size
        )


class SkipConnection(ArchitectureOperation):
    """Skip connection operation."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if in_channels != out_channels:
            self.projection = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.projection = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)
    
    def get_flops(self, input_shape: Tuple[int, ...]) -> int:
        """Calculate FLOPs for skip connection."""
        if self.in_channels != self.out_channels:
            batch_size, _, seq_len = input_shape
            return batch_size * seq_len * self.in_channels * self.out_channels
        return 0


class ZeroOperation(ArchitectureOperation):
    """Zero operation (no connection)."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len = x.shape
        return torch.zeros(batch_size, self.out_channels, seq_len, device=x.device)
    
    def get_flops(self, input_shape: Tuple[int, ...]) -> int:
        return 0


class MixedOperation(nn.Module):
    """Mixed operation for DARTS."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        operations: List[ArchitectureOperation]
    ):
        super().__init__()
        
        self.operations = nn.ModuleList(operations)
        self.num_ops = len(operations)
        
        # Architecture parameters (alpha)
        self.alpha = nn.Parameter(torch.randn(self.num_ops))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weighted combination of operations."""
        # Apply softmax to architecture parameters
        weights = F.softmax(self.alpha, dim=0)
        
        # Weighted combination of operations
        output = sum(w * op(x) for w, op in zip(weights, self.operations))
        
        return output
    
    def get_dominant_operation(self) -> int:
        """Get the index of the dominant operation."""
        return torch.argmax(self.alpha).item()


class SearchableCell(nn.Module):
    """Searchable cell for DARTS-based NAS."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: NASConfig
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.config = config
        
        # Create operations for search space
        operations = []
        
        # Convolution operations
        for kernel_size in config.kernel_choices:
            for dilation in config.dilation_choices:
                operations.append(
                    ConvOperation(in_channels, out_channels, kernel_size, dilation)
                )
        
        # Skip connection
        operations.append(SkipConnection(in_channels, out_channels))
        
        # Zero operation
        operations.append(ZeroOperation(in_channels, out_channels))
        
        # Mixed operation
        self.mixed_op = MixedOperation(in_channels, out_channels, operations)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixed_op(x)
    
    def get_architecture(self) -> Dict[str, Any]:
        """Get the current architecture configuration."""
        dominant_op_idx = self.mixed_op.get_dominant_operation()
        operation = self.mixed_op.operations[dominant_op_idx]
        
        if isinstance(operation, ConvOperation):
            return {
                'type': 'conv',
                'kernel_size': operation.kernel_size,
                'dilation': operation.dilation,
                'in_channels': operation.in_channels,
                'out_channels': operation.out_channels
            }
        elif isinstance(operation, SkipConnection):
            return {
                'type': 'skip',
                'in_channels': operation.in_channels,
                'out_channels': operation.out_channels
            }
        else:
            return {'type': 'zero'}


class ProgressiveGrowingNetwork(nn.Module):
    """Network with progressive growing capability."""
    
    def __init__(self, config: CNNEnsembleConfig):
        super().__init__()
        
        self.config = config
        self.current_depth = config.nas_config.min_layers
        self.max_depth = config.nas_config.max_depth
        
        # Initialize with minimum depth
        self.cells = nn.ModuleList()
        self._build_initial_network()
        
        # Global average pooling and output projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # The output projection will be set dynamically based on the actual last layer size
        self.output_proj = None
        
    def _build_initial_network(self):
        """Build initial network with minimum depth."""
        in_channels = self.config.input_channels
        
        for i in range(self.current_depth):
            out_channels = self.config.nas_config.filter_choices[
                min(i, len(self.config.nas_config.filter_choices) - 1)
            ]
            
            cell = SearchableCell(in_channels, out_channels, self.config.nas_config)
            self.cells.append(cell)
            
            in_channels = out_channels
    
    def grow_network(self):
        """Add a new layer to the network."""
        if self.current_depth < self.max_depth:
            in_channels = self.config.nas_config.filter_choices[
                min(self.current_depth - 1, len(self.config.nas_config.filter_choices) - 1)
            ]
            out_channels = self.config.nas_config.filter_choices[
                min(self.current_depth, len(self.config.nas_config.filter_choices) - 1)
            ]
            
            new_cell = SearchableCell(in_channels, out_channels, self.config.nas_config)
            self.cells.append(new_cell)
            self.current_depth += 1
            
            return True
        return False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the progressive network."""
        for cell in self.cells:
            x = cell(x)
        
        # Global pooling and output projection
        x = self.global_pool(x)
        x = x.squeeze(-1)
        
        # Create output projection if not exists or if dimensions changed
        if self.output_proj is None or self.output_proj.in_features != x.shape[1]:
            self.output_proj = nn.Linear(x.shape[1], self.config.output_dim).to(x.device)
        
        x = self.output_proj(x)
        
        return x
    
    def get_architecture_config(self) -> List[Dict[str, Any]]:
        """Get the current architecture configuration."""
        return [cell.get_architecture() for cell in self.cells]


class AdaptiveDepthController(nn.Module):
    """Controller for adaptive depth mechanism."""
    
    def __init__(self, config: NASConfig, input_dim: int):
        super().__init__()
        
        self.config = config
        self.depth_predictor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict whether to continue processing."""
        return self.depth_predictor(features)


class CNNEnsembleNAS(nn.Module):
    """
    CNN Ensemble with Neural Architecture Search.
    
    This model implements a sophisticated ensemble of CNN architectures discovered
    through DARTS-based neural architecture search, with progressive growing,
    adaptive depth, and ensemble distillation capabilities.
    """
    
    def __init__(self, config: CNNEnsembleConfig):
        super().__init__()
        
        self.config = config
        
        # Create ensemble of progressive growing networks
        self.ensemble_networks = nn.ModuleList([
            ProgressiveGrowingNetwork(config)
            for _ in range(config.ensemble_size)
        ])
        
        # Adaptive depth controllers
        if config.nas_config.enable_adaptive_depth:
            self.depth_controllers = nn.ModuleList([
                AdaptiveDepthController(config.nas_config, config.output_dim)
                for _ in range(config.ensemble_size)
            ])
        else:
            self.depth_controllers = None
        
        # Ensemble combination weights
        self.ensemble_weights = nn.Parameter(torch.ones(config.ensemble_size))
        
        # Diversity regularization
        self.diversity_regularizer = DiversityRegularizer(config.diversity_regularization)
        
        # Distillation components
        if config.nas_config.enable_distillation:
            self.teacher_network = None  # Will be set during distillation
            self.distillation_loss = DistillationLoss(
                temperature=config.nas_config.temperature,
                alpha=config.nas_config.distillation_weight
            )
        
    def forward(self, x: torch.Tensor, return_individual: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the ensemble.
        
        Args:
            x: Input tensor (batch_size, input_channels, sequence_length)
            return_individual: Whether to return individual network outputs
            
        Returns:
            Ensemble output or dictionary with individual outputs
        """
        individual_outputs = []
        
        # Forward pass through each network
        for i, network in enumerate(self.ensemble_networks):
            output = network(x)
            
            # Apply adaptive depth if enabled
            if self.depth_controllers is not None:
                depth_score = self.depth_controllers[i](output)
                # Apply depth-based weighting (simplified)
                output = output * depth_score
            
            individual_outputs.append(output)
        
        # Ensemble combination
        ensemble_weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_output = sum(
            w * output for w, output in zip(ensemble_weights, individual_outputs)
        )
        
        if return_individual:
            return {
                'ensemble_output': ensemble_output,
                'individual_outputs': individual_outputs,
                'ensemble_weights': ensemble_weights
            }
        
        return ensemble_output
    
    def grow_networks(self) -> List[bool]:
        """Grow all networks in the ensemble."""
        growth_results = []
        for network in self.ensemble_networks:
            result = network.grow_network()
            growth_results.append(result)
        return growth_results
    
    def get_ensemble_architectures(self) -> List[List[Dict[str, Any]]]:
        """Get architecture configurations for all networks."""
        return [network.get_architecture_config() for network in self.ensemble_networks]
    
    def calculate_diversity_loss(self, individual_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Calculate diversity regularization loss."""
        return self.diversity_regularizer(individual_outputs)
    
    def set_teacher_network(self, teacher: nn.Module):
        """Set teacher network for distillation."""
        if self.config.nas_config.enable_distillation:
            self.teacher_network = teacher
    
    def distillation_forward(self, x: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with distillation loss calculation."""
        if not self.config.nas_config.enable_distillation or self.teacher_network is None:
            raise ValueError("Distillation not enabled or teacher network not set")
        
        # Student predictions
        student_output = self.forward(x)
        
        # Teacher predictions
        with torch.no_grad():
            teacher_output = self.teacher_network(x)
        
        # Calculate distillation loss
        distillation_loss = self.distillation_loss(student_output, teacher_output, targets)
        
        return {
            'student_output': student_output,
            'teacher_output': teacher_output,
            'distillation_loss': distillation_loss
        }


class DiversityRegularizer(nn.Module):
    """Diversity regularization for ensemble training."""
    
    def __init__(self, diversity_weight: float):
        super().__init__()
        self.diversity_weight = diversity_weight
    
    def forward(self, individual_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Calculate diversity loss to encourage different predictions."""
        if len(individual_outputs) < 2:
            return torch.tensor(0.0, device=individual_outputs[0].device)
        
        diversity_loss = 0.0
        num_pairs = 0
        
        for i in range(len(individual_outputs)):
            for j in range(i + 1, len(individual_outputs)):
                # Calculate correlation between outputs
                output_i = individual_outputs[i]
                output_j = individual_outputs[j]
                
                # Normalize outputs
                output_i_norm = F.normalize(output_i, dim=1)
                output_j_norm = F.normalize(output_j, dim=1)
                
                # Calculate cosine similarity
                similarity = torch.sum(output_i_norm * output_j_norm, dim=1).mean()
                
                # Diversity loss (negative similarity)
                diversity_loss += similarity
                num_pairs += 1
        
        diversity_loss = diversity_loss / num_pairs if num_pairs > 0 else 0.0
        
        return self.diversity_weight * diversity_loss


class DistillationLoss(nn.Module):
    """Knowledge distillation loss."""
    
    def __init__(self, temperature: float, alpha: float):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self, 
        student_logits: torch.Tensor, 
        teacher_logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculate distillation loss."""
        # Soft targets from teacher
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # Distillation loss
        distillation_loss = self.kl_div(soft_student, soft_targets) * (self.temperature ** 2)
        
        # Hard target loss
        hard_loss = self.ce_loss(student_logits, targets)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss
        
        return total_loss


class NASTrainer:
    """Trainer for Neural Architecture Search."""
    
    def __init__(self, model: CNNEnsembleNAS, config: NASConfig):
        self.model = model
        self.config = config
        
        # Separate optimizers for architecture and model parameters
        self.arch_optimizer = torch.optim.Adam(
            self._get_arch_parameters(),
            lr=config.learning_rate_arch,
            weight_decay=config.arch_weight_decay
        )
        
        self.model_optimizer = torch.optim.Adam(
            self._get_model_parameters(),
            lr=config.learning_rate_model,
            weight_decay=config.model_weight_decay
        )
        
        self.epoch = 0
        
    def _get_arch_parameters(self):
        """Get architecture parameters for optimization."""
        arch_params = []
        for network in self.model.ensemble_networks:
            for cell in network.cells:
                arch_params.append(cell.mixed_op.alpha)
        return arch_params
    
    def _get_model_parameters(self):
        """Get model parameters (excluding architecture parameters)."""
        model_params = []
        arch_param_ids = {id(p) for p in self._get_arch_parameters()}
        
        for param in self.model.parameters():
            if id(param) not in arch_param_ids:
                model_params.append(param)
        
        return model_params
    
    def train_step(
        self, 
        train_data: torch.Tensor, 
        train_targets: torch.Tensor,
        val_data: torch.Tensor, 
        val_targets: torch.Tensor
    ) -> Dict[str, float]:
        """Single training step for DARTS."""
        # Update architecture parameters on validation data
        self.arch_optimizer.zero_grad()
        val_output = self.model(val_data, return_individual=True)
        val_loss = F.cross_entropy(val_output['ensemble_output'], val_targets)
        
        # Add diversity regularization
        diversity_loss = self.model.calculate_diversity_loss(val_output['individual_outputs'])
        total_val_loss = val_loss + diversity_loss
        
        total_val_loss.backward()
        self.arch_optimizer.step()
        
        # Update model parameters on training data
        self.model_optimizer.zero_grad()
        train_output = self.model(train_data, return_individual=True)
        train_loss = F.cross_entropy(train_output['ensemble_output'], train_targets)
        
        # Add diversity regularization
        train_diversity_loss = self.model.calculate_diversity_loss(train_output['individual_outputs'])
        total_train_loss = train_loss + train_diversity_loss
        
        total_train_loss.backward()
        self.model_optimizer.step()
        
        # Progressive growing
        if (self.config.enable_progressive_growing and 
            self.epoch % self.config.growth_epochs == 0 and 
            self.epoch > 0):
            growth_results = self.model.grow_networks()
            print(f"Progressive growing results: {growth_results}")
        
        self.epoch += 1
        
        return {
            'train_loss': train_loss.item(),
            'val_loss': val_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'total_train_loss': total_train_loss.item(),
            'total_val_loss': total_val_loss.item()
        }


def create_cnn_ensemble_nas(config: Optional[CNNEnsembleConfig] = None) -> CNNEnsembleNAS:
    """
    Factory function to create a CNN Ensemble with NAS.
    
    Args:
        config: Optional configuration. If None, uses default configuration.
        
    Returns:
        Initialized CNNEnsembleNAS model
    """
    if config is None:
        config = CNNEnsembleConfig()
    
    return CNNEnsembleNAS(config)


if __name__ == "__main__":
    # Example usage and testing
    nas_config = NASConfig(
        max_layers=6,
        min_layers=3,
        filter_choices=[32, 64, 128, 256],
        enable_progressive_growing=True,
        enable_adaptive_depth=True,
        num_architectures=3
    )
    
    config = CNNEnsembleConfig(
        input_channels=5,
        sequence_length=100,
        nas_config=nas_config,
        ensemble_size=3,
        output_dim=512
    )
    
    model = create_cnn_ensemble_nas(config)
    
    # Create sample data
    batch_size = 16
    sample_data = torch.randn(batch_size, 5, 100)
    
    # Forward pass
    output = model(sample_data, return_individual=True)
    print(f"Ensemble output shape: {output['ensemble_output'].shape}")
    print(f"Number of individual outputs: {len(output['individual_outputs'])}")
    print(f"Ensemble weights: {output['ensemble_weights']}")
    
    # Get architecture configurations
    architectures = model.get_ensemble_architectures()
    print(f"Number of architectures: {len(architectures)}")
    for i, arch in enumerate(architectures):
        print(f"Architecture {i}: {len(arch)} layers")
    
    # Test progressive growing
    growth_results = model.grow_networks()
    print(f"Growth results: {growth_results}")
    
    # Test NAS trainer
    trainer = NASTrainer(model, nas_config)
    
    # Create sample training data
    train_data = torch.randn(batch_size, 5, 100)
    train_targets = torch.randint(0, 10, (batch_size,))
    val_data = torch.randn(batch_size, 5, 100)
    val_targets = torch.randint(0, 10, (batch_size,))
    
    # Single training step
    losses = trainer.train_step(train_data, train_targets, val_data, val_targets)
    print(f"Training losses: {losses}")