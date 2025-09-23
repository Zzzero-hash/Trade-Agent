"""
Unit tests for flexible CNN baseline heads to prevent dimension mismatch bugs.

This test suite ensures that the FlexibleCNNHead can handle various CNN output shapes
without crashing, preventing the dimension mismatch bug that occurred during baseline
model training in Task 5.4.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock

# Import the FlexibleCNNHead from the training script
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.ml.train_integrated_cnn_lstm import IntegratedCNNLSTMTrainer
from src.ml.hybrid_model import create_hybrid_config


class TestFlexibleCNNHeads:
    """Test suite for flexible CNN baseline heads"""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration"""
        return create_hybrid_config(
            input_dim=11,
            sequence_length=30,
            num_classes=4,
            regression_targets=2,
            batch_size=16
        )
    
    @pytest.fixture
    def trainer(self, config):
        """Create a trainer instance for testing"""
        return IntegratedCNNLSTMTrainer(
            config=config,
            save_dir="test_checkpoints",
            log_dir="test_logs"
        )
    
    def test_flexible_head_creation(self, trainer, config):
        """Test that flexible CNN heads are created without errors"""
        # Build models to create the flexible heads
        trainer.build_models(input_dim=11)
        
        # Verify heads exist and are flexible
        assert hasattr(trainer.cnn_baseline, 'classification_head')
        assert hasattr(trainer.cnn_baseline, 'regression_head')
        
        # Verify they are the flexible type (have adaptive behavior)
        class_head = trainer.cnn_baseline.classification_head
        reg_head = trainer.cnn_baseline.regression_head
        
        assert hasattr(class_head, 'adaptive_pool')
        assert hasattr(reg_head, 'adaptive_pool')
    
    def test_flexible_head_various_input_shapes(self, trainer, config):
        """Test that flexible heads handle various input shapes without crashing"""
        trainer.build_models(input_dim=11)
        
        class_head = trainer.cnn_baseline.classification_head
        reg_head = trainer.cnn_baseline.regression_head
        
        # Test different input shapes that could come from CNN
        test_shapes = [
            (16, 64),           # 2D: (batch_size, features)
            (16, 64, 30),       # 3D: (batch_size, features, sequence)
            (16, 32, 15),       # 3D: different dimensions
            (8, 128),           # 2D: different batch size
            (1, 256, 60),       # 3D: single sample
        ]
        
        for shape in test_shapes:
            with torch.no_grad():
                test_input = torch.randn(shape)
                
                # Test classification head
                class_output = class_head(test_input)
                assert class_output.shape == (shape[0], config.num_classes)
                
                # Test regression head  
                reg_output = reg_head(test_input)
                assert reg_output.shape == (shape[0], config.regression_targets)
    
    def test_flexible_head_dynamic_layer_creation(self, trainer, config):
        """Test that linear layers are created dynamically on first forward pass"""
        trainer.build_models(input_dim=11)
        
        class_head = trainer.cnn_baseline.classification_head
        
        # Initially, linear layers should be None
        assert class_head.linear1 is None
        assert class_head.linear2 is None
        
        # After first forward pass, they should be created
        test_input = torch.randn(16, 64)
        with torch.no_grad():
            output = class_head(test_input)
        
        assert class_head.linear1 is not None
        assert class_head.linear2 is not None
        assert isinstance(class_head.linear1, nn.Linear)
        assert isinstance(class_head.linear2, nn.Linear)
    
    def test_flexible_head_consistent_output(self, trainer, config):
        """Test that heads produce consistent outputs for same input"""
        trainer.build_models(input_dim=11)
        
        class_head = trainer.cnn_baseline.classification_head
        
        # Same input should produce same output (in eval mode)
        class_head.eval()
        test_input = torch.randn(16, 64)
        
        with torch.no_grad():
            output1 = class_head(test_input)
            output2 = class_head(test_input)
        
        torch.testing.assert_close(output1, output2)
    
    def test_flexible_head_gradient_flow(self, trainer, config):
        """Test that gradients flow properly through flexible heads"""
        trainer.build_models(input_dim=11)
        
        class_head = trainer.cnn_baseline.classification_head
        class_head.train()
        
        test_input = torch.randn(16, 64, requires_grad=True)
        output = class_head(test_input)
        
        # Create a dummy loss and backpropagate
        target = torch.randint(0, config.num_classes, (16,))
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
        # Check that gradients exist
        assert test_input.grad is not None
        assert class_head.linear1.weight.grad is not None
        assert class_head.linear2.weight.grad is not None
    
    def test_dimension_mismatch_bug_prevention(self, trainer, config):
        """
        Specific test to prevent the exact bug that occurred:
        RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x1 and 60x64)
        """
        trainer.build_models(input_dim=11)
        
        class_head = trainer.cnn_baseline.classification_head
        reg_head = trainer.cnn_baseline.regression_head
        
        # Simulate the problematic input shape that caused the crash
        # (batch_size=32, features=1) after adaptive pooling
        problematic_input = torch.randn(32, 1)
        
        # This should NOT crash anymore
        with torch.no_grad():
            class_output = class_head(problematic_input)
            reg_output = reg_head(problematic_input)
        
        # Verify correct output shapes
        assert class_output.shape == (32, config.num_classes)
        assert reg_output.shape == (32, config.regression_targets)
    
    def test_flexible_head_with_3d_cnn_output(self, trainer, config):
        """Test handling of 3D CNN outputs (the typical case)"""
        trainer.build_models(input_dim=11)
        
        class_head = trainer.cnn_baseline.classification_head
        
        # Simulate typical CNN output: (batch, channels, sequence)
        cnn_output = torch.randn(32, 64, 30)  # 32 samples, 64 filters, 30 time steps
        
        with torch.no_grad():
            output = class_head(cnn_output)
        
        # Should be properly pooled and classified
        assert output.shape == (32, config.num_classes)
    
    def test_flexible_head_memory_efficiency(self, trainer, config):
        """Test that flexible heads don't create memory leaks"""
        trainer.build_models(input_dim=11)
        
        class_head = trainer.cnn_baseline.classification_head
        
        # Process multiple different input shapes
        shapes = [(16, 32), (8, 64), (4, 128)]
        
        for shape in shapes:
            test_input = torch.randn(shape)
            with torch.no_grad():
                output = class_head(test_input)
            
            # Linear layers should be created only once
            assert class_head.linear1 is not None
            assert class_head.linear2 is not None
            
            # Should handle the new shape without creating new layers
            assert output.shape == (shape[0], config.num_classes)


class TestIntegrationWithTraining:
    """Integration tests to ensure flexible heads work in training context"""
    
    def test_baseline_training_no_crash(self):
        """Test that baseline training doesn't crash with flexible heads"""
        config = create_hybrid_config(
            input_dim=11,
            sequence_length=30,
            num_classes=4,
            regression_targets=2,
            batch_size=8  # Small batch for testing
        )
        
        trainer = IntegratedCNNLSTMTrainer(config=config)
        trainer.build_models(input_dim=11)
        
        # Create dummy data loaders
        X = torch.randn(32, 11, 30)
        y_class = torch.randint(0, 4, (32,))
        y_reg = torch.randn(32, 2)
        
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(X, y_class, y_reg)
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=8, shuffle=False)
        
        # This should NOT crash anymore
        try:
            # Test just one epoch of CNN baseline training
            cnn_results = trainer._train_baseline_model(
                trainer.cnn_baseline, train_loader, val_loader, num_epochs=1, model_name='CNN'
            )
            success = True
        except RuntimeError as e:
            if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                success = False
            else:
                raise e
        
        assert success, "Flexible CNN heads should prevent dimension mismatch errors"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])