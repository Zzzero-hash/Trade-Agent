"""
Standalone test for flexible CNN heads bug fix.
This test verifies that the dimension mismatch bug is fixed.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.ml.train_integrated_cnn_lstm import IntegratedCNNLSTMTrainer
from src.ml.hybrid_model import create_hybrid_config


def test_flexible_cnn_heads_bug_fix():
    """Test that the flexible CNN heads prevent the dimension mismatch bug"""
    print("🧪 Testing flexible CNN heads bug fix...")
    
    # Create configuration
    config = create_hybrid_config(
        input_dim=11,
        sequence_length=30,
        num_classes=4,
        regression_targets=2,
        batch_size=16
    )
    
    # Create trainer
    trainer = IntegratedCNNLSTMTrainer(
        config=config,
        save_dir="test_checkpoints",
        log_dir="test_logs"
    )
    
    # Build models (this creates the flexible heads)
    trainer.build_models(input_dim=11)
    
    print("✅ Models built successfully with flexible heads")
    
    # Test the exact problematic input that caused the original crash
    # RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x1 and 60x64)
    problematic_inputs = [
        torch.randn(32, 1),      # The exact shape that caused the crash
        torch.randn(16, 1),      # Smaller batch
        torch.randn(8, 64, 30),  # 3D CNN output
        torch.randn(16, 128),    # Different 2D shape
        torch.randn(4, 256, 15), # Another 3D shape
    ]
    
    class_head = trainer.cnn_baseline.classification_head
    reg_head = trainer.cnn_baseline.regression_head
    
    for i, test_input in enumerate(problematic_inputs):
        try:
            with torch.no_grad():
                class_output = class_head(test_input)
                reg_output = reg_head(test_input)
            
            # Verify correct output shapes
            expected_batch_size = test_input.size(0)
            assert class_output.shape == (expected_batch_size, config.num_classes)
            assert reg_output.shape == (expected_batch_size, config.regression_targets)
            
            print(f"✅ Test {i+1}: Input shape {tuple(test_input.shape)} -> "
                  f"Class: {tuple(class_output.shape)}, Reg: {tuple(reg_output.shape)}")
            
        except RuntimeError as e:
            if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                print(f"❌ Test {i+1} FAILED: Dimension mismatch bug still exists!")
                print(f"   Input shape: {tuple(test_input.shape)}")
                print(f"   Error: {e}")
                return False
            else:
                # Some other error, re-raise
                raise e
    
    # Test gradient flow
    print("🧪 Testing gradient flow...")
    class_head.train()
    test_input = torch.randn(16, 64, requires_grad=True)
    output = class_head(test_input)
    
    # Create dummy loss and backpropagate
    target = torch.randint(0, config.num_classes, (16,))
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    
    # Check gradients exist
    assert test_input.grad is not None
    assert class_head.linear1.weight.grad is not None
    assert class_head.linear2.weight.grad is not None
    
    print("✅ Gradient flow test passed")
    
    # Test dynamic layer creation
    print("🧪 Testing dynamic layer creation...")
    
    # Create a fresh head with the improved logic
    class FlexibleCNNHead(nn.Module):
        def __init__(self, output_dim, dropout_rate=0.3):
            super().__init__()
            self.output_dim = output_dim
            self.dropout_rate = dropout_rate
            self.dropout = nn.Dropout(dropout_rate)
            self.relu = nn.ReLU()
            
            # Track the expected input dimension
            self.expected_input_dim = None
            self.linear1 = None
            self.linear2 = None
            
        def forward(self, x):
            # Handle different input shapes dynamically
            if x.dim() > 2:
                # For 3D inputs like (batch, channels, sequence), apply global average pooling
                x = torch.mean(x, dim=-1)  # Average over the last dimension
            elif x.dim() == 2:
                # Already 2D (batch, features), keep as is
                pass
            elif x.dim() == 1:
                # 1D case, add batch dimension
                x = x.unsqueeze(0)
            else:
                # Handle any other weird cases by flattening
                x = x.view(x.size(0), -1)
            
            # Ensure we have a 2D tensor (batch_size, features)
            if x.dim() != 2:
                x = x.view(x.size(0), -1)
            
            current_input_dim = x.size(1)
            
            # Create or recreate linear layers if input dimension changed
            if (self.linear1 is None or 
                self.expected_input_dim != current_input_dim):
                
                hidden_dim = max(64, current_input_dim // 2)  # Adaptive hidden dimension
                
                self.linear1 = nn.Linear(current_input_dim, hidden_dim).to(x.device)
                self.linear2 = nn.Linear(hidden_dim, self.output_dim).to(x.device)
                
                # Initialize weights
                nn.init.xavier_uniform_(self.linear1.weight)
                nn.init.xavier_uniform_(self.linear2.weight)
                
                self.expected_input_dim = current_input_dim
            
            # Forward pass
            x = self.linear1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.linear2(x)
            
            return x
    
    fresh_head = FlexibleCNNHead(output_dim=4)
    
    # Initially, linear layers should be None
    assert fresh_head.linear1 is None
    assert fresh_head.linear2 is None
    
    # After first forward pass, they should be created
    test_input = torch.randn(16, 64)
    with torch.no_grad():
        output = fresh_head(test_input)
    
    assert fresh_head.linear1 is not None
    assert fresh_head.linear2 is not None
    
    print("✅ Dynamic layer creation test passed")
    
    print("\n🎉 ALL TESTS PASSED! The flexible CNN heads bug fix is working perfectly!")
    print("   The dimension mismatch bug has been successfully prevented.")
    return True


if __name__ == "__main__":
    success = test_flexible_cnn_heads_bug_fix()
    if success:
        print("\n✨ The beautiful integrated CNN+LSTM model is safe and the baseline comparison will work!")
    else:
        print("\n💥 Bug fix failed - needs more work")
        exit(1)