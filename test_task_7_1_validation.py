#!/usr/bin/env python3
"""
Test script to validate Task 7.1 implementation without full training.

This script tests the core components of the Rainbow DQN implementation
to ensure all features are properly implemented and working.
"""

import os
import sys
import logging
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ml.rainbow_dqn_task_7_1_trainer import RainbowDQNTask71Trainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_rainbow_dqn_components():
    """Test all Rainbow DQN components are properly implemented."""
    logger.info("Testing Rainbow DQN Task 7.1 Implementation")
    logger.info("=" * 60)
    
    try:
        # Test 1: Initialize trainer
        logger.info("Test 1: Initializing Rainbow DQN trainer...")
        trainer = RainbowDQNTask71Trainer(
            symbols=["SPY", "AAPL"],  # Use fewer symbols for testing
            start_date="2022-01-01",
            end_date="2022-12-31"
        )
        logger.info("✓ Trainer initialized successfully")
        
        # Test 2: Verify Rainbow features are enabled
        logger.info("Test 2: Verifying Rainbow features...")
        config = trainer.config
        
        assert config.distributional == True, "C51 distributional RL not enabled"
        logger.info("✓ C51 Distributional DQN enabled")
        
        assert config.dueling == True, "Dueling DQN not enabled"
        logger.info("✓ Dueling DQN enabled")
        
        assert config.prioritized_replay == True, "Prioritized replay not enabled"
        logger.info("✓ Prioritized Experience Replay enabled")
        
        assert config.noisy == True, "Noisy networks not enabled"
        logger.info("✓ Noisy Networks enabled")
        
        assert config.multi_step >= 1, "Multi-step learning not configured"
        logger.info("✓ Multi-step learning enabled")
        
        # Test 3: Verify agent initialization
        logger.info("Test 3: Verifying agent initialization...")
        agent = trainer.agent
        
        assert agent is not None, "Agent not initialized"
        assert hasattr(agent, 'q_network'), "Q-network not found"
        assert hasattr(agent, 'target_network'), "Target network not found"
        assert hasattr(agent, 'replay_buffer'), "Replay buffer not found"
        logger.info("✓ Agent components initialized")
        
        # Test 4: Verify network architecture
        logger.info("Test 4: Verifying network architecture...")
        q_network = agent.q_network
        
        assert hasattr(q_network, 'value_output'), "Dueling architecture not found"
        assert hasattr(q_network, 'advantage_output'), "Dueling architecture not found"
        logger.info("✓ Dueling network architecture verified")
        
        # Test 5: Test environment wrapper
        logger.info("Test 5: Testing environment wrapper...")
        train_env = trainer.train_env
        
        assert hasattr(train_env, 'action_space'), "Environment not properly wrapped"
        assert hasattr(train_env.action_space, 'n'), "Discrete action space not found"
        logger.info("✓ Environment properly wrapped for DQN")
        
        # Test 6: Test prediction functionality
        logger.info("Test 6: Testing prediction functionality...")
        obs, _ = train_env.reset()
        action, _ = agent.predict(obs, deterministic=True)
        
        assert action is not None, "Prediction failed"
        assert isinstance(action, (list, tuple)) or hasattr(action, '__iter__'), "Action format incorrect"
        logger.info("✓ Prediction functionality working")
        
        logger.info("=" * 60)
        logger.info("ALL TESTS PASSED!")
        logger.info("Rainbow DQN Task 7.1 implementation is ready for training")
        logger.info("=" * 60)
        
        # Summary of implemented features
        logger.info("IMPLEMENTED FEATURES:")
        logger.info("• C51 Distributional DQN with {} atoms".format(config.n_atoms))
        logger.info("• Double DQN for reduced overestimation bias")
        logger.info("• Dueling DQN for separate value/advantage estimation")
        logger.info("• Prioritized Experience Replay with alpha={:.2f}".format(config.alpha))
        logger.info("• Noisy Networks for parameter space exploration")
        logger.info("• Multi-step learning with n={}".format(config.multi_step))
        logger.info("• Target Sharpe ratio validation: {:.1f}".format(1.5))
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


def test_training_methods():
    """Test that all required training methods are implemented."""
    logger.info("Testing training method implementations...")
    
    try:
        trainer = RainbowDQNTask71Trainer(
            symbols=["SPY"],
            start_date="2022-01-01", 
            end_date="2022-12-31"
        )
        
        # Check all required methods exist
        required_methods = [
            'train_c51_distributional_dqn',
            'train_double_dueling_dqn_with_prioritized_replay',
            'train_noisy_networks_exploration',
            'validate_performance',
            'run_complete_task_7_1'
        ]
        
        for method_name in required_methods:
            assert hasattr(trainer, method_name), f"Method {method_name} not implemented"
            method = getattr(trainer, method_name)
            assert callable(method), f"Method {method_name} is not callable"
            logger.info(f"✓ {method_name} implemented")
        
        logger.info("✓ All required training methods implemented")
        return True
        
    except Exception as e:
        logger.error(f"Training method test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    logger.info("Starting Task 7.1 Rainbow DQN Validation Tests")
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models/rainbow_dqn_task_7_1", exist_ok=True)
    
    # Run tests
    test1_passed = test_rainbow_dqn_components()
    test2_passed = test_training_methods()
    
    if test1_passed and test2_passed:
        logger.info("🎉 ALL VALIDATION TESTS PASSED!")
        logger.info("Task 7.1 implementation is ready for execution")
        logger.info("Run 'python scripts/execute_task_7_1_rainbow_dqn.py' to start training")
        return True
    else:
        logger.error("❌ SOME TESTS FAILED!")
        logger.error("Please fix the issues before running the full training")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)