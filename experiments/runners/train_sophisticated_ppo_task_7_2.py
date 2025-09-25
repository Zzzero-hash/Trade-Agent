#!/usr/bin/env python3
"""
Experiment Runner for Task 7.2: Train sophisticated PPO agent with policy optimization.

This script implements the complete training pipeline for a sophisticated PPO agent
with all the advanced features required by the task:
- GAE (Generalized Advantage Estimation)
- Adaptive KL penalty scheduling
- Entropy regularization
- Trust region constraints
- Natural policy gradients
- Parallel environment collection
- Performance validation (>1.0 Sortino ratio, <10% max drawdown)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from ml.sophisticated_ppo_trainer import SophisticatedPPOTrainer
from ml.yfinance_trading_environment import YFinanceConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ppo_task_7_2_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def create_advanced_env_config() -> YFinanceConfig:
    """Create advanced environment configuration for PPO training."""
    from config.trading_configs import TradingConfigFactory
    
    return TradingConfigFactory.create_training_config(
        max_position_size=0.15,  # Conservative position sizing
        reward_scaling=1.0,      # Reduced reward scaling to prevent overflow
        # Advanced overrides for PPO
        max_total_exposure=0.75,  # Allow 75% total exposure
        slippage_base=0.0005,     # 0.05% base slippage
        slippage_impact=0.0001,   # Additional impact slippage
        max_drawdown_limit=0.08,  # 8% max drawdown before episode ends
        stop_loss_threshold=0.025,  # 2.5% stop loss per position
        position_timeout=120,     # Max 120 steps per position
        prediction_horizon=5,     # 5-step prediction horizon
        min_episode_length=250,   # Minimum 250 steps per episode
        risk_free_rate=0.02,      # 2% annual risk-free rate
        sharpe_weight=0.3,        # Balanced Sharpe weight
        return_weight=0.4,        # Balanced return weight
        drawdown_penalty=1.5,     # Moderate drawdown penalty
        volatility_window=20,
        trend_window=50,
        regime_threshold=0.02
    )


def select_trading_symbols() -> list:
    """Select diverse trading symbols for robust training."""
    from config.trading_configs import get_default_symbols
    
    # Extend default symbols with additional diversity
    base_symbols = get_default_symbols()
    additional_symbols = [
        # High growth/volatility
        'META', 'NVDA', 'NFLX',
        # Traditional sectors
        'JPM', 'JNJ', 'PG', 'KO',
        # ETFs for diversification
        'SPY', 'QQQ', 'IWM'
    ]
    
    return base_symbols + additional_symbols


def train_sophisticated_ppo(
    total_timesteps: int = 500000,  # Reduced for initial validation
    n_envs: int = 4,  # Reduced for stability
    symbols: list = None,
    start_date: str = "2020-01-01",  # Shorter period for faster training
    end_date: str = "2023-12-31",
    log_dir: str = None
) -> dict:
    """Train sophisticated PPO agent with all advanced features.
    
    Args:
        total_timesteps: Total training timesteps (3M for 3000+ episodes)
        n_envs: Number of parallel environments
        symbols: Trading symbols to use
        start_date: Training data start date
        end_date: Training data end date
        log_dir: Logging directory
        
    Returns:
        Training results dictionary
    """
    if symbols is None:
        symbols = select_trading_symbols()
        
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/sophisticated_ppo_task_7_2_{timestamp}"
        
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    logger.info("="*80)
    logger.info("TASK 7.2: SOPHISTICATED PPO AGENT TRAINING")
    logger.info("="*80)
    logger.info(f"Training configuration:")
    logger.info(f"  Total timesteps: {total_timesteps:,}")
    logger.info(f"  Parallel environments: {n_envs}")
    logger.info(f"  Trading symbols: {symbols}")
    logger.info(f"  Data period: {start_date} to {end_date}")
    logger.info(f"  Log directory: {log_dir}")
    logger.info("="*80)
    
    # Create environment configuration
    env_config = create_advanced_env_config()
    
    # Initialize trainer
    trainer = SophisticatedPPOTrainer(
        env_config=env_config,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        n_envs=n_envs,
        log_dir=log_dir
    )
    
    try:
        # Advanced PPO training configuration
        training_config = {
            'total_timesteps': total_timesteps,
            'learning_rate': 3e-4,
            'n_steps': 1024,          # Reduced steps per rollout for stability
            'batch_size': 32,         # Smaller batch size
            'n_epochs': 4,            # Fewer epochs to prevent overfitting
            'gamma': 0.99,            # Discount factor
            'gae_lambda': 0.95,       # GAE lambda parameter
            'clip_range': 0.2,        # PPO clipping range
            'clip_range_vf': None,    # Value function clipping
            'normalize_advantage': True,  # Normalize advantages
            'ent_coef': 0.01,         # Entropy coefficient (with scheduling)
            'vf_coef': 0.5,           # Value function coefficient
            'max_grad_norm': 0.5,     # Gradient clipping
            'use_sde': False,         # State Dependent Exploration
            'target_kl': 0.05,        # Increased target KL to prevent early stopping
            'verbose': 1,
            'seed': 42,
            'device': 'auto'  # Let device optimizer choose
        }
        
        logger.info("Starting sophisticated PPO training with advanced features:")
        logger.info("  * Generalized Advantage Estimation (GAE)")
        logger.info("  * Adaptive KL penalty scheduling")
        logger.info("  * Entropy regularization with decay")
        logger.info("  * Trust region constraints")
        logger.info("  * Natural policy gradients")
        logger.info("  * Parallel environment collection")
        logger.info("  * Advanced performance monitoring")
        
        # Start training
        start_time = time.time()
        results = trainer.train(**training_config)
        training_time = time.time() - start_time
        
        # Add training metadata
        results.update({
            'task': '7.2 Train sophisticated PPO agent with policy optimization',
            'training_config': training_config,
            'environment_config': env_config.__dict__,
            'symbols': symbols,
            'data_period': {'start': start_date, 'end': end_date},
            'actual_training_time': training_time,
            'timestamp': datetime.now().isoformat()
        })
        
        # Validate performance requirements
        final_eval = results.get('final_evaluation', {})
        sortino_ratio = final_eval.get('mean_sortino_ratio', 0)
        max_drawdown = final_eval.get('max_drawdown', 1)
        
        # Use centralized performance validation
        from config.trading_configs import validate_performance, format_performance_report
        
        validation_results = validate_performance(results)
        performance_report = format_performance_report(validation_results)
        
        logger.info("TRAINING COMPLETED - PERFORMANCE VALIDATION")
        logger.info(performance_report)
        
        if validation_results['all_targets_met']:
            logger.info("🎉 TASK 7.2 REQUIREMENTS SUCCESSFULLY ACHIEVED!")
        else:
            logger.warning("⚠️  Task requirements not fully met - consider additional training")
        
        # Save comprehensive results
        results_file = os.path.join(log_dir, "task_7_2_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Complete results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        trainer.cleanup()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Train sophisticated PPO agent for Task 7.2"
    )
    parser.add_argument(
        '--timesteps', 
        type=int, 
        default=500000,
        help='Total training timesteps (default: 500K for validation)'
    )
    parser.add_argument(
        '--envs', 
        type=int, 
        default=4,
        help='Number of parallel environments (default: 4)'
    )
    parser.add_argument(
        '--start-date', 
        type=str, 
        default="2018-01-01",
        help='Training data start date (default: 2018-01-01)'
    )
    parser.add_argument(
        '--end-date', 
        type=str, 
        default="2023-12-31",
        help='Training data end date (default: 2023-12-31)'
    )
    parser.add_argument(
        '--log-dir', 
        type=str, 
        default=None,
        help='Custom log directory (default: auto-generated)'
    )
    parser.add_argument(
        '--quick-test', 
        action='store_true',
        help='Run quick test with reduced timesteps'
    )
    
    args = parser.parse_args()
    
    # Adjust for quick test
    if args.quick_test:
        args.timesteps = 100000  # 100K timesteps for quick test
        args.envs = 4
        logger.info("Running in quick test mode with reduced timesteps")
    
    try:
        # Execute training
        results = train_sophisticated_ppo(
            total_timesteps=args.timesteps,
            n_envs=args.envs,
            start_date=args.start_date,
            end_date=args.end_date,
            log_dir=args.log_dir
        )
        
        # Print summary
        final_eval = results.get('final_evaluation', {})
        print("\n" + "="*60)
        print("TASK 7.2 EXECUTION SUMMARY")
        print("="*60)
        print(f"Training Status: {'COMPLETED' if results.get('training_completed') else 'FAILED'}")
        print(f"Total Timesteps: {results.get('total_timesteps', 0):,}")
        print(f"Training Time: {results.get('training_time', 0):.2f} seconds")
        print(f"Final Sortino Ratio: {final_eval.get('mean_sortino_ratio', 0):.4f}")
        print(f"Maximum Drawdown: {final_eval.get('max_drawdown', 0):.4f}")
        print(f"Requirements Met: {final_eval.get('performance_threshold_met', False) and final_eval.get('drawdown_threshold_met', False)}")
        print("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())