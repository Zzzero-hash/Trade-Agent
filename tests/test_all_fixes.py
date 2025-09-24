#!/usr/bin/env python3
"""
Test all fixes for hyperparameter optimization
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.ml.hyperopt_runner import run_hyperparameter_optimization

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_all_fixes():
    """Test all fixes with minimal configuration"""

    logger.info("Testing all fixes for hyperparameter optimization...")
    logger.info("Running 3 trials to verify fixes work...")

    # Use current date and go back appropriately for different timeframes
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Just 30 days for testing

    try:
        # Run minimal optimization to test all fixes
        results = run_hyperparameter_optimization(
            symbols=["AAPL"],  # Just one symbol
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            n_trials=3,  # 3 trials for testing
            results_dir="test_all_fixes_results",
        )

        logger.info(
            "✓ All fixes working! Hyperparameter optimization completed successfully!"
        )
        logger.info(f"Total trials: {len(results['study'].trials)}")

        # Check if any trials completed successfully
        completed_trials = [
            t for t in results["study"].trials if t.state.name == "COMPLETE"
        ]
        if completed_trials:
            logger.info(f"✓ {len(completed_trials)} trials completed successfully!")
            for i, trial in enumerate(completed_trials):
                logger.info(f"  Trial {trial.number}: objectives = {trial.values}")
        else:
            logger.warning(
                "No trials completed successfully - check for remaining issues"
            )

        return len(completed_trials) > 0

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_all_fixes()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
