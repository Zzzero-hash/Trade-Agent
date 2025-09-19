#!/usr/bin/env python3
"""
Advanced ML Trading Environment Setup Script

This script sets up the complete development environment for the ML trading system,
including dependency installation, GPU configuration, and experiment tracking setup.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any

import yaml
from utils.gpu_utils import get_device_info, GPUManager
from experiments.tracking import ExperimentTracker


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnvironmentSetup:
    """Environment setup and validation"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.gpu_manager = GPUManager()
        
    def check_python_version(self) -> bool:
        """Check if Python version meets requirements"""
        version = sys.version_info
        if version.major == 3 and version.minor >= 9:
            logger.info(f"Python version OK: {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            logger.error(f"Python 3.9+ required, found: {version.major}.{version.minor}.{version.micro}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies"""
        try:
            logger.info("Installing dependencies from requirements.txt...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], check=True)
            logger.info("Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def setup_gpu_environment(self) -> bool:
        """Setup GPU environment and validate configuration"""
        device_info = get_device_info()
        logger.info("GPU Environment Information:")
        for key, value in device_info.items():
            logger.info(f"  {key}: {value}")
        
        if device_info["cuda_available"]:
            # Set optimal GPU settings
            self.gpu_manager.set_memory_fraction(0.9)
            logger.info("GPU environment configured successfully")
            return True
        else:
            logger.warning("CUDA not available, using CPU")
            return False
    
    def setup_experiment_tracking(self) -> bool:
        """Setup experiment tracking systems"""
        try:
            # Create MLflow directories
            mlflow_dir = self.project_root / "mlruns"
            mlflow_dir.mkdir(exist_ok=True)
            
            # Initialize experiment tracker
            tracker = ExperimentTracker()
            logger.info("Experiment tracking setup completed")
            return True
        except Exception as e:
            logger.error(f"Failed to setup experiment tracking: {e}")
            return False
    
    def create_data_directories(self) -> bool:
        """Create necessary data directories"""
        directories = [
            "data/raw",
            "data/processed",
            "data/features",
            "models/checkpoints",
            "models/saved",
            "experiments/logs",
            "experiments/results",
            "outputs/plots",
            "outputs/reports"
        ]
        
        try:
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
            
            logger.info("Data directories created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            return False
    
    def validate_installation(self) -> Dict[str, bool]:
        """Validate the complete installation"""
        results = {}
        
        # Test core imports
        try:
            import torch
            import numpy as np
            import pandas as pd
            import mlflow
            import optuna
            results["core_imports"] = True
            logger.info("Core imports successful")
        except ImportError as e:
            results["core_imports"] = False
            logger.error(f"Core import failed: {e}")
        
        # Test GPU functionality
        try:
            import torch
            if torch.cuda.is_available():
                x = torch.randn(10, 10).cuda()
                y = torch.matmul(x, x.T)
                results["gpu_test"] = True
                logger.info("GPU test successful")
            else:
                results["gpu_test"] = False
                logger.warning("GPU not available for testing")
        except Exception as e:
            results["gpu_test"] = False
            logger.error(f"GPU test failed: {e}")
        
        # Test experiment tracking
        try:
            tracker = ExperimentTracker()
            run_id = tracker.start_run("validation_test")
            tracker.log_params({"test_param": 1.0})
            tracker.log_metrics({"test_metric": 0.95})
            tracker.end_run()
            results["experiment_tracking"] = True
            logger.info("Experiment tracking test successful")
        except Exception as e:
            results["experiment_tracking"] = False
            logger.error(f"Experiment tracking test failed: {e}")
        
        return results
    
    def run_setup(self) -> bool:
        """Run complete environment setup"""
        logger.info("Starting ML Trading Environment Setup...")
        
        steps = [
            ("Python version check", self.check_python_version),
            ("Dependency installation", self.install_dependencies),
            ("GPU environment setup", self.setup_gpu_environment),
            ("Experiment tracking setup", self.setup_experiment_tracking),
            ("Data directories creation", self.create_data_directories),
        ]
        
        success = True
        for step_name, step_func in steps:
            logger.info(f"Running: {step_name}")
            if not step_func():
                logger.error(f"Failed: {step_name}")
                success = False
            else:
                logger.info(f"Completed: {step_name}")
        
        if success:
            logger.info("Running validation tests...")
            validation_results = self.validate_installation()
            
            logger.info("Validation Results:")
            for test, result in validation_results.items():
                status = "PASS" if result else "FAIL"
                logger.info(f"  {test}: {status}")
            
            overall_success = all(validation_results.values())
            if overall_success:
                logger.info("🎉 Environment setup completed successfully!")
            else:
                logger.warning("⚠️ Environment setup completed with warnings")
            
            return overall_success
        else:
            logger.error("❌ Environment setup failed")
            return False


def main():
    """Main setup function"""
    setup = EnvironmentSetup()
    success = setup.run_setup()
    
    if success:
        print("\n" + "="*60)
        print("ML Trading Environment Setup Complete!")
        print("="*60)
        print("\nNext steps:")
        print("1. Review configuration files in configs/")
        print("2. Set up your data sources in data/ingestion/")
        print("3. Start developing models in models/")
        print("4. Run experiments using experiments/runners/")
        print("\nFor GPU training, ensure CUDA drivers are properly installed.")
        print("For experiment tracking, configure your W&B account if needed.")
    else:
        print("\n" + "="*60)
        print("Setup encountered issues. Please check the logs above.")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()