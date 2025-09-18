"""Test basic project setup and imports"""

import sys
import os
from pathlib import Path

# Add src to path for testing
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

# Change to src directory for relative imports to work
os.chdir(Path(__file__).parent.parent / "src")

def test_basic_imports():
    """Test that basic modules can be imported"""
    
    # Test configuration
    from config.settings import Settings, get_settings, Environment
    print("✓ Configuration module imported successfully")
    
    # Test that we can create settings
    settings = Settings()
    assert settings.environment == Environment.DEVELOPMENT
    assert settings.debug == True
    print("✓ Settings creation successful")
    
    # Test that we can get settings
    settings = get_settings()
    print("✓ Settings retrieval successful")


def test_project_structure():
    """Test that project structure is correct"""
    
    # Check that all required directories exist
    base_path = Path(__file__).parent.parent / "src"
    
    required_dirs = [
        "config",
        "models", 
        "services",
        "repositories",
        "api",
        "exchanges",
        "ml",
        "utils"
    ]
    
    for dir_name in required_dirs:
        dir_path = base_path / dir_name
        assert dir_path.exists(), f"Directory {dir_name} does not exist"
        
        init_file = dir_path / "__init__.py"
        assert init_file.exists(), f"__init__.py missing in {dir_name}"
    
    print("✓ Project structure is correct")


def test_config_files():
    """Test that configuration files exist"""
    
    config_path = Path(__file__).parent.parent / "config"
    
    required_files = [
        "settings.yaml",
        "production.yaml"
    ]
    
    for file_name in required_files:
        file_path = config_path / file_name
        assert file_path.exists(), f"Configuration file {file_name} does not exist"
    
    print("✓ Configuration files exist")


if __name__ == "__main__":
    test_basic_imports()
    test_project_structure()
    test_config_files()
    print("\n🎉 All tests passed! Project structure is set up correctly.")