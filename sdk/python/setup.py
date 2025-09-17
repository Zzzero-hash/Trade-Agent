"""
Setup script for AI Trading Platform Python SDK
"""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-trading-platform-sdk",
    version="1.0.0",
    author="AI Trading Platform Team",
    author_email="sdk@tradingplatform.com",
    description="Python SDK for AI Trading Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ai-trading-platform/python-sdk",
    project_urls={
        "Bug Tracker": "https://github.com/ai-trading-platform/python-sdk/issues",
        "Documentation": "https://docs.tradingplatform.com/sdk/python",
        "Source Code": "https://github.com/ai-trading-platform/python-sdk",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "sphinx-autodoc-typehints>=1.22.0",
        ],
        "examples": [
            "jupyter>=1.0.0",
            "matplotlib>=3.6.0",
            "pandas>=1.5.0",
            "numpy>=1.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "trading-codegen=ai_trading_platform.codegen:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ai_trading_platform": [
            "templates/**/*",
            "examples/**/*",
        ],
    },
    keywords=[
        "trading",
        "finance",
        "ai",
        "machine-learning",
        "api",
        "sdk",
        "stocks",
        "forex",
        "crypto",
        "portfolio",
        "risk-management",
    ],
    zip_safe=False,
)