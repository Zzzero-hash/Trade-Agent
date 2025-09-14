# Project Structure & Architecture

## Directory Organization

```
src/                          # Source code (main package)
├── main.py                   # Application entry point
├── config/                   # Configuration management
├── models/                   # Pydantic data models with validation
├── services/                 # Business logic layer
├── repositories/             # Data access layer
├── api/                      # FastAPI endpoints
├── exchanges/                # Exchange connector implementations
├── ml/                       # Machine learning components
└── utils/                    # Utility functions

tests/                        # Test suite with pytest
├── test_*_connector.py       # Exchange integration tests
├── test_*_model.py          # ML model tests
└── test_*.py                # Other component tests

config/                       # Configuration files
├── settings.yaml             # Development configuration
└── production.yaml           # Production configuration

examples/                     # Usage demonstrations
docs/                         # Documentation
```

## Architecture Patterns

### Layered Architecture
- **API Layer**: FastAPI endpoints for external interfaces
- **Service Layer**: Business logic and orchestration
- **Repository Layer**: Data access abstraction
- **Model Layer**: Pydantic models for validation

### Abstract Base Classes
- **ExchangeConnector**: Common interface for all exchanges
- **BaseMLModel**: Common interface for ML models
- **FeatureTransformer**: Pluggable feature engineering

### Configuration-Driven Design
- Environment-specific settings in `config/`
- Override via environment variables
- Hierarchical configuration loading

## Code Organization Principles

### Module Structure
- Each major component has its own directory under `src/`
- Abstract base classes in `base.py` files
- Concrete implementations in separate files
- `__init__.py` files expose public interfaces

### Naming Conventions
- **Files**: snake_case (e.g., `market_data.py`)
- **Classes**: PascalCase (e.g., `MarketData`)
- **Functions/Variables**: snake_case (e.g., `get_historical_data`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_TIMEOUT`)

### Import Organization
```python
# Standard library imports
import asyncio
from datetime import datetime

# Third-party imports
import pandas as pd
import torch

# Local imports
from src.models import MarketData
from src.exchanges.base import ExchangeConnector
```

## Key Architectural Components

### Exchange Connectors (`src/exchanges/`)
- Inherit from `ExchangeConnector` abstract base class
- Implement async methods for data retrieval and trading
- Handle exchange-specific authentication and rate limiting

### Data Models (`src/models/`)
- Use Pydantic for validation and serialization
- Include comprehensive field validation
- Support multiple exchange formats

### ML Components (`src/ml/`)
- Inherit from `BaseMLModel` or `BasePyTorchModel`
- Modular feature engineering pipeline
- Configurable model architectures

### Services (`src/services/`)
- Business logic orchestration
- Data aggregation across exchanges
- Trading decision engines

## File Naming Patterns

- **Models**: `{domain}_model.py` (e.g., `market_data.py`)
- **Connectors**: `{exchange}_connector.py` (e.g., `coinbase_connector.py`)
- **Services**: `{function}_service.py` (e.g., `data_aggregator.py`)
- **Tests**: `test_{component}.py` (e.g., `test_coinbase_connector.py`)
- **Examples**: `{feature}_demo.py` (e.g., `hybrid_model_demo.py`)

## Configuration Structure

### Settings Hierarchy
1. Default values in dataclasses
2. Configuration files (YAML/JSON)
3. Environment variables (highest priority)

### Environment-Specific Configs
- `config/settings.yaml`: Development defaults
- `config/production.yaml`: Production overrides
- Local configs ignored by git

## Testing Organization

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: Exchange connector testing
- **ML Tests**: Model training and prediction validation
- **End-to-End Tests**: Full pipeline testing

### Test Structure
- Mirror source structure in `tests/`
- Use fixtures for common test data
- Mock external dependencies (exchange APIs)