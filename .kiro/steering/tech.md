# Technology Stack

## Core Technologies

- **Python 3.9+**: Primary development language
- **PyTorch**: Deep learning framework for CNN+LSTM models
- **FastAPI**: REST API framework with async support
- **Pydantic**: Data validation and serialization
- **SQLAlchemy**: Database ORM with PostgreSQL
- **Redis**: Caching and session management
- **Ray**: Distributed computing and ML training

## Machine Learning Stack

- **PyTorch**: Neural network implementation
- **Stable Baselines3**: Reinforcement learning algorithms
- **NumPy/Pandas**: Data manipulation and analysis
- **Scikit-learn**: Traditional ML algorithms and preprocessing

## Data & Infrastructure

- **PostgreSQL**: Primary database for structured data
- **Redis**: Real-time data caching and pub/sub
- **WebSockets**: Real-time market data streaming
- **HTTPX/Requests**: HTTP client for exchange APIs

## Development Tools

- **pytest**: Testing framework with async support
- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking

## Configuration Management

- **YAML/JSON**: Configuration files in `config/` directory
- **Environment Variables**: Override any configuration setting
- **Hierarchical Config**: Environment-specific settings (dev/staging/prod)

## Common Commands

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run application
python src/main.py

# Set configuration
export CONFIG_FILE=config/local.yaml
```

### Testing
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test categories
pytest tests/test_*_connector.py  # Exchange tests
pytest tests/test_*_model.py      # ML model tests
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Build System

- **pip**: Package management via requirements.txt
- **Virtual Environment**: Use `.venv/` for isolation
- **No complex build system**: Direct Python execution
