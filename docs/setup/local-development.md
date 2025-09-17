# Local Development Setup

This guide walks you through setting up the AI Trading Platform for local development.

## Prerequisites

### System Requirements
- **Python**: 3.9 or higher
- **Node.js**: 16.x or higher (for frontend development)
- **PostgreSQL**: 12 or higher
- **Redis**: 6.0 or higher
- **Git**: Latest version

### Hardware Requirements
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: 10GB free space
- **CPU**: Multi-core processor (for ML training)
- **GPU**: Optional but recommended for faster model training

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/ai-trading-platform.git
cd ai-trading-platform
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Set Up Database

#### PostgreSQL Setup

```bash
# Install PostgreSQL (Ubuntu/Debian)
sudo apt-get install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql
CREATE DATABASE ai_trading_platform;
CREATE USER trading_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE ai_trading_platform TO trading_user;
\q
```

#### Redis Setup

```bash
# Install Redis (Ubuntu/Debian)
sudo apt-get install redis-server

# Start Redis service
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

### 4. Configure Environment

Create a `.env` file in the project root:

```bash
# Database Configuration
DATABASE_URL=postgresql://trading_user:your_password@localhost:5432/ai_trading_platform
REDIS_URL=redis://localhost:6379/0

# API Configuration
SECRET_KEY=your-secret-key-here
DEBUG=true
LOG_LEVEL=INFO

# Exchange API Keys (for testing)
ROBINHOOD_USERNAME=your_username
ROBINHOOD_PASSWORD=your_password
OANDA_API_KEY=your_oanda_key
OANDA_ACCOUNT_ID=your_account_id
COINBASE_API_KEY=your_coinbase_key
COINBASE_API_SECRET=your_coinbase_secret
COINBASE_PASSPHRASE=your_passphrase

# ML Configuration
MODEL_CACHE_DIR=./models
FEATURE_CACHE_SIZE=1000
BATCH_SIZE=32
```

### 5. Initialize Database

```bash
# Run database migrations
python -m alembic upgrade head

# Load initial data (optional)
python scripts/load_sample_data.py
```

### 6. Start the Application

```bash
# Start the main application
python src/main.py

# Or use uvicorn directly
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 7. Set Up Frontend (Optional)

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Development Workflow

### Code Quality Tools

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Run all quality checks
make lint
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

# Run integration tests
pytest tests/test_integration.py -v
```

### Database Management

```bash
# Create new migration
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1

# Reset database (development only)
python scripts/reset_database.py
```

## Configuration Management

### Environment-Specific Configs

The platform uses hierarchical configuration:

1. **Default values**: Defined in `src/config/settings.py`
2. **Config files**: `config/settings.yaml` (development)
3. **Environment variables**: Override any setting

### Configuration Structure

```yaml
# config/settings.yaml
database:
  url: "postgresql://localhost:5432/ai_trading_platform"
  pool_size: 10
  echo: false

redis:
  url: "redis://localhost:6379/0"
  max_connections: 20

api:
  host: "0.0.0.0"
  port: 8000
  debug: true
  cors_origins: ["http://localhost:3000"]

ml:
  model_cache_dir: "./models"
  feature_cache_size: 1000
  batch_size: 32
  device: "auto"  # auto, cpu, cuda

exchanges:
  robinhood:
    paper_trading: true
    rate_limit: 100  # requests per minute
  
  oanda:
    environment: "practice"  # practice or live
    rate_limit: 120
  
  coinbase:
    sandbox: true
    rate_limit: 10
```

### Environment Variable Overrides

Any configuration can be overridden with environment variables:

```bash
# Override database URL
export DATABASE_URL="postgresql://user:pass@localhost/db"

# Override ML batch size
export ML__BATCH_SIZE=64

# Override exchange settings
export EXCHANGES__ROBINHOOD__PAPER_TRADING=false
```

## Development Tools

### VS Code Setup

Recommended VS Code extensions:

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.black-formatter",
    "ms-python.flake8",
    "ms-python.mypy-type-checker",
    "ms-vscode.vscode-json",
    "redhat.vscode-yaml"
  ]
}
```

VS Code settings (`.vscode/settings.json`):

```json
{
  "python.defaultInterpreterPath": "./.venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/"]
}
```

### Docker Development (Alternative)

For containerized development:

```bash
# Build development image
docker-compose -f docker-compose.dev.yml build

# Start all services
docker-compose -f docker-compose.dev.yml up

# Run tests in container
docker-compose -f docker-compose.dev.yml exec api pytest tests/
```

## Debugging

### Application Debugging

```python
# Add breakpoints in code
import pdb; pdb.set_trace()

# Or use ipdb for better experience
import ipdb; ipdb.set_trace()
```

### Database Debugging

```bash
# Connect to database
psql postgresql://trading_user:password@localhost:5432/ai_trading_platform

# View logs
tail -f /var/log/postgresql/postgresql-*.log
```

### Redis Debugging

```bash
# Connect to Redis
redis-cli

# Monitor commands
redis-cli monitor

# View memory usage
redis-cli info memory
```

## Performance Optimization

### Development Performance

```bash
# Use faster test database
export TEST_DATABASE_URL="sqlite:///test.db"

# Enable parallel testing
pytest tests/ -n auto

# Skip slow tests during development
pytest tests/ -m "not slow"
```

### ML Development

```bash
# Use smaller datasets for faster iteration
export ML__SAMPLE_SIZE=1000

# Use CPU for debugging (faster startup)
export ML__DEVICE=cpu

# Enable model caching
export ML__CACHE_MODELS=true
```

## Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>
```

#### Database Connection Issues
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check connection
pg_isready -h localhost -p 5432
```

#### Redis Connection Issues
```bash
# Check Redis status
sudo systemctl status redis-server

# Test connection
redis-cli ping
```

#### Python Import Issues
```bash
# Ensure virtual environment is activated
which python

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Getting Help

1. **Check logs**: Application logs are in `logs/` directory
2. **Run diagnostics**: `python scripts/diagnose.py`
3. **Check configuration**: `python scripts/check_config.py`
4. **Reset environment**: `make clean && make setup`

## Next Steps

Once your development environment is set up:

1. **Explore the codebase**: Start with `src/main.py`
2. **Run the test suite**: Ensure everything works
3. **Try the examples**: Check out `examples/` directory
4. **Read the architecture docs**: Understand the system design
5. **Make your first change**: Follow the contributing guide

For production deployment, see the [Cloud Deployment Guide](../deployment/cloud-deployment.md).