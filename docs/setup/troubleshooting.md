# Troubleshooting Guide

This comprehensive guide helps you diagnose and resolve common issues with the AI Trading Platform.

## Quick Diagnostics

### Health Check Script

Run the built-in diagnostics to identify issues:

```bash
# Run comprehensive health check
python scripts/diagnose.py --verbose

# Check specific components
python scripts/diagnose.py --component database
python scripts/diagnose.py --component redis
python scripts/diagnose.py --component exchanges
python scripts/diagnose.py --component models
```

### System Status

Check overall system health:

```bash
# API health endpoint
curl http://localhost:8000/health

# Detailed system metrics
curl http://localhost:8000/api/v1/monitoring/metrics

# Service-specific health
curl http://localhost:8000/api/v1/monitoring/services
```

## Common Issues

### Installation and Setup Issues

#### Issue: Python Dependencies Conflict

**Symptoms:**
- Import errors when starting the application
- Version conflicts during pip install
- ModuleNotFoundError exceptions

**Solutions:**

1. **Clean Virtual Environment:**
   ```bash
   # Remove existing environment
   rm -rf .venv
   
   # Create fresh environment
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate     # Windows
   
   # Install dependencies
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Use Conda Environment:**
   ```bash
   # Create conda environment
   conda create -n ai-trading python=3.9
   conda activate ai-trading
   
   # Install PyTorch with CUDA support
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   
   # Install other dependencies
   pip install -r requirements.txt
   ```

3. **Check Python Version:**
   ```bash
   python --version  # Should be 3.9+
   
   # If using wrong version
   pyenv install 3.9.18
   pyenv local 3.9.18
   ```

#### Issue: Database Connection Failed

**Symptoms:**
- `psycopg2.OperationalError: could not connect to server`
- Application fails to start with database errors
- Connection timeout errors

**Solutions:**

1. **Check PostgreSQL Status:**
   ```bash
   # Linux/Mac
   sudo systemctl status postgresql
   sudo systemctl start postgresql
   
   # Windows
   net start postgresql-x64-14
   
   # Check if PostgreSQL is listening
   netstat -an | grep 5432
   ```

2. **Verify Database Configuration:**
   ```bash
   # Test connection manually
   psql -h localhost -p 5432 -U trading_user -d ai_trading_platform
   
   # Check environment variables
   echo $DATABASE_URL
   
   # Verify .env file
   cat .env | grep DATABASE
   ```

3. **Reset Database:**
   ```bash
   # Drop and recreate database
   sudo -u postgres psql
   DROP DATABASE IF EXISTS ai_trading_platform;
   CREATE DATABASE ai_trading_platform;
   GRANT ALL PRIVILEGES ON DATABASE ai_trading_platform TO trading_user;
   \q
   
   # Run migrations
   alembic upgrade head
   ```

#### Issue: Redis Connection Failed

**Symptoms:**
- `redis.exceptions.ConnectionError`
- Caching not working
- Session management issues

**Solutions:**

1. **Check Redis Status:**
   ```bash
   # Check if Redis is running
   redis-cli ping
   
   # Start Redis service
   sudo systemctl start redis-server  # Linux
   brew services start redis          # Mac
   
   # Check Redis configuration
   redis-cli config get "*"
   ```

2. **Verify Redis Configuration:**
   ```bash
   # Test connection
   redis-cli -h localhost -p 6379
   
   # Check memory usage
   redis-cli info memory
   
   # Clear Redis cache if needed
   redis-cli flushall
   ```

### API and Authentication Issues

#### Issue: JWT Token Expired or Invalid

**Symptoms:**
- 401 Unauthorized responses
- "Token has expired" errors
- Authentication failures

**Solutions:**

1. **Refresh Token:**
   ```bash
   # Get new token
   curl -X POST "http://localhost:8000/auth/login" \
     -H "Content-Type: application/json" \
     -d '{"username": "your_username", "password": "your_password"}'
   
   # Use refresh endpoint
   curl -X POST "http://localhost:8000/auth/refresh" \
     -H "Content-Type: application/json" \
     -d '{"refresh_token": "your_refresh_token"}'
   ```

2. **Check Token Configuration:**
   ```python
   # Verify JWT settings in config
   import jwt
   from datetime import datetime
   
   # Decode token to check expiry
   token = "your_jwt_token"
   decoded = jwt.decode(token, options={"verify_signature": False})
   exp_time = datetime.fromtimestamp(decoded['exp'])
   print(f"Token expires at: {exp_time}")
   ```

#### Issue: Rate Limiting Errors

**Symptoms:**
- 429 Too Many Requests responses
- "Rate limit exceeded" messages
- API calls being rejected

**Solutions:**

1. **Check Current Usage:**
   ```bash
   # Check usage limits
   curl -H "Authorization: Bearer YOUR_TOKEN" \
     "http://localhost:8000/api/v1/usage/current"
   ```

2. **Implement Exponential Backoff:**
   ```python
   import time
   import random
   
   def api_call_with_backoff(func, max_retries=5):
       for attempt in range(max_retries):
           try:
               return func()
           except RateLimitError:
               if attempt == max_retries - 1:
                   raise
               wait_time = (2 ** attempt) + random.uniform(0, 1)
               time.sleep(wait_time)
   ```

### Model Training Issues

#### Issue: CUDA Out of Memory

**Symptoms:**
- `RuntimeError: CUDA out of memory`
- Training crashes during forward/backward pass
- GPU memory errors

**Solutions:**

1. **Reduce Batch Size:**
   ```python
   # In training config
   training:
     batch_size: 16  # Reduce from 32
     gradient_accumulation_steps: 2  # Maintain effective batch size
   ```

2. **Enable Gradient Checkpointing:**
   ```python
   # In model configuration
   model = HybridCNNLSTM(
       use_checkpoint=True,  # Enable gradient checkpointing
       checkpoint_segments=4
   )
   ```

3. **Use Mixed Precision:**
   ```python
   # Enable automatic mixed precision
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   
   with autocast():
       outputs = model(inputs)
       loss = criterion(outputs, targets)
   
   scaler.scale(loss).backward()
   ```

4. **Clear GPU Cache:**
   ```python
   import torch
   
   # Clear GPU cache
   torch.cuda.empty_cache()
   
   # Monitor GPU memory
   print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
   print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
   ```

#### Issue: Model Not Converging

**Symptoms:**
- Loss not decreasing
- Validation metrics not improving
- Training appears stuck

**Solutions:**

1. **Check Learning Rate:**
   ```python
   # Try different learning rates
   learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
   
   for lr in learning_rates:
       optimizer = torch.optim.Adam(model.parameters(), lr=lr)
       # Train for a few epochs and check loss
   ```

2. **Verify Data Quality:**
   ```python
   # Check for NaN values
   import pandas as pd
   
   print(f"NaN values in features: {features.isna().sum().sum()}")
   print(f"Infinite values: {np.isinf(features.values).sum()}")
   
   # Check target distribution
   print(f"Target distribution:\n{targets.value_counts()}")
   ```

3. **Adjust Model Architecture:**
   ```python
   # Simplify model if overfitting
   model = HybridCNNLSTM(
       hidden_dim=128,  # Reduce from 256
       num_layers=1,    # Reduce from 2
       dropout=0.3      # Increase dropout
   )
   ```

### Exchange Connector Issues

#### Issue: Robinhood Authentication Failed

**Symptoms:**
- Login failures
- "Invalid credentials" errors
- 2FA challenges not handled

**Solutions:**

1. **Check Credentials:**
   ```python
   # Verify credentials in environment
   import os
   
   username = os.getenv('ROBINHOOD_USERNAME')
   password = os.getenv('ROBINHOOD_PASSWORD')
   
   print(f"Username configured: {username is not None}")
   print(f"Password configured: {password is not None}")
   ```

2. **Handle 2FA:**
   ```python
   # Enable 2FA handling
   from src.exchanges.robinhood_connector import RobinhoodConnector
   
   connector = RobinhoodConnector(
       username=username,
       password=password,
       enable_2fa=True,  # Enable 2FA support
       device_token="your_device_token"
   )
   ```

3. **Use Paper Trading:**
   ```python
   # Switch to paper trading for testing
   connector = RobinhoodConnector(
       paper_trading=True,  # Use paper trading mode
       username=username,
       password=password
   )
   ```

#### Issue: OANDA API Rate Limits

**Symptoms:**
- HTTP 429 responses from OANDA
- "Rate limit exceeded" errors
- API calls being throttled

**Solutions:**

1. **Implement Rate Limiting:**
   ```python
   from src.utils.rate_limiter import RateLimiter
   
   # Create rate limiter for OANDA (120 requests/minute)
   rate_limiter = RateLimiter(max_calls=120, time_window=60)
   
   @rate_limiter.limit
   def make_oanda_request():
       return oanda_client.get_prices(instruments="EUR_USD")
   ```

2. **Use Streaming API:**
   ```python
   # Use streaming for real-time data instead of polling
   from src.exchanges.oanda_connector import OANDAStreamingConnector
   
   streaming_connector = OANDAStreamingConnector(
       api_key=api_key,
       account_id=account_id,
       instruments=["EUR_USD", "GBP_USD"]
   )
   
   streaming_connector.start_stream()
   ```

### Performance Issues

#### Issue: Slow API Response Times

**Symptoms:**
- API responses taking >1 second
- Timeout errors
- Poor user experience

**Solutions:**

1. **Enable Caching:**
   ```python
   # Configure Redis caching
   from src.utils.cache import CacheManager
   
   cache = CacheManager(
       redis_url="redis://localhost:6379/0",
       default_ttl=300  # 5 minutes
   )
   
   @cache.cached(ttl=60)
   def get_trading_signals(symbol):
       return expensive_signal_calculation(symbol)
   ```

2. **Database Query Optimization:**
   ```sql
   -- Add indexes for common queries
   CREATE INDEX idx_signals_symbol_timestamp ON trading_signals(symbol, timestamp);
   CREATE INDEX idx_portfolio_user_id ON portfolio_positions(user_id);
   
   -- Analyze query performance
   EXPLAIN ANALYZE SELECT * FROM trading_signals WHERE symbol = 'AAPL';
   ```

3. **Use Connection Pooling:**
   ```python
   # Configure connection pooling
   from sqlalchemy import create_engine
   from sqlalchemy.pool import QueuePool
   
   engine = create_engine(
       database_url,
       poolclass=QueuePool,
       pool_size=20,
       max_overflow=30,
       pool_pre_ping=True
   )
   ```

#### Issue: High Memory Usage

**Symptoms:**
- Application consuming excessive RAM
- Out of memory errors
- System becoming unresponsive

**Solutions:**

1. **Profile Memory Usage:**
   ```python
   import psutil
   import tracemalloc
   
   # Start memory profiling
   tracemalloc.start()
   
   # Your code here
   
   # Get memory statistics
   current, peak = tracemalloc.get_traced_memory()
   print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
   print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
   ```

2. **Optimize Data Loading:**
   ```python
   # Use chunked data loading
   def load_data_in_chunks(file_path, chunk_size=10000):
       for chunk in pd.read_csv(file_path, chunksize=chunk_size):
           yield process_chunk(chunk)
   
   # Clear unused variables
   del large_dataframe
   import gc
   gc.collect()
   ```

3. **Configure Garbage Collection:**
   ```python
   import gc
   
   # Tune garbage collection
   gc.set_threshold(700, 10, 10)  # More aggressive GC
   
   # Force garbage collection periodically
   if iteration % 100 == 0:
       gc.collect()
   ```

## Debugging Tools

### Logging Configuration

```python
# Configure detailed logging
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

# Enable SQL query logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

### Performance Profiling

```python
# Profile function performance
import cProfile
import pstats

def profile_function(func):
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = func()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    
    return result
```

### Database Debugging

```sql
-- Monitor active connections
SELECT pid, usename, application_name, client_addr, state, query_start, query
FROM pg_stat_activity
WHERE state = 'active';

-- Check slow queries
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;

-- Monitor table sizes
SELECT schemaname, tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

## Frequently Asked Questions (FAQ)

### General Questions

**Q: What are the minimum system requirements?**

A: 
- **RAM**: 8GB minimum, 16GB+ recommended
- **CPU**: Multi-core processor (4+ cores recommended)
- **Storage**: 10GB free space for installation, 50GB+ for data
- **GPU**: Optional but recommended for ML training (NVIDIA with CUDA support)
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10+

**Q: Can I run the platform without GPU support?**

A: Yes, the platform works with CPU-only mode. Set `ML__DEVICE=cpu` in your environment variables. Training will be slower but functional.

**Q: How do I update to a new version?**

A:
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Run database migrations
alembic upgrade head

# Restart services
python src/main.py
```

### Configuration Questions

**Q: How do I configure multiple exchanges?**

A: Edit your configuration file or set environment variables:

```yaml
exchanges:
  robinhood:
    enabled: true
    paper_trading: true
    username: "your_username"
    password: "your_password"
  
  oanda:
    enabled: true
    environment: "practice"
    api_key: "your_api_key"
    account_id: "your_account_id"
  
  coinbase:
    enabled: false  # Disable if not needed
    sandbox: true
    api_key: "your_api_key"
```

**Q: How do I change the default port?**

A: Set the `API__PORT` environment variable or update `config/settings.yaml`:

```bash
export API__PORT=9000
# or
echo "API__PORT=9000" >> .env
```

**Q: Can I use a different database?**

A: The platform is designed for PostgreSQL, but you can configure other databases by updating the `DATABASE_URL`. Note that some features may require PostgreSQL-specific functionality.

### Trading Questions

**Q: Is my money safe? Are trades executed automatically?**

A: By default, the platform runs in paper trading mode and does not execute real trades. You must explicitly enable live trading and provide real API credentials. Always test thoroughly in paper trading mode first.

**Q: How accurate are the trading signals?**

A: Signal accuracy varies based on market conditions, model training, and configuration. Historical backtesting shows typical accuracy ranges of 55-75%, but past performance doesn't guarantee future results.

**Q: Can I customize the trading strategies?**

A: Yes, you can:
- Modify model architectures in `src/ml/`
- Adjust risk parameters in configuration files
- Create custom feature engineering pipelines
- Implement custom reward functions for RL agents

### Technical Questions

**Q: How do I backup my data?**

A:
```bash
# Backup database
pg_dump ai_trading_platform > backup_$(date +%Y%m%d).sql

# Backup model checkpoints
tar -czf models_backup_$(date +%Y%m%d).tar.gz checkpoints/

# Backup configuration
cp -r config/ config_backup_$(date +%Y%m%d)/
```

**Q: How do I monitor system performance?**

A: Use the built-in monitoring endpoints:
- Health: `GET /health`
- Metrics: `GET /api/v1/monitoring/metrics`
- System status: `GET /api/v1/monitoring/system`

You can also integrate with external monitoring tools like Prometheus and Grafana.

**Q: How do I scale the platform for production?**

A: See the [Cloud Deployment Guide](../deployment/cloud-deployment.md) for:
- Kubernetes deployment
- Horizontal pod autoscaling
- Load balancing
- Database clustering
- Redis clustering

### Model Training Questions

**Q: How long does model training take?**

A: Training time depends on:
- **Dataset size**: 1-5 years of data typically
- **Model complexity**: CNN+LSTM hybrid takes 2-8 hours
- **Hardware**: GPU training is 5-10x faster than CPU
- **Hyperparameter optimization**: Can take 12-48 hours

**Q: How often should I retrain models?**

A: The platform includes automated retraining triggers based on:
- Model drift detection (recommended: weekly checks)
- Performance degradation (recommended: <5% accuracy drop)
- New data availability (recommended: monthly retraining)

**Q: Can I use my own data sources?**

A: Yes, implement a custom exchange connector by extending the `ExchangeConnector` base class in `src/exchanges/base.py`.

## Getting Help

### Self-Service Resources

1. **Check Logs**: Application logs are in `logs/` directory
2. **Run Diagnostics**: `python scripts/diagnose.py --verbose`
3. **Check Configuration**: `python scripts/check_config.py`
4. **Health Check**: `curl http://localhost:8000/health`

### Community Support

- **Documentation**: [docs.ai-trading-platform.com](https://docs.ai-trading-platform.com)
- **GitHub Issues**: [Report bugs and request features](https://github.com/your-org/ai-trading-platform/issues)
- **Discord**: [Join our community](https://discord.gg/ai-trading-platform)
- **Stack Overflow**: Tag questions with `ai-trading-platform`

### Emergency Procedures

For production issues:

```bash
# Emergency rollback
./scripts/rollback.sh emergency

# Stop all services
docker-compose down

# Check system resources
htop
df -h
free -h

# Restart with safe mode
SAFE_MODE=true python src/main.py
```

### Contact Information

- **Technical Support**: support@ai-trading-platform.com
- **Security Issues**: security@ai-trading-platform.com
- **Business Inquiries**: business@ai-trading-platform.com

**Response Times:**
- Critical issues: 2-4 hours
- High priority: 24 hours
- Normal priority: 48-72 hours