# Contributing to AI Trading Platform

Thank you for your interest in contributing to the AI Trading Platform! This guide will help you get started with contributing to our open-source project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Code Style and Standards](#code-style-and-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

### Our Pledge

We are committed to making participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Positive behavior includes:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes:**
- The use of sexualized language or imagery
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information without explicit permission
- Other conduct which could reasonably be considered inappropriate in a professional setting

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team at conduct@ai-trading-platform.com. All complaints will be reviewed and investigated promptly and fairly.

## Getting Started

### Ways to Contribute

There are many ways to contribute to the AI Trading Platform:

1. **Code Contributions**
   - Bug fixes
   - New features
   - Performance improvements
   - Code refactoring

2. **Documentation**
   - API documentation improvements
   - Tutorial creation
   - Translation
   - Example code

3. **Testing**
   - Writing test cases
   - Manual testing
   - Performance testing
   - Security testing

4. **Design and UX**
   - UI/UX improvements
   - Dashboard enhancements
   - Visualization improvements

5. **Community Support**
   - Answering questions in discussions
   - Helping with issue triage
   - Code reviews
   - Mentoring new contributors

### Prerequisites

Before contributing, ensure you have:

- **Git**: Version control system
- **Python 3.9+**: Programming language
- **Node.js 16+**: For frontend development
- **PostgreSQL**: Database system
- **Redis**: Caching system
- **Docker**: For containerized development (optional)

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/ai-trading-platform.git
cd ai-trading-platform

# Add upstream remote
git remote add upstream https://github.com/original-org/ai-trading-platform.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Install pre-commit hooks
pre-commit install
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your configuration
nano .env
```

### 4. Set Up Database

```bash
# Start PostgreSQL and Redis
sudo systemctl start postgresql redis-server

# Create database
createdb ai_trading_platform

# Run migrations
alembic upgrade head
```

### 5. Verify Setup

```bash
# Run tests to verify setup
pytest tests/ -v

# Start development server
python src/main.py

# Check health endpoint
curl http://localhost:8000/health
```

## Contributing Guidelines

### Branch Naming Convention

Use descriptive branch names that follow this pattern:

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `hotfix/description` - Critical fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test improvements

**Examples:**
```bash
git checkout -b feature/add-binance-connector
git checkout -b bugfix/fix-memory-leak-in-data-aggregator
git checkout -b docs/update-api-documentation
```

### Commit Message Format

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(exchanges): add Binance connector with WebSocket support

- Implement real-time data streaming
- Add authentication and rate limiting
- Include comprehensive error handling

Closes #123
```

```
fix(ml): resolve memory leak in feature extraction

The feature cache was not properly clearing expired entries,
causing memory usage to grow indefinitely.

Fixes #456
```

### Development Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run all tests
   pytest tests/
   
   # Run specific test categories
   pytest tests/test_exchanges/ -v
   pytest tests/test_ml/ -v
   
   # Run linting and formatting
   make lint
   make format
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Pull Request Process

### Before Submitting

- [ ] Code follows project style guidelines
- [ ] All tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated if needed
- [ ] Commit messages follow conventional format
- [ ] Branch is up to date with main

### PR Template

When creating a pull request, use this template:

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced

## Related Issues
Closes #issue_number
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs automatically
2. **Code Review**: At least one maintainer reviews the code
3. **Testing**: Automated and manual testing
4. **Approval**: Maintainer approval required for merge
5. **Merge**: Squash and merge to main branch

### Review Criteria

Reviewers will check for:

- **Functionality**: Does the code work as intended?
- **Code Quality**: Is the code clean, readable, and maintainable?
- **Testing**: Are there adequate tests with good coverage?
- **Documentation**: Is the code properly documented?
- **Performance**: Are there any performance implications?
- **Security**: Are there any security concerns?

## Issue Guidelines

### Reporting Bugs

Use the bug report template:

```markdown
**Bug Description**
A clear and concise description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
A clear description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment:**
- OS: [e.g. Ubuntu 20.04]
- Python Version: [e.g. 3.9.7]
- Platform Version: [e.g. v1.2.3]

**Additional Context**
Add any other context about the problem here.
```

### Feature Requests

Use the feature request template:

```markdown
**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Additional context**
Any other context or screenshots about the feature request.
```

### Issue Labels

We use labels to categorize issues:

- **Type**: `bug`, `enhancement`, `documentation`, `question`
- **Priority**: `low`, `medium`, `high`, `critical`
- **Status**: `needs-triage`, `in-progress`, `blocked`, `ready-for-review`
- **Component**: `api`, `ml`, `frontend`, `exchanges`, `docs`
- **Difficulty**: `good-first-issue`, `help-wanted`, `advanced`

## Code Style and Standards

### Python Code Style

We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# Use Black for formatting
black src/ tests/

# Use flake8 for linting
flake8 src/ tests/

# Use mypy for type checking
mypy src/
```

### Code Formatting Configuration

**.flake8**
```ini
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git,__pycache__,docs/source/conf.py,old,build,dist
```

**pyproject.toml**
```toml
[tool.black]
line-length = 88
target-version = ['py39']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''
```

### Type Hints

Use type hints for all function signatures:

```python
from typing import List, Dict, Optional, Union
from datetime import datetime

def process_market_data(
    data: List[Dict[str, Union[float, str]]],
    start_date: datetime,
    end_date: Optional[datetime] = None
) -> Dict[str, float]:
    """Process market data and return aggregated metrics."""
    # Implementation here
    pass
```

### Documentation Strings

Use Google-style docstrings:

```python
def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """Calculate the Sharpe ratio for a series of returns.
    
    Args:
        returns: List of portfolio returns
        risk_free_rate: Risk-free rate for calculation (default: 0.02)
        
    Returns:
        The calculated Sharpe ratio
        
    Raises:
        ValueError: If returns list is empty
        
    Example:
        >>> returns = [0.1, 0.05, -0.02, 0.08]
        >>> sharpe = calculate_sharpe_ratio(returns)
        >>> print(f"Sharpe ratio: {sharpe:.3f}")
    """
    if not returns:
        raise ValueError("Returns list cannot be empty")
    
    # Implementation here
    pass
```

### Error Handling

Use specific exception types and proper error messages:

```python
from src.exceptions import ExchangeConnectionError, DataValidationError

class RobinhoodConnector:
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        try:
            response = self._make_request(f"/quotes/{symbol}")
        except requests.RequestException as e:
            raise ExchangeConnectionError(
                f"Failed to fetch data for {symbol}: {str(e)}"
            ) from e
        
        if not self._validate_response(response):
            raise DataValidationError(
                f"Invalid response format for {symbol}"
            )
        
        return response.json()
```

### Logging

Use structured logging with appropriate levels:

```python
import logging
from src.utils.logging import get_logger

logger = get_logger(__name__)

class DataAggregator:
    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(
            "Processing market data",
            extra={
                "symbol": data.get("symbol"),
                "timestamp": data.get("timestamp"),
                "data_size": len(data)
            }
        )
        
        try:
            result = self._process_internal(data)
            logger.debug("Data processing completed successfully")
            return result
        except Exception as e:
            logger.error(
                "Data processing failed",
                extra={"error": str(e), "symbol": data.get("symbol")},
                exc_info=True
            )
            raise
```

## Testing Requirements

### Test Structure

Organize tests to mirror the source structure:

```
tests/
├── test_api/
│   ├── test_auth.py
│   ├── test_trading_endpoints.py
│   └── test_websocket.py
├── test_exchanges/
│   ├── test_robinhood_connector.py
│   ├── test_oanda_connector.py
│   └── test_coinbase_connector.py
├── test_ml/
│   ├── test_cnn_model.py
│   ├── test_lstm_model.py
│   └── test_hybrid_model.py
└── conftest.py
```

### Test Categories

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Test performance requirements

### Writing Tests

Use pytest with clear, descriptive test names:

```python
import pytest
from unittest.mock import Mock, patch
from src.exchanges.robinhood_connector import RobinhoodConnector

class TestRobinhoodConnector:
    @pytest.fixture
    def connector(self):
        return RobinhoodConnector(
            username="test_user",
            password="test_pass",
            paper_trading=True
        )
    
    def test_get_market_data_success(self, connector):
        """Test successful market data retrieval."""
        with patch.object(connector, '_make_request') as mock_request:
            mock_request.return_value.json.return_value = {
                'symbol': 'AAPL',
                'price': 150.0,
                'volume': 1000000
            }
            
            result = connector.get_market_data('AAPL')
            
            assert result['symbol'] == 'AAPL'
            assert result['price'] == 150.0
            mock_request.assert_called_once()
    
    def test_get_market_data_connection_error(self, connector):
        """Test handling of connection errors."""
        with patch.object(connector, '_make_request') as mock_request:
            mock_request.side_effect = requests.ConnectionError("Network error")
            
            with pytest.raises(ExchangeConnectionError):
                connector.get_market_data('AAPL')
    
    @pytest.mark.parametrize("symbol,expected_url", [
        ("AAPL", "/quotes/AAPL"),
        ("GOOGL", "/quotes/GOOGL"),
        ("TSLA", "/quotes/TSLA"),
    ])
    def test_url_construction(self, connector, symbol, expected_url):
        """Test URL construction for different symbols."""
        with patch.object(connector, '_make_request') as mock_request:
            connector.get_market_data(symbol)
            mock_request.assert_called_with(expected_url)
```

### Test Coverage

Maintain high test coverage:

```bash
# Run tests with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Coverage requirements
# - Overall coverage: >90%
# - New code coverage: >95%
# - Critical paths: 100%
```

### Mocking External Dependencies

Mock external services and APIs:

```python
@pytest.fixture
def mock_exchange_api():
    with patch('src.exchanges.base.requests.Session') as mock_session:
        mock_response = Mock()
        mock_response.json.return_value = {'status': 'success'}
        mock_response.status_code = 200
        mock_session.return_value.get.return_value = mock_response
        yield mock_session

def test_api_integration(mock_exchange_api):
    """Test API integration with mocked external service."""
    connector = ExchangeConnector()
    result = connector.fetch_data()
    
    assert result['status'] == 'success'
    mock_exchange_api.return_value.get.assert_called_once()
```

## Documentation

### Code Documentation

- **Docstrings**: All public functions and classes must have docstrings
- **Type Hints**: Use type hints for all function signatures
- **Comments**: Explain complex logic and business rules
- **README**: Update README files for new components

### API Documentation

- **OpenAPI**: Keep OpenAPI specification up to date
- **Examples**: Provide working code examples
- **Error Codes**: Document all error responses
- **Rate Limits**: Document rate limiting behavior

### User Documentation

- **Tutorials**: Step-by-step guides for common tasks
- **How-to Guides**: Solution-oriented documentation
- **Reference**: Complete API and configuration reference
- **Troubleshooting**: Common issues and solutions

### Documentation Standards

```markdown
# Use clear, descriptive headings

## Structure content logically

### Use code blocks with syntax highlighting

```python
# Example code should be complete and runnable
from src.api.client import TradingClient

client = TradingClient(api_key="your_key")
signals = client.get_signals(symbol="AAPL")
```

**Use formatting for emphasis**

- Use bullet points for lists
- Use tables for structured data
- Include screenshots for UI elements
```

## Community

### Communication Channels

- **GitHub Discussions**: General questions and discussions
- **GitHub Issues**: Bug reports and feature requests
- **Discord**: Real-time chat and community support
- **Stack Overflow**: Technical questions (tag: `ai-trading-platform`)

### Community Guidelines

1. **Be Respectful**: Treat all community members with respect
2. **Be Helpful**: Share knowledge and help others learn
3. **Be Patient**: Remember that everyone has different experience levels
4. **Be Constructive**: Provide actionable feedback and suggestions
5. **Be Inclusive**: Welcome newcomers and diverse perspectives

### Getting Help

1. **Search First**: Check existing issues and documentation
2. **Provide Context**: Include relevant details in your questions
3. **Use Templates**: Use issue templates for bug reports and feature requests
4. **Be Specific**: Provide minimal reproducible examples
5. **Follow Up**: Update issues with additional information as needed

### Mentorship Program

We offer mentorship for new contributors:

- **Mentee Application**: Apply to be paired with an experienced contributor
- **Mentor Volunteer**: Experienced contributors can volunteer to mentor
- **Good First Issues**: Issues labeled for newcomers
- **Pair Programming**: Scheduled sessions for complex contributions

### Recognition

We recognize contributors through:

- **Contributors Page**: Listed on our website and documentation
- **Release Notes**: Contributions acknowledged in release notes
- **Swag**: Stickers and merchandise for significant contributions
- **Conference Opportunities**: Speaking opportunities at events

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Schedule

- **Major Releases**: Quarterly
- **Minor Releases**: Monthly
- **Patch Releases**: As needed for critical fixes

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version numbers bumped
- [ ] Release notes prepared
- [ ] Security review completed

## Security

### Reporting Security Issues

**DO NOT** report security vulnerabilities through public GitHub issues.

Instead, email security@ai-trading-platform.com with:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Security Best Practices

- **Input Validation**: Validate all user inputs
- **Authentication**: Use secure authentication methods
- **Authorization**: Implement proper access controls
- **Encryption**: Encrypt sensitive data in transit and at rest
- **Dependencies**: Keep dependencies up to date
- **Secrets**: Never commit secrets to version control

## License

By contributing to the AI Trading Platform, you agree that your contributions will be licensed under the MIT License.

## Questions?

If you have questions about contributing, please:

1. Check the [FAQ](../setup/troubleshooting.md#frequently-asked-questions-faq)
2. Search [GitHub Discussions](https://github.com/your-org/ai-trading-platform/discussions)
3. Ask in our [Discord community](https://discord.gg/ai-trading-platform)
4. Email us at contributors@ai-trading-platform.com

Thank you for contributing to the AI Trading Platform! 🚀