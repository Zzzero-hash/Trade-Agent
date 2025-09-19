# ML Trading Ensemble Makefile

.PHONY: setup install test clean lint format gpu-test

# Setup and Installation
setup:
	python setup_environment.py

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Development
test:
	pytest tests/ -v

lint:
	flake8 models/ experiments/ data/ utils/
	mypy models/ experiments/ data/ utils/

format:
	black models/ experiments/ data/ utils/ configs/
	isort models/ experiments/ data/ utils/

# GPU and Performance
gpu-test:
	python -c "from utils.gpu_utils import get_device_info; print(get_device_info())"

gpu-benchmark:
	python -c "import torch; x=torch.randn(1000,1000).cuda(); print('GPU benchmark:', torch.matmul(x,x).sum())"

# Experiment Management
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

clean-experiments:
	rm -rf mlruns/
	rm -rf wandb/
	rm -rf outputs/

# Data Management
clean-data:
	rm -rf data/raw/*
	rm -rf data/processed/*
	rm -rf data/features/*

# Model Management
clean-models:
	rm -rf models/checkpoints/*
	rm -rf models/saved/*

# Complete cleanup
clean: clean-experiments clean-data clean-models
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.log" -delete

# Documentation
docs:
	sphinx-build -b html docs/ docs/_build/

# Docker (if needed)
docker-build:
	docker build -t ml-trading-ensemble .

docker-run:
	docker run --gpus all -it ml-trading-ensemble

# Help
help:
	@echo "Available commands:"
	@echo "  setup          - Run complete environment setup"
	@echo "  install        - Install dependencies"
	@echo "  install-dev    - Install with development dependencies"
	@echo "  test           - Run tests"
	@echo "  lint           - Run linting"
	@echo "  format         - Format code"
	@echo "  gpu-test       - Test GPU availability"
	@echo "  mlflow-ui      - Start MLflow UI"
	@echo "  clean          - Clean all generated files"
	@echo "  help           - Show this help message"