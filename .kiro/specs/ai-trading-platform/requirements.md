# Requirements Document

## Platform Goal
The AI Trading Platform delivers machine learning driven trade recommendations and portfolio management across equities, forex, and digital assets. The system combines CNN+LSTM feature extraction with reinforcement-learning ensembles, exposes orchestration services through FastAPI, and surfaces insights in a Next.js dashboard.

## Scope Snapshot
- **Delivered baseline**: Multi-exchange connectors, unified data aggregation with quality checks, an end-to-end CNN+LSTM hybrid plus RL training stack, portfolio and risk services, monitoring and interpretability modules, and deployment assets.
- **Stabilization in progress**: Production-friendly APIs (signal history, streaming, rate limiting), automated backups and retraining pipelines, experiment management, alert channel integrations, and secure secret storage.
- **Next focus**: Harden the real-time data plane, close outstanding TODO placeholders, document runbooks, and validate deployment workflows across Ray, Kubernetes, and the frontend.

## Functional Requirements

### Requirement 1: Market Data Foundation
- **Status**: Baseline delivered; streaming and retention hardening pending.
- **Delivered**
  - Async exchange connectors for Robinhood, OANDA, and Coinbase with historical and live polling interfaces (`src/exchanges/*`).
  - Aggregated ingestion with normalization, timestamp synchronization, and quality validation (`src/services/data_aggregator.py`).
  - Extensive validation tests covering multi-exchange scenarios (`tests/test_data_aggregator.py`).
- **Outstanding**
  - Real-time broadcasting from the aggregator into WebSocket clients and the frontend (`src/api/trading_endpoints.py` placeholders).
  - Durable signal/data persistence and automated backfills/archival through the storage service (`src/services/data_storage_service.py`).
  - Operational tooling for credential rotation and rate-limit tuning per exchange.

### Requirement 2: Feature Engineering & Dataset Management
- **Status**: Core library delivered; external data enrichment pending.
- **Delivered**
  - Comprehensive technical, volume, and volatility feature set plus advanced transforms (wavelet, Fourier, fractal, cross-asset scaffolding) in `src/ml/feature_engineering.py`.
  - Feature caching, dataset wrappers, and unit coverage in `tests/test_feature_extraction.py` and `tests/test_advanced_features.py`.
- **Outstanding**
  - Replace placeholder correlation/sentiment enrichments with live data sources and calibration logic (`CrossAssetFeatures`, sentiment adapters).
  - Dataset versioning hash generation in training metadata (`src/ml/training_pipeline.py` TODO).
  - Documentation of feature governance and refresh cadence.

### Requirement 3: Modeling & Training
- **Status**: CNN/LSTM hybrid and RL stack implemented; scalability tasks open.
- **Delivered**
  - CNN and bidirectional LSTM modules with attention, hybrid fusion, Monte Carlo dropout, and uncertainty calibration (`src/ml/*`).
  - Reinforcement-learning environments, agent ensemble management, and reward strategies (`src/ml/trading_environment.py`, `rl_agents.py`, `rl_ensemble.py`).
  - Distributed training orchestrator, Ray Tune integration, and automated retraining workflows (`src/ml/distributed_training.py`, `src/services/automated_retraining_service.py`).
- **Outstanding**
  - Parallel hyperparameter search on Ray (`src/ml/rl_hyperopt.py` TODO) and validation of Ray cluster topology in production.
  - End-to-end model evaluation hooks (placeholder score aggregation in `distributed_training.py`) and registry metadata hashing.
  - Governance workflows for promoting models into A/B testing and live serving.

### Requirement 4: Decisioning & Portfolio Services
- **Status**: Decision engine and risk tooling operational; analytics gaps remain.
- **Delivered**
  - TradingDecisionEngine combining ML outputs with RL ensemble policies, risk calculators, and position sizing (`src/services/trading_decision_engine.py`).
  - Portfolio optimization, rebalancing, and supporting calculators (`src/services/portfolio_management_service.py`, `portfolio_optimizer.py`, `portfolio_rebalancer.py`).
  - Risk management, stress testing, and monitoring services with alert callbacks (`src/services/risk_manager.py`, `risk_monitoring_service.py`).
  - Backtesting engine and audit trails (`src/services/backtesting_engine.py`, `audit_service.py`).
- **Outstanding**
  - Implement correlation-aware risk signals in the decision engine and align with analytics dashboards (TODO noted in `trading_decision_engine.py`).
  - Persisted trade/signal history, performance metrics, and attribution reports (placeholders in `src/api/trading_endpoints.py`).
  - Integration with broker execution flows for automated order placement and fail-safe handling.

### Requirement 5: API & User Experience
- **Status**: FastAPI surface and dashboard scaffolded; production polish pending.
- **Delivered**
  - Modular FastAPI app with auth, trading, risk, monitoring, model-serving, and usage endpoints (`src/api/*`).
  - WebSocket manager for push notifications and usage tracking middleware.
  - Next.js dashboard components for signals, portfolio views, and monitoring overlays (`frontend/app`, `frontend/components`).
- **Outstanding**
  - Redis-backed rate limiting and quota enforcement (`src/api/app.py` TODO).
  - Signal history/performance endpoints backed by storage, plus WebSocket real-time market data streaming (placeholders in `trading_endpoints.py`).
  - UX polish, onboarding flows, and documentation for non-technical traders.

### Requirement 6: Monitoring, Explainability & Governance
- **Status**: Monitoring and interpretability stack delivered; alert channel integrations pending.
- **Delivered**
  - Monitoring orchestrator, drift detector, performance tracker, and automated retraining triggers (`src/services/monitoring/*`, `model_monitoring_service.py`).
  - SHAP explanations, attention visualization, decision auditing, and uncertainty calibration modules (`src/ml/shap_explainer.py`, `attention_visualizer.py`, `decision_auditor.py`).
  - Usage and compliance auditing services with structured logging (`src/services/audit_service.py`, `usage_tracking_service.py`).
- **Outstanding**
  - Production-ready alert delivery (email/Slack integrations currently placeholders in `risk_monitoring_service.py`).
  - Compliance reporting pipelines and governance dashboards.
  - Automated documentation of explanation artefacts for regulators.

### Requirement 7: Infrastructure, Deployment & Operations
- **Status**: Deployment assets in place; resilience work pending.
- **Delivered**
  - Docker, docker-compose, Helm, and Kubernetes manifests (`docker/`, `helm/`, `k8s/`).
  - Connection pooling, caching, and background workers (`src/connection_pool`, `src/services/data_aggregator.py`).
  - CI/CD workflow definitions and local automation scripts (`.github/workflows`, `scripts/`).
  - Ray Serve A/B testing framework and tests (`src/ml/ray_serve/ab_testing.py`, `test_ab_testing.py`).
- **Outstanding**
  - Finish storage backup/restore automation (`src/services/data_storage_service.py` TODO) and model artefact backup in the retraining service.
  - Validate Ray Serve deployment in cluster environments and integrate with API controls.
  - Observability pipeline rollout (metrics/log shipping and on-call runbooks).

### Requirement 8: Security, Compliance & Quality
- **Status**: Security primitives and QA harness established; secret management pending.
- **Delivered**
  - Encryption/key management services, authentication middleware, and security monitoring utilities (`src/services/security_service.py`, `src/api/auth.py`, `src/utils/security.py`).
  - Config management, structured logging, and extensive unit/integration test suites (`src/config`, `tests/`).
  - Usage tracking, audit logs, and compliance scaffolding.
- **Outstanding**
  - Integrate secure key vault/secret storage (placeholder key persistence in `security_service.py`).
  - Expand penetration/resilience testing and document incident response procedures.
  - Complete documentation set (API guides, deployment runbooks, end-user manuals) and maintain CI coverage gates.

## Traceability
- Detailed architectural decisions live in `design.md`.
- Implementation roadmap with milestone status is tracked in `tasks.md`.
- TODO markers in source highlight remaining acceptance criteria; they align directly with the outstanding items above.
