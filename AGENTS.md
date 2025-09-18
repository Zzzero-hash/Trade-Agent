# Repository Guidelines

## Project Structure & Module Organization
- `src/main.py` exposes the FastAPI bootstrapper; route handlers stay in `src/api`, orchestration in `src/services`, ML artifacts under `src/ml`, with shared helpers in `src/repositories`, `src/utils`, and `src/connection_pool`.
- `tests/` mirrors backend packages with `test_*.py`, plus helpers in `tests/test_helpers/` and orchestration scripts like `tests/run_integration_tests.py`.
- `frontend/` hosts the Next.js dashboard (`app/`, `components/`, `__tests__/`), while deployment and automation assets live in `docker/`, `helm/`, `k8s/`, `config/`, and `scripts/`.

## Build, Test, and Development Commands
- `python -m venv .venv` then `pip install -r requirements.txt` to bootstrap backend dependencies.
- `uvicorn src.api.app:app --reload` starts the API with live reload.
- `pytest tests -v` for quick feedback; `pytest tests --cov=src --cov-report=html` before merging.
- `python tests/run_integration_tests.py` coordinates multi-service checks.
- Inside `frontend/`, run `npm install`, `npm run dev`, and `npm test` (or `npm run test:watch`) for UI development.
- `docker-compose up --build` spins up the full stack locally.

## Coding Style & Naming Conventions
- Follow PEP 8 with four-space indents and `snake_case` modules/functions; classes use `CamelCase`.
- Format Python via `black src tests`; lint with `flake8 src tests`; maintain `mypy src` cleanliness.
- TypeScript relies on `npm run lint`; components adopt PascalCase filenames, shared utilities live in `frontend/lib/`, and hooks use `useFeature.ts`.

## Testing Guidelines
- Place new suites beside the feature (e.g., `tests/test_risk_management_models.py`) and reuse fixtures from `tests/conftest.py`.
- Tag slow or external tests so they can be deselected; review coverage using `htmlcov/index.html`.
- Refresh frontend snapshots whenever UI contracts shift.

## Commit & Pull Request Guidelines
- Use conventional commits (`feat:`, `fix:`, `refactor:`) with imperative subjects under 72 characters.
- PRs should link issues, summarize impact, call out API or config changes, and attach relevant terminal output or UI screenshots.
- Verify CI locally by running formatters, linters, backend/frontend tests, and any config migrations before requesting review.

## Deployment & Configuration Notes
- Runtime settings live under `config/*.yaml`; update sample configs when toggling behavior.
- Container or Helm updates belong in `docker/`, `helm/`, and `k8s/`; keep scripts in `scripts/` executable and documented.
