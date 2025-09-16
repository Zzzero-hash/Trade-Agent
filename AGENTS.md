# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains the FastAPI entrypoint `main.py` plus domain packages: `api/` (routes), `services/` (orchestration), `ml/` (models), and supporting modules in `repositories/`, `utils/`, and `connection_pool/`.
- `tests/` mirrors backend modules with `test_*.py`, helpers in `tests/test_helpers/`, and orchestration scripts like `run_integration_tests.py`.
- `frontend/` provides the Next.js dashboard (TypeScript under `app/` and `components/`, Jest setup in `__tests__/`).
- Deployment assets live in `docker/`, `helm/`, and `k8s/`; runtime configuration stays in `config/*.yaml`; automation scripts (Bash and PowerShell) live in `scripts/`.

## Build, Test, and Development Commands
- Backend setup: `python -m venv .venv`, `pip install -r requirements.txt`, then `uvicorn src.api.app:app --reload`.
- Quality gates: `black src tests`, `flake8 src tests`, and `mypy src` match `.github/workflows/ci-cd.yml`.
- Tests: `pytest tests -v` for quick checks, `pytest tests --cov=src --cov-report=html` before merging, `python tests/run_integration_tests.py` for service coordination.
- Frontend: inside `frontend/`, run `npm install`, `npm run dev`, and `npm test` or `npm run test:watch`.
- Full stack validation: `docker-compose up --build` runs API, data services, and monitoring locally.

## Coding Style & Naming Conventions
- Use PEP 8, four-space indentation, `snake_case` modules and functions, `CamelCase` classes, and keep async endpoints grouped in `src/api`.
- Run `black` before commits, keep `flake8` clean, and resolve `mypy` warnings; notebooks or experiments stay in `docs/` or `checkpoints/`.
- TypeScript relies on `npm run lint`; React components use PascalCase filenames, hooks follow `useFeature.ts`, and shared utilities belong in `frontend/lib/`.

## Testing Guidelines
- Store new backend suites alongside features in `tests/` with explicit names (`test_risk_management_models.py`), reusing fixtures from `conftest.py`.
- Tag slow or externally dependent tests so they can be deselected; review coverage via `pytest tests --cov=src` and `htmlcov/index.html`.
- Frontend suites stay in `frontend/__tests__/` using Testing Library; refresh snapshots whenever UI contracts change.
- After major refactors, run `pytest tests/smoke_tests.py` and document notable findings in `docs/`.

## Commit & Pull Request Guidelines
- Commits follow conventional prefixes (`feat:`, `refactor:`, `fix:`) with imperative subjects under 72 characters.
- PRs should link issues, describe impact, and flag API or config updates; include terminal output or screenshots for UX changes.
- Confirm CI locally: run lint, backend/frontend tests, and any config migrations before requesting review.
- Update `README.md`, `docs/`, and sample configs whenever behavior or operational procedures change.
