# Repository Cleanup Plan

## DRY Violations Identified

### 1. Duplicate Services (Critical)
- `src/services/auth_service.py` vs `src/security/auth_service.py`
- `src/services/encryption_service.py` vs `src/security/encryption_service.py`
- `src/services/compliance_service.py` vs `src/security/compliance_service.py`
- `src/services/audit_service.py` vs `src/security/audit_service.py`
- `src/services/kyc_service.py` vs `src/security/kyc_service.py`

**Decision**: Keep `src/services/` versions (more comprehensive), remove `src/security/` duplicates

### 2. Root Directory Test Files
- `test_all_fixes.py`
- `test_critical_fixes.py` 
- `test_task_7_1_validation.py`
- `test_visualization.py`

**Decision**: Move to `tests/` directory or remove if redundant

### 3. Multiple Hyperopt Result Directories
- `hyperopt_results/`
- `hyperopt_results_final/`
- `hyperopt_results_task_5_5/`
- `hyperopt_results_task_5_5_final/`
- `hyperopt_results_task_5_5_fixed/`
- `hyperopt_results_task_5_5_working/`

**Decision**: Keep only `hyperopt_results/` (latest), archive others

### 4. Redundant ML Files
- Multiple similar trainer implementations
- Duplicate model validation files
- Old interpretability documentation files

**Decision**: Keep working implementations, remove drafts/duplicates

### 5. Visualization Files
- `benchmark_visualization.py`
- `demo_visualization.py`
- `live_viewer_enhanced.py`
- `test_visualization.py`

**Decision**: Consolidate into `src/ml/visualization/`

## Cleanup Actions

1. Remove duplicate services from `src/security/`
2. Move/consolidate test files
3. Archive old hyperopt results
4. Clean up ML directory
5. Consolidate visualization files
6. Update imports and dependencies
7. Run comprehensive tests