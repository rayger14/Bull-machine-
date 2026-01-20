# Ghost → Live v2 Upgrade: Risks & Rollback Plan

**Version:** 2.0.0
**Date:** 2025-11-19
**Status:** Risk Assessment Complete
**Purpose:** Comprehensive risk analysis and rollback procedures for Ghost → Live v2 upgrade

---

## Executive Summary

This document identifies all risks associated with the Ghost → Live v2 upgrade and provides detailed mitigation strategies and rollback procedures for each risk category.

**Risk Categories:**
1. **Data Integrity Risks:** Feature store corruption, NaN propagation
2. **Performance Degradation Risks:** Backtest metrics below baseline
3. **Integration Risks:** Module conflicts, API breakage
4. **Technical Debt Risks:** Incomplete upgrades, ghost code
5. **Operational Risks:** Deployment failures, configuration errors

**Mitigation Strategy:** Multi-layered defense (backup, validation, incremental testing, rollback)

---

## 1. Risk Matrix

### 1.1 Risk Assessment Summary

| Risk ID | Risk Description | Probability | Impact | Severity | Mitigation |
|---------|-----------------|-------------|--------|----------|------------|
| R1 | Feature store corruption | LOW | CRITICAL | HIGH | Backup before each phase |
| R2 | NaN propagation in new columns | MEDIUM | HIGH | HIGH | Schema validation after each phase |
| R3 | Performance degradation (PF < baseline) | MEDIUM | CRITICAL | HIGH | Gold standard validation gates |
| R4 | API breakage (backward compatibility) | LOW | CRITICAL | MEDIUM | API stability tests |
| R5 | Module import conflicts | LOW | HIGH | MEDIUM | Integration tests |
| R6 | Config incompatibility | LOW | MEDIUM | LOW | Config validation script |
| R7 | Incomplete module upgrades (PARTIAL stuck) | MEDIUM | MEDIUM | MEDIUM | Module status tracker |
| R8 | Dead code accumulation | LOW | LOW | LOW | Dead code detector |
| R9 | Test failures in CI/CD | MEDIUM | HIGH | MEDIUM | Pre-commit hooks |
| R10 | Deployment rollback failure | LOW | CRITICAL | MEDIUM | Tested rollback procedures |

**Severity Calculation:** Severity = Probability × Impact

---

## 2. Data Integrity Risks

### 2.1 R1: Feature Store Corruption

**Risk:** Feature store becomes corrupted during upgrade, losing historical data

**Probability:** LOW
**Impact:** CRITICAL (unrecoverable data loss)
**Severity:** HIGH

**Mitigation:**

1. **Backup Before Each Phase:**
```bash
# Phase 0 backup
cp data/features_mtf/BTC_1H_2022-2024.parquet \
   data/features_mtf/BTC_1H_2022-2024_phase0_backup.parquet

# Phase 1 backup
cp data/features_mtf/BTC_1H_2022-2024.parquet \
   data/features_mtf/BTC_1H_2022-2024_phase1_backup.parquet

# ... etc for each phase
```

2. **Checksum Validation:**
```bash
# Generate checksum before modification
sha256sum data/features_mtf/BTC_1H_2022-2024.parquet > \
  data/features_mtf/BTC_1H_2022-2024.parquet.sha256

# Validate after modification
sha256sum -c data/features_mtf/BTC_1H_2022-2024.parquet.sha256
```

3. **Read-Only Original:**
```bash
# Mark original as read-only
chmod 444 data/features_mtf/BTC_1H_2022-2024_original.parquet
```

**Rollback Procedure:**
```bash
# Restore from backup
cp data/features_mtf/BTC_1H_2022-2024_phase<N>_backup.parquet \
   data/features_mtf/BTC_1H_2022-2024.parquet

# Validate restoration
sha256sum -c data/features_mtf/BTC_1H_2022-2024.parquet.sha256
```

**Recovery Time:** < 1 minute (file copy)

---

### 2.2 R2: NaN Propagation in New Columns

**Risk:** New columns introduce NaN values that propagate through calculations

**Probability:** MEDIUM
**Impact:** HIGH (backtest failures, invalid signals)
**Severity:** HIGH

**Mitigation:**

1. **Schema Validation After Each Phase:**
```bash
python bin/validate_feature_store_schema.py \
  --input data/features_mtf/BTC_1H_2022-2024.parquet \
  --schema docs/FEATURE_STORE_SCHEMA_v2.md \
  --strict

# Must pass before proceeding to next phase
```

2. **NaN Detection:**
```python
def detect_nans(df):
    nan_counts = df.isnull().sum()
    if nan_counts.sum() > 0:
        print("❌ NaN values detected:")
        print(nan_counts[nan_counts > 0])
        raise ValueError("NaN values found - blocking phase progression")
    return True
```

3. **Column-Level Validation:**
```python
# Validate each new column individually
new_columns = ['liquidity_score', 'oi_change_24h', 'fvg_below', ...]
for col in new_columns:
    assert df[col].notna().all(), f"{col} contains NaN values"
    assert (df[col] >= min_val).all(), f"{col} below min range"
    assert (df[col] <= max_val).all(), f"{col} above max range"
```

**Rollback Procedure:**
```bash
# Revert to previous phase backup
cp data/features_mtf/BTC_1H_2022-2024_phase<N-1>_backup.parquet \
   data/features_mtf/BTC_1H_2022-2024.parquet

# Fix NaN source
# Re-run feature calculation with fix
# Re-validate
```

**Recovery Time:** 1-2 hours (identify NaN source, fix, re-calculate)

---

## 3. Performance Degradation Risks

### 3.1 R3: Performance Degradation (PF < Baseline)

**Risk:** Upgraded system produces worse backtest results than baseline

**Probability:** MEDIUM
**Impact:** CRITICAL (system unusable for trading)
**Severity:** HIGH

**Gold Standard Baseline (2024-01-01 to 2024-09-30):**
- Profit Factor: 1.16
- Trade Count: 330
- Max Drawdown: 4.4%
- Win Rate: 65.8%

**Acceptance Criteria (±5% tolerance):**
- Profit Factor: 1.10 - 1.22
- Trade Count: 297 - 363
- Max Drawdown: 3.96% - 4.84%
- Win Rate: 62.5% - 69.1%

**Mitigation:**

1. **Gold Standard Validation Gates:**
```bash
# Run after each phase
pytest tests/integration/test_gold_standard.py

# Expected output:
# ✓ Profit Factor: 1.14 (within 1.10-1.22 range)
# ✓ Trade Count: 315 (within 297-363 range)
# ✓ Max Drawdown: 4.2% (within 3.96-4.84% range)
# ✓ Win Rate: 64.3% (within 62.5-69.1% range)
#
# GOLD STANDARD: PASSED
```

2. **Performance Regression Detection:**
```python
def detect_performance_regression(current_metrics, baseline_metrics, tolerance=0.05):
    for metric, baseline_value in baseline_metrics.items():
        current_value = current_metrics[metric]
        lower_bound = baseline_value * (1 - tolerance)
        upper_bound = baseline_value * (1 + tolerance)

        if not (lower_bound <= current_value <= upper_bound):
            print(f"❌ REGRESSION: {metric} = {current_value:.2f}")
            print(f"   Expected: {lower_bound:.2f} - {upper_bound:.2f}")
            raise ValueError(f"Performance regression detected in {metric}")
```

3. **Archetype Balance Monitoring:**
```python
# Ensure no single archetype dominates (> 50% of matches)
archetype_distribution = matches.groupby('archetype').size() / len(matches)
if (archetype_distribution > 0.5).any():
    print("⚠ WARNING: Archetype imbalance detected")
    print(archetype_distribution[archetype_distribution > 0.5])
```

**Rollback Procedure:**

If PF degradation > 5%:
```bash
# Step 1: Identify cause
python bin/diagnose_performance_degradation.py \
  --baseline configs/frozen/btc_1h_v2_baseline.json \
  --current configs/mvp/mvp_bull_market_v1.json

# Step 2: Rollback to last good phase
git reset --hard phase<N-1>_complete
cp data/features_mtf/BTC_1H_2022-2024_phase<N-1>_backup.parquet \
   data/features_mtf/BTC_1H_2022-2024.parquet

# Step 3: Re-run gold standard validation
pytest tests/integration/test_gold_standard.py

# Step 4: Document root cause
echo "Performance degradation root cause: ..." >> docs/ROLLBACK_LOG.md
```

**Recovery Time:** 30 minutes - 4 hours (depending on root cause complexity)

---

## 4. Integration Risks

### 4.1 R4: API Breakage (Backward Compatibility)

**Risk:** Upgraded modules break existing APIs used by downstream code

**Probability:** LOW
**Impact:** CRITICAL (production failures)
**Severity:** MEDIUM

**Protected APIs:**
- `from engine.fusion import k2_fusion`
- `from engine.archetypes.logic_v2_adapter import detect_all`
- `from engine.smc.order_blocks import detect_order_blocks`
- `from engine.context.regime_classifier import classify`

**Mitigation:**

1. **API Stability Tests:**
```python
# tests/integration/test_api_stability.py
def test_k2_fusion_signature():
    from engine.fusion import k2_fusion
    import inspect

    sig = inspect.signature(k2_fusion.compute_fusion_score)
    expected_params = ['ctx', 'wyckoff_score', 'liquidity_score', 'momentum_score']
    actual_params = list(sig.parameters.keys())

    assert actual_params == expected_params, "k2_fusion API signature changed"

def test_archetype_detector_signature():
    from engine.archetypes.logic_v2_adapter import detect_all
    import inspect

    sig = inspect.signature(detect_all)
    expected_params = ['ctx', 'row']
    actual_params = list(sig.parameters.keys())

    assert actual_params == expected_params, "detect_all API signature changed"
```

2. **Deprecation Warnings:**
```python
# If API change required, add deprecation warning
import warnings

def old_api_function(ctx):
    warnings.warn(
        "old_api_function is deprecated, use new_api_function instead",
        DeprecationWarning,
        stacklevel=2
    )
    return new_api_function(ctx)
```

3. **Version Pinning:**
```python
# engine/__init__.py
__version__ = "2.0.0"
__api_version__ = "1.0.0"  # API version separate from implementation version

def check_api_compatibility(required_api_version):
    if __api_version__ != required_api_version:
        raise ValueError(f"API version mismatch: required {required_api_version}, got {__api_version__}")
```

**Rollback Procedure:**

If API breakage detected:
```bash
# Step 1: Identify broken API
pytest tests/integration/test_api_stability.py -v

# Step 2: Revert to last good commit
git revert <commit-that-broke-api>

# Step 3: Re-run tests
pytest tests/integration/test_api_stability.py

# Step 4: Document breaking change
echo "API breakage: <description>" >> docs/ROLLBACK_LOG.md
```

**Recovery Time:** 15-30 minutes (revert commit)

---

### 4.2 R5: Module Import Conflicts

**Risk:** Upgraded modules conflict with existing imports

**Probability:** LOW
**Impact:** HIGH (import errors, runtime failures)
**Severity:** MEDIUM

**Mitigation:**

1. **Import Tests:**
```python
# tests/smoke/test_imports.py
def test_all_imports():
    """Verify all critical imports work"""
    import engine.fusion.k2_fusion
    import engine.archetypes.logic_v2_adapter
    import engine.smc.order_blocks
    import engine.context.regime_classifier
    import engine.wyckoff.wyckoff_engine
    # ... all critical modules
```

2. **Circular Import Detection:**
```bash
# Use import graph tool
python bin/detect_circular_imports.py engine/

# Expected output:
# ✓ No circular imports detected
```

3. **Namespace Isolation:**
```python
# Avoid import * (namespace pollution)
# BAD:
from engine.fusion import *

# GOOD:
from engine.fusion import k2_fusion, domain_fusion
```

**Rollback Procedure:**

If import conflict detected:
```bash
# Step 1: Identify conflict
python -c "import engine.problematic_module"

# Step 2: Fix import path or revert
git revert <commit-that-caused-conflict>

# Step 3: Re-test imports
pytest tests/smoke/test_imports.py
```

**Recovery Time:** 10-20 minutes

---

## 5. Technical Debt Risks

### 5.1 R7: Incomplete Module Upgrades (PARTIAL Stuck)

**Risk:** Modules stuck in PARTIAL state after upgrade, not fully functional

**Probability:** MEDIUM
**Impact:** MEDIUM (reduced feature coverage)
**Severity:** MEDIUM

**Mitigation:**

1. **Module Status Tracker:**
```python
# bin/track_module_status.py
module_status = {
    'engine/features/fib_retracement.py': 'PARTIAL',
    'engine/features/swing_detection.py': 'PARTIAL',
    'engine/psychology/fear_greed.py': 'PARTIAL',
    # ... all modules
}

# Track upgrade progress
def upgrade_module(module_path, new_status):
    module_status[module_path] = new_status
    print(f"✓ {module_path}: {new_status}")

# Generate report
def generate_status_report():
    live_count = sum(1 for s in module_status.values() if s == 'LIVE')
    partial_count = sum(1 for s in module_status.values() if s == 'PARTIAL')
    idea_count = sum(1 for s in module_status.values() if s == 'IDEA')

    print(f"Module Status Report:")
    print(f"  LIVE: {live_count} / {len(module_status)}")
    print(f"  PARTIAL: {partial_count} / {len(module_status)}")
    print(f"  IDEA: {idea_count} / {len(module_status)}")
```

2. **Upgrade Checklist:**
```markdown
## features/fib_retracement.py Upgrade Checklist

- [ ] Complete missing logic (fib extensions)
- [ ] Add unit tests (test_fib_retracement.py)
- [ ] Add integration tests
- [ ] Add feature store columns (fib_0.236, fib_0.382, etc.)
- [ ] Validate feature coverage
- [ ] Update documentation
- [ ] Mark as LIVE
```

3. **Automated Validation:**
```python
def validate_module_live_status(module_path):
    """Verify module meets LIVE criteria"""
    checks = [
        has_unit_tests(module_path),
        has_integration_tests(module_path),
        has_documentation(module_path),
        no_todo_comments(module_path),
        code_coverage_above_80(module_path)
    ]

    if all(checks):
        return 'LIVE'
    elif any(checks):
        return 'PARTIAL'
    else:
        return 'IDEA'
```

**Rollback Procedure:**

If module upgrade incomplete:
```bash
# Option 1: Complete upgrade in next phase
# Mark module as PARTIAL, continue

# Option 2: Revert module to previous state
git checkout phase<N-1>_complete -- engine/features/incomplete_module.py

# Re-run tests
pytest tests/unit/test_incomplete_module.py
```

**Recovery Time:** Depends on module complexity (2 hours - 2 days)

---

### 5.2 R8: Dead Code Accumulation

**Risk:** Old code paths remain after upgrade, creating confusion

**Probability:** LOW
**Impact:** LOW (technical debt, confusion)
**Severity:** LOW

**Mitigation:**

1. **Dead Code Detector:**
```python
# bin/detect_dead_features.py
def detect_unused_features(feature_store_columns, code_references):
    """Find features in store but never used in code"""
    unused_features = []
    for col in feature_store_columns:
        if col not in code_references:
            unused_features.append(col)

    if unused_features:
        print("⚠ Unused features detected:")
        print("\n".join(unused_features))
    return unused_features

# Run after each phase
python bin/detect_dead_features.py
```

2. **Code Coverage Analysis:**
```bash
# Generate coverage report
pytest --cov=engine --cov-report=html

# Identify uncovered code (potential dead code)
open htmlcov/index.html
```

3. **Deprecation Markers:**
```python
# Mark deprecated code
@deprecated(version='2.0', reason='Use new_function instead')
def old_function():
    pass
```

**Rollback Procedure:**

If dead code detected:
```bash
# Remove dead code in separate cleanup commit
git checkout -b cleanup/remove-dead-code
# Remove dead code
git commit -m "chore: remove dead code from v1.0"
git push origin cleanup/remove-dead-code
```

**Recovery Time:** 1-3 hours (identify and remove dead code)

---

## 6. Operational Risks

### 6.1 R9: Test Failures in CI/CD

**Risk:** Automated tests fail after upgrade, blocking merge

**Probability:** MEDIUM
**Impact:** HIGH (delayed deployment)
**Severity:** MEDIUM

**Mitigation:**

1. **Pre-Commit Hooks:**
```bash
# .git/hooks/pre-commit
#!/bin/bash
set -e

# Run unit tests
pytest tests/unit/ --maxfail=1

# Run linter
flake8 engine/

# Run type checker
mypy engine/

# Run dead feature detector
python bin/detect_dead_features.py

echo "✓ Pre-commit checks passed"
```

2. **CI/CD Pipeline:**
```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run unit tests
        run: pytest tests/unit/
      - name: Run integration tests
        run: pytest tests/integration/
      - name: Run gold standard validation
        run: pytest tests/integration/test_gold_standard.py
      - name: Validate feature store schema
        run: python bin/validate_feature_store_schema.py
```

3. **Test Isolation:**
```python
# Ensure tests don't depend on each other
@pytest.fixture(autouse=True)
def reset_state():
    """Reset global state before each test"""
    yield
    # Cleanup after test
```

**Rollback Procedure:**

If CI/CD fails:
```bash
# Step 1: Identify failing test
pytest tests/integration/ -v

# Step 2: Fix test or code
# ... fix ...

# Step 3: Re-run CI/CD
git push origin feature/ghost-modules-to-live-v2

# Step 4: Monitor CI/CD
# Wait for green checkmark
```

**Recovery Time:** 30 minutes - 4 hours (depending on fix complexity)

---

### 6.2 R10: Deployment Rollback Failure

**Risk:** Rollback procedure fails, leaving system in broken state

**Probability:** LOW
**Impact:** CRITICAL (system down)
**Severity:** MEDIUM

**Mitigation:**

1. **Tested Rollback Procedures:**
```bash
# Test rollback procedure BEFORE production deployment
# 1. Simulate upgrade
cp data/features_mtf/BTC_1H_2022-2024.parquet \
   data/features_mtf/BTC_1H_2022-2024_backup.parquet

# 2. Make breaking change
# ... intentionally break something ...

# 3. Test rollback
cp data/features_mtf/BTC_1H_2022-2024_backup.parquet \
   data/features_mtf/BTC_1H_2022-2024.parquet

git reset --hard HEAD~1

# 4. Validate rollback
pytest tests/integration/test_gold_standard.py
```

2. **Rollback Checklist:**
```markdown
## Rollback Procedure Checklist

- [ ] Backup feature store
- [ ] Restore from backup
- [ ] Validate checksum
- [ ] Reset git to last good commit
- [ ] Run gold standard validation
- [ ] Verify no NaN values
- [ ] Document rollback reason
```

3. **Automated Rollback Script:**
```bash
# bin/emergency_rollback.sh
#!/bin/bash
set -e

PHASE=$1

echo "⚠ EMERGENCY ROLLBACK TO PHASE $PHASE"

# Restore feature store
cp data/features_mtf/BTC_1H_2022-2024_phase${PHASE}_backup.parquet \
   data/features_mtf/BTC_1H_2022-2024.parquet

# Reset git
git reset --hard phase${PHASE}_complete

# Validate
python bin/validate_feature_store_schema.py --strict
pytest tests/integration/test_gold_standard.py

echo "✓ Rollback complete. System restored to Phase $PHASE"
```

**Rollback Procedure:**

If rollback fails:
```bash
# Nuclear option: restore from original backup
cp data/features_mtf/BTC_1H_2022-2024_phase0_backup.parquet \
   data/features_mtf/BTC_1H_2022-2024.parquet

git reset --hard feature/ghost-modules-to-live-v2^

# Validate
pytest tests/integration/test_gold_standard.py

# Document incident
echo "Emergency rollback executed at $(date)" >> docs/ROLLBACK_LOG.md
```

**Recovery Time:** 5-10 minutes (automated script)

---

## 7. Backup Strategy

### 7.1 Feature Store Backups

**Backup Schedule:**
- Phase 0: Before any changes
- Phase 1: After Tier 1 validation
- Phase 2: After Tier 2 completion
- Phase 3: After Tier 3 completion
- Phase 4: Before final validation

**Backup Location:**
```
data/features_mtf/
├── BTC_1H_2022-2024.parquet (current)
├── BTC_1H_2022-2024_phase0_backup.parquet
├── BTC_1H_2022-2024_phase1_backup.parquet
├── BTC_1H_2022-2024_phase2_backup.parquet
├── BTC_1H_2022-2024_phase3_backup.parquet
└── BTC_1H_2022-2024_phase4_backup.parquet
```

**Retention Policy:**
- Keep all phase backups until merge to main
- Delete after successful production deployment (except phase0)
- Phase 0 backup: Keep forever (original baseline)

**Storage Requirements:**
- Per backup: ~10 MB
- Total backups: 5 × 10 MB = 50 MB

---

### 7.2 Code Backups (Git Tags)

**Tag Strategy:**
```bash
# Tag each phase completion
git tag phase0_complete -m "Phase 0: Planning complete"
git tag phase1_complete -m "Phase 1: Tier 1 validation complete"
git tag phase2_complete -m "Phase 2: Tier 2 upgrade complete"
git tag phase3_complete -m "Phase 3: Tier 3 upgrade complete"
git tag phase4_validated -m "Phase 4: Final validation passed"

# Push tags to remote
git push origin --tags
```

**Rollback via Tags:**
```bash
# Rollback to Phase 2
git reset --hard phase2_complete

# Restore feature store
cp data/features_mtf/BTC_1H_2022-2024_phase2_backup.parquet \
   data/features_mtf/BTC_1H_2022-2024.parquet
```

---

### 7.3 Config Backups

**Backup Strategy:**
```bash
# Frozen configs never modified (protected)
chmod 444 configs/frozen/*.json

# Archive configs before modification
mkdir -p configs/archive/2025-11-19/
cp configs/mvp/*.json configs/archive/2025-11-19/
```

---

## 8. Validation Gates

### 8.1 Phase Gate Checklist

**Phase 1 Gate (Tier 1 Validation):**
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Feature store validation clean (no NaNs)
- [ ] Gold standard metrics within ±5%
- [ ] No API breakage detected
- [ ] Backup created

**Phase 2 Gate (Tier 2 Completion):**
- [ ] All PARTIAL modules upgraded to LIVE (or documented why not)
- [ ] New feature store columns validated (no NaNs)
- [ ] Bear archetype detection functional (S1, S2, S4)
- [ ] Gold standard metrics within ±5%
- [ ] Backup created

**Phase 3 Gate (Tier 3 Completion):**
- [ ] IDEA modules upgraded to PARTIAL/LIVE
- [ ] New feature store columns validated
- [ ] Feature coverage increased (140+ columns)
- [ ] Gold standard metrics within ±5%
- [ ] Backup created

**Phase 4 Gate (Final Validation):**
- [ ] Full regression test suite passes
- [ ] Gold standard metrics within ±5%
- [ ] No performance degradation
- [ ] CI/CD guardrails implemented
- [ ] Documentation complete
- [ ] PR ready for review

---

## 9. Communication Plan

### 9.1 Rollback Notification

**If Rollback Required:**
1. Create incident report in `docs/ROLLBACK_LOG.md`
2. Tag as `incident/rollback-<date>`
3. Notify team via communication channel
4. Document root cause analysis
5. Create fix plan
6. Re-attempt upgrade (if applicable)

**Incident Report Template:**
```markdown
## Rollback Incident Report

**Date:** 2025-11-19
**Phase:** Phase 2
**Reason:** Performance degradation (PF dropped to 1.05, below 1.10 threshold)
**Rollback Target:** Phase 1
**Recovery Time:** 45 minutes

### Root Cause
Liquidity score calculation introduced bias in bear market conditions.

### Fix Plan
1. Revise liquidity score calculation to normalize by regime
2. Re-run Phase 2 with fix
3. Re-validate gold standard

### Status
- [x] Rollback complete
- [ ] Fix implemented
- [ ] Re-validation passed
```

---

## 10. Testing the Rollback

### 10.1 Rollback Dry Run

**Before Production Upgrade, Test Rollback:**

```bash
# 1. Create test environment
git checkout -b test/rollback-dry-run

# 2. Simulate Phase 1 upgrade
cp data/features_mtf/BTC_1H_2022-2024.parquet \
   data/features_mtf/BTC_1H_2022-2024_phase1_backup.parquet
# ... make changes ...

# 3. Simulate Phase 2 upgrade
cp data/features_mtf/BTC_1H_2022-2024.parquet \
   data/features_mtf/BTC_1H_2022-2024_phase2_backup.parquet
# ... make changes ...

# 4. Simulate failure (intentional)
# ... introduce bug ...

# 5. Test rollback to Phase 1
bash bin/emergency_rollback.sh 1

# 6. Validate rollback
pytest tests/integration/test_gold_standard.py

# 7. Clean up test branch
git checkout feature/ghost-modules-to-live-v2
git branch -D test/rollback-dry-run
```

---

## 11. Recovery Time Objectives

| Incident | Target Recovery Time | Maximum Acceptable Downtime |
|----------|---------------------|----------------------------|
| Feature store corruption | 5 minutes | 10 minutes |
| Performance degradation | 4 hours | 8 hours |
| API breakage | 30 minutes | 1 hour |
| Module import conflict | 20 minutes | 30 minutes |
| CI/CD failure | 2 hours | 4 hours |
| Deployment rollback | 10 minutes | 20 minutes |

**Overall RTO:** 4 hours (worst-case performance degradation)
**Overall RPO:** 0 (no data loss tolerated)

---

## 12. References

- **Architecture:** `docs/GHOST_TO_LIVE_ARCHITECTURE.md`
- **Brain Blueprint:** `docs/BRAIN_BLUEPRINT_SNAPSHOT_v2.md`
- **Dev Workflow:** `docs/DEV_WORKFLOW.md`
- **Feature Schema:** `docs/FEATURE_STORE_SCHEMA_v2.md`
- **CI Guardrails:** `docs/CI_GUARDRAILS_SPEC.md`

---

## Appendix A: Emergency Contacts

**System Architect:** Primary escalation
**Data Engineer:** Feature store issues
**DevOps Engineer:** CI/CD, deployment issues

---

## Appendix B: Rollback Log Template

**File:** `docs/ROLLBACK_LOG.md`

```markdown
# Rollback Log

## 2025-11-19: Phase 2 Rollback
- **Reason:** Performance degradation
- **Root Cause:** Liquidity score calculation bias
- **Recovery Time:** 45 minutes
- **Status:** Fixed and re-validated

## (Template for future incidents)
- **Reason:**
- **Root Cause:**
- **Recovery Time:**
- **Status:**
```

---

## Version History

- **v2.0.0** (2025-11-19): Complete risk assessment and rollback procedures
- **v1.0.0** (2025-11-14): Initial risk assessment
