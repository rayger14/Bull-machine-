# Workspace Cleanup Strategy

**Date**: 2026-01-13
**Status**: 350 uncommitted files + 18 modified files
**Objective**: Clean commit history for v4 deployment readiness

---

## Current State Analysis

```bash
Modified files:       18
Untracked files:     332
Total uncommitted:   350
```

### Critical Files Modified
- `archetype_registry.yaml` - Archetype specs (KEEP)
- `engine/context/hmm_regime_model.py` - HMM model code (KEEP)
- `bin/smoke_test_all_archetypes.py` - Smoke tests (KEEP)
- `engine/archetypes/logic_v2_adapter.py` - Adapter logic (KEEP)

### Untracked Files Breakdown
- **Reports**: ~150 markdown documentation files
- **Scripts**: ~80 Python validation/test scripts
- **Results**: ~60 result CSV/parquet files
- **Config**: ~20 configuration files
- **Misc**: ~20 logs, temp files

---

## Commit Strategy

### Phase 1: Core Engine Improvements (Commit 1)

**Scope**: Production-ready archetype and regime improvements

```bash
git add archetype_registry.yaml
git add engine/archetypes/logic_v2_adapter.py
git add engine/strategies/archetypes/bear/__init__.py
git add engine/strategies/archetypes/bull/__init__.py
git add engine/models/archetype_model.py
git add engine/context/hmm_regime_model.py
git add engine/optimization/__init__.py
```

**Commit Message**:
```
feat(archetypes): Production archetype fixes and canonical spec system

- Add canonical archetype spec system (single source of truth)
- Fix direction inversions (funding_divergence SHORT→LONG)
- Implement runtime assertions preventing mismatches
- Update archetype registry with correct directions/regimes
- Refactor logic_v2_adapter for spec compliance

Performance Impact:
- Direction bug fixed: +$1,375 PnL swing (2022)
- Zero direction mismatches after fix
- All archetypes validated against specs

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

### Phase 2: Validation & Testing Infrastructure (Commit 2)

**Scope**: Comprehensive validation suite from parity ladder + A/B testing

**Core Validation Scripts**:
```bash
git add bin/backtest_with_real_signals.py
git add bin/backtest_phase1_dynamic_regime.py
git add bin/run_phase3_streaming_backtest.py
git add bin/validate_hysteresis_fix.py
git add bin/test_raw_regime_model.py
```

**Regime Training Scripts**:
```bash
git add bin/train_logistic_regime_v3.py
git add bin/validate_hybrid_regime.py
git add engine/context/hybrid_regime_model.py
git add engine/context/regime_service.py
git add engine/context/regime_hysteresis.py
git add engine/context/logistic_regime_model.py
```

**Archetype Integration**:
```bash
git add engine/archetypes/archetype_spec.py
git add engine/archetypes/archetype_factory.py
```

**Commit Message**:
```
test(validation): Add parity ladder validation + regime model training

Parity Ladder (3 phases):
- Phase 1: Dynamic regime detection (0 static reads)
- Phase 2: Real MTF computation (0 hardcoded boosts)
- Phase 3: Streaming mode (0 discontinuities)

A/B Testing:
- Hybrid vs v3 ML regime comparison
- Hysteresis integration validation
- Raw model diagnostics (discovered 0.173 confidence issue)

Training Infrastructure:
- LogisticRegimeModel v3 (2023-2024 training)
- Hybrid regime model (crisis rules + ML)
- RegimeService integration with hysteresis

Validation Results:
- All architectural components validated ✓
- Model confidence issue identified (deployment blocker)
- Hysteresis cannot fix low-confidence model

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

### Phase 3: Documentation & Reports (Commit 3)

**Critical Reports Only** (not all 150 files):

```bash
# Core validation reports
git add PARITY_LADDER_COMPLETE_REPORT.md
git add HYSTERESIS_FIX_FINAL_REPORT.md
git add PHASE3_AB_TEST_RESULTS.md
git add PHASE3_DECISION_CARD.md

# Regime model reports
git add HYBRID_REGIME_IMPLEMENTATION_REPORT.md
git add LOGISTIC_REGIME_V3_TRAINING_REPORT.md

# Quick start guides
git add REGIME_DETECTION_QUICK_START.md
git add ARCHETYPE_INTEGRATION_QUICK_START.md

# Data coverage
git add DATA_COVERAGE_NOTICE.md
```

**Commit Message**:
```
docs(validation): Add comprehensive validation reports

Validation Documentation:
- PARITY_LADDER_COMPLETE_REPORT: All 3 phases validated
- HYSTERESIS_FIX_FINAL_REPORT: Deployment blocker analysis
- PHASE3_AB_TEST_RESULTS: Hybrid vs v3 ML comparison

Key Findings:
- Direction bug impact: +$1,375 PnL (critical fix)
- Regime v3 confidence: 0.173 avg (too low for hysteresis)
- Deployment decision: Retrain v4 on 2018-2024 data

Next Steps:
- Acquire 2018-2021 historical data
- Train v4 with 6 years of data
- Target: >0.40 confidence (vs 0.173)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

### Phase 4: Cleanup - Archive Non-Essential Files

**Strategy**: Don't commit everything - archive old reports

```bash
# Create archive directory
mkdir -p archive/validation_reports_2026_01
mkdir -p archive/smoke_tests_2026_01
mkdir -p archive/results_2026_01

# Move non-essential files
mv *_REPORT.md archive/validation_reports_2026_01/ 2>/dev/null || true
mv SMOKE_TEST_*.md archive/smoke_tests_2026_01/ 2>/dev/null || true
mv results/*.csv results/*.parquet archive/results_2026_01/ 2>/dev/null || true

# Add archive to .gitignore
echo "archive/" >> .gitignore
```

**Then clean**:
```bash
git add .gitignore
git status  # Should show much cleaner state
```

---

## .gitignore Additions

Add these patterns to prevent future clutter:

```gitignore
# Validation artifacts
*_VALIDATION_*.md
*_DIAGNOSIS_*.md
*_DIAGNOSTIC_*.md
*_ANALYSIS_*.md
*_COMPARISON_*.md

# Temporary results
results/phase*
results/*_validation/
results/*_diagnostic/

# Logs
*.log
*.output

# Archive
archive/

# Jupyter checkpoints
.ipynb_checkpoints/

# Python cache
__pycache__/
*.pyc
*.pyo

# Data downloads (large files)
data/raw/historical_*/
```

---

## Execution Plan

### Step 1: Review Current State
```bash
git status --short | grep "^M" > modified_files.txt
git status --short | grep "^?" > untracked_files.txt
```

### Step 2: Execute Commits (in order)
```bash
# Commit 1: Engine improvements
<add files from Phase 1>
git commit -m "<message from Phase 1>"

# Commit 2: Validation infrastructure
<add files from Phase 2>
git commit -m "<message from Phase 2>"

# Commit 3: Documentation
<add files from Phase 3>
git commit -m "<message from Phase 3>"

# Commit 4: Cleanup
<execute Phase 4 cleanup>
git add .gitignore
git commit -m "chore: Archive old validation reports and cleanup workspace"
```

### Step 3: Verify Clean State
```bash
git status  # Should show minimal uncommitted files
git log --oneline -5  # Verify commit messages
```

---

## Files to EXCLUDE from Commits

**Never commit**:
- `*.pkl` (model files - too large)
- Large parquet files (>10MB)
- CSV results files
- Log files
- Temporary output files
- IDE-specific files (`.vscode/`, `.idea/`)

**Already in .gitignore** (verify):
- `models/*.pkl`
- `data/raw/*.parquet`
- `results/*.csv`

---

## Post-Cleanup Validation

After commits, verify:

1. **Build succeeds**:
```bash
python3 -m pytest tests/ -v
```

2. **Imports work**:
```python
from engine.context.regime_service import RegimeService
from engine.archetypes.archetype_spec import ArchetypeRegistry
```

3. **No regressions**:
```bash
python3 bin/validate_hysteresis_fix.py  # Should still run
```

---

## Timeline

- **Step 1** (Review): 15 minutes
- **Step 2** (Commits): 30 minutes
- **Step 3** (Verify): 15 minutes
- **Total**: ~1 hour

---

## Success Criteria

✓ Modified files committed (18 files)
✓ Core validation scripts committed (~30 files)
✓ Critical documentation committed (~10 files)
✓ Archive created for old reports (~150 files)
✓ Clean git status (<20 uncommitted files)
✓ .gitignore updated (prevent future clutter)
✓ Build and tests pass

---

## Next Actions After Cleanup

Once workspace is clean:

1. **Create v4 branch**:
```bash
git checkout -b feature/regime-model-v4
```

2. **Acquire 2018-2021 data** (in progress)
3. **Train LogisticRegimeModel v4**
4. **Deploy to paper trading**

---

**Prepared by**: Claude Code
**Date**: 2026-01-13
**Status**: Ready for execution
