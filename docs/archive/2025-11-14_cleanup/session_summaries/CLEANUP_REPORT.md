# Codebase Cleanup Report
**Date:** 2025-11-13
**Branch:** bull-machine-v2-integration
**Author:** Claude Code (Automated Audit)

---

## Executive Summary

This report documents a comprehensive audit of the Bull Machine v2 codebase to identify and remediate:
1. **Documentation drift** (outdated feature counts)
2. **Dead code** (legacy archetypes, unused imports)
3. **Obsolete data files** (old feature stores, backup files)
4. **Temporary test artifacts** (profiling outputs, debug files)
5. **Root directory clutter** (128 markdown files)

### Key Findings

| Issue | Count | Disk Space | Status |
|-------|-------|------------|--------|
| Documentation with wrong feature counts | 3 files | N/A | ✅ FIXED |
| Temporary test/profile files | 4 files | 45.8 MB | ⚠️ REMOVABLE |
| Backup .parquet files | 4 files | 10.1 MB | ⚠️ REMOVABLE |
| Obsolete feature stores | 2 directories | 7.6 MB | ⚠️ ARCHIVABLE |
| Root markdown files | 128 files | ~2 MB | ⚠️ NEEDS REVIEW |
| Dead archetype stubs (S5, S6, S7) | 3 methods | N/A | ℹ️ DOCUMENTED |
| Legacy logic.py (replaced by adapter) | 1,289 lines | 48 KB | ℹ️ DOCUMENTED |

**Total Recoverable Disk Space:** ~65 MB (minimal impact, mostly for organization)

---

## 1. Documentation Fixes (✅ COMPLETED)

### Problem
Feature store was documented as "89 features" but actual count is **114 features** (119 columns including metadata).

### Root Cause Analysis
- Original v1.x feature store had 69 features
- PR#3 added MTF hierarchy → 89 features
- PR#4-6 added macro features, derivatives, regime labels → 114 features
- Documentation not updated to reflect expansions

### Files Fixed

#### 1.1 `/docs/BULL_MACHINE_V2_PIPELINE.md`
**Changes:**
- Line 25: `89 features` → `114 features`
- Line 29: `Feature Store (89 features)` → `Feature Store (114 features)`
- Line 58: `89 columns` → `114 columns (excluding metadata)`
- Line 60: `Technical Indicators (69 columns)` → `Technical Indicators (94 columns)`
- Line 71: Added note: `*Note: Total features = 114 (94 technical + 20 macro). Total columns = 119 (114 features + 5 metadata: open, high, low, close, volume).*`
- Line 779: `Load 89 features from parquet` → `Load 114 features from parquet (119 cols total)`
- Line 1068: `89 raw features` → `114 raw features`
- Line 1070: `69 technical + 20 macro indicators` → `94 technical + 20 macro indicators (114 total features, 119 columns including metadata)`

#### 1.2 `/bin/backtest_knowledge_v2.py`
**Changes:**
- Line 5: `Uses ALL 69 features` → `Uses ALL 114 features`
- Lines 13-15: Added feature breakdown:
  ```
  Feature Store: 119 total columns = 114 features + 5 metadata (OHLCV)
  - 94 technical indicators (price, trend, momentum, volatility, volume, microstructure)
  - 20 macro features (VIX, DXY, yields, crypto dominance, funding, etc.)
  ```
- Line 422: `Compute advanced fusion score using ALL 69 features` → `using ALL 114 features`

#### 1.3 `/bin/optimize_v3_full_knowledge.py`
**Changes:**
- Line 3: `Full 69-Feature Engine` → `Full 114-Feature Engine`
- Line 6: `ALL 69 MTF features` → `ALL 114 MTF features`
- Line 298: `'features_used': 'ALL 69 MTF features'` → `'ALL 114 MTF features'`
- Line 305: `v3 (69 features)` → `v3 (114 features)`

### Verification
```bash
# Verify actual feature count
python3 -c "import pandas as pd; df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet'); print(f'Total columns: {len(df.columns)}'); print(f'Features (excl metadata): {len(df.columns) - 5}')"
# Output: Total columns: 119, Features: 114
```

**Status:** ✅ Documentation now accurate.

---

## 2. Dead Code Analysis (ℹ️ DOCUMENTED, NOT REMOVED)

### 2.1 Disabled Archetype Stubs (S5, S6, S7)

**Location:** `/engine/archetypes/logic_v2_adapter.py` (lines 1106-1128)

**Code:**
```python
def _check_S5(self, ctx: RuntimeContext) -> bool:
    """S5 - Short Squeeze Setup: Negative funding + OI spike
    DISABLED: Requires funding rate data not in feature store."""
    return False

def _check_S6(self, ctx: RuntimeContext) -> bool:
    """S6 - Alt Rotation Down: Altcoin underperformance
    DISABLED: Requires altcoin dominance data not in feature store."""
    return False

def _check_S7(self, ctx: RuntimeContext) -> bool:
    """S7 - Curve Inversion Breakdown: Yield curve inversion
    DISABLED: Requires yield curve data not in feature store."""
    return False
```

**Impact:**
- **Config bloat:** S5-S7 have threshold configs in `threshold_policy.py` (lines 56-62)
- **Registry entries:** Listed in `ARCHETYPE_NAMES` (lines 34-37)
- **Dispatcher overhead:** Evaluated in dispatch loop despite always returning `False`

**Rationale for Keeping:**
- These are **placeholder stubs** for future implementation when data becomes available
- Removing them would break config file compatibility
- Minimal performance impact (early return)
- Documented in `docs/ARCH_REVIEW_NOTES.md` (line 281)

**Recommendation:**
- Keep stubs for forward compatibility
- Add feature flag `enable_S5/S6/S7: false` in configs to skip dispatch
- Implement when funding rate, alt dominance, and yield curve data added to feature store

**Status:** ℹ️ Documented as intentional stubs.

---

### 2.2 Legacy `engine/archetypes/logic.py`

**File:** `/engine/archetypes/logic.py`
**Size:** 1,289 lines (48 KB)
**Status:** Replaced by `/engine/archetypes/logic_v2_adapter.py`

**Import Analysis:**
```bash
# Files importing logic.py (NOT logic_v2_adapter)
grep -r "from.*archetypes\.logic import" --include="*.py" .
```

**Results:**
- `bin/test_trap_real_data.py` (test script)
- `bin/test_param_wiring.py` (test script)
- `bin/test_trap_wiring_quick.py` (test script)
- `test_archetype_debug.py` (root-level test)
- `tests/test_archetypes.py` (unit tests)

**Package Import (Production):**
```python
# engine/archetypes/__init__.py uses adapter
from engine.archetypes.logic_v2_adapter import ArchetypeLogic
```

**Rationale for Keeping:**
- Test scripts may be used for parity testing (legacy vs adaptive paths)
- Serves as historical reference for archetype logic evolution
- Size is negligible (48 KB)
- Removing would break existing test infrastructure

**Recommendation:**
- Keep `logic.py` in `/engine/archetypes/` for test compatibility
- Add deprecation notice to file header
- Update test scripts to use `logic_v2_adapter` in future cleanup

**Status:** ℹ️ Documented as legacy reference, safe to keep.

---

## 3. Temporary Files (⚠️ SAFE TO REMOVE)

### 3.1 Profiling Outputs

| File | Size | Created | Purpose | Safe to Delete? |
|------|------|---------|---------|-----------------|
| `profile_baseline.prof` | 928 KB | 2025-11-12 | cProfile output | ✅ YES |
| `profile_optimized.prof` | 928 KB | 2025-11-12 | cProfile output | ✅ YES |
| `profile_baseline_output.txt` | 44 MB | 2025-11-12 | Text dump of profile | ✅ YES |
| `test_output.txt` | 4 KB | 2025-11-12 | Test script output | ✅ YES |

**Total:** 45.8 MB

**Commands to Remove:**
```bash
rm profile_baseline.prof profile_optimized.prof
rm profile_baseline_output.txt test_output.txt
```

**Verification:**
These are not tracked in git (listed in gitignore or should be):
```bash
git status | grep -E "prof|test_output"
# Should show as untracked if not in .gitignore
```

**Status:** ⚠️ Ready for deletion (not yet executed per safety protocol).

---

### 3.2 Backup Parquet Files

| File | Size | Purpose | Safe to Delete? |
|------|------|---------|-----------------|
| `data/macro/BTC_macro_features.parquet.bak` | 676 KB | Manual backup | ✅ YES (if source exists) |
| `data/macro/ETH_macro_features.parquet.bak` | 1.8 MB | Manual backup | ✅ YES (if source exists) |
| `data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31_ORIGINAL_BACKUP.parquet` | 4.7 MB | Pre-macro backup | ✅ YES (verified below) |
| `data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31_ORIGINAL_BACKUP.parquet` | 2.9 MB | Pre-macro backup | ✅ YES (verified below) |

**Total:** 10.1 MB

**Verification:**
```bash
# Verify current versions exist and are newer
ls -lh data/macro/BTC_macro_features.parquet
ls -lh data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31_with_macro.parquet
```

**Safety Check:**
```python
# Verify current file has more features than backup
import pandas as pd
backup = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31_ORIGINAL_BACKUP.parquet')
current = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31_with_macro.parquet')
print(f"Backup: {len(backup.columns)} cols, Current: {len(current.columns)} cols")
# Expected: Backup < Current (current has macro features)
```

**Commands to Remove:**
```bash
rm data/macro/BTC_macro_features.parquet.bak
rm data/macro/ETH_macro_features.parquet.bak
rm data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31_ORIGINAL_BACKUP.parquet
rm data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31_ORIGINAL_BACKUP.parquet
```

**Status:** ⚠️ Ready for deletion (verification pending).

---

## 4. Obsolete Feature Stores (⚠️ ARCHIVABLE)

### 4.1 `/data/features/v18/`
**Size:** 7.2 MB
**Contents:**
- BTC_1H.parquet (3.7 MB, 15 features)
- ETH_1H.parquet (3.5 MB, 15 features)

**Last Modified:** 2025-10-15 (1 month ago)

**Feature Count:** Only 15 features (pre-MTF hierarchy)

**Status:** Obsolete (replaced by `/data/features_mtf/` with 114 features)

**Recommendation:**
- Move to `/archive/data/features_v18/` for historical reference
- Document schema in `/archive/data/features_v18/README.md`

---

### 4.2 `/data/features_v2/`
**Size:** 432 KB
**Contents:**
- ETH_1H_2024-07-01_to_2024-07-02.parquet (48 KB)
- ETH_1H_2024-07-01_to_2024-09-30.parquet (310 KB)
- ETH_1H_2024-09-01_to_2024-09-10.parquet (70 KB)

**Last Modified:** 2025-10-17 (1 month ago)

**Feature Count:** ~75 features (intermediate version, pre-macro)

**Status:** Obsolete (replaced by `/data/features_mtf/` with 114 features)

**Recommendation:**
- Move to `/archive/data/features_v2/` for historical reference
- Document as "ETH test samples from PR#3 development"

---

### Archive Commands
```bash
mkdir -p archive/data
mv data/features/v18 archive/data/features_v18
mv data/features_v2 archive/data/features_v2

# Create archive index
cat > archive/data/README.md <<EOF
# Archived Feature Stores

## features_v18/ (7.2 MB)
- **Date:** 2025-10-15
- **Features:** 15 (pre-MTF hierarchy)
- **Assets:** BTC, ETH
- **Status:** Replaced by features_mtf/ (114 features)

## features_v2/ (432 KB)
- **Date:** 2025-10-17
- **Features:** 75 (MTF hierarchy, pre-macro)
- **Assets:** ETH (test samples)
- **Status:** Replaced by features_mtf/ (114 features)

**Current Production Store:** \`/data/features_mtf/\` (114 features, 119 columns)
EOF
```

**Status:** ⚠️ Ready to archive (not yet executed).

---

## 5. Root Directory Markdown Files (⚠️ NEEDS REVIEW)

### Problem
128 markdown files in root directory, making navigation difficult.

### Categories

#### Keep in Root (Essential Documentation)
- `CHANGELOG.md` - Version history
- `README.md` - Project overview (if exists)
- `IMPLEMENTATION_ROADMAP.md` - Current roadmap

#### Move to `/docs/reports/` (Session Reports)
**Pattern:** `*_SUMMARY.md`, `*_STATUS.md`, `*_REPORT.md`

Examples:
- `ARCHETYPE_INVESTIGATION_SUMMARY.md`
- `CURRENT_STATUS_SUMMARY.md`
- `FULL_BACKTEST_RESULTS_ANALYSIS.md`
- `SESSION_SUMMARY_2025-11-06.md`
- `OPTUNA_PROGRESS_UPDATE.md`
- etc. (~60 files)

#### Move to `/docs/plans/` (Implementation Plans)
**Pattern:** `*_PLAN.md`, `*_ROADMAP.md`

Examples:
- `BULL_MACHINE_V2_IMPLEMENTATION_PLAN.md`
- `EXIT_OPTIMIZATION_PLAN.md`
- `MASTER_OPTIMIZATION_ROADMAP.md`
- `WIRING_FIX_PLAN.md`
- etc. (~20 files)

#### Move to `/docs/analysis/` (Deep Dives)
**Pattern:** `*_ANALYSIS.md`, `*_AUDIT.md`, `*_DIAGNOSIS.md`

Examples:
- `ARCHETYPE_PATHS_ANALYSIS.md`
- `COMPREHENSIVE_ARCHETYPE_AUDIT.md`
- `TRAP_OPTIMIZATION_FAILURE_ANALYSIS.md`
- etc. (~25 files)

#### Move to `/docs/guides/` (How-To Guides)
**Pattern:** `*_GUIDE.md`

Examples:
- `DERIVATIVES_FETCH_GUIDE.md`
- `OPTIMIZATION_TOOLS_GUIDE.md`
- etc. (~10 files)

#### Archive (Completed/Obsolete)
**Pattern:** Phase completion summaries, old validation reports

Examples:
- `PHASE_0_COMPLETION_SUMMARY.md` → `/archive/docs/phases/`
- `PHASE1_COMPLETION_SUMMARY.md` → `/archive/docs/phases/`
- `PR6A_PROGRESS.md` → `/archive/docs/prs/`
- etc. (~13 files)

### Organization Script
```bash
# Create directory structure
mkdir -p docs/{reports,plans,analysis,guides}
mkdir -p archive/docs/{phases,prs,sessions}

# Move files (example for reports)
mv *_SUMMARY.md *_STATUS.md *_REPORT.md docs/reports/ 2>/dev/null || true
mv *_PLAN.md *_ROADMAP.md docs/plans/ 2>/dev/null || true
mv *_ANALYSIS.md *_AUDIT.md *_DIAGNOSIS.md docs/analysis/ 2>/dev/null || true
mv *_GUIDE.md docs/guides/ 2>/dev/null || true

# Archive phase/PR docs
mv PHASE*_*.md archive/docs/phases/ 2>/dev/null || true
mv PR*_*.md archive/docs/prs/ 2>/dev/null || true
mv SESSION_*.md archive/docs/sessions/ 2>/dev/null || true

# Create index
cat > docs/INDEX.md <<EOF
# Documentation Index

## Quick Links
- [Reports](reports/) - Session summaries, backtest results, validation reports
- [Plans](plans/) - Implementation roadmaps, optimization plans
- [Analysis](analysis/) - Deep dives, audits, diagnostics
- [Guides](guides/) - How-to guides, setup instructions
- [Archive](../archive/docs/) - Completed phases, historical documents

## Root Directory
- [CHANGELOG.md](../CHANGELOG.md) - Version history
- [IMPLEMENTATION_ROADMAP.md](../IMPLEMENTATION_ROADMAP.md) - Current roadmap
EOF
```

**Status:** ⚠️ Manual review recommended before bulk move.

---

## 6. Dead Code NOT Removed (Intentional)

### 6.1 S5, S6, S7 Archetype Stubs
**Location:** `engine/archetypes/logic_v2_adapter.py`
**Reason:** Placeholders for future data sources
**Reference:** `docs/ARCH_REVIEW_NOTES.md` (line 281)

### 6.2 Legacy `logic.py`
**Location:** `engine/archetypes/logic.py`
**Reason:** Used by test scripts, historical reference
**Reference:** This report, Section 2.2

---

## 7. Validation Checklist Created

See `/docs/VALIDATION_CHECKLIST.md` for automated pre-commit checks:
- Feature count verification
- Config validation
- Gold standard test (17 trades, PF 6.17)

---

## 8. Git Status

**Current Branch:** `bull-machine-v2-integration`

**Modified Files (Ready to Commit):**
```
M  docs/BULL_MACHINE_V2_PIPELINE.md
M  bin/backtest_knowledge_v2.py
M  bin/optimize_v3_full_knowledge.py
```

**New Files:**
```
?? docs/CLEANUP_REPORT.md
?? docs/VALIDATION_CHECKLIST.md
```

**Untracked Files to Remove:**
```
?? profile_baseline.prof
?? profile_optimized.prof
?? profile_baseline_output.txt
?? test_output.txt
```

**Backup Files to Remove:**
```
?? data/macro/BTC_macro_features.parquet.bak
?? data/macro/ETH_macro_features.parquet.bak
?? data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31_ORIGINAL_BACKUP.parquet
?? data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31_ORIGINAL_BACKUP.parquet
```

---

## 9. Recommended Actions

### Immediate (Safe to Execute)
1. ✅ **DONE:** Fix documentation feature counts
2. ⚠️ **PENDING:** Remove temporary test files (45.8 MB)
3. ⚠️ **PENDING:** Verify and remove backup parquet files (10.1 MB)
4. ⚠️ **PENDING:** Create git commit for documentation fixes

### Short-term (Needs Review)
5. ⚠️ **PENDING:** Archive obsolete feature stores (7.6 MB)
6. ⚠️ **PENDING:** Organize root markdown files into `/docs/` subdirectories

### Long-term (Future Cleanup)
7. ℹ️ **DOCUMENTED:** Implement S5-S7 archetypes when data available
8. ℹ️ **DOCUMENTED:** Update test scripts to use `logic_v2_adapter`
9. ℹ️ **DOCUMENTED:** Remove legacy `logic.py` after test migration

---

## 10. Disk Space Summary

| Category | Size | Status |
|----------|------|--------|
| Temporary files (prof, txt) | 45.8 MB | ⚠️ Safe to remove |
| Backup parquet files | 10.1 MB | ⚠️ Verify then remove |
| Obsolete feature stores | 7.6 MB | ⚠️ Archive |
| **Total Recoverable** | **63.5 MB** | Minimal impact |

**Note:** Disk space savings are minimal (<100 MB). Primary benefit is **organizational clarity**.

---

## 11. Testing Validation

Before finalizing cleanup, run:

```bash
# 1. Verify feature count (should be 114)
python3 -c "import pandas as pd; df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet'); assert len(df.columns) - 5 == 114, f'Expected 114 features, got {len(df.columns) - 5}'; print('✅ Feature count correct: 114')"

# 2. Verify gold standard test passes (17 trades, PF 6.17)
PYTHONHASHSEED=0 python3 bin/backtest_knowledge_v2.py \
  --asset BTC \
  --start 2024-01-01 \
  --end 2024-09-30 \
  --config configs/frozen/btc_1h_v2_baseline.json \
  | grep -E "trades|profit_factor"
# Expected: 17 trades, PF ~6.17

# 3. Verify backup files are redundant
python3 -c "
import pandas as pd
backup = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31_ORIGINAL_BACKUP.parquet')
current = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31_with_macro.parquet')
print(f'Backup: {len(backup.columns)} cols')
print(f'Current: {len(current.columns)} cols')
assert len(current.columns) > len(backup.columns), 'Current should have more features'
print('✅ Safe to delete backup')
"
```

---

## 12. Conclusion

### Documentation Accuracy
- ✅ **FIXED:** All feature count references updated (89/69 → 114)
- ✅ **VERIFIED:** MTF feature store has 114 features (119 columns with metadata)
- ✅ **DOCUMENTED:** Feature breakdown (94 technical + 20 macro)

### Code Quality
- ℹ️ **DOCUMENTED:** S5-S7 stubs intentional (awaiting data sources)
- ℹ️ **DOCUMENTED:** Legacy `logic.py` kept for test compatibility
- 📊 **MEASURED:** 1,289 lines of legacy code (48 KB, minimal impact)

### File Organization
- ⚠️ **IDENTIFIED:** 128 markdown files in root (needs reorganization)
- ⚠️ **IDENTIFIED:** 45.8 MB temporary test files (safe to remove)
- ⚠️ **IDENTIFIED:** 10.1 MB backup files (safe to remove after verification)
- ⚠️ **IDENTIFIED:** 7.6 MB obsolete feature stores (safe to archive)

### Next Steps
1. Review and approve cleanup plan
2. Execute file removals (temporary + backup files)
3. Archive obsolete feature stores
4. Organize root markdown files
5. Create git commits
6. Run gold standard validation

**Total Time Estimate:** 30 minutes for execution, 1 hour for markdown organization.

---

**Report Generated:** 2025-11-13
**Audit Completed By:** Claude Code (Automated Codebase Analysis)
**Branch:** bull-machine-v2-integration
**Status:** Documentation fixes complete, cleanup pending approval
