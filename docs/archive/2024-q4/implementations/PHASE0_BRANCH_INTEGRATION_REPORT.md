# PHASE 0: Bull Machine v2 - Branch Integration Report

**Date:** 2025-11-12
**System Architect:** Claude Code
**Mission:** Consolidate all Bull Machine v2 work into single integration branch

---

## Executive Summary

**Integration Branch Created:** `bull-machine-v2-integration` ✓
**Base Branch Used:** `pr6a-archetype-expansion` ✓
**Gold Standard Validation:** ⚠ FAILED - Performance Degradation Detected

**Critical Issue Discovered:** Uncommitted modifications in working directory affecting core v2 components

---

## 1. Branch Audit Findings

### A. Integration Base Selection: pr6a-archetype-expansion

**Commits Ahead of Main:** 74 commits
**Last Commit:** 2025-11-05 18:04:03 -0800 (4246fee)
**Message:** feat(router-v10): lock baseline with corrected 2022-2024 results

**Why This Branch:**
- Contains most complete Bull Machine v2 implementation
- Has GMM v3.2 balanced regime classifier (latest model)
- Has state-aware gates implementation
- Has fixed cooldown logic (inherited from v1.5.1)
- Has all 11 archetype detectors
- Has frozen baseline configurations
- Already merged work from all upstream branches:
  - feature/phase2-regime-classifier (via lineage)
  - feature/ml-meta-optimizer (via phase2)
  - pr1-pr5 (sequential integration)

### B. Branch Lineage Map

```
main (2932808) @ 2025-10-12
 |
 +-- feature/phase2-regime-classifier (55 commits) @ 2025-10-23
     |   [GMM, max-hold, feature store v2, regime foundation]
     |
     +-- (merged) feature/ml-meta-optimizer (35 commits)
     |   [Knowledge hooks, ML enhancements, psychology layers]
     |
     +-- pr1-infra-patch-tool (66 commits) @ 2025-10-23
     |   [Infrastructure safety, column patcher]
     |
     +-- pr2-feature-calculators (66 commits) @ 2025-10-23
     |   [Feature quality validation]
     |
     +-- pr3-wire-calculators (66 commits) @ 2025-10-23
     |   [BOMS calculations fix]
     |
     +-- pr4-runtime-intelligence (67 commits) @ 2025-10-23
     |   [Liquidity scoring]
     |
     +-- pr5-decision-gates (67 commits) @ 2025-10-23
         |   [Merged PR1-PR4]
         |
         +-- pr6a-archetype-expansion (74 commits) @ 2025-11-05 ⭐ BASE
             |   [State gates, GMM v3.2, threshold tuning, router v10]
             |
             +-- bull-machine-v2-integration ⭐ NEW INTEGRATION BRANCH
```

### C. Branches Fully Integrated (via lineage)

1. feature/phase2-regime-classifier (55 commits) ✓
2. feature/ml-meta-optimizer (35 commits) ✓
3. pr1-infra-patch-tool (66 commits) ✓
4. pr2-feature-calculators (66 commits) ✓
5. pr3-wire-calculators (66 commits) ✓
6. pr4-runtime-intelligence (67 commits) ✓
7. pr5-decision-gates (67 commits) ✓

**Total Unique Commits in Integration Branch:** 74 commits ahead of main

### D. Branches NOT Merged (Decision Pending)

**feat/v7-trackB-eth-exits-sizing** (1 unique commit - dbf1661)
- **Content:** Paper trading runbook documentation only
- **Risk:** NONE (documentation, no code changes)
- **Recommendation:** Merge if paper trading is imminent, otherwise skip

### E. Legacy Branches (Superseded - Historical Reference Only)

1. feature/macro-fusion-v186 (v1.8.6 era)
2. feature/v1.7 (v1.7 era)
3. feature/v1.7.3-live (v1.7.3 live trading)
4. v1.5.1-core-trader (v1.5.1 era)
5. v1.5.0-features (v1.5.0 era)
6. v1.5.0-optimize-4h (v1.5.0 era)

**None contain unique Bull Machine v2 work.**

---

## 2. Critical Components Verification

### A. GMM v3.2 Regime Classifier ✓

**Model File:** `models/regime_gmm_v3.2_balanced.pkl`
- Size: 37,285 bytes
- Last Modified: 2025-11-11 15:41
- Status: PRESENT

**Loader Module:** `engine/context/regime_classifier.py`
- Version: GMM-based regime classifier
- Features: VIX_Z, DXY_Z, YC_SPREAD, funding_Z, etc. (19 features)
- Status: PRESENT

**Config Reference:** `configs/frozen/btc_1h_v2_baseline.json`
- Line 48: model_path points to GMM v3.2
- Zero-fill mode: enabled
- Status: CONFIGURED CORRECTLY

### B. State-Aware Gates ✓

**Module:** `engine/archetypes/state_aware_gates.py`
- Version: state_aware_gates@v1
- Size: 10,204 bytes
- Last Modified: 2025-11-11 16:13
- Status: PRESENT

**Config:** `configs/frozen/btc_1h_v2_baseline.json` (lines 72-93)
- Enabled: true
- Weights configured (ADX, ATR, funding, 4H alignment)
- Thresholds configured (adx_weak: 18, adx_strong: 30, etc.)
- Status: CONFIGURED

### C. Cooldown Logic ✓

**Inherited From:** v1.5.1 commits (d0ff21f, 68b88d5)
**Files:**
- `engine/archetypes/logic.py` (modified 2025-11-11)
- `engine/archetypes/logic_v2_adapter.py` (modified 2025-11-12)
- Status: PRESENT

### D. Archetype Detectors ✓

**Primary Logic:** `engine/archetypes/logic.py` (47,382 bytes)
**V2 Adapter:** `engine/archetypes/logic_v2_adapter.py` (48,277 bytes)
**Threshold Policy:** `engine/archetypes/threshold_policy.py` (13,365 bytes)
**Registry:** `engine/archetypes/registry.py` (9,384 bytes)
**Status:** ALL PRESENT, recently modified (2025-11-11/12)

### E. Backtest Engine ✓

**Script:** `bin/backtest_knowledge_v2.py`
- Size: 122,166 bytes
- Version: 2.0.0 (69-feature knowledge engine)
- Last Modified: 2025-11-12 17:59
- Status: PRESENT

### F. Frozen Baselines ✓

**Files:**
1. `configs/frozen/btc_1h_v2_baseline.json` (11,577 bytes)
   - Version: 2.0.0-v2-baseline-frozen
   - Profile: v2_baseline_state_gates_cooldown_fixed
2. `configs/frozen/btc_1h_v2_frontier.json` (11,061 bytes)

**Status:** PRESENT

---

## 3. Gold Standard Validation Results

### Test Configuration

**Command:**
```bash
PYTHONHASHSEED=0 python3 bin/backtest_knowledge_v2.py \
  --asset BTC \
  --start 2024-01-01 \
  --end 2024-09-30 \
  --config configs/frozen/btc_1h_v2_baseline.json
```

**Environment:** bull-machine-v2-integration branch
**Date Executed:** 2025-11-12

### Actual vs Expected Results

| Metric | Expected | Actual | Delta | Status |
|--------|----------|--------|-------|--------|
| **Profit Factor** | ~1.16 | 1.03 | -11.2% | ⚠ FAIL |
| **Total Trades** | ~330 | 246 | -25.5% | ⚠ FAIL |
| **Max Drawdown** | ~4.4% | 11.8% | +168% | ⚠ FAIL |
| **Win Rate** | ~50% | 46.7% | -6.6% | ⚠ WARNING |
| **Total PNL** | - | $154.18 | - | - |

**Pass Criteria (±10% tolerance):**
- PF: 1.10 - 1.22 → **FAIL** (1.03 below threshold)
- Trades: 297 - 363 → **FAIL** (246 below threshold)
- DD: 3.96% - 4.84% → **FAIL** (11.8% exceeds threshold)

**Overall Status:** ⚠ INTEGRATION BLOCKED - Performance Degradation

### Archetype Distribution Analysis

**Total Checks:** 6,452 bars
**Total Matches:** 1,542 (23.9% match rate)

| Archetype | Matches | % of Total | Status |
|-----------|---------|------------|--------|
| trap_within_trend | 1,403 | 91.0% | ⚠ OVER-DOMINANT |
| volume_exhaustion | 134 | 8.7% | ✓ Active |
| order_block_retest | 2 | 0.1% | ⚠ MINIMAL |
| failed_continuation | 1 | 0.1% | ⚠ MINIMAL |
| **Other 7 archetypes** | 2 | 0.1% | ⚠ INACTIVE |

**Issue:** Archetype diversity severely reduced. 91% of entries are trap_within_trend, indicating:
- Other archetypes being filtered out by gates
- Threshold tuning may be too restrictive
- Regime classifier may be over-conservative

---

## 4. Root Cause Investigation

### A. Uncommitted Modifications Detected ⚠

**Critical Finding:** 11 core Bull Machine v2 files have uncommitted changes:

```
M  bin/append_macro_to_feature_store.py
M  bin/backtest_knowledge_v2.py
M  configs/profile_experimental.json
M  engine/archetypes/logic.py
M  engine/archetypes/logic_v2_adapter.py
M  engine/archetypes/threshold_policy.py
M  engine/context/loader.py
M  engine/context/regime_classifier.py
M  engine/exits/macro_echo.py
M  engine/exits/multi_modal_exits.py
M  engine/runtime/context.py
```

**Impact:**
- These modifications are NOT part of the pr6a-archetype-expansion commit history
- The gold standard benchmark is running against MODIFIED code, not committed code
- Performance degradation may be due to in-progress work

**Status:** WORKING DIRECTORY NOT CLEAN

### B. Potential Issues

1. **State-Aware Gates Too Restrictive**
   - ADX weak penalty: 0.06
   - ATR low penalty: 0.05
   - Funding high penalty: 0.05
   - Combined penalty could be filtering too many entries

2. **Regime Classifier Over-Filtering**
   - GMM v3.2 model may be too conservative
   - Zero-fill mode enabled (could affect classification)
   - All trades show "neutral" regime (no regime diversity)

3. **Archetype Threshold Tuning**
   - threshold_policy.py was modified (uncommitted)
   - Thresholds may have been adjusted too high
   - Only 2 archetypes active (trap_within_trend, volume_exhaustion)

4. **Working Directory State**
   - 11 modified files not committed
   - Unknown changes from in-progress development
   - Cannot validate "frozen baseline" with uncommitted changes

---

## 5. Integration Branch Status

### Created Successfully ✓

**Branch Name:** `bull-machine-v2-integration`
**Based On:** pr6a-archetype-expansion (4246fee)
**Commits Ahead of Main:** 74 commits
**Merge Conflicts:** NONE (no additional merges performed)

**Command Used:**
```bash
git checkout pr6a-archetype-expansion
git checkout -b bull-machine-v2-integration
```

**Branch Verification:**
```bash
$ git log --oneline -5
4246fee feat(router-v10): lock baseline with corrected 2022-2024 results
ac9d658 feat(pr-a): add parity testing infrastructure for legacy vs adaptive paths
b251474 chore: pin PF-20 bull winner to risk_on profile + diagnostic analysis
e140430 feat(pr6b): complete regime-aware architecture with threshold tuning
2c00b4c feat(frontiers): add empirical BTC/ETH guardrails from performance mapping
```

### Validation Status ⚠

**Gold Standard Benchmark:** FAILED
**Reason:** Performance degradation + uncommitted changes in working directory
**Action Required:** Clean working directory and re-validate

---

## 6. Branches Summary Table

| Branch | Commits vs Main | Last Modified | Bull Machine v2 | Status |
|--------|----------------|---------------|-----------------|---------|
| **pr6a-archetype-expansion** | +74, -0 | 2025-11-05 | ✓ COMPLETE | ⭐ INTEGRATION BASE |
| **bull-machine-v2-integration** | +74, -0 | 2025-11-05 | ✓ COMPLETE | ⭐ NEW - VALIDATION FAILED |
| feature/phase2-regime-classifier | +55, -0 | 2025-10-23 | ✓ Partial | MERGED (via pr6a) |
| feature/ml-meta-optimizer | +35, -0 | 2025-10-16 | ✓ Partial | MERGED (via phase2) |
| pr5-decision-gates | +67, -0 | 2025-10-23 | ✓ Partial | MERGED (via pr6a) |
| pr4-runtime-intelligence | +67, -0 | 2025-10-23 | ✓ Partial | MERGED (via pr5) |
| pr3-wire-calculators | +66, -0 | 2025-10-23 | ✓ Partial | MERGED (via pr5) |
| pr2-feature-calculators | +66, -0 | 2025-10-23 | ✓ Partial | MERGED (via pr5) |
| pr1-infra-patch-tool | +66, -0 | 2025-10-23 | ✓ Partial | MERGED (via pr5) |
| feat/v7-trackB-eth-exits-sizing | +74, -0 | 2025-10-28 | Documentation | OPTIONAL (1 doc commit) |
| feature/macro-fusion-v186 | +35, -0 | 2025-10-14 | No (v1.8.6) | LEGACY - SUPERSEDED |
| feature/v1.7* | Various | 2025-10-07 | No | LEGACY - SUPERSEDED |
| v1.5.* | Various | 2025-09-25 | No | LEGACY - SUPERSEDED |

---

## 7. Action Items & Recommendations

### Immediate Actions (CRITICAL)

1. **Clean Working Directory**
   ```bash
   # Option A: Stash uncommitted changes
   git stash save "WIP: uncommitted v2 modifications from pr6a"

   # Option B: Commit uncommitted changes
   git add -A
   git commit -m "chore: capture in-progress v2 modifications"

   # Option C: Review and selectively commit
   git diff engine/archetypes/logic.py
   # Review each modified file individually
   ```

2. **Re-run Gold Standard Validation**
   ```bash
   # After cleaning working directory
   PYTHONHASHSEED=0 python3 bin/backtest_knowledge_v2.py \
     --asset BTC \
     --start 2024-01-01 \
     --end 2024-09-30 \
     --config configs/frozen/btc_1h_v2_baseline.json
   ```

3. **Document Baseline Performance**
   - If validation passes: Proceed with v2 integration
   - If validation fails: Investigate committed code issues

### Short-Term Actions (HIGH PRIORITY)

4. **Investigate Archetype Imbalance**
   - Why are only 2 of 11 archetypes active?
   - Review threshold_policy.py logic
   - Check state-aware gate penalties
   - Validate regime classifier is not over-filtering

5. **Validate State-Aware Gates**
   - Test with gates disabled (set enable: false)
   - Compare trade count and archetype distribution
   - Adjust penalty weights if too restrictive

6. **Check Regime Classifier**
   - Verify GMM v3.2 model loading correctly
   - Confirm feature extraction working
   - Test with regime override (force risk_on)
   - Analyze why all trades show "neutral" regime

### Medium-Term Actions

7. **Optional: Merge Paper Trading Docs**
   ```bash
   # If paper trading is imminent
   git merge feat/v7-trackB-eth-exits-sizing --no-ff \
     -m "docs: merge paper trading runbook from feat/v7"
   ```

8. **Create Baseline Snapshot**
   - Once validation passes, create git tag
   - Document known-good configuration
   - Establish regression test suite

9. **Archive Legacy Branches**
   ```bash
   # DO NOT DELETE, just mark as archived
   git tag archive/feature-macro-fusion-v186 feature/macro-fusion-v186
   git tag archive/v1.5.1-core-trader v1.5.1-core-trader
   # etc.
   ```

### Long-Term Strategy

10. **Future Development Protocol**
    - ALL work proceeds on `bull-machine-v2-integration` ONLY
    - NO new feature branches from main
    - Working directory MUST be clean before benchmarks
    - Gold standard validation required before any merge

11. **PR to Main (When Ready)**
    - After validation passes
    - After archetype balance restored
    - After performance meets targets (PF ≥ 1.16, trades ≥ 330, DD ≤ 4.4%)

---

## 8. Risk Assessment

### Current Risks

| Risk | Severity | Impact | Mitigation |
|------|----------|--------|------------|
| **Uncommitted changes affecting validation** | 🔴 HIGH | Cannot trust gold standard results | Clean working directory immediately |
| **Archetype imbalance (91% one archetype)** | 🔴 HIGH | System not diversifying risk | Investigate threshold tuning |
| **Performance degradation (PF 1.03 vs 1.16)** | 🔴 HIGH | Below production standards | Root cause analysis required |
| **DD increased 168% (11.8% vs 4.4%)** | 🔴 HIGH | Unacceptable risk profile | Review exit logic and sizing |
| **Trade count down 25% (246 vs 330)** | 🟡 MEDIUM | Missing opportunities | Review gate penalties |
| **No regime diversity (all "neutral")** | 🟡 MEDIUM | Classifier may not be working | Validate GMM v3.2 loading |

### Mitigated Risks ✓

| Risk | Status | Resolution |
|------|--------|------------|
| Branch lineage unclear | ✓ RESOLVED | Full lineage map documented |
| Multiple conflicting branches | ✓ RESOLVED | pr6a contains all work via lineage |
| Stacked branches creating complexity | ✓ RESOLVED | Clean linear path main→phase2→pr6a |
| Missing Bull Machine v2 components | ✓ RESOLVED | All components verified present |

---

## 9. Conclusions

### What Was Accomplished ✓

1. **Comprehensive Branch Audit** - All 15 local branches analyzed
2. **Branch Lineage Mapped** - Clear visualization of merge history
3. **Integration Base Identified** - pr6a-archetype-expansion contains complete v2 work
4. **Integration Branch Created** - bull-machine-v2-integration ready
5. **Component Verification** - All Bull Machine v2 components present and located
6. **Gold Standard Test Executed** - Benchmark ran (though failed validation)
7. **Documentation Complete** - Full audit in docs/BRANCH_AUDIT.md

### What Blocked Integration ⚠

1. **Uncommitted Modifications** - 11 core files have uncommitted changes
2. **Performance Degradation** - Gold standard validation failed all 3 metrics
3. **Archetype Imbalance** - 91% of entries from single archetype
4. **Working Directory Not Clean** - Cannot validate frozen baseline with WIP changes

### Integration Status

**Branch Creation:** ✓ COMPLETE
**Validation:** ⚠ BLOCKED - Clean working directory required
**Ready for Development:** ⚠ NOT YET - Performance issues must be resolved

---

## 10. Path Forward

### Option A: Stash and Validate (RECOMMENDED)

```bash
# 1. Save uncommitted work
git stash save "WIP: uncommitted v2 modifications from pr6a"

# 2. Re-run gold standard on clean branch
PYTHONHASHSEED=0 python3 bin/backtest_knowledge_v2.py \
  --asset BTC --start 2024-01-01 --end 2024-09-30 \
  --config configs/frozen/btc_1h_v2_baseline.json

# 3. If passes: Document and proceed
# 4. If fails: Investigate committed code

# 5. Restore stashed work (if validation passed)
git stash pop
```

### Option B: Commit WIP and Document

```bash
# 1. Commit all uncommitted changes
git add -A
git commit -m "chore: capture in-progress v2 modifications

Uncommitted changes from pr6a working directory:
- Modified archetype logic, threshold_policy, state_aware_gates
- Modified regime_classifier, context loader
- Modified backtest engine, macro echo exits
- Modified runtime context

Note: These changes may explain gold standard validation failure.
Performance: PF 1.03 (expected 1.16), 246 trades (expected 330), DD 11.8% (expected 4.4%)
"

# 2. Create checkpoint tag
git tag checkpoint/bull-machine-v2-wip-state

# 3. Investigate performance degradation
```

### Option C: Rollback to Last Known Good

```bash
# 1. Discard uncommitted changes
git checkout .
git clean -fd

# 2. Rollback to pr6a base
git reset --hard pr6a-archetype-expansion

# 3. Re-validate
PYTHONHASHSEED=0 python3 bin/backtest_knowledge_v2.py \
  --asset BTC --start 2024-01-01 --end 2024-09-30 \
  --config configs/frozen/btc_1h_v2_baseline.json

# 4. If passes: This is the clean baseline
```

---

## 11. Deliverables

### Documentation

1. ✓ **docs/BRANCH_AUDIT.md** - Comprehensive branch analysis
2. ✓ **PHASE0_BRANCH_INTEGRATION_REPORT.md** (this document)

### Git State

1. ✓ **bull-machine-v2-integration branch** - Created from pr6a
2. ⚠ **Working directory** - NOT CLEAN (11 modified files)
3. ⚠ **Gold standard validation** - FAILED (needs investigation)

### Next Agent Handoff

**Status:** ⚠ INTEGRATION INCOMPLETE - ACTION REQUIRED

**Before Next Agent:**
1. Clean working directory (stash or commit)
2. Re-run gold standard validation
3. If validation fails, document root cause
4. If validation passes, mark integration COMPLETE

**For Next Agent:**
- Use `bull-machine-v2-integration` branch ONLY
- DO NOT create new branches from main
- Working directory MUST be clean before any benchmarks
- Gold standard validation MUST pass before proceeding

---

## Appendix A: Git Commands Reference

### View Branch Audit
```bash
cat docs/BRANCH_AUDIT.md
```

### Check Current Branch
```bash
git branch --show-current
# Should show: bull-machine-v2-integration
```

### View Uncommitted Changes
```bash
git status
git diff engine/archetypes/logic.py
git diff --stat
```

### Run Gold Standard Validation
```bash
PYTHONHASHSEED=0 python3 bin/backtest_knowledge_v2.py \
  --asset BTC --start 2024-01-01 --end 2024-09-30 \
  --config configs/frozen/btc_1h_v2_baseline.json
```

### Clean Working Directory
```bash
# Option 1: Stash
git stash save "WIP: description"

# Option 2: Discard (DANGER)
git checkout .
git clean -fd

# Option 3: Commit
git add -A
git commit -m "chore: message"
```

---

## Appendix B: Component Locations

### Models
- `models/regime_gmm_v3.2_balanced.pkl` - GMM v3.2 regime classifier (37KB)
- `models/regime_gmm_v3.1_fixed.pkl` - GMM v3.1 (46KB)
- `models/regime_gmm_v3_full.pkl` - GMM v3 full (46KB)

### Engine Components
- `engine/context/regime_classifier.py` - Regime classification logic
- `engine/archetypes/state_aware_gates.py` - Dynamic gate computation
- `engine/archetypes/logic.py` - Primary archetype detection (47KB)
- `engine/archetypes/logic_v2_adapter.py` - V2 adapter layer (48KB)
- `engine/archetypes/threshold_policy.py` - Threshold management
- `engine/archetypes/registry.py` - Archetype registry

### Configs
- `configs/frozen/btc_1h_v2_baseline.json` - Gold standard baseline (11KB)
- `configs/frozen/btc_1h_v2_frontier.json` - Frontier config (11KB)

### Scripts
- `bin/backtest_knowledge_v2.py` - Full backtest engine (122KB)

---

**Report Status:** COMPLETE
**Date Generated:** 2025-11-12
**Author:** System Architect Agent (Claude Code)
**Version:** 1.0
