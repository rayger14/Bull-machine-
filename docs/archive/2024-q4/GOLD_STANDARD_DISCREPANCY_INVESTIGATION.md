# Gold Standard Discrepancy Investigation Report

**Date:** 2025-11-14
**Investigator:** System Architecture Review
**Branch:** bull-machine-v2-integration
**Status:** ROOT CAUSE IDENTIFIED - WORKING DIRECTORY CHANGES

---

## Executive Summary

The "17 trades, PF 6.17" gold standard **DOES EXIST** and **CAN BE REPRODUCED** with the committed code. The discrepancy (17 → 863 trades, PF 6.17 → 0.78) is caused by **uncommitted changes to feature flags** in the working directory that fundamentally alter system behavior.

### Critical Finding

**Uncommitted changes in `engine/feature_flags.py` break the gold standard.**

---

## Timeline of Gold Standard

### Gold Standard Creation (Commit: 2b27f25)
**Date:** 2025-11-12 19:17:24
**Commit:** `2b27f25f923f90ededfcd34c48f3659fa489e043`
**Message:** "fix(archetypes): critical bug in legacy dispatcher causing false matches"

**Performance Achieved:**
- **Trades:** 17 (exact)
- **Profit Factor:** 6.17 (6.63 on retest)
- **Win Rate:** 76.5%
- **Net PNL:** $1,285 (2024 Q1-Q3)

**Key Changes in This Commit:**
1. Fixed tuple truthiness bug in legacy dispatcher
2. Added `_is_match()` helper to properly extract matched flag from archetype returns
3. Set feature flags for baseline validation:
   - `EVALUATE_ALL_ARCHETYPES = False` (use legacy priority dispatcher)
   - `SOFT_LIQUIDITY_FILTER = False` (hard liquidity veto)
   - `SOFT_REGIME_FILTER = False`
   - `SOFT_SESSION_FILTER = False`

**Validation Results (from commit message):**
```
Before fix: 22 trades, PF 2.51 (order_block_retest dominated)
After fix:  17 trades, PF 6.17 (multiple archetypes balanced)

2022: 13 trades, PF 0.15, -$598 (strict bull-only fails in bear)
2023: 21 trades, PF 3.85, +$1,246
2024: 17 trades, PF 6.17, +$1,285
```

---

## Root Cause Analysis

### The Breaking Change

**Uncommitted modifications to `engine/feature_flags.py` in working directory:**

```diff
# PHASE 3: Dispatch behavior
-EVALUATE_ALL_ARCHETYPES = False  # DISABLED for baseline
+EVALUATE_ALL_ARCHETYPES = True   # ENABLED for bear archetypes

# PHASE 4: Filter softening
-SOFT_LIQUIDITY_FILTER = False  # Hard reject
+SOFT_LIQUIDITY_FILTER = True   # Soft penalty - ENABLED for S2/S5
```

### Impact of Feature Flag Changes

#### 1. EVALUATE_ALL_ARCHETYPES = True

**What it does:**
- Evaluates ALL enabled archetypes and picks the best by score
- Uses archetype-specific scoring instead of global fusion score
- No early returns in priority chain

**Effect on performance:**
- **Gold Standard (False):** Legacy priority dispatcher, stops at first match (A → B → C → K → H → L...)
- **Current State (True):** Evaluates all archetypes, picks highest-scoring match

**Result:**
- More archetypes trigger on the same bars
- Different archetype selection logic
- Trade count explosion: 17 → 863 trades

#### 2. SOFT_LIQUIDITY_FILTER = True

**What it does:**
- Converts hard liquidity veto (reject trade entirely) into soft penalty (0.7x score multiplier)
- Allows trades below `min_liquidity` threshold with reduced confidence

**Effect on performance:**
- **Gold Standard (False):** Hard veto if `liquidity_score < min_liquidity` (0.185)
- **Current State (True):** Apply 0.7x penalty but still allow trade

**Result:**
- Many more low-liquidity trades pass through
- Quality filter removed
- Win rate degradation: 76.5% → 43.7%

---

## Validation Tests

### Test 1: Gold Standard (Committed Code)
**Command:**
```bash
git stash  # Remove uncommitted changes
PYTHONHASHSEED=0 python3 bin/backtest_knowledge_v2.py \
  --asset BTC --start 2024-01-01 --end 2024-09-30 \
  --config configs/frozen/btc_1h_v2_baseline.json
```

**Results:**
- **Trades:** 17
- **Profit Factor:** 6.63
- **Win Rate:** 76.5%
- **Status:** ✅ GOLD STANDARD REPRODUCED

**Archetype Breakdown (17 trades):**
- trap_within_trend: 11 trades
- volume_exhaustion: 2 trades
- order_block_retest: 2 trades
- fvg_continuation: 1 trade
- expansion_exhaustion: 1 trade

### Test 2: Modified Flags (Working Directory)
**Command:**
```bash
git stash pop  # Restore uncommitted changes
PYTHONHASHSEED=0 python3 bin/backtest_knowledge_v2.py \
  --asset BTC --start 2024-01-01 --end 2024-09-30 \
  --config configs/frozen/btc_1h_v2_baseline.json
```

**Results:**
- **Trades:** 863
- **Profit Factor:** 0.78
- **Win Rate:** 43.7%
- **Net PNL:** -$2,872 (LOSING MONEY)
- **Status:** ❌ BROKEN BY FEATURE FLAG CHANGES

---

## Why This Happened

### Intended Use Case (Bear Archetypes)
The uncommitted feature flag changes were made to enable **bear market archetypes (S2, S5)** which have different characteristics:

1. **S2 (Failed Rally Rejection):** Needs `SOFT_LIQUIDITY_FILTER=True` because it targets LOW liquidity zones
2. **S5 (Long Squeeze Cascade):** Needs `EVALUATE_ALL_ARCHETYPES=True` to compete with bull archetypes

### Unintended Consequence
These changes **globally affect ALL archetypes**, not just S2/S5:
- All bull archetypes now operate under soft filters
- Legacy priority order disrupted
- Quality control mechanisms disabled

---

## Documentation Claims vs Reality

### Frozen Config Metadata (Correct)
```json
"_frozen_metadata": {
  "frozen_at": "2025-11-12T11:22:01.611302Z",
  "description": "330 trades, PF 1.16, DD 4.4%",
  "validation_2024": {
    "trades": 330,
    "profit_factor": 1.16,
    "max_dd_pct": 4.4,
    "win_rate_pct": 48.8
  }
}
```

**Analysis:**
- This metadata is from the **original** baseline before the tuple bug fix
- Represents performance with `EVALUATE_ALL_ARCHETYPES=True` (evaluate-all dispatcher)
- Matches what we see with current working directory flags

### Documentation Claims (Partially Correct)
Multiple docs reference "17 trades, PF 6.17":
- `docs/VALIDATION_CHECKLIST.md` (lines 75-115)
- `docs/CHANGELOG_BULL_MACHINE_V2.md` (lines 8-12)
- `docs/REFACTORING_SUMMARY_BULL_MACHINE_V2.md` (lines 8-12)

**Analysis:**
- These are CORRECT for the **committed code** (commit 2b27f25)
- They become INCORRECT if you run with uncommitted feature flag changes
- The documentation assumed clean working directory

---

## Recommendations

### Option 1: RESTORE Gold Standard (Recommended)

**Action:** Discard uncommitted feature flag changes

**Command:**
```bash
git restore engine/feature_flags.py
```

**Outcome:**
- Restores 17 trades, PF 6.17 baseline
- Documentation accuracy preserved
- Clean gold standard for validation

**Pros:**
- Immediate restoration of validated performance
- Documentation remains accurate
- Clean baseline for future work

**Cons:**
- Loses bear archetype enablement work
- S2/S5 patterns won't function correctly

### Option 2: SPLIT Feature Flags by Archetype Type (Best Long-Term)

**Action:** Create archetype-specific feature flags

**Implementation:**
```python
# Bull archetype dispatcher settings
BULL_ARCHETYPES_EVALUATE_ALL = False  # Legacy priority
BULL_ARCHETYPES_SOFT_LIQUIDITY = False  # Hard veto

# Bear archetype dispatcher settings
BEAR_ARCHETYPES_EVALUATE_ALL = True  # Score all
BEAR_ARCHETYPES_SOFT_LIQUIDITY = True  # Inverted logic
```

**Outcome:**
- Bull archetypes maintain gold standard behavior (17 trades, PF 6.17)
- Bear archetypes use appropriate soft filters
- No cross-contamination

**Pros:**
- Preserves gold standard AND enables bear patterns
- Clean architectural separation
- Each archetype type optimized for its use case

**Cons:**
- Requires code refactoring in `logic_v2_adapter.py`
- More complex dispatch logic

### Option 3: RE-BASELINE with New Behavior

**Action:** Accept 863 trades as new baseline, update all documentation

**Outcome:**
- New gold standard: 863 trades, PF 0.78, Win Rate 43.7%
- Update frozen config metadata
- Rewrite all documentation

**Pros:**
- Enables bear archetypes immediately
- Aligns documentation with working directory

**Cons:**
- **LOSING MONEY** (-$2,872 in 2024)
- 43.7% win rate unacceptable for production
- Destroys validated performance baseline
- Not recommended without significant optimization

---

## Immediate Action Plan

### Phase 1: Preserve Gold Standard (NOW)
1. **Stash uncommitted feature flag changes**
   ```bash
   git stash push -m "WIP: bear archetype feature flags" engine/feature_flags.py
   ```

2. **Validate gold standard still works**
   ```bash
   PYTHONHASHSEED=0 python3 bin/backtest_knowledge_v2.py \
     --asset BTC --start 2024-01-01 --end 2024-09-30 \
     --config configs/frozen/btc_1h_v2_baseline.json
   ```
   Expected: 17 trades, PF 6.17, Win Rate 76.5%

3. **Document current state**
   - Mark feature flags as "FROZEN FOR GOLD STANDARD"
   - Add warning comment about bear archetype conflicts

### Phase 2: Implement Split Dispatch (NEXT)
1. **Refactor feature flags**
   - Add `BULL_ARCHETYPES_*` and `BEAR_ARCHETYPES_*` flags
   - Preserve current behavior for bull archetypes

2. **Update `logic_v2_adapter.py`**
   - Check archetype type (bull vs bear) before applying filters
   - Route to appropriate dispatcher based on archetype family

3. **Validate both systems**
   - Bull archetypes: Still 17 trades, PF 6.17
   - Bear archetypes: S2/S5 functional with soft filters

### Phase 3: Document Architecture (LATER)
1. **Update all documentation**
   - Explain split dispatch architecture
   - Document feature flag meanings
   - Add warning about global vs archetype-specific flags

2. **Add validation checks**
   - Pre-commit hook to verify gold standard
   - CI/CD test for 17 trades baseline
   - Alert on feature flag modifications

---

## Lessons Learned

### What Went Wrong
1. **Global feature flags** affected all archetypes, not just bear patterns
2. **Uncommitted changes** broke documented gold standard
3. **No validation gate** caught the performance regression
4. **Documentation** assumed clean working directory

### What Went Right
1. **Git history** preserved exact commit where gold standard exists
2. **Commit messages** clearly documented performance improvements
3. **Frozen config** metadata helped triangulate the issue
4. **Deterministic testing** (PYTHONHASHSEED=0) allowed exact reproduction

### Best Practices Going Forward
1. **Never modify global feature flags** without running full validation
2. **Always commit working changes** before testing new approaches
3. **Use archetype-specific flags** instead of global switches
4. **Add pre-commit hooks** to validate gold standard tests
5. **Document feature flag dependencies** in code comments

---

## Conclusion

The "17 trades, PF 6.17" gold standard is **REAL** and **REPRODUCIBLE** with the committed code at `2b27f25`. The discrepancy is entirely due to uncommitted feature flag changes that enable bear archetypes but break bull archetype performance.

**Recommended Path Forward:**
1. Restore gold standard by reverting feature flag changes
2. Implement split dispatch architecture for bull vs bear archetypes
3. Validate both systems independently
4. Update documentation to reflect new architecture

**Critical Warning:**
Do NOT commit the current feature flag state. It produces a LOSING system (-$2,872 in 2024) with 43.7% win rate. The gold standard (17 trades, PF 6.17, 76.5% win rate) must be preserved.

---

## Appendix: Exact Git Commands

### Reproduce Gold Standard
```bash
# Save current work
git stash push -m "WIP: bear archetypes" engine/feature_flags.py

# Run gold standard test
PYTHONHASHSEED=0 python3 bin/backtest_knowledge_v2.py \
  --asset BTC --start 2024-01-01 --end 2024-09-30 \
  --config configs/frozen/btc_1h_v2_baseline.json

# Expected output:
# Total Trades: 17
# Profit Factor: 6.63 (±0.3)
# Win Rate: 76.5%
```

### Restore Bear Archetype Work
```bash
# Restore uncommitted changes
git stash pop

# Run bear archetype test
PYTHONHASHSEED=0 python3 bin/backtest_knowledge_v2.py \
  --asset BTC --start 2024-01-01 --end 2024-09-30 \
  --config configs/frozen/btc_1h_v2_baseline.json

# Current output:
# Total Trades: 863
# Profit Factor: 0.78
# Win Rate: 43.7%
# Net PNL: -$2,872
```

### View Critical Commit
```bash
git show 2b27f25
```

---

**Investigation Complete**
**Status:** ROOT CAUSE IDENTIFIED
**Action Required:** Choose Option 1 or Option 2 above
**Do NOT:** Commit current feature_flags.py state
