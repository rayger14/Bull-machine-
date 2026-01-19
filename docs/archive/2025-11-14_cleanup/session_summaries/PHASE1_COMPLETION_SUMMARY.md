# Phase 1 Completion Summary - Parameter Flow Fixes

**Date**: 2025-11-08
**Status**: ✅ CORE FIXES COMPLETED
**Branch**: pr6a-archetype-expansion

---

## What Was Fixed

Phase 1 focused on correcting the **critical parameter wiring bugs** that were causing zero-variance in optimization. Three core fixes were implemented:

### 1. Fixed Archetype Slug Names (CRITICAL)

**File**: `engine/archetypes/logic_v2_adapter.py`

**Problem**: Archetype H and K were using the WRONG canonical slugs when reading parameters:
- `_check_H` used `'momentum_continuation'` ❌ (should be `'trap_within_trend'` ✓)
- `_check_K` used `'trap_within_trend'` ❌ (should be `'wick_trap_moneytaur'` ✓)

**Fix Applied**:
- Line 556-558: Changed `_check_H` to use `'trap_within_trend'`
- Line 578-580: Changed `_check_K` to use `'wick_trap_moneytaur'`

**Impact**: Parameters written by optimizer can now be read by correct archetype logic.

---

### 2. Added Validation Logging to ThresholdPolicy

**File**: `engine/archetypes/threshold_policy.py`

**Problem**: No visibility into what parameters were being loaded from config.

**Fix Applied** (Lines 179-194):
```python
# PHASE 1 FIX: Comprehensive logging to verify parameter loading
loaded_count = sum(1 for v in base_map.values() if v)
empty_count = sum(1 for v in base_map.values() if not v)

logger.info(f"[PHASE1] ThresholdPolicy loaded {loaded_count} archetypes with params, {empty_count} empty")

# Log key archetypes for optimization debugging
for key_arch in ['trap_within_trend', 'wick_trap_moneytaur', 'wyckoff_spring_utad']:
    if key_arch in base_map:
        params = base_map[key_arch]
        if params:
            param_summary = ', '.join(f"{k}={v}" for k, v in list(params.items())[:3])
            logger.info(f"[PHASE1] {key_arch}: {len(params)} params ({param_summary}...)")
        else:
            logger.warning(f"[PHASE1] {key_arch}: EMPTY (optimizer params may not be reaching context!)")
```

**Impact**: Can now verify at runtime that parameters are flowing from config → context.

---

### 3. Added Warning Logging to RuntimeContext

**File**: `engine/runtime/context.py`

**Problem**: Silent failures when parameters not found - always used hardcoded defaults.

**Fix Applied** (Lines 56-73):
```python
# PHASE 1 FIX: Always warn on critical failures (not just first N calls)
if not self.thresholds:
    logger.warning(
        f"[PHASE1] CRITICAL: Thresholds dict is EMPTY! "
        f"get_threshold('{archetype}', '{param}') using default={default}"
    )
elif archetype not in self.thresholds:
    logger.warning(
        f"[PHASE1] Archetype '{archetype}' NOT FOUND in thresholds! "
        f"Available: {list(self.thresholds.keys())[:5]}... "
        f"Using default={default} for '{param}'"
    )
elif param not in self.thresholds[archetype]:
    logger.warning(
        f"[PHASE1] Parameter '{param}' NOT FOUND in archetype '{archetype}'! "
        f"Available params: {list(self.thresholds[archetype].keys())} "
        f"Using default={default}"
    )
```

**Impact**: Will immediately show in logs when parameters are missing or using wrong names.

---

## Expected Behavior After Fixes

### Before Phase 1:
```
[PR-A THRESHOLD DEBUG] get_threshold('momentum_continuation', 'fusion_threshold', default=0.35)
  → 0.35 | arch_data=NOT_FOUND
```
**Result**: Always uses hardcoded default 0.35 (zero variance!)

### After Phase 1:
```
[PHASE1] ThresholdPolicy loaded 11 archetypes with params, 2 empty
[PHASE1] trap_within_trend: 5 params (fusion_threshold=0.35, adx_threshold=25.0, liquidity_threshold=0.30...)
[PHASE1] wick_trap_moneytaur: 3 params (fusion_threshold=0.36, adx_threshold=25.0, liquidity_threshold=0.30...)

[PR-A THRESHOLD DEBUG] get_threshold('trap_within_trend', 'fusion_threshold', default=0.35)
  → 0.35 | arch_data={'fusion_threshold': 0.35, 'adx_threshold': 25.0, ...}
```
**Result**: Parameters successfully loaded and read (variance restored!)

---

## What's Still Needed (Next Steps)

Phase 1 focused on **correcting the wiring**. There are still issues to address:

### Remaining from Original Phase 1 Plan:

1. **Remove Hardcoded Defaults** (HIGH PRIORITY)
   - Lines 556-558 in `logic_v2_adapter.py` (_check_H) still have defaults
   - Lines 578-580 in `logic_v2_adapter.py` (_check_K) still have defaults
   - Should fail loudly if param not found instead of silently using default

2. **Expand to ALL Archetypes** (MEDIUM PRIORITY)
   - Only H and K fixed so far
   - Need to audit and fix A, B, C, D, E, F, G, L, M archetypes
   - Use same pattern: correct canonical slug, no hardcoded defaults

3. **Legacy Config Compatibility** (MEDIUM PRIORITY)
   - Baseline configs still use letter codes: `config['archetypes']['thresholds']['H']`
   - Optimizer writes to: `config['archetypes']['trap_within_trend']`
   - ThresholdPolicy handles both, but this needs testing

---

## How to Test the Fixes

### Test 1: Check Logging Output

Run any backtest and check logs for `[PHASE1]` markers:

```bash
python3 bin/backtest_knowledge_v2.py \
  --cache data/cached/btc_features_2022-01-01_2024-12-31_cached.parquet \
  --config configs/baseline_btc_bull_pf20.json \
  --capital 10000 2>&1 | grep "PHASE1"
```

**Expected output**:
```
[PHASE1] ThresholdPolicy loaded X archetypes with params, Y empty
[PHASE1] trap_within_trend: N params (fusion_threshold=..., ...)
[PHASE1] wick_trap_moneytaur: N params (fusion_threshold=..., ...)
```

### Test 2: Parameter Variance Test

Run 5 trials with EXTREME parameter differences:

```python
# Trial 1: fusion_threshold = 0.10 (very low)
# Trial 2: fusion_threshold = 0.90 (very high)

# If results are IDENTICAL → bug still exists
# If results are DIFFERENT → fix working!
```

### Test 3: Optimizer Re-Run

Re-run the trap_within_trend optimizer to verify parameters flow correctly:

```bash
python3 bin/optuna_trap_v2.py \
  --n-trials 5 \
  --cache data/cached/btc_features_2022-01-01_2024-12-31_cached.parquet \
  --output results/optuna_trap_phase1_test
```

Check logs for:
- `[PHASE1]` markers showing parameters loaded
- `[PR-A THRESHOLD DEBUG]` showing parameters found (not NOT_FOUND)
- Different trial results (not identical scores)

---

## Files Modified

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `engine/archetypes/logic_v2_adapter.py` | 556-558, 578-580 | Fix archetype slug names for H and K |
| `engine/archetypes/threshold_policy.py` | 179-194 | Add validation logging to _build_base_map() |
| `engine/runtime/context.py` | 56-73 | Add warning logging to get_threshold() |

**Total**: 3 files, ~30 lines of changes

---

## Success Criteria

Phase 1 is considered successful when:

1. ✅ Code imports without syntax errors
2. ⏳ Logs show `[PHASE1]` markers with parameter counts
3. ⏳ `trap_within_trend` parameters are found (not NOT_FOUND)
4. ⏳ `wick_trap_moneytaur` parameters are found (not NOT_FOUND)
5. ⏳ Optimization trials show parameter variance (different results for different params)

**Current Status**: Items 1 completed, items 2-5 require runtime testing.

---

## Next Phase Preview

**Phase 2 Plan**:
1. Remove ALL hardcoded defaults from archetype logic
2. Update configs to use canonical slugs everywhere
3. Standardize parameter names (long form: `fusion_threshold` not `fusion`)
4. Audit and fix remaining 9 archetypes (A, B, C, D, E, F, G, L, M)

**Estimated Effort**: 4-8 hours
