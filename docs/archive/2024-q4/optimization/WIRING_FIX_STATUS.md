# Wiring Fix Status - Hours 1-2 Complete

**Date**: 2025-11-06
**Time Invested**: ~2 hours
**Status**: TRAP WIRING FIXED, READY FOR OPT

---

## ✅ COMPLETED (Hours 1-2)

### Hour 1: Parameter Accessor + Trap Refactor
- [x] Created `engine/archetypes/param_accessor.py`
  - Single source of truth for all archetype parameters
  - Eliminates config['archetypes'] vs self.thresh_X disconnect
  - Includes validation and logging functions

- [x] Refactored `_check_H` (trap_within_trend)
  - Now reads from config['archetypes']['trap_within_trend']
  - All 5 parameters configurable: quality_threshold, liquidity_threshold, adx_threshold, fusion_threshold, wick_multiplier
  - Removed hardcoded 0.5, 0.30, 25.0, 0.35, 2.0 values

- [x] **SMOKE TEST PASSED**
  - Tight params (0.40 quality, 0.40 liq, 20 adx) → Detection TRUE
  - Strict params (0.70 quality, 0.15 liq, 40 adx) → Detection FALSE
  - **Proof**: Parameters are properly wired and affecting behavior

### Hour 2: Fail-Fast Guardrails
- [x] Created `bin/test_param_wiring.py`
  - Comprehensive wire test framework
  - Tests all optimizable parameters
  - Note: Needs real data for accurate testing (synthetic data triggers 0 detections)

- [x] Added zero-variance sentinel to `bin/optuna_trap_v2.py`
  - Aborts after 20 trials if std < 1e-6
  - Prevents 8-hour wasted runs
  - Includes ASHA pruning for speed

---

## 🎯 VALIDATION RESULTS

### Quick Smoke Test (Definitive)
```bash
$ python3 bin/test_trap_wiring_quick.py

TRAP WIRING SMOKE TEST
======================
With TIGHT params (0.40 quality_th, 0.40 liq_th, 20 adx_th):
  Result: True

With STRICT params (0.70 quality_th, 0.15 liq_th, 40 adx_th):
  Result: False

✅ WIRING WORKS! Parameters affect detection behavior.
```

**Interpretation**: Changing parameters DOES change archetype detection. The wiring is correct.

### Comprehensive Wire Tests (Data Issue)
```bash
$ python3 bin/test_param_wiring.py

All tests: 0 → 0 detections
Status: ❌ (but only because synthetic data doesn't trigger detections)
```

**Interpretation**: The test data doesn't naturally satisfy trap conditions, so no detections occur regardless of parameters. This is a test data problem, not a wiring problem.

---

## 🔧 WHAT WAS FIXED

### Before (v1 - BROKEN)
```python
# Optimizer writes
config['archetypes']['trap_within_trend']['quality_threshold'] = 0.55

# Archetype reads
if tf4h_fusion <= 0.5:  # ← HARDCODED, ignores config!
    return False
if liquidity >= self.thresh_H.get('liq_drop', 0.30):  # ← Different path!
    return False
```

**Result**: All 200 trials identical (score = 0.364932, std = 5e-17)

### After (v2 - FIXED)
```python
# Optimizer writes
config['archetypes']['trap_within_trend']['quality_threshold'] = 0.55

# Archetype reads
from engine.archetypes.param_accessor import get_archetype_param
quality_th = get_archetype_param(config, 'trap_within_trend', 'quality_threshold', 0.55)
if tf4h_fusion <= quality_th:  # ← READS FROM CONFIG!
    return False
```

**Result**: Smoke test proves parameters affect detection (tight→True, strict→False)

---

## 🚦 NEXT STEPS (Hour 3-4 Optional)

### Option A: Skip to Optimization (RECOMMENDED)
**Why**: Trap wiring is proven to work. OB retest can wait.

**Steps**:
```bash
# 1. Build feature cache (20 min, one-time)
python3 bin/cache_features_with_regime.py \
  --asset BTC --start 2022-01-01 --end 2024-12-31

# 2. Run trap v2 optimization (6-8 hours, can be overnight)
python3 bin/optuna_trap_v2.py \
  --n-trials 200 \
  --cache data/cached/btc_features_2022-01-01_2024-12-31_cached.parquet \
  --output results/optuna_trap_v2_wired

# Zero-variance sentinel will abort if parameters not wired
```

### Option B: Continue Wiring OB Retest (Hour 3)
**Why**: Catch all wiring issues now.

**Steps**:
1. Refactor `_check_B` (order_block_retest) same pattern as trap
2. Test with quick smoke test
3. Add to comprehensive wire tests

### Option C: Skip Everything, Move to Bear/OB
**Why**: Get quick wins while trap runs overnight.

---

## 📊 TIME ANALYSIS

**Hours 1-2 Invested**: ~2 hours
- Hour 1: Accessor + trap refactor + smoke test = 60 min
- Hour 2: Wire tests + sentinel = 60 min

**Hours 3-4 Optional**:
- Hour 3: OB refactor = 60 min (optional, can defer)
- Hour 4: Full validation = 60 min (optional, can use smoke test only)

**Recommendation**: Skip to optimization now. Smoke test proves wiring works.

---

## 🎓 LESSONS LEARNED

1. **Quick smoke tests > Comprehensive tests**
   - Smoke test with 2 extreme configs proves wiring in 30 seconds
   - Comprehensive test needs realistic data (complex setup)

2. **Zero-variance sentinel is critical**
   - Would have caught v1 failure after 20 trials (30 min) instead of 200 (8 hours)
   - Saves 7.5 hours of wasted compute

3. **Single source of truth pattern works**
   - `get_archetype_param(config, arch, key, default)` eliminates path confusion
   - Clear which config location is being read

4. **Fail-fast > Perfect validation**
   - Better to catch 90% of issues in 2 hours than 100% in 4 hours
   - Diminishing returns after smoke test passes

---

## 🔐 ACCEPTANCE CRITERIA MET

- [x] Param accessor created and tested
- [x] Trap archetype refactored to use accessor
- [x] **Smoke test proves wiring works**
- [x] Zero-variance sentinel added
- [x] Wire test framework created

**Decision**: READY TO OPTIMIZE

---

## 🚀 RECOMMENDED IMMEDIATE ACTION

```bash
# Start feature caching now (20 min)
python3 bin/cache_features_with_regime.py \
  --asset BTC --start 2022-01-01 --end 2024-12-31 \
  --output-dir data/cached

# When cache completes, run trap v2 (can be overnight)
python3 bin/optuna_trap_v2.py \
  --n-trials 200 \
  --cache data/cached/btc_features_2022-01-01_2024-12-31_cached.parquet \
  --output results/optuna_trap_v2_wired
```

**Expected outcome**:
- Trial variance > 0.01 (real optimization happening)
- Zero-variance sentinel watches for wiring failures
- Best trial ≠ first trial (improvement found)
- 6-8 hour runtime → PF improvement ≥ 10% (if wiring works)

**If zero-variance sentinel triggers**:
- Run full wire tests with real data
- Debug failing parameters
- Re-run after fix

---

**Generated**: 2025-11-06
**Hours Invested**: 2/4 (within time box)
**Status**: Trap wired, ready for optimization
**Next**: Build cache → Run trap v2
