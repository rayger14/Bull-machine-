# Walk-Forward Validation Bug Fix Report

**Date:** 2026-01-17
**Issue:** Critical archetype isolation failure in walk-forward validation
**Status:** FIXED - Re-running validation (ETA: 22 hours)
**Impact:** Original walk-forward results were completely invalid

---

## Executive Summary

The original walk-forward validation results showed all 6 archetypes failing validation with 54-74% OOS degradation. Investigation revealed a **critical bug**: all archetypes were running together instead of individually, completely invalidating the results.

### The Bug

The `FullEngineBacktest` engine hardcoded `enable_A=True, enable_B=True, ... enable_S5=True`, causing all archetypes to run simultaneously regardless of which archetype was being tested. This meant:

- H, K, and S5 generated **identical** equity curves (same trades down to the penny)
- Trade counts were inflated by all archetypes trading together
- OOS metrics reflected portfolio performance, not individual archetype performance
- Individual archetype validation was impossible

### The Evidence

**Before Fix:**
```
H (trap_within_trend):  217 trades, -0.54% return, final capital $9945.65
K (wick_trap_moneytaur): 217 trades, -0.54% return, final capital $9945.65
S5 (long_squeeze):       217 trades, -0.54% return, final capital $9945.65
```
All three archetypes had **identical equity curves down to the penny**.

**After Fix (Smoke Test on Same Window):**
```
H (trap_within_trend):  0 trades, 0.00% return, final capital $10000.00
K (wick_trap_moneytaur): 1 trade,  +2.92% return, final capital $10291.99
S5 (long_squeeze):       0 trades, 0.00% return, final capital $10000.00
```
Results are now **different** - each archetype runs independently.

---

## Root Cause Analysis

### Location 1: bin/backtest_full_engine_replay.py:178-191

**Problem:**
```python
archetype_config = {
    'use_archetypes': True,
    'enable_A': True,   # Hardcoded
    'enable_B': True,   # Hardcoded
    'enable_C': True,   # Hardcoded
    'enable_H': True,   # Hardcoded
    'enable_K': True,   # Hardcoded
    'enable_S1': True,  # Hardcoded
    'enable_S4': True,  # Hardcoded
    'enable_S5': True   # Hardcoded
}
```

The `archetype_config` was hardcoded inside `__init__()` and never read from the `config` parameter.

**Fix:**
```python
# Default: Enable all archetypes (for normal backtesting)
default_archetype_config = {
    'use_archetypes': True,
    'enable_A': True,
    'enable_B': True,
    # ... etc
}

# CRITICAL FIX: Allow config to override archetype enable flags
archetype_config = default_archetype_config.copy()
for key in ['enable_A', 'enable_B', 'enable_C', 'enable_G', 'enable_H',
            'enable_K', 'enable_S1', 'enable_S4', 'enable_S5']:
    if key in config:
        archetype_config[key] = config[key]
```

### Location 2: bin/walk_forward_production_engine.py:316-329

**Problem:**
```python
engine_config = {
    'symbol': 'BTC',
    'initial_capital': current_capital,
    'position_size_pct': 0.12,
    # ... NO archetype enable flags
}
```

The walk-forward script never passed archetype enable flags to the engine.

**Fix:**
```python
engine_config = {
    'symbol': 'BTC',
    'initial_capital': current_capital,
    'position_size_pct': 0.12,
    # CRITICAL: Disable all archetypes except the one being tested
    'enable_A': False,
    'enable_B': archetype_slug == 'order_block_retest',
    'enable_C': False,
    'enable_G': False,
    'enable_H': archetype_slug == 'trap_within_trend',
    'enable_K': archetype_slug == 'wick_trap_moneytaur',
    'enable_S1': archetype_slug == 'liquidity_vacuum',
    'enable_S4': archetype_slug == 'funding_divergence',
    'enable_S5': archetype_slug == 'long_squeeze'
}
```

---

## Validation of Fix

### Smoke Test Results

Created `bin/test_archetype_isolation_fix.py` to verify fix on a single window (2019-01-03 to 2019-04-03):

**Test Output:**
```
================================================================================
Testing: H (trap_within_trend)
================================================================================
Results:
  Trades: 0
  Return: 0.00%
  Final Capital: $10000.00

================================================================================
Testing: K (wick_trap_moneytaur)
================================================================================
Results:
  Trades: 1
  Return: 2.92%
  Final Capital: $10291.99

================================================================================
Testing: S5 (long_squeeze)
================================================================================
Results:
  Trades: 0
  Return: 0.00%
  Final Capital: $10000.00

✅ PASS: Results are DIFFERENT - bug is fixed!
Each archetype is now running independently as expected.
```

### Verification Against TPE Optimization

Checked that TPE optimization scripts (`bin/optimize_s4_multi_objective_v2.py`) correctly isolated archetypes during optimization:

```python
"archetypes": {
    "use_archetypes": True,
    # Only S4 enabled
    "enable_A": False,
    "enable_B": False,
    "enable_S4": True,  # Only S4 is enabled
    "enable_S5": False,
    # ... all others False
}
```

✅ **Confirmed:** TPE optimization was done correctly with individual archetypes isolated.

This means:
- In-sample metrics from optimization results are valid for individual archetypes
- Walk-forward validation SHOULD compare individual archetype OOS to individual archetype IS
- The bug was introduced only in the walk-forward script, not the optimization

---

## Expected Results from Fixed Validation

### Original (Buggy) Results

All 6 archetypes appeared to fail:

```csv
archetype_id,in_sample_sortino,oos_sortino,oos_degradation_pct,total_oos_trades,passed_validation
B,1.77,0.48,72.74%,2148,False
S1,1.64,0.43,74.12%,2142,False
H,1.29,0.40,69.37%,2127,False
S5,0.00,0.40,0.00%,2127,False  # BUG: S5 should have ~0 trades (crisis-only)
K,0.86,0.40,54.13%,2127,False
S4,0.05,0.26,-417.48%,1794,False  # BUG: Negative degradation is nonsensical
```

**Critical Anomalies:**
1. **H, K, S5 identical OOS Sortino (0.40)** - impossible unless running same trades
2. **H, K, S5 identical trade counts (2,127)** - impossible unless running same trades
3. **S5 had 2,127 trades** - should have ~0 (crisis-only archetype)
4. **S4 had -417% degradation** - mathematically nonsensical

All of these anomalies are explained by the archetypes running together.

### Expected Results After Fix

Based on smoke test patterns:

**Conservative Estimates:**
- **Trade counts will drop significantly** (possibly 80-95% reduction)
  - Many archetypes may have 0 trades in some windows
  - This is EXPECTED and not a bug (archetypes are selective)

- **Individual performance may vary widely:**
  - Some archetypes (B, S1) may pass validation
  - Others (H, K, S5) may have insufficient trades to validate
  - S5 should have ~0 trades (crisis-only)

- **OOS degradation will be calculable:**
  - Previous results were comparing portfolio IS to portfolio OOS
  - New results will compare individual archetype IS to individual archetype OOS
  - This is the correct comparison for walk-forward validation

**Key Questions to Answer:**
1. Which archetypes trade frequently enough to validate? (>=8 trades)
2. Which archetypes maintain Sortino > 0.5 OOS?
3. Which archetypes have <20% OOS degradation?
4. Is low trade frequency due to overfitting or correct selectivity?

---

## Timeline

| Event | Date/Time | Duration |
|-------|-----------|----------|
| Original walk-forward started | 2026-01-16 20:00 | - |
| Original walk-forward completed | 2026-01-17 18:00 | ~22 hours |
| Bug discovered | 2026-01-17 20:00 | - |
| Bug fixed and smoke test passed | 2026-01-17 20:15 | 15 min |
| Fixed walk-forward started | 2026-01-17 20:12 | - |
| **Expected completion** | **2026-01-18 18:00** | **~22 hours** |

---

## Files Modified

### Production Code
1. **bin/backtest_full_engine_replay.py** (lines 178-200)
   - Modified archetype_config to accept overrides from config parameter
   - Preserves backward compatibility (defaults to all enabled)

2. **bin/walk_forward_production_engine.py** (lines 316-345)
   - Added archetype enable flags to engine_config
   - Ensures only the archetype being tested is enabled

### Test Code
3. **bin/test_archetype_isolation_fix.py** (NEW - 205 lines)
   - Smoke test to verify archetype isolation
   - Tests H, K, S5 on single window
   - Confirms results are now different (not identical)

---

## Process Monitoring

### Current Status

**Process:** RUNNING
**PID:** 76289
**Log File:** `logs/walk_forward_fixed_20260117_201220.log`
**Size:** 2.2 MB (as of 20:20)
**Progress:** Processing archetype H, window ~8-9 (2021-02 data)

### How to Monitor Progress

```bash
# Check if process is still running
ps aux | grep walk_forward_production_engine | grep -v grep

# View recent log output
tail -100 logs/walk_forward_fixed_20260117_201220.log

# Count completed windows (each archetype has 24 windows)
grep "Window.*complete:" logs/walk_forward_fixed_20260117_201220.log | wc -l

# Check for errors
grep -i "error\|exception\|traceback" logs/walk_forward_fixed_20260117_201220.log
```

### Expected Output Files

When complete, results will be saved to:
```
results/walk_forward_YYYY-MM-DD/
├── B_walk_forward_results.json
├── S1_walk_forward_results.json
├── H_walk_forward_results.json
├── K_walk_forward_results.json
├── S5_walk_forward_results.json
├── S4_walk_forward_results.json
└── walk_forward_summary.csv
```

---

## Lessons Learned

### Why This Bug Went Undetected

1. **No unit tests for archetype isolation**
   - The FullEngineBacktest had no tests verifying enable flags work
   - Walk-forward script had no tests verifying single-archetype execution

2. **Plausible metrics**
   - OOS degradation of 54-74% is high but not impossible
   - Trade counts of 2,000+ seemed reasonable for portfolio
   - No automated checks for impossible patterns (identical equity curves)

3. **Complex system integration**
   - ArchetypeFactory → FullEngineBacktest → WalkForwardScript
   - Configuration passed through multiple layers
   - Easy to lose archetype enable flags in the chain

### Prevention for Future

1. **Add archetype isolation tests**
   - Test that enabling only one archetype produces unique results
   - Test that different archetypes produce different equity curves
   - Add to CI/CD pipeline

2. **Validation checks in walk-forward script**
   - Assert that archetypes produce different results
   - Warn if equity curves are suspiciously similar
   - Check trade count ratios (portfolio vs individual)

3. **Better configuration visibility**
   - Log which archetypes are enabled at engine creation
   - Log total active archetypes in each window
   - Surface configuration in debug output

---

## Recommendations

### Immediate (While Waiting for Results)

1. **Monitor for new bugs**
   - Watch for errors in log file
   - Check that trade counts are reasonable (not all 0, not all 2000+)
   - Verify S5 has low trade count (crisis-only)

2. **Prepare for low trade counts**
   - Some archetypes may have insufficient trades to validate
   - This is EXPECTED for selective archetypes
   - Consider alternative validation: precision/recall on signals, not just returns

3. **Review optimization methodology**
   - Verify in-sample windows match walk-forward train windows
   - Check if optimization used same data as walk-forward train set
   - Ensure no lookahead bias in optimization

### After Results Complete

1. **Compare new vs old results**
   - Calculate how trade counts changed (individual vs portfolio)
   - Identify archetypes with insufficient trades
   - Flag archetypes that still fail validation

2. **Root cause investigation for failures**
   - If archetypes still fail after fix, investigate why
   - Check if overfitting is real vs low sample size
   - Consider if regime detection has lookahead bias

3. **Implement recommendations from research report**
   - Add regularization penalties to TPE optimization
   - Implement Combinatorial Purged Cross-Validation (CPCV)
   - Consider no-fitting approach per Rob Carver

---

## Conclusion

The archetype isolation bug was a **critical system-level failure** that completely invalidated the original walk-forward validation results. The bug caused all archetypes to run together, producing portfolio-level metrics instead of individual archetype metrics.

**Status: FIXED**

The bug has been fixed and verified via smoke test. A corrected walk-forward validation is now running with proper archetype isolation. Results are expected in ~22 hours (2026-01-18 18:00).

The corrected results will provide the first valid assessment of which archetypes have real generalization ability vs. which were overfitting the in-sample data.

---

**Report Author:** Claude Code (Sonnet 4.5)
**Report Date:** 2026-01-17 20:30
**Validation Status:** Fix verified, re-run in progress
