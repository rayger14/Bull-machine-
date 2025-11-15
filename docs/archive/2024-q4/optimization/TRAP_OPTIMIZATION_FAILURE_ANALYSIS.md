# Trap Optimization v1 - Failure Analysis

**Date**: 2025-11-06
**Status**: FAILED - Zero variance across all 200 trials
**Optimization completed**: ✅ (8+ hours runtime)
**Results**: ❌ All trials identical score (0.36493200337296494)

---

## 🚨 CRITICAL ISSUE: Parameters Not Connected to Archetype Logic

### The Problem

All 200 Optuna trials produced **exactly the same score** because the parameters being optimized are not actually used by the archetype detection logic.

### Root Cause Analysis

#### 1. **What We Thought We Were Optimizing**

The optimizer set these config parameters:
```python
config['archetypes']['trap_within_trend'] = {
    'quality_threshold': 0.45-0.65,      # ← NOT USED
    'confirmation_bars': 2-5,            # ← NOT USED
    'volume_ratio': 1.5-2.5,             # ← NOT USED
    'stop_multiplier': 0.8-1.5           # ← NOT USED (maybe used in exits?)
}
```

#### 2. **What The Archetype Actually Reads**

The `trap_within_trend` archetype (archetype 'H') is implemented in `engine/archetypes/logic.py:674`:

```python
def _check_H(self, row, prev_row, df, index, fusion_score) -> bool:
    """H - Trap Within Trend"""

    # HARDCODED THRESHOLD
    tf4h_fusion = row.get('tf4h_fusion_score', 0.0)
    if tf4h_fusion <= 0.5:  # ← HARDCODED, NOT FROM CONFIG
        return False

    # Reads from self.thresh_H, NOT from config['archetypes']
    liquidity = self._get_liquidity_score(row)
    if liquidity >= self.thresh_H.get('liq_drop', 0.30):  # ← Different location
        return False

    adx = row.get('adx_14', 0.0)
    if adx <= self.thresh_H.get('adx', 25.0):  # ← Different location
        return False

    # More hardcoded logic for wick detection...

    if fusion_score < self.thresh_H.get('fusion', 0.35):  # ← Different location
        return False
```

#### 3. **The Disconnect**

- **Optimizer writes to**: `config['archetypes']['trap_within_trend']`
- **Archetype reads from**: `self.thresh_H` (from constructor)
- **Result**: Changing config parameters has **zero effect** on archetype detection

### Evidence

```bash
$ python3 -c "
import pandas as pd
df = pd.read_csv('results/optuna_trap_v10_full/trials.csv')
print('Unique scores:', df['value'].nunique())
print('Std dev:', df['value'].std())
"

Unique scores: 1
Std dev: 5.565045e-17  # Essentially zero (floating point noise)
```

All 200 trials with wildly different parameters produced identical backtests.

---

## 🔍 Where thresh_H Actually Comes From

Looking at `engine/archetypes/logic.py:57`:

```python
def __init__(self, thresholds: dict, enabled: dict):
    # ...
    self.thresh_H = thresholds.get('H', {})
```

The thresholds come from a separate `thresholds` parameter passed during initialization, NOT from the `config['archetypes']` section.

### Checking Baseline Config Structure

```bash
$ cat configs/baseline_btc_bull_pf20.json | jq '.archetypes.trap_within_trend'
null
```

The baseline config has **no archetype-specific section** at all!

---

## 💡 Why This Wasn't Caught Earlier

1. **No validation errors**: The optimizer ran successfully for 8+ hours
2. **Configs looked correct**: The `trap_optimized_bull.json` file was generated with parameters
3. **RouterV10 accepted configs**: No complaints from the backtest engine
4. **Metrics were plausible**: Training PF 0.88, Val PF 3.37 looked reasonable

The system silently ignored the parameters we set because they were in the wrong location.

---

## 🎯 The Actual Configurable Parameters

Based on the archetype logic, the **actual** parameters that affect trap_within_trend detection are:

1. `thresh_H['liq_drop']` (default: 0.30) - Maximum liquidity score
2. `thresh_H['adx']` (default: 25.0) - Minimum ADX
3. `thresh_H['fusion']` (default: 0.35) - Minimum fusion score
4. Hardcoded `tf4h_fusion > 0.5` threshold
5. Hardcoded wick detection logic (wick > 2× body)

### Where stop_multiplier Might Be Used

The `stop_multiplier` parameter we were optimizing *might* be used in exit logic, but NOT in entry detection. Even if it affects stop placement, it wouldn't explain why ALL trials had identical scores.

---

## 📊 Wasted Compute Analysis

- **Runtime**: 8 hours 19 minutes
- **Trials**: 200
- **CPU time per trial**: ~2.5 minutes
- **Total CPU time**: ~500 minutes (8.3 hours)
- **Actual optimization performed**: **ZERO**
- **Result**: All trials were duplicates of the baseline

---

## 🔧 How To Fix This

### Option 1: Use threshold_policy (Existing System)

The codebase has a `threshold_policy` system that IS read by archetypes:

```python
# In engine/archetypes/logic_v2_adapter.py
fusion_th = ctx.get_threshold('trap_within_trend', 'fusion', 0.36)
```

This reads from a different config structure. We need to:

1. Understand how `threshold_policy` configs work
2. Modify optimizer to set thresholds via this system
3. Test that changes actually affect detection

### Option 2: Fix The Archetype Logic (Recommended)

Modify `_check_H` in `logic.py` to read from the archetype config:

```python
def _check_H(self, row, prev_row, df, index, fusion_score) -> bool:
    """H - Trap Within Trend with configurable params"""

    # Read from archetype config (passed during init)
    archetype_config = self.archetype_configs.get('trap_within_trend', {})

    quality_threshold = archetype_config.get('quality_threshold', 0.5)
    adx_threshold = archetype_config.get('adx_threshold', 25.0)
    liquidity_threshold = archetype_config.get('liquidity_threshold', 0.30)
    fusion_threshold = archetype_config.get('fusion_threshold', 0.35)

    # Now use these configurable thresholds
    if tf4h_fusion <= quality_threshold:
        return False
    # ... etc
```

This requires:
1. Passing archetype configs to ArchetypeLogic constructor
2. Storing them in self.archetype_configs
3. Reading them in detection methods

### Option 3: Bypass Detection, Optimize Exits Only

Maybe the trap archetype detection is intentionally hardcoded, and only exit parameters (stops, targets) are meant to be tunable. In that case:

1. Accept that entry detection is fixed
2. Optimize only `stop_multiplier`, `target_multiplier`, `partial_exits`, etc.
3. Run a focused optimization on exit parameters

---

## 🚫 Why v2 Optimizer Won't Help

The improved trap optimizer v2 (`bin/optuna_trap_v2.py`) I built has:
- Fixed sizing ✅
- Rolling OOS validation ✅
- Better objective function ✅
- Feature caching ✅

**BUT** it still suffers from the same fundamental issue: it sets parameters in the wrong config location that the archetype logic doesn't read.

Running v2 right now would just waste another 6-8 hours of compute producing 200 identical results.

---

## 📋 Immediate Action Plan

### Step 1: Investigation (30 minutes)

1. Search codebase for how `threshold_policy` works
2. Check if `stop_multiplier` is used anywhere in exit logic
3. Verify which archetype logic file is actually being used (logic.py vs logic_v2_adapter.py)
4. Document the correct config structure

### Step 2: Diagnostic Test (15 minutes)

Create a minimal test that:
1. Runs baseline config
2. Runs config with dramatically different "trap" parameters
3. Compares trade counts and confirms they differ
4. If they don't differ → parameters definitely not connected

### Step 3: Fix Implementation (2-4 hours)

Based on investigation, choose:
- **Fix archetype logic** to read from config['archetypes']['trap_within_trend']
- **Use threshold_policy** system and update optimizer accordingly
- **Optimize exits only** and skip entry detection tuning

### Step 4: Re-run Optimization

Only after confirming that parameter changes actually affect backtest results.

---

## 💰 Cost-Benefit Analysis

### If We Fix This

- **Time to fix**: 2-4 hours (investigation + code changes + testing)
- **Time to re-run**: 6-8 hours (200 trials with v2 optimizer)
- **Chance of success**: HIGH (if parameters are properly connected)
- **Expected gain**: +$400-600/year (from original Phase 1 goals)

### If We Skip This

- **Time saved**: 8-12 hours
- **Move to**: OB retest scaling or bear optimization
- **Accept**: Trap archetype stays broken (46% WR, -$353 PNL)
- **Impact**: Miss ~35% of potential Phase 1 gains

---

## 🎓 Lessons Learned

1. **Always validate parameter connectivity**: Test that changing a parameter actually changes behavior
2. **Start with 5-10 trial smoke test**: Would have caught this in 30 minutes instead of 8+ hours
3. **Check for zero variance early**: If first 20 trials are identical, stop immediately
4. **Understand the codebase first**: We optimized against the wrong API
5. **Document config schemas**: Need clear docs on what lives where

---

## 📝 Recommended Next Steps

### Immediate (Today)

1. Run diagnostic test to confirm stop_multiplier doesn't work either
2. Investigate threshold_policy system
3. Decide: Fix & re-run OR skip trap for now

### If Fixing (This Week)

1. Implement archetype config reading
2. Test parameter connectivity with 5-trial smoke test
3. Run full 200-trial optimization with v2 (with fixes)
4. Validate results

### If Skipping (Move On)

1. Document this failure for future reference
2. Move to OB retest scaling (different approach)
3. Or move to bear optimization (2022 loss reduction)
4. Return to trap optimization in Phase 2 with better tools

---

**Generated**: 2025-11-06
**Status**: Complete failure analysis
**Recommendation**: Fix parameter connectivity before any re-run
**Next Action**: Diagnostic test + decision on fix vs skip
