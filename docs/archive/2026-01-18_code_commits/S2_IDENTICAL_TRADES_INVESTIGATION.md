# S2 Identical Trades Investigation Report

**Date:** 2025-11-20
**Issue:** All 50 Optuna trials produced identical trade counts (135.9 trades/year)
**Status:** ROOT CAUSE IDENTIFIED

---

## Executive Summary

**CRITICAL BUG CONFIRMED:** S2 optimization produced zero variance across 50 trials because:

1. **Runtime features are NOT being applied** - The optimizer calls `backtest_knowledge_v2.py` which does NOT run S2 runtime enrichment
2. **Params are written to config but never used** - S2 logic falls back to legacy detection which ignores `fusion_threshold`, `wick_ratio_min`, etc.
3. **Wrong backtest script** - Should call `backtest_s2_enriched.py` (applies enrichment) but calls generic `backtest_knowledge_v2.py`

**Result:** Every trial runs identical S2 logic → identical trades (54/54/94) → identical annual rate (135.9).

---

## Evidence Chain

### 1. Identical Trade Counts Across All Trials

From `results/s2_optimization.log`:

```
Trial 0: 2022_H1: trades=54, 2022_H2: trades=54, 2023_H1: trades=94
Trial 1: 2022_H1: trades=54, 2022_H2: trades=54, 2023_H1: trades=94
Trial 2: 2022_H1: trades=54, 2022_H2: trades=54, 2023_H1: trades=94
...
Trial 49: 2022_H1: trades=54, 2022_H2: trades=54, 2023_H1: trades=94
```

**Annual rate:** `(54+54+94) / (181+184+181) * 365 = 135.9 trades/year`

Every single trial was PRUNED for exceeding 30 trades/year threshold.

### 2. Parameters ARE Being Varied by Optuna

Search ranges from `results/s2_calibration/fusion_percentiles_2022.json`:

```json
{
  "fusion_threshold": [0.66, 0.725],
  "wick_ratio_min": [2.0, 4.0],
  "rsi_min": [75.0, 85.0],
  "volume_z_max": [-2.0, 0.0],
  "liquidity_max": [0.05, 0.25],
  "cooldown_bars": [4, 20]
}
```

Optuna is correctly sampling from these ranges, but the sampled values **have zero effect** on backtest results.

### 3. Config IS Being Created Correctly

From `bin/optimize_s2_calibration.py` lines 131-270:

```python
def create_s2_backtest_config(params: S2Parameters) -> Dict:
    return {
        "archetypes": {
            "enable_S2": True,
            "failed_rally": {
                "fusion_threshold": params.fusion_threshold,  # ✓ Params written
                "wick_ratio_min": params.wick_ratio_min,      # ✓ Params written
                "rsi_min": params.rsi_min,                    # ✓ Params written
                "use_runtime_features": True,                 # ✓ Feature flag set
                ...
            }
        }
    }
```

The config is correctly built from Optuna trial parameters.

### 4. Wrong Backtest Script Is Called

From `bin/optimize_s2_calibration.py` lines 292-299:

```python
cmd = [
    'python3',
    'bin/backtest_knowledge_v2.py',  # ❌ WRONG SCRIPT
    '--asset', 'BTC',
    '--start', fold['start'],
    '--end', fold['end'],
    '--config', config_path
]
```

**Problem:** `backtest_knowledge_v2.py` does NOT apply S2 runtime enrichment!

### 5. S2 Runtime Enrichment Is Never Applied

From `engine/strategies/archetypes/bear/failed_rally_runtime.py`:

The enrichment module exists and implements:
- `wick_upper_ratio` calculation
- `volume_fade_flag` detection
- `rsi_bearish_div` divergence analysis
- `ob_retest_flag` resistance detection

But checking `bin/backtest_knowledge_v2.py`:

```bash
$ grep -n "S2Runtime\|apply_runtime_enrichment\|failed_rally_runtime" bin/backtest_knowledge_v2.py
# NO MATCHES FOUND
```

**The generic backtest script NEVER applies S2 enrichment.**

### 6. S2 Falls Back to Legacy Detection

From `engine/archetypes/logic_v2_adapter.py` lines 1275-1371:

```python
def _check_S2(self, context: RuntimeContext) -> Tuple[bool, float, Dict]:
    use_runtime_features = context.get_threshold('failed_rally', 'use_runtime_features', False)

    if use_runtime_features:
        # Enhanced logic with runtime features
        wick_upper_ratio = self.g(context.row, 'wick_upper_ratio', None)
        ...
        if all(x is not None for x in [wick_upper_ratio, ...]):
            return self._check_S2_enhanced(...)  # ✓ Uses thresholds

    # FALLBACK: Original logic (HARDCODED THRESHOLDS)
    wick_ratio_min = 2.0  # ❌ HARDCODED (ignores config)
    ob_high = self.g(context.row, 'tf1h_ob_high', None)
    ...
```

**The legacy path uses HARDCODED thresholds**, completely ignoring `fusion_threshold`, `wick_ratio_min`, etc.

---

## Root Cause Analysis

### Why Parameters Don't Affect Results

**Flow diagram:**

```
Optuna Trial
  ↓
create_s2_backtest_config(params) → writes params to temp config ✓
  ↓
backtest_knowledge_v2.py --config temp_config.json
  ↓
[NO S2 ENRICHMENT APPLIED] ❌
  ↓
S2 detection logic checks: use_runtime_features = True
  ↓
Tries to read: wick_upper_ratio, volume_fade_flag, etc.
  ↓
[FEATURES NOT FOUND IN DATAFRAME] ❌
  ↓
Falls back to legacy logic (lines 1295-1371)
  ↓
Uses HARDCODED thresholds (wick_ratio_min=2.0, etc.) ❌
  ↓
Produces IDENTICAL trade count every time
```

### The Missing Link

The optimizer should call `bin/backtest_s2_enriched.py` which:

1. Loads feature data
2. **Applies S2 runtime enrichment** (adds wick_upper_ratio, etc.)
3. Runs backtest with enriched data
4. Thresholds are read from config and ACTUALLY USED

But instead it calls `backtest_knowledge_v2.py` which skips step 2.

---

## Detailed Code Path Analysis

### Working Path (Not Used)

```python
# bin/backtest_s2_enriched.py (lines 31-87)
from engine.strategies.archetypes.bear.failed_rally_runtime import apply_runtime_enrichment

df = pd.read_parquet(feature_path)
df_enriched = apply_runtime_enrichment(df, lookback=14)  # ✓ Adds runtime features

# Now when S2 detection runs:
# - wick_upper_ratio EXISTS in dataframe
# - volume_fade_flag EXISTS in dataframe
# - S2 enhanced logic can use configurable thresholds
```

### Broken Path (Currently Used)

```python
# bin/optimize_s2_calibration.py calls backtest_knowledge_v2.py
# backtest_knowledge_v2.py loads features directly:

df = pd.read_parquet(feature_path)  # ❌ NO enrichment

# When S2 detection runs:
# - wick_upper_ratio MISSING from dataframe
# - volume_fade_flag MISSING from dataframe
# - Falls back to legacy hardcoded logic
```

---

## Proof of Zero Variance

### Statistical Evidence

**Perfect correlation** across 50 trials:

| Metric | Trial 0 | Trial 25 | Trial 49 | Variance |
|--------|---------|----------|----------|----------|
| 2022_H1 Trades | 54 | 54 | 54 | **0.0** |
| 2022_H2 Trades | 54 | 54 | 54 | **0.0** |
| 2023_H1 Trades | 94 | 94 | 94 | **0.0** |
| Win Rate (2022_H1) | 31.5% | 31.5% | 31.5% | **0.0** |
| Profit Factor (2022_H1) | 0.32 | 0.32 | 0.32 | **0.0** |

**Probability this is random:** `P(identical 50 times) ≈ 0.0` (astronomical odds)

---

## Why This Went Undetected

### Silent Failure Mode

1. **Config validation passes** - The config JSON is syntactically correct
2. **No errors raised** - The fallback logic is a FEATURE (graceful degradation)
3. **Trials complete successfully** - Backtest runs without exceptions
4. **Metrics look reasonable** - PF, WR, trades are in plausible ranges

The only clue was **identical output**, which requires looking at ALL trials.

### Design Flaw

The `use_runtime_features` flag creates a **silent failure mode**:

```python
if use_runtime_features:
    # Try to use enhanced features
    if features_available:
        use_enhanced_logic()  # ✓ Uses config thresholds
    else:
        fall_back_to_legacy()  # ❌ Uses hardcoded thresholds (SILENT)
```

**No warning is logged** when falling back! The code assumes features might be missing and degrades gracefully.

---

## Fix Implementation

### Option 1: Use Correct Backtest Script (Quick Fix)

**Change:** `bin/optimize_s2_calibration.py` line 294

```python
# BEFORE (broken)
cmd = ['python3', 'bin/backtest_knowledge_v2.py', ...]

# AFTER (fixed)
cmd = ['python3', 'bin/backtest_s2_enriched.py', ...]
```

**Pros:**
- One line change
- Uses existing infrastructure
- Will immediately produce variance

**Cons:**
- Requires separate backtest script for each archetype
- Doesn't scale to multi-archetype optimization

### Option 2: Apply Enrichment in backtest_knowledge_v2.py (Proper Fix)

**Change:** Add runtime enrichment hook to main backtest

```python
# bin/backtest_knowledge_v2.py (after loading features)

# Check if S2 is enabled
if runtime_config.get('archetypes', {}).get('enable_S2', False):
    s2_config = runtime_config['archetypes'].get('failed_rally', {})
    if s2_config.get('use_runtime_features', False):
        from engine.strategies.archetypes.bear.failed_rally_runtime import apply_runtime_enrichment
        logger.info("[S2] Applying runtime feature enrichment...")
        df = apply_runtime_enrichment(df, lookback=14)
```

**Pros:**
- Cleaner architecture
- Works for all optimizers automatically
- Scales to other runtime-enriched archetypes

**Cons:**
- Requires backtest modification
- Adds complexity to main backtest loop

### Option 3: Log Warning on Fallback (Diagnostic Fix)

**Change:** `engine/archetypes/logic_v2_adapter.py` line 1293

```python
if use_runtime_features:
    wick_upper_ratio = self.g(context.row, 'wick_upper_ratio', None)
    ...
    if all(x is not None for x in [wick_upper_ratio, ...]):
        return self._check_S2_enhanced(...)
    else:
        # ✓ ADD WARNING
        if not hasattr(self, '_s2_fallback_warned'):
            logger.warning("[S2] Runtime features not found! Falling back to legacy logic (thresholds will be ignored)")
            self._s2_fallback_warned = True
```

**Pros:**
- Prevents silent failures in future
- Easy diagnostic tool

**Cons:**
- Doesn't fix the optimizer (still needs Option 1 or 2)

---

## Recommended Fix (Hybrid Approach)

**Phase 1 (Immediate):** Option 1 - Fix optimizer to call correct script

```bash
# Edit bin/optimize_s2_calibration.py line 294
sed -i '' 's/backtest_knowledge_v2.py/backtest_s2_enriched.py/' bin/optimize_s2_calibration.py
```

**Phase 2 (Short-term):** Option 3 - Add fallback warning

Prevents future silent failures during development.

**Phase 3 (Long-term):** Option 2 - Integrate enrichment into main backtest

Cleaner architecture for production use.

---

## Validation Plan

### Test 1: Verify Fix Produces Variance

```bash
# After applying fix, run 10 trials
python3 bin/optimize_s2_calibration.py --trials 10

# Expected:
# - Trade counts should VARY across trials
# - At least 3 different trade count patterns
# - Variance in PF, WR, DD
```

### Test 2: Verify Thresholds Actually Work

```bash
# Manual test with extreme thresholds
# Config A: fusion_threshold=0.90 (ultra-strict)
# Config B: fusion_threshold=0.55 (relaxed)

# Expected:
# - Config A: 0-10 trades (very few signals)
# - Config B: 200+ trades (many signals)
```

### Test 3: Verify Runtime Features Are Present

```bash
# Add debug logging to _check_S2_enhanced
logger.info(f"[S2 DEBUG] wick_upper_ratio={wick_upper_ratio:.3f}")

# Expected in backtest log:
# [S2 DEBUG] wick_upper_ratio=0.652
# [S2 DEBUG] wick_upper_ratio=0.234
# ...
```

---

## Impact Assessment

### Wasted Compute

- **50 trials × 3 folds × ~6 seconds = 900 seconds (15 minutes)**
- All trials were running **identical code path**
- Zero information gain from trials 1-49

### Optimizer Integrity

- **Optuna study is CORRUPTED** - All trials have identical results
- Pareto frontier is meaningless (all points are the same)
- Must **delete database and restart** after fix

### Configuration Trust

- S2 configs generated from this study are **invalid**
- Recommended thresholds are arbitrary (not optimized)
- Any downstream work using these configs must be **re-validated**

---

## Lessons Learned

### 1. Silent Fallbacks Are Dangerous

**Principle:** If a feature is REQUIRED for correctness, FAIL LOUD instead of degrading silently.

```python
# BAD (current code)
if features_available:
    use_features()
else:
    fall_back()  # Silent degradation

# GOOD (defensive)
if not features_available and features_required:
    raise RuntimeError("Runtime features required but not found!")
```

### 2. Validate Optimizer Integration

**Principle:** Add sanity checks to optimizer objective function.

```python
# Add to objective() function:
if trial.number > 0:
    # Compare to previous trial
    prev_trades = get_previous_trial_trades()
    if current_trades == prev_trades:
        logger.warning(f"Trial {trial.number} has IDENTICAL trades to previous trial!")
```

### 3. Test End-to-End Before Large Runs

**Principle:** Run 3-trial smoke test with EXTREME params to verify variance.

```python
# Smoke test configs:
# Trial A: fusion_threshold=0.99 (expect 0 trades)
# Trial B: fusion_threshold=0.01 (expect 500+ trades)
# Trial C: fusion_threshold=0.50 (expect ~50 trades)

# If all produce same trades → ABORT optimization
```

---

## Next Steps

### Immediate (Next 1 Hour)

1. ✅ Create this investigation report
2. ⏳ Apply Option 1 fix (1-line change to optimizer)
3. ⏳ Delete corrupted Optuna database
4. ⏳ Run 3-trial smoke test with extreme params
5. ⏳ Verify variance is present

### Short-term (Next 1 Day)

6. Apply Option 3 fix (add fallback warning)
7. Re-run full 50-trial optimization
8. Validate Pareto frontier has diversity
9. Generate new S2 configs from valid study

### Long-term (Next 1 Week)

10. Apply Option 2 fix (integrate enrichment into main backtest)
11. Add optimizer sanity checks (detect identical trials)
12. Document runtime enrichment architecture
13. Add integration tests for S2 optimization flow

---

## Appendix: Code Snippets

### A. Current Optimizer Call (Broken)

```python
# bin/optimize_s2_calibration.py lines 292-307
cmd = [
    'python3',
    'bin/backtest_knowledge_v2.py',  # ❌ No enrichment
    '--asset', 'BTC',
    '--start', fold['start'],
    '--end', fold['end'],
    '--config', config_path
]

result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
```

### B. Fixed Optimizer Call

```python
# bin/optimize_s2_calibration.py lines 292-307 (FIXED)
cmd = [
    'python3',
    'bin/backtest_s2_enriched.py',  # ✓ Applies enrichment
    '--asset', 'BTC',
    '--start', fold['start'],
    '--end', fold['end'],
    '--config', config_path,
    '--lookback', '14'  # ✓ Pass lookback param
]

result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
```

### C. Fallback Detection Warning

```python
# engine/archetypes/logic_v2_adapter.py line 1290 (ADD THIS)
if use_runtime_features:
    wick_upper_ratio = self.g(context.row, 'wick_upper_ratio', None)
    volume_fade_flag = self.g(context.row, 'volume_fade_flag', None)
    rsi_bearish_div = self.g(context.row, 'rsi_bearish_div', None)
    ob_retest_flag = self.g(context.row, 'ob_retest_flag', None)

    if all(x is not None for x in [wick_upper_ratio, volume_fade_flag, rsi_bearish_div, ob_retest_flag]):
        return self._check_S2_enhanced(...)
    else:
        # ✓ ADD WARNING ON FALLBACK
        missing = [name for name, val in [
            ('wick_upper_ratio', wick_upper_ratio),
            ('volume_fade_flag', volume_fade_flag),
            ('rsi_bearish_div', rsi_bearish_div),
            ('ob_retest_flag', ob_retest_flag)
        ] if val is None]

        if not hasattr(self, '_s2_fallback_logged'):
            logger.error(f"[S2 FALLBACK] Runtime features missing: {missing}")
            logger.error(f"[S2 FALLBACK] Config thresholds will be IGNORED!")
            logger.error(f"[S2 FALLBACK] Using legacy hardcoded logic instead")
            self._s2_fallback_logged = True
```

### D. Smoke Test Script

```python
#!/usr/bin/env python3
"""Smoke test for S2 optimizer variance"""

import subprocess
import json

configs = [
    {"fusion_threshold": 0.99, "wick_ratio_min": 5.0, "name": "ultra_strict"},
    {"fusion_threshold": 0.01, "wick_ratio_min": 1.0, "name": "ultra_relaxed"},
    {"fusion_threshold": 0.50, "wick_ratio_min": 2.5, "name": "moderate"}
]

results = {}

for cfg in configs:
    # Create temp config
    with open(f'/tmp/smoke_{cfg["name"]}.json', 'w') as f:
        json.dump({"archetypes": {"failed_rally": cfg}}, f)

    # Run backtest
    cmd = [
        'python3', 'bin/backtest_s2_enriched.py',
        '--config', f'/tmp/smoke_{cfg["name"]}.json',
        '--start', '2022-01-01',
        '--end', '2022-06-30'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Extract trade count
    for line in result.stdout.split('\n'):
        if 'Total Trades:' in line:
            trades = int(line.split(':')[1].strip())
            results[cfg['name']] = trades
            break

# Validate variance
print("Smoke Test Results:")
print(json.dumps(results, indent=2))

if len(set(results.values())) == 1:
    print("\n❌ FAILED: All configs produced identical trades!")
    exit(1)
elif results['ultra_strict'] >= results['moderate']:
    print("\n❌ FAILED: Strict config produced more trades than moderate!")
    exit(1)
elif results['ultra_relaxed'] <= results['moderate']:
    print("\n❌ FAILED: Relaxed config produced fewer trades than moderate!")
    exit(1)
else:
    print("\n✅ PASSED: Configs produce expected variance")
    exit(0)
```

---

## Conclusion

**Root Cause:** Wrong backtest script called → runtime enrichment never applied → fallback to hardcoded logic → zero variance.

**Fix:** One-line change to call `backtest_s2_enriched.py` instead of `backtest_knowledge_v2.py`.

**Validation:** Run smoke test with extreme params to verify variance before full optimization run.

**Impact:** All 50 trials wasted, must re-run after fix. Lessons learned about silent fallbacks and optimizer validation.
