# S2 Optimizer Fix - Implementation Guide

**Issue:** Zero variance across 50 trials because runtime enrichment not applied

**Root Cause:** `bin/optimize_s2_calibration.py` calls `backtest_knowledge_v2.py` which doesn't apply S2 enrichment

**Fix Complexity:** MEDIUM (backtest_s2_enriched.py uses wrong import pattern)

---

## Problem Analysis

### What's Broken

```python
# bin/optimize_s2_calibration.py line 292-299
cmd = [
    'python3',
    'bin/backtest_knowledge_v2.py',  # ❌ No S2 enrichment
    '--asset', 'BTC',
    '--start', fold['start'],
    '--end', fold['end'],
    '--config', config_path
]
```

### Why backtest_s2_enriched.py Won't Work

```python
# bin/backtest_s2_enriched.py line 122
from bin.backtest_knowledge_v2 import run_backtest  # ❌ Function doesn't exist
```

The main backtest script doesn't export a `run_backtest()` function. It's structured as a CLI script with `if __name__ == '__main__'`.

---

## Recommended Fix: Add Enrichment Hook to Main Backtest

### Step 1: Modify backtest_knowledge_v2.py

Add enrichment hook AFTER feature loading (around line 2580):

```python
# bin/backtest_knowledge_v2.py (add after df is loaded and filtered)

# ============================================================================
# PR#6D: Apply S2 runtime enrichment if enabled
# ============================================================================
if runtime_config and runtime_config.get('archetypes', {}).get('enable_S2', False):
    s2_config = runtime_config['archetypes'].get('failed_rally', {})
    if s2_config.get('use_runtime_features', False):
        try:
            from engine.strategies.archetypes.bear.failed_rally_runtime import apply_runtime_enrichment
            logger.info("[S2] Applying runtime feature enrichment...")
            lookback = s2_config.get('lookback_window', 14)
            df = apply_runtime_enrichment(df, lookback=lookback)
            logger.info(f"[S2] Enriched {len(df)} bars with runtime features")
        except Exception as e:
            logger.error(f"[S2] Runtime enrichment failed: {e}")
            logger.error("[S2] Continuing with legacy S2 logic (thresholds will be ignored)")
```

### Step 2: Test Fix with Smoke Test

```bash
# Create test configs with extreme parameters
python3 -c "
import json

# Ultra-strict (should fire rarely)
config_strict = {
    'archetypes': {
        'enable_S2': True,
        'failed_rally': {
            'fusion_threshold': 0.90,
            'wick_ratio_min': 5.0,
            'use_runtime_features': True
        }
    }
}

# Ultra-relaxed (should fire often)
config_relaxed = {
    'archetypes': {
        'enable_S2': True,
        'failed_rally': {
            'fusion_threshold': 0.10,
            'wick_ratio_min': 0.5,
            'use_runtime_features': True
        }
    }
}

with open('/tmp/s2_strict.json', 'w') as f:
    json.dump(config_strict, f, indent=2)

with open('/tmp/s2_relaxed.json', 'w') as f:
    json.dump(config_relaxed, f, indent=2)

print('✓ Created test configs')
"

# Run smoke tests
python3 bin/backtest_knowledge_v2.py --asset BTC --start 2022-01-01 --end 2022-06-30 --config /tmp/s2_strict.json | grep "Total Trades"
python3 bin/backtest_knowledge_v2.py --asset BTC --start 2022-01-01 --end 2022-06-30 --config /tmp/s2_relaxed.json | grep "Total Trades"

# Expected:
# Strict: 0-10 trades
# Relaxed: 100+ trades
# If both show 54 trades → fix didn't work
```

### Step 3: Re-run Optimizer

```bash
# Delete corrupted Optuna database
rm results/s2_calibration/optuna_s2_calibration.db

# Run 10-trial test
python3 bin/optimize_s2_calibration.py --trials 10 --timeout 3600

# Verify variance:
python3 -c "
import sqlite3
import pandas as pd

db = sqlite3.connect('results/s2_calibration/optuna_s2_calibration.db')
df = pd.read_sql_query('''
    SELECT trial_id,
           json_extract(value, '\$.user_attrs.2022_H1_trades') as trades_h1,
           json_extract(value, '\$.user_attrs.2022_H2_trades') as trades_h2
    FROM trials
''', db)

print(df)

# Check variance
variance = df['trades_h1'].var()
print(f'\\nTrade count variance: {variance:.2f}')

if variance == 0:
    print('❌ FAILED: Still zero variance!')
else:
    print('✅ PASSED: Variance detected!')
"
```

---

## Alternative Quick Fix (Less Elegant)

If modifying the main backtest is too risky, use a wrapper script:

### Create bin/optimize_s2_calibration_fixed.py

```python
#!/usr/bin/env python3
"""
S2 Optimizer with Runtime Enrichment
Wrapper that applies enrichment before calling main backtest
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import subprocess
import tempfile
import json
from engine.strategies.archetypes.bear.failed_rally_runtime import apply_runtime_enrichment

# ... (copy rest of optimize_s2_calibration.py)

def run_backtest_with_enrichment(config: Dict, fold: Dict, trial_num: int) -> FoldMetrics:
    """
    Run backtest with S2 enrichment applied first.
    """
    # 1. Load features
    feature_path = f"data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet"
    df = pd.read_parquet(feature_path)

    # Filter to fold
    start_ts = pd.Timestamp(fold['start'], tz='UTC')
    end_ts = pd.Timestamp(fold['end'], tz='UTC')
    df_fold = df[(df.index >= start_ts) & (df.index <= end_ts)].copy()

    # 2. Apply S2 enrichment
    lookback = config['archetypes']['failed_rally'].get('lookback_window', 14)
    df_enriched = apply_runtime_enrichment(df_fold, lookback=lookback)

    # 3. Save to temp file
    temp_features = f"/tmp/s2_enriched_trial{trial_num}_fold{fold['name']}.parquet"
    df_enriched.to_parquet(temp_features)

    # 4. Write config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        config_path = f.name

    # 5. Run backtest (needs modification to accept --features-path)
    cmd = [
        'python3',
        'bin/backtest_knowledge_v2.py',
        '--asset', 'BTC',
        '--start', fold['start'],
        '--end', fold['end'],
        '--config', config_path,
        # TODO: Add --features-path argument to backtest
    ]

    # ... (rest same as original)
```

**Problem:** This requires adding `--features-path` argument to backtest, making it equally invasive.

---

## Recommended Approach: Main Backtest Modification

**Reason:** Cleaner, works for all future optimizations, minimal risk

**Location:** `/Users/raymondghandchi/Bull-machine-/Bull-machine-/bin/backtest_knowledge_v2.py` line ~2580

**Code to add:**

```python
# After loading df and before creating backtest instance:

# ============================================================================
# PR#6D: Apply archetype-specific runtime enrichment
# ============================================================================
if runtime_config and 'archetypes' in runtime_config:
    archetypes = runtime_config['archetypes']

    # S2 (Failed Rally) enrichment
    if archetypes.get('enable_S2', False):
        s2_cfg = archetypes.get('failed_rally', {})
        if s2_cfg.get('use_runtime_features', False):
            try:
                from engine.strategies.archetypes.bear.failed_rally_runtime import apply_runtime_enrichment
                print("[S2] Applying runtime enrichment...")
                df = apply_runtime_enrichment(df, lookback=s2_cfg.get('lookback_window', 14))
                print(f"[S2] ✓ Enriched {len(df)} bars")
            except Exception as e:
                print(f"[S2] ✗ Enrichment failed: {e}")

    # TODO: Add similar blocks for other runtime-enriched archetypes
```

---

## Validation Checklist

- [ ] Smoke test shows variance (strict: <10 trades, relaxed: >100 trades)
- [ ] Optimizer produces different trade counts across trials
- [ ] Optuna study has > 0 variance in `mean_annual_trades`
- [ ] Log shows `[S2] Applying runtime enrichment...` message
- [ ] Features `wick_upper_ratio`, `volume_fade_flag` present in archetype detection
- [ ] No `[S2 FALLBACK]` warnings in log

---

## Timeline

**Estimated Time:** 30 minutes

1. Modify backtest (5 min)
2. Run smoke test (5 min)
3. Delete Optuna DB and restart (1 min)
4. Run 10-trial validation (15 min)
5. Verify variance and inspect logs (4 min)

---

## Rollback Plan

If fix breaks existing functionality:

```bash
# Revert backtest changes
git checkout bin/backtest_knowledge_v2.py

# Use old optimizer (without S2 enrichment requirement)
# Disable use_runtime_features in config
```

---

## Next Steps After Fix

1. Complete 50-trial optimization with working variance
2. Validate Pareto frontier has diversity
3. Generate S2 configs from valid study
4. Compare performance to baseline (PF, WR, trades)
5. Document runtime enrichment pattern for future archetypes
