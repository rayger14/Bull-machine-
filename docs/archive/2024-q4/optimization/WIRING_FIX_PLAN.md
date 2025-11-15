# Wiring Fix Plan - No More Endless Loops

**Created**: 2025-11-06
**Status**: READY TO EXECUTE
**Time Budget**: 4 hours total (hard stop)
**Goal**: Fix parameter wiring + add fail-fast guardrails

---

## 🚨 PROBLEM SUMMARY

**What happened**: 8 hours of optimization, 200 trials, **zero variance** because:
- Optimizer wrote: `config['archetypes']['trap_within_trend']['quality_threshold']`
- Code read: `self.thresh_H.get('liq_drop', 0.30)` ← different location
- Result: All parameters ignored

**Why it matters**: Same issue likely affects OB retest and other archetypes. Without fixing the wiring system, **every future optimization risks the same failure**.

---

## ✅ THE FIX (Time-Boxed, 4 Hours)

### Hour 1: Single Source of Truth (60 min)

#### 1.1 Create Parameter Accessor (15 min)

```python
# engine/archetypes/param_accessor.py
"""Single source of truth for archetype parameters."""

def get_archetype_param(config: dict, archetype: str, key: str, default):
    """
    Read archetype parameter from config.

    Args:
        config: Full config dict
        archetype: Archetype name (e.g., 'trap_within_trend')
        key: Parameter key (e.g., 'quality_threshold')
        default: Default value if not found

    Returns:
        Parameter value from config or default

    Usage:
        quality_th = get_archetype_param(config, 'trap_within_trend', 'quality_threshold', 0.55)
    """
    return config.get('archetypes', {}).get(archetype, {}).get(key, default)


def log_params_used(config: dict, archetype: str, output_path: str):
    """
    Log all parameters actually used by an archetype.

    Creates a params_used.json file for audit trail.
    """
    import json
    from pathlib import Path

    arch_config = config.get('archetypes', {}).get(archetype, {})

    params_snapshot = {
        'archetype': archetype,
        'params': arch_config,
        'timestamp': datetime.now().isoformat()
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(params_snapshot, f, indent=2)

    print(f"[PARAMS] {archetype}: {arch_config}")
```

#### 1.2 Refactor Trap Archetype Logic (30 min)

```python
# engine/archetypes/logic.py (modify _check_H)

def _check_H(self, row, prev_row, df, index, fusion_score, config: dict) -> bool:
    """
    H - Trap Within Trend (NOW READS FROM CONFIG)

    Configurable parameters:
    - quality_threshold: HTF fusion minimum (default: 0.55)
    - liquidity_threshold: Max liquidity score (default: 0.30)
    - adx_threshold: Minimum ADX (default: 25.0)
    - fusion_threshold: Minimum fusion score (default: 0.35)
    - wick_multiplier: Wick size vs body (default: 2.0)
    """
    from engine.archetypes.param_accessor import get_archetype_param

    # READ FROM CONFIG (not hardcoded!)
    quality_th = get_archetype_param(config, 'trap_within_trend', 'quality_threshold', 0.55)
    liquidity_th = get_archetype_param(config, 'trap_within_trend', 'liquidity_threshold', 0.30)
    adx_th = get_archetype_param(config, 'trap_within_trend', 'adx_threshold', 25.0)
    fusion_th = get_archetype_param(config, 'trap_within_trend', 'fusion_threshold', 0.35)
    wick_mult = get_archetype_param(config, 'trap_within_trend', 'wick_multiplier', 2.0)

    # Now use these configurable thresholds
    tf4h_fusion = row.get('tf4h_fusion_score', 0.0)
    if tf4h_fusion <= quality_th:  # ← WAS HARDCODED 0.5
        return False

    liquidity = self._get_liquidity_score(row)
    if liquidity >= liquidity_th:  # ← WAS self.thresh_H.get(...)
        return False

    adx = row.get('adx_14', 0.0)
    if adx <= adx_th:  # ← WAS self.thresh_H.get(...)
        return False

    # Wick check
    close = row.get('close', 0.0)
    open_price = row.get('open', close)
    high = row.get('high', close)
    low = row.get('low', close)

    body = abs(close - open_price)
    upper_wick = high - max(close, open_price)
    lower_wick = min(close, open_price) - low

    wick_against_trend = (lower_wick > wick_mult * body) or (upper_wick > wick_mult * body)
    if not wick_against_trend:
        return False

    bos_flag = self._get_bos_flag(row)
    if bos_flag == 0:
        return False

    if fusion_score < fusion_th:  # ← WAS self.thresh_H.get(...)
        return False

    return True
```

#### 1.3 Update Constructor to Pass Config (15 min)

```python
# engine/archetypes/logic.py (modify __init__)

def __init__(self, config: dict, enabled: dict):
    """
    Initialize archetype logic with config.

    Args:
        config: Full config dict (includes archetypes section)
        enabled: Dict of enabled archetypes
    """
    self.config = config
    self.enabled = enabled

    # REMOVE: self.thresh_H and other shadow dicts
    # Now everything reads from self.config via get_archetype_param
```

---

### Hour 2: Wire Tests + Fail-Fast Guardrails (60 min)

#### 2.1 Wire Test Script (30 min)

```python
# bin/test_param_wiring.py
"""Test that parameter changes actually affect outcomes."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import json
from bin.backtest_router_v10_full import RouterAwareBacktest

def test_param_wiring(archetype: str, param: str, val_min, val_max):
    """
    Test that changing a parameter changes outcomes.

    Returns:
        True if parameter is properly wired, False otherwise
    """
    # Load baseline data (200 bars for speed)
    df = pd.read_parquet('data/BTC_1H_features.parquet')
    df = df.tail(200)

    # Load baseline configs
    with open('configs/baseline_btc_bull_pf20.json') as f:
        bull_base = json.load(f)
    with open('configs/baseline_btc_bear_defensive.json') as f:
        bear_base = json.load(f)

    # Test MIN value
    bull_min = bull_base.copy()
    bear_min = bear_base.copy()

    if 'archetypes' not in bull_min:
        bull_min['archetypes'] = {}
    if archetype not in bull_min['archetypes']:
        bull_min['archetypes'][archetype] = {}

    bull_min['archetypes'][archetype][param] = val_min
    bear_min['archetypes'][archetype][param] = val_min

    # Test MAX value
    bull_max = bull_base.copy()
    bear_max = bear_base.copy()

    if 'archetypes' not in bull_max:
        bull_max['archetypes'] = {}
    if archetype not in bull_max['archetypes']:
        bull_max['archetypes'][archetype] = {}

    bull_max['archetypes'][archetype][param] = val_max
    bear_max['archetypes'][archetype][param] = val_max

    # Run backtests
    from bin.backtest_router_v10_full import run_backtest

    results_min = run_backtest(bull_min, bear_min, df)
    results_max = run_backtest(bull_max, bear_max, df)

    # Check if anything changed
    trades_min = results_min.get('total_trades', 0)
    trades_max = results_max.get('total_trades', 0)
    pnl_min = results_min.get('total_pnl', 0)
    pnl_max = results_max.get('total_pnl', 0)

    trade_delta = abs(trades_min - trades_max)
    pnl_delta = abs(pnl_min - pnl_max)

    is_wired = (trade_delta > 0) or (pnl_delta > 1.0)

    print(f"\n{'='*60}")
    print(f"WIRE TEST: {archetype}.{param}")
    print(f"{'='*60}")
    print(f"Range: {val_min} → {val_max}")
    print(f"Trades: {trades_min} → {trades_max} (Δ{trade_delta})")
    print(f"PNL: ${pnl_min:.2f} → ${pnl_max:.2f} (Δ${pnl_delta:.2f})")
    print(f"Status: {'✅ WIRED' if is_wired else '❌ NOT WIRED'}")
    print(f"{'='*60}\n")

    return is_wired


def main():
    """Run wire tests for all optimizable parameters."""
    tests = [
        # Trap parameters
        ('trap_within_trend', 'quality_threshold', 0.45, 0.65),
        ('trap_within_trend', 'liquidity_threshold', 0.25, 0.40),
        ('trap_within_trend', 'adx_threshold', 20.0, 30.0),
        ('trap_within_trend', 'fusion_threshold', 0.30, 0.45),

        # OB Retest parameters (after refactoring)
        ('order_block_retest', 'boms_strength_min', 0.25, 0.40),
        ('order_block_retest', 'wyckoff_min', 0.30, 0.45),
        ('order_block_retest', 'ob_quality_threshold', 0.30, 0.50),
    ]

    results = {}
    for archetype, param, val_min, val_max in tests:
        try:
            is_wired = test_param_wiring(archetype, param, val_min, val_max)
            results[f"{archetype}.{param}"] = is_wired
        except Exception as e:
            print(f"❌ ERROR testing {archetype}.{param}: {e}")
            results[f"{archetype}.{param}"] = False

    # Summary
    print("\n" + "="*60)
    print("WIRE TEST SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for key, status in results.items():
        print(f"  {'✅' if status else '❌'} {key}")

    print(f"\n{passed}/{total} tests passed")

    if passed < total:
        print("\n❌ WIRE TESTS FAILED - Fix wiring before running Optuna!")
        return 1
    else:
        print("\n✅ All parameters properly wired")
        return 0


if __name__ == '__main__':
    exit(main())
```

#### 2.2 Zero-Variance Sentinel in Optimizer (15 min)

```python
# Add to optuna_trap_v2.py in the objective function

def run_optimization(self):
    """Execute Optuna study with zero-variance detection."""

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(  # ASHA-like pruning
            n_startup_trials=10,
            n_warmup_steps=5
        )
    )

    # Add callback for zero-variance detection
    def check_variance_callback(study, trial):
        """Abort if no variance detected after N trials."""
        if len(study.trials) >= 20:
            values = [t.value for t in study.trials if t.value is not None]
            if len(values) >= 20:
                std = np.std(values)
                if std < 1e-6:
                    raise RuntimeError(
                        f"❌ ZERO VARIANCE DETECTED after {len(values)} trials!\n"
                        f"All trials produced identical scores (std={std:.2e}).\n"
                        f"This indicates parameters are not affecting outcomes.\n"
                        f"Run wire tests: python3 bin/test_param_wiring.py"
                    )

    study.optimize(
        self.objective,
        n_trials=self.n_trials,
        callbacks=[check_variance_callback],
        show_progress_bar=True
    )

    return study
```

#### 2.3 Param Echo on Every Run (15 min)

```python
# Add to RouterAwareBacktest.run() or backtest entry point

def run(self):
    """Run backtest with parameter logging."""
    from engine.archetypes.param_accessor import log_params_used

    # Log parameters being used
    output_dir = Path(self.output_dir) if hasattr(self, 'output_dir') else Path('results/debug')

    for archetype in ['trap_within_trend', 'order_block_retest']:
        log_params_used(
            self.bull_config,
            archetype,
            output_dir / f'params_used_{archetype}.json'
        )

    # Continue with normal backtest...
    results = self._run_backtest()
    return results
```

---

### Hour 3: Refactor OB Retest (60 min)

Same pattern as trap:

```python
# engine/archetypes/logic.py (_check_B for order_block_retest)

def _check_B(self, row, prev_row, df, index, fusion_score, config: dict) -> bool:
    """B - Order Block Retest (NOW READS FROM CONFIG)"""
    from engine.archetypes.param_accessor import get_archetype_param

    boms_min = get_archetype_param(config, 'order_block_retest', 'boms_strength_min', 0.30)
    wyckoff_min = get_archetype_param(config, 'order_block_retest', 'wyckoff_min', 0.35)
    ob_quality_th = get_archetype_param(config, 'order_block_retest', 'ob_quality_threshold', 0.374)

    # Now use configurable thresholds (not hardcoded!)
    boms_str = self.g(row, "boms_strength", 0.0)
    wyckoff = self.g(row, "wyckoff_score", 0.0)

    return (bos_bullish and
            boms_str >= boms_min and      # ← WAS HARDCODED 0.30
            wyckoff >= wyckoff_min and    # ← WAS HARDCODED 0.35
            fusion >= ob_quality_th)      # ← WAS from threshold_policy only
```

---

### Hour 4: Validation + Documentation (60 min)

#### 4.1 Run Wire Tests (10 min)

```bash
python3 bin/test_param_wiring.py
```

**Acceptance**: All tests pass (✅ WIRED for each parameter)

#### 4.2 5-Trial Smoke Test (15 min)

```bash
python3 bin/optuna_trap_v2.py --n-trials 5 --cache <cache> --output results/trap_smoke
```

**Check**: `results/trap_smoke/trials.csv` should show variance > 0.001

#### 4.3 Diff Backtest Validation (15 min)

```bash
# Create two extreme configs
python3 bin/create_test_configs.py --extreme

# Run diff
python3 bin/diff_backtest.py \
  --config-a configs/trap_min_params.json \
  --config-b configs/trap_max_params.json \
  --output results/param_diff_test

# Check detection count deltas
cat results/param_diff_test/summary.txt
```

**Acceptance**: Detection counts differ by >5% between min/max configs

#### 4.4 Document Wiring System (20 min)

Update `OPTIMIZATION_TOOLS_GUIDE.md` with:
- How to add new optimizable parameters
- Wire test requirements
- Param accessor usage examples
- CI integration instructions

---

## 🎯 ACCEPTANCE CRITERIA (Must Pass All)

### Before Running ANY Optimization:

- [ ] Wire tests pass for all parameters
- [ ] 5-trial smoke test shows variance > 0.001
- [ ] Param echo logs appear in `params_used.json`
- [ ] Diff backtest shows detection count changes

### System-Level Guardrails Installed:

- [ ] `get_archetype_param()` accessor exists
- [ ] Trap archetype reads from config (not hardcoded)
- [ ] OB retest reads from config (not hardcoded)
- [ ] Zero-variance sentinel in optimizer
- [ ] Param echo on every run
- [ ] Wire test script executable

---

## ⏱️ TIME BUDGET ENFORCEMENT

**HARD STOP at 4 hours**. If any step takes longer:

- Skip OB refactor (do it later)
- Run wire tests on trap only
- Document what's left and move to optimization

**Why**: Better to have trap working + good guardrails than both half-done.

---

## 🚀 AFTER THE FIX (Decision Tree)

### If Wire Tests Pass (Expected):

```bash
# Build cache (one-time, 20 min)
python3 bin/cache_features_with_regime.py \
  --asset BTC --start 2022-01-01 --end 2024-12-31

# Run Optuna v2 (6-8 hours, can be overnight)
python3 bin/optuna_trap_v2.py \
  --n-trials 200 \
  --cache data/cached/btc_features_2022-01-01_2024-12-31_cached.parquet \
  --output results/optuna_trap_v2_fixed

# Zero-variance sentinel will abort early if parameters still not wired
```

### If Wire Tests Fail After 4 Hours:

**Move to OB Retest Scaling** (it has fewer parameters, easier to wire)

```bash
# Simpler: just tune 3 thresholds for OB
python3 bin/optuna_ob_retest.py --n-trials 100
```

**OR Move to Bear Optimization** (different archetype set, may be easier)

---

## 📊 SUCCESS METRICS

### Immediate (After Fix):

- Wire tests: 100% pass rate
- Smoke test: std(trial_values) > 0.001
- Param echo: logs appear with correct values

### After Re-Run:

- Trial variance > 0.01 (real search happening)
- Best trial ≠ first trial (improvement found)
- Fixed-size validation: PF improvement ≥ 10%

---

## 🎓 LESSONS ENCODED

1. **Always run wire tests** before multi-hour optimizations
2. **5-trial smoke test** catches zero-variance early
3. **Param accessor pattern** prevents shadow dict issues
4. **Zero-variance sentinel** auto-aborts bad runs
5. **Param echo** creates audit trail

---

**Generated**: 2025-11-06
**Estimated Duration**: 4 hours
**Hard Stop**: Yes (move to fallback if exceeds)
**Outcome**: Properly wired parameter system + fail-fast guardrails
