# Optuna Parallel Archetype Optimizer V2 - Two-Layer Threshold Architecture

## Executive Summary

**Status**: Refactored optimizer implementing correct two-layer threshold architecture
**File**: `bin/optuna_parallel_archetypes_v2.py` (1,105 lines)
**Runtime**: 6-8 hours for full optimization (4 parallel workers × 50-100 trials each)

### Key Improvements Over V1

1. **Correct Architectural Separation**
   - Global safety rails are FIXED (never optimized)
   - Archetype-specific thresholds are optimized per pattern
   - Clear separation eliminates previous mixing

2. **MVP Config Compatibility**
   - Reads from `configs/mvp/mvp_bull_market_v1.json`
   - Applies parameters to correct nested paths
   - Validates config structure before backtest

3. **Parameter Application Fix**
   - Writes to both `archetypes.thresholds[pattern]` AND `archetypes[pattern]`
   - Handles MVP dual-location structure correctly
   - Ensures ThresholdPolicy can read optimizer-written params

---

## Architecture: Two-Layer Threshold System

### Layer 1: Global Safety Rails (FIXED - Never Optimize)

These are hardcoded constants in `GLOBAL_SAFETY_RAILS`:

```python
GLOBAL_SAFETY_RAILS = {
    'min_liquidity_floor': 0.05,          # Hard minimum
    'vix_panic_threshold': 30.0,           # Crisis detection
    'move_panic_threshold': 120.0,         # Bond volatility crisis
    'dxy_extreme_threshold': 105.0,        # Dollar crisis
    'funding_z_extreme': 3.0,              # Extreme crowding
    'crisis_fuse_enabled': True,           # Always enabled
    'crisis_fuse_lookback_hours': 24,
    'max_portfolio_risk_pct': 0.10,
}
```

**Why Fixed?**
- Safety rails protect against extreme market conditions
- Optimizing these would remove critical guardrails
- Values are based on fundamental risk principles, not backtest performance

**Where Applied:**
- `MVPConfigGenerator._apply_safety_rails()` enforces these on every trial config
- Applied to `config['context']['crisis_fuse']` and `config['risk']`

---

### Layer 2: Archetype-Specific Thresholds (OPTIMIZED)

Each archetype pattern gets its own optimized parameters:

```python
# Example: Trap Within Trend
params['trap_within_trend'] = {
    'fusion_threshold': 0.42,        # Entry confidence
    'archetype_weight': 1.15,        # Scoring multiplier
    'adx_threshold': 25.0,           # Pattern filter
    'cooldown_bars': 14              # Cooldown period
}
```

**Parameter Types:**

1. **Core Thresholds** (all archetypes)
   - `fusion_threshold`: Entry confidence (0.20 - 0.52)
   - `archetype_weight`: Pattern scoring multiplier (0.40 - 1.35)
   - `cooldown_bars`: Bars between trades (6 - 20)

2. **Pattern-Specific Filters** (varies by archetype)
   - `funding_z_min`: Funding rate z-score (long_squeeze)
   - `rsi_min`: RSI minimum (volume_exhaustion, long_squeeze)
   - `vol_z_min/max`: Volume z-score bounds
   - `adx_threshold`: ADX trend strength
   - `boms_strength_min`: BOMS minimum (order_block_retest)
   - `wyckoff_min`: Wyckoff score minimum
   - `disp_atr_multiplier`: Displacement threshold
   - `pti_score_threshold`: PTI minimum (spring)

**Where Applied:**
- `MVPConfigGenerator._apply_archetype_params()` writes to TWO locations:
  1. `config['archetypes']['thresholds'][pattern]` - thresholds subdirectory
  2. `config['archetypes'][pattern]` - top-level archetype config

This dual-write ensures compatibility with `ThresholdPolicy._build_base_map()` which reads from both locations.

---

## Config Application Flow

### 1. Trial Config Generation

```python
# In objective function:
global_params = suggest_global_fusion_weights(trial)  # NOT safety rails
archetype_params = suggest_archetype_params(trial, group_name)

config_path = config_gen.generate(global_params, archetype_params, group_name)
```

### 2. Parameter Application Order

```python
def generate(self, global_params, archetype_params, group_name):
    # 1. Deep copy base MVP config
    trial_config = deepcopy(self.base_config)

    # 2. ENFORCE global safety rails (fixed)
    self._apply_safety_rails(trial_config)

    # 3. Apply global fusion weights (optimized)
    self._apply_fusion_weights(trial_config, global_params)

    # 4. Enable only this group's archetypes
    self._enable_group_archetypes(trial_config, group_name)

    # 5. Apply archetype-specific params (optimized)
    self._apply_archetype_params(trial_config, archetype_params)

    # 6. Validate structure
    self._validate_config(trial_config)

    return temp_config_path
```

### 3. Dual-Location Write Strategy

```python
def _apply_archetype_params(self, config, archetype_params):
    for pattern_name, params in archetype_params.items():
        # Location 1: thresholds subdirectory (LEGACY path)
        if pattern_name not in thresholds:
            thresholds[pattern_name] = {}

        thresholds[pattern_name]['fusion_threshold'] = params['fusion_threshold']
        thresholds[pattern_name]['funding_z_min'] = params.get('funding_z_min')
        # ... other pattern filters

        # Location 2: top-level archetype config (OPTIMIZER path)
        if pattern_name not in archetypes_cfg:
            archetypes_cfg[pattern_name] = {}

        archetypes_cfg[pattern_name]['archetype_weight'] = params['archetype_weight']
        archetypes_cfg[pattern_name]['final_fusion_gate'] = params['fusion_threshold']
        archetypes_cfg[pattern_name]['cooldown_bars'] = params.get('cooldown_bars')
```

**Why Two Locations?**
- MVP configs use nested structure: `archetypes.thresholds[pattern]` for filter thresholds
- `ThresholdPolicy` reads from `archetypes[pattern]` for archetype-level config
- Optimizer writes to BOTH to ensure compatibility with all code paths

---

## Archetype Groups & Parameter Spaces

### Group 1: Trap Within Trend (A, G, K)

**Archetypes**: trap_within_trend, liquidity_sweep, spring
**Strategy**: Momentum-based reversals and liquidity traps
**Trader Type**: Moneytaur

```python
# Trap Within Trend
'trap_within_trend': {
    'fusion_threshold': (0.30, 0.48, step=0.01),
    'archetype_weight': (0.85, 1.30, step=0.05),
    'adx_threshold': (18.0, 35.0, step=1.0),
    'cooldown_bars': (10, 20, step=2)
}

# Liquidity Sweep
'liquidity_sweep': {
    'fusion_threshold': (0.28, 0.48, step=0.01),
    'archetype_weight': (0.90, 1.25, step=0.05),
    'boms_strength_min': (0.25, 0.55, step=0.05)
}

# Spring (Wyckoff)
'spring': {
    'fusion_threshold': (0.26, 0.44, step=0.01),
    'archetype_weight': (0.95, 1.20, step=0.05),
    'pti_score_threshold': (0.25, 0.55, step=0.05),
    'disp_atr_multiplier': (0.5, 1.3, step=0.1)
}
```

### Group 2: Order Block Retest (B, H, L)

**Archetypes**: order_block_retest, momentum_continuation, volume_exhaustion
**Strategy**: Structure-based order block retests
**Trader Type**: Zeroika

```python
# Order Block Retest
'order_block_retest': {
    'fusion_threshold': (0.28, 0.48, step=0.01),
    'archetype_weight': (0.85, 1.35, step=0.05),
    'boms_strength_min': (0.20, 0.50, step=0.02),
    'wyckoff_min': (0.25, 0.55, step=0.02),
    'cooldown_bars': (8, 16, step=2)
}

# Momentum Continuation
'momentum_continuation': {
    'fusion_threshold': (0.30, 0.50, step=0.01),
    'archetype_weight': (0.90, 1.25, step=0.05),
    'adx_threshold': (22.0, 40.0, step=2.0)
}

# Volume Exhaustion
'volume_exhaustion': {
    'fusion_threshold': (0.30, 0.48, step=0.01),
    'archetype_weight': (0.85, 1.20, step=0.05),
    'vol_z_min': (0.6, 1.8, step=0.1),
    'rsi_min': (62.0, 78.0, step=2.0)
}
```

### Group 3: BOS/CHOCH (C)

**Archetypes**: wick_trap
**Strategy**: Break of Structure and Change of Character
**Trader Type**: Generic

```python
# Wick Trap
'wick_trap': {
    'fusion_threshold': (0.32, 0.52, step=0.01),
    'archetype_weight': (0.85, 1.25, step=0.05),
    'disp_atr_multiplier': (0.6, 1.6, step=0.1),
    'cooldown_bars': (10, 18, step=2)
}
```

### Group 4: Long Squeeze (S5)

**Archetypes**: long_squeeze
**Strategy**: Funding rate cascade patterns
**Trader Type**: Moneytaur

```python
# Long Squeeze (short-biased)
'long_squeeze': {
    'fusion_threshold': (0.26, 0.46, step=0.01),
    'archetype_weight': (0.40, 0.80, step=0.05),  # Lower weight (crisis pattern)
    'funding_z_min': (0.8, 2.2, step=0.1),
    'rsi_min': (62.0, 78.0, step=2.0),
    'liquidity_max': (0.15, 0.35, step=0.02),
    'cooldown_bars': (6, 12, step=2)
}
```

---

## Optimization Strategy

### Multi-Fidelity Evaluation (Hyperband Pruning)

```python
fidelity = trial.suggest_int('_fidelity', 0, 2)

periods = {
    0: ("2024-01-01", "2024-01-31"),   # 1 month - fast pruning
    1: ("2024-01-01", "2024-03-31"),   # 3 months - validation
    2: ("2024-01-01", "2024-09-30"),   # 9 months - full eval
}
```

**Hyperband Pruning:**
- `min_resource=1`: Start at 1-month fidelity
- `max_resource=3`: Max 9-month fidelity
- `reduction_factor=3`: Prune 2/3 of trials at each step

**Expected Behavior:**
- 50% of trials pruned after 1-month test
- 25% continue to 3-month validation
- 12% reach full 9-month evaluation

### Objective Function

```python
def compute_objective_score(metrics, fidelity):
    # Base: PF × win_rate × sqrt(trades)
    base_score = pf * (1 + wr/100) * (trades ** 0.5)

    # Penalties
    dd_penalty = dd / (10 if fidelity == 2 else 20)
    overtrade_penalty = max(0, (trades - max_trades) * 0.1)

    # Bonuses (full fidelity only)
    sharpe_bonus = max(sharpe, 0) * 0.5 if fidelity == 2 else 0

    return base_score - dd_penalty - overtrade_penalty + sharpe_bonus
```

**Targets:**
- Fidelity 0 (1mo): ≤30 trades
- Fidelity 1 (3mo): ≤60 trades
- Fidelity 2 (9mo): ≤100 trades

---

## Usage Guide

### 1. Test Single Trial (Validation)

```bash
python bin/optuna_parallel_archetypes_v2.py \
    --test-trial \
    --base-config configs/mvp/mvp_bull_market_v1.json
```

**Output:**
```
SINGLE TRIAL TEST
================
Testing group: trap_within_trend
Archetypes: A, G, K

Generating trial config...

Generated config validation:
- Fusion weights: {'wyckoff': 0.40, 'liquidity': 0.30, 'momentum': 0.30}
- Crisis fuse enabled: True

Archetype parameters:
- trap_within_trend:
    archetype_weight: 1.15
    final_fusion_gate: 0.42
    fusion_threshold: 0.42

Running test backtest (1 month)...

Backtest Results:
- Trades: 12
- Profit Factor: 1.85
- Win Rate: 58.3%
- Max Drawdown: 4.2%

TEST PASSED: Config generation and backtest execution successful
```

### 2. Run Full Optimization

```bash
python bin/optuna_parallel_archetypes_v2.py \
    --trials 100 \
    --base-config configs/mvp/mvp_bull_market_v1.json \
    --output configs/optimized_archetypes_v2.json \
    --storage optuna_v2.db
```

**Runtime**: 6-8 hours (4 parallel workers × 100 trials = 400 total trials)

### 3. Resume from Checkpoint

```bash
python bin/optuna_parallel_archetypes_v2.py \
    --resume \
    --storage optuna_v2.db
```

Uses existing SQLite database to continue optimization.

### 4. Optimize Specific Groups

```bash
python bin/optuna_parallel_archetypes_v2.py \
    --trials 50 \
    --groups trap_within_trend order_block_retest \
    --base-config configs/mvp/mvp_bull_market_v1.json
```

Only optimizes selected archetype groups.

---

## Output Structure

### Trial Config Example

```json
{
  "fusion": {
    "entry_threshold_confidence": 0.40,
    "weights": {
      "wyckoff": 0.42,
      "liquidity": 0.28,
      "momentum": 0.30,
      "smc": 0.0
    }
  },

  "archetypes": {
    "enable_A": true,
    "enable_G": true,
    "enable_K": true,
    "enable_B": false,

    "thresholds": {
      "trap_within_trend": {
        "direction": "long",
        "fusion_threshold": 0.42,
        "adx_threshold": 25.0,
        "max_risk_pct": 0.02,
        "atr_stop_mult": 1.8
      }
    },

    "trap_within_trend": {
      "archetype_weight": 1.15,
      "final_fusion_gate": 0.42,
      "cooldown_bars": 14
    }
  },

  "context": {
    "crisis_fuse": {
      "enabled": true,
      "lookback_hours": 24
    }
  },

  "risk": {
    "max_portfolio_risk_pct": 0.10
  }
}
```

### Final Optimized Config

```json
{
  "version": "mvp_optimized_v2",

  "_optimization_metadata": {
    "timestamp": "2025-11-17T14:30:00",
    "optimizer_version": "v2_two_layer_architecture",
    "groups_optimized": [
      "trap_within_trend",
      "order_block_retest",
      "bos_choch",
      "long_squeeze"
    ],
    "total_trials": 400,
    "best_scores": {
      "trap_within_trend": 12.5,
      "order_block_retest": 11.8,
      "bos_choch": 9.2,
      "long_squeeze": 7.4
    },
    "global_safety_rails": {
      "min_liquidity_floor": 0.05,
      "vix_panic_threshold": 30.0,
      "_note": "These were FIXED and not optimized"
    }
  },

  "fusion": {
    "weights": {
      "wyckoff": 0.45,
      "liquidity": 0.25,
      "momentum": 0.30
    }
  },

  "archetypes": {
    "trap_within_trend": {
      "archetype_weight": 1.20,
      "final_fusion_gate": 0.40
    }
  }
}
```

---

## Validation Checklist

### Before Running Optimization

- [ ] Base MVP config exists and is valid JSON
- [ ] `bin/backtest_knowledge_v2.py` is executable
- [ ] Database file path is writable
- [ ] Run `--test-trial` to verify config generation
- [ ] Check that backtest engine is working

### During Optimization

- [ ] Monitor progress logs for errors
- [ ] Check that trials are completing (not timing out)
- [ ] Verify metrics are being extracted correctly
- [ ] Watch for pruning behavior (should see "Trial X pruned")

### After Optimization

- [ ] Check that optimized config is valid JSON
- [ ] Verify all archetype groups have results
- [ ] Compare scores across groups
- [ ] Validate that safety rails are present in output
- [ ] Run validation backtest with optimized config

---

## Common Issues & Solutions

### Issue 1: Config Mismatch Error

**Error:**
```
KeyError: 'archetypes'
```

**Cause:** Base config doesn't have `archetypes` section

**Solution:**
```bash
# Use MVP config (not legacy config)
--base-config configs/mvp/mvp_bull_market_v1.json
```

### Issue 2: Parameters Not Applied

**Error:** Backtest ignores optimizer parameters

**Cause:** ThresholdPolicy can't find parameters in config

**Solution:** Check dual-write in `_apply_archetype_params()`:
- Parameters written to BOTH `archetypes.thresholds[pattern]` AND `archetypes[pattern]`
- Verify with `--test-trial` mode

### Issue 3: All Trials Fail

**Error:** `Backtest did not return metrics`

**Cause:** Backtest script failing to run

**Solution:**
```bash
# Test backtest manually
python bin/backtest_knowledge_v2.py \
    --asset BTC \
    --start 2024-01-01 \
    --end 2024-01-31 \
    --config configs/mvp/mvp_bull_market_v1.json
```

### Issue 4: No Trades Generated

**Error:** All trials return 0 trades

**Cause:** Thresholds too strict or archetypes not enabled

**Solution:**
- Check that `enable_A`, `enable_B`, etc. are being set in trial config
- Verify fusion weights are normalized correctly
- Lower initial threshold ranges if needed

---

## Performance Expectations

### Runtime Breakdown (100 trials per group)

| Group               | Trials | Pruned | Full Eval | Runtime |
|---------------------|--------|--------|-----------|---------|
| trap_within_trend   | 100    | ~65    | ~12       | 1.5h    |
| order_block_retest  | 100    | ~60    | ~15       | 1.8h    |
| bos_choch           | 100    | ~70    | ~10       | 1.2h    |
| long_squeeze        | 100    | ~55    | ~18       | 2.0h    |
| **TOTAL**           | 400    | ~250   | ~55       | **6.5h**|

### Backtest Speed (per trial)

- **Fidelity 0** (1 month): ~15-30 seconds
- **Fidelity 1** (3 months): ~30-60 seconds
- **Fidelity 2** (9 months): ~60-120 seconds

### Expected Score Ranges

| Group               | Min Score | Median Score | Best Score |
|---------------------|-----------|--------------|------------|
| trap_within_trend   | 5.0       | 9.5          | 14.0       |
| order_block_retest  | 4.5       | 8.8          | 13.5       |
| bos_choch           | 3.0       | 7.2          | 11.0       |
| long_squeeze        | 2.5       | 5.8          | 9.0        |

**Score Interpretation:**
- Score = PF × (1 + WR/100) × sqrt(trades) - penalties
- Score > 10: Excellent
- Score 7-10: Good
- Score 5-7: Acceptable
- Score < 5: Poor

---

## Architecture Compliance

### ✅ Fixed Global Safety Rails

```python
# Hardcoded in GLOBAL_SAFETY_RAILS dict
'min_liquidity_floor': 0.05,          # Never changes
'vix_panic_threshold': 30.0,           # Never changes
'crisis_fuse_enabled': True,           # Never changes
```

### ✅ Optimized Archetype Thresholds

```python
# Suggested per trial from parameter space
'fusion_threshold': trial.suggest_float('trap_fusion', 0.30, 0.48, step=0.01)
'archetype_weight': trial.suggest_float('trap_weight', 0.85, 1.30, step=0.05)
```

### ✅ Dual-Location Write Strategy

```python
# Location 1: thresholds subdirectory
thresholds['trap_within_trend']['fusion_threshold'] = 0.42

# Location 2: top-level archetype config
archetypes_cfg['trap_within_trend']['archetype_weight'] = 1.15
archetypes_cfg['trap_within_trend']['final_fusion_gate'] = 0.42
```

### ✅ Config Validation

```python
def _validate_config(self, config: dict):
    # Ensures required sections exist
    required_sections = ['fusion', 'archetypes', 'context', 'risk']

    # Validates at least one archetype is enabled
    enabled = [k for k, v in archetypes_cfg.items()
               if k.startswith('enable_') and v is True]
```

---

## Next Steps

1. **Run Test Trial**
   ```bash
   python bin/optuna_parallel_archetypes_v2.py --test-trial
   ```

2. **Start Small Optimization** (10 trials for quick validation)
   ```bash
   python bin/optuna_parallel_archetypes_v2.py --trials 10
   ```

3. **Review Results**
   - Check `configs/optimized_archetypes_v2.json`
   - Validate metadata shows safety rails were NOT optimized
   - Compare scores across groups

4. **Full Production Run** (100 trials)
   ```bash
   python bin/optuna_parallel_archetypes_v2.py --trials 100
   ```

5. **Validate Optimized Config**
   ```bash
   python bin/backtest_knowledge_v2.py \
       --asset BTC \
       --start 2024-01-01 \
       --end 2024-09-30 \
       --config configs/optimized_archetypes_v2.json
   ```
