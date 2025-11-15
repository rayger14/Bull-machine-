# Optimization Auto-Bounds Feature

## Overview

The auto-bounds feature prevents parameter range mismatches in Optuna optimizations by computing bounds directly from actual data quantiles. This eliminates wasted search space and ensures all sampled values are realistic.

## Problem

Manual parameter bounds often don't match data distributions:
- Data range: `tf4h_fusion_score = [0.00, 0.22]`
- Manual bounds: `quality_threshold = [0.05, 0.30]`
- Result: 27% of search space (0.22-0.30) never occurs in data

## Solution

Use `engine/optimization/auto_bounds.py` to compute data-derived bounds:

```python
from engine.optimization.auto_bounds import compute_parameter_bounds, TRAP_PARAM_MAP

# Compute bounds from cached features
bounds = compute_parameter_bounds(
    df=df,
    param_to_feature_map=TRAP_PARAM_MAP,
    quantiles=(0.05, 0.95),  # Use 5th-95th percentile
    expand_factor=0.1,       # Add 10% buffer
    min_range=0.01          # Minimum range
)

# Use in Optuna
trial.suggest_float('fusion_threshold', *bounds['fusion_threshold'])
```

## Predefined Mappings

### Trap Archetype
```python
TRAP_PARAM_MAP = {
    'quality_threshold': 'tf4h_fusion_score',
    'fusion_threshold': 'tf4h_fusion_score',
    'adx_threshold': 'adx_14',
    'rsi_threshold': 'rsi_14',
    'atr_threshold': 'atr_20',
}
```

## Key Bugs Fixed in bin/optuna_trap_v2.py

### Bug 1: Parameter Range Mismatch
**Problem**: Hardcoded ranges didn't match data
**Fix**: Use auto-bounds feature
**Location**: `bin/optuna_trap_v2.py:186-192`

### Bug 2: Router Integration Issue
**Problem**: RouterAwareBacktest switched to CASH mode (no trades) on rolling windows
**Root Cause**: Missing macro features → all bars classified as 'neutral' → CASH mode
**Fix**: Use KnowledgeAwareBacktest directly for single-archetype optimization
**Location**: `bin/optuna_trap_v2.py:128-136`

### Bug 3: Shallow Copy Bug
**Problem**: `config.copy()` only copies top-level dict; nested dicts remain shared references
**Symptom**: All trials used same config despite different parameters
**Fix**: Use deep copy via JSON: `json.loads(json.dumps(config))`
**Location**: `bin/optuna_trap_v2.py:102`

### Bug 4: Wrong Config Location (CRITICAL)
**Problem**: Writing parameters to `config['archetypes']['trap_within_trend']` (never read)
**Root Cause**: Backtest uses letter codes (A-M), trap = 'H'
**Correct Location**: `config['archetypes']['thresholds']['H']`
**Fix**: Write to correct thresholds path
**Location**: `bin/optuna_trap_v2.py:104-119`

```python
# CORRECT config structure for trap archetype (letter code 'H')
config['archetypes']['thresholds']['H']['fusion'] = trap_params['fusion_threshold']
config['archetypes']['thresholds']['min_liquidity'] = trap_params['liquidity_threshold']
```

## Config Structure Reference

Archetypes use single-letter codes:
- A = spring
- B = order_block_retest
- C = wick_trap
- **H = trap_within_trend** (the one optimized in v2)
- K, L, M = other archetypes

Parameters are stored at:
```
config['archetypes']['thresholds'][letter_code][param_name]
```

## Validation Results

After all 4 bugs fixed, variance confirmed (5-trial test):
- Trial 0: value = -5.351 (liquidity=0.45)
- Trial 1: value = 0.769 (liquidity=0.15) ← BEST
- Trial 2: value = -5.351 (liquidity=0.45)
- Trial 3: value = -5.351 (liquidity=0.25)
- Trial 4: value = 0.769 (liquidity=0.15)

**Key Finding**: Liquidity threshold drives performance. Values ~0.15 outperform higher values (0.25-0.45).

## Usage for Future Optimizers

1. **Define parameter-to-feature mapping** in `auto_bounds.py`
2. **Compute bounds** before optimization starts
3. **Use KnowledgeAwareBacktest directly** for single-archetype optimization
4. **Use deep copy** for config mutations
5. **Write to correct thresholds location** using letter codes

## Testing

Validate with test script:
```bash
python3 bin/test_auto_bounds.py --cache data/cached/btc_features_2022-01-01_2024-12-31_cached.parquet
```

Expected output shows computed bounds for each parameter with actual data ranges.
