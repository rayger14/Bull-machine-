# Archetype System Wiring Diagnosis

**Date**: 2025-11-07
**Issue**: Zero-variance Optuna optimization despite parameter variation
**Status**: ROOT CAUSE IDENTIFIED

## Executive Summary

The trap archetype optimizer (optuna_trap_v2.py) produces identical scores across all trials because:

1. **Wrong method is being called**: Optimizer targets `_check_H()` but backtest actually calls `_check_K()`
2. **Priority order issue**: K (Wick Trap) has higher priority than H (Trap Within Trend) in dispatcher
3. **Config location mismatch**: Optimizer writes to one location, _check_K() reads from another
4. **Letter code confusion**: Documentation and code have conflicting mappings

## Logic Diagram: Current Call Flow

```
optuna_trap_v2.py (optimizer)
│
├─> suggest_params() → {fusion_threshold, liquidity_threshold, adx_threshold, ...}
│
├─> write config to: config['archetypes']['trap_within_trend']
│   └─> location: config['archetypes']['trap_within_trend'] = {...}
│
└─> KnowledgeAwareBacktest.run()
    │
    └─> for each bar:
        │
        └─> ArchetypeLogic.check_archetype(row, ...)
            │
            ├─> Priority order: A, B, C, K, H, L, F, D, G, E, M
            │
            ├─> if enabled['K'] and _check_K():  ← CALLED FIRST!
            │   │
            │   ├─> reads from: self.thresh_K = config['archetypes']['thresholds']['K']
            │   │   └─> NOT from config['archetypes']['trap_within_trend']!
            │   │
            │   ├─> uses hardcoded defaults:
            │   │   • adx_threshold: 25.0
            │   │   • liq_threshold: 0.30
            │   │   • fusion_threshold: 0.36
            │   │
            │   └─> return 'wick_trap' ← ENDS SEARCH
            │
            └─> if enabled['H'] and _check_H():  ← NEVER REACHED!
                │
                ├─> reads from: config['archetypes']['trap_within_trend']
                │   via get_archetype_param()  ← CORRECT LOCATION!
                │
                └─> return 'trap_within_trend'
```

## The Three-Layer Problem

### Layer 1: Priority Order Prevents _check_H() from Running

**File**: engine/archetypes/logic.py:320-367

```python
def check_archetype(self, ...):
    # Priority: A, B, C, K, H, L, F, D, G, E, M

    if self.enabled['K']:  # Line 335
        if self._check_K(row, prev_row, df, index, fusion_score):
            return 'wick_trap', fusion_score, liquidity_score

    if self.enabled['H']:  # Line 339 - NEVER REACHED IF K MATCHES
        if self._check_H(row, prev_row, df, index, fusion_score):
            return 'trap_within_trend', fusion_score, liquidity_score
```

**Impact**: If K matches, H is never called. Optimizer's parameters for H are ignored.

### Layer 2: Config Location Mismatch

**Optimizer writes to**: `config['archetypes']['trap_within_trend']`
**File**: bin/optuna_trap_v2.py:104-122

```python
# Write to legacy archetype param location
trap_config = {
    'fusion_threshold': ...,
    'liquidity_threshold': ...,
    ...
}
config['archetypes']['trap_within_trend'] = trap_config
```

**_check_K() reads from**: `config['archetypes']['thresholds']['K']`
**File**: engine/archetypes/logic.py:58, 763, 768, 775

```python
# In __init__:
self.thresh_K = thresholds.get('K', {})

# In _check_K:
if adx <= self.thresh_K.get('adx', 25.0):  # Line 763
if liquidity < self.thresh_K.get('liq', 0.30):  # Line 768
if fusion_score < self.thresh_K.get('fusion', 0.36):  # Line 775
```

**_check_H() reads from**: `config['archetypes']['trap_within_trend']` ✅
**File**: engine/archetypes/logic.py:692-696

```python
# Correctly uses get_archetype_param accessor!
quality_th = get_archetype_param(self.config, 'trap_within_trend', 'quality_threshold', 0.55)
liquidity_th = get_archetype_param(self.config, 'trap_within_trend', 'liquidity_threshold', 0.30)
adx_th = get_archetype_param(self.config, 'trap_within_trend', 'adx_threshold', 25.0)
```

**Impact**: Even if H were called, K reads from a different location with hardcoded defaults.

### Layer 3: Letter Code Naming Confusion

**Documentation** (logic.py:16-17):
```python
# - H: Trap Within Trend (HTF trend + liquidity drop + wick against trend)
# - K: Wick Trap (Moneytaur) (Wick anomaly + ADX > 25 + BOS context)
```

**LEGACY_ARCHETYPE_MAP** (threshold_policy.py:32-44):
```python
LEGACY_ARCHETYPE_MAP = {
    'momentum_continuation': 'H',  # ← H is NOT trap_within_trend!
    'trap_within_trend': 'K',      # ← K IS trap_within_trend!
}
```

**check_archetype() returns** (logic.py:337, 341):
```python
if self.enabled['K']:
    return 'wick_trap', ...         # ← K returns 'wick_trap'

if self.enabled['H']:
    return 'trap_within_trend', ... # ← H returns 'trap_within_trend'
```

**Impact**: Fundamental inconsistency across docstrings, legacy maps, and return values.

## Why Zero Variance Occurred

1. All 10 trials wrote **different** parameters to `config['archetypes']['trap_within_trend']`
2. Backtest called `_check_K()` which reads from `config['archetypes']['thresholds']['K']`
3. Location `['thresholds']['K']` was **never written to** by optimizer
4. `_check_K()` used **same hardcoded defaults** every trial: adx=25.0, liq=0.30, fusion=0.36
5. Result: **Identical behavior** across all trials → **identical score**: 0.7689968789394481

## Evidence: Diagnostic Logs Never Appeared

**Expected**: `_check_H()` diagnostic log at line 701:
```python
logging.info(f"[CONFIG READ] trap_within_trend params: fusion={fusion_th}, liq={liquidity_th}")
```

**Observed**: This log NEVER appeared in any trial output.

**Conclusion**: `_check_H()` was never called. `_check_K()` intercepted all matches.

## Config Structure Comparison

### Optimizer Writes To:
```json
{
  "archetypes": {
    "trap_within_trend": {
      "fusion_threshold": 0.35,    ← VARIES per trial
      "liquidity_threshold": 0.30, ← VARIES per trial
      "adx_threshold": 25.0,       ← VARIES per trial
      "quality_threshold": 0.55,
      "wick_multiplier": 2.0
    }
  }
}
```

### _check_K() Reads From:
```json
{
  "archetypes": {
    "thresholds": {
      "K": {
        "adx": ???,      ← NOT SET! Uses default 25.0
        "liq": ???,      ← NOT SET! Uses default 0.30
        "fusion": ???    ← NOT SET! Uses default 0.36
      }
    }
  }
}
```

### _check_H() Reads From (Correct but Never Called):
```json
{
  "archetypes": {
    "trap_within_trend": {
      "fusion_threshold": ...,  ← CORRECT LOCATION!
      "liquidity_threshold": ..., ← CORRECT LOCATION!
      "adx_threshold": ...      ← CORRECT LOCATION!
    }
  }
}
```

## The Wiring Issue Visualized

```
OPTIMIZER                    CONFIG                         BACKTEST
─────────                    ──────                         ────────

write params     ────────>   ['archetypes']
to 'trap_         ────────>     ['trap_within_trend']
within_trend'                     {fusion: 0.35, ...}  ────> _check_H() reads ✅
                                                      │      BUT NEVER CALLED ❌
                                                      │
                                ['thresholds']        │
                                  ['K']               │
                                    {} ←─────────────┘──── _check_K() reads ❌
                                    EMPTY!                  Uses defaults!
                                    Uses:                   SAME EVERY TRIAL
                                    • adx: 25.0
                                    • liq: 0.30
                                    • fusion: 0.36
```

## Solution Options

### Option 1: Quick Fix - Write to K's Location
**Change**: Optimizer writes to `config['archetypes']['thresholds']['K']`

```python
# In optuna_trap_v2.py:
if 'archetypes' not in config:
    config['archetypes'] = {}
if 'thresholds' not in config['archetypes']:
    config['archetypes']['thresholds'] = {}

config['archetypes']['thresholds']['K'] = {
    'adx': trap_params.get('adx_threshold', 25.0),
    'liq': trap_params.get('liquidity_threshold', 0.30),
    'fusion': trap_params.get('fusion_threshold', 0.35)
}
```

**Pros**: Minimal change, works immediately
**Cons**: Doesn't fix naming confusion, perpetuates letter codes

### Option 2: Canonical Slug System (Recommended)
**Change**: Implement unified naming as specified in user's architecture

1. Create `engine/archetypes/registry.py`:
```python
ARCHETYPES = {
    'wick_trap': {
        'aliases': ['K'],
        'description': 'Wick anomaly reversal (Moneytaur)',
        'priority': 4
    },
    'trap_within_trend': {
        'aliases': ['H'],
        'description': 'Trap reversal within HTF trend',
        'priority': 5
    }
}

def resolve_key(key: str) -> str:
    """Resolve alias to canonical slug."""
    for slug, meta in ARCHETYPES.items():
        if key == slug or key in meta['aliases']:
            return slug
    return key
```

2. Single parameter accessor:
```python
def get_param(ctx, slug: str, key: str, default):
    """Single source of truth for archetype params."""
    resolved = resolve_key(slug)
    return ctx.config.get('archetypes', {}).get(resolved, {}).get(key, default)
```

3. Refactor all `_check_X()` methods to use canonical names

**Pros**: Clean architecture, future-proof, eliminates confusion
**Cons**: Larger refactor, requires config migration

### Option 3: Disable K, Enable H Only
**Change**: Force optimizer to work with H by disabling K

```python
# In base config:
config['archetypes']['enable_K'] = False  # Disable Wick Trap
config['archetypes']['enable_H'] = True   # Enable Trap Within Trend
```

**Pros**: Immediate unblocking for optimization
**Cons**: Loses K archetype, doesn't fix underlying issue

## Recommended Path Forward

1. **Immediate**: Option 3 (disable K) to unblock optimization
2. **Short-term**: Option 1 (quick fix) to verify parameters actually work
3. **Long-term**: Option 2 (canonical slugs) for clean architecture

## Archetype Method Comparison

| Method | Description | Config Location | Uses | Priority |
|--------|-------------|----------------|------|----------|
| `_check_K()` | Wick Trap (Moneytaur) | `['thresholds']['K']` | `self.thresh_K.get()` | 4 (higher) |
| `_check_H()` | Trap Within Trend | `['trap_within_trend']` | `get_archetype_param()` | 5 (lower) |

## Files Involved

1. **bin/optuna_trap_v2.py** - Optimizer (writes config)
2. **engine/archetypes/logic.py** - Detection logic (reads config)
3. **engine/archetypes/threshold_policy.py** - Naming maps
4. **engine/archetypes/param_accessor.py** - Config accessor
5. **configs/baseline_btc_bull_pf20.json** - Base config structure

## Next Steps

1. User to choose solution path (Option 1, 2, or 3)
2. Implement chosen solution
3. Re-run 10-trial validation test
4. Verify variance appears and parameters affect scores
5. If successful, run full 200-trial optimization
6. Long-term: Migrate to canonical slug system (PR#6A)
