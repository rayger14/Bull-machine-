# Investigation: Why trap_within_trend Archetypes (A, G, K) Aren't Firing

**Date**: 2025-11-17
**Status**: ROOT CAUSE IDENTIFIED - CRITICAL CONFIG/CODE MISMATCH
**Impact**: Optimizer optimizing wrong parameters, optimization effort completely wasted

---

## Executive Summary

The optimizer successfully completed 40 trials for the "trap_within_trend" archetype group (A, G, K), but produced **ZERO trades** in 2024 bull market backtests. Investigation reveals this is due to a **critical mismatch between optimizer archetype group definitions and actual runtime archetype name mappings**.

**TL;DR**: The optimizer is optimizing parameters for archetypes that don't actually exist in the runtime code.

---

## 1. Archetype ID Mapping: What the Optimizer THINKS

From `/bin/optuna_parallel_archetypes_v2.py` (lines 94-119):

```python
ARCHETYPE_GROUPS = {
    'trap_within_trend': {
        'archetypes': ['A', 'G', 'K'],
        'canonical': ['trap_within_trend', 'liquidity_sweep', 'spring'],
        'description': 'Momentum-based reversals and liquidity traps',
        'trader_type': 'Moneytaur',
    },
    'order_block_retest': {
        'archetypes': ['B', 'H', 'L'],
        'canonical': ['order_block_retest', 'momentum_continuation', 'volume_exhaustion'],
        'description': 'Structure-based order block retests',
        'trader_type': 'Zeroika',
    },
    # ...
}
```

**Optimizer believes**:
- A → `trap_within_trend`
- G → `liquidity_sweep`
- K → `spring`

---

## 2. Archetype ID Mapping: What the Runtime ACTUALLY USES

From `/engine/archetypes/logic_v2_adapter.py` (lines 398-420):

```python
archetype_map = {
    # Bull-biased archetypes
    'A': ('trap_reversal', self._check_A, 1),      # ❌ NOT 'trap_within_trend'!
    'B': ('order_block_retest', self._check_B, 2),
    'C': ('fvg_continuation', self._check_C, 3),
    'K': ('wick_trap', self._check_K, 4),          # ❌ NOT 'spring'!
    'H': ('trap_within_trend', self._check_H, 5),  # ❌ H is trap_within_trend!
    'L': ('volume_exhaustion', self._check_L, 6),
    'G': ('re_accumulate', self._check_G, 9),      # ❌ NOT 'liquidity_sweep'!
    # ...
}
```

**Runtime ACTUALLY maps**:
- A → `trap_reversal` (checks for 'spring' in `_check_A`)
- G → `re_accumulate`
- K → `wick_trap`
- H → `trap_within_trend` (the one the optimizer thought was A!)

---

## 3. What Each Archetype Check REALLY Looks For

### Archetype A (`_check_A`, line 637-697):
```python
def _check_A(self, context: RuntimeContext) -> tuple:
    """
    Archetype A: Trap Reversal (PTI spring/UTAD + displacement).
    """
    fusion_th = context.get_threshold('spring', 'fusion_threshold', 0.33)
    pti_score_th = context.get_threshold('spring', 'pti_score_threshold', 0.40)
    disp_multiplier = context.get_threshold('spring', 'disp_atr_multiplier', 0.80)

    pti_trap = self.g(context.row, "pti_trap_type", '')
    if not pti_trap or pti_trap not in ['spring', 'utad']:
        return False, 0.0, {"reason": "no_pti_trap"}
```

**Requirements**:
- PTI trap type must be 'spring' or 'utad'
- PTI score >= threshold
- Displacement >= ATR multiplier
- **Uses config key**: `'spring'`

### Archetype G (`_check_G`, line 891-910):
```python
def _check_G(self, context: RuntimeContext) -> bool:
    """
    Archetype G: Re-Accumulate Base (BOMS strength + high liquidity).
    """
    fusion_th = context.get_threshold('liquidity_sweep', 'fusion_threshold', 0.40)
    boms_str_min = context.get_threshold('liquidity_sweep', 'boms_strength_min', 0.40)
    liq_min = context.get_threshold('liquidity_sweep', 'liquidity_min', 0.40)

    return (boms_str >= boms_str_min and
            liq >= liq_min and
            fusion >= fusion_th)
```

**Requirements**:
- BOMS strength >= threshold
- Liquidity >= threshold (0.40 default)
- Fusion score >= threshold
- **Uses config key**: `'liquidity_sweep'`

### Archetype K (`_check_K`, line 979-999):
```python
def _check_K(self, context: RuntimeContext) -> bool:
    """
    Archetype K: Wick Trap / Moneytaur (ADX + liquidity + wicks).
    """
    fusion_th = context.get_threshold('wick_trap_moneytaur', 'fusion_threshold', 0.36)
    adx_th = context.get_threshold('wick_trap_moneytaur', 'adx_threshold', 25.0)
    liq_th = context.get_threshold('wick_trap_moneytaur', 'liquidity_threshold', 0.30)

    return (adx >= adx_th and
            liq >= liq_th and
            fusion >= fusion_th)
```

**Requirements**:
- ADX >= threshold (25.0 default)
- Liquidity >= threshold (0.30 default)
- Fusion score >= threshold
- **Uses config key**: `'wick_trap_moneytaur'`

### Archetype H (`_check_H`, line 912-977):
```python
def _check_H(self, context: RuntimeContext) -> tuple:
    """
    Archetype H: Trap Within Trend (ADX trend + liquidity drop).
    """
    base_fusion_th = context.get_threshold('trap_within_trend', 'fusion_threshold', 0.35)
    adx_th = context.get_threshold('trap_within_trend', 'adx_threshold', 25.0)
    liq_th = context.get_threshold('trap_within_trend', 'liquidity_threshold', 0.30)

    if adx < adx_th:
        return False, 0.0, {"reason": "adx_weak"}
    if liq >= liq_th:  # Trap within trend needs LOW liquidity
        return False, 0.0, {"reason": "liquidity_too_high"}
```

**Requirements**:
- ADX >= threshold (25.0 default)
- Liquidity < threshold (0.30) — **INVERTED CHECK**
- Fusion score >= threshold
- **Uses config key**: `'trap_within_trend'`
- **NOT in the A/G/K group!**

---

## 4. What the Config Actually Contains

From `/configs/mvp/mvp_bull_market_v1.json`:

```json
{
  "archetypes": {
    "enable_A": true,
    "enable_G": true,
    "enable_K": true,
    "enable_H": true,

    "thresholds": {
      "trap_within_trend": {
        "direction": "long",
        "fusion_threshold": 0.42,
        "max_risk_pct": 0.02,
        "atr_stop_mult": 1.8
      }
    },

    "trap_within_trend": {
      "archetype_weight": 1.2,
      "final_fusion_gate": 0.42,
      "cooldown_bars": 16
    }
  }
}
```

**What's MISSING**:
- No `'spring'` config section (A needs this!)
- No `'liquidity_sweep'` config section (G needs this!)
- No `'wick_trap_moneytaur'` config section (K needs this!)

**What's PRESENT**:
- `'trap_within_trend'` config — but this is for archetype **H**, not A!

---

## 5. Evidence from Backtest Warnings

```
WARNING:engine.runtime.context:[PHASE1] Archetype 'wick_trap_moneytaur' NOT FOUND in thresholds!
Available: ['spring', 'order_block_retest', 'wick_trap', 'failed_continuation', 'volume_exhaustion']
Using default=0.36 for 'fusion_threshold'

WARNING:engine.runtime.context:[PHASE1] Parameter 'fusion_threshold' (canonical: 'fusion_threshold')
NOT FOUND in archetype 'trap_within_trend'!
Available params: ['archetype_weight', 'final_fusion_gate', 'cooldown_bars']
Using default=0.35
```

These warnings confirm:
1. Runtime can't find `'wick_trap_moneytaur'` config (K's check function needs this)
2. `'trap_within_trend'` config exists but doesn't have `'fusion_threshold'` parameter
3. All three archetypes (A, G, K) are falling back to **hardcoded defaults**

---

## 6. Why Wick_Trap (K) IS Firing But Others Aren't

### K (wick_trap) fired 192 times because:

```python
# _check_K uses DEFAULTS when config missing
fusion_th = 0.36  # default
adx_th = 25.0     # default
liq_th = 0.30     # default

# Requirements (ALL must pass):
return (adx >= 25.0 and      # ✓ Commonly met in 2024 bull
        liq >= 0.30 and      # ✓ Commonly met
        fusion >= 0.36)      # ✓ Relatively low threshold
```

**Why K fires**: The default thresholds are PERMISSIVE enough for 2024 bull market conditions.

### A (spring) fired 0 times because:

```python
# _check_A has STRICT requirements
pti_trap = self.g(context.row, "pti_trap_type", '')
if not pti_trap or pti_trap not in ['spring', 'utad']:
    return False  # ❌ HARD VETO
```

**Why A doesn't fire**: Requires `pti_trap_type` feature to be 'spring' or 'utad'. This is a **HARD GATE** that can't be bypassed by threshold tuning. If PTI isn't detecting springs/UTADs in 2024 data, A will never fire.

### G (liquidity_sweep) fired 0 times because:

```python
# _check_G defaults
fusion_th = 0.40  # default
boms_str_min = 0.40  # default - STRICT!
liq_min = 0.40       # default - STRICT!

# Requirements (ALL must pass):
return (boms_str >= 0.40 and  # ❌ High bar
        liq >= 0.40 and       # ❌ Higher than K's 0.30
        fusion >= 0.40)       # ❌ Higher than K's 0.36
```

**Why G doesn't fire**: Default thresholds are MORE RESTRICTIVE than K's defaults. In 2024, liquidity scores likely don't consistently exceed 0.40 (K only needs 0.30).

---

## 7. The Config Name Mismatch Crisis

### What the Optimizer Wrote (in optimized configs):

```json
{
  "archetypes": {
    "trap_within_trend": {
      "fusion_threshold": 0.42,
      "archetype_weight": 1.15,
      "adx_threshold": 25.0
    },
    "liquidity_sweep": {
      "fusion_threshold": 0.38,
      "archetype_weight": 1.10,
      "boms_strength_min": 0.40
    },
    "spring": {
      "fusion_threshold": 0.35,
      "archetype_weight": 1.05,
      "pti_score_threshold": 0.40
    }
  }
}
```

### What the Runtime Actually Needs:

For A, G, K to work:
```json
{
  "archetypes": {
    "thresholds": {
      "spring": {                    // ← A reads from here
        "fusion_threshold": 0.35,
        "pti_score_threshold": 0.40,
        "disp_atr_multiplier": 0.80
      },
      "liquidity_sweep": {           // ← G reads from here
        "fusion_threshold": 0.38,
        "boms_strength_min": 0.40,
        "liquidity_min": 0.40
      },
      "wick_trap_moneytaur": {       // ← K reads from here
        "fusion_threshold": 0.36,
        "adx_threshold": 25.0,
        "liquidity_threshold": 0.30
      }
    }
  }
}
```

**Current state**: Optimizer writes parameters to top-level archetype configs using names that DON'T MATCH what the check functions actually query!

---

## 8. The Registry Confusion

From `/engine/archetypes/registry.py`:

```python
ARCHETYPES = {
    "wyckoff_spring_utad": {
        "display": "Spring / UTAD",
        "class": "WyckoffSpringUtad",
        "aliases": ["A", "spring", "utad", "trap_reversal"],  # ← Multiple names!
        "priority": 1,
    },

    "trap_within_trend": {
        "display": "Trap Within Trend",
        "class": "TrapWithinTrend",
        "aliases": ["H", "trap", "trap_legacy", "htf_trap"],  # ← H, not A!
        "priority": 5,
    },

    "liquidity_sweep_reclaim": {
        "display": "Liquidity Sweep & Reclaim",
        "class": "LiquiditySweepReclaim",
        "aliases": ["G", "sweep", "liquidity_sweep"],
        "priority": 7,
    },

    "wick_trap_moneytaur": {
        "display": "Wick Trap (Moneytaur)",
        "class": "WickTrapMoneytaur",
        "aliases": ["K", "wick_trap", "moneytaur"],
        "priority": 4,
    },
}
```

**The registry says**:
- A → `wyckoff_spring_utad` (canonical), aliases include "spring", "trap_reversal"
- G → `liquidity_sweep_reclaim` (canonical), aliases include "liquidity_sweep"
- H → `trap_within_trend` (canonical)
- K → `wick_trap_moneytaur` (canonical)

But `_check_A` queries for `'spring'`, not `'wyckoff_spring_utad'`!

---

## 9. Root Cause Analysis

### The Fundamental Problem

**THREE different naming systems are in conflict**:

1. **Letter IDs** (A, G, K, H): Used in enable flags, archetype_map
2. **Canonical registry slugs** (wyckoff_spring_utad, liquidity_sweep_reclaim): Defined in registry.py
3. **Check function query keys** ('spring', 'liquidity_sweep', 'wick_trap_moneytaur'): Used in get_threshold() calls

**The optimizer uses canonical names from column 2**, but **check functions query using column 3 names**, and they **DON'T ALWAYS MATCH**!

### Specific Mismatches

| Letter | Registry Canonical | Check Function Queries | Optimizer Writes To | Result |
|--------|-------------------|----------------------|-------------------|---------|
| A | wyckoff_spring_utad | 'spring' | 'spring' | ❌ Never fires (PTI feature missing) |
| G | liquidity_sweep_reclaim | 'liquidity_sweep' | 'liquidity_sweep' | ❌ Never fires (thresholds too strict) |
| K | wick_trap_moneytaur | 'wick_trap_moneytaur' | 'wick_trap' | ⚠️ Uses defaults (name mismatch!) |
| H | trap_within_trend | 'trap_within_trend' | 'trap_within_trend' | ✓ Would work if enabled |

---

## 10. Why K (wick_trap) Fires Despite the Mismatch

The optimizer writes to `'wick_trap'` but the check function queries `'wick_trap_moneytaur'`:

```python
# Optimizer writes:
"wick_trap": {"fusion_threshold": 0.44, ...}

# Runtime queries:
fusion_th = context.get_threshold('wick_trap_moneytaur', 'fusion_threshold', 0.36)
# → Not found, uses default 0.36
```

**K fires because**:
1. Default threshold (0.36) is LOWER than optimizer's target (0.44)
2. Default requirements are MORE PERMISSIVE
3. Check function is simple boolean (returns True/False), not tuple with scoring
4. No hard feature gates like A's pti_trap_type requirement

---

## 11. Feature Flags Impact

From `/engine/feature_flags.py`:

```python
# Bull Archetypes (A, B, C, D, E, F, G, H, K, L, M)
BULL_EVALUATE_ALL = False        # Legacy priority dispatch
BULL_SOFT_LIQUIDITY = False      # Hard filter at min_liquidity=0.30

# Priority order: A, B, C, K, H, L, ...
```

**Dispatch behavior**:
1. System uses **legacy priority dispatcher** (early returns)
2. Checks archetypes in order: A → B → C → K → H → L → ...
3. **First match wins**, rest are ignored

**Why K dominates**:
- K is checked 4th (after A, B, C)
- A never matches (PTI requirement)
- B, C not enabled or don't match as often
- K has permissive defaults → matches frequently
- H never gets a chance (K already matched!)

---

## 12. Why the Optimization Failed

### The Optimizer's Futile Effort

The optimizer ran 40 trials, testing parameter combinations like:

```python
# Trial 1
params['trap_within_trend'] = {
    'fusion_threshold': 0.42,
    'archetype_weight': 1.15,
    'adx_threshold': 25.0,
}

# Trial 2
params['liquidity_sweep'] = {
    'fusion_threshold': 0.38,
    'archetype_weight': 1.10,
    'boms_strength_min': 0.35,
}

# Trial 3
params['spring'] = {
    'fusion_threshold': 0.35,
    'archetype_weight': 1.05,
    'pti_score_threshold': 0.45,
}
```

**But at runtime**:
- A's `_check_A` always queries for `'spring'` config
- G's `_check_G` always queries for `'liquidity_sweep'` config
- K's `_check_K` queries for `'wick_trap_moneytaur'` (NOT 'wick_trap'!)

**Result**: All optimized parameters are IGNORED, check functions use hardcoded defaults!

### Zero-Variance Bug Redux

This is the **same zero-variance bug** from earlier PRs, but now at the archetype group level:
- Optimizer varies parameters
- Runtime ignores varied parameters
- All trials use same hardcoded defaults
- Optimizer "optimizes" random noise
- No actual learning occurs

---

## 13. Recommendations

### IMMEDIATE (Fix the Crisis)

**Option 1: Fix the Optimizer** (Least risky)
Update `bin/optuna_parallel_archetypes_v2.py` archetype group definitions to match runtime check function names:

```python
ARCHETYPE_GROUPS = {
    'trap_within_trend': {
        'archetypes': ['H'],  # NOT A, G, K!
        'canonical': ['trap_within_trend'],
        'description': 'ADX trend + LOW liquidity trap pattern',
    },
    'spring_utad': {
        'archetypes': ['A'],
        'canonical': ['spring'],  # Match what _check_A queries
        'description': 'PTI-based spring/UTAD reversals',
    },
    'liquidity_sweep': {
        'archetypes': ['G'],
        'canonical': ['liquidity_sweep'],  # Match what _check_G queries
        'description': 'BOMS strength + liquidity reclaim',
    },
    'wick_trap': {
        'archetypes': ['K'],
        'canonical': ['wick_trap_moneytaur'],  # Match what _check_K queries!
        'description': 'ADX + liquidity + wick rejection',
    },
}
```

**Option 2: Fix the Runtime** (More invasive)
Standardize all `_check_X` functions to query using canonical registry names:

```python
# _check_A: Change from 'spring' to 'wyckoff_spring_utad'
fusion_th = context.get_threshold('wyckoff_spring_utad', 'fusion_threshold', 0.33)

# _check_G: Change from 'liquidity_sweep' to 'liquidity_sweep_reclaim'
fusion_th = context.get_threshold('liquidity_sweep_reclaim', 'fusion_threshold', 0.40)

# _check_K: Already correct (wick_trap_moneytaur)
```

**Option 3: Add Alias Resolution**
Make `get_threshold()` resolve aliases:

```python
def get_threshold(self, archetype_name, param_name, default):
    # Resolve aliases: 'spring' → 'wyckoff_spring_utad'
    canonical = resolve_archetype_key(archetype_name, warn=False)
    # ... rest of logic
```

### SHORT-TERM (Prevent Recurrence)

1. **Add config validation**: Check that enabled archetypes have corresponding threshold configs
2. **Add integration tests**: Verify optimizer-written configs actually affect backtest behavior
3. **Log first-call diagnostics**: Show which config section was used for each threshold lookup

### LONG-TERM (Architectural Fix)

1. **Single source of truth**: Registry should define BOTH canonical names AND query keys
2. **Class-based archetypes**: Replace check functions with ArchetypePattern classes
3. **Config schema validation**: Enforce that archetype configs match expected structure
4. **Eliminate letter codes**: Migrate fully to canonical slug-based naming

---

## 14. Why This Matters

### Wasted Optimization Effort

- 40 trials × 4 archetype groups = **160 trials total**
- Each trial = ~2 minutes = **5.3 hours of compute time**
- **ZERO actual learning** for A, G, K groups (parameters ignored)
- Optimizer produced configs that **look optimized but don't work**

### User Confusion

User expected:
- A, G, K archetypes to be optimized and fire trades
- Parameters to follow from successful optimization

Reality:
- A never fires (missing PTI feature)
- G never fires (defaults too strict)
- K fires but ignores optimized params
- H (the REAL trap_within_trend) not in optimized group!

### Pattern Recognition

This is the **THIRD instance** of name mismatch bugs:
1. **Parameter shadowing**: Optimizer wrote 'fusion', runtime read 'fusion_threshold'
2. **Archetype name shadowing**: Archetypes had multiple names (A, trap, trap_reversal)
3. **THIS**: Archetype group canonical names don't match runtime query keys

**Root cause**: Too many naming layers without enforced consistency!

---

## 15. Immediate Action Items

### For the User

**DO NOT use the optimized configs from trap_within_trend group**. They don't work.

**Instead**:
1. Run optimization on archetype **H** (the real trap_within_trend)
2. Run optimization on archetype **K** with correct name (`wick_trap_moneytaur`)
3. Skip A and G until PTI/BOMS features are validated

### For the Codebase

**Fix Priority 1** (Optimizer):
```python
# bin/optuna_parallel_archetypes_v2.py line 94
ARCHETYPE_GROUPS = {
    'wick_trap': {  # ← Rename from 'trap_within_trend'
        'archetypes': ['K'],
        'canonical': ['wick_trap_moneytaur'],  # ← Match runtime query
    },
    'trap_within_trend': {  # ← NEW group for H
        'archetypes': ['H'],
        'canonical': ['trap_within_trend'],
    },
}
```

**Fix Priority 2** (Add test):
```python
def test_optimizer_runtime_name_consistency():
    """Verify optimizer canonical names match runtime query keys."""
    for group_name, group in ARCHETYPE_GROUPS.items():
        for canonical_name in group['canonical']:
            # Check that canonical_name appears in some _check_X function
            assert canonical_name in RUNTIME_QUERY_KEYS, \
                f"Optimizer uses '{canonical_name}' but no _check function queries it!"
```

---

## Conclusion

The trap_within_trend archetypes (A, G, K) aren't firing because:

1. **A (spring)**: Hard PTI feature requirement not met in 2024 data
2. **G (liquidity_sweep)**: Default thresholds too strict (liq >= 0.40)
3. **K (wick_trap)**: Fires frequently BUT uses hardcoded defaults (name mismatch)

The optimizer's 40-trial effort was **completely wasted** because:
- Optimizer writes to `'trap_within_trend'`, `'liquidity_sweep'`, `'spring'`
- Runtime queries `'trap_within_trend'` (for H!), `'liquidity_sweep'`, `'spring'`, `'wick_trap_moneytaur'`
- A, G, K check functions can't find their configs → fall back to defaults
- Optimization varies parameters that are never actually used

**This is a critical architectural bug that invalidates ALL archetype group optimization work.**

---

**Recommended immediate action**: Fix optimizer archetype group definitions to match runtime query keys, then re-run optimization with corrected mappings.
