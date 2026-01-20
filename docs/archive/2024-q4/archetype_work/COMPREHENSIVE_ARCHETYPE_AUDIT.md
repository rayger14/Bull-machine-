# Comprehensive Architectural Audit: Archetype Parameter System

**Date:** 2025-11-08  
**Scope:** Complete parameter flow from config to archetype logic  
**Status:** CRITICAL ISSUES IDENTIFIED

---

## EXECUTIVE SUMMARY

The archetype parameter system is in a **transitional state** between legacy (letter codes, hardcoded thresholds) and modern (canonical slugs, runtime config) architectures. This creates:

1. **Multiple incompatible parallel code paths** (logic.py vs logic_v2_adapter.py)
2. **Naming mismatches** between config keys and code expectations
3. **Hardcoded defaults** in multiple layers that bypass config
4. **Zero-variance bug** where optimizer writes don't reach archetype logic
5. **Dead code** from incomplete migration

The system is **NOT unified** and blocking optimization work.

---

## 1. COMPLETE PARAMETER FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CONFIG FILE (JSON)                                                          │
│ ├─ config['archetypes']['thresholds'][LETTER_CODE] (LEGACY)                 │
│ │  └─ {'fusion': 0.35, 'pti': 0.40, 'disp_atr': 0.80, ...}                 │
│ │                                                                             │
│ ├─ config['archetypes'][SLUG_NAME] (NEW)                                    │
│ │  └─ {'fusion_threshold': 0.35, 'pti_score_threshold': 0.40, ...}         │
│ │                                                                             │
│ ├─ config['gates_regime_profiles'] (REGIME BLENDING)                        │
│ │  └─ {'risk_on': {'final_fusion_floor': 0.52, 'min_liquidity': 0.20}}     │
│ │                                                                             │
│ └─ config['archetype_overrides'] (PER-REGIME DELTAS)                        │
│    └─ {'trap_within_trend': {'risk_on': {'fusion': -0.02}}}                │
└─────────────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ THRESHOLD POLICY (engine/archetypes/threshold_policy.py)                    │
│                                                                              │
│ ThresholdPolicy.__init__(base_cfg, regime_profiles, overrides)             │
│  └─ Line 90: self.base_arch_thresholds = config['archetypes']['thresholds']│
│                                                                              │
│ ThresholdPolicy.resolve(regime_probs, regime_label)                        │
│  ├─ Line 125: final = self._build_base_map()                               │
│  │  └─ Line 156: Reads BOTH top-level archetype AND thresholds/            │
│  │     PRIORITY: top-level > thresholds/slug > thresholds/letter > {}      │
│  ├─ Line 128: blended_gates = self._blend_regime_gates(regime_probs)       │
│  ├─ Line 131: self._apply_regime_floors(final, blended_gates)              │
│  ├─ Line 134: self._apply_archetype_overrides(final, regime_label)         │
│  └─ Line 137: self._clamp(final)                                           │
│                                                                              │
│ RETURNS: Dict[archetype_slug → Dict[param_name → value]]                  │
│  Example: {'order_block_retest': {'fusion_threshold': 0.33, ...}, ...}    │
└─────────────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ RUNTIME CONTEXT (engine/runtime/context.py)                                │
│                                                                              │
│ @dataclass RuntimeContext                                                   │
│  ├─ thresholds: Dict[str, Dict[str, float]]                                │
│  │  └─ Populated by ThresholdPolicy.resolve()                              │
│  │     Example: thresholds={'order_block_retest': {'fusion_threshold': ... }}│
│  │                                                                           │
│  └─ get_threshold(archetype: str, param: str, default: float)              │
│     └─ Line 54: return self.thresholds.get(archetype, {}).get(param, default)│
│        Returns threshold from context or default                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ ARCHETYPE LOGIC (2 PARALLEL IMPLEMENTATIONS)                               │
│                                                                              │
│ PATH 1: logic_v2_adapter.py (NEW - RuntimeContext-aware)                   │
│ ├─ detect(ctx: RuntimeContext) → (archetype_name, fusion_score, liq_score) │
│ │  └─ Dispatches to _check_X(ctx) for each enabled archetype               │
│ │                                                                             │
│ └─ Each _check_X reads thresholds:                                         │
│    ├─ Line 385 (_check_A):                                                 │
│    │  fusion_th = ctx.get_threshold('spring', 'fusion_threshold', 0.33)    │
│    │  pti_score_th = ctx.get_threshold('spring', 'pti_score_threshold', ...) │
│    │  disp_multiplier = ctx.get_threshold('spring', 'disp_atr_multiplier', ...)│
│    │                                                                        │
│    ├─ Line 411 (_check_B):                                                 │
│    │  fusion_th = ctx.get_threshold('order_block_retest', 'fusion_..', 0.374)│
│    │  boms_str_th = ctx.get_threshold('order_block_retest', 'boms_strength_min')│
│    │  wyckoff_th = ctx.get_threshold('order_block_retest', 'wyckoff_min', ...)│
│    │  ...                                                                   │
│    │                                                                        │
│    └─ CRITICAL: Uses canonical SLUG names + SPECIFIC param names           │
│       Example: get_threshold('order_block_retest', 'fusion_threshold', ...)│
│                                                                             │
│ PATH 2: logic.py (LEGACY - non-RuntimeContext, hardcoded)                  │
│ ├─ check_archetype(row, prev_row, df, index)                              │
│ │  └─ Uses self.thresh_X dicts from __init__                              │
│ │     self.thresh_A = config.get('thresholds', {}).get('A', {})          │
│ │                                                                           │
│ └─ Each _check_X reads from self.thresh_X:                                │
│    ├─ Line 512 (_check_A): self.thresh_A.get('disp_atr', 0.80)           │
│    ├─ Line 531 (_check_B): self.thresh_B.get('boms_strength', 0.30)      │
│    └─ PROBLEM: Uses SHORT param names (disp_atr, boms_strength, NOT       │
│       fusion_threshold)                                                     │
│       This causes ZERO-VARIANCE: optimizer writes to 'fusion_threshold'   │
│       but logic reads from 'fusion' ← MISMATCH!                           │
│                                                                             │
│ SPECIAL CASES - _check_H and _check_K in logic.py:                       │
│ ├─ Line 802 (_check_H): get_param(self, 'trap_within_trend', ...)        │
│ ├─ Line 863 (_check_K): get_param(self, 'wick_trap_moneytaur', ...)      │
│ │  └─ Uses param_accessor.get_param() with canonical slugs!               │
│ │     This is the CORRECT approach but only for 2 archetypes              │
│ └─ OTHER _check_X methods still use self.thresh_X (BROKEN)                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                   ↓
        (Archetype matching + positioning)
```

---

## 2. CRITICAL INCONSISTENCIES FOUND

### A. NAMING MISMATCHES: Letter Codes vs Canonical Slugs

| Archetype | Letter | logic_v2_adapter.py Slug | logic.py Uses | Config Key |
|-----------|--------|--------------------------|----------------|-----------|
| A | A | 'spring' | 'A' in thresholds | 'archetypes']['A'] |
| B | B | 'order_block_retest' | 'B' in thresholds | 'archetypes']['B'] |
| C | C | 'wick_trap' | 'C' in thresholds | 'archetypes']['C'] |
| D | D | 'failed_continuation' | 'D' in thresholds | 'archetypes']['D'] |
| E | E | 'volume_exhaustion' | 'E' in thresholds | 'archetypes']['E'] |
| F | F | 'exhaustion_reversal' | 'F' in thresholds | 'archetypes']['F'] |
| G | G | 'liquidity_sweep' | 'G' in thresholds | 'archetypes']['G'] |
| H | H | 'momentum_continuation' OR 'trap_within_trend' | 'H' in thresholds | 'archetypes']['H'] |
| K | K | 'trap_within_trend' | 'K' in thresholds | 'archetypes']['K'] |
| L | L | 'retest_cluster' | 'L' in thresholds | 'archetypes']['L'] |
| M | M | 'confluence_breakout' | 'M' in thresholds | 'archetypes']['M'] |

**PROBLEM**: logic.py expects letter codes in config['archetypes']['thresholds'][LETTER],  
but optimizer writes to config['archetypes'][CANONICAL_SLUG]. **ZERO VARIANCE!**

### B. PARAMETER NAME MISMATCHES

**logic.py reads SHORT names from config['archetypes']['thresholds'][LETTER]:**

```python
# Line 504 (logic.py _check_A):
self.thresh_A.get('disp_atr', 0.80)    # ← expects 'disp_atr' key

# Line 531 (logic.py _check_B):
self.thresh_B.get('boms_strength', 0.30)  # ← expects 'boms_strength' key
self.thresh_B.get('wyckoff', 0.35)        # ← expects 'wyckoff' key

# Line 536 (logic.py _check_B):
self.thresh_C.get('disp_atr', 1.00)       # ← expects 'disp_atr' key
```

**logic_v2_adapter.py reads LONG names via context.get_threshold():**

```python
# Line 385 (logic_v2_adapter.py _check_A):
ctx.get_threshold('spring', 'fusion_threshold', 0.33)        # ← LONG name
ctx.get_threshold('spring', 'pti_score_threshold', 0.40)    # ← LONG name

# Line 411 (logic_v2_adapter.py _check_B):
ctx.get_threshold('order_block_retest', 'fusion_threshold', 0.374)  # ← LONG
ctx.get_threshold('order_block_retest', 'boms_strength_min', 0.30) # ← LONG
```

**Actual config (baseline_btc_bull_pf20.json) uses SHORT names:**

```json
{
  "archetypes": {
    "thresholds": {
      "A": {"pti": 0.4, "disp_atr": 0.8, "fusion": 0.33},
      "B": {"fusion": 0.359...},
      ...
    }
  }
}
```

### C. HARDCODED DEFAULTS EVERYWHERE

**Problem**: Even when thresholds dict is populated, defaults are HARDCODED as fallback.

#### In logic_v2_adapter.py - Hardcoded Defaults:

| Line | Archetype | Parameter | Hardcoded Default | Should Read From |
|------|-----------|-----------|-------------------|------------------|
| 385 | spring | fusion_threshold | **0.33** | config |
| 386 | spring | pti_score_threshold | **0.40** | config |
| 387 | spring | disp_atr_multiplier | **0.80** | config |
| 411 | order_block_retest | fusion_threshold | **0.374** | config |
| 412 | order_block_retest | boms_strength_min | **0.30** | config |
| 413 | order_block_retest | wyckoff_min | **0.35** | config |
| 434 | wick_trap | fusion_threshold | **0.42** | config |
| 436 | wick_trap | momentum_min | **0.45** | config |
| 463 | failed_continuation | fusion_threshold | **0.42** | config |
| 464 | failed_continuation | rsi_max | **50.0** | config |
| 485 | volume_exhaustion | fusion_threshold | **0.35** | config |
| 486 | volume_exhaustion | atr_percentile_max | **0.25** | config |
| 511 | exhaustion_reversal | fusion_threshold | **0.38** | config |
| 512 | exhaustion_reversal | rsi_min | **78.0** | config |
| 535 | liquidity_sweep | fusion_threshold | **0.40** | config |
| 536 | liquidity_sweep | boms_strength_min | **0.40** | config |
| 556 | momentum_continuation | fusion_threshold | **0.35** | config |
| 557 | momentum_continuation | adx_threshold | **25.0** | config |
| 578 | trap_within_trend | fusion_threshold | **0.36** | config |
| 579 | trap_within_trend | adx_threshold | **25.0** | config |
| 599 | retest_cluster | fusion_threshold | **0.38** | config |
| 600 | retest_cluster | vol_z_min | **1.0** | config |
| 620 | confluence_breakout | fusion_threshold | **0.35** | config |
| 621 | confluence_breakout | atr_percentile_max | **0.30** | config |

**Impact**: Even if config has values, these defaults are used if threshold not in context.  
**Optimizer writes to config['archetypes'][SLUG], but logic reads defaults → ZERO VARIANCE.**

#### In logic.py - Hardcoded Defaults in __init__:

```python
# Line 46-60 (logic.py __init__):
self.min_liquidity = thresholds.get('min_liquidity', 0.30)  # ← HARDCODED 0.30

# These are loaded from config but NEVER read from if optimizer writes to canonical location!
self.thresh_A = thresholds.get('A', {})
self.thresh_B = thresholds.get('B', {})
...
```

### D. MULTIPLE INCOMPATIBLE CODE PATHS

**Problem**: Two entirely different archetype implementations coexist:

```
engine/archetypes/logic.py
├─ Uses non-RuntimeContext API (row, prev_row, df, index)
├─ Reads from self.thresh_X dicts
├─ Has legacy priority dispatch with early returns
└─ Used by: backtest_knowledge_v2.py (?)

engine/archetypes/logic_v2_adapter.py  
├─ Uses RuntimeContext API
├─ Reads from context.get_threshold()
├─ Has new all-archetype evaluation
└─ Used by: new router code
```

**Question**: Which is actually being used? **UNCLEAR FROM CODEBASE.**

### E. PARAMETER NAME INCONSISTENCIES ACROSS ARCHETYPES

**In logic_v2_adapter.py, the naming is INCONSISTENT:**

```python
# Some use 'fusion_threshold' (BASE PARAM):
ctx.get_threshold('spring', 'fusion_threshold', 0.33)

# Some use archetype-specific suffixes:
ctx.get_threshold('order_block_retest', 'boms_strength_min', 0.30)
ctx.get_threshold('order_block_retest', 'wyckoff_min', 0.35)
ctx.get_threshold('wick_trap', 'momentum_min', 0.45)

# Inconsistent: some use underscores, some don't
ctx.get_threshold('momentum_continuation', 'adx_threshold', 25.0)
vs
ctx.get_threshold('order_block_retest', 'boms_strength_min', 0.30)
```

**Config actually has SHORT names:**

```json
{
  "thresholds": {
    "B": {"fusion": 0.359, "boms_strength": 0.30, "wyckoff": 0.35}
  }
}
```

**MISMATCH**: Code reads 'boms_strength_min' but config has 'boms_strength' → NOT FOUND → default used.

---

## 3. THE ZERO-VARIANCE BUG EXPLAINED

### Root Cause

**Optimizer writes:**
```python
config['archetypes']['trap_within_trend']['quality_threshold'] = 0.55
config['archetypes']['trap_within_trend']['adx_threshold'] = 25.0
```

**logic.py reads from:**
```python
self.thresh_H = config.get('thresholds', {}).get('H', {})  # Empty dict!
# So self.thresh_H['quality_threshold'] → KeyError → uses hardcoded default
```

**logic_v2_adapter.py reads from context, but context is populated wrong:**
```python
ctx.get_threshold('trap_within_trend', 'quality_threshold', 0.55)
# Context thresholds dict has 'trap_within_trend' key... MAYBE
# But only if ThresholdPolicy._build_base_map() found it
```

### Where It Goes Wrong

**ThresholdPolicy._build_base_map() (Line 154-177):**
```python
for arch_name in ARCHETYPE_NAMES:  # ['spring', 'order_block_retest', ...]
    # PR#6A: Try top-level archetype config FIRST (where optimizer writes)
    if arch_name in archetypes and isinstance(archetypes[arch_name], dict):
        base_map[arch_name] = deepcopy(archetypes[arch_name])
        # ✓ This SHOULD pick up optimizer-written params
        
    # Fallback to letter code in thresholds subdirectory (LEGACY)
    elif arch_name in LEGACY_ARCHETYPE_MAP:
        letter_code = LEGACY_ARCHETYPE_MAP[arch_name]
        if letter_code in self.base_arch_thresholds:
            base_map[arch_name] = deepcopy(self.base_arch_thresholds[letter_code])
```

**BUT**: This only works if config has CANONICAL SLUG at top level:
```json
{
  "archetypes": {
    "trap_within_trend": {             // ← Optimizer writes here
      "quality_threshold": 0.55,
      "adx_threshold": 25.0
    }
  }
}
```

**If config uses legacy format, it FAILS:**
```json
{
  "archetypes": {
    "thresholds": {
      "H": {                          // ← Old code reads here
        "quality_threshold": 0.55
      }
    }
  }
}
```

---

## 4. FALLBACK/DEFAULT MECHANISMS

### Layer 1: ThresholdPolicy.resolve()

```
If thresholds dict is empty (locked_regime='static'):
  → Line 298-300: Returns EMPTY dict
  → Causes get_threshold() to use hardcoded defaults
  → This is INTENTIONAL for parity testing
```

### Layer 2: RuntimeContext.get_threshold()

```
get_threshold(archetype, param, default=0.0):
  → Line 54: self.thresholds.get(archetype, {}).get(param, default)
  → Waterfall: if param NOT in thresholds dict
    → Uses 'default' parameter (0.0 if not specified)
    → Does NOT fallback to hardcoded logic defaults
```

### Layer 3: Archetype Logic Defaults

**In logic_v2_adapter.py _check_X methods:**
```python
fusion_th = ctx.get_threshold('spring', 'fusion_threshold', 0.33)
                                                            ↑ This is the default!
# If context has NO entry for 'spring'.'fusion_threshold':
#   → get_threshold returns 0.33
# Even if config has {'spring': {'other_param': ...}}
```

### Layer 4: logic.py Defaults

```python
# Line 504 (_check_A):
if tf4h_disp < self.thresh_A.get('disp_atr', 0.80) * atr:  # ← hardcoded 0.80
# If config['thresholds']['A'] exists but has NO 'disp_atr' key:
#   → defaults to 0.80
```

---

## 5. CONFIG READING PATHS

### Path A: ThresholdPolicy (PR#6A/PR#6B)

**File**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/engine/archetypes/threshold_policy.py`

```python
def __init__(base_cfg):
  Line 90: self.base_arch_thresholds = base_cfg.get('archetypes', {}).get('thresholds', {})
           # Only reads thresholds subdirectory!
           
def _build_base_map():
  Line 152: archetypes = self.base.get('archetypes', {})
  Line 154-161: for arch_name in ARCHETYPE_NAMES:
                  if arch_name in archetypes:  # ← Checks TOP-LEVEL
                    base_map[arch_name] = deepcopy(archetypes[arch_name])
                    # ✓ Good! Picks up optimizer-written params
                    
  Line 168-172: Fallback to LEGACY_ARCHETYPE_MAP
                # Maps 'trap_within_trend' → 'H'
                # Tries self.base_arch_thresholds['H']
```

**Issues**:
- Only works if config uses BOTH top-level archetype AND legacy letter code
- Inconsistent parameter names in fallback

### Path B: legacy logic.py

**File**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/engine/archetypes/logic.py`

```python
def __init__(config):
  Line 43: self.config = config
  Line 46: thresholds = config.get('thresholds', {})  # ← WRONG!
           # Should be config.get('archetypes', {}).get('thresholds', {})
           
  Line 50-60: self.thresh_A = thresholds.get('A', {})
             self.thresh_B = thresholds.get('B', {})
             # Reads letter codes from thresholds dict
             
  Line 299 (check_archetype):
    thresholds = self._build_thresholds_from_config()
    
  Line 342 (_build_thresholds_from_config):
    for arch_name in ARCHETYPE_NAMES:
      if arch_name in self.config:  # ← checks self.config directly
        arch_config = self.config[arch_name]
        if isinstance(arch_config, dict):
          thresholds[arch_name] = arch_config
```

**Issues**:
- Line 46 reads from wrong location (thresholds directly, not archetypes.thresholds)
- self.config is the ENTIRE config, not just archetypes section
- Inconsistent: line 50-60 use letter codes, line 342 uses canonical slugs

### Path C: param_accessor.get_param()

**File**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/engine/archetypes/param_accessor.py`

```python
def get_param(ctx, slug, key, default):
  Line 67: config = ctx.config if hasattr(ctx, 'config') else ctx
  
  Line 77: # Try canonical (NEW): config['archetypes'][slug][key]
           value = config.get('archetypes', {}).get(canonical_slug, {}).get(key)
           
  Line 88: # Try legacy letter code (LEGACY): thresholds[letter][key]
           letter_val = config.get('archetypes', {}).get('thresholds', {}).get(alias, {}).get(key)
           
  Line 101: # Use default
            return default
```

**Good**: Migration-safe fallback chain  
**Used by**: Only _check_H and _check_K in logic.py (NOT all archetypes!)

---

## 6. HARDCODED VALUES AUDIT

### Critical Hardcoded Thresholds in logic_v2_adapter.py

```
Location: /Users/raymondghandchi/Bull-machine-/Bull-machine-/engine/archetypes/logic_v2_adapter.py

_check_A (Line 378-403):
  385:  fusion_th = ctx.get_threshold('spring', 'fusion_threshold', 0.33)
  386:  pti_score_th = ctx.get_threshold('spring', 'pti_score_threshold', 0.40)
  387:  disp_multiplier = ctx.get_threshold('spring', 'disp_atr_multiplier', 0.80)

_check_B (Line 404-426):
  411:  fusion_th = ctx.get_threshold('order_block_retest', 'fusion_threshold', 0.374)
  412:  boms_str_th = ctx.get_threshold('order_block_retest', 'boms_strength_min', 0.30)
  413:  wyckoff_th = ctx.get_threshold('order_block_retest', 'wyckoff_min', 0.35)

_check_C (Line 427-455):
  434:  fusion_th = ctx.get_threshold('wick_trap', 'fusion_threshold', 0.42)
  436:  momentum_th = ctx.get_threshold('wick_trap', 'momentum_min', 0.45)
  437:  tf4h_fusion_th = ctx.get_threshold('wick_trap', 'tf4h_fusion_min', 0.25)

_check_D (Line 456-477):
  463:  fusion_th = ctx.get_threshold('failed_continuation', 'fusion_threshold', 0.42)
  464:  rsi_max = ctx.get_threshold('failed_continuation', 'rsi_max', 50.0)

_check_E (Line 478-503):
  485:  fusion_th = ctx.get_threshold('volume_exhaustion', 'fusion_threshold', 0.35)
  486:  atr_pct_max = ctx.get_threshold('volume_exhaustion', 'atr_percentile_max', 0.25)
  487:  vol_z_min = ctx.get_threshold('volume_exhaustion', 'vol_z_min', 0.5)
  488:  vol_z_max = ctx.get_threshold('volume_exhaustion', 'vol_z_max', 1.5)
  489:  vol_cluster_min = ctx.get_threshold('volume_exhaustion', 'vol_cluster_min', 0.70)

_check_F (Line 504-527):
  511:  fusion_th = ctx.get_threshold('exhaustion_reversal', 'fusion_threshold', 0.38)
  512:  rsi_min = ctx.get_threshold('exhaustion_reversal', 'rsi_min', 78.0)
  513:  atr_pct_min = ctx.get_threshold('exhaustion_reversal', 'atr_percentile_min', 0.90)
  514:  vol_z_min = ctx.get_threshold('exhaustion_reversal', 'vol_z_min', 1.0)

_check_G (Line 528-548):
  535:  fusion_th = ctx.get_threshold('liquidity_sweep', 'fusion_threshold', 0.40)
  536:  boms_str_min = ctx.get_threshold('liquidity_sweep', 'boms_strength_min', 0.40)
  537:  liq_min = ctx.get_threshold('liquidity_sweep', 'liquidity_min', 0.40)

_check_H (Line 549-569):
  556:  fusion_th = ctx.get_threshold('momentum_continuation', 'fusion_threshold', 0.35)
  557:  adx_th = ctx.get_threshold('momentum_continuation', 'adx_threshold', 25.0)
  558:  liq_th = ctx.get_threshold('momentum_continuation', 'liquidity_threshold', 0.30)

_check_K (Line 570-591):
  578:  fusion_th = ctx.get_threshold('trap_within_trend', 'fusion_threshold', 0.36)
  579:  adx_th = ctx.get_threshold('trap_within_trend', 'adx_threshold', 25.0)
  580:  liq_th = ctx.get_threshold('trap_within_trend', 'liquidity_threshold', 0.30)

_check_L (Line 592-612):
  599:  fusion_th = ctx.get_threshold('retest_cluster', 'fusion_threshold', 0.38)
  600:  vol_z_min = ctx.get_threshold('retest_cluster', 'vol_z_min', 1.0)
  601:  rsi_min = ctx.get_threshold('retest_cluster', 'rsi_min', 70.0)

_check_M (Line 613-636):
  620:  fusion_th = ctx.get_threshold('confluence_breakout', 'fusion_threshold', 0.35)
  621:  atr_pct_max = ctx.get_threshold('confluence_breakout', 'atr_percentile_max', 0.30)
  622:  poc_dist_max = ctx.get_threshold('confluence_breakout', 'poc_dist_max', 0.50)
  623:  boms_str_min = ctx.get_threshold('confluence_breakout', 'boms_strength_min', 0.40)
```

### Additional Hardcoded in Helper Methods

```
_momentum_score (Line 149-174):
  161:  rsi = self.g(row, "rsi", 50.0)           ← hardcoded default
  166:  rsi_comp = _norm01(abs(rsi - 50.0), 0.0, 25.0)  ← 50.0, 25.0 hardcoded
  169:  adx_comp = _norm01(adx, 10.0, 40.0)     ← 10.0, 40.0 hardcoded
  172:  vol_comp = max(0.0, min(1.0, vol_z / 2.0))  ← 2.0 hardcoded

_liquidity_score (Line 176-206):
  190:  bstr = self.g(row, "boms_strength", 0.0)  ← default 0.0
  197:  disp_n = max(0.0, min(1.0, disp / (2.0 * atr)))  ← 2.0 hardcoded
  199:  liq_derived = (0.5 * bstr + 0.25 * fvg + 0.25 * disp_n)  ← weights hardcoded!

_fusion (Line 208-241):
  227:  wy = self.g(row, "wyckoff_score", 0.0)
  235:  self.fakeout_penalty * fake  ← 0.075 hardcoded (line 94)
  232:  w.get("wyckoff", 0.331) * wy  ← 0.331 hardcoded fallback
  233:  w.get("liquidity", 0.392) * liq  ← 0.392 hardcoded fallback
  234:  w.get("momentum", 0.205) * mom  ← 0.205 hardcoded fallback

__init__ (Line 57-95):
  67:  self.min_liquidity = thresholds.get('min_liquidity', 0.30)  ← 0.30 hardcoded
  89-93: fusion_weights fallback to hardcoded values if not in config
```

---

## 7. NAMING MISMATCH CATALOG

### Archetype Name Mappings (Confusion Point)

| Slug (logic_v2_adapter.py) | Letter Code (logic.py) | Registry Canonical | Config Top-Level |
|----------------------------|------------------------|--------------------|------------------|
| 'spring' | 'A' | 'wyckoff_spring_utad' | 'A' or 'spring' |
| 'order_block_retest' | 'B' | 'order_block_retest' | 'B' |
| 'wick_trap' | 'C' | 'bos_choch_reversal' | 'C' |
| 'failed_continuation' | 'D' | 'failed_continuation' | 'D' |
| 'volume_exhaustion' | 'E' | 'liquidity_compression' | 'E' |
| 'exhaustion_reversal' | 'F' | 'expansion_exhaustion' | 'F' |
| 'liquidity_sweep' | 'G' | 'liquidity_sweep_reclaim' | 'G' |
| 'momentum_continuation' | 'H' | 'momentum_continuation' | 'H' |
| 'trap_within_trend' | 'K' | 'trap_within_trend' | 'K' |
| 'retest_cluster' | 'L' | 'fakeout_real_move' | 'L' |
| 'confluence_breakout' | 'M' | 'ratio_coil_break' | 'M' |

**Problem**: Three different naming systems for same archetype!

### Parameter Name Mismatches (Detailed)

**What optimizer writes (optuna_trap_v2.py:118):**
```python
config['archetypes']['trap_within_trend'] = {
    'fusion_threshold': trap_params.get('fusion_threshold', 0.35),
    'liquidity_threshold': trap_params.get('liquidity_threshold', 0.30),
    'quality_threshold': trap_params.get('quality_threshold', 0.55),
    'adx_threshold': trap_params.get('adx_threshold', 25.0),
    'wick_multiplier': trap_params.get('wick_multiplier', 2.0)
}
```

**What logic.py reads (line 804-806 using get_param):**
```python
quality_th = get_param(self, 'trap_within_trend', 'quality_threshold', 0.55)
liquidity_th = get_param(self, 'trap_within_trend', 'liquidity_threshold', 0.30)
adx_th = get_param(self, 'trap_within_trend', 'adx_threshold', 25.0)
```

**What is in config['archetypes']['thresholds']['H'] (legacy):**
```json
{
  "H": {
    "fusion": 0.544,
    "quality": 0.55,
    "adx": 25.0
  }
}
```

**Mismatch**: Optimizer uses 'fusion_threshold', config has 'fusion', code reads 'quality_threshold'

---

## 8. DEAD CODE / DEPRECATED PATHS

### Dead Code in logic.py

```
Line 312-340: check_archetype() - DEPRECATED backward compat wrapper
  └─ Comment says: "DEPRECATED: Backward compatibility wrapper"
  └─ Only used if called directly with (row, prev_row, df, index) signature
  └─ Modern code uses ArchetypeLogic_v2_adapter with RuntimeContext

Line 418-479: _check_legacy_priority() - EARLY RETURN DISPATCH
  └─ Causes archetype starvation (K blocks H)
  └─ Replaced by _check_all_archetypes()
  └─ Still present but guarded by feature flag

Line 89-107: _get_liquidity_score() - PATCHED COMPOSITE
  └─ Duplicates logic_v2_adapter.py implementation
  └─ Suggests two implementations are incomplete migration
```

### Unused Imports/Methods

```
logic.py Line 312: @deprecated marker missing
         Should have: @deprecated("Use ArchetypeLogic_v2_adapter.detect() instead")

logic.py Line 368: 'get_archetype_meta' imported but...
         Used for priority lookup in _check_all_archetypes()
         But _check_legacy_priority() doesn't use it (early returns instead)

param_accessor.py Line 105-110: _log_param_source() 
         Only logs if features.PARAM_ECHO_ENABLED
         No documentation on how to enable

threshold_policy.py Line 87: self.locked_regime
         Only used in _resolve_locked() for special test modes
         Comment mentions "legacy behavior" but used for PR#6A parity
```

---

## 9. SPECIFIC FILE:LINE REFERENCES

### Critical Disconnects

| Issue | File | Line | Code | Impact |
|-------|------|------|------|--------|
| Wrong config read path | logic.py | 46 | `thresholds = config.get('thresholds', {})` | Reads root level instead of archetypes.thresholds |
| Letter code hardcoded | logic.py | 50-82 | `self.thresh_A = config.get('enable_A', True)` | Still uses letter codes |
| Param name mismatch | logic.py | 504 | `self.thresh_A.get('disp_atr', 0.80)` | Expects 'disp_atr' not 'disp_atr_multiplier' |
| Two implementations | repo | — | logic.py vs logic_v2_adapter.py | DUPLICATE CODE |
| Slug vs Letter | logic_v2_adapter.py | 385 | `get_threshold('spring', ...)` | Uses slug but config has letter code |
| Hardcoded defaults everywhere | logic_v2_adapter.py | 385-623 | 70+ hardcoded values in defaults | Blocks optimizer variance |
| Context thresholds dict populate | threshold_policy.py | 154-177 | `_build_base_map()` | Only works if config uses BOTH top-level AND legacy |

---

## 10. COMPLETE LIST OF ALL INCONSISTENCIES

### Naming Inconsistencies (11 total)

1. **Archetype A**: Called 'spring' in logic_v2, 'A' in logic.py, 'wyckoff_spring_utad' in registry
2. **Archetype B**: Called 'order_block_retest' consistently (GOOD)
3. **Archetype C**: Called 'wick_trap'/'fvg_continuation' in logic_v2, 'C' in logic.py, 'bos_choch_reversal' in registry
4. **Archetype D**: Called 'failed_continuation' consistently (GOOD)
5. **Archetype E**: Called 'volume_exhaustion'/'liquidity_compression' inconsistently
6. **Archetype F**: Called 'exhaustion_reversal'/'expansion_exhaustion' inconsistently
7. **Archetype G**: Called 'liquidity_sweep'/'liquidity_sweep_reclaim' inconsistently
8. **Archetype H**: Called 'momentum_continuation' OR 'trap_within_trend' (DUAL IDENTITY!)
9. **Archetype K**: Called 'trap_within_trend' in logic_v2, 'K' in logic.py, 'wick_trap_moneytaur' in registry
10. **Archetype L**: Called 'retest_cluster'/'volume_exhaustion' inconsistently, registry has 'fakeout_real_move'
11. **Archetype M**: Called 'confluence_breakout' consistently (GOOD)

### Parameter Name Mismatches (20+ total)

1. **fusion_threshold**: logic_v2 expects 'fusion_threshold', config has 'fusion'
2. **pti_score**: logic expects 'pti', logic_v2 expects 'pti_score_threshold'
3. **disp_atr**: logic expects 'disp_atr', logic_v2 expects 'disp_atr_multiplier'
4. **boms_strength**: logic expects 'boms_strength', logic_v2 expects 'boms_strength_min'
5. **wyckoff**: logic expects 'wyckoff', logic_v2 expects 'wyckoff_min'
6. **momentum**: logic expects 'momentum', logic_v2 expects 'momentum_min'
7. **adx**: logic expects 'adx' (via thresh_H), logic_v2 expects 'adx_threshold'
8. **liquidity**: logic expects 'liq' OR 'liquidity', logic_v2 expects 'liquidity_threshold'
9. **atr_pctile**: logic expects 'atr_pctile', logic_v2 expects 'atr_percentile_max'
10. **rsi_max**: logic expects 'rsi_max' in thresh_D, logic_v2 expects 'rsi_max' (match!)
... (10+ more minor variations)

### Hardcoded Values (70+ total)

See section 6 above - every _check_X method has 3-5 hardcoded thresholds.

### Dead Code (4 instances)

1. **logic.py check_archetype()**: Deprecated wrapper
2. **logic.py _check_legacy_priority()**: Feature-gated dead code
3. **param_accessor._log_param_source()**: Only logs if feature flag
4. **Multiple duplicate implementations**: logic.py vs logic_v2_adapter.py

---

## 11. PARAMETER FLOW SUMMARY TABLE

| Stage | Format Expected | Format Actual | Location | Issues |
|-------|-----------------|---------------|----------|--------|
| **Config** | Letter code thresholds | Letter code LEGACY, slug NEW | config['archetypes']['thresholds'][LETTER] | Dual format |
| **Optimizer** | Writes canonical slug | Writes to 'trap_within_trend', 'wick_trap_moneytaur' | config['archetypes'][SLUG] | Correct! |
| **ThresholdPolicy** | Reads slug, writes to context | Reads from LEGACY, outputs canonical | Converts letter→slug | Works IF legacy exists |
| **RuntimeContext** | Receives slug→param dict | Receives from ThresholdPolicy | thresholds dict | Correct IF populated |
| **Archetype (v2_adapter)** | Read slug+param names | Uses slug but hardcoded defaults | get_threshold() calls | 70+ hardcoded values |
| **Archetype (legacy)** | Read letter codes | Uses self.thresh_X dicts | reads thresholds[LETTER] | Never gets optimizer values |

---

## 12. RECOMMENDATIONS FOR UNIFICATION

### Phase 1: Immediate (Block Optimization)

1. **Standardize parameter names**: Use LONG descriptive names everywhere
   - 'fusion_threshold' not 'fusion'
   - 'adx_threshold' not 'adx'
   - 'boms_strength_min' not 'boms_strength'

2. **Standardize archetype names**: Use canonical slugs everywhere
   - Replace all letter codes (A, B, C, ...) with slugs
   - Update configs to use top-level archetype keys
   - Keep letter→slug mapping ONLY in registry

3. **Consolidate implementations**:
   - Remove logic.py entirely OR make it a wrapper around logic_v2_adapter
   - Use RuntimeContext exclusively going forward
   - Fix the dual identity issue (H vs K for trap_within_trend)

### Phase 2: Remove Hardcoding

1. **Delete hardcoded defaults** in _check_X methods
   - Remove the default parameter from ctx.get_threshold()
   - Make context.thresholds mandatory (fail loudly if missing)

2. **Validate config schema** at startup
   - Use ARCHETYPE_SCHEMAS from param_accessor.py
   - Reject configs with missing required parameters

### Phase 3: Migration

1. **Update all optimizers** to write correct format
2. **Update all configs** to use canonical slugs
3. **Add migration warnings** for old-format configs

