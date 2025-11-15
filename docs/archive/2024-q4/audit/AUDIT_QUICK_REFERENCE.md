# Archetype Parameter System - Quick Reference

**Problem**: System has two incompatible code paths + hardcoded defaults = zero-variance bug  
**Status**: BLOCKING OPTIMIZATION  
**Solution**: Unify to single code path, remove hardcoding, standardize naming

---

## CRITICAL FILES

| File | Issue | Action |
|------|-------|--------|
| `engine/archetypes/logic.py` | Duplicate implementation | DELETE entirely OR make wrapper |
| `engine/archetypes/logic_v2_adapter.py` | Hardcoded defaults | REMOVE defaults from lines 385-623 |
| `engine/archetypes/threshold_policy.py` | Correct but needs validation | ADD logging in _build_base_map() |
| `engine/runtime/context.py` | Good but missing validation | ADD check that thresholds dict populated |
| `engine/archetypes/param_accessor.py` | CORRECT PATTERN | EXTEND to ALL archetypes |

---

## ZERO-VARIANCE BUG ROOT CAUSE

```
OPTIMIZER WRITES:                    CODE READS FROM:
config['archetypes']                 config['thresholds']  ← WRONG!
  ['trap_within_trend']              Only letter codes!
  ['quality_threshold'] = 0.55       Empty when optimizer writes

RESULT: Parameter never flows → always hardcoded default → no variance
```

---

## 5 HARDCODED THRESHOLD VALUES

Example from logic_v2_adapter.py (70+ total):

```python
# Line 385 - Archetype A
fusion_th = ctx.get_threshold('spring', 'fusion_threshold', 0.33)  # ← HARDCODED

# Line 411 - Archetype B  
fusion_th = ctx.get_threshold('order_block_retest', 'fusion_threshold', 0.374)  # ← HARDCODED

# Line 557 - Archetype H
adx_th = ctx.get_threshold('momentum_continuation', 'adx_threshold', 25.0)  # ← HARDCODED
```

**FIX**: Remove default parameter from ctx.get_threshold() call

---

## 3 NAMING SYSTEMS CONFLICT

Same archetype called 3 different things:

```
ARCHETYPE H (Trap Within Trend):

logic.py code:          'H'
logic_v2_adapter.py:    'momentum_continuation' OR 'trap_within_trend'
registry.py:            'trap_within_trend'  
config['thresholds']:   'H'
config top-level:       'trap_within_trend' (new)

RESULT: Lookup failures in ALL layers!
```

**FIX**: Use ONLY registry canonical slug everywhere

---

## CONFIG PARAMETER MISMATCH

What code expects vs what config has:

```
CODE EXPECTS (logic_v2_adapter.py):    CONFIG HAS (baseline_btc_bull_pf20.json):
'fusion_threshold'                     'fusion'
'pti_score_threshold'                  'pti'
'disp_atr_multiplier'                  'disp_atr'
'boms_strength_min'                    'boms_strength'
'wyckoff_min'                          'wyckoff'

RESULT: Threshold NOT FOUND → hardcoded default used
```

**FIX**: Standardize all parameter names to LONG form

---

## IMMEDIATE FIXES (Order of priority)

1. **CRITICAL**: Fix logic_v2_adapter._check_K slug name
   - Line 578: Currently reads 'trap_within_trend' 
   - Should read 'wick_trap_moneytaur' to match registry!

2. **CRITICAL**: Verify ThresholdPolicy context population
   - Test that config['archetypes']['trap_within_trend'] reaches context
   - Add logging output to _build_base_map()

3. **HIGH**: Remove hardcoded defaults (lines 385-623)
   - Make get_threshold() calls without default parameter
   - Fail loudly if threshold not found

4. **HIGH**: Update all configs to canonical slug format
   - Replace all letter codes with slugs
   - Use LONG parameter names (fusion_threshold not fusion)

5. **MEDIUM**: Delete logic.py duplicate code
   - Too risky until (1-3) complete
   - After verified, can consolidate

6. **MEDIUM**: Update registry mappings
   - Fix archetype 'A' (spring vs wyckoff_spring_utad)
   - Fix archetype 'H'/'K' dual identity

---

## WHERE TO READ PARAMETERS

### CORRECT PATTERN (Only 2 archetypes do this):
```python
# logic.py lines 802-806 (_check_H):
from engine.archetypes.param_accessor import get_param
quality_th = get_param(self, 'trap_within_trend', 'quality_threshold', 0.55)
```

**WHY**: Migration-safe fallback chain. EXTEND TO ALL ARCHETYPES.

### WRONG PATTERN (All other archetypes):
```python
# logic_v2_adapter.py line 385 (_check_A):
fusion_th = ctx.get_threshold('spring', 'fusion_threshold', 0.33)
           └─ WRONG SLUG! Should match config key
           └─ Hardcoded default blocks variance!
           └─ Config parameter names don't match!
```

---

## PARAMETER FLOW CHECKLIST

For each archetype, verify this chain works:

```
[ ] Config has: config['archetypes'][SLUG]
    where SLUG = registry canonical name
    with param names = LONG descriptive (fusion_threshold not fusion)

[ ] ThresholdPolicy._build_base_map():
    Reads config['archetypes'][SLUG]
    Populates context.thresholds[SLUG]

[ ] RuntimeContext.get_threshold(SLUG, param_name):
    Returns context.thresholds[SLUG][param_name]

[ ] Archetype logic._check_X(ctx):
    Calls ctx.get_threshold(SLUG, param_name)
    WITHOUT hardcoded default (or with minimal fallback)

[ ] Optimizer can vary parameters in config['archetypes'][SLUG]
    And see results change in backtest
```

---

## EXPECTED OUTCOME (After fixes)

**Before:**
- Optimizer trial #1: quality_threshold=0.50 → result_pnl=1000
- Optimizer trial #2: quality_threshold=0.75 → result_pnl=1000 (SAME!)
- Conclusion: Parameter has zero variance

**After:**
- Optimizer trial #1: quality_threshold=0.50 → result_pnl=850 (different!)
- Optimizer trial #2: quality_threshold=0.75 → result_pnl=1200 (different!)
- Conclusion: Parameter affects results, optimization works!

---

## DOCUMENT REFERENCES

**Full comprehensive audit** (12 pages):  
→ `COMPREHENSIVE_ARCHETYPE_AUDIT.md`

**Executive summary** (8 pages):  
→ `AUDIT_EXECUTIVE_SUMMARY.md`

**This quick reference**:  
→ `AUDIT_QUICK_REFERENCE.md`

---

## KEY STATS

- **Lines of code duplicated**: 961 (logic.py) + 636 (logic_v2_adapter.py) = 1,597
- **Hardcoded threshold values**: 70+
- **Parameter name mismatches**: 20+
- **Naming conflicts**: 11 archetypes × 3 naming systems
- **Critical issues blocking optimization**: 5
- **Estimated fix time**: 3-7 days

