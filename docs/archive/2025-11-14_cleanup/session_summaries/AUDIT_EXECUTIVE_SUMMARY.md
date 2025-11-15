# Archetype Parameter System Audit - Executive Summary

**Date:** 2025-11-08  
**Finding:** CRITICAL BLOCKING ISSUES - System is broken and blocking optimization  
**Full Report:** See `COMPREHENSIVE_ARCHETYPE_AUDIT.md`

---

## THE CORE PROBLEM

The archetype parameter system is in a **broken state** with two mutually incompatible code paths:

1. **Legacy Path (logic.py)**: Reads parameters from letter codes in config['archetypes']['thresholds'][LETTER]
2. **Modern Path (logic_v2_adapter.py)**: Reads from config['archetypes'][SLUG] via RuntimeContext

**Result**: Optimizer writes to modern path, but code reads from legacy path → ZERO VARIANCE.

---

## TOP 5 CRITICAL ISSUES

### 1. ZERO-VARIANCE BUG (Blocks All Optimization)

**Symptom**: Optimizer trials all have identical results despite different parameters

**Root Cause**: 
- Optimizer writes: `config['archetypes']['trap_within_trend']['quality_threshold'] = 0.55`
- logic.py expects: `config['archetypes']['thresholds']['H']['quality']` (WRONG LOCATION!)
- Result: Parameter never reaches archetype logic, always uses hardcoded default

**Evidence**:
- optuna_trap_v2.py Line 118: Writes to `config['archetypes']['trap_within_trend']`
- logic.py Line 46: Reads from `config['thresholds']` (WRONG!)
- logic.py Line 50-82: Uses `self.thresh_H` dict which is empty when optimizer writes to canonical location

**Impact**: **BLOCKS OPTIMIZATION WORK** - trials cannot vary parameters

---

### 2. DUPLICATE ARCHETYPE IMPLEMENTATIONS

**Two completely different implementations exist:**

| Aspect | logic.py | logic_v2_adapter.py |
|--------|----------|-------------------|
| Location | /engine/archetypes/logic.py | /engine/archetypes/logic_v2_adapter.py |
| API | (row, prev_row, df, index) | RuntimeContext |
| Config Reading | self.thresh_X dicts | context.get_threshold() |
| Dispatch | Priority order (early returns) | All archetypes evaluation |
| Line Count | 961 lines | 636 lines |
| Parameter Names | Short ('fusion', 'adx') | Long ('fusion_threshold', 'adx_threshold') |

**Problem**: Unclear which is actually used. Code paths conflict.

---

### 3. NAMING CHAOS - 3 DIFFERENT SYSTEMS

Same archetype has 3 different names:

Example (Trap Within Trend):
- **Letter code in logic.py**: 'H' 
- **Slug in logic_v2_adapter.py**: 'trap_within_trend'
- **Registry canonical**: 'trap_within_trend'
- **Config thresholds**: 'H' (legacy) or 'trap_within_trend' (new)

This causes lookup failures across ALL layers.

---

### 4. HARDCODED VALUES EVERYWHERE (70+ instances)

Every archetype check has hardcoded defaults that BYPASS config:

```python
# logic_v2_adapter.py lines 385-623
fusion_th = ctx.get_threshold('spring', 'fusion_threshold', 0.33)  # ← hardcoded!
pti_score_th = ctx.get_threshold('spring', 'pti_score_threshold', 0.40)  # ← hardcoded!
# ... 70+ more hardcoded values
```

**Impact**: Even if context is populated, defaults are used if threshold not found

---

### 5. PARAMETER NAME MISMATCHES (20+ instances)

Code reads one name, config has another:

| Code Reads | Config Has | Result |
|------------|-----------|--------|
| 'fusion_threshold' | 'fusion' | NOT FOUND → hardcoded default |
| 'adx_threshold' | 'adx' | NOT FOUND → hardcoded default |
| 'boms_strength_min' | 'boms_strength' | NOT FOUND → hardcoded default |
| 'pti_score_threshold' | 'pti' | NOT FOUND → hardcoded default |

---

## IMMEDIATE ACTION ITEMS

### Phase 1: UNBLOCK OPTIMIZATION (1-2 days)

1. **Delete logic.py** OR convert to wrapper around logic_v2_adapter.py
   - Keep only one implementation
   - Choose: logic_v2_adapter.py is the future

2. **Fix config location bug in ThresholdPolicy**
   - Ensure config['archetypes'][SLUG] is read (already partially fixed in PR#6A)
   - Validate context thresholds dict is populated before use

3. **Remove hardcoded defaults** from all _check_X methods
   - Replace hardcoded defaults with mandatory config values
   - Fail loudly if threshold not found (vs silently using 0.33)

### Phase 2: STANDARDIZE NAMING (1-2 days)

1. **Canonical slug everywhere**: Use registry values
   - No more letter codes in new code
   - Keep letter→slug mapping ONLY for legacy config compatibility

2. **Standard parameter names**: Use LONG descriptive names
   - 'fusion_threshold' not 'fusion'
   - 'adx_threshold' not 'adx'
   - Update all configs to use new names

3. **Update registry**:
   - Fix archetype 'A' mapping (currently 'spring' vs 'wyckoff_spring_utad')
   - Fix archetype 'H'/'K' dual identity (momentum_continuation vs trap_within_trend)
   - Resolve all aliases

### Phase 3: VALIDATION (1 day)

1. **Add schema validation** at config load time
   - Use ARCHETYPE_SCHEMAS from param_accessor.py
   - Reject configs with missing required parameters

2. **Add parameter tracing** 
   - Log which parameters are actually being read
   - Prove optimizer-written values reach archetype logic

3. **Add test coverage**
   - Wire test: Optimizer parameter → config → context → archetype logic
   - Verify each archetype actually uses parameters

---

## FILES THAT NEED CHANGES

### MUST DELETE/REFACTOR
- [ ] `/engine/archetypes/logic.py` - Duplicate implementation
  - **Lines to preserve**: registry import, param_accessor functions
  - **Action**: Delete entire _check_X logic, create wrapper calling logic_v2_adapter

### MUST FIX
- [ ] `/engine/archetypes/logic_v2_adapter.py` - Remove hardcoded defaults
  - **Lines 385-623**: Remove all hardcoded default values
  - **Action**: Make context.get_threshold() call mandatory, fail if not found

- [ ] `/engine/archetypes/threshold_policy.py` - Validate parameter flow
  - **Lines 154-177**: Already correct (PR#6A fix), but needs logging
  - **Action**: Add debug output showing what gets populated in context

- [ ] `/engine/runtime/context.py` - Add validation
  - **Lines 41-67**: Add check that thresholds dict is populated
  - **Action**: Log warning if threshold dict is empty

- [ ] `/engine/archetypes/param_accessor.py` - Already correct!
  - This is the RIGHT pattern (used only by _check_H, _check_K)
  - **Action**: Extend to ALL archetypes, delete get_param fallback chain

### CONFIG FILES
- [ ] All config files need updating to canonical slug format
  - Replace all letter codes with slugs
  - Update parameter names to long form

---

## EVIDENCE FOR ZERO-VARIANCE BUG

### Optimizer writes here (optuna_trap_v2.py:118):
```python
config['archetypes']['trap_within_trend'] = {
    'fusion_threshold': 0.35,
    'liquidity_threshold': 0.30,
    'quality_threshold': 0.55,
    'adx_threshold': 25.0
}
```

### logic.py reads here (Line 46):
```python
thresholds = config.get('thresholds', {})  # ← WRONG! Should be config['archetypes']['thresholds']
self.thresh_H = thresholds.get('H', {})    # ← Empty dict!
```

### Result:
```
get_threshold('trap_within_trend', 'quality_threshold', default=0.55)
  → Not in context
  → Uses default 0.55
  → SAME VALUE FOR ALL TRIALS!
```

---

## PARAMETER FLOW (CURRENT - BROKEN)

```
Optimizer Writes
    ↓
config['archetypes']['trap_within_trend'] = {quality_threshold: 0.55, ...}
    ↓
ThresholdPolicy._build_base_map() 
    ↓
context.thresholds['trap_within_trend'] = {quality_threshold: 0.55, ...} ✓
    ↓
logic_v2_adapter._check_H reads:
    ctx.get_threshold('momentum_continuation', 'quality_threshold', 0.55)
                       ↑ WRONG SLUG!
    ↓
Threshold NOT FOUND
    ↓
Uses hardcoded default 0.55
    ↓
ZERO VARIANCE!
```

---

## PARAMETER FLOW (FIXED)

```
Optimizer Writes
    ↓
config['archetypes']['trap_within_trend'] = {quality_threshold: 0.55, ...}
    ↓
ThresholdPolicy._build_base_map() 
    ↓
context.thresholds['trap_within_trend'] = {quality_threshold: 0.55, ...} ✓
    ↓
logic_v2_adapter._check_H reads:
    ctx.get_threshold('trap_within_trend', 'quality_threshold')
                       ↑ CORRECT SLUG!
    ↓
Threshold FOUND: 0.55
    ↓
Parameter flows to archetype logic
    ↓
VARIANCE RESTORED!
```

---

## KEY FILES FOR REFERENCE

**Full details in**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/COMPREHENSIVE_ARCHETYPE_AUDIT.md`

**Config reading paths**: See Section 5 of audit
**Hardcoded values**: See Section 6 of audit (70+ values listed)
**Naming catalog**: See Section 7 of audit (20+ mismatches)
**All file:line references**: See Section 9 of audit

---

## ESTIMATED EFFORT

- **Phase 1 (Unblock)**: 4-8 hours (1-2 days)
- **Phase 2 (Standardize)**: 8-16 hours (2-3 days)  
- **Phase 3 (Validate)**: 4-8 hours (1-2 days)

**Total: 3-7 days of focused work**

---

## SUCCESS CRITERIA

After fixes:

1. **Optimizer variance**: Different trial parameters produce different results
2. **Single code path**: Only logic_v2_adapter.py used for archetype detection
3. **Config clarity**: All parameter names standardized (no short/long mismatch)
4. **Naming consistency**: One name per archetype (canonical slug only)
5. **Zero hardcoding**: No magic numbers in threshold logic (all from config)

