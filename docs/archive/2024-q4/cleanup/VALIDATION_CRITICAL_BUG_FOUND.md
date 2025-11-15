# CRITICAL BUG FOUND - Parameter Wiring Mismatch

**Date**: 2025-11-08
**Status**: 🚨 BLOCKING - Invalidates trap_within_trend optimization results

## Summary

The validation revealed a **critical parameter wiring bug** in both the validation script AND the 200-trial trap_within_trend optimizer. Parameters are being written to the WRONG config location, causing zero archetype trades.

## Evidence

### Validation Results
- **Expected**: ~104 trap_within_trend trades (from baseline diagnostic)
- **Actual**: 0 trap_within_trend trades in both baseline AND optimized runs
- **Root Cause**: Parameters not being read by archetype logic

### Config Structure Mismatch

**Baseline config structure** (baseline_btc_bull_pf20.json:195-196):
```json
"archetypes": {
  "thresholds": {
    "K": {
      "fusion": 0.435
    }
  }
}
```

**What the optimizer writes** (optuna_trap_v2.py:104-113):
```python
config['archetypes']['trap_within_trend'] = {
    'quality_threshold': 0.55,
    'fusion_threshold': 0.35,  # ❌ Should be 'fusion'
    ...
}
```

**What Context.get_threshold() reads**:
```python
value = self.thresholds.get('trap_within_trend', {}).get('fusion_threshold', default)
# Looks for: thresholds['trap_within_trend']['fusion_threshold']
# But config has: thresholds['K']['fusion']
# Result: Always uses default value!
```

### Log Evidence

```
[PR-A THRESHOLD DEBUG] get_threshold('trap_within_trend', 'fusion_threshold', default=0.36)
  → 0.36 | thresholds=1 archetypes | arch_data=NOT_FOUND
```

- **0.36 returned**: Default value (not from config)
- **arch_data=NOT_FOUND**: trap_within_trend key doesn't exist in thresholds dict
- **Result**: Archetype uses hardcoded defaults, not optimized parameters!

### Archetype Detection Logic

From engine/archetypes/logic_v2_adapter.py:578:
```python
fusion_th = ctx.get_threshold('trap_within_trend', 'fusion_threshold', 0.36)
```

Archetype asks for `('trap_within_trend', 'fusion_threshold')` but config only has `('K', 'fusion')`.

## Impact Assessment

###  200-Trial Optimization is INVALID
The 200-trial trap_within_trend optimization (results/optuna_trap_200trial_PRODUCTION/):
- ✅ Generated parameters: quality_threshold=0.14, fusion_threshold=0.22, etc.
- ❌ Parameters were NEVER USED during optimization
- ❌ All trials used hardcoded defaults
- ❌ best_value=0.8608 reflects DEFAULT params, not optimized ones
- **Conclusion**: Optimization results are meaningless

### Wick Trap 10-Trial Probe is QUESTIONABLE
The wick trap variance probe (results/optuna_wick_trap_probe/):
- Passed with score_std = 0.895
- Uses same `_create_fixed_sizing_config()` method
- **Needs Investigation**: May have same bug or different config structure

## Root Cause Analysis

### Letter Code vs Canonical Slug Mismatch

**ThresholdPolicy Letter Mapping** (threshold_policy.py:42):
```python
'trap_within_trend': 'K'
```

**Expected Flow**:
1. Optimizer writes to canonical location: `config['archetypes']['trap_within_trend']`
2. ThresholdPolicy converts `'trap_within_trend'` → `'K'` when building thresholds dict
3. Context.get_threshold('trap_within_trend') reads converted value

**Actual Flow**:
1. Optimizer writes to `config['archetypes']['trap_within_trend']` ✓
2. ThresholdPolicy ONLY reads from `config['archetypes']['thresholds']['K']` ❌
3. Context finds nothing at 'trap_within_trend', uses default ❌

### Parameter Name Mismatch

**Baseline uses**: `"fusion": 0.435`
**Optimizer uses**: `"fusion_threshold": 0.22`
**Archetype logic expects**: `"fusion_threshold"` (per engine/archetypes/logic_v2_adapter.py:578)

**Problem**: ThresholdPolicy doesn't translate `fusion` → `fusion_threshold`

## Fix Required

### Option A: Write to Legacy Location (Quick Fix)
```python
# In _create_fixed_sizing_config():
config['archetypes']['thresholds']['K'] = {
    'fusion': trap_params.get('fusion_threshold', 0.36),  # Translate names
    'adx': trap_params.get('adx_threshold', 25.0),
    'liq': trap_params.get('liquidity_threshold', 0.30),
}
```

**Pros**: Works with existing baseline configs
**Cons**: Doesn't align with PR#6A canonical slug migration

### Option B: Fix ThresholdPolicy to Read Canonical Location
Update ThresholdPolicy to check BOTH locations:
1. First try: `config['archetypes']['trap_within_trend']['fusion_threshold']` (new)
2. Fallback: `config['archetypes']['thresholds']['K']['fusion']` (old)

**Pros**: Enables PR#6A migration
**Cons**: More complex, requires ThresholdPolicy refactor

### Option C: Hybrid Approach (Recommended)
1. Fix optimizer to write to BOTH locations for backward compat
2. Update validation to compare using correct baseline structure
3. Plan ThresholdPolicy refactor for PR#6A Phase 6

## Immediate Action Items

1. ❌ **STOP using trap_within_trend optimization results** - they're invalid
2. 🔧 **Fix validation script** to use correct config structure
3. 🔍 **Investigate wick_trap variance probe** - check if it has same bug
4. 🔁 **Re-run 200-trial trap optimization** with fixed parameter wiring
5. 📋 **Document PR#6A parameter reading requirements** to prevent future bugs

## Test Plan for Fix Verification

### Wire Test
```python
# 1. Write test config
config = {'archetypes': {'thresholds': {'K': {'fusion': 0.99}}}}

# 2. Run single-bar backtest
results = backtest.run()

# 3. Check logs for threshold read
# Expected: "[PR-A THRESHOLD DEBUG] ... → 0.99" (not 0.36)

# 4. Verify archetype fires
# Expected: >0 trap_within_trend trades (if setup exists)
```

### Variance Test
```python
# Run 5 trials with EXTREME parameter variance
trial_1: fusion=0.10 → expect different results
trial_2: fusion=0.90 → expect different results
# If results are identical → bug still exists
```

## Related Files

- `bin/optuna_trap_v2.py:94-129` - Buggy config creation
- `bin/validate_trap_optimization.py:83-98` - Buggy baseline config
- `engine/archetypes/threshold_policy.py:42` - Letter→slug mapping
- `engine/runtime/context.py:41-54` - get_threshold() implementation
- `engine/archetypes/logic_v2_adapter.py:578-580` - Archetype param reads
- `configs/baseline_btc_bull_pf20.json:195-196` - Legacy config structure

## Lessons Learned

1. **Always verify parameter wiring** before long optimizations
2. **Run variance probes** with extreme values to catch zero-variance bugs early
3. **Test with actual baseline configs** not synthetic ones
4. **Check logs** for "THRESHOLD DEBUG" messages to verify reads
5. **Config migration requires end-to-end testing** of the full read/write path
