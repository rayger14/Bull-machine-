# PR-A: Parity Test Failure - Root Cause Analysis

**Date**: 2025-11-02
**Status**: ROOT CAUSE IDENTIFIED ✅

## Test Results Summary

| Config | Path | Trades | Notes |
|--------|------|--------|-------|
| Legacy (baseline_btc_bull_pf20.json) | check_archetype() → detect() with empty thresholds | 64 | Target baseline (PF 3.13) |
| Adaptive (first test) | detect() with ThresholdPolicy + regime blending | 84 | Over-trading (+31%) |
| Adaptive (full lockdown) | detect() with ThresholdPolicy + all adaptive features disabled | 19 | Under-trading (-70%) |

## Root Cause: Threshold Resolution Discrepancy

### The Two Code Paths

**Path 1: Legacy (baseline_btc_bull_pf20.json)**

bin/backtest_knowledge_v2.py:482-512:
```python
else:
    # Fallback to old API if ThresholdPolicy not available
    archetype_name, fusion_score, liquidity_score = self.archetype_logic.check_archetype(
        row=row_with_runtime,
        prev_row=prev_row,
        df=self.df,
        index=current_idx
    )
```

engine/archetypes/logic_v2_adapter.py:311-333:
```python
def check_archetype(...):
    """DEPRECATED: Backward compatibility wrapper for old API."""
    # Create minimal context without regime data
    ctx = RuntimeContext(
        ts=row.name if hasattr(row, 'name') else index,
        row=row,
        regime_probs={'neutral': 1.0},
        regime_label='neutral',
        adapted_params={},
        thresholds={}  # ⚠️ EMPTY THRESHOLDS - uses defaults
    )
    return self.detect(ctx)
```

**Path 2: Adaptive (btc_v8_adaptive.json)**

bin/backtest_knowledge_v2.py:482-502:
```python
if self.threshold_policy and 'adapted_params' in context and context['adapted_params']:
    # Resolve thresholds using ThresholdPolicy
    thresholds = self.threshold_policy.resolve(regime_probs, regime_label)

    runtime_ctx = RuntimeContext(
        ts=row.name,
        row=row_with_runtime,
        regime_probs=regime_probs,
        regime_label=regime_label,
        adapted_params=adapted_params,
        thresholds=thresholds  # ✅ Resolved from ThresholdPolicy
    )

    archetype_name, fusion_score, liquidity_score = self.archetype_logic.detect(runtime_ctx)
```

### How Thresholds Are Retrieved

engine/runtime/context.py:35-47:
```python
def get_threshold(self, archetype: str, param: str, default: float = 0.0) -> float:
    """Safely get threshold for archetype parameter."""
    return self.thresholds.get(archetype, {}).get(param, default)
```

**Legacy path behavior**:
- `thresholds={}` (empty dict)
- `ctx.get_threshold('order_block_retest', 'fusion', 0.374)` returns `0.374` (hardcoded default)
- **Config values are NEVER read!**

**Adaptive path behavior**:
- `thresholds=<dict from ThresholdPolicy.resolve()>`
- `ctx.get_threshold('order_block_retest', 'fusion', 0.374)` returns resolved value (e.g., 0.359 from config)

## The Mystery: Where Does Legacy Get Config Thresholds?

**CRITICAL QUESTION**: The legacy config has `archetypes.thresholds.B.fusion = 0.35912...`, but the legacy path uses `thresholds={}`. How does it read config values?

**HYPOTHESIS**: There may be a THIRD code path or the legacy config actually uses a different version of ArchetypeLogic that reads thresholds directly from config.

## Verification Steps Needed

1. **Check which ArchetypeLogic class is instantiated**:
   - Does legacy use `engine/archetypes/logic.py`?
   - Does adaptive use `engine/archetypes/logic_v2_adapter.py`?

2. **Trace threshold loading**:
   - Where does `archetypes.thresholds` get read in legacy path?
   - Is there a separate threshold loading mechanism?

3. **Compare hardcoded defaults vs config values**:
   ```python
   # Archetype B (order_block_retest)
   # Hardcoded default: 0.374
   # Config value (legacy): 0.35912732076623655
   # Config value (adaptive): 0.35912732076623655

   # Delta: -0.015 (config is MORE LENIENT)
   ```

## BREAKTHROUGH: Root Cause Identified (2025-11-02 continued)

### Debug Logging Revealed the Issue

1. **Legacy config (baseline_btc_bull_pf20.json)**:
   - No `gates_regime_profiles` → ThresholdPolicy NOT created
   - Takes legacy path → `check_archetype()` → `RuntimeContext(thresholds={})`
   - Uses hardcoded defaults: `get_threshold('order_block_retest', 'fusion', 0.374) → 0.374`

2. **Adaptive locked config (btc_v8_adaptive_locked_parity.json)**:
   - Has `gates_regime_profiles` → ThresholdPolicy created in locked mode
   - **PROBLEM**: Archetype detection returns `tier1_market` (fallback name!)
   - **NO archetypes matching** → `get_threshold()` never called → No debug logs!

### The Real Issue: Code Path Divergence

The branching condition at bin/backtest_knowledge_v2.py:482 required **BOTH** `threshold_policy` AND `adapted_params`. When locked configs don't have `adapted_params`, they fell back to legacy path.

**FIX APPLIED**: Changed condition from:
```python
if self.threshold_policy and 'adapted_params' in context and context['adapted_params']:
```
To:
```python
if self.threshold_policy:  # Always use RuntimeContext if ThresholdPolicy exists
```

### New Mystery: Why No Archetype Matches?

After the fix, adaptive locked config now uses RuntimeContext path, BUT all trades show `archetype=tier1_market` (fallback), meaning archetype detection is failing completely. This needs investigation.

## Next Actions

1. ✅ Fix branching condition to always use RuntimeContext when ThresholdPolicy exists
2. 🔄 Investigate why archetype detection returns no matches for locked config
3. Compare archetype detection between legacy and adaptive locked paths
4. Re-run parity test after fixing archetype detection

## Key Files

- bin/backtest_knowledge_v2.py:482-512 (branching logic)
- engine/archetypes/logic_v2_adapter.py (RuntimeContext-based detection)
- engine/archetypes/logic.py (possibly old logic?)
- engine/runtime/context.py:35-47 (threshold getter)
- engine/archetypes/threshold_policy.py (threshold resolution)

## Parity Test Command

```bash
python3 tests/test_parity_legacy_vs_adaptive.py \
  --asset BTC \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --legacy-config configs/baseline_btc_bull_pf20.json \
  --adaptive-config /tmp/btc_v8_adaptive_FULLY_locked.json
```

---

**Status**: Investigation ongoing. Need to find threshold loading mechanism in legacy path.
