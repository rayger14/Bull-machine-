# Soft Gating Integration Report

## Summary

Successfully integrated regime-conditioned soft gating into `engine/archetypes/logic_v2_adapter.py` to replace binary on/off switches with continuous capital allocation weights.

## Completed Work

### 1. Import and Initialization
- **Added import**: `from engine.portfolio.regime_allocator import RegimeWeightAllocator`
- **Added initialization** in `__init__` method (lines 256-296):
  - Loads edge table and config
  - Creates `self.regime_allocator` instance
  - Handles graceful fallback if edge data not available
  - Logs initialization status

### 2. Helper Method Created
- **Method**: `_apply_soft_gating()` (lines 474-540)
- **Purpose**: Centralized soft gating logic for all archetypes
- **Inputs**:
  - `archetype`: Name matching ARCHETYPE_REGIMES keys
  - `raw_score`: Score before soft gating
  - `regime_label`: Current regime (risk_on, neutral, risk_off, crisis)
  - `min_threshold`: Minimum score to pass
- **Returns**: `(passed: bool, gated_score: float, metadata: dict)`
- **Features**:
  - Gets regime weight from RegimeWeightAllocator
  - Applies multiplicative gating: `gated_score = raw_score * regime_weight`
  - Checks if gated score passes threshold
  - Logs for observability
  - Returns detailed metadata

### 3. Applied to Archetypes
- **_check_B** (order_block_retest): ✓ Complete
- **_check_C** (wick_trap): ✓ Complete

## Archetype Name Mapping

Based on ARCHETYPE_REGIMES and code analysis:

| Check Method | Archetype Name | Type | Regimes Allowed |
|--------------|----------------|------|-----------------|
| _check_A | spring | Bull | risk_on, neutral |
| _check_B | order_block_retest | Bull | neutral |
| _check_C | wick_trap | Bull | risk_on, neutral |
| _check_D | failed_continuation | Bull | risk_on, neutral |
| _check_E | volume_exhaustion | Bull | risk_on, neutral |
| _check_F | exhaustion_reversal | Bull | risk_on, neutral |
| _check_G | liquidity_sweep | Bull | risk_on, neutral |
| _check_H | momentum_continuation | Bull | risk_on, neutral |
| _check_K | trap_within_trend | Bull | risk_on, neutral |
| _check_L | retest_cluster | Bull | risk_on, neutral |
| _check_M | confluence_breakout | Bull | risk_on, neutral |
| _check_S1 | liquidity_vacuum | Bear | risk_off, crisis |
| _check_S2 | failed_rally | Bear | risk_off, neutral |
| _check_S3 | whipsaw | Bear | risk_off, crisis |
| _check_S4 | funding_divergence | Bear | risk_off, neutral |
| _check_S5 | long_squeeze | Contrarian Short | risk_on, neutral |
| _check_S6 | alt_rotation_down | Bear | risk_off, crisis |
| _check_S7 | curve_inversion | Bear | risk_off, crisis |
| _check_S8 | volume_fade_chop | Neutral | neutral |

## Integration Pattern

For each remaining `_check_*()` method, add this code block **AFTER** `_apply_regime_soft_penalty()` and **BEFORE** the threshold gate check:

```python
# ============================================================================
# SOFT GATING: Apply regime-conditioned weight (NEW!)
# ============================================================================
regime_label = self.g(context.row, "regime_label", "neutral")
raw_score_before_gating = score
passed_gating, score, gating_metadata = self._apply_soft_gating(
    archetype='<ARCHETYPE_NAME>',  # See mapping table above
    raw_score=score,
    regime_label=regime_label,
    min_threshold=fusion_th  # Or appropriate threshold variable
)

if not passed_gating:
    return (
        False,
        score,
        {
            **gating_metadata,
            "score_before_gating": raw_score_before_gating,
            "base_score": base_score,  # If available
            # Include any other existing metadata
        },
    )
```

And in the successful return statement, add gating metadata:

```python
return (
    True,
    score,
    {
        # ... existing metadata ...
        **gating_metadata,  # Add this line
    },
    "LONG",  # or "SHORT"
)
```

## Remaining Work

Apply soft gating to these methods (in priority order):

### High Priority (Used in Production)
1. **_check_S1** (liquidity_vacuum) - Bear archetype, crisis regime
2. **_check_S4** (funding_divergence) - Bear archetype
3. **_check_K** (trap_within_trend) - Bull archetype
4. **_check_H** (momentum_continuation) - Bull archetype
5. **_check_S5** (long_squeeze) - Contrarian short

### Medium Priority (Enabled but Less Active)
6. **_check_A** (spring) - Wyckoff spring
7. **_check_D** (failed_continuation)
8. **_check_E** (volume_exhaustion)
9. **_check_F** (exhaustion_reversal)
10. **_check_G** (liquidity_sweep)
11. **_check_L** (retest_cluster)
12. **_check_M** (confluence_breakout)
13. **_check_S2** (failed_rally)
14. **_check_S8** (volume_fade_chop)

### Low Priority (Disabled in Config)
15. **_check_S3** (whipsaw) - Currently disabled
16. **_check_S6** (alt_rotation_down) - Currently disabled
17. **_check_S7** (curve_inversion) - Currently disabled

## Example: _check_S1 (liquidity_vacuum)

This is a critical archetype that currently has hard vetoes in RISK_ON regime. Here's how to apply soft gating:

1. Find the line with `_apply_regime_soft_penalty()`
2. Add soft gating block immediately after
3. Update successful return to include `**gating_metadata`

```python
# After _apply_regime_soft_penalty()
regime_label = self.g(context.row, "regime_label", "neutral")
raw_score_before_gating = score
passed_gating, score, gating_metadata = self._apply_soft_gating(
    archetype='liquidity_vacuum',
    raw_score=score,
    regime_label=regime_label,
    min_threshold=fusion_th
)

if not passed_gating:
    return (
        False,
        score,
        {
            **gating_metadata,
            "score_before_gating": raw_score_before_gating,
        },
    )
```

## Configuration

To enable soft gating, add to archetype config JSON:

```json
{
  "soft_gating": {
    "enabled": true,
    "edge_table_path": "data/regime_edge_table.csv",
    "config_path": "configs/regime_allocator_config.json"
  }
}
```

## Validation

After integration, validate with:

1. **Unit tests**: Verify soft gating applies weights correctly
2. **Integration tests**: Check that scores are modified as expected
3. **Backtest**: Compare results with/without soft gating enabled
4. **Logging**: Monitor soft gating decisions in logs

## Files Modified

1. `/Users/raymondghandchi/Bull-machine-/Bull-machine-/engine/archetypes/logic_v2_adapter.py`
   - Added import (line 19)
   - Added initialization (lines 256-296)
   - Added `_apply_soft_gating()` method (lines 474-540)
   - Applied to `_check_B()` (lines 2161-2185)
   - Applied to `_check_C()` (lines 2318-2340)

## Next Steps

1. Apply soft gating to remaining high-priority methods (S1, S4, K, H, S5)
2. Create regime edge table CSV with empirical weights
3. Create regime allocator config JSON
4. Enable soft gating in archetype configs
5. Run smoke tests to validate integration
6. Compare backtest results before/after soft gating

## Benefits

- **Graceful degradation**: Poor-performing archetypes get reduced allocation, not disabled
- **Continuous adaptation**: Weights can be updated based on ongoing performance
- **Quantitative approach**: Allocation based on empirical edge metrics, not manual decisions
- **Preserves architecture**: All archetypes remain active, portfolio manager decides allocation
- **Observable**: Detailed logging shows exactly how soft gating affects each signal
