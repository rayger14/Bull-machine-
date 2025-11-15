# PR-A: Parity Testing Infrastructure

**Date**: 2025-11-01
**Branch**: pr6a-archetype-expansion
**Priority**: CRITICAL - Must pass before any regime work

## Executive Summary

Successfully implemented parity testing infrastructure to ensure the RuntimeContext adaptive path can reproduce legacy static results when locked. This is the foundation for proving code path equivalence before proceeding with GMM retraining and regime-specific optimization.

## Motivation

From PF20_RECOVERY_STATUS.md analysis:

```
Static PF-20:  64 trades, PF 3.13, WR 64.1%  (legacy check_archetype path)
Adaptive:     183 trades, PF 1.62, WR 51.9%  (RuntimeContext detect path)
```

**Root Cause**: Two different code paths with no proof of equivalence when locked to same parameters.

**Solution**: Build parity testing guardrails to force equivalence, then prove it with tests.

## Implementation Complete

### ✅ Component 1: Locked Mode in ThresholdPolicy

**File**: `engine/archetypes/threshold_policy.py`

**Changes**:
- Added `locked_regime` parameter to `__init__`
- Modified `resolve()` to bypass blending when locked
- Added `_resolve_locked()` helper method

**Modes**:
```python
# Mode 1: Static (return base thresholds only, no regime profiles)
ThresholdPolicy(..., locked_regime='static')

# Mode 2: Locked to specific regime (force 100% weight to one regime)
ThresholdPolicy(..., locked_regime='risk_on')
```

**Usage in Config**:
```json
{
  "locked_regime": "static",
  "description": "Locked to static mode for parity testing"
}
```

**Integration**: `bin/backtest_knowledge_v2.py:217` reads `locked_regime` from config and passes to ThresholdPolicy.

### ✅ Component 2: Parity Test Script

**File**: `tests/test_parity_legacy_vs_adaptive.py`

**Workflow**:
1. Create locked version of adaptive config (`"locked_regime": "static"`)
2. Run legacy static config → Export trades
3. Run locked adaptive config → Export trades
4. Compare trade lists:
   - Trade counts
   - Entry timestamps
   - Archetype distributions

**Expected Outcome**:
- PASS: Identical trades (proves code paths equivalent when locked)
- FAIL: Divergence detected (must fix adaptive path)

**Usage**:
```bash
python3 tests/test_parity_legacy_vs_adaptive.py \
  --asset BTC \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --legacy-config configs/baseline_btc_bull_pf20.json \
  --adaptive-config configs/btc_v8_adaptive.json
```

### ✅ Component 3: Legacy Threshold Copy Utility

**File**: `bin/copy_legacy_thresholds.py`

**Purpose**: Extract thresholds from legacy static config and convert to `archetype_overrides` delta format for merging into adaptive config.

**Workflow**:
1. Read legacy config (e.g., `baseline_btc_bull_pf20.json`)
2. Read base adaptive config
3. Compute deltas: `override_delta = legacy_threshold - base_threshold`
4. Generate `archetype_overrides` format:
   ```json
   {
     "order_block_retest": {
       "static": {"fusion": -0.02}
     }
   }
   ```

**Usage**:
```bash
# Extract overrides to standalone file
python3 bin/copy_legacy_thresholds.py \
  --legacy-config configs/baseline_btc_bull_pf20.json \
  --base-config configs/btc_v8_adaptive.json \
  --output configs/archetype_overrides_pf20.json

# Or merge directly into adaptive config
python3 bin/copy_legacy_thresholds.py \
  --legacy-config configs/baseline_btc_bull_pf20.json \
  --base-config configs/btc_v8_adaptive.json \
  --merge
```

## Architecture Benefits

### Before PR-A
- Legacy and adaptive paths diverge silently
- No way to prove equivalence
- Impossible to know if adaptive bugs exist
- Can't trust regime blending built on broken foundation

### After PR-A
- Locked mode forces parity for testing
- Golden test suite to catch regressions
- Proof of correctness before regime work
- Confidence in adaptive path when unlocked

## Testing Protocol

### Phase 1: Prove Parity (Baseline)
```bash
# Test 1: Static baseline (PF-20 winner)
python3 tests/test_parity_legacy_vs_adaptive.py \
  --legacy-config configs/baseline_btc_bull_pf20.json \
  --adaptive-config configs/btc_v8_adaptive.json \
  --asset BTC --start 2024-01-01 --end 2024-12-31

# Expected: PASS (64 trades both paths)
```

### Phase 2: Prove Regime Locking Works
```bash
# Test 2: Lock to risk_on profile
# Edit config: "locked_regime": "risk_on"
python3 bin/backtest_knowledge_v2.py \
  --asset BTC --start 2024-01-01 --end 2024-12-31 \
  --config configs/btc_v8_adaptive.json

# Expected: Deterministic results, no regime blending
```

### Phase 3: Unlock and Verify Blending
```bash
# Test 3: Remove locked_regime, enable full blending
# Edit config: remove "locked_regime" key
python3 bin/backtest_knowledge_v2.py \
  --asset BTC --start 2024-01-01 --end 2024-12-31 \
  --config configs/btc_v8_adaptive.json

# Expected: Smooth regime-aware morphing (after GMM fix)
```

## Known Issues

### Issue 1: Legacy Path Still Uses check_archetype()

**Status**: EXPECTED

The legacy config (`baseline_btc_bull_pf20.json`) does NOT have `fusion_regime_profiles`, so it never initializes ThresholdPolicy or RuntimeContext. It uses the old `check_archetype(row, prev_row, df, index)` method.

**Why This Is Fine**:
- Legacy path is frozen baseline (production-ready PF 3.13)
- Parity test compares OUTPUT (trades), not CODE PATH
- If locked adaptive matches legacy trades, paths are equivalent at decision boundary

### Issue 2: First Parity Test May Fail

**Status**: EXPECTED

The test is DESIGNED to fail if the adaptive path has bugs. If it fails:

1. Examine diagnostic output:
   ```
   Missing in adaptive: 12 trades
   Extra in adaptive: 45 trades
   ```

2. Fix adaptive path (e.g., threshold enforcement logic)

3. Re-run parity test until PASS

4. ONLY THEN proceed with GMM/regime work

## Files Modified

**Core Infrastructure**:
- `engine/archetypes/threshold_policy.py` - Added locked_regime support
- `bin/backtest_knowledge_v2.py:217` - Pass locked_regime to ThresholdPolicy

**Testing & Utilities**:
- `tests/test_parity_legacy_vs_adaptive.py` - Parity test suite (NEW)
- `bin/copy_legacy_thresholds.py` - Legacy threshold converter (NEW)

**Documentation**:
- `docs/PRA_PARITY_TESTING.md` - This file (NEW)

## Success Criteria

PR-A is complete when:

- ✅ ThresholdPolicy supports locked_regime parameter
- ✅ Parity test script created and runnable
- ✅ Legacy threshold copy utility created
- ⏳ Parity test PASSES on BTC 2024 (locked adaptive == legacy static)

## Next Steps (PR-B: GMM Retraining)

**Only after parity test passes**:

1. Train GMM v3 with corrected label mapping
2. Validate on held-out 2024 data
3. Re-run adaptive backtest with fixed GMM
4. Expected: Fewer trades in risk_off, similar/better in risk_on

## Conclusion

PR-A establishes the critical "parity or fail" guardrail. This prevents wasting time on regime optimization when the underlying adaptive path may be broken.

**Key Insight**: You can't fix what you can't measure. Parity testing makes code path equivalence measurable and testable.

**Status**: Infrastructure complete. Ready for parity validation.

