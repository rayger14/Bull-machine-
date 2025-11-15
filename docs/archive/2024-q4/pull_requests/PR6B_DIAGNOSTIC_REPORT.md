# PR#6B Diagnostic Report - Overtrading Issue

**Date**: 2025-10-30
**Status**: BLOCKED - RuntimeContext integration complete but not reducing trade count

## Problem Statement

Acceptance test on BTC 2024 shows **185 trades** (expected: 20-35 trades)
- Profit Factor: 1.62
- Win Rate: 54.6%
- Max Drawdown: 2.7%

## Root Cause Analysis

### 1. Code Integration: ✅ COMPLETE

All infrastructure is correctly implemented:
- `RuntimeContext` dataclass (engine/runtime/context.py) ✅
- `ThresholdPolicy` with 5-step resolution (engine/archetypes/threshold_policy.py) ✅
- All 11 archetype methods refactored to use `ctx.get_threshold()` ✅
- Backtest engine building RuntimeContext per bar ✅
- Bug fix: `regime_probs_ema` key extraction ✅

### 2. Regime Classifier Issue: ❌ MAJOR PROBLEM

**GMM Label Map**: `{0: 'risk_on', 1: 'risk_on', 2: 'risk_off', 3: 'risk_on'}`

**Impact**: 3 out of 4 clusters → 'risk_on'
- Result: 100% risk_on probability for all of 2024
- Causes lowest threshold settings: `final_fusion_floor=0.35`

**Evidence** from logs:
```
INFO:engine.fusion.adaptive:[ADAPTIVE_UPDATE] curr_probs={'risk_on': 1.0, 'risk_off': 0.0}
INFO:engine.fusion.adaptive:[ADAPT_GATES] Final blended gates: {'min_liquidity': 0.12, 'final_fusion_floor': 0.35}
```

### 3. Configuration Threshold Analysis

**Risk_On Regime Thresholds** (from `btc_v8_adaptive.json`):
- `final_fusion_floor`: 0.35
- `min_liquidity`: 0.12

**Archetype Base Thresholds**:
- Archetype B (order_block_retest): 0.359
- Archetype D (failed_continuation): 0.42
- Archetype E (volume_exhaustion): 0.35
- Archetype H (momentum_continuation): 0.544

**The Problem**: Base thresholds are already at or above the risk_on floor!
- ThresholdPolicy enforces MAX(base, floor)
- When floor (0.35) ≤ base (0.35-0.54), no change occurs
- Result: No gating effect, massive overtrading

### 4. RuntimeContext Execution: ❓ UNCERTAIN

**Key Finding**: No RuntimeContext-specific logs found in backtest output
- No evidence of `ctx.get_threshold()` calls
- No logs from `detect(RuntimeContext)` method
- Suggests fallback to `check_archetype()` legacy path

**Suspected Cause**: `context['adapted_params']` may be None
- Line 1721-1736 in backtest_knowledge_v2.py sets `adapted_params = None` on exception
- Line 479 condition: `context['adapted_params']` will be False if None
- Need to verify with debug logging

## The Cascade of Issues

1. **GMM Model** returns 100% risk_on (bad label mapping)
   ↓
2. **ThresholdPolicy** resolves to risk_on floors (0.35 fusion, 0.12 liquidity)
   ↓
3. **Archetype thresholds** already at or above floors (no change)
   ↓
4. **Overtrading** continues (185 trades vs expected 20-35)

## Solutions (Ranked by Impact)

### Option A: Fix GMM Label Mapping (HIGH IMPACT)
**Action**: Retrain regime classifier with proper labels
- Ensure crisis/risk_off clusters properly mapped
- Validate on 2022-2024 data
- Expected: More balanced regime distribution

**Effort**: 4-6 hours (data prep + training + validation)

### Option B: Raise Risk_On Floors (MEDIUM IMPACT)
**Action**: Update `gates_regime_profiles.risk_on` in config:
```json
"risk_on": {
  "min_liquidity": 0.18,      // was 0.12
  "final_fusion_floor": 0.45  // was 0.35
}
```

**Effort**: 15 minutes (config + retest)
**Risk**: May suppress valid bull trades

### Option C: Add RuntimeContext Debug Logging (DIAGNOSTIC)
**Action**: Add logging to verify RuntimeContext path execution:
```python
# In classify_entry_archetype
logger.info(f"[CLASSIFY] threshold_policy={self.threshold_policy is not None}, "
            f"has_adapted={('adapted_params' in context)}, "
            f"adapted_not_none={context.get('adapted_params') is not None}")
```

**Effort**: 30 minutes (add logs + rerun)

### Option D: Bypass GMM for Testing (QUICK TEST)
**Action**: Force neutral regime in adaptive config:
```python
# In backtest, hardcode:
regime_probs = {'neutral': 1.0}
regime_label = 'neutral'
```

**Effort**: 5 minutes (test if architecture works)

## Recommended Path Forward

1. **Immediate** (30 min): Add debug logging (Option C) to confirm RuntimeContext execution
2. **Quick Test** (15 min): Raise risk_on floors (Option B) to validate gating logic
3. **Proper Fix** (4-6 hrs): Retrain GMM with corrected labels (Option A)

## Test Plan

After implementing fixes:

**Quick Validation** (15 min):
```bash
python3 bin/backtest_knowledge_v2.py --asset BTC --start 2024-01-01 --end 2024-12-31 \
  --config configs/btc_v8_adaptive.json
```

**Expected**: 20-35 trades, PF ≥ 1.5, WR ≥ 60%

**Full Validation** (1 hour):
- BTC 2024 (bull): 20-40 trades
- BTC 2022-2023 (bear/chop): Fewer entries, DD ≤ 8%
- Verify regime response (expansion in risk_on, contraction in risk_off)

## Status Summary

**✅ Complete**:
- RuntimeContext infrastructure
- ThresholdPolicy implementation
- Archetype logic refactor
- Backtest engine integration

**❌ Blocked**:
- GMM label mapping prevents proper regime detection
- Config thresholds insufficient to gate trades
- RuntimeContext execution uncertain (needs logging)

**⏳ Pending**:
- Debug logging to verify execution path
- GMM retraining
- Acceptance testing with fixes
- Unit tests for ThresholdPolicy

**Estimated Time to Resolution**: 6-8 hours (GMM retrain + validation + testing)
