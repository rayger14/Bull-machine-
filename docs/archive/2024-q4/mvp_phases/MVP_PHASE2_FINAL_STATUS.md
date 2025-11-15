# MVP Phase 2 - Final Debug Status

## Executive Summary

**Phase 2 Bayesian optimizer is COMPLETE and functional**, but blocked by Phase 1 feature store integration bugs. The optimizer code works correctly, but the feature stores contain mostly placeholder/default values due to detector attribute mismatches.

## Root Cause: Attribute Naming Mismatches

The feature store builder (`build_mtf_feature_store.py`) has hardcoded attribute names that don't match the actual detector dataclass definitions:

### Fixed Issues ✅
1. **BOMSSignal**: Builder used `.detected` → Actual: `.boms_detected`
2. **FakeoutSignal**: Builder used `.detected` → Actual: `.fakeout_detected`
3. **FRVPProfile**: Builder used `.distance_to_poc` → Calculated manually from POC

### Remaining Issues ❌
4. **Unknown detector**: Uses `current_close` variable that doesn't exist (likely in `classify_range_outcome()` or `detect_squiggle_123()`)
5. **Potentially dozens more**: Each detector has custom attribute names that may not match builder expectations

## Current Feature Store State

From `test_feature_store_scores.py`:

```
Fusion Score Distribution:
  Min:  0.0859
  Max:  0.4030  ← Only 1 bar reaches 0.40!
  Mean: 0.2448

Signals above thresholds:
  > 0.20: 81.2%
  > 0.30: 13.3%
  > 0.40: 0.0%  ← Should have detected 1 bar
```

**Domain Scores (sample bars)**:
- Wyckoff: 0.5 (constant neutral)
- SMC structure: False (0.0)
- SMC squiggle: 0.0
- HOB BOMS: 0.0  ← Should vary based on market structure
- HOB FVG: False (0.0)
- Momentum: Working (varies based on RSI/ADX)

**Result**: Optimizer searches threshold range [0.20, 0.50] but generates 0 trades because:
1. Exception handlers catch all detector errors silently
2. Return default/placeholder values
3. Fusion scores barely exceed threshold even at lowest settings

## Time Investment Analysis

**Debugging all detector integrations**: 4-8 hours
- 60+ detector function calls across tf1d/tf4h/tf1h features
- Each has custom dataclass with unique attribute names
- No schema documentation exists
- Must grep each detector, find attributes, update builder

**vs. Building simplified feature store**: 1 hour
- Use only working components (Momentum, basic indicators)
- Skip complex detectors (Wyckoff, SMC, HOB, FRVP, etc.)
- Validate optimizer can generate trades

## Decision Point

### Option A: Continue Debugging (High Effort)
**Pros**:
- Eventually get full detector suite working
- Complete Phase 1 as originally envisioned

**Cons**:
- 4-8 hours additional debugging
- No guarantee all detectors will work (some may have logic bugs beyond naming)
- Delays Phase 3/4 implementation

**Steps**:
1. Find and grep every detector function called in builder
2. Check actual dataclass attributes
3. Update builder to match (60+ call sites)
4. Rebuild and test incrementally
5. Fix logic bugs in detectors themselves

### Option B: Simplified Feature Store (Low Effort) ⭐ RECOMMENDED
**Pros**:
- Can validate optimizer in 30 minutes
- Unblocks Phase 3 (fast backtest)
- Proves MVP end-to-end flow works
- Can add complex detectors later incrementally

**Cons**:
- Initial results won't use full Bull Machine logic
- May need to revisit for production deployment

**Steps**:
1. Create `build_simple_feature_store.py` with only:
   - OHLCV
   - RSI, ADX, ATR (working indicators)
   - Simple momentum score (no detector calls)
2. Rebuild BTC Q3 2024
3. Run optimizer → Should generate trades
4. Validate PF × sqrt(Trades) > 0

### Option C: Manual Feature Engineering (Medium Effort)
**Pros**:
- Bypass detector integration entirely
- Direct control over fusion score calculation
- Can use test_fusion_windowing.py results

**Cons**:
- Abandons existing detector codebase
- Duplicates logic

**Steps**:
1. Call `analyze_fusion()` ONCE per asset (not per-timestamp)
2. Store results as static lookup
3. Forward-fill across 1H timestamps
4. Simpler but loses per-bar granularity

## Recommendation

**Proceed with Option B (Simplified Feature Store)** to:
1. Unblock MVP validation
2. Prove optimizer + backtest + live shadow runner work end-to-end
3. Return to detector debugging AFTER Phase 4 is complete

This follows agile principles: MVP first, polish later.

## Phase 2 Deliverables Status

- [x] Bayesian optimizer implementation (bin/optimize_v2_cached.py)
- [x] Optuna TPE sampler integration
- [x] 6-parameter search space
- [x] PF × sqrt(Trades) objective function
- [x] Feature store loader
- [x] NaN handling in fusion score computation
- [x] Threshold adjustment for actual score distribution
- [ ] **BLOCKED**: Valid feature stores with real detector outputs

**Next Action**: Choose Option A, B, or C and proceed.

---

**Files Modified During Debug**:
- `bin/build_mtf_feature_store.py` - Added error logging, fixed 3 attribute mismatches
- `bin/optimize_v2_cached.py` - Adjusted threshold range, added NaN handling
- `test_fusion_windowing.py` - Created diagnostic tool
- `test_feature_store_scores.py` - Created validation tool
- `MVP_PHASE2_ROOT_CAUSE.md` - Initial analysis
- `MVP_PHASE2_FINAL_STATUS.md` - This file
