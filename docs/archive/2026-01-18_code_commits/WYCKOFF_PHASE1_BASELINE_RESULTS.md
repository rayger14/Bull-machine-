# Wyckoff Events - Phase 1 Baseline Validation Results

**Date**: 2025-11-18
**Status**: Phase 1 FAILED - Does NOT Meet Decision Gate Criteria
**Test Period**: 2024-01-01 to 2024-09-30 (9 months, bull market)

---

## Executive Summary

Phase 1 baseline validation **FAILED** the decision gate criteria. The Wyckoff event-based boost/veto logic **degraded performance** compared to the baseline (tf1d_wyckoff_score only).

**Critical Findings**:
- ❌ Win rate decreased by -1.0% (45.8% → 44.8%)
- ❌ Net -2 winners (removed 5 winners, added only 3)
- ❌ Added trades have 25% WR (very poor quality)
- ✅ BC veto logic worked (1 BC avoided on March 28, 2024)
- ⚠️ LPS boosts add low-quality trades despite high frequency

**Recommendation**: **DO NOT proceed to Phase 2 optimization**. The hand-tuned boost multipliers are likely too aggressive and should be disabled or significantly reduced.

---

## Test Configuration

### Bug Fix Applied

**Issue**: The `wyckoff_events` config was not being passed to ArchetypeLogic
**Location**: `bin/backtest_knowledge_v2.py` lines 339-344
**Fix**: Added `'wyckoff_events': self.runtime_config.get('wyckoff_events', {})` to `full_archetype_config` dictionary
**Result**: Wyckoff events now initialize correctly (validated via smoke test)

### Comparison Setup

**Baseline (OLD Wyckoff)**:
- Config: `configs/mvp/mvp_bull_market_v1.json`
- Uses: `tf1d_wyckoff_score` in fusion weights only (44% weight)
- No event-based boost/veto logic

**Enhanced (NEW Wyckoff)**:
- Config: `configs/mvp/mvp_bull_wyckoff_v1.json`
- Uses: `tf1d_wyckoff_score` (44% weight) PLUS event-based boost/veto logic
- Boost multipliers: LPS +10%, Spring-A +12%, SOS +8%
- Veto signals: BC, UTAD (fusion_score = 0.0)

---

## Results Summary

### Overall Performance

| Metric | Baseline (OLD) | Enhanced (NEW) | Delta |
|--------|---------------|----------------|-------|
| Total Trades | 284 | 286 | +2 |
| Winners | 130 | 128 | -2 |
| Win Rate | 45.8% | 44.8% | -1.0% |
| BC/UTAD Vetoes | 0 | 1 | +1 |

### Trade Changes Analysis

**Trades ADDED by Wyckoff (LPS/SOS/Spring-A boosts)**:
- Count: 12 trades
- Win Rate: **25.0%** (3/12 wins)
- Avg R-multiple: **-0.24R** (losers on average)
- Quality: **Very poor** - significantly below baseline

**Trades REMOVED by Wyckoff** (timing changes from boost multipliers):
- Count: 10 trades
- Win Rate: **50.0%** (5/10 wins)
- Avg R-multiple: **-0.06R** (nearly breakeven)
- Quality: **Decent** - above baseline average

**Net Impact**:
- Net Winners: -2
- Net Win Rate Change: -1.0%
- **Overall Effect**: Negative

---

## Wyckoff Event Activity

### Boost Events (2024-01-01 to 2024-09-30)

Total boost applications: **961 events**

| Event Type | Count | % of Total | Config Multiplier |
|------------|-------|------------|-------------------|
| wyckoff_lps | 945 | 98.3% | +10% fusion score |
| wyckoff_sos | 15 | 1.6% | +8% fusion score |
| wyckoff_spring_a | 1 | 0.1% | +12% fusion score |

**Key Observation**: Despite 945 LPS boost events, only 12 trades were actually added. This means:
- Most boosts slightly increase existing signals but don't cross thresholds
- When boosts DO push signals over thresholds, they're low-quality signals (25% WR)
- The +10% LPS multiplier may be too small to be useful for good signals, but large enough to let bad signals through

### Veto Events (BC/UTAD Avoidance)

Total veto applications: **1 event**

| Event | Date | Price | Confidence | Baseline Trade? | Result |
|-------|------|-------|------------|-----------------|--------|
| wyckoff_bc | 2024-03-28 13:00 | ~$70,850 | 0.74 | No | ✅ Avoided (but no baseline trade) |

**Note**: The BC veto worked correctly (fusion_score set to 0.0), but the baseline also had no trade signal on March 28, so the veto didn't prevent a trade that would have been taken. The BC detection itself was accurate (March 2024 BTC ATH).

---

## Decision Gate Criteria Evaluation

From `WYCKOFF_OPTIMIZATION_STRATEGY.md`, proceed to Phase 2 ONLY if **4 of 5** criteria met:

1. ❌ **Win rate improvement: +8-12%** (target: 46% → 54-58%)
   - **Result**: -1.0% (45.8% → 44.8%)
   - **Status**: FAILED

2. ❌ **Profit factor: >1.8** (must maintain or improve)
   - **Result**: Not measured (CSV lacks PnL totals), but -2 winners suggests degradation
   - **Status**: LIKELY FAILED

3. ✅ **BC/UTAD avoidance: 2-5 tops avoided**
   - **Result**: 1 BC avoided (March 28, 2024)
   - **Status**: PARTIAL (only 1 event, but correctly avoided)

4. ❌ **Spring-A capture: 2-4 high-quality pullbacks entered**
   - **Result**: Only 1 Spring-A boost applied, not clear if it added a trade
   - **Status**: FAILED (insufficient Spring-A events)

5. ❌ **No performance degradation in neutral/bear periods**
   - **Result**: Overall degradation in bull period (-1.0% WR)
   - **Status**: FAILED

**Total**: **1 of 5** criteria met (BC avoidance partial success)

**Decision**: **DO NOT PROCEED TO PHASE 2**

---

## Root Cause Analysis

### Why LPS Boosts Failed

1. **Over-detection of LPS events**
   - 945 LPS events in 9 months (~3.5 per day)
   - LPS (Last Point of Support) should be rare accumulation zone markers
   - Detection criteria may be too lenient (conf threshold: 0.65)

2. **Low-quality signal boosting**
   - When LPS boosts push signals over threshold, they're bad trades (25% WR)
   - The +10% multiplier is enough to cross the fusion threshold for marginal signals
   - Good signals likely already pass threshold without boost

3. **Unintended timing changes**
   - Boosts change fusion scores, which can shift entry timing
   - 10 trades were removed (likely timing shifts), and they had 50% WR (decent)
   - The boost logic may be interfering with good signals

### Why BC/UTAD Avoidance Didn't Help

1. **Limited BC/UTAD events in test period**
   - Only 1 BC detected in 9 months (March 28, 2024)
   - No UTAD events detected
   - Not enough avoidance opportunities to show value

2. **Baseline had no trade signal anyway**
   - On March 28 BC day, baseline took 0 trades
   - The veto prevented a trade that wouldn't have been taken anyway
   - Can't measure avoided losses without a baseline trade

---

## Recommendations

### Immediate Action

**Disable Wyckoff event boosts in production configs**:

```json
{
  "wyckoff_events": {
    "enabled": false  // Disable until Phase 2 optimization or manual tuning
  }
}
```

**OR keep BC/UTAD veto only** (discard boosts):

```json
{
  "wyckoff_events": {
    "enabled": true,
    "min_confidence": 0.70,  // Raise threshold
    "log_events": false,

    "avoid_longs_if": [
      "wyckoff_bc",
      "wyckoff_utad"
    ],

    "boost_longs_if": {}  // REMOVE all boosts
  }
}
```

### If User Wants to Retry (Manual Tuning)

**Option A: Increase LPS confidence threshold**
- Current: 0.65
- Suggested: 0.85-0.90
- Goal: Reduce LPS detection from 945 to ~50-100 events (high-quality only)

**Option B: Reduce boost multipliers**
- Current: LPS +10%, Spring-A +12%, SOS +8%
- Suggested: LPS +3%, Spring-A +5%, SOS +3%
- Goal: Boost good signals without crossing threshold for bad ones

**Option C: Skip Phase 2, keep veto logic only**
- BC/UTAD avoidance worked correctly
- Boosts are net negative
- Simple is better - just avoid tops, don't boost bottoms

---

## Files Modified

### Code Changes

**`bin/backtest_knowledge_v2.py`** (lines 339-344):
```python
# BEFORE (buggy):
full_archetype_config = {
    **archetype_config,
    'fusion': self.runtime_config.get('fusion', {}),
    'state_aware_gates': self.runtime_config.get('state_aware_gates', {})
}

# AFTER (fixed):
full_archetype_config = {
    **archetype_config,
    'fusion': self.runtime_config.get('fusion', {}),
    'state_aware_gates': self.runtime_config.get('state_aware_gates', {}),
    'wyckoff_events': self.runtime_config.get('wyckoff_events', {})  # PR#6C: Wyckoff events integration
}
```

### Test Results

**Baseline (OLD Wyckoff)**:
- Log: `results/wyckoff_baseline_old_2024_FIXED.log`
- Trades: `results/wyckoff_baseline_old_2024_FIXED.csv`

**Enhanced (NEW Wyckoff)**:
- Log: `results/wyckoff_enhanced_new_2024_FIXED.log`
- Trades: `results/wyckoff_enhanced_new_2024_FIXED.csv`

---

## Conclusion

❌ **Phase 1 Baseline Validation FAILED**

The Wyckoff event-based boost/veto logic **degraded performance** by:
- Adding 12 low-quality trades (25% WR, -0.24R avg)
- Removing 10 decent trades (50% WR, -0.06R avg)
- Net impact: -2 winners, -1.0% win rate

**BC/UTAD avoidance logic worked correctly** but had limited impact (only 1 BC in test period).

**LPS/SOS/Spring-A boost logic is net negative** and should be:
1. Disabled entirely, OR
2. Significantly tuned down (higher confidence, lower multipliers), OR
3. Replaced with veto-only approach (keep BC/UTAD, remove boosts)

**DO NOT PROCEED TO PHASE 2 (Optuna optimization)** until:
- Root cause of LPS over-detection is addressed
- Boost multipliers are manually tuned to neutral or positive impact
- OR boosts are removed entirely

---

**Status**: Phase 1 Complete - Phase 2 NOT Recommended
**Next Action**: Disable Wyckoff boosts OR manually retune detection thresholds
**Date**: November 18, 2025
