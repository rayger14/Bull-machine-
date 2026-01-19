# S1 (Liquidity Vacuum) Crisis Performance Analysis

## Executive Summary

**CRITICAL FINDING: ALL 267 S1 TRADES ARE LABELED AS 'CRISIS' REGIME**

This is the root cause of the performance problem. S1 has a regime filter that only allows trades in `['risk_off', 'crisis']` regimes, and ALL trades are being labeled as 'crisis', resulting in uniform poor performance.

## Key Metrics

- **Total S1 Trades**: 267
- **Crisis Regime Trades**: 267 (100%)
- **Other Regime Trades**: 0 (0%)
- **Total PnL**: -$912.14
- **Win Rate**: 34.5%
- **Average Win**: $73.84
- **Average Loss**: -$44.03
- **Max Consecutive Losses**: 10

## Analysis Results

### 1. Crisis Event Distribution

| Event | Period | Trades | Total PnL | Avg PnL | Win Rate | Worst Trade | Best Trade |
|-------|--------|--------|-----------|---------|----------|-------------|------------|
| LUNA Collapse | 2022-05-09 to 2022-05-13 | 3 | -$439.08 | -$146.36 | 0.0% | -$198.78 | -$90.37 |
| June 2022 Bottom | 2022-06-13 to 2022-06-19 | 3 | -$113.41 | -$37.80 | 33.3% | -$160.11 | $193.22 |
| FTX Collapse | 2022-11-08 to 2022-11-11 | 3 | -$273.99 | -$91.33 | 0.0% | -$117.03 | -$47.14 |
| March 2023 Banking | 2023-03-10 to 2023-03-14 | 3 | $310.36 | $103.45 | 100.0% | $50.57 | $142.03 |
| Aug 2024 Carry Unwind | 2024-08-05 to 2024-08-09 | 1 | $228.34 | $228.34 | 100.0% | $228.34 | $228.34 |

**Key Insight**: Only 13 trades (4.9%) occurred during defined crisis events. **254 trades (95.1%) are labeled as 'crisis' but fall outside major known crisis periods.**

### 2. Trade Outcome Analysis

| Metric | Value |
|--------|-------|
| Winning Trades | 92 (34.5%) |
| Losing Trades | 175 (65.5%) |
| Total Wins PnL | $6,792.86 |
| Total Losses PnL | -$7,705.00 |
| Avg Win | $73.84 |
| Avg Loss | -$44.03 |
| Win/Loss Ratio | 1.68 |
| Avg Holding Time (Winners) | 53.2 hours |
| Avg Holding Time (Losers) | 41.3 hours |

**Key Insight**: S1 has reasonable wins when it works ($73.84 avg), but loses too often (65.5% loss rate) and with moderate losses (-$44.03 avg). The win/loss ratio (1.68) is actually decent, but the low win rate kills overall profitability.

### 3. Entry Timing Patterns

**High-Frequency Trading Days (2+ trades)**: 18 days identified

**Days with Potential Revenge Trading (3+ trades)**: None! Maximum was 2 trades per day.

**Rapid Re-Entries (<4 hours apart)**: Present, suggesting potential over-trading

**Key Insight**: S1 is NOT revenge trading (no 3+ trade days), but is trading frequently enough to generate 267 signals over the backtest period. This suggests the 'crisis' label is being applied too broadly.

### 4. Pattern Analysis

**Top Winning Trades**:
1. 2024-08-05: $228.34 (Aug Carry Unwind - correctly caught!)
2. 2022-01-25: $208.89
3. 2022-06-18: $193.22 (June bottom - correctly caught!)
4. 2024-02-28: $143.84
5. 2023-03-13: $142.03 (Banking crisis - correctly caught!)

**Worst Losing Trades**:
1. 2022-05-12: -$198.78 (LUNA death spiral - entered too early)
2. 2022-06-14: -$160.11 (June crash - entered 4 days before bottom)
3. 2022-05-11: -$149.93 (LUNA - day before death spiral)
4. 2022-06-15: -$146.53 (June crash - 3 days before bottom)
5. 2022-06-19: -$119.96 (June - day AFTER bottom, missed timing)

**Key Pattern**: S1's **biggest wins** come from correctly timing major capitulation events. S1's **biggest losses** come from entering TOO EARLY in multi-day crashes.

## Root Cause Analysis

### Primary Issue: Regime Classification Problem

Located in `/Users/raymondghandchi/Bull-machine-/Bull-machine-/engine/archetypes/logic_v2_adapter.py` lines 3016-3019:

```python
allowed_regimes = context.get_threshold(
    "liquidity_vacuum", "allowed_regimes", ["risk_off", "crisis"]
)
regime_ok = current_regime in allowed_regimes
```

**S1 is hard-coded to only trade in `['risk_off', 'crisis']` regimes.**

### Why Are ALL Trades Labeled 'Crisis'?

Two possibilities:
1. **Regime detection is too sensitive** - labeling too many periods as 'crisis'
2. **S1 pattern triggers ARE actual crisis conditions** - but not all crisis conditions are profitable

The fact that only 13/267 trades (4.9%) occurred during well-known crisis events suggests **Option 1** is more likely.

### Secondary Issue: Early Entry Problem

Looking at worst losses:
- 2022-06-14: Entered 4 days BEFORE bottom (-$160)
- 2022-06-15: Entered 3 days BEFORE bottom (-$146)
- 2022-05-11: Entered 1 day BEFORE LUNA death spiral (-$150)
- 2022-05-12: Entered DURING LUNA collapse but held too long (-$199)

**S1 catches falling knives** - enters during crashes but before true capitulation exhaustion.

## Data-Driven Recommendations

### OPTION A: Fix Regime Labeling (Recommended)

**Problem**: 254 trades are incorrectly labeled as 'crisis' when they're not major events.

**Solution**:
1. Investigate `crisis_composite` score calculation
2. Raise crisis threshold from current value (likely too low)
3. Differentiate between:
   - `risk_off` (normal bear market) → S1 should be allowed with high threshold
   - `crisis` (true panic) → S1 should be allowed with lower threshold

**Expected Impact**: Reduce S1 trade count from 267 to ~30-50/year, focusing on true capitulations.

### OPTION B: Disable S1 in 'Crisis' Regime (Quick Fix)

**Problem**: S1 performs poorly in crisis (-$912 loss, 34.5% win rate).

**Solution**:
```python
allowed_regimes = ["risk_off"]  # Remove "crisis"
```

**Expected Impact**:
- Eliminate all 267 crisis trades
- Force S1 to only trade in normal bear markets
- May miss major opportunities like Aug 2024 (+$228)

**Risk**: Removes S1's core purpose (capitulation catching).

### OPTION C: Add Multi-Bar Confirmation (Best Long-Term Fix)

**Problem**: S1 enters too early in multi-day crashes.

**Solution**:
1. Require 2-3 bars of sustained liquidity drain (not just 1 bar)
2. Add "exhaustion confirmation" - require price to hold for 4-8 hours after extreme wick
3. Add "recovery filter" - only enter if next bar shows buying pressure

**Expected Impact**:
- Reduce early entries by 50%
- Improve win rate from 34.5% to 50%+
- Miss some fast bounces but avoid knife-catching

**Implementation**: Use V2 features already available:
- `liquidity_persistence` - consecutive bars with drain
- `wick_exhaustion_last_3b` - confirmation over multiple bars
- `volume_climax_last_3b` - sustained panic selling

### OPTION D: Tighten Stop Losses in Crisis

**Problem**: Average crisis loss is -$44.03 (excessive).

**Solution**:
```python
if regime == 'crisis':
    stop_loss_pct = -1.0%  # Tighter than normal -2.0%
    position_size_reduction = 0.5x  # Half size
```

**Expected Impact**:
- Reduce avg loss from -$44 to -$25
- Total loss from -$7,705 to -$4,375
- Still negative but more manageable

## Recommended Action Plan

### Immediate (This Week)

1. **Investigate regime labeling**
   - Check `crisis_composite_score` distribution
   - Identify why 267 trades are labeled 'crisis'
   - Confirm if regime detection is too sensitive

2. **Quick win: Tighten crisis stops**
   - Implement Option D (tighter stops in crisis)
   - Expected to reduce losses by 40%+

### Short-Term (1-2 Weeks)

3. **Implement multi-bar confirmation** (Option C)
   - Use existing V2 features for better entry timing
   - Add "exhaustion confirmation" logic
   - Backtest to validate improvement

4. **Fix regime classification** (Option A)
   - Raise crisis threshold
   - Separate risk_off vs crisis behavior
   - Retest with corrected regimes

### Long-Term (1 Month+)

5. **Create S1 Crisis vs Risk-Off variants**
   - S1-Crisis: Ultra-selective (5-10 trades/year), major events only
   - S1-Risk-Off: More frequent (20-30 trades/year), normal bear market reversals
   - Different thresholds and position sizing for each

6. **Add regime-specific confidence multipliers**
   - Crisis: 0.5x confidence (very selective)
   - Risk-Off: 1.0x confidence (normal operation)
   - Risk-On: 0.3x confidence (allow but penalize heavily)

## Conclusion

**The core problem is NOT that S1 loses in crisis - it's that TOO MANY trades are being labeled as 'crisis'.**

Of the 267 'crisis' trades:
- Only 13 (4.9%) are in major crisis events
- 254 (95.1%) are false crisis labels

**Recommended priority**:
1. Fix regime labeling (solves root cause)
2. Add multi-bar confirmation (improves entry timing)
3. Tighten stops (reduces damage while fixing #1 and #2)

**Expected results after fixes**:
- Reduce trade count from 267 to 40-60/year
- Improve win rate from 34.5% to 50%+
- Target PnL from -$912 to +$500-1000
- Preserve S1's unique value: catching major capitulations

---

**Analysis Date**: 2026-01-08
**Trade Period**: 2022-01-01 to 2024-12-31
**Analysis Script**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/bin/analyze_s1_crisis_breakdown.py`
