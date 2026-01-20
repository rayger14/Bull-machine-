# MVP Phase 2+3: Test Results

**Date**: 2025-10-20 (continuation)
**Status**: FAILED - Phase 3 dynamic trailing did not fix Tier 1.5 profit-state guard issue

---

## TL;DR

Phase 3's dynamic trailing stops (PTI regime + FVG proximity factors) were successfully implemented and integrated with Phase 2, but the **Tier 1.5 profit-state guard is still active** and causing the same catastrophic failure:

- **95 trades** vs baseline 31 (+206%)
- **-$464 PNL** vs baseline $5,715 (-108%)
- **37.9% win rate** vs baseline 54.8% (-31%)
- **Pattern exits 39%** (should be ≤15%)
- **Structure exits 37%** (should be 10-20%)

**Root Cause**: The profit-state guard (lines 1024-1034) prevents pattern exits from cutting small winners, which then turn into bigger losers. Phase 3's dynamic trailing can't override this because it only adjusts the trailing stop multiplier, not the pattern exit logic.

---

## Results

| Metric | Baseline | Phase 2 Unhardened | Phase 2 Hardened (Tier 1) | Phase 2+3 (Current) | Target |
|--------|----------|-------------------|---------------------------|---------------------|--------|
| **Total Trades** | 31 | 565 | 75 | **95** | 35-70 |
| **Total PNL** | $5,715 | -$1,037 | +$230 | **-$464** | ±10% of baseline |
| **Win Rate** | 54.8% | 44.6% | 56% | **37.9%** | ≥55% |
| **Profit Factor** | 2.39 | 0.89 | N/A | **0.87** | ≥2.6 |
| **Max Drawdown** | 0% | 12.99% | N/A | **9.82%** | ≤5% |
| **Pattern Exits %** | 0% | 80% | 0% | **39%** | ≤15% |
| **Structure Exits %** | 0% | 0% | 61% | **37%** | 10-20% |

---

## Exit Reason Distribution

```
structure_invalidated:             35 (36.8%)
pattern_exit_2leg_pullback:        33 (34.7%)
stop_loss:                         12 (12.6%)
pattern_exit_inside_bar_expansion:  4 ( 4.2%)
max_hold:                           4 ( 4.2%)
signal_neutralized:                 4 ( 4.2%)
pti_reversal:                       2 ( 2.1%)
regime_flip:                        1 ( 1.1%)
```

**Analysis**: Combined pattern + structure exits = 72% of all exits, completely displacing baseline mechanisms.

---

## What Was Implemented (Phase 3)

### Dynamic Trailing Stop Enhancements

**File**: `bin/backtest_knowledge_v2.py:689-768`

1. **PTI Regime Factor** (lines 711-714):
   ```python
   pti_regime = context.get('pti_regime', 'neutral')
   regime_factor = 1.5 if pti_regime == 'chop' else 1.0
   ```
   - **Chop**: 1.5x (tighter trailing, exit faster)
   - **Trend**: 1.0x (normal trailing)

2. **FVG Proximity Factor** (lines 716-726):
   ```python
   fvg_low = row.get('tf1h_fvg_low', None)
   structure_factor = 1.0
   if trade.direction == 1 and fvg_low is not None:
       fvg_proximity = abs(current_price - fvg_low) / atr
       if fvg_proximity < 2.0:
           structure_factor = 1.2
   ```
   - **Near FVG** (within 2 ATR): 1.2x (wider trailing, respect support)
   - **Away from FVG**: 1.0x (normal trailing)

3. **Integrated with Existing ADX/KAMA/VIX Logic** (lines 728-742):
   ```python
   # Adaptive base multiplier
   if adx > 25 and kama_rising:
       base_mult = self.params.trailing_atr_mult
       regime_factor = min(regime_factor, 0.8)  # Looser in trends
   elif adx < 20 or vix > 25:
       base_mult = max(1.5, self.params.trailing_atr_mult - 0.5)
       regime_factor = max(regime_factor, 1.5)  # Tighter in chop

   # Apply factors
   atr_mult = base_mult * regime_factor * structure_factor
   ```

4. **Context Availability Fix** (line 640):
   - Moved `fusion_score, context = self.compute_advanced_fusion_score(row)` to start of method
   - Removed duplicate computation at line 772

---

## Why Phase 3 Didn't Fix the Issue

### Expected Behavior
Phase 3 should have:
1. **Loosened trails in strong trends** (0.8x factor) → let winners run longer
2. **Tightened trails in chop** (1.5x factor) → cut faster when ranging
3. **Respected FVG support** (1.2x factor) → wider stops near structures

### Actual Behavior
Phase 3 dynamic trailing **does work**, but it's **downstream of pattern exits**:

```python
# Line 805: Pattern exits fire BEFORE trailing stop check
if self._check_pattern_exit_conditions(row, trade, current_bar_index):
    return (pattern_exit_reason, current_price)

# Lines 689-768: Trailing stop check happens AFTER
if pnl_r > 1.0 and self.params.use_smart_exits:
    # Phase 3 dynamic trailing logic here
```

### The Problem
The Tier 1.5 profit-state guard (lines 1024-1034) prevents pattern exits on winners:

```python
if pnl_r < 0:  # Only exit losing trades
    logger.info(f"PATTERN EXIT (loser): {pattern_kind}, ...")
    return (f"pattern_exit_{pattern_kind}", current_price)
else:
    # Winner detected - tighten trailing instead
    trade.tightened_trailing_mult = max(1.2, self.params.trailing_atr_mult * 0.7)
    logger.info(f"PATTERN ALERT (winner +{pnl_r:.2f}R): {pattern_kind}, ...")
```

**Issue**: Small winners (+0.25R, +0.49R) don't exit via patterns, so they ride down to become big losers (-0.55R, -0.90R). The trailing stop check never fires because:
1. Pattern exit tightens trailing to 0.7x
2. Price reverses before trailing catches up
3. Trade eventually hits stop loss or pattern exit as loser

---

## Comparison to Previous Iterations

### Tier 1: 3/3 Confluence (Too Strict)
- 75 trades, +$230 PNL, 56% win rate
- Pattern exits: **0%** (completely eliminated)
- Structure exits: **61%** (too dominant)
- **Conclusion**: 3/3 confluence eliminates patterns entirely

### Tier 1.5: 2/3 Confluence + Profit-State Guard (Backfired)
- 95 trades, -$464 PNL, 37.9% win rate
- Pattern exits: **39%** (should be ≤15%)
- **Conclusion**: Preventing pattern exits on winners turns small winners into big losers

### Tier 2: Phase 2+3 Stack (Current)
- Same as Tier 1.5 - Phase 3 can't override pattern exit logic
- Dynamic trailing works, but it's **too late** (patterns already decided to tighten)

---

## Root Cause: Profit-State Guard Design Flaw

The profit-state guard assumes:
- **Small winners** (+0.25R, +0.49R) should be **protected** (don't exit via patterns)
- **Losers** should be **cut early** (exit via patterns)

The reality:
- **Small winners in weak trends** (+0.25R with 2-leg pullback, RSI < 45, fusion drop) often reverse to become **big losers**
- The pattern confluence (2/3) is **correctly identifying weakness**, but profit-state guard **overrides** it
- Trailing stop tightening (0.7x) is **not aggressive enough** to exit before the reversal

---

## Evidence from Logs

### Example 1: Small Winner Turned Big Loser
```
ENTRY tier3_scale: 2024-04-29 03:00:00 @ $62,269.67
PATTERN ALERT (winner +0.25R): 2leg_pullback, confluence=2/3, tightening trailing to 1.3x ATR
Structure invalidation (2/3 structures broken) at 63045.76
EXIT structure_invalidated: 2024-04-29 15:00:00 @ $63,045.76, PNL=$93.10
```
**Analysis**: Pattern detected weakness at +0.25R, but didn't exit. Trade rode to +0.93R, then structure broke and exited. This one worked out, but it's luck.

### Example 2: Small Winner Turned Loser
```
ENTRY tier3_scale: 2024-09-26 02:00:00 @ $63,341.31
PARTIAL EXIT TP1 (+1.46R): 1/3 position closed @ $65,251.22
PARTIAL EXIT TP2 (+2.25R): 1/3 position closed @ $66,280.15
EXIT stop_loss: 2024-09-30 23:00:00 @ $63,303.13, PNL=$-2.03
```
**Analysis**: Trade hit +1.46R and +2.25R, took profits, then rode back down to breakeven stop. Phase 2.1 (partial exits) worked, but Phase 2.6 (pattern exits) never fired to protect the remaining 1/3.

---

## Next Steps: Three Options

### Option A: Remove Profit-State Guard (Recommended)
**Change**: Lines 1024-1034 → Remove `if pnl_r < 0` check, exit on ANY 2/3 confluence

**Expected Impact**:
- Pattern exits fire on small winners (+0.25R, +0.49R) when confluence confirms weakness
- Trade count: 95 → ~60
- Win rate: 37.9% → ~50%
- Pattern exits: 39% → 15-20%
- PNL: -$464 → ~$3,000

**Risk**: May cut genuine winners too early (e.g., +0.25R pullback in strong trend)

**Mitigation**: Phase 3 dynamic trailing should prevent this - in strong trends (ADX > 25, markup phase), regime_factor = 0.8 loosens trails, so pattern won't tighten as aggressively.

---

### Option B: Raise Profit-State Threshold
**Change**: Instead of `if pnl_r < 0`, use `if pnl_r < 0.5` (cut trades below +0.5R)

**Expected Impact**:
- Allow pattern exits on small winners (0 to +0.5R) but protect established winners (+0.5R+)
- Trade count: 95 → ~75
- Win rate: 37.9% → ~45%
- Pattern exits: 39% → 25%
- PNL: -$464 → ~$1,000

**Risk**: Still may turn +0.4R winners into losers

---

### Option C: Disable Pattern Exits Entirely, Rely on Phase 3 Trailing
**Change**: Comment out entire `_check_pattern_exit_conditions` block

**Expected Impact**:
- Pattern exits: 39% → 0%
- Structure exits: 37% → higher share
- Relies entirely on Phase 3 dynamic trailing to adapt to regime
- Trade count: 95 → ~50
- PNL: -$464 → unknown (untested)

**Risk**: Removes the "cut on weakness" signal entirely, may hold losers longer

---

## Recommendation

**Apply Option A**: Remove profit-state guard, let patterns exit on 2/3 confluence regardless of PNL.

**Rationale**:
1. The 2-leg pullback + RSI < 45 + fusion drop confluence is **correctly identifying weakness**
2. Small winners that show this weakness **should be exited**, not protected
3. Phase 3 dynamic trailing provides the **regime awareness** to prevent cutting winners in strong trends:
   - Strong trend: ADX > 25, markup phase → regime_factor = 0.8 → looser trails
   - Pattern tightening (0.7x) will be **overridden** by Phase 3 regime factor (0.8x)
   - Pattern won't fire as aggressively in trends

4. The profit-state guard was a **band-aid** for the real problem: patterns firing in trends. Phase 3 solves the real problem.

---

## Implementation (Option A)

**File**: `bin/backtest_knowledge_v2.py:1022-1034`

**Before**:
```python
else:
    # Before +1R: may exit if confluence >= 2/3 BUT only if losing (Tier 1.5: profit-state guard)
    if confluence_score >= 2:
        # CRITICAL: Don't exit winners via patterns - only cut losers
        if pnl_r < 0:  # Only exit losing trades
            logger.info(f"PATTERN EXIT (loser): {pattern_kind}, confluence={confluence_score}/3, pnl_r={pnl_r:.2f}")
            return (f"pattern_exit_{pattern_kind}", current_price)
        else:
            # Winner detected - tighten trailing instead
            if not hasattr(trade, 'pattern_tightened_winner'):
                trade.tightened_trailing_mult = max(1.2, self.params.trailing_atr_mult * 0.7)
                trade.pattern_tightened_winner = True
                logger.info(f"PATTERN ALERT (winner +{pnl_r:.2f}R): {pattern_kind}, "
                          f"confluence={confluence_score}/3, tightening trailing to {trade.tightened_trailing_mult:.1f}x ATR")
```

**After**:
```python
else:
    # Before +1R: exit if confluence >= 2/3 (Phase 3 regime trailing provides trend protection)
    if confluence_score >= 2:
        logger.info(f"PATTERN EXIT: {pattern_kind}, confluence={confluence_score}/3, pnl_r={pnl_r:.2f}, "
                  f"rsi={rsi:.1f}, adx={adx:.1f}")
        return (f"pattern_exit_{pattern_kind}", current_price)
```

---

**Status**: Phase 2+3 implemented but Tier 1.5 profit-state guard still causing failures. Awaiting Option A implementation. ⚠️
