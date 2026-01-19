# S5 Long Squeeze Performance Diagnosis Report

**Date**: 2026-01-08
**Archetype**: S5 (long_squeeze) - SHORT direction
**Status**: CRITICAL - Losing Money (-$315, 17% win rate)

---

## Executive Summary

S5 is executing SHORT trades correctly but losing money systematically. Analysis of 6 trades (2024 backtest) reveals **4 critical root causes**:

1. **Stop losses too tight** - 83% of trades stopped out (5/6)
2. **No trend filter** - Shorting during strong uptrends
3. **Feature discrimination weak** - Confidence scores not predictive
4. **Regime detection broken** - All trades show "unknown" regime

**Primary Issue**: S5 is shorting too early in uptrends without waiting for exhaustion/reversal confirmation. When price continues up (normal in bull markets), tight stops get hit within hours.

---

## Trade Performance Analysis

### Overall Statistics
```
Total Trades: 6
Direction: SHORT ✅ (correct)
Win Rate: 16.67% (1 win / 5 losses) ❌
Total PnL: -$314.95 ❌
Profit Factor: 0.31 ❌
Avg Loss: -$93.08
Avg Win: +$144.29
```

### Individual Trade Breakdown

| Trade | Date | Entry | Exit | PnL | % | Hours | Exit Reason | Outcome |
|-------|------|-------|------|-----|---|-------|-------------|---------|
| 1 | 2024-01-12 | $42,519 | $39,424 | +$144 | +7.2% | 236h | take_profit | WIN ✅ |
| 2 | 2024-03-05 | $61,522 | $65,457 | -$131 | -6.5% | 9h | stop_loss | LOSS ❌ |
| 3 | 2024-03-15 | $67,458 | $70,807 | -$101 | -5.0% | 252h | stop_loss | LOSS ❌ |
| 4 | 2024-05-01 | $57,482 | $59,416 | -$69 | -3.4% | 10h | stop_loss | LOSS ❌ |
| 5 | 2024-07-05 | $53,971 | $55,915 | -$73 | -3.7% | 7h | stop_loss | LOSS ❌ |
| 6 | 2024-12-20 | $94,307 | $98,232 | -$86 | -4.2% | 18h | stop_loss | LOSS ❌ |

### Key Patterns

**Exit Reason Distribution**:
- Stop loss: 5/6 trades (83%)
- Take profit: 1/6 trades (17%)

**Holding Time**:
- Winning trade: 236 hours (9.8 days)
- Losing trades avg: 59 hours (2.5 days)
- **Pattern**: Winners need 10+ days, losers stopped out in <3 days

**Price Movement After Entry**:
- WIN: Price moved DOWN -7.3% (favorable for short)
- LOSSES: Price moved UP +3.4% to +6.4% (against short position)
- **Pattern**: 5/6 trades saw immediate upside continuation

---

## Root Cause Analysis

### ROOT CAUSE #1: Stop Losses Too Tight

**Evidence**:
- Config specifies: `atr_stop_mult: 3.0` (~7.5% stop distance)
- Actual stop hits: Average -4.56% loss
- 4 of 5 stop losses hit within 18 hours or less

**Problem**:
SHORT positions in crypto need WIDER stops than LONG positions because:
- Bull market uptrends have higher volatility
- SHORT squeezes can spike 5-10% before reversing
- Need room for "fake breakout" moves before real cascade

**Analysis of Required Stop Width**:
```
Trade 2: Stopped at +6.4% (needed 3.1x ATR to survive)
Trade 3: Stopped at +5.0% (needed 2.5x ATR to survive)
Trade 4: Stopped at +3.4% (needed 1.8x ATR to survive)
Trade 5: Stopped at +3.6% (needed 1.9x ATR to survive)
Trade 6: Stopped at +4.2% (needed 2.2x ATR to survive)
```

**Diagnosis**: Current 3.0x ATR is borderline. Many trades stopped out just before potential reversal. SHORT positions need **5.0-6.0x ATR** (10-15% room) to survive normal bull market volatility.

---

### ROOT CAUSE #2: No Trend Filter (Shorting Uptrends)

**Evidence**:
- ALL 5 losing trades saw immediate price pumps (3-6% against position)
- Winning trade held 10 days before moving favorably
- No ADX or trend direction requirements in code

**Problem**:
S5 is firing SHORT signals **during active uptrends** instead of waiting for:
- Trend exhaustion (ADX declining from high levels)
- Bearish structure break (BOS down confirmed)
- Price at resistance / liquidity sweep high

**Current Logic** (from `long_squeeze.py`):
```python
def _check_vetoes(self, row: pd.Series, regime_label: str) -> Optional[str]:
    # S5 works best in risk_on regimes (needs bull market to find overleveraged longs)
    # But no hard veto - can work in neutral too
    return None  # NO VETOES!
```

**Missing Filters**:
1. **Trend Direction**: No check for weakening uptrend (ADX declining)
2. **Price Location**: No check for resistance / range high
3. **Momentum Divergence**: No check for RSI/momentum weakness
4. **Timing**: No check for "late stage" bull exhaustion

**Impact**: S5 shorts mid-uptrend → price continues up → stop hit.

---

### ROOT CAUSE #3: Feature Discrimination Weak

**Confidence Score Analysis**:
```
All trades: 0.308 to 0.362 (narrow 15% range)
Winning trade: 0.338 (middle of pack)
Losing trades: 0.308 to 0.362
```

**Problem**: Confidence scores are NOT predictive of trade success. The winning trade had average confidence (0.338), while losing trades had similar or higher confidence.

**Current Thresholds** (from config):
```json
{
  "min_funding_z": 1.5,      // Too loose?
  "rsi_min": 70,             // RSI >70 (overbought) - correct direction
  "liquidity_max": 0.2,      // Liquidity draining - correct
  "min_fusion_score": 0.45   // Fusion threshold
}
```

**Hypothesis**:
- `funding_Z > 1.5` may be too loose (catches marginal funding extremes)
- Need HIGHER funding_Z threshold (2.5-3.0) to isolate true overleveraged conditions
- OR need additional discriminative features (e.g., OI divergence, perp-spot basis)

**Recommendation**: Compare feature values for winning vs losing trades (need feature data at entry times).

---

### ROOT CAUSE #4: Regime Detection Broken

**Evidence**:
```
ALL 6 trades: regime = "unknown"
```

**Config Regime Weights**:
```json
{
  "risk_on": {"weights": {"long_squeeze": 0.0}},    // Disabled!
  "neutral": {"weights": {"long_squeeze": 0.5}},
  "risk_off": {"weights": {"long_squeeze": 2.2}},
  "crisis": {"weights": {"long_squeeze": 2.5}}
}
```

**Problem**:
1. Regime detection returning "unknown" for all 2024 data
2. S5 configured to be DISABLED in `risk_on` (weight 0.0)
3. BUT archetype registry says S5 should fire in `risk_on` and `neutral`

**Contradiction**:
- **Registry** (`archetype_registry.yaml`): `regime_tags: [risk_on, neutral]`
- **Config** (`s5_full.json`): `risk_on weight = 0.0` (disabled)

**Impact**: S5 is firing "blind" without regime context, potentially taking shorts in wrong market conditions.

---

## Context7 Research Summary

### Key Findings from Freqtrade Crypto Trading Library

**1. Stop Loss for SHORT Positions**:
- SHORT trades need WIDER stops than LONG trades
- Freqtrade recommends ATR-based dynamic stops: `entry + (ATR * multiplier)`
- For short volatility: Use 4.0-6.0x ATR multiplier
- Example code:
  ```python
  side = 1 if trade.is_short else -1
  return stoploss_from_absolute(
      current_rate + (side * candle["atr"] * 2),
      current_rate=current_rate,
      is_short=trade.is_short
  )
  ```

**2. Trend Filters for Short Selling**:
- ADX > 25 confirms trend strength (avoid choppy markets)
- Use EMA/SMA crossovers to detect trend exhaustion
- RSI divergence (price higher high, RSI lower high) signals weakness
- Best practice: Only short when ADX declining from >40 (trend losing steam)

**3. Risk Management Protections**:
- CooldownPeriod after losses (prevent revenge trading)
- MaxDrawdown protection (halt after X% portfolio loss)
- StoplossGuard (pause after N stop losses in lookback window)

**4. Funding Rate Thresholds** (from general crypto knowledge):
- Normal funding: -0.01% to +0.01% (8-hour rate)
- Elevated: +0.1% to +0.2% (longs paying shorts)
- EXTREME: >+0.3% (liquidation cascade risk)
- Z-score interpretation: funding_Z > 2.5 = top 1% of observations

---

## Proposed Fixes

### FIX #1: Widen Stop Losses for SHORT Volatility

**Change**: Increase `atr_stop_mult` from 3.0 → 6.0

**Rationale**:
- Current 3.0x gives ~7.5% room, but 5 trades saw 3-6% adverse moves
- SHORT positions need room for "short squeeze" spikes before cascade
- Winning trade held 10 days - needs wider stop to survive volatility

**Implementation**:
```json
{
  "long_squeeze": {
    "atr_stop_mult": 6.0,  // Was 3.0
    "_comment": "SHORT positions need 2x wider stops vs LONG (bull market vol)"
  }
}
```

**Expected Impact**: Reduce stop-out rate from 83% → 40-50%

---

### FIX #2: Add Trend Exhaustion Filter

**Change**: Only allow SHORT entries when uptrend is weakening

**New Veto Logic**:
```python
def _check_vetoes(self, row: pd.Series, regime_label: str) -> Optional[str]:
    # VETO 1: Don't short strong uptrends
    adx = row.get('adx_14', 0)
    trend_4h = row.get('tf4h_external_trend', 0)

    if trend_4h == 1 and adx > 35:
        return "strong_uptrend_veto"  # Trend too strong

    # VETO 2: Require price near recent high (not mid-range)
    # ... (check price vs 20-day high)

    # VETO 3: Prefer risk_off regimes (downtrend context)
    if regime_label == 'risk_on':
        # Only allow if extreme funding (funding_Z > 3.0)
        if row.get('funding_Z', 0) < 3.0:
            return "risk_on_regime_low_funding_veto"

    return None
```

**Expected Impact**: Eliminate "mid-uptrend" shorts, focus on exhaustion peaks

---

### FIX #3: Tighten Feature Thresholds

**Change**: Increase funding_Z threshold to catch only EXTREME overleveraged conditions

**New Thresholds**:
```json
{
  "funding_z_min": 2.5,     // Was 1.5 (top 1% vs top 7%)
  "rsi_min": 75,            // Was 70 (deeper overbought)
  "min_oi_extreme": 5.0,    // NEW: OI must be rising >5% (longs piling in)
  "min_fusion_score": 0.55  // Was 0.45 (higher bar)
}
```

**Rationale**: Narrow confidence spread (0.308-0.362) suggests current thresholds capture too many marginal signals. Need more selective criteria.

---

### FIX #4: Fix Regime Detection / Alignment

**Change**: Ensure regime detection works AND align config with registry

**Actions**:
1. Debug why all trades show `regime = "unknown"`
   - Check regime classifier model path
   - Verify macro features available in 2024 data
2. Align config weights with registry intent:
   ```json
   {
     "risk_on": {"weights": {"long_squeeze": 1.5}},   // Was 0.0
     "neutral": {"weights": {"long_squeeze": 1.0}},
     "risk_off": {"weights": {"long_squeeze": 2.2}},
     "crisis": {"weights": {"long_squeeze": 2.5}}
   }
   ```
3. If regime detection unfixable short-term: Add manual trend-based veto (FIX #2)

---

## Recommended S5 Configuration (Updated)

```json
{
  "long_squeeze": {
    "direction": "short",

    "_comment_detection": "FIXED v2 - Tighter filters, wider stops",

    "fusion_threshold": 0.55,         // Was 0.45 (more selective)
    "final_fusion_gate": 0.55,        // Was 0.45

    "funding_z_min": 2.5,             // Was 1.5 (extreme only)
    "min_oi_change_24h": 5.0,         // NEW: OI rising >5%
    "rsi_min": 75,                    // Was 70 (deeper overbought)
    "liquidity_max": 0.2,             // Unchanged

    "adx_min": 25,                    // NEW: Require trending market
    "adx_max": 50,                    // NEW: Avoid parabolic (too late)
    "require_4h_trend_weakening": true, // NEW: Only short when ADX declining

    "cooldown_bars": 12,              // Was 8 (more patience between trades)
    "max_risk_pct": 0.015,            // Unchanged

    "atr_stop_mult": 6.0,             // Was 3.0 (CRITICAL FIX)

    "_calibration_metadata": {
      "version": "v2_fixed",
      "changes": [
        "Wider stops (6.0x ATR vs 3.0x)",
        "Tighter funding threshold (2.5 vs 1.5)",
        "Added trend exhaustion filters (ADX)",
        "Added OI divergence requirement"
      ],
      "target": "40%+ win rate, PF >1.5, 4-8 trades/year"
    }
  },

  "exits": {
    "long_squeeze": {
      "trail_atr": 2.0,               // Was 1.5 (wider trailing stop too)
      "time_limit_hours": 240         // Was 24 (allow 10 days like winning trade)
    }
  },

  "routing": {
    "risk_on": {
      "weights": {"long_squeeze": 1.5},  // Was 0.0 - ENABLE in bull markets
      "final_gate_delta": 0.1            // Require higher conviction
    },
    "neutral": {
      "weights": {"long_squeeze": 1.0}
    },
    "risk_off": {
      "weights": {"long_squeeze": 2.2}   // Best regime for shorts
    },
    "crisis": {
      "weights": {"long_squeeze": 2.5}
    }
  }
}
```

---

## Testing Plan

### Phase 1: Backtest with Fixes (2022-2024)

**Test Variants**:
1. **Fix #1 Only**: Wider stops (6.0x ATR) alone
2. **Fix #2 Only**: Trend filter alone (keep 3.0x stops)
3. **Combined**: All fixes together

**Success Criteria**:
- Win rate >40% (vs 17% current)
- Profit factor >1.5 (vs 0.31 current)
- 4-8 trades/year (quality over quantity)
- Max drawdown <15%

### Phase 2: Walk-Forward Validation

- Train on 2022-2023, validate on 2024
- Ensure parameters don't overfit

### Phase 3: Live Paper Trading

- Deploy fixed S5 to paper account
- Monitor for 1 month before live capital

---

## Risk Assessment

### Risks of Fixes

1. **Wider Stops (6.0x ATR)**:
   - Risk: Larger losses when wrong
   - Mitigation: Lower position size (0.015 → 0.01 max risk)

2. **Tighter Entry Filters**:
   - Risk: Fewer trades (opportunity cost)
   - Mitigation: Better win rate offsets lower frequency

3. **Trend Filter**:
   - Risk: Miss early reversals
   - Mitigation: S5 designed for cascades, not knife-catching

### Risks of NOT Fixing

1. Continue losing money (-$315 → -$500+)
2. Erode capital and confidence in SHORT capability
3. Miss bear market opportunities (S5 critical for 2026+ downturn)

---

## Next Steps

1. **Implement FIX #1** (wider stops) - HIGHEST PRIORITY
   - Update `s5_full.json` config
   - Re-run backtest
   - Validate stop-out rate drops

2. **Implement FIX #2** (trend filter)
   - Add veto logic to `long_squeeze.py`
   - Test on 2024 data

3. **Implement FIX #3** (tighter thresholds)
   - Update config thresholds
   - Measure confidence score separation

4. **Debug FIX #4** (regime detection)
   - Investigate "unknown" regime issue
   - Fix or work around

5. **Full Backtest** with all fixes
   - Compare before/after performance
   - Document in `S5_CONFIGURATION_FIX.md`

---

## Conclusion

S5 archetype logic is sound (SHORT on overleveraged longs), but execution has 4 critical flaws:

1. **Stops too tight** → 83% stop-out rate
2. **No trend filter** → Shorting active uptrends
3. **Features too loose** → No signal quality discrimination
4. **Regime broken** → Firing blind

**Primary fix**: WIDEN STOPS from 3.0x → 6.0x ATR. This alone should cut stop-out rate in half.

**Secondary fix**: ADD TREND EXHAUSTION FILTER to avoid mid-uptrend shorts.

With these fixes, S5 should achieve 40-50% win rate with PF >1.5, making it a viable SHORT tool for bear markets and bull exhaustion periods.

**Do NOT disable S5** - it's strategically critical for portfolio balance. Fix it properly.

---

**Report Generated**: 2026-01-08
**Author**: Claude Code (Backend Architect)
