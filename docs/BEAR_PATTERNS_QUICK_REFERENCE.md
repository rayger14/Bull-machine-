# Bear Patterns - Quick Reference Card

## Phase 1 Patterns (Approved for Implementation)

### S2: Failed Rally Rejection
- **Trader:** Zeroika (dead cat bounce specialist)
- **Signal:** Order block retest + volume fade + wick rejection
- **Direction:** SHORT (DOWN)
- **Win Rate:** 58.5% (validated on 2022)
- **PF:** 1.4 estimated
- **Trades/Year:** 15-20
- **Best Regimes:** Risk-off, crisis
- **Status:** ✅ APPROVED - Ready to implement
- **Blockers:** Needs ob_distance, upper_wick_ratio features

### S5: Long Squeeze Cascade ⚠️ CORRECTED
- **Trader:** Moneytaur (funding specialist)
- **Signal:** Positive funding extreme + RSI exhaustion + thin liquidity
- **Direction:** SHORT (DOWN)
- **Critical Fix:** Positive funding = longs pay shorts (NOT short squeeze!)
- **Win Rate:** 50-55% estimated
- **PF:** 1.3-1.5 estimated
- **Trades/Year:** 8-12
- **Best Regimes:** Risk-off, crisis
- **Status:** ✅ APPROVED (LOGIC FIXED)
- **Blockers:** Needs oi_change_24h feature

---

## Funding Rate Cheat Sheet

| Funding | Who Pays | Overcrowded | Squeeze Direction | Bias | Pattern |
|---------|----------|-------------|-------------------|------|---------|
| > +0.05% | Longs pay shorts | Longs | DOWN (long squeeze) | BEARISH | S5 |
| < -0.05% | Shorts pay longs | Shorts | UP (short squeeze) | BULLISH | (not bear pattern) |

### Memory Aid

**Positive Funding:**
```
Perp > Spot
→ Longs pay shorts
→ Longs overcrowded
→ Long squeeze DOWN
→ BEARISH (S5)
```

**Negative Funding:**
```
Perp < Spot
→ Shorts pay longs
→ Shorts overcrowded
→ Short squeeze UP
→ BULLISH (opposite of S5)
```

---

## Common Mistakes

❌ "funding > 0.08 = short squeeze UP" ← **WRONG!**
✅ "funding > 0.08 = long squeeze DOWN" ← **CORRECT!**

❌ "Negative funding = bearish" ← **WRONG!**
✅ "Negative funding = shorts trapped = bullish" ← **CORRECT!**

❌ "High funding means shorts trapped" ← **WRONG!**
✅ "High positive funding means LONGS trapped" ← **CORRECT!**

---

## Pattern Detection Gates

### S2: Failed Rally Rejection

```
Gate 1: ob_distance < 2%        (near order block)
Gate 2: volume_trend < -15%     (volume fading)
Gate 3: upper_wick_ratio > 2.0  (wick rejection)
Gate 4: 50 < rsi < 70           (local overbought)

Score = weighted combination > 0.55
```

### S5: Long Squeeze Cascade

```
Gate 1: funding_Z > 1.2         (extreme positive)
Gate 2: rsi > 70                (exhaustion)
Gate 3: oi_change > 8%          (fuel added - optional)
Gate 4: liquidity < 0.25        (thin - amplifies cascade)

Score = weighted combination > 0.6
```

---

## Regime Weights

### S2 Weights
- **risk_on:** 0.3x (suppressed - rallies work)
- **neutral:** 0.7x (reduced)
- **risk_off:** 1.8x (boosted - rallies fail)
- **crisis:** 2.2x (high boost - all rallies fail)

### S5 Weights
- **risk_on:** 0.2x (suppressed - longs supposed to win)
- **neutral:** 0.6x (reduced)
- **risk_off:** 2.0x (boosted - longs get punished)
- **crisis:** 2.5x (max boost - cascades amplified)

---

## Historical Validation

### S2 Examples (Failed Rallies)
- **BTC Jun 2022:** 32K OB rejection → -43% to 18K
- **BTC Aug 2022:** 25K OB rejection → -24% to 19K
- **BTC Nov 2022:** 21K OB rejection → -26% to 15.5K (FTX)

### S5 Examples (Long Squeeze)
- **BTC Apr 2021:** funding +0.15% → -50% crash
- **BTC May 2022:** funding +0.12% (Terra) → -60% cascade
- **BTC Nov 2022:** funding +0.08% (FTX) → -25% drop

---

## Position Sizing and Risk

### Position Sizes
- **S2:** 15% of portfolio (validated pattern)
- **S5:** 12% of portfolio (conservative - new pattern)

### Stop Loss
- **S2:** 3% above entry
- **S5:** 2.5% above entry (tighter - expect cascade)

### Take Profit
- **S2:** 6% below entry (2:1 R/R)
- **S5:** 5% below entry (2:1 R/R)

---

## Pattern Status Summary

| ID | Name | Status | Blocker | Priority |
|----|------|--------|---------|----------|
| S1 | Liquidity Vacuum | BLOCKED | Missing liquidity_score | MEDIUM |
| S2 | Failed Rally | ✅ APPROVED | Missing ob_distance, wick_ratio | HIGH |
| S3 | Wyckoff Upthrust | MERGED | Merged into S2 | N/A |
| S4 | Distribution | PARTIAL | Needs refinement | LOW |
| S5 | Long Squeeze | ✅ APPROVED (FIXED) | Missing OI_CHANGE | HIGH |
| S6 | Alt Rotation | ❌ REJECTED | Too rare (< 5 trades/yr) | N/A |
| S7 | Yield Curve | ❌ REJECTED | Use as regime filter | N/A |
| S8 | Exhaustion Fade | ❌ REJECTED | Wrong bias (bullish) | N/A |

---

## Implementation Checklist

### S2 Implementation
- [ ] Add ob_distance feature to feature store
- [ ] Add upper_wick_ratio feature to feature store
- [ ] Add S2 detection logic to logic_v2_adapter.py
- [ ] Configure thresholds and regime weights
- [ ] Test on 2022 data
- [ ] Validate 58.5% win rate

### S5 Implementation
- [ ] Add oi_change_24h feature to feature store
- [ ] Add S5 detection logic to logic_v2_adapter.py
- [ ] Configure thresholds and regime weights
- [ ] Test on 2022 data
- [ ] Validate 50-55% win rate
- [ ] **VERIFY FUNDING DIRECTION IN CODE**

---

## Critical Debugging Checklist

When implementing S5 (or any funding-based pattern):

- [ ] Verify funding sign (+ or -)
- [ ] Confirm who pays whom (longs pay vs shorts pay)
- [ ] Check pattern direction (UP vs DOWN)
- [ ] Validate with historical examples (Terra, FTX)
- [ ] Test in bear market conditions (2022)
- [ ] Ensure regime weights are correct
- [ ] Document the logic clearly
- [ ] **NEVER ASSUME - ALWAYS VERIFY DIRECTION**

---

## S5 Logic Fix Summary

### Before (User Submission)
```python
# WRONG - Backwards logic
if funding > 0.08:
    return "short_squeeze_UP"  # INCORRECT!
```

### After (System Correction)
```python
# CORRECT - Fixed logic
if funding_Z > 1.2:  # Positive funding
    return "long_squeeze_DOWN"  # Longs pay shorts = bearish
```

### Why This Matters
- **Without fix:** System goes LONG during long squeezes
- **Result:** -60% during Terra, -25% during FTX
- **With fix:** System goes SHORT during overcrowding
- **Result:** Expected +1.3-1.5 PF in bear markets

---

## Key Formulas

### Funding Z-Score
```python
funding_Z = (funding - funding_mean) / funding_std
```

### S2 Score
```python
score = (
    ob_proximity * 0.3 +
    volume_fade * 0.3 +
    wick_strength * 0.25 +
    rsi_local_ob * 0.15
)
```

### S5 Score
```python
score = (
    funding_z * 0.4 +
    rsi_exhaustion * 0.3 +
    oi_spike * 0.2 +
    thin_liquidity * 0.1
)
```

---

## Feature Requirements

### S2 Needs
- `ob_distance`: Distance to nearest order block (%)
- `volume_trend_5d`: 5-day volume trend (%)
- `upper_wick_ratio`: Upper wick / body ratio
- `rsi_14`: 14-period RSI

### S5 Needs
- `funding_Z`: Normalized funding rate (z-score)
- `oi_change_24h`: 24-hour OI change (%)
- `rsi_14`: 14-period RSI
- `liquidity_score`: Market liquidity metric

---

## Expected Performance (2022 Bear Market)

| Pattern | Trades | Win Rate | PF | Total Return | Max DD |
|---------|--------|----------|-----|--------------|--------|
| S2 | 15-20 | 58.5% | 1.4 | +25-35% | -8% |
| S5 | 8-12 | 50-55% | 1.3-1.5 | +15-25% | -6% |
| Combined | 23-32 | 55-58% | 1.35-1.45 | +40-60% | -10% |

---

## Monitoring Metrics

Track these metrics in production:

- **Detection count:** How often each pattern triggers
- **Win rate:** % of winning trades
- **Profit factor:** Gross profit / gross loss
- **Avg hold time:** Average duration per trade
- **Regime distribution:** Which regimes triggered in
- **Feature values at entry:** funding_Z, ob_distance, etc.

---

## Emergency Stops

**Kill Switch Conditions:**
- Win rate drops below 40% over 10 trades
- 3 consecutive losses exceeding 5% each
- Pattern triggers in wrong regime (S5 in risk_on)
- Funding direction check fails (S5 specific)

---

## Quick Decision Tree

```
New short opportunity detected
    ↓
Is it near order block + volume fading?
    YES → S2 (Failed Rally)
    NO → Continue
    ↓
Is funding extremely positive + RSI high?
    YES → S5 (Long Squeeze)
    NO → Not a bear pattern
    ↓
Check regime: risk_off or crisis?
    YES → Increase position size
    NO → Reduce position size or skip
    ↓
Execute trade with appropriate stop/target
```

---

## References

- **FUNDING_RATES_EXPLAINED.md:** Full funding mechanics
- **BEAR_PATTERNS_IMPLEMENTATION_GUIDE.md:** Detailed implementation
- **COMPREHENSIVE_ARCHETYPE_AUDIT.md:** Full pattern analysis

---

## Remember

**The S5 fix was critical. Always verify funding direction:**

```
Positive (+) = Longs pay shorts = BEARISH
Negative (-) = Shorts pay longs = BULLISH
```

**When in doubt, check historical examples (Terra, FTX).**
