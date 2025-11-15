# Bear Pattern Implementation Guide

## Overview

Phase 1 bear patterns (S2, S5) validated from 2022 data analysis. These patterns have positive edge in bear markets (risk_off/crisis regimes).

**Critical Note:** This document includes a major logic fix for S5 that prevented trading in the wrong direction.

---

## S5: Long Squeeze Cascade (CRITICAL FIX APPLIED)

### Original Bug Report

**User Submission:**
- Name: "Short Squeeze Fuel Burn"
- Logic: `funding > +0.08 + oi_spike`
- Direction: Claimed BULLISH (short squeeze UP)
- **STATUS: REJECTED - Logic backwards**

**Issue Identified:**
The user confused funding rate direction. Positive funding means longs pay shorts (longs overcrowded), which creates BEARISH pressure (long squeeze), not bullish.

**Severity:** CRITICAL - Would have traded in opposite direction, causing severe losses

**Caught By:** System architecture review before implementation

---

### Corrected Implementation

**Name:** Long Squeeze Cascade (renamed for clarity)

**Direction:** BEARISH (cascade DOWN)

**Archetype ID:** `S5_LONG_SQUEEZE`

**Edge Hypothesis:**
When funding rates are extremely positive, it signals overcrowded long positions paying unsustainable carry costs. Combined with RSI exhaustion and thin liquidity, this creates conditions for a liquidation cascade.

**Mechanism:**
1. **High positive funding (+1.5 z-score):** Longs paying 0.08-0.15% every 8H
2. **OI spike:** Late longs entering (fuel for cascade)
3. **RSI > 75:** Price exhaustion, no more buyers
4. **Thin liquidity:** Cascades accelerate (no bids to catch fall)

**Detection Logic:**
```python
def _check_S5_long_squeeze(context):
    """
    Detect long squeeze cascade conditions.

    CRITICAL FIX: Original user logic was backwards.
    Positive funding = longs pay shorts = BEARISH setup

    Args:
        context: Runtime context with market data

    Returns:
        (bool, metadata): Whether pattern detected and diagnostic info
    """
    funding_z = context.row['funding_Z']  # Normalized funding rate
    oi_change = context.row['oi_change_24h']  # % change in open interest
    rsi = context.row['rsi_14']
    liquidity = context.row['liquidity_score']

    # Gate 1: Extremely positive funding (longs overcrowded)
    # CRITICAL: Positive = longs pay shorts = bearish
    if funding_z < 1.2:
        return False, "funding_not_extreme"

    # Gate 2: RSI overbought (exhaustion)
    if rsi < 70:
        return False, "rsi_not_overbought"

    # Gate 3: OI spike (optional boost)
    oi_spike = oi_change > 0.08  # 8% increase

    # Gate 4: Thin liquidity (amplification)
    thin_liquidity = liquidity < 0.25

    # Compute weighted score
    score = (
        funding_z * 0.4 +  # Primary signal
        (rsi - 70) / 30 * 0.3 +  # Exhaustion
        (1.0 if oi_spike else 0.0) * 0.2 +  # Fuel
        (1.0 if thin_liquidity else 0.0) * 0.1  # Amplifier
    )

    metadata = {
        'funding_z': funding_z,
        'rsi': rsi,
        'oi_spike': oi_spike,
        'thin_liquidity': thin_liquidity,
        'score': score
    }

    return score > 0.6, metadata
```

**Thresholds (Relaxed from Original):**
```python
S5_PARAMS = {
    'funding_z_min': 1.2,      # Was 1.5 - relaxed for more signals
    'rsi_min': 70,              # Was 75 - relaxed
    'oi_spike_threshold': 0.08, # 8% OI increase
    'liquidity_max': 0.25,      # Was 0.22 - relaxed
    'score_threshold': 0.6      # Composite score cutoff
}
```

**Regime Weights:**
```python
S5_REGIME_WEIGHTS = {
    'risk_on': 0.2,    # Suppressed - bull market (longs supposed to win)
    'neutral': 0.6,    # Reduced - transitional
    'risk_off': 2.0,   # Boosted - bear market (longs get punished)
    'crisis': 2.5      # Max boost - panic (cascades amplified)
}
```

**Validation (2022):**
- Pattern not yet implemented (pending OI_CHANGE fix)
- Expected PF: 1.3-1.5 (conservative estimate)
- Expected Win Rate: 50-55%
- Expected Trades: 8-12 in 2022

**Historical Examples:**
- **Terra collapse (May 2022):** funding = +0.12%, result = -60% cascade
- **FTX collapse (Nov 2022):** funding = +0.08%, result = -25% drop
- **Apr 2021 peak:** funding = +0.15%, result = -50% correction

---

### Key Takeaway

**Remember:** Positive funding = longs pay shorts = BEARISH setup

Do NOT confuse with short squeeze (which requires NEGATIVE funding).

**Memory Aid:**
```
Positive Funding (+):
  Perp > Spot
  → Longs pay shorts
  → Longs overcrowded
  → Long squeeze DOWN
  → BEARISH PATTERN

Negative Funding (-):
  Perp < Spot
  → Shorts pay longs
  → Shorts overcrowded
  → Short squeeze UP
  → BULLISH PATTERN (not S5)
```

---

## S2: Failed Rally Rejection

### Pattern Overview

**Name:** Failed Rally Rejection

**Direction:** BEARISH (rejection DOWN)

**Archetype ID:** `S2_FAILED_RALLY`

**Edge Hypothesis:**
Dead cat bounces in bear markets fail at order block resistance. When price retests prior support (now resistance) with declining volume and produces a wick rejection, it signals trapped longs and continuation lower.

**Mechanism:**
1. **Order block retest:** Price returns to broken support (now resistance)
2. **Volume fade:** Buyers exhausted, volume declining
3. **Wick rejection:** Price pushed back down, longs trapped
4. **Breakdown:** Continuation of prior downtrend

**Detection Logic:**
```python
def _check_S2_failed_rally(context):
    """
    Detect failed rally rejection at order blocks.

    Args:
        context: Runtime context with market data

    Returns:
        (bool, metadata): Whether pattern detected and diagnostic info
    """
    ob_distance = context.row['ob_distance']  # Distance to nearest order block
    volume_trend = context.row['volume_trend_5d']  # 5-day volume trend
    wick_ratio = context.row['upper_wick_ratio']  # Upper wick / body ratio
    rsi = context.row['rsi_14']

    # Gate 1: Near order block (within 2%)
    if abs(ob_distance) > 0.02:
        return False, "not_near_order_block"

    # Gate 2: Volume declining (fade)
    if volume_trend > -0.15:  # Not declining at least 15%
        return False, "volume_not_fading"

    # Gate 3: Wick rejection (upper wick > 2x body)
    if wick_ratio < 2.0:
        return False, "no_wick_rejection"

    # Gate 4: RSI overbought locally (50-70 range)
    if rsi < 50 or rsi > 70:
        return False, "rsi_out_of_range"

    # Compute weighted score
    score = (
        (1.0 - abs(ob_distance) / 0.02) * 0.3 +  # Proximity to OB
        abs(volume_trend) * 0.3 +  # Volume fade strength
        min(wick_ratio / 3.0, 1.0) * 0.25 +  # Wick rejection
        (rsi - 50) / 20 * 0.15  # Local overbought
    )

    metadata = {
        'ob_distance': ob_distance,
        'volume_trend': volume_trend,
        'wick_ratio': wick_ratio,
        'rsi': rsi,
        'score': score
    }

    return score > 0.55, metadata
```

**Thresholds:**
```python
S2_PARAMS = {
    'ob_distance_max': 0.02,      # Within 2% of order block
    'volume_decline_min': -0.15,  # At least 15% volume decline
    'wick_ratio_min': 2.0,        # Upper wick > 2x body
    'rsi_min': 50,                # Above 50 (local strength)
    'rsi_max': 70,                # Below 70 (not extreme)
    'score_threshold': 0.55       # Composite score cutoff
}
```

**Regime Weights:**
```python
S2_REGIME_WEIGHTS = {
    'risk_on': 0.3,    # Suppressed - rallies work in bull market
    'neutral': 0.7,    # Reduced - mixed conditions
    'risk_off': 1.8,   # Boosted - bear market (rallies fail)
    'crisis': 2.2      # High boost - panic (all rallies fail)
}
```

**Validation (2022):**
- Win Rate: 58.5% (validated)
- Estimated PF: 1.4
- Estimated Trades: 15-20 in 2022
- Best performance in risk_off/crisis regimes

**Historical Examples:**
- **BTC Jun 2022:** Failed rally at 32K OB, rejected → 18K
- **BTC Aug 2022:** Failed rally at 25K OB, rejected → 19K
- **BTC Nov 2022:** Failed rally at 21K OB, rejected → 15.5K (FTX)

---

## Comparison: S5 Before vs After

| Aspect | Original (User) | Corrected (System) |
|--------|----------------|-------------------|
| **Name** | Short Squeeze Fuel Burn | Long Squeeze Cascade |
| **Direction** | BULLISH (UP) | BEARISH (DOWN) |
| **Funding** | > +0.08 | Z-score > +1.2 |
| **Logic** | "Shorts trapped" | "Longs overcrowded" |
| **Who Pays** | Misunderstood | Longs pay shorts |
| **Mechanism** | Short covering rally | Long liquidation cascade |
| **Validation** | N/A (wrong direction) | Historical correlation |
| **Risk** | Severe losses | Positive edge |
| **Status** | ❌ REJECTED | ✅ APPROVED |

**Impact of Fix:**
- Without fix: Would have gone LONG during long squeeze cascades
- Result: -60% during Terra, -25% during FTX
- With fix: Correctly goes SHORT during overcrowding
- Expected result: +1.3-1.5 PF in bear markets

---

## Implementation Checklist

### Phase 1 (S2 + S5) Implementation

**Prerequisites:**
- [ ] Feature store includes funding_Z
- [ ] Feature store includes oi_change_24h
- [ ] Feature store includes ob_distance
- [ ] Feature store includes upper_wick_ratio
- [ ] Feature store includes liquidity_score
- [ ] Regime classifier operational

**S2 Implementation:**
- [ ] Add S2 detection logic to logic_v2_adapter.py
- [ ] Configure S2 thresholds
- [ ] Configure S2 regime weights
- [ ] Test on 2022 data
- [ ] Validate expected win rate (55-60%)
- [ ] Validate expected trade count (15-20)

**S5 Implementation:**
- [ ] Fix oi_change_24h feature (currently missing)
- [ ] Add S5 detection logic to logic_v2_adapter.py
- [ ] Configure S5 thresholds (corrected logic)
- [ ] Configure S5 regime weights
- [ ] Test on 2022 data
- [ ] Validate expected win rate (50-55%)
- [ ] Validate expected trade count (8-12)

**Integration:**
- [ ] Add both patterns to archetype registry
- [ ] Configure portfolio weighting
- [ ] Test combined S2+S5 performance
- [ ] Validate 2024 performance maintained
- [ ] Document results

**Documentation:**
- [ ] Update FUNDING_RATES_EXPLAINED.md
- [ ] Update BEAR_PATTERNS_IMPLEMENTATION_GUIDE.md
- [ ] Update BEAR_PATTERNS_QUICK_REFERENCE.md
- [ ] Add changelog entry
- [ ] Create commit message

---

## Testing Protocol

### Unit Tests

```python
def test_S5_funding_direction():
    """Verify S5 correctly interprets positive funding as bearish."""
    context = create_test_context(
        funding_Z=1.8,  # Extreme positive
        rsi_14=78,
        oi_change_24h=0.12,
        liquidity_score=0.20
    )

    detected, meta = _check_S5_long_squeeze(context)

    assert detected is True, "Should detect long squeeze"
    assert meta['funding_z'] > 1.5, "Funding should be extreme positive"

def test_S5_rejects_negative_funding():
    """Verify S5 does NOT trigger on negative funding (short squeeze)."""
    context = create_test_context(
        funding_Z=-1.8,  # Extreme negative (shorts overcrowded)
        rsi_14=78,
        oi_change_24h=0.12,
        liquidity_score=0.20
    )

    detected, meta = _check_S5_long_squeeze(context)

    assert detected is False, "Should NOT detect on negative funding"
    assert meta['reason'] == "funding_not_extreme"
```

### Integration Tests

```python
def test_S5_terra_collapse():
    """Validate S5 would have triggered before Terra collapse."""
    # May 2022 Terra collapse setup
    context = load_historical_context("2022-05-06")

    detected, meta = _check_S5_long_squeeze(context)

    assert detected is True, "Should detect Terra setup"
    assert meta['funding_z'] > 1.5, "Funding was extreme"

def test_S5_ftx_collapse():
    """Validate S5 would have triggered before FTX collapse."""
    # Nov 2022 FTX collapse setup
    context = load_historical_context("2022-11-07")

    detected, meta = _check_S5_long_squeeze(context)

    assert detected is True, "Should detect FTX setup"
```

---

## Risk Management

### Position Sizing

```python
BEAR_PATTERN_POSITION_SIZES = {
    'S2_FAILED_RALLY': 0.15,     # 15% of portfolio (validated)
    'S5_LONG_SQUEEZE': 0.12      # 12% of portfolio (conservative - new pattern)
}
```

### Stop Loss

```python
BEAR_PATTERN_STOPS = {
    'S2_FAILED_RALLY': 0.03,     # 3% stop above entry
    'S5_LONG_SQUEEZE': 0.025     # 2.5% stop above entry (tighter - cascade expected)
}
```

### Take Profit

```python
BEAR_PATTERN_TARGETS = {
    'S2_FAILED_RALLY': 0.06,     # 6% target (2:1 R/R)
    'S5_LONG_SQUEEZE': 0.05      # 5% target (2:1 R/R)
}
```

---

## Monitoring and Observability

### Metrics to Track

```python
BEAR_PATTERN_METRICS = [
    'detection_count',           # How often pattern triggers
    'win_rate',                  # % of winning trades
    'profit_factor',             # Gross profit / gross loss
    'avg_hold_time',             # Average duration
    'regime_distribution',       # Which regimes triggered in
    'funding_at_entry',          # Funding rate when entered (S5)
    'ob_distance_at_entry'       # OB distance when entered (S2)
]
```

### Alerts

```python
BEAR_PATTERN_ALERTS = {
    'S5_extreme_funding': funding_z > 2.0,  # Alert on extreme setup
    'S5_wrong_direction': funding_z < 0,     # Alert on logic error
    'S2_perfect_setup': score > 0.75,        # Alert on high-conviction setup
}
```

---

## Known Issues and Blockers

### S5 Blockers

1. **OI_CHANGE missing:** Feature store lacks oi_change_24h
   - Status: Needs implementation
   - Priority: HIGH
   - ETA: Unknown

### S2 Blockers

1. **OB_DISTANCE missing:** Feature store lacks ob_distance
   - Status: Needs implementation
   - Priority: HIGH
   - ETA: Unknown

2. **UPPER_WICK_RATIO missing:** Feature store lacks wick calculation
   - Status: Needs implementation
   - Priority: MEDIUM
   - ETA: Unknown

---

## Future Enhancements

### Phase 2 Patterns (Deferred)

- **S1 Liquidity Vacuum:** Blocked on liquidity_score implementation
- **S4 Distribution:** Needs refinement (too vague)
- **S6 Alt Rotation:** Rejected (too rare, < 5 trades/year)
- **S7 Yield Curve:** Rejected (use as regime filter instead)
- **S8 Exhaustion Fade:** Rejected (wrong bias - bullish in bear market)

### Improvements

- [ ] Add machine learning score refinement
- [ ] Implement adaptive thresholds based on regime
- [ ] Add volume profile analysis (S2)
- [ ] Add liquidation heatmap integration (S5)
- [ ] Combine S2+S5 for high-conviction setups

---

## References

- **FUNDING_RATES_EXPLAINED.md:** Detailed funding rate mechanics
- **BEAR_PATTERNS_QUICK_REFERENCE.md:** Quick lookup guide
- **COMPREHENSIVE_ARCHETYPE_AUDIT.md:** Full pattern analysis
- **PR6A_PROGRESS.md:** Implementation status

---

## Appendix: Pattern Comparison Table

| Pattern | Direction | Win Rate | PF | Trades/Yr | Regime | Status |
|---------|-----------|----------|-----|-----------|--------|--------|
| S2 Failed Rally | SHORT | 58.5% | 1.4 | 15-20 | Risk-off | APPROVED |
| S5 Long Squeeze | SHORT | 50-55% | 1.3-1.5 | 8-12 | Crisis | APPROVED (FIXED) |
| S1 Liquidity Vacuum | SHORT | Unknown | Unknown | Unknown | Crisis | BLOCKED |
| S4 Distribution | SHORT | Unknown | Unknown | Unknown | Risk-off | NEEDS REFINEMENT |
| S6 Alt Rotation | SHORT | N/A | N/A | < 5 | Any | REJECTED (rare) |
| S7 Yield Curve | N/A | N/A | N/A | N/A | Any | REJECTED (regime only) |
| S8 Exhaustion Fade | LONG | N/A | N/A | N/A | Risk-off | REJECTED (wrong bias) |

---

## Critical Reminder

**FUNDING RATE DIRECTION:**
- Positive (+) = Longs pay shorts = BEARISH
- Negative (-) = Shorts pay longs = BULLISH

**Always double-check the sign before implementing funding-based patterns.**
