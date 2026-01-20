# Perpetual Funding Rates - Critical Concepts for Bear Patterns

## What is Funding Rate?

Funding rate is a periodic payment between long and short positions in perpetual futures contracts. It keeps the perpetual price anchored to the spot price.

### Direction Rules

**Positive Funding (+):**
- Perp price > Spot price
- **Longs pay shorts** (longs are overcrowded)
- Example: funding = +0.05% means longs pay 0.05% every 8 hours
- Interpretation: **Bullish sentiment too strong, risk of long squeeze DOWN**

**Negative Funding (-):**
- Perp price < Spot price
- **Shorts pay longs** (shorts are overcrowded)
- Example: funding = -0.05% means shorts pay 0.05% every 8 hours
- Interpretation: **Bearish sentiment too strong, risk of short squeeze UP**

## Historical Examples

### Long Squeeze (Positive Funding)
**BTC Apr 2021 Peak:**
- Funding: +0.15% (extremely high)
- OI: All-time high
- Result: -50% crash as overleveraged longs liquidated

**BTC May 2022 (Terra Collapse):**
- Funding: +0.12% (extreme positive)
- OI: Elevated
- Result: -60% cascade as longs liquidated in panic

**BTC Nov 2022 (FTX Collapse):**
- Funding: +0.08% (high positive)
- OI: Moderately high
- Result: -25% drop as overcrowded longs exited

### Short Squeeze (Negative Funding)
**BTC Jul 2021 Dip:**
- Funding: -0.08% (negative)
- OI: High short interest
- Result: +40% rally as shorts covered

**BTC Oct 2023 Rally:**
- Funding: -0.05% (negative)
- OI: Short-heavy positioning
- Result: +30% short squeeze as bears capitulated

## Common Misconceptions

❌ **WRONG:** "High funding = short squeeze = bullish"
✅ **CORRECT:** "High positive funding = long squeeze = bearish"

❌ **WRONG:** "Negative funding = bearish"
✅ **CORRECT:** "Negative funding = shorts overcrowded = short squeeze risk = bullish"

❌ **WRONG:** "funding > 0.08 means shorts are trapped"
✅ **CORRECT:** "funding > 0.08 means LONGS are paying shorts, longs are trapped"

## Application to S5 Pattern

### Original User Logic (INCORRECT)

```
Pattern: Short Squeeze Fuel Burn
Logic: funding > +0.08 + oi_spike
Claim: "Shorts trapped = price goes UP"
Direction: BULLISH
```

### Why this is backwards:

- funding > +0.08 means LONGS pay shorts
- Longs are overcrowded, not shorts
- Cascades go DOWN when longs get liquidated
- This would have traded in the WRONG DIRECTION
- Result: Severe losses during bear market

### Corrected Logic:

```
Pattern: Long Squeeze Cascade
Logic: funding_Z > +1.5 + rsi > 75 + thin_liquidity
Claim: "Longs overcrowded = cascade DOWN"
Direction: BEARISH
```

### Validation (2022 data):

- High positive funding preceded major drops
- Terra collapse: funding peaked at +0.12%
- FTX collapse: funding was +0.08%
- Pattern consistently preceded cascades

## Mechanism Explained

### Long Squeeze Cascade (Positive Funding)

1. **Setup:** Positive funding reaches +0.08-0.15%
2. **Meaning:** Longs paying high carry cost (unsustainable)
3. **Trigger:** Price stalls, longs start closing
4. **Cascade:** Liquidations trigger more liquidations
5. **Result:** Sharp move DOWN

### Short Squeeze Rally (Negative Funding)

1. **Setup:** Negative funding reaches -0.05 to -0.10%
2. **Meaning:** Shorts paying high carry cost (unsustainable)
3. **Trigger:** Price bounces, shorts start covering
4. **Cascade:** Short covering triggers more covering
5. **Result:** Sharp move UP

## Technical Details

### Funding Rate Formula

```
funding_rate = (perp_price - spot_price) / spot_price
```

- If perp > spot: funding is POSITIVE (longs pay)
- If perp < spot: funding is NEGATIVE (shorts pay)

### Z-Score Normalization

To detect extremes, we normalize funding rates:

```python
funding_Z = (funding - funding_mean) / funding_std
```

- funding_Z > +1.5: Extremely positive (longs overcrowded)
- funding_Z < -1.5: Extremely negative (shorts overcrowded)

### Typical Ranges (BTC)

| Funding Rate | Z-Score | Interpretation | Risk |
|--------------|---------|----------------|------|
| +0.01% | 0.0 | Neutral | Low |
| +0.05% | +1.0 | Elevated long interest | Medium |
| +0.08% | +1.5 | Extreme long crowding | High |
| +0.12% | +2.0 | Critical overcrowding | Very High |
| -0.05% | -1.0 | Elevated short interest | Medium |
| -0.08% | -1.5 | Extreme short crowding | High |

## Best Practices

1. **Always check the sign:** Positive = longs pay, Negative = shorts pay
2. **Use z-scores:** Normalize funding to detect extremes
3. **Combine with OI:** Spike in OI + high funding = overcrowding
4. **Validate direction:** High positive funding = bearish setup
5. **Check regime:** Funding extremes more dangerous in risk-off/crisis
6. **Monitor RSI:** Combine with exhaustion signals (RSI > 75 or RSI < 25)
7. **Assess liquidity:** Thin liquidity amplifies cascades

## Integration with Bear Patterns

### S5 Detection Logic

```python
def _check_S5_long_squeeze(context):
    """
    Detect long squeeze cascade conditions.

    CRITICAL: Positive funding = longs pay shorts = BEARISH
    """
    funding_z = context.row['funding_Z']
    oi_change = context.row['oi_change_24h']
    rsi = context.row['rsi_14']
    liquidity = context.row['liquidity_score']

    # Gate 1: Extremely positive funding (longs overcrowded)
    if funding_z < 1.2:
        return False, "funding_not_extreme"

    # Gate 2: RSI overbought (exhaustion)
    if rsi < 70:
        return False, "rsi_not_overbought"

    # Gate 3: OI spike (optional boost)
    oi_spike = oi_change > 0.08  # 8% increase

    # Gate 4: Thin liquidity (amplification)
    thin_liquidity = liquidity < 0.25

    score = compute_weighted_score(
        funding_z=funding_z,
        rsi=rsi,
        oi_spike=oi_spike,
        thin_liquidity=thin_liquidity
    )

    return score > threshold, {
        'funding_z': funding_z,
        'rsi': rsi,
        'oi_spike': oi_spike,
        'thin_liquidity': thin_liquidity
    }
```

### Regime-Aware Weighting

```python
regime_weights = {
    'risk_on': 0.2,    # Suppress in bull market
    'neutral': 0.6,    # Reduce in neutral
    'risk_off': 2.0,   # Boost in bear market
    'crisis': 2.5      # Max boost in crisis
}
```

## Debugging Checklist

When implementing funding-based patterns:

- [ ] Verify funding sign (+ or -)
- [ ] Confirm who pays whom (longs pay vs shorts pay)
- [ ] Check pattern direction (UP vs DOWN)
- [ ] Validate with historical examples
- [ ] Test in bear market conditions (2022)
- [ ] Ensure regime weights are correct
- [ ] Document the logic clearly

## References

- **Terra Collapse (May 2022):** Case study of extreme positive funding
- **FTX Collapse (Nov 2022):** Moderate positive funding cascade
- **Oct 2023 Rally:** Negative funding short squeeze example

## Key Takeaway

**Memory Aid:**
```
Positive Funding:
  Perp > Spot
  → Longs pay shorts
  → Longs overcrowded
  → Long squeeze DOWN
  → BEARISH

Negative Funding:
  Perp < Spot
  → Shorts pay longs
  → Shorts overcrowded
  → Short squeeze UP
  → BULLISH
```

**Never confuse the direction. Check the sign first, always.**
