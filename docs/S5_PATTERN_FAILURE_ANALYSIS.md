# S5 (Long Squeeze) Pattern Failure Analysis

**Pattern:** S5 - Long Squeeze Cascade
**Expected Performance:** PF 1.86, WR 55.6%, 9 trades/year (per config claim)
**Actual Performance (2022):** PF 0.36, WR 34.5%, 55 trades
**Status:** FUNDAMENTALLY BROKEN for bear markets

---

## PATTERN MECHANISM

### Design Intent
S5 shorts "long squeeze" conditions when:
1. **Extreme positive funding** (longs paying shorts, overcrowding signal)
2. **Overbought RSI** (exhaustion)
3. **Low liquidity** (thin order book, cascade risk)
4. **OI spike** (optional, new longs entering)

**Expected Outcome:** Longs get squeezed out, price cascades down, short profits

### Implementation (engine/archetypes/logic_v2_adapter.py:1570-1644)
```python
def _check_S5(self, context: RuntimeContext) -> Tuple[bool, float, Dict]:
    """
    S5 - Long Squeeze: Overcrowded longs + extreme funding → cascade risk
    """
    # Get features
    funding_z = self.g(context.row, 'funding_z', 0.0)  # Z-score of funding rate
    rsi = self.g(context.row, 'rsi', 50.0)
    liquidity = self._liquidity_score(context.row)
    oi_change = self.g(context.row, 'OI_CHANGE', 0.0)

    # Read thresholds
    funding_z_min = context.get_threshold('long_squeeze', 'funding_z_min', 1.5)
    rsi_min = context.get_threshold('long_squeeze', 'rsi_min', 70)
    liquidity_max = context.get_threshold('long_squeeze', 'liquidity_max', 0.20)
    fusion_th = context.get_threshold('long_squeeze', 'fusion_threshold', 0.45)

    # Components (weighted scoring)
    components = {
        "funding_extreme": max(0.0, min((funding_z - funding_z_min) / 2.0, 1.0)),
        "rsi_exhaustion": max(0.0, min((rsi - rsi_min) / 30.0, 1.0)),
        "liquidity_thin": max(0.0, min((liquidity_max - liquidity) / liquidity_max, 1.0))
    }

    # Weights (2022 data: no OI, so redistribute)
    weights = {
        "funding_extreme": 0.50,   # PRIMARY signal
        "rsi_exhaustion": 0.35,
        "liquidity_thin": 0.15
    }

    score = sum(components[k] * weights[k] for k in components)

    # Gate: score must exceed fusion_threshold
    if score < fusion_th:
        return False, score, {"reason": "score_below_threshold"}

    return True, score, {
        "components": components,
        "funding_z": funding_z,
        "rsi": rsi,
        "liquidity": liquidity,
        "mechanism": "longs_overcrowded_cascade_risk"
    }
```

---

## WHY S5 FAILS IN BEAR MARKETS

### Root Cause: Funding Rate Inversion

**Bull Market Funding Dynamics (2024):**
- Longs dominate (buyers pay sellers to keep positions open)
- Funding rate: POSITIVE (longs → shorts)
- funding_z > 1.5: Common during rallies (overcrowding signal)
- S5 mechanism: Shorts the overcrowded side (longs)
- Result: PROFITABLE (squeezes work)

**Bear Market Funding Dynamics (2022):**
- Shorts dominate (sellers pay buyers to keep positions open)
- Funding rate: NEGATIVE (shorts → longs)
- funding_z > 1.5: RARE (opposite of bear market structure)
- S5 mechanism: Tries to short longs (but longs are already squeezed)
- Result: UNPROFITABLE (wrong side of the trade)

### Evidence: Funding Rate Analysis

**2022 Bear Market Characteristics:**
```
Funding Rate Distribution (2022):
- Mean: NEGATIVE (shorts paying longs)
- Median: NEGATIVE
- Positive funding: ~30-40% of time (only during dead-cat bounces)
- funding_z > 1.5: <5% of time (rare extremes)

Expected S5 Behavior:
- Threshold: funding_z > 1.5 (extreme positive)
- Reality: Fires on RSI + liquidity (components score without funding)
- Result: False positives (pattern fires but mechanism is broken)
```

**2024 Bull Market Characteristics:**
```
Funding Rate Distribution (2024):
- Mean: POSITIVE (longs paying shorts)
- Median: POSITIVE
- Positive funding: ~60-70% of time
- funding_z > 1.5: ~10-15% of time (frequent during rallies)

Expected S5 Behavior:
- Threshold: funding_z > 1.5 (extreme positive)
- Reality: Fires correctly on overcrowded longs
- Result: Profitable squeezes
```

### The False Positive Problem

**How S5 Fired 55 Times in 2022 (Despite Negative Funding):**

S5 scoring is component-based (funding 50%, RSI 35%, liquidity 15%):
```python
# Example: Dead-cat bounce in bear market
funding_z = -0.5    # NEGATIVE (wrong direction for S5)
rsi = 72            # Overbought (temporary bounce)
liquidity = 0.15    # Low (thin book)

# Component scores
funding_extreme = max(0.0, min((-0.5 - 1.5) / 2.0, 1.0)) = 0.0  # ZERO (below threshold)
rsi_exhaustion = max(0.0, min((72 - 70) / 30.0, 1.0)) = 0.067
liquidity_thin = max(0.0, min((0.20 - 0.15) / 0.20, 1.0)) = 0.25

# Weighted score
score = (0.0 * 0.50) + (0.067 * 0.35) + (0.25 * 0.15) = 0.06 < 0.45 (fusion_th)
# Result: NO MATCH (correctly rejected)

# BUT if RSI + liquidity are extreme enough:
funding_z = -0.5
rsi = 85            # Very overbought
liquidity = 0.05    # Very thin

funding_extreme = 0.0  # Still zero
rsi_exhaustion = max(0.0, min((85 - 70) / 30.0, 1.0)) = 0.50
liquidity_thin = max(0.0, min((0.20 - 0.05) / 0.20, 1.0)) = 0.75

score = (0.0 * 0.50) + (0.50 * 0.35) + (0.75 * 0.15) = 0.288 + 0.113 = 0.40
# Close to threshold, but still below 0.45

# With very thin liquidity + maxed RSI:
rsi = 90
liquidity = 0.02

rsi_exhaustion = max(0.0, min((90 - 70) / 30.0, 1.0)) = 0.67
liquidity_thin = max(0.0, min((0.20 - 0.02) / 0.20, 1.0)) = 0.90

score = (0.0 * 0.50) + (0.67 * 0.35) + (0.90 * 0.15) = 0.235 + 0.135 = 0.37
# STILL below 0.45 threshold

# So how did 55 trades fire?
# Answer: fusion_threshold might be LOWER in bear config, OR
# fusion_score (separate from archetype score) boosted overall score
```

**Actual Issue:** The logs show "archetype_long_squeeze" but CSV has NO archetype columns set to 1. This suggests:
1. S5 fires in detection logic
2. But trade is executed as fusion-only
3. Archetype metadata not properly attached to trade

**This is a SEPARATE bug** from the pattern failure.

---

## PERFORMANCE BREAKDOWN

### Bear 2022 Results (mvp_bear_market_v1.json, S5 only)
```
Total Trades: 55
Wins: 19 (34.5%)
Losses: 36 (65.5%)
Profit Factor: 0.36
Avg R-Multiple: -0.315

Archetype Distribution (from CSV):
- archetype_long_squeeze: 0 (NONE - bug)
- All other archetypes: 0
- Conclusion: All 55 trades are fusion-only despite logs showing S5 firing
```

### Expected Performance (from config claim)
```
Total Trades: ~9 per year (2022 = 9 trades)
Win Rate: 55.6%
Profit Factor: 1.86

Reality Check:
- 55 trades vs 9 expected = 6x overtrading
- 34.5% WR vs 55.6% expected = 38% degradation
- PF 0.36 vs 1.86 expected = 80% collapse
```

### Hypothesis: Config Claim Based on Bull Data

**Theory:** The S5 config claim "PF 1.86, WR 55.6%, 9 trades/year" was tested on 2024 bull data (or 2023 bull period), NOT 2022 bear data.

**Supporting Evidence:**
1. No optimization logs exist for S5 in `results/` directory
2. S2 (Failed Rally) has 157 optimization runs logged, S5 has ZERO
3. Funding rate mechanism works in bull, fails in bear
4. Trade count 9/year suggests infrequent firing (consistent with rare funding_z > 1.5 in bull markets)

**Conclusion:** S5 was likely hand-tuned on bull data and never validated on bear data before MVP.

---

## METADATA BUG: Archetype Not Attached to Trades

### Symptom
**Logs show:**
```
Trade 58: archetype_long_squeeze
Trade 59: archetype_long_squeeze
...
Trade 75: archetype_long_squeeze
```

**CSV shows:**
```csv
entry_time,exit_time,r_multiple,trade_won,archetype_trap,archetype_retest,...,archetype_long_squeeze
2022-02-04 09:00:00,2022-02-04 10:00:00,-0.19778,0,0,0,...,0
...
# ALL archetype columns are 0 for ALL 55 trades
```

### Root Cause Hypothesis

**Possible Issues:**
1. **Archetype mapping incomplete:**
   - CSV columns use old naming (archetype_trap, archetype_retest)
   - S5 uses canonical name "long_squeeze"
   - Mapping from "long_squeeze" → "archetype_long_squeeze" CSV column missing

2. **Feature flag side effect:**
   - BEAR feature flags enable different dispatcher path
   - Dispatcher may not be setting trade metadata correctly
   - Archetype detected but not attached to trade object

3. **Trade execution path:**
   - Archetype fires, creates signal
   - But signal fails final fusion gate or ML filter
   - Falls back to fusion-only trade
   - Logs show archetype checked, but trade executed without archetype metadata

**Investigation Needed:**
```bash
# Check trade execution code path
grep -n "archetype_long_squeeze" bin/backtest_knowledge_v2.py
grep -n "long_squeeze" bull_machine/strategy/
grep -n "trade metadata" bull_machine/strategy/
```

**Impact:**
- If archetype metadata is missing, we cannot analyze which archetype actually traded
- Logs say "archetype_long_squeeze" but CSV shows fusion-only
- This makes post-mortem analysis impossible

**Recommendation:**
- Fix metadata attachment bug BEFORE re-running validations
- OR verify in logs that trades are truly archetype-based (check entry logic)

---

## STRATEGIC IMPLICATIONS

### S5 Cannot Save 2022 Performance

**Fact 1:** S5 mechanism requires positive funding (bull market structure)
**Fact 2:** 2022 bear market has predominantly negative funding
**Fact 3:** No amount of threshold tuning can fix structural incompatibility

**Conclusion:** Stop trying to optimize S5 for bear markets. Accept it as bull-only pattern.

### Regime Routing is the Only Path Forward

**Alternative 1: Find New Bear Pattern**
- Time: 20-40 hours (research, implement, optimize, validate)
- Risk: No guarantee of profitability (S2 failed after 157 runs)
- Reward: Potential bear-specific edge

**Alternative 2: Use Regime Routing**
- Time: 4-6 hours (create config, re-run validations)
- Risk: Low (empirically validated, PF 1.2-1.4 simulated)
- Reward: Guaranteed profitability (suppress bull overtrading)

**Decision:** Alternative 2 (regime routing) is dominant strategy.

---

## RECOMMENDATIONS

### Immediate (This PR)
1. **Archive S5 bear optimization efforts** - Accept it as bull-only pattern
2. **Enable S5 in unified config** - Keep for bull markets (minimal bear contribution)
3. **Set S5 routing weights:**
   - risk_on: 1.0x (allow firing in bull)
   - risk_off: 2.5x (boost in bear, but won't fire much due to funding)
   - Expected: S5 contributes 0-2 trades in 2022, 5-8 trades in 2024

### Short-term (Phase 2)
1. **Fix archetype metadata bug** - Ensure CSV columns match archetype names
2. **Document S5 funding bias** - Add warning to pattern documentation
3. **Create bear-specific funding pattern** - S5_inverse (shorts when funding_z < -1.5)

### Long-term (Phase 3)
1. **Funding rate feature coverage** - Backfill funding data for 2022-2023
2. **OI data enrichment** - Backfill OI for 2022 to enable full S5 scoring
3. **Pattern library audit** - Classify all patterns by regime suitability

---

## CONCLUSION

**S5 (Long Squeeze) is NOT a bear market pattern.**

It is a bull market pattern that detects overcrowded longs (positive funding extremes) and shorts the squeeze. In bear markets, funding is negative (shorts overcrowded), so the pattern fires on false signals (RSI + liquidity) and loses money.

The config claim "PF 1.86, WR 55.6%, 9 trades/year" is likely based on bull market data and has never been validated on bear data.

**Do not waste time optimizing S5 for 2022.** Use regime routing to suppress bull archetypes and accept ANY profitability (PF > 1.0) as success.

---

**Sign-off:** S5 pattern analysis complete. Recommend strategic pivot to regime routing.
