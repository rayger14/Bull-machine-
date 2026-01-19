# S1 Liquidity Vacuum V2 - Known Issues and Limitations

**Version**: V2 Production (2025-11-23)
**Config**: `configs/s1_v2_production.json`
**Last Updated**: 2025-11-23

This document catalogs known edge cases, limitations, and failure modes of the S1 Liquidity Vacuum pattern. Understanding these issues helps operators set appropriate expectations and avoid misdiagnosis.

---

## Issue 1: Microstructure Breaks (FTX-Type Events)

### Description

Fast exchange collapses (hours to days) don't always build macro stress signals (VIX, DXY, MOVE) quickly enough. The pattern relies on `crisis_composite` to filter noise, but sudden shocks can occur before macro indicators react.

### Example: FTX Collapse (Nov 9, 2022)

**Timeline**:
- **Nov 8 (Tue)**: FTX solvency concerns emerge
- **Nov 9 (Wed)**: Full collapse, withdrawals halted, BTC drops -25%
- **Nov 10 (Thu)**: VIX spikes (macro finally reacts)

**Issue**: FTX collapse occurred Wed-Thu, but VIX/macro stress didn't register until Thu-Fri. The `crisis_composite` score was 0.34 on Nov 9 (below 0.40 threshold at the time).

**Impact**:
- With `crisis_composite_min=0.40`: FTX event MISSED
- With `crisis_composite_min=0.35`: FTX event CAUGHT
- Lowering threshold from 0.40 to 0.35 added ~117 signals over 2.9 years (+40 trades/year)

**Current Status**: PARTIALLY MITIGATED
- Production config uses `crisis_composite_min=0.35` (catches FTX)
- Confluence logic filters most of the +117 false signals (3-of-4 conditions + weighted score)
- Trade frequency increased from 0 to 60.7/year (acceptable)

### Workaround

**Short-term** (currently deployed):
- Keep `crisis_composite_min=0.35` to catch exchange-specific shocks
- Rely on confluence logic to filter false positives
- Expect 40-60 trades/year (up from ideal 30-40)

**Long-term** (future enhancement):
- Add exchange-specific stress indicators (e.g., FTX premium to Binance, withdrawal delays)
- Create "microstructure crisis" composite separate from macro crisis
- Add social sentiment indicators (Twitter panic, Reddit fear)

### When to Adjust

**If FTX-type events recurring** (multiple exchange collapses):
- Consider adding exchange health monitoring
- Lower `crisis_composite_min` to 0.30 (accept more trades)
- Tighten confluence to 4-of-4 conditions (compensate for lower crisis gate)

**If false positive rate too high** (>100 trades/year):
- Raise `crisis_composite_min` to 0.40 (accept risk of missing exchange-specific events)
- Accept that microstructure breaks may be missed unless confluence very strong

---

## Issue 2: Regime Classifier Lag

### Description

GMM regime classifier uses 60-day rolling windows to classify market regime (risk_on, neutral, risk_off, crisis). During rapid regime transitions (bull→bear or bear→bull), the classifier lags by 1-2 weeks.

### Example: Bear Market Start (May 2022)

**Timeline**:
- **May 5-10**: LUNA collapse begins, BTC drops -30%
- **May 12**: LUNA death spiral, extreme capitulation event
- **May 13**: Regime classifier still shows "neutral" (lag effect)
- **May 20**: Regime classifier finally updates to "risk_off"

**Issue**: First major capitulation of bear market occurred while regime classifier still in "neutral" (or even "risk_on").

**Impact**:
- If `allowed_regimes=["risk_off", "crisis"]` with no override: Event MISSED
- If `drawdown_override_pct=0.10` enabled: Event CAUGHT (30% drawdown >> 10% override)

**Current Status**: MITIGATED
- Production config uses `drawdown_override_pct=0.10`
- If drawdown >10%, bypass regime check entirely
- LUNA May-12 had -30% drawdown → override triggered → event caught

### Workaround

**Current mitigation**:
- `drawdown_override_pct=0.10` allows detection in any regime if drawdown severe
- Set to 0.15 for more conservative (only extreme crashes bypass regime)
- Set to 0.05 for more aggressive (smaller crashes bypass regime)

**Alternative**: Use faster regime indicators
- Add 7-day MA of VIX as fast regime proxy (reacts within days)
- Add funding rate extreme as crisis indicator (reacts within hours)
- Combine GMM (slow, accurate) with fast indicators (quick, noisy)

### When to Adjust

**If missing early capitulations** (first event after regime change):
- Lower `drawdown_override_pct` from 0.10 to 0.08 (more sensitive)
- Add "neutral" to `allowed_regimes` temporarily during transitions
- Monitor manually and override config during obvious regime changes

**If too many regime-edge trades** (low quality entries at regime boundaries):
- Raise `drawdown_override_pct` from 0.10 to 0.15 (more conservative)
- Remove "neutral" from `allowed_regimes` if present
- Require `require_regime_or_drawdown=true` strictly

---

## Issue 3: 2023 Zero Trades

### Description

S1 produced near-zero trades in 2023 despite some mini-dips and volatility spikes.

### Analysis

**2023 Market Characteristics**:
- Regime: Mostly risk_on (bull recovery after 2022 bear)
- No >10% drawdowns from 30d high (steady grind up)
- No sustained crisis periods (VIX ranged 12-22, mostly calm)
- Mini-dips (5-8%) but recovered quickly without building stress

**Why No Trades**:
- Regime filter: risk_on not in `allowed_regimes` (requires risk_off or crisis)
- Drawdown override: No dips >10% (override didn't trigger)
- Crisis composite: Low macro stress (VIX calm, DXY normal)
- Capitulation depth: Most dips were -5% to -8% (below -20% threshold)

### Assessment: CORRECT BEHAVIOR

**Why this is correct**:
- 2023 was NOT a bear market (recovery year)
- Mini-dips were NOT capitulation events (healthy corrections in uptrend)
- S1 designed for bear market capitulations, NOT bull market dips
- Zero trades in bull recovery is EXPECTED AND DESIRED

**What would have happened if S1 fired**:
- Buying dips in strong uptrend = good (but not S1's purpose)
- S1 optimized for EXTREME stress → overkill for normal corrections
- Other archetypes (Trap Within Trend, Order Block Retest) better suited for bull dips

### When to Be Concerned

**NOT concerning** (expected behavior):
- Zero trades in 2023 (bull recovery)
- Zero trades in strong bull markets (2024 Q1-Q3 during rally)
- Low trade frequency during low-volatility periods

**Concerning** (investigate):
- Zero trades in 2022 (confirmed bear market) → config error
- Zero trades during LUNA, FTX events → thresholds too tight
- Zero trades in risk_off regime with >20% drawdowns → regime filter broken

### Action: None Required

2023 zero-trade result validates that S1 correctly filters for bear/crisis environments. No tuning needed.

---

## Issue 4: Liquidity Paradox (V1 Issue - Fixed in V2)

### Description (Historical - Fixed)

**V1 Problem**: Single-bar liquidity scoring missed June 18, 2022 capitulation despite being major event.

**Root Cause**: June 18 was FINAL capitulation after months of selling. Orderbook had ALREADY been drained (liquidity_score was low for days). Single-bar scoring couldn't distinguish "ongoing low liquidity" from "sudden drain".

**Why V1 Missed It**:
- `liquidity_score=0.12` on June 18 (below 0.15 threshold) → REJECTED
- But liquidity had been 0.10-0.15 for entire week (persistent stress, not new drain)
- Pattern needed to detect "multi-bar sustained stress" not just "single-bar snapshot"

### V2 Solution (Current)

**Multi-bar capitulation encoding**:
1. **Liquidity drain percentage**: RELATIVE drain vs 7d avg (catches June 18 even if absolute level low)
2. **Liquidity velocity**: RATE of drain (fast drain = capitulation, slow = grind)
3. **Liquidity persistence**: Count of consecutive stress bars (sustained stress)
4. **Volume climax last 3 bars**: Peak panic in recent bars (not just current)
5. **Wick exhaustion last 3 bars**: Peak rejection in recent bars (multi-bar pattern)

**Result**: June 18, 2022 now CAUGHT
- `liquidity_drain_pct=0.35` (35% drain vs 7d avg despite low absolute level)
- `volume_climax_last_3b=0.68` (extreme volume in last 3 bars)
- `wick_exhaustion_last_3b=0.52` (deep rejection visible)
- `capitulation_depth=-0.28` (28% drawdown from 30d high)
- Confluence score: 0.71 (high confidence) → TRADE TAKEN

### Current Status: RESOLVED

V2 multi-bar features solve liquidity paradox. No further action needed.

---

## Issue 5: High False Positive Ratio (Acceptable by Design)

### Description

S1 has false positive ratio of ~10-15:1 (10-15 signals for every true major capitulation event).

### Analysis

**Why so high**:
- True major capitulations are RARE (3-4 per year in bear markets)
- Minor stress events are COMMON (30-50 per year)
- Distinguishing major from minor is HARD (similar features, different magnitude)
- Confluence reduces ratio from 236:1 (binary mode) to 10-15:1 (acceptable)

**Historical Data**:
- 2022-2024 (2.9 years): 7 major events identified
- S1 V2 (quick fix): 60.7 trades/year over period
- 60.7 trades/year * 2.9 years = ~176 signals
- 176 signals / 7 events = 25:1 ratio (if assuming trades uncorrelated with events)
- Actual: 4 events caught, 60 trades → ~15:1 ratio

**Is this acceptable?**

**YES, because**:
- Small losses on false positives (stopped out quickly with -1R)
- Large gains on true positives (explosive bounces, +3R to +5R)
- Net expectancy positive: (0.55 * 2.5R) - (0.45 * 1R) = +0.93R per trade
- Profit factor >1.5 validates ratio is acceptable

**Comparison to benchmarks**:
- Mean reversion patterns typically 5-10:1 ratio (S1 comparable)
- Trend-following patterns typically 2-5:1 ratio (S1 higher, but different use case)
- Market timing models typically 10-20:1 ratio (S1 comparable)

### When to Be Concerned

**NOT concerning**:
- False positive ratio 10-20:1 (expected for rare event detection)
- Win rate 50-60% (acceptable given R:R)
- Small losses on false signals (-1R typical)

**Concerning**:
- False positive ratio >30:1 (too noisy, tighten confluence)
- Win rate <45% (signal quality degrading)
- Large losses on false signals (-2R or worse, stops too wide)

### Mitigation Strategies

**Current** (already deployed):
- Confluence logic (3-of-4 conditions + 65% weighted score)
- Regime filter (only trade bear/crisis or >10% drawdowns)
- Exhaustion gates (require volume climax OR wick exhaustion)

**If ratio increases** (>20:1):
1. Raise confluence threshold: 0.65 → 0.70
2. Require 4-of-4 conditions instead of 3-of-4
3. Tighten exhaustion gates: volume 0.50→0.60, wick 0.60→0.70

**If ratio decreases but missing events** (<8:1 but recall <40%):
1. Lower confluence threshold: 0.65 → 0.60
2. Allow 2-of-4 conditions instead of 3-of-4
3. Lower exhaustion gates: volume 0.50→0.40, wick 0.60→0.50

### Action: Monitor, Adjust if Needed

Current 10-15:1 ratio is acceptable. Monitor quarterly. Adjust if drifts >20:1 or recall drops <40%.

---

## Issue 6: Weekend and Low-Liquidity Gaps

### Description

Crypto markets trade 24/7, but liquidity varies significantly. Weekends and holidays often have 40-50% lower liquidity, making S1 signals less reliable.

### Example: Weekend False Signals

**Pattern**:
- Friday evening: Liquidity drops (traders offline)
- Saturday: Small sell-off on low volume looks like "capitulation" to metrics
- Sunday: Rebounds (thin market overreaction, not true capitulation)

**Why it happens**:
- Volume Z-score calculated vs 24-hour baseline (includes high-liquidity weekdays)
- Weekend volume looks "elevated" relative to weekend normal, but it's just thin
- Wick ratios exaggerated (small absolute ranges create large percentage wicks)

### Current Mitigation

**Partial mitigation** (not perfect):
- Confluence logic requires multiple signals (not just volume or wick alone)
- Crisis composite gate (macro stress doesn't spike on weekends)
- Regime filter (weekends alone don't change regime)

**What doesn't help**:
- Cooldown period (12h) prevents clustering but doesn't filter bad entries
- Drawdown override (applies equally to weekend and weekday)

### Potential Solutions

**Short-term** (can implement now):
1. Add day-of-week check: penalize weekend signals
   ```python
   is_weekend = df['timestamp'].dt.dayofweek >= 5
   weekend_penalty = 0.1 if is_weekend else 0.0
   confluence_score -= weekend_penalty
   ```

2. Use day-adjusted volume Z-score (compare Sat to Sat, not Sat to Wed)
   ```python
   volume_z = (volume - volume_mean_by_dow[dow]) / volume_std_by_dow[dow]
   ```

3. Require higher confluence threshold on weekends (0.65 → 0.70)

**Long-term** (requires more work):
- Build time-of-week liquidity profile
- Add exchange-specific liquidity metrics (orderbook depth, not just score)
- Use intraday liquidity cycles (e.g., Asia vs US session)

### When to Act

**Monitor**:
- Percentage of trades on Saturday/Sunday (target <15%)
- Win rate on weekend trades vs weekday trades (should be similar)
- Check if weekend trades are false positives or legitimate

**Act if**:
- >25% of trades on weekends (overweighting low-liquidity periods)
- Weekend win rate <40% (significantly worse than weekday)
- Manual review shows most weekend trades are false

**Implementation**:
```python
# Add to config (future enhancement)
"liquidity_vacuum": {
    "use_day_of_week_filter": true,
    "weekend_confluence_penalty": 0.10,
    "weekend_min_confluence": 0.70
}
```

---

## Issue 7: Exchange-Specific Microstructure

### Description

S1 was developed primarily on Binance data. Other exchanges (Coinbase, Kraken, etc.) may have different liquidity profiles, order book dynamics, and price discovery mechanisms.

### Known Differences

**Binance vs Coinbase**:
- Binance: Higher leverage, more retail, faster moves
- Coinbase: Lower leverage, more institutional, slower moves
- Coinbase capitulations may look "milder" in metrics (less panic visible)

**Binance vs Kraken**:
- Kraken: Lower volume, wider spreads
- Kraken orderbook depth metrics not directly comparable to Binance

### Current Status

**S1 optimized for**: Binance spot/perps data
**Tested on**: Binance only (as of 2025-11-23)
**Untested on**: Coinbase, Kraken, Bybit, OKX

### Workaround

**For production deployment**:
1. Use Binance data for S1 signals (primary exchange)
2. Execute on any exchange (price discovery same, just timing differs)
3. Accept 1-2 bar lag on non-Binance exchanges

**For multi-exchange deployment**:
1. Run S1 separately on each exchange's data
2. Tune thresholds per exchange (Coinbase may need lower thresholds)
3. Use ensemble: take trade if 2+ exchanges fire

### Future Enhancement

**Exchange-specific calibration**:
- Optimize S1 separately for Binance, Coinbase, Kraken
- Create exchange-specific configs (e.g., `s1_v2_production_coinbase.json`)
- Use exchange health indicators (premium/discount, withdrawal delays)

---

## Issue 8: Feature Availability and Data Quality

### Description

S1 V2 relies on 12+ features (capitulation depth, crisis composite, liquidity metrics, volume/wick exhaustion, etc.). If features missing or poor quality, pattern detection fails silently.

### Known Feature Dependencies

**Critical features** (missing = no detection):
1. `capitulation_depth` (requires 30d high calculation)
2. `crisis_composite` (requires VIX, DXY, MOVE data)
3. `volume_climax_last_3b` (requires volume history)
4. `wick_exhaustion_last_3b` (requires OHLC data)

**Important features** (missing = degraded detection):
5. `liquidity_drain_pct` (requires liquidity_score + 7d history)
6. `funding_reversal` (requires funding rate data)
7. `oversold` (requires RSI or equivalent)

### Common Data Issues

**Issue 1: Macro data gaps**
- VIX/DXY/MOVE data may have gaps during weekends
- Crisis composite fails to calculate → all signals rejected
- **Solution**: Forward-fill macro data (use Friday close for Saturday/Sunday)

**Issue 2: Liquidity score missing**
- Older data may not have liquidity_score calculated
- Liquidity drain metrics fail → pattern detection incomplete
- **Solution**: Backfill liquidity_score or disable liquidity features for historical testing

**Issue 3: Funding rate unavailable**
- Spot markets don't have funding rates
- `funding_reversal` feature returns NaN → confluence score incomplete
- **Solution**: Use perp data, or set funding_reversal weight to 0 for spot-only testing

### Mitigation

**Runtime checks**:
```python
# Verify critical features exist
required_features = [
    'capitulation_depth', 'crisis_composite',
    'volume_climax_last_3b', 'wick_exhaustion_last_3b'
]

missing = [f for f in required_features if f not in df.columns]
if missing:
    logger.error(f"S1 DISABLED: Missing critical features {missing}")
    return None  # Disable pattern
```

**Graceful degradation**:
```python
# Optional features: use if available, skip if missing
if 'liquidity_drain_pct' in df.columns:
    confluence_score += weights['liquidity_drain'] * df['liquidity_drain_pct']
else:
    logger.warning("S1: liquidity_drain_pct missing, excluding from confluence")
```

### When to Act

**Before deployment**:
- Verify all 12 V2 features present in production data
- Run data quality checks (no NaNs, reasonable ranges)
- Test with 1-week live data before enabling real trading

**During operation**:
- Monitor logs for feature warnings
- Alert if >5% of bars have missing features
- Disable S1 if critical features unavailable

---

## Issue 9: Optimization Overfitting Risk

### Description

S1 V2 has 10+ tunable parameters. Over-optimization on historical data can lead to parameter sets that work great in-sample but fail out-of-sample.

### Warning Signs of Overfitting

**Quantitative**:
- In-sample profit factor >3.0, out-of-sample <1.5 (>50% degradation)
- In-sample win rate 70%, out-of-sample 45% (>25% drop)
- Parameters at extreme ends of search range (e.g., confluence_threshold=0.80)

**Qualitative**:
- Configuration catches every historical event perfectly (suspiciously good)
- Parameters don't make intuitive sense (e.g., crisis_min=0.25 seems too low)
- Slight parameter changes cause massive performance swings

### Prevention

**During optimization**:
1. Use walk-forward validation (multiple OOS periods)
2. Limit parameter search space (avoid extremes)
3. Regularize: penalize extreme parameter values
4. Multi-objective: optimize PF AND trade frequency (prevents overtrading)

**During validation**:
1. Check parameter sensitivity (Monte Carlo with ±10% noise)
2. Compare to "reasonable baseline" (e.g., v2 quick fix params)
3. Manual review of trades (do they make sense?)

### When Overfitting Detected

**Action**:
1. Discard overfit parameters
2. Re-optimize with:
   - Longer training period (more data)
   - Fewer parameters (Tier 1 only)
   - Stronger regularization
3. Use simpler model temporarily (revert to quick fix params)

---

## Summary: Issue Priority

### Critical (Address Immediately)

1. **Feature availability** (Issue 8): Verify all V2 features in production data
2. **Data quality** (Issue 8): Check for NaNs, gaps, outliers before deployment

### Important (Monitor and Adjust)

3. **False positive ratio** (Issue 5): Target 10-15:1, adjust if >20:1
4. **Regime lag** (Issue 2): Use drawdown override, monitor early bear market periods
5. **Weekend gaps** (Issue 6): Track weekend trade performance, filter if needed

### Moderate (Long-term Enhancements)

6. **Microstructure breaks** (Issue 1): Consider exchange-specific indicators
7. **Exchange differences** (Issue 7): Calibrate per exchange if deploying multi-exchange
8. **Optimization overfitting** (Issue 9): Use validation framework, avoid over-tuning

### Low (Expected Behavior)

9. **2023 zero trades** (Issue 3): Correct behavior, no action needed
10. **Liquidity paradox** (Issue 4): Resolved in V2, no further action

---

## Document History

| Date | Version | Changes |
|------|---------|---------|
| 2025-11-23 | 1.0 | Initial creation for S1 V2 production deployment |

## Related Documents

- **Operator Guide**: `docs/S1_V2_OPERATOR_GUIDE.md` - Deployment and monitoring
- **Tuning Guide**: `docs/S1_V2_TUNING_GUIDE.md` - How to optimize thresholds
- **Implementation**: `engine/strategies/archetypes/bear/liquidity_vacuum_runtime.py` - Feature calculations
- **Production Config**: `configs/s1_v2_production.json` - Current deployment settings
