# S1 Liquidity Vacuum V2 - Operator Guide

**Version**: V2 Production (2025-11-23)
**Config**: `configs/s1_v2_production.json`
**Pattern Type**: Capitulation Reversal (Long Bias)
**Target Frequency**: 40-60 trades/year
**Operating Regime**: Bear markets (risk_off/crisis)

---

## 1. What is S1?

### Pattern Description

S1 Liquidity Vacuum detects major capitulation events where orderbook liquidity evaporates during sell-offs, creating "air pockets" where sellers exhaust themselves. The resulting vacuum creates explosive short-covering bounces as there's no resistance.

**Key Characteristics**:
- Extreme liquidity drain (orderbook depth collapses)
- Panic volume spike (selling climax visible in volume)
- Deep lower wicks (sellers exhausted, buyers stepping in)
- Macro crisis environment (VIX elevated, DXY stress)
- Multi-bar capitulation dynamics (not just single candle)

### When It Fires

S1 fires during major capitulation events characterized by:
1. **20%+ drawdown** from 30-day high
2. **Crisis macro environment** (VIX/DXY/MOVE elevated)
3. **Volume climax OR wick exhaustion** (panic visible in microstructure)
4. **Multiple confirmation signals** (confluence of 3-4 conditions)
5. **Bear market context** (risk_off regime OR >10% drawdown)

### Historical Examples

| Event | Date | Context | Drawdown | Outcome |
|-------|------|---------|----------|---------|
| **LUNA Death Spiral** | May 12, 2022 | Stablecoin collapse | -80% | Violent 25% bounce in 24h |
| **LUNA Final Capitulation** | Jun 18, 2022 | Bear market bottom | -70% | Major reversal, trend change |
| **FTX Collapse** | Nov 9, 2022 | Exchange bankruptcy | -25% | Liquidity vacuum bounce |
| **Japan Carry Unwind** | Aug 5, 2024 | Global macro shock | -15% flash | Mean reversion spike |

---

## 2. Configuration

### How to Enable/Disable S1

**Enable S1 only**:
```json
{
  "archetypes": {
    "enable_S1": true,
    "enable_S2": false,  // Disable all other archetypes
    "enable_S3": false,
    // ... etc
  }
}
```

**Disable S1**:
```json
{
  "archetypes": {
    "enable_S1": false
  }
}
```

### Key Parameters and What They Control

#### Detection Mode Parameters

| Parameter | Default | Range | What It Controls |
|-----------|---------|-------|------------------|
| `use_v2_logic` | `true` | boolean | Enable V2 multi-bar capitulation detection (REQUIRED) |
| `use_confluence` | `true` | boolean | Enable confluence scoring (RECOMMENDED) |
| `use_regime_filter` | `true` | boolean | Enable regime-aware filtering (RECOMMENDED) |

#### Hard Gate Thresholds

These are **minimum requirements** - if any fails, trade rejected immediately:

| Parameter | Default | Range | Effect of Lowering | Effect of Raising |
|-----------|---------|-------|-------------------|-------------------|
| `capitulation_depth_max` | `-0.20` | -0.15 to -0.30 | More trades (detects shallower dips) | Fewer trades (only deep capitulations) |
| `crisis_composite_min` | `0.35` | 0.25 to 0.50 | More trades (lower macro stress required) | Fewer trades (only severe crises) |

#### Exhaustion Signal Thresholds

Require at least **ONE** to pass (OR gate):

| Parameter | Default | Range | Effect of Lowering | Effect of Raising |
|-----------|---------|-------|-------------------|-------------------|
| `volume_climax_3b_min` | `0.50` | 0.30 to 0.70 | More trades (catches moderate volume spikes) | Fewer trades (only extreme panics) |
| `wick_exhaustion_3b_min` | `0.60` | 0.40 to 0.80 | More trades (catches moderate wicks) | Fewer trades (only deep rejections) |

#### Confluence Parameters

After gates pass, require multiple confirmations:

| Parameter | Default | Range | What It Controls |
|-----------|---------|-------|------------------|
| `confluence_min_conditions` | `3` | 2 to 4 | Minimum number of conditions that must pass (out of 4) |
| `confluence_threshold` | `0.65` | 0.50 to 0.80 | Minimum weighted score required (0.65 = high confidence) |

**Confluence Conditions** (need 3 of 4):
1. Capitulation depth score
2. Crisis environment score
3. Volume climax score
4. Wick exhaustion score

#### Regime Filter Parameters

| Parameter | Default | Options | What It Controls |
|-----------|---------|---------|------------------|
| `allowed_regimes` | `["risk_off", "crisis"]` | Any combo of: risk_on, neutral, risk_off, crisis | Which regimes allow trading |
| `drawdown_override_pct` | `0.10` | 0.05 to 0.20 | Drawdown that bypasses regime check (10% = allow any regime if >10% drop) |
| `require_regime_or_drawdown` | `true` | boolean | If true, MUST be in allowed regime OR exceed drawdown override |

### Confluence vs Binary Mode

**Confluence Mode** (RECOMMENDED):
```json
{
  "use_confluence": true,
  "confluence_min_conditions": 3,
  "confluence_threshold": 0.65
}
```
- Requires multiple confirmation signals (3-of-4 conditions + 65% weighted score)
- Reduces false positive ratio from 236:1 to 10-15:1
- Catches 3-4 out of 7 major events (high precision, moderate recall)

**Binary Mode** (NOT RECOMMENDED):
```json
{
  "use_confluence": false
}
```
- Uses only hard gates (depth + crisis) + exhaustion (volume OR wick)
- Higher recall but MUCH higher false positive rate
- 237 trades/year vs 40-60 with confluence

### Regime Filter Behavior

**Strict Regime Filtering** (DEFAULT):
```json
{
  "use_regime_filter": true,
  "allowed_regimes": ["risk_off", "crisis"],
  "require_regime_or_drawdown": true,
  "drawdown_override_pct": 0.10
}
```
- Only trades in bear markets (risk_off) or crisis periods
- Drawdown >10% bypasses regime check (catches flash crashes in bull markets)
- 2023 (bull recovery) = 0 trades is CORRECT

**Permissive Filtering** (for testing):
```json
{
  "use_regime_filter": true,
  "allowed_regimes": ["risk_off", "crisis", "neutral"],
  "drawdown_override_pct": 0.05
}
```
- Allows trading in neutral regime
- Lower drawdown override (5%) catches smaller moves
- Expect more trades, lower win rate

**No Filtering** (NOT RECOMMENDED):
```json
{
  "use_regime_filter": false
}
```
- Trades in any regime
- High false positive rate in bull markets

---

## 3. What to Expect

### Trade Frequency

**Annual**: 40-60 trades/year
- **Bear markets** (risk_off/crisis): 1-2 trades per week
- **Bull markets** (risk_on): Near-zero trades (by design)
- **2022 (bear)**: 50-80 trades
- **2023 (bull recovery)**: 0-5 trades (CORRECT)
- **2024 (mixed)**: 10-30 trades

**Pattern**: Concentrated bursts during capitulation events, then long quiet periods.

### Win Rate and Variance

- **Win Rate**: 50-60%
- **Variance**: HIGH (big wins when right, small losses when wrong)
- **R:R**: 2-3:1 on winning trades (explosive bounces)
- **Max Drawdown**: Expect -30% to -40% during extended bear markets

**Performance Characteristics**:
- Pattern is **event-driven**, not trend-following
- Win rate varies by market phase (higher in crisis, lower in grind)
- Most profits come from 3-4 major events per year
- Small losses during false signals (stopped out quickly)

### Major Events It Should Catch

**Historical Examples** (2022-2024):

**CAUGHT** (4 of 7):
1. **LUNA Death Spiral** (May 12, 2022): -80% crash → 25% bounce in 24h
2. **LUNA Final Capitulation** (Jun 18, 2022): Final bear bottom → trend reversal
3. **FTX Collapse** (Nov 9, 2022): Exchange bankruptcy → liquidity vacuum bounce
4. **Japan Carry Unwind** (Aug 5, 2024): Global macro shock → mean reversion

**MISSED BY DESIGN** (3 of 7):
1. **SVB Bank Run** (Mar 10, 2023): Moderate event, no volume climax
2. **August Flush** (Aug 17, 2022): Mild selloff, regime uncertain
3. **September Flush** (Sep 6, 2022): Mild selloff, no crisis confirmation

**False Positive Ratio**: ~10-15:1 (10-15 signals for every true capitulation event)

### Known Edge Cases

#### FTX-Type Microstructure Breaks

**Issue**: Fast exchange collapses don't always build macro stress (VIX stays calm)
**Impact**: May miss if `crisis_composite` < 0.35 and volume/wick weak
**Why It Happened**: FTX collapsed over weekend (VIX didn't react until Monday)
**Workaround**: Confluence logic helps (depth + 2 other signals may catch it)
**FTX Outcome**: Caught by lowering `crisis_composite_min` from 0.40 to 0.35

#### Regime Classifier Lag

**Issue**: GMM regime lags by 1-2 weeks during rapid transitions
**Impact**: May miss first capitulation after bear market starts
**Why It Happens**: Regime model uses 60-day windows (slow to react)
**Workaround**: Drawdown override (>10%) bypasses regime check
**Example**: 2024 Japan carry unwind caught via drawdown override despite risk_on regime

#### 2023 Zero Trades

**Issue**: No S1 trades in 2023 despite some mini-dips
**Root Cause**: Regime stayed risk_on/neutral, no >10% drawdowns
**Assessment**: CORRECT BEHAVIOR (2023 was recovery, not true capitulations)
**Action**: None needed

---

## 4. Monitoring

### What Logs to Watch

**Entry Logs** (INFO level):
```
[S1 Liquidity Vacuum] ENTRY SIGNAL
  Time: 2022-11-09 12:00:00
  Price: $17,500
  Capitulation Depth: -0.28 (28% drawdown)
  Crisis Composite: 0.52 (high stress)
  Volume Climax: 0.68 (extreme panic)
  Wick Exhaustion: 0.45 (moderate)
  Confluence Score: 0.71 (3/4 conditions, HIGH CONFIDENCE)
  Regime: crisis
  Position Size: 2% risk
```

**What to Check**:
- Confluence score >0.65 (high confidence)
- At least 3 of 4 conditions pass
- Regime is risk_off or crisis (or drawdown >10%)
- Volume climax OR wick exhaustion >threshold

**Exit Logs** (INFO level):
```
[S1 Liquidity Vacuum] EXIT
  Time: 2022-11-11 08:00:00
  Entry Price: $17,500
  Exit Price: $19,200
  P&L: +9.7%
  Exit Reason: Time limit (72h)
  Hold Duration: 44 hours
```

**What to Check**:
- P&L matches expected R:R (2-3:1 on wins, -1:1 on losses)
- Exit reason (stop loss, time limit, trailing stop)
- Hold duration (24-72 hours typical)

### Warning Signs

| Warning Sign | What It Means | Action |
|--------------|---------------|--------|
| **>100 trades/year** | Too many false positives | Tighten confluence threshold or raise exhaustion gates |
| **Wrong regime trades** | Regime filter not working | Check regime classifier, verify `allowed_regimes` |
| **Very low confluence scores** | Borderline signals trading | Raise `confluence_threshold` to 0.70 |
| **Missing obvious bottoms** | Too conservative | Lower `confluence_min_conditions` to 2, or reduce `crisis_composite_min` |
| **Many 3h-exits** | Failing immediately | Review exhaustion thresholds, may be too loose |

### Performance Metrics to Track

**Daily Monitoring**:
- Trades per day (should be 0-1, rarely 2)
- Confluence score distribution (should be >0.65)
- Regime at entry (should be mostly risk_off/crisis)

**Weekly Monitoring**:
- Win rate (target 50-60%)
- Average R:R on wins (target 2-3:1)
- False positive count (expect 10-15:1 ratio)

**Monthly Monitoring**:
- Trades per month (target 3-5 in bear markets, 0-1 in bull)
- Major events caught vs missed (target 50-70% recall)
- Drawdown (expect -20% to -40% max in bear markets)

**Quarterly Review**:
- Annual trade frequency (target 40-60)
- Profit factor (target >1.5)
- Sharpe ratio (target >1.0)
- Compare to major capitulation events (did we catch them?)

### When to Adjust Thresholds

**Immediate Adjustment** (within 24h):
- Trade frequency >5 per day → tighten confluence or raise exhaustion gates
- Wrong regime entries (risk_on when you want bear-only) → fix `allowed_regimes`

**Weekly Adjustment** (if pattern persists):
- Trade frequency >100/year → raise `confluence_threshold` to 0.70
- Missing obvious bottoms (>50% miss rate) → lower `confluence_min_conditions` to 2

**Monthly Adjustment** (after regime change):
- Bear→Bull transition → verify regime filter active
- Bull→Bear transition → may need to lower thresholds slightly for new regime

**Quarterly Tuning** (see Tuning Guide):
- After major regime change
- If win rate drops below 45%
- If false positive ratio >20:1

---

## 5. Troubleshooting

### "Why didn't S1 fire on [event]?"

**Step 1**: Check if event was detectable
- Was drawdown >20% from 30d high?
- Was crisis_composite >0.35?
- Was there volume climax OR wick exhaustion?

**Step 2**: Check confluence scores
```python
# Run backtest with debug logging
python bin/backtest.py --config configs/s1_v2_production.json --start 2022-11-08 --end 2022-11-10 --log-level DEBUG
```

Look for log line:
```
[S1 Debug] Confluence check FAILED
  Conditions passed: 2/4 (need 3)
  Weighted score: 0.58 (need 0.65)
  Missing: volume_climax (0.42 < 0.50)
```

**Step 3**: Determine if miss was acceptable
- If event was mild/moderate → CORRECT (by design)
- If event was major but score was 0.60-0.64 → Consider lowering threshold
- If conditions were 2/4 → Consider lowering `confluence_min_conditions`

### "Too many trades"

**Symptom**: >100 trades/year, low win rate (<45%)

**Quick Fix**:
1. Raise confluence threshold: `0.65 → 0.70`
2. OR raise exhaustion gates: `volume_climax_3b_min: 0.50 → 0.60`

**Medium Fix** (if quick fix not enough):
1. Raise crisis threshold: `0.35 → 0.40`
2. AND tighten drawdown override: `0.10 → 0.15`

**Long-term Fix** (see Tuning Guide):
1. Run Optuna optimization for current market regime
2. Analyze Pareto frontier for trade frequency vs win rate
3. Select configuration matching your risk tolerance

### "Missed obvious bottom"

**Symptom**: Major capitulation event occurred, S1 didn't fire

**Diagnosis**:
1. Check if V2 features were available in dataset
   ```python
   # Verify V2 features exist
   df[['liquidity_drain_pct', 'crisis_composite', 'volume_climax_last_3b']].isna().sum()
   ```
   If features missing → need to enrich dataset first

2. Check regime filter
   ```python
   # Check regime at event time
   df.loc['2022-11-09', 'regime']  # Should be 'risk_off' or 'crisis'
   ```
   If regime was risk_on but drawdown <10% → adjust `drawdown_override_pct`

3. Check confluence scores
   ```python
   # Run backtest with debug logging (see above)
   ```
   If score was 0.60-0.64 → lower threshold
   If conditions were 2/4 → lower `confluence_min_conditions`

### "Regime filter blocking everything"

**Symptom**: S1 not firing despite events, logs show "Regime filter rejected"

**Step 1**: Verify regime classifier working
```python
# Check regime predictions
python bin/validate_regime_classifier.py --config configs/s1_v2_production.json
```

**Step 2**: Check allowed regimes
```json
{
  "allowed_regimes": ["risk_off", "crisis"]  // Should include current regime
}
```

**Step 3**: Adjust drawdown override if needed
```json
{
  "drawdown_override_pct": 0.05  // Lower to 5% to catch smaller crashes
}
```

**Step 4**: Temporarily disable for testing (NOT FOR PRODUCTION)
```json
{
  "use_regime_filter": false  // Only for diagnosis
}
```

---

## Quick Start (3 Steps)

### Step 1: Deploy Configuration
```bash
# Copy production config to active location
cp configs/s1_v2_production.json configs/active/s1_production.json

# Verify config valid
python bin/validate_config.py configs/active/s1_production.json
```

### Step 2: Run Validation Backtest
```bash
# Test on known period (2022 bear market)
python bin/backtest.py \
  --config configs/active/s1_production.json \
  --start 2022-01-01 \
  --end 2022-12-31 \
  --output results/s1_validation_2022.csv

# Should see 50-80 trades, 50-60% win rate, catch LUNA + FTX events
```

### Step 3: Enable Live Trading
```bash
# Update live trading config
python bin/update_live_config.py \
  --archetype S1 \
  --enable \
  --config configs/active/s1_production.json

# Monitor logs
tail -f logs/live_trading.log | grep "S1 Liquidity Vacuum"
```

---

## Additional Resources

- **Tuning Guide**: `docs/S1_V2_TUNING_GUIDE.md` - How to optimize thresholds
- **Known Issues**: `docs/S1_V2_KNOWN_ISSUES.md` - Edge cases and limitations
- **Feature Documentation**: `engine/strategies/archetypes/bear/liquidity_vacuum_runtime.py` - Implementation details
- **Example Config**: `configs/s1_v2_quick_fix.json` - Validated research config

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review logs for warning signs
3. Consult tuning guide for optimization
4. Review known issues document for edge cases

**Remember**: S1 is designed for rare, high-conviction events. Zero trades in bull markets is CORRECT behavior.
