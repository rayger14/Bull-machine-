# Archetype Systems Production Guide

**Version**: 1.0  
**Date**: 2025-12-04  
**Audience**: Operators, System Administrators, Trading Teams  
**Scope**: S4 (Funding Divergence), S5 (Long Squeeze), S1 V2 (Liquidity Vacuum)

---

## Executive Summary

This guide covers deployment, operation, and monitoring of three validated archetype systems that work correctly in the original Bull Machine engine:

- **S4 (Funding Divergence)**: Bear market short squeeze specialist (LONG positions), PF 2.22
- **S5 (Long Squeeze)**: Bear market long squeeze specialist (SHORT positions), PF 1.86
- **S1 V2 (Liquidity Vacuum)**: Capitulation reversal specialist (LONG positions), 60.7 trades/year

All three archetypes are **regime-aware specialists** designed for specific market conditions. They should be deployed as part of a multi-archetype portfolio, not as standalone systems.

**Production Readiness Status**:
- ✅ S1 V2: PRODUCTION READY (validated 2022-2024)
- ✅ S4: CONDITIONALLY READY (excellent in bear/volatile markets)
- ✅ S5: READY WITH CAVEATS (bear specialist, OI data gaps for 2022)

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Deployment Prerequisites](#deployment-prerequisites)
3. [Configuration Setup](#configuration-setup)
4. [Expected Behavior by Archetype](#expected-behavior-by-archetype)
5. [Monitoring and Alerts](#monitoring-and-alerts)
6. [Known Limitations](#known-limitations)
7. [Troubleshooting](#troubleshooting)
8. [Staging Recommendations](#staging-recommendations)

---

## System Overview

### S4 (Funding Divergence)

**Pattern**: Detects short squeeze opportunities when funding rates are extremely negative

**Key Characteristics**:
- Direction: LONG (betting on short squeeze reversals)
- Regime: Bear/Volatile markets (risk_off, crisis)
- Trade Frequency: 12/year in bear markets, 0-2/year in bull markets
- Performance: PF 2.22 (2022 bear), PF 2.32 (2024 volatility)

**Signal Requirements**:
- Funding z-score < -1.976σ (extreme negative, shorts overcrowded)
- Price resilience > 0.555 (price holding despite negative funding)
- Liquidity < 0.348 (thin orderbook amplifies squeeze potential)
- S4 fusion threshold > 0.7824

**Use Case**: Captures short squeezes during bear market selloffs (FTX aftermath, August 2022 squeeze)

### S5 (Long Squeeze)

**Pattern**: Detects long squeeze opportunities when funding rates are extremely positive

**Key Characteristics**:
- Direction: SHORT (betting against overleveraged longs)
- Regime: Bear market rallies (risk_off, crisis)
- Trade Frequency: 9/year in bear markets, 0-1/year in bull markets
- Performance: PF 1.86 (2022 bear), 55.6% win rate

**Signal Requirements**:
- Funding z-score > +1.5σ (extreme positive, longs overcrowded)
- RSI > 70 (overbought)
- Liquidity < 0.20 (thin orderbook)
- S5 fusion threshold > 0.45

**Use Case**: Captures failed rallies in bear markets (LUNA, 3AC, FTX continuation)

**Data Limitation**: ⚠️ OI data unavailable for 2022 - pattern validated without OI component. May perform better with OI data (2024+).

### S1 V2 (Liquidity Vacuum)

**Pattern**: Detects capitulation reversals during extreme market stress

**Key Characteristics**:
- Direction: LONG (betting on bounce from capitulation)
- Regime: Bear markets and crisis periods (risk_off, crisis)
- Trade Frequency: 40-60/year in bear markets, 0-5/year in bull markets
- Performance: 60.7 trades/year (2022-2024), captures 4 of 7 major events

**Signal Requirements** (Multi-Gate System):

*Hard Gates (ALL must pass):*
- Capitulation depth < -0.20 (20% drawdown from 30d high)
- Crisis composite > 0.35 (VIX/DXY/MOVE stress)

*Confluence Logic (3 of 4 conditions + 65% weighted score):*
- Capitulation depth score
- Crisis environment score
- Volume climax in last 3 bars (z-score > 0.50)
- Wick exhaustion in last 3 bars (ratio > 0.60)

*Regime Filter:*
- Allowed regimes: risk_off, crisis
- OR drawdown override > 10% (catches flash crashes in any regime)

**Use Case**: Captures major capitulation events (LUNA, FTX, Japan carry unwind Aug 2024)

---

## Deployment Prerequisites

### Data Requirements

**All Archetypes**:
- ✅ Price data (OHLCV, 1H timeframe)
- ✅ Feature store with 114 features
- ✅ Regime classifier model (GMM trained on macro data)

**S4 Specific**:
- ✅ Funding rate data (100% coverage required)
- ✅ Liquidity score calculation (runtime)
- ✅ S4 runtime enrichment features

**S5 Specific**:
- ✅ Funding rate data (100% coverage required)
- ✅ RSI indicator
- ✅ Liquidity score calculation
- ❌ OI data (optional but recommended, 0% coverage 2022)

**S1 V2 Specific**:
- ✅ Volume data with z-score calculation
- ✅ Macro data (VIX, DXY, MOVE)
- ✅ 30-day high/low tracking
- ✅ Wick/candle analysis

### System Dependencies

```bash
# Python 3.8+
python3 --version

# Required packages
pip install pandas numpy joblib scikit-learn

# Feature store generation
python3 bin/generate_features.py --asset BTC --start 2022-01-01 --end 2024-12-31

# Regime classifier
# Ensure models/regime_classifier_gmm.pkl exists
ls -la models/regime_classifier_gmm.pkl
```

### Configuration Files

```bash
# S4 Production Config
configs/system_s4_production.json

# S5 Production Config  
configs/system_s5_production.json

# S1 V2 Production Config
configs/s1_v2_production.json
```

---

## Configuration Setup

### Single Archetype Deployment

To deploy a single archetype (e.g., S4 only):

```json
{
  "archetypes": {
    "use_archetypes": true,
    "max_trades_per_day": 3,

    "enable_S4": true,
    "enable_S5": false,
    "enable_S1": false,

    "thresholds": {
      "funding_divergence": {
        "direction": "long",
        "fusion_threshold": 0.7824,
        "funding_z_max": -1.976,
        "resilience_min": 0.555,
        "liquidity_max": 0.348
      }
    }
  },

  "fusion": {
    "entry_threshold_confidence": 0.99
  }
}
```

**Critical**: Set `fusion.entry_threshold_confidence` to 0.99 to disable legacy tier1 fallback trades. This ensures ONLY archetype trades fire.

### Multi-Archetype Portfolio

To deploy all three archetypes together:

```json
{
  "archetypes": {
    "use_archetypes": true,
    "max_trades_per_day": 8,

    "enable_S4": true,
    "enable_S5": true,
    "enable_S1": true,

    "thresholds": {
      "funding_divergence": { ... },
      "long_squeeze": { ... },
      "liquidity_vacuum": { ... }
    },

    "routing": {
      "risk_off": {
        "weights": {
          "funding_divergence": 1.0,
          "long_squeeze": 2.2,
          "liquidity_vacuum": 1.5
        }
      },
      "crisis": {
        "weights": {
          "funding_divergence": 1.5,
          "long_squeeze": 2.5,
          "liquidity_vacuum": 2.0
        }
      }
    }
  }
}
```

**Regime Routing**:
- `risk_off` (bear market): All archetypes active, S5 highest weight
- `crisis`: All archetypes maximum weight
- `risk_on` (bull market): S4/S5 reduced or disabled, S1 reduced weight

---

## Expected Behavior by Archetype

### S4 (Funding Divergence)

**Bull Market (risk_on)**:
- Expected trades: 0-2/year ✅ **NORMAL**
- Activity level: Idle (funding rarely negative)
- Alert: "No S4 signals in bull market" = EXPECTED

**Bear Market (risk_off)**:
- Expected trades: 10-15/year
- Activity level: Active during selloffs
- Entry: Funding < -2σ + price resilience + thin liquidity

**Volatility/Crisis**:
- Expected trades: 15-20/year
- Activity level: High (optimal environment)
- Entry: Same criteria as bear, increased weight

**False Positive Triggers**:
- Funding negative but price not resilient → rejected
- Funding extreme but liquidity high → rejected
- S4 fusion < 0.7824 → rejected

### S5 (Long Squeeze)

**Bull Market (risk_on)**:
- Expected trades: 0-1/year ✅ **NORMAL**
- Activity level: Disabled (longs not overleveraged in healthy uptrend)
- Alert: "No S5 signals in bull market" = EXPECTED

**Bear Market (risk_off)**:
- Expected trades: 8-12/year
- Activity level: Active during failed rallies
- Entry: Funding > +1.5σ + RSI > 70 + thin liquidity

**Crisis**:
- Expected trades: 12-18/year
- Activity level: High (overleveraged longs squeezed fastest)

**Known Misses by Design**:
- Main crash events (LUNA May, FTX Nov): 1H timeframe lag
- Regime reversals: Loses on bottoming patterns

### S1 V2 (Liquidity Vacuum)

**Bull Market (risk_on)**:
- Expected trades: 0-5/year ✅ **NORMAL**
- Activity level: Low (capitulations rare in bulls)
- Drawdown override: Catches flash crashes > 10% in any regime

**Bear Market (risk_off)**:
- Expected trades: 40-60/year (primary environment)
- Activity level: High (1-2 per week during stress periods)
- Entry: Capitulation depth + crisis + exhaustion + confluence > 65%

**Capitulation Event Capture Rate**: 3-4 of 7 major events (57% recall)

**Known Misses by Design**:
- Mild drawdowns without volume climax (SVB Mar-10)
- Slow grinds without wick exhaustion (Aug-17, Sep-6)

---

## Monitoring and Alerts

### Using monitor_archetypes.py

```bash
# Check current conditions for all archetypes
python3 bin/monitor_archetypes.py --mode alert

# Check only S4 and S5
python3 bin/monitor_archetypes.py --mode alert --archetypes S4,S5

# Historical scan
python3 bin/monitor_archetypes.py --mode historical --start 2024-01-01 --end 2024-12-31
```

**Output**:
```
ARCHETYPE ALERT STATUS
====================================
Timestamp: 2024-12-04 12:00:00
Alert Status: ACTIVE

Active Signals: S4_FUNDING_DIVERGENCE

S4:
  Signal Active: True
  funding_z: -2.145
  resilience: 0.612
  liquidity: 0.289
```

### Monitoring Checklist

**Daily**:
- [ ] Verify feature store is updating (latest timestamp within 1H)
- [ ] Check funding rate data feed is live
- [ ] Review active archetype signals
- [ ] Monitor position count (should match expected trade frequency)

**Weekly**:
- [ ] Review regime classification accuracy
- [ ] Check archetype trade distribution by regime
- [ ] Validate liquidity score calculation
- [ ] Review stop loss hit rate (should be < 50%)

**Monthly**:
- [ ] Compare actual vs expected trade frequency
- [ ] Review PF by archetype (should be near validation results)
- [ ] Check for data quality issues (gaps, anomalies)
- [ ] Assess portfolio composition (bear/bull archetype balance)

### Key Metrics to Monitor

| Metric | S4 Target | S5 Target | S1 V2 Target |
|--------|-----------|-----------|--------------|
| **Trades/Year (Bear)** | 10-15 | 8-12 | 40-60 |
| **Trades/Year (Bull)** | 0-2 | 0-1 | 0-5 |
| **Win Rate** | 45-55% | 50-60% | 50-60% |
| **Profit Factor** | 2.0-2.4 | 1.6-2.0 | 1.4-1.8 |
| **Avg Hold Time** | 24-48h | 12-24h | 48-72h |

---

## Known Limitations

### S4 (Funding Divergence)

1. **Low Trade Frequency in Bull Markets**
   - Impact: S4 idle 50% of time (2023 H1+H2: 1 trade)
   - Mitigation: Deploy with bull-biased patterns (not yet implemented)
   - Severity: Low (expected behavior)

2. **Slippage in Volatile Markets**
   - Impact: S4 fires during high volatility (by design)
   - Mitigation: Add 2-5 bps slippage buffer in backtests
   - Severity: Moderate

3. **Sample Size Concerns**
   - Impact: Only 8 OOS trades across 12 months
   - Mitigation: Extended live testing, larger OOS window
   - Severity: Medium

### S5 (Long Squeeze)

1. **OI Data Unavailable for 2022**
   - Impact: Pattern validated without OI component
   - Mitigation: May perform better with OI data (2024+)
   - Severity: Medium

2. **Low Trade Frequency**
   - Impact: 9 trades/year - not suitable as standalone
   - Mitigation: Deploy in multi-archetype portfolio
   - Severity: Low

3. **Short Position Risk**
   - Impact: Requires margin/futures capability
   - Mitigation: Position sizing limits, tight stops
   - Severity: High (operational)

4. **1H Timeframe Lag**
   - Impact: May miss intra-hour crashes (LUNA May, FTX Nov)
   - Mitigation: Design tradeoff for lower false positives
   - Severity: Medium

### S1 V2 (Liquidity Vacuum)

1. **False Positive Rate**
   - Baseline: 236:1 (before confluence)
   - With confluence: 10-15:1 (improved but still high)
   - Mitigation: Confluence logic reduces by 95%
   - Severity: Medium

2. **Regime Dependency**
   - Impact: Concentrated in bear markets
   - Mitigation: Drawdown override for flash crashes
   - Severity: Low

3. **High Variance**
   - Impact: Big wins when right, small losses when wrong
   - Mitigation: Conservative position sizing (2% max)
   - Severity: Medium

---

## Troubleshooting

### Issue: No S4/S5 Signals in Bull Market

**Status**: ✅ EXPECTED BEHAVIOR

**Explanation**: S4/S5 are bear market specialists. Zero trades in bull markets indicates correct regime filtering.

**Action**: None required. Verify regime classification is "risk_on" or "neutral".

### Issue: S4 Firing with Positive Funding

**Status**: ❌ ERROR

**Diagnosis**:
1. Check funding_z calculation (should be < -1.976 for S4)
2. Verify funding rate data feed is correct
3. Check S4 runtime enrichment is enabled

**Action**:
```bash
# Verify funding data
python3 -c "import pandas as pd; df=pd.read_parquet('data/feature_store.parquet'); print(df['funding_z'].describe())"

# Check S4 config
grep -A 5 "funding_divergence" configs/system_s4_production.json
```

### Issue: S5 Not Firing Despite Overbought RSI

**Status**: ⚠️ CHECK ADDITIONAL CONDITIONS

**Diagnosis**:
- S5 requires ALL conditions: funding > +1.5σ AND RSI > 70 AND liquidity < 0.20
- Check if only RSI is met but funding/liquidity conditions fail

**Action**:
```bash
# Check current conditions
python3 bin/monitor_archetypes.py --mode alert --archetypes S5
```

### Issue: S1 V2 High False Positives

**Status**: ⚠️ VALIDATE CONFLUENCE LOGIC

**Diagnosis**:
1. Check confluence_threshold (should be 0.65)
2. Verify confluence_min_conditions (should be 3)
3. Confirm regime filter is active

**Action**:
```bash
# Check S1 config
grep -A 30 "liquidity_vacuum" configs/s1_v2_production.json | grep -E "(confluence|regime)"
```

### Issue: Tier1 Market Trades Firing Instead of Archetypes

**Status**: ❌ CONFIG ERROR

**Diagnosis**: Legacy fusion threshold too low (< 0.99)

**Action**:
```bash
# Fix: Set fusion.entry_threshold_confidence to 0.99
# This disables tier1 fallback trades, ensuring archetype-only operation
```

---

## Staging Recommendations

### Phase 1: Paper Trading (2-4 weeks)

**Objective**: Validate signal detection and execution logic without capital risk

**Setup**:
1. Deploy single archetype (start with S1 V2 - highest frequency)
2. Run on live data feed
3. Log all signals and hypothetical fills
4. Compare to validation results

**Success Criteria**:
- Signal frequency within expected range (±20%)
- No unexpected errors or data gaps
- Liquidity/funding/regime calculations match backtest

### Phase 2: Limited Capital (4-8 weeks)

**Objective**: Validate live execution with minimal risk

**Setup**:
1. Enable S1 V2 with 1% position sizing (50% of production)
2. Add S4 or S5 after 2 weeks if S1 performing
3. Monitor slippage, fill quality, stop loss behavior

**Success Criteria**:
- PF within 30% of validation results
- Win rate within 10pp of target
- Slippage < 5 bps per trade
- No operational issues (data, connectivity, execution)

### Phase 3: Full Production (Ongoing)

**Objective**: Scale to full portfolio allocation

**Setup**:
1. Increase position sizing to production levels (2% S4, 1.5% S5, 2% S1)
2. Enable multi-archetype portfolio
3. Implement regime-based routing

**Success Criteria**:
- Maintain target PF by archetype
- Portfolio correlation < 0.6 between archetypes
- Sharpe ratio improvement vs single archetype
- Drawdown within expected range

### Rollback Triggers

**Immediate Rollback**:
- PF < 1.0 for > 30 days
- Drawdown > 25%
- Data integrity issues (funding/liquidity)
- Operational errors (execution, connectivity)

**Phased Reduction**:
- PF 20-40% below target for > 60 days
- Win rate < 40% for > 30 trades
- False positive rate > 30:1 for S1 V2

---

## Additional Resources

**Configuration Files**:
- /configs/system_s4_production.json
- /configs/system_s5_production.json
- /configs/s1_v2_production.json

**Validation Reports**:
- /S4_PRODUCTION_READINESS_ASSESSMENT.md
- /S4_OPTIMIZATION_FINAL_REPORT.md
- /S5_DEPLOYMENT_SUMMARY.md (if exists)
- /docs/S1_V2_PRODUCTION_DEPLOYMENT_SUMMARY.md (if exists)

**Monitoring Scripts**:
- /bin/monitor_archetypes.py
- /bin/backtest_knowledge_v2.py

**Support Contacts**:
- System Architecture: [Team Lead]
- Data Pipeline: [Data Engineer]
- Trading Operations: [Operator]

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-04  
**Next Review**: 2025-01-04 (monthly review cycle)
