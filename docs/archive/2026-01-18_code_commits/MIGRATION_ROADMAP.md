# Migration Roadmap: Ghost Modules to Live v2

**Version:** 1.0.0
**Date:** 2025-12-03
**Status:** Ready for Execution
**Owner:** System Architecture Team

---

## Executive Summary

**Current Blocker:** Archetype models exist but need validation against baseline performance (Baseline-Conservative PF 3.17)

**Immediate Goal:** Determine if archetype-based strategies add value over simple drawdown-based baselines

**Long-Term Goal:** Production-ready feature store v2 with 140+ columns supporting optimized archetype models

**Critical Path:** Phase 0 (comparison) → Phase 1 (feature store v2) → Phase 2 (optimization) → Production

---

## Current State Analysis

### ✅ What's Working
1. **Baseline Models Benchmarked**
   - Conservative: Test PF 3.17, 42.9% WR, 7 trades
   - Aggressive: Test PF 2.10, 33.3% WR, 36 trades
   - Both show excellent generalization (negative overfit)

2. **Archetype Wrapper Complete**
   - `ArchetypeModel` class implemented and tested
   - BaseModel interface fully compatible
   - 5/6 tests passing (1 expected fail with synthetic data)

3. **Feature Store v1 Available**
   - 116 columns, 26,236 rows (2022-2024)
   - 97.4% valid (3 broken columns)
   - Data: `data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet`

### ❌ Current Blockers

1. **No Archetype Comparison Yet**
   - Need to run full 4-model comparison (2 baselines + 2 archetypes)
   - Answer: "Do S1/S4 beat baseline PF 3.17?"

2. **Feature Store Gaps**
   - Missing: `liquidity_score` (needed for S1, S4, S5)
   - Broken: `OI_CHANGE`, `oi_change_24h`, `oi_z` (all NaN)
   - Partial: Macro data only 2024 (missing 2022-2023)

3. **Archetype Config Compatibility**
   - S4 config exists but may need relaxed thresholds
   - Unknown if S4 will generate >0 trades on 2022-2023 data

---

## Phase 0: Immediate - Get Comparison Working

**Timeline:** TODAY - 4 hours
**Goal:** Answer "Do archetypes beat baseline PF 3.17?"
**Owner:** Agent 1 + Agent 2

### Tasks

#### Task 1: Test Archetype Configs with Real Data (1 hour)
```bash
# Test S4 archetype on existing feature store
python bin/test_archetype_model.py --config configs/s4_optimized_oos_2024.json \
  --data data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet \
  --period 2023-01-01:2023-12-31

# Expected output:
# - Number of signals generated
# - If 0 trades: thresholds too strict
# - If >0 trades: ready for comparison
```

**Acceptance Criteria:**
- [ ] S4 generates >0 trades on 2023 test period
- [ ] Signal generation confirmed on real data
- [ ] No import errors or crashes

#### Task 2: Create Relaxed S4 Config (If Needed) (30 min)
If S4 generates 0 trades, create relaxed version:

```json
{
  "archetype": "S4",
  "fusion_threshold": 0.50,  // Lowered from 0.65
  "min_fusion_score": 0.45,  // Lowered from 0.60
  "atr_stop_multiplier": 2.0,
  "volume_confirmation": false  // Disable for testing
}
```

Save as: `configs/s4_relaxed_test.json`

**Acceptance Criteria:**
- [ ] Relaxed config generates >5 trades on 2023 test period
- [ ] Trades are reasonable (not garbage signals)
- [ ] Config documented with rationale

#### Task 3: Run Full 4-Model Comparison (1.5 hours)
```bash
# Uncomment archetype models in comparison script
# Edit: examples/baseline_vs_archetype_comparison.py

# Run comparison
python examples/baseline_vs_archetype_comparison.py

# Output: results/baseline_vs_archetype_comparison.csv
# Output: results/baseline_vs_archetype_report.txt
```

**Acceptance Criteria:**
- [ ] All 4 models run successfully
- [ ] Comparison shows >0 trades for all models
- [ ] CSV and text reports generated
- [ ] Clear winner identified

#### Task 4: Decision Point - Continue or Pivot? (1 hour)
Analyze results and decide:

**If Archetypes WIN (S1 or S4 > 3.17 PF):**
- ✅ Continue to Phase 1 (feature store v2)
- ✅ Invest in archetype optimization
- Document winning archetype(s)

**If Archetypes LOSE (S1 and S4 < 3.17 PF):**
- ❌ Investigate why complexity fails
- Consider hybrid: Baseline filter + archetype confirmation
- May pivot to ML baseline instead (Phase 4)

**Acceptance Criteria:**
- [ ] Decision documented in `PHASE0_DECISION.md`
- [ ] Rationale clear (data-driven)
- [ ] Next steps identified

### Exit Criteria
- [ ] Comparison shows S4 with >0 trades
- [ ] Clear answer: archetypes beat/lose to baseline
- [ ] Decision: Worth investing in v2 or not?
- [ ] Results documented and reproducible

### Risk Mitigation
**Risk:** S4 config too strict, generates 0 trades
**Mitigation:** Create relaxed config, lower thresholds by 20%

**Risk:** Real data missing features S4 needs
**Mitigation:** Check feature availability, add runtime fallbacks

**Risk:** Results inconclusive (archetypes ~= baseline)
**Mitigation:** Run extended test (2022-2024), check regime-specific performance

---

## Phase 1: Feature Store v2 - Canonical Dataset

**Timeline:** NEXT - 1-2 days
**Goal:** Create versioned, enriched feature store that all models use
**Dependencies:** Phase 0 decision = "Continue with archetypes"

### Tasks

#### Task 1: Create Feature Store v2 Builder Script (4 hours)
**File:** `bin/build_feature_store_v2.py`

```python
"""
Build Feature Store v2 with enriched features.

Adds:
- Derived features for bear archetypes (fvg_below, ob_retest, etc.)
- Fixed OI derivatives (oi_change_24h, oi_z)
- Backfilled liquidity_score
- Swing detection (swing_high_1h, swing_low_1h)
"""
```

**Acceptance Criteria:**
- [ ] Script loads v1 parquet
- [ ] Adds 10+ derived features
- [ ] Backfills liquidity_score (26k rows)
- [ ] Fixes OI derivatives (no NaN)
- [ ] Saves as v2 parquet
- [ ] Runtime < 5 minutes

#### Task 2: Implement Derived Features (2 hours)
Add 6 derived features for bear archetypes:

```python
# fvg_below: FVG below current price
df['fvg_below'] = (df['tf1h_fvg_high'] < df['close']).astype(int)

# ob_retest: Order block retest flag
df['ob_retest'] = (
    (df['high'] >= df['tf1h_ob_low']) &
    (df['low'] <= df['tf1h_ob_high'])
).astype(int)

# rsi_divergence: RSI bearish divergence (simplified)
df['rsi_divergence'] = (
    (df['high'] > df['high'].shift(4)) &
    (df['rsi_14'] < df['rsi_14'].shift(4))
).astype(int)

# vol_fade: Volume fading
df['vol_fade'] = (
    df['volume_zscore'] < df['volume_zscore'].shift(4)
).astype(int)

# wick_ratio: Upper wick / total range
df['wick_ratio'] = (df['high'] - df['close']) / (df['high'] - df['low'])

# volume_spike: Volume z-score > 2.0
df['volume_spike'] = (df['volume_zscore'] > 2.0).astype(int)
```

**Acceptance Criteria:**
- [ ] All 6 features calculated correctly
- [ ] No NaN values introduced
- [ ] Validation tests pass

#### Task 3: Fix OI Derivatives (1 hour)
```python
# oi_change_24h: Absolute change in OI (24 hours = 24 bars)
df['oi_change_24h'] = df['oi'].diff(24)

# oi_change_pct_24h: Percentage change in OI
df['oi_change_pct_24h'] = df['oi'].pct_change(24) * 100

# oi_z: Z-score (252-hour = ~10.5 day rolling window)
df['oi_z'] = (df['oi'] - df['oi'].rolling(252).mean()) / df['oi'].rolling(252).std()

# Fill initial NaNs with 0 (first 252 hours have no z-score)
df['oi_z'] = df['oi_z'].fillna(0)
df['oi_change_24h'] = df['oi_change_24h'].fillna(0)
df['oi_change_pct_24h'] = df['oi_change_pct_24h'].fillna(0)
```

**Acceptance Criteria:**
- [ ] OI derivatives calculated correctly
- [ ] No all-NaN columns
- [ ] Distribution looks reasonable (z-score: -3 to +3 for 99% of data)

#### Task 4: Backfill Liquidity Score (2 hours)
```python
# Composite liquidity score (simplified version)
def calculate_liquidity_score(row):
    """
    Liquidity score: 0.0 (illiquid) to 1.0 (highly liquid)

    Components:
    - Volume percentile (40%)
    - ATR percentile (30%)
    - Spread proxy (30%)
    """
    vol_pctl = percentileofscore(df['volume'], row['volume']) / 100
    atr_pctl = percentileofscore(df['atr_14'], row['atr_14']) / 100
    spread_proxy = 1.0 - (row['atr_14'] / row['close'])  # Lower ATR = tighter spread

    liquidity_score = (
        0.4 * vol_pctl +
        0.3 * atr_pctl +
        0.3 * spread_proxy
    )

    return np.clip(liquidity_score, 0.0, 1.0)

df['liquidity_score'] = df.apply(calculate_liquidity_score, axis=1)
```

**Acceptance Criteria:**
- [ ] Liquidity score calculated for all 26k rows
- [ ] Range: 0.0-1.0 (no outliers)
- [ ] Distribution: median ~0.5, p90 ~0.85

#### Task 5: Schema Validation (1 hour)
```bash
# Run validation script
python bin/validate_feature_store_schema.py \
  --input data/features_mtf/BTC_1H_2022-2024_v2.parquet \
  --schema docs/FEATURE_STORE_SCHEMA_v2.md \
  --strict

# Expected output:
# ✓ 126 columns validated (116 + 10 new)
# ✓ 0 NaN values found
# ✓ All data types correct
# ✓ All ranges valid
# ✓ Timestamp continuity verified
# ✓ Logical consistency passed
```

**Acceptance Criteria:**
- [ ] All validation checks pass
- [ ] No NaN values
- [ ] Column count: 126+ (v1: 116, added: 10+)
- [ ] Schema documented

#### Task 6: Update Scripts to Use v2 (1 hour)
Update paths in:
- `examples/baseline_vs_archetype_comparison.py`
- `engine/backtesting/backtest_engine.py`
- `bin/test_archetype_model.py`

```python
# Old
DATA_PATH = 'data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet'

# New
DATA_PATH = 'data/features_mtf/BTC_1H_2022-2024_v2.parquet'
```

**Acceptance Criteria:**
- [ ] All scripts use v2 parquet by default
- [ ] Backward compatibility maintained (can still use v1)
- [ ] Documentation updated

### Exit Criteria
- [ ] v2 parquet exists with 126+ columns
- [ ] All new code uses v2 by default
- [ ] Comparison runs with archetypes generating trades
- [ ] Validation passes (no NaN, correct ranges)

### Risk Mitigation
**Risk:** Liquidity score backfill too slow
**Mitigation:** Use vectorized operations, progress bar, estimate 2-3 min runtime

**Risk:** OI data missing for 2022-2023
**Mitigation:** Accept NaN for missing data, fill with 0, document gaps

**Risk:** Feature calculations break existing tests
**Mitigation:** Run full test suite before/after, compare v1 vs v2 metrics

---

## Phase 2: Model Optimization - Beat Baseline

**Timeline:** NEXT - 2-3 days
**Goal:** Optimize archetypes to beat PF 3.17
**Dependencies:** Phase 1 complete (v2 feature store ready)

### Tasks

#### Task 1: Regime-Aware S1 Optimization (6 hours)
```bash
# Run regime-aware optimization for S1 (Liquidity Vacuum)
python bin/optimize_s1_regime_aware.py \
  --data data/features_mtf/BTC_1H_2022-2024_v2.parquet \
  --train 2022-01-01:2022-12-31 \
  --test 2023-01-01:2023-12-31 \
  --trials 100 \
  --parallel 4

# Output: configs/s1_optimized_v2.json
```

**Parameters to optimize:**
- `fusion_threshold`: [0.45, 0.75]
- `min_liquidity_score`: [0.40, 0.80]
- `atr_stop_multiplier`: [1.5, 3.0]
- `volume_confirmation`: [True, False]
- `regime_filter`: ['risk_on', 'all', 'neutral']

**Acceptance Criteria:**
- [ ] Optimization completes in <2 hours
- [ ] Best config: Test PF > 3.17
- [ ] Trade count: >5 trades on test period
- [ ] Overfit: <+1.0 (train PF - test PF)

#### Task 2: S4 Multi-Period Validation (4 hours)
Test S4 on different time periods:

```bash
# 2023 validation (recovery)
python bin/validate_archetype.py --archetype S4 --period 2023

# 2024 validation (bull run)
python bin/validate_archetype.py --archetype S4 --period 2024

# 2022 validation (bear market)
python bin/validate_archetype.py --archetype S4 --period 2022
```

**Acceptance Criteria:**
- [ ] S4 profitable on 2+ periods (PF > 1.2)
- [ ] Performance consistent across regimes
- [ ] Trade frequency acceptable (>3 trades/year)

#### Task 3: Ensemble Model Creation (4 hours)
Create portfolio combining S1 + S4:

```python
class EnsembleModel(BaseModel):
    """
    Ensemble combining S1 (Liquidity Vacuum) and S4 (Funding Divergence).

    Entry: When EITHER archetype fires
    Position size: Split 50/50 if both fire, 100% if one fires
    Exit: Individual stop losses per archetype
    """

    def __init__(self):
        self.s1 = ArchetypeModel('configs/s1_optimized_v2.json', 'S1')
        self.s4 = ArchetypeModel('configs/s4_optimized_v2.json', 'S4')

    def predict(self, bar, position=None):
        s1_signal = self.s1.predict(bar, position)
        s4_signal = self.s4.predict(bar, position)

        # Combine signals
        if s1_signal.is_entry and s4_signal.is_entry:
            # Both fire: highest confidence
            return Signal(
                direction='long',
                confidence=max(s1_signal.confidence, s4_signal.confidence),
                entry_price=bar['close']
            )
        elif s1_signal.is_entry:
            return s1_signal
        elif s4_signal.is_entry:
            return s4_signal
        else:
            return Signal(direction='hold', confidence=0.0)
```

**Acceptance Criteria:**
- [ ] Ensemble outperforms individual archetypes
- [ ] Test PF > 3.5 (beat baseline by 10%+)
- [ ] Diversification benefit visible (lower DD)

#### Task 4: Walk-Forward Validation (3 hours)
```bash
# Walk-forward validation (3 folds)
python bin/validate_walk_forward.py \
  --model ensemble \
  --folds 3 \
  --train-size 12m \
  --test-size 6m

# Output:
# Fold 1: Train 2022-01 to 2022-12, Test 2023-01 to 2023-06
# Fold 2: Train 2022-07 to 2023-06, Test 2023-07 to 2023-12
# Fold 3: Train 2023-01 to 2023-12, Test 2024-01 to 2024-06
```

**Acceptance Criteria:**
- [ ] All 3 folds profitable (PF > 1.2)
- [ ] Average test PF > 3.0
- [ ] Consistency: StdDev of test PF < 1.0

### Exit Criteria
- [ ] At least one archetype beats baseline Test PF 3.17
- [ ] Overfit acceptable (<+1.0)
- [ ] Production-ready config created
- [ ] Walk-forward validation passes

### Risk Mitigation
**Risk:** Optimization overfits to 2022 bear market
**Mitigation:** Use walk-forward validation, test on multiple regimes

**Risk:** S1/S4 both underperform baseline
**Mitigation:** Investigate why, consider pivot to ML baseline (Phase 4)

**Risk:** Ensemble doesn't beat individual models
**Mitigation:** Try weighted ensemble, voting system, or ML meta-learner

---

## Phase 3: Advanced Features - Wyckoff & Temporal

**Timeline:** LATER - 1 week
**Goal:** Add Phase 3 intelligence (Wyckoff events, temporal fusion)
**Dependencies:** Phase 2 complete (optimized archetypes ready)
**Priority:** LOW (optional enhancement)

### Tasks

#### Task 1: Backfill Wyckoff Events (8 hours)
```python
# Detect Wyckoff structural events
events = [
    'SC',  # Selling Climax
    'AR',  # Automatic Rally
    'ST',  # Secondary Test
    'Spring',  # Spring (false breakdown)
    'SOS',  # Sign of Strength
    'LPS',  # Last Point of Support
]

# Add event flags to feature store
df['wyckoff_event'] = detect_wyckoff_events(df)
df['wyckoff_event_confidence'] = calculate_event_confidence(df)
```

**Acceptance Criteria:**
- [ ] Wyckoff events detected for 2022-2024
- [ ] Event recall tested (LUNA crash, FTX collapse, June 18)
- [ ] New columns added to v3 feature store

#### Task 2: Temporal Fusion Features (6 hours)
```python
# Fibonacci time clusters
fib_periods = [21, 34, 55, 89, 144]
df['fib_cluster_score'] = calculate_fib_cluster_pressure(df, fib_periods)

# Gann cycles (30/60/90 day)
df['gann_cycle_score'] = calculate_gann_cycle_score(df)

# Time-at-level (how long price stayed near key level)
df['time_at_level'] = calculate_time_at_level(df)
```

**Acceptance Criteria:**
- [ ] Temporal features add measurable value (A/B test)
- [ ] Feature importance: temporal in top 20 features
- [ ] No performance degradation

#### Task 3: Create v3 Feature Store (2 hours)
```bash
# Build v3 with Wyckoff + temporal
python bin/build_feature_store_v3.py

# Output: data/features_mtf/BTC_1H_2022-2024_v3.parquet
# Columns: 140+ (v2: 126, added: 14+)
```

**Acceptance Criteria:**
- [ ] v3 parquet created
- [ ] 140+ columns validated
- [ ] Backward compatible with v2 scripts

#### Task 4: Re-Test Archetypes with Temporal (4 hours)
```bash
# Test S1 with temporal fusion
python bin/test_temporal_fusion.py --archetype S1 --version v3

# Expected: Marginal improvement in PF (+5-10%)
```

**Acceptance Criteria:**
- [ ] Temporal fusion shows improvement
- [ ] Event recall improved (catch LUNA, FTX, June 18)
- [ ] Production config updated

### Exit Criteria
- [ ] v3 parquet with Wyckoff + temporal features
- [ ] Temporal fusion showing measurable improvement
- [ ] Event recall improved (catch major market events)

### Risk Mitigation
**Risk:** Wyckoff event detection too subjective
**Mitigation:** Use rule-based heuristics, validate with historical examples

**Risk:** Temporal features add noise, not signal
**Mitigation:** A/B test, feature importance analysis, ablation study

---

## Phase 4: ML Baseline - Benchmark (FUTURE)

**Timeline:** FUTURE - 1 week
**Goal:** Test if ML can beat archetype + baseline
**Dependencies:** Phase 2 complete (archetype performance known)
**Priority:** MEDIUM (research project)

### Tasks

#### Task 1: Create ML Baseline (6 hours)
```python
from sklearn.ensemble import RandomForestClassifier

# Train on v2 features (126 columns)
features = [
    'rsi_14', 'adx_14', 'atr_14', 'volume_zscore',
    'tf1h_fusion_score', 'liquidity_score',
    'fvg_below', 'ob_retest', 'wick_ratio',
    # ... 117 more features
]

target = 'profitable_long_entry'  # Label: 1 if long entry profitable, 0 if not

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=50,
    class_weight='balanced'
)

rf.fit(X_train[features], y_train)
```

**Acceptance Criteria:**
- [ ] ML model trains without errors
- [ ] Feature importance analysis complete
- [ ] Cross-validation accuracy > 55%

#### Task 2: Comparison: Baseline vs Archetype vs ML (4 hours)
```bash
# Run 3-way comparison
python examples/three_way_comparison.py

# Models:
# 1. Baseline-Conservative (PF 3.17)
# 2. Ensemble (S1+S4, optimized)
# 3. ML Baseline (Random Forest)
```

**Acceptance Criteria:**
- [ ] All 3 models run on same test period
- [ ] Fair comparison (same risk, same period)
- [ ] Clear winner identified

#### Task 3: Identify Production Model (2 hours)
**Decision Matrix:**

| Model | Test PF | Complexity | Maintainability | Production Score |
|-------|---------|------------|-----------------|------------------|
| Baseline-Conservative | 3.17 | Low | High | 8/10 |
| Ensemble (S1+S4) | TBD | Medium | Medium | TBD |
| ML Baseline | TBD | High | Low | TBD |

**Acceptance Criteria:**
- [ ] Production model selected
- [ ] Rationale documented
- [ ] Deployment plan created

### Exit Criteria
- [ ] ML model in comparison framework
- [ ] Clear winner identified
- [ ] Production model selected

---

## Phase 5: Live Preparation (FUTURE)

**Timeline:** FUTURE - 2 weeks
**Goal:** Prepare for paper trading / live deployment
**Dependencies:** Production model selected (Phase 4)
**Priority:** HIGH (once model proven)

### Tasks

#### Task 1: Streaming Feature Pipeline (1 week)
```python
class StreamingFeatureEngine:
    """
    Incremental feature calculation for live trading.

    Updates v2 feature store in real-time as new 1H candles close.
    """

    def update(self, new_bar):
        # Append new bar to rolling window
        self.window.append(new_bar)

        # Recalculate features incrementally
        new_features = {
            'rsi_14': self._update_rsi(new_bar),
            'atr_14': self._update_atr(new_bar),
            'liquidity_score': self._update_liquidity_score(new_bar),
            # ... 123 more features
        }

        return new_features
```

**Acceptance Criteria:**
- [ ] Streaming pipeline runs without errors
- [ ] Latency < 1 second (bar close to signal)
- [ ] Parity with batch: streaming features == batch features

#### Task 2: Live Simulator (Paper Trading) (4 days)
```bash
# Run paper trading simulation
python bin/paper_trading.py \
  --model production_ensemble \
  --duration 30d \
  --capital 10000

# Output: Daily PnL, trades, metrics
```

**Acceptance Criteria:**
- [ ] Paper trading runs for 30 days
- [ ] Performance matches backtest (±20%)
- [ ] No execution errors

#### Task 3: Risk Management Hardening (2 days)
```python
# Add production risk limits
risk_limits = {
    'max_position_size': 0.15,  # 15% of capital
    'max_daily_loss': 0.05,  # 5% daily stop
    'max_drawdown': 0.10,  # 10% max DD
    'max_leverage': 1.0,  # No leverage
    'min_liquidity_score': 0.60,  # Require liquid markets
}
```

**Acceptance Criteria:**
- [ ] All risk limits enforced
- [ ] Kill switch tested (manual override)
- [ ] Alerting system integrated

#### Task 4: Monitoring and Alerts (2 days)
```python
# Add monitoring hooks
monitors = [
    PerformanceMonitor(),  # Track PnL, Sharpe, DD
    HealthMonitor(),  # Check API, data feed, latency
    RiskMonitor(),  # Check position size, exposure
    AnomalyMonitor(),  # Detect unusual behavior
]
```

**Acceptance Criteria:**
- [ ] Real-time monitoring dashboard
- [ ] Slack/email alerts configured
- [ ] Daily performance reports automated

### Exit Criteria
- [ ] Paper trading running for 1 month
- [ ] Performance matches backtest
- [ ] Ready for live capital

---

## Quick Wins (Next 4 Hours)

### Priority 1: Archetype Comparison (CRITICAL)
**Task:** Run 4-model comparison to answer "Do archetypes add value?"
**Time:** 2 hours
**Value:** HIGH - Determines entire strategy direction

```bash
# 1. Test S4 on real data (30 min)
python bin/test_archetype_model.py --config configs/s4_optimized_oos_2024.json

# 2. Create relaxed config if needed (30 min)
# Edit configs/s4_relaxed_test.json

# 3. Run comparison (1 hour)
python examples/baseline_vs_archetype_comparison.py
```

### Priority 2: Document Phase 0 Decision (MEDIUM)
**Task:** Create decision document with results
**Time:** 1 hour
**Value:** MEDIUM - Communicates findings to stakeholders

```markdown
# PHASE0_DECISION.md

## Results
- Baseline-Conservative: PF 3.17
- S1-LiquidityVacuum: PF X.XX
- S4-FundingDivergence: PF X.XX

## Winner
[Model name]

## Decision
[Continue to Phase 1 / Pivot to ML / Hybrid approach]

## Rationale
[Data-driven justification]
```

### Priority 3: Validate Feature Store v1 (LOW)
**Task:** Check current feature store health
**Time:** 30 min
**Value:** LOW - Nice to have before building v2

```bash
python bin/validate_feature_store_schema.py \
  --input data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet
```

### Priority 4: Plan Feature Store v2 Build (LOW)
**Task:** Draft build script outline
**Time:** 30 min
**Value:** LOW - Prep for Phase 1

---

## Risk Register

### Critical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Archetypes underperform baseline** | HIGH | MEDIUM | Pivot to ML baseline or hybrid approach |
| **S4 generates 0 trades** | MEDIUM | MEDIUM | Create relaxed config, lower thresholds |
| **OI data missing for 2022-2023** | MEDIUM | HIGH | Accept NaN, fill with 0, document gaps |
| **Liquidity score backfill breaks** | MEDIUM | LOW | Use simplified calculation, vectorized ops |
| **Optimization overfits to 2022** | HIGH | MEDIUM | Use walk-forward validation, test multiple regimes |

### Medium Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Feature store v2 too slow** | MEDIUM | LOW | Optimize with Dask, parallel processing |
| **Schema validation fails** | MEDIUM | LOW | Fix data issues, relax constraints temporarily |
| **Ensemble doesn't beat individuals** | MEDIUM | MEDIUM | Try weighted ensemble, voting, ML meta-learner |
| **Wyckoff events too subjective** | LOW | MEDIUM | Use rule-based heuristics, validate with examples |

### Low Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Temporal features add noise** | LOW | MEDIUM | A/B test, feature importance, ablation study |
| **ML baseline overfits** | LOW | MEDIUM | Use proper cross-validation, regularization |
| **Paper trading differs from backtest** | MEDIUM | LOW | Test execution logic, slippage modeling |

---

## Decision Points

### After Phase 0: Continue or Pivot?

**If Archetypes WIN (PF > 3.17):**
- ✅ Proceed to Phase 1 (feature store v2)
- ✅ Invest in optimization (Phase 2)
- ✅ Consider ensemble approach
- Document winning archetype(s) and metrics

**If Archetypes LOSE (PF < 3.17):**
- ❌ Investigate root cause:
  - Config too strict?
  - Missing features?
  - Pattern not applicable to BTC?
- Consider alternatives:
  - Hybrid: Baseline filter + archetype confirmation
  - Pivot to ML baseline (Phase 4)
  - Optimize baseline further
- Document lessons learned

**If Results INCONCLUSIVE (PF ~= 3.17):**
- Run extended test (2022-2024 full period)
- Check regime-specific performance
- Analyze trade quality (WR, avg win/loss)
- Consider ensemble or hybrid approach

### After Phase 1: Which Archetypes to Optimize?

**If S1 wins:**
- Optimize S1 parameters
- Backfill liquidity_score (critical for S1)
- Test S1 on multiple regimes

**If S4 wins:**
- Optimize S4 parameters
- Ensure OI features available
- Test S4 on multiple time periods

**If Both win:**
- Create ensemble (S1 + S4)
- Optimize ensemble weights
- Test diversification benefit

**If Neither wins:**
- Re-evaluate archetype selection
- Consider other archetypes (S2, S5, etc.)
- Or pivot to ML baseline

### After Phase 2: Add ML or Go Live?

**If Optimization succeeds (PF > 4.0):**
- ✅ Proceed to Phase 5 (live preparation)
- Skip Phase 4 (ML not needed if archetypes work)
- Focus on production deployment

**If Optimization marginal (PF 3.5-4.0):**
- Test ML baseline (Phase 4)
- Compare archetype vs ML
- Pick best performer for production

**If Optimization fails (PF < 3.5):**
- ❌ Stop archetype development
- Pivot to ML baseline (Phase 4)
- Or deploy Baseline-Conservative to production

---

## Success Metrics

### Phase 0 Success
- ✅ Comparison complete
- ✅ Decision documented
- ✅ Direction clear (continue/pivot)

### Phase 1 Success
- ✅ Feature store v2 created (126+ columns)
- ✅ All validation checks pass
- ✅ Archetypes generate >0 trades on v2

### Phase 2 Success
- ✅ Optimized model beats baseline PF 3.17
- ✅ Overfit <+1.0
- ✅ Walk-forward validation passes

### Phase 3 Success (Optional)
- ✅ Temporal features add value (+5-10% PF)
- ✅ Event recall improved
- ✅ v3 feature store ready

### Phase 4 Success (Optional)
- ✅ ML baseline trained
- ✅ 3-way comparison complete
- ✅ Production model selected

### Phase 5 Success
- ✅ Paper trading: 30 days, performance matches backtest
- ✅ Risk management tested
- ✅ Monitoring live

---

## Timeline Summary

| Phase | Duration | Cumulative | Critical Path |
|-------|----------|------------|---------------|
| **Phase 0** | 4 hours | Day 0 | ✅ CRITICAL |
| **Phase 1** | 1-2 days | Day 2 | ✅ CRITICAL |
| **Phase 2** | 2-3 days | Day 5 | ✅ CRITICAL |
| **Phase 3** | 1 week | Day 12 | ❌ Optional |
| **Phase 4** | 1 week | Day 19 | ❌ Optional |
| **Phase 5** | 2 weeks | Day 33 | ✅ CRITICAL |

**Critical Path:** Phase 0 → Phase 1 → Phase 2 → Phase 5
**Total Time (Critical Path Only):** ~3 weeks
**Total Time (All Phases):** ~5 weeks

---

## Resource Requirements

### Phase 0 (4 hours)
- 1 developer (Agent 1)
- 1 data scientist (Agent 2)
- Existing infrastructure (no new resources)

### Phase 1 (1-2 days)
- 1 data engineer
- Compute: Local machine (feature store build <5 min)
- Storage: +10 MB (v2 parquet)

### Phase 2 (2-3 days)
- 1 ML engineer
- Compute: 4-8 cores (Optuna optimization)
- Storage: +50 MB (optimization results)

### Phase 3 (1 week)
- 1 quant researcher
- Compute: Local machine
- Storage: +15 MB (v3 parquet)

### Phase 4 (1 week)
- 1 ML engineer
- Compute: 8-16 cores (Random Forest training)
- Storage: +100 MB (ML models)

### Phase 5 (2 weeks)
- 1 DevOps engineer
- 1 backend developer
- Infrastructure: VPS for paper trading ($20/month)
- Monitoring: Grafana/Prometheus stack

---

## Rollback Plan

### Phase 0 Rollback
**Trigger:** Archetypes fail comparison
**Action:**
1. Document findings
2. Deploy Baseline-Conservative to production
3. Investigate alternative approaches (ML, hybrid)

**No code changes needed - safe rollback**

### Phase 1 Rollback
**Trigger:** v2 feature store validation fails
**Action:**
1. Restore v1 parquet from backup
2. Revert script changes (git reset)
3. Continue with v1 feature store

```bash
# Restore v1
cp data/features_mtf/BTC_1H_2022-2024_backup.parquet \
   data/features_mtf/BTC_1H_2022-2024.parquet

# Revert code
git reset --hard HEAD~1
```

### Phase 2 Rollback
**Trigger:** Optimization fails to beat baseline
**Action:**
1. Use unoptimized archetype configs
2. Or deploy Baseline-Conservative
3. Document lessons learned

**No infrastructure changes - safe rollback**

### Phase 3 Rollback
**Trigger:** Temporal features degrade performance
**Action:**
1. Restore v2 feature store
2. Remove temporal feature columns
3. Continue with v2 (non-temporal)

```bash
# Restore v2
cp data/features_mtf/BTC_1H_2022-2024_v2_backup.parquet \
   data/features_mtf/BTC_1H_2022-2024_v2.parquet
```

### Phase 5 Rollback
**Trigger:** Paper trading underperforms backtest
**Action:**
1. Stop paper trading
2. Investigate execution logic (slippage, fees, latency)
3. Fix issues and restart

**Infrastructure can be paused - low cost rollback**

---

## Communication Plan

### Stakeholder Updates

**Daily (During Phase 0-2):**
- Slack: `#bull-machine-dev`
- Quick status: "Phase X in progress, Y% complete"

**Weekly:**
- Email: Project status report
- Metrics: Current PF, trade count, validation status

**Milestone:**
- Document: `PHASEX_COMPLETE.md`
- Demo: Show results, metrics, next steps

### Escalation

**If Phase 0 fails (archetypes lose):**
- Escalate to: System Architect
- Decision needed: Pivot strategy or continue investigation

**If Phase 1 fails (v2 validation fails):**
- Escalate to: Data Engineering Lead
- Decision needed: Relax validation or fix data issues

**If Phase 2 fails (optimization doesn't improve):**
- Escalate to: ML Lead
- Decision needed: Pivot to ML or deploy baseline

---

## Next Steps (Immediate)

### Today (Next 4 Hours)

1. **Run archetype comparison** (2 hours)
   ```bash
   python examples/baseline_vs_archetype_comparison.py
   ```

2. **Document results** (1 hour)
   - Create `PHASE0_DECISION.md`
   - Share findings with team
   - Decide: Continue to Phase 1 or pivot?

3. **Plan Phase 1** (1 hour)
   - Draft `bin/build_feature_store_v2.py` outline
   - Identify data gaps (OI, macro)
   - Estimate time to complete

### Tomorrow (If Phase 0 succeeds)

1. **Start Phase 1** (full day)
   - Implement feature store v2 builder
   - Add derived features
   - Fix OI derivatives
   - Run validation

2. **Test v2** (evening)
   - Run comparison on v2 data
   - Verify archetypes still work
   - Document any regressions

### Day 3-5 (If Phase 1 succeeds)

1. **Start Phase 2** (3 days)
   - Run regime-aware optimization
   - Test S4 on multiple periods
   - Create ensemble model
   - Run walk-forward validation

---

## Appendix A: File Locations

### Scripts
- Comparison: `examples/baseline_vs_archetype_comparison.py`
- Feature store builder: `bin/build_feature_store_v2.py` (to be created)
- Validation: `bin/validate_feature_store_schema.py`
- Optimization: `bin/optimize_s1_regime_aware.py`

### Data
- v1 feature store: `data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet`
- v2 feature store: `data/features_mtf/BTC_1H_2022-2024_v2.parquet` (to be created)
- Backups: `data/features_mtf/BTC_1H_*_backup.parquet`

### Configs
- S1: `configs/s1_v2_production.json`
- S4: `configs/s4_optimized_oos_2024.json`
- S4 relaxed: `configs/s4_relaxed_test.json` (to be created)

### Results
- Comparison: `results/baseline_vs_archetype_comparison.csv`
- Report: `results/baseline_vs_archetype_report.txt`
- Optimization: `results/phase2_optimization/`

### Documentation
- Architecture: `docs/GHOST_TO_LIVE_ARCHITECTURE.md`
- Schema: `docs/FEATURE_STORE_SCHEMA_v2.md`
- This roadmap: `MIGRATION_ROADMAP.md`

---

## Appendix B: Command Reference

### Phase 0 Commands
```bash
# Test archetype on real data
python bin/test_archetype_model.py --config configs/s4_optimized_oos_2024.json

# Run 4-model comparison
python examples/baseline_vs_archetype_comparison.py

# View results
cat results/baseline_vs_archetype_report.txt
```

### Phase 1 Commands
```bash
# Build feature store v2
python bin/build_feature_store_v2.py

# Validate v2
python bin/validate_feature_store_schema.py --input data/features_mtf/BTC_1H_2022-2024_v2.parquet

# Backup v1
cp data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet \
   data/features_mtf/BTC_1H_2022-2024_v1_backup.parquet
```

### Phase 2 Commands
```bash
# Optimize S1
python bin/optimize_s1_regime_aware.py --trials 100 --parallel 4

# Validate S4 multi-period
python bin/validate_archetype.py --archetype S4 --period 2023

# Walk-forward validation
python bin/validate_walk_forward.py --model ensemble --folds 3
```

### Phase 3 Commands
```bash
# Build v3 with temporal features
python bin/build_feature_store_v3.py

# Test temporal fusion
python bin/test_temporal_fusion.py --archetype S1 --version v3
```

---

## Version History

- **v1.0.0** (2025-12-03): Initial migration roadmap created
  - 5 phases defined
  - Decision points identified
  - Risk mitigation strategies documented
  - Quick wins prioritized

---

**End of Migration Roadmap**
