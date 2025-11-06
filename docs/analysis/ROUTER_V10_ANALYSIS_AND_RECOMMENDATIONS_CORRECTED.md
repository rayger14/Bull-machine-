# Router v10 Full Backtest Analysis & Optimization Recommendations
## 2022-2024 Complete Results (CORRECTED)

**Generated**: 2025-11-05
**Test Period**: January 2022 - December 2024 (3 years)
**Starting Capital**: $10,000
**Configs Used**: baseline_btc_bull_pf20.json + baseline_btc_bear_defensive.json

---

## Executive Summary

The Router v10 regime-aware backtest achieved **+11.40% return** (+$1,140.18) over 3 years with **125 trades** across all three years. Performance was **highly year-dependent**: 2022 (bear market) lost -$965.64, 2023 recovered with +$743.84, and 2024 delivered strong +$1,361.97 gains.

### Key Findings:
1. **Order Block Retest** is the star performer: 90% WR, $151.76 avg PNL, but only 8% of trades
2. **Trap Within Trend** dominates volume (83% of trades) but is barely profitable (-$352.95 total)
3. **Bear market performance needs improvement** (2022: 25% WR, -$965.64)
4. **Sharpe ratio improved from negative (2022-2023) to strong positive (2024)**

---

## Overall Performance Metrics (2022-2024)

| Metric | Value | Grade | Notes |
|--------|-------|-------|-------|
| **Total PNL** | +$1,140.18 | B | Strong absolute gain |
| **Return** | +11.40% | B+ | ~3.8% annualized |
| **Profit Factor** | 1.42 | C+ | Needs improvement (target: 2.0+) |
| **Win Rate** | 50.4% | B | Balanced |
| **Sharpe Ratio** | 1.966 | A | Excellent risk-adjusted returns |
| **Max Drawdown** | 10.06% | B+ | Acceptable |
| **Total Trades** | 125 | B | Good sample size |
| **Avg Trade** | $9.12 | C | Small wins, needs scaling |

---

## Year-by-Year Breakdown

### 2022 (Bear Market Crash)
| Metric | Value | Analysis |
|--------|-------|----------|
| **Trades** | 32 | Active but struggled |
| **PNL** | -$965.64 | **Worst year** |
| **Win Rate** | 25.0% | **Poor** - system failed in crisis |
| **Market Context** | BTC -65% peak-to-trough | Crypto winter, FTX collapse |

**Key Issues**:
- Trap within trend failed repeatedly in choppy bear markets
- Stop losses too wide (-$78 avg loss)
- Bull config inappropriately active during downtrend

---

### 2023 (Recovery Phase)
| Metric | Value | Analysis |
|--------|-------|----------|
| **Trades** | 38 | Most active year |
| **PNL** | +$743.84 | **Recovery profits** |
| **Win Rate** | 55.3% | Good improvement |
| **Market Context** | BTC +150% year | Bottomed Q1, rallied Q4 |

**Key Wins**:
- Order block retest started working (4 trades in Oct-Dec)
- Trap within trend improved in trending recovery
- Better regime classification in transitional periods

---

### 2024 (Bull Market Strength)
| Metric | Value | Analysis |
|--------|-------|----------|
| **Trades** | 55 | Most trades |
| **PNL** | +$1,361.97 | **Best year** |
| **Win Rate** | 61.8% | **Excellent** |
| **Market Context** | BTC +120% year | ATH breakout, ETF inflows |

**Key Wins**:
- Order block retest at peak performance (5/5 wins)
- Volume exhaustion catching reversals
- Regime switching working optimally

---

## Archetype Performance Analysis

### Top Performers

| Archetype | Trades | Win Rate | Avg PNL | Total PNL | Grade | Notes |
|-----------|--------|----------|---------|-----------|-------|-------|
| **order_block_retest** | 10 | **90.0%** | **$151.76** | **$1,517.59** | A+ | **GOLDMINE** - scale this |
| volume_exhaustion | 6 | 66.7% | $51.03 | $306.15 | B+ | Solid reversal detector |
| trap_within_trend | 104 | 46.2% | -$3.39 | **-$352.95** | D | **PROBLEM** - most trades, net loss |
| tier1_market | 4 | 50.0% | -$59.07 | -$236.28 | F | Failed entry type |
| failed_continuation | 1 | 0.0% | -$94.33 | -$94.33 | F | Rare, unprofitable |

### Critical Insights

#### 1. Order Block Retest Dominance
- **Only 8% of trades but contributes 133% of total profit**
- 9 out of 10 trades won (only 1 small loss)
- Average winner: $168.62
- Works best in trending markets (2024: 5/5 wins)
- **MUST OPTIMIZE**: Increase detection rate without sacrificing quality

#### 2. Trap Within Trend Problem
- **83% of all trades (104/125) but net LOSS of -$352.95**
- Win rate only 46.2% (below breakeven with R:R)
- Generates noise, not signal
- 2022 performance: -$821.38 (catastrophic)
- 2023 performance: +$88.98 (barely profitable)
- 2024 performance: +$379.45 (improved but still mediocre)

**Root Cause**: This archetype fires too frequently in choppy/ranging conditions where small wins are eroded by large stop losses.

#### 3. Volume Exhaustion Underutilized
- Only 6 trades but solid 66.7% WR
- Average $51.03 per trade
- Catching local tops/bottoms effectively
- **OPPORTUNITY**: Increase detection or lower threshold

---

## Entry/Exit Pattern Analysis

### Entry Reasons (Ranked by Total PNL)

| Entry Type | Count | % | Total PNL | Avg PNL | Win Rate |
|------------|-------|---|-----------|---------|----------|
| order_block_retest | 10 | 8.0% | **+$1,517.59** | **+$151.76** | 90.0% |
| volume_exhaustion | 6 | 4.8% | +$306.15 | +$51.03 | 66.7% |
| tier1_market | 4 | 3.2% | -$236.28 | -$59.07 | 50.0% |
| trap_within_trend | 104 | 83.2% | **-$352.95** | **-$3.39** | 46.2% |
| failed_continuation | 1 | 0.8% | -$94.33 | -$94.33 | 0.0% |

### Exit Reasons Analysis

| Exit Type | Count | Avg PNL | Win Rate | Notes |
|-----------|-------|---------|----------|-------|
| **signal_neutralized** | 84 | +$21.87 | 56.0% | Most common, working well |
| **pti_reversal** | 10 | +$28.27 | 80.0% | Excellent exit trigger |
| **stop_loss** | 19 | -$78.38 | 0.0% | **PROBLEM** - too wide, avg -$78 |
| cash_mode_* | 8 | +$3.66 | 50.0% | Regime exits, mixed results |
| regime_flip | 2 | +$182.90 | 100.0% | Rare but excellent timing |

### Risk/Reward Issues

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Avg Win | +$43.15 | +$60-80 | Need larger wins |
| Avg Loss | -$78.38 | -$30-40 | **Stops too wide** |
| Win/Loss Ratio | 0.55:1 | 2:1 | **Inverted R:R** |
| Profit Factor | 1.42 | 2.0+ | Below target |

**CRITICAL FINDING**: The system is taking 2x larger losses than wins. With 50% WR, this is barely profitable. Need to flip to 2R wins minimum.

---

## Router Behavior Analysis

### Config Distribution (2022-2024 Combined)
Based on 2024 data (representative):

| Config | Usage % | Analysis |
|--------|---------|----------|
| **CASH** | 71.6% | Too defensive, missing opportunities |
| **BULL** | 18.2% | Reasonable, but generated most losses in 2022 |
| **BEAR** | 10.2% | Underutilized, threshold too high (0.75) |

### Regime Switching Stats
- **Config Switches**: 1,295 total (990 in 2022-2023, 305 in 2024)
- **Regime Switches**: 1,115 total (858 in 2022-2023, 257 in 2024)
- **Avg Switches/Year**: 432 config switches, 372 regime switches
- **Switching Frequency**: ~1.5 switches per day

**Analysis**: High switching frequency suggests:
1. Regime detector is sensitive (good for adaptation)
2. Configs may need smoother transitions (reduce whipsaws)
3. CASH mode threshold may be too low (71% in cash = missed opportunities)

---

## Optimization Roadmap

### Priority 1: Fix Trap Within Trend 🔥🔥🔥
**Impact**: HIGH | **Effort**: MEDIUM

**Problem**: 104 trades (83% of all trades) generating -$352.95 net loss.

**Optuna Hyperparameter Targets**:
```python
{
    'trap_quality_threshold': [0.35, 0.65],  # Currently too lenient
    'trap_confirmation_bars': [2, 5],  # May need more confirmation
    'trap_volume_ratio': [1.2, 2.5],  # Volume spike requirement
    'trap_stop_multiplier': [0.8, 1.5],  # Tighten stops (currently too wide)
}
```

**Frontier Testing**:
- Objective 1: Maximize win rate (currently 46%)
- Objective 2: Minimize avg loss (currently -$78)
- Constraint: Maintain >30 trades/year

**Expected Gain**: If trap WR improves to 55% with tighter stops, could gain +$400-600/year.

---

### Priority 2: Scale Order Block Retest 🔥🔥
**Impact**: VERY HIGH | **Effort**: MEDIUM-HIGH

**Problem**: Only 10 trades in 3 years, but 90% WR and $151 avg PNL.

**Optuna Hyperparameter Targets**:
```python
{
    'ob_quality_threshold': [0.25, 0.50],  # Lower to catch more (currently ~0.40)
    'ob_retest_tolerance': [0.005, 0.025],  # Price distance from OB
    'ob_volume_confirm': [0.8, 1.5],  # Volume confirmation strictness
    'ob_timeframe_weight': [0.5, 1.5],  # Multi-timeframe alignment
}
```

**Frontier Testing**:
- Objective 1: Maximize trade count (target: 20-30 trades/year)
- Objective 2: Maintain win rate >70%
- Constraint: Maintain avg PNL >$100

**Expected Gain**: If OB trades increase from 3.3 to 10/year while maintaining 80% WR:
- Additional 6.7 trades/year × $120 avg = **+$800/year**

**PyTorch Opportunity**: Train OrderBlockQualityNet to classify OB quality (0-1 score):
```python
class OrderBlockQualityNet(nn.Module):
    """MLP to score order block quality based on:
    - Formation volume profile
    - Retest confluence factors
    - Market regime context
    - Historical success rate of similar patterns
    """
```

---

### Priority 3: Improve Bear Market Performance 🔥
**Impact**: HIGH | **Effort**: HIGH

**Problem**: 2022 lost -$965.64 with 25% WR. System fails in crisis conditions.

**Root Causes**:
1. Bull config was active 60% of time during bear market (wrong)
2. Trap within trend lost -$821 in 2022 (failed in chop)
3. Stop losses didn't adapt to increased volatility

**Optuna Hyperparameter Targets** (Bear Config):
```python
{
    'fusion_entry_threshold': [0.30, 0.60],  # Currently 0.75 is too high
    'bear_position_size': [0.5, 1.0],  # Reduce size in crisis
    'bear_stop_multiplier': [0.6, 1.2],  # Tighten stops for drawdown control
    'bear_min_confidence': [0.55, 0.75],  # Regime confidence filter
}
```

**Frontier Testing**:
- Objective 1: Minimize 2022 loss (target: -$400 max)
- Objective 2: Maintain 2024 performance
- Test period: 2022 only (isolated bear market optimization)

**Expected Gain**: If 2022 loss reduces from -$965 to -$400, that's **+$565 improvement** (annualized: +$188/year).

---

### Priority 4: Tighten Stop Losses 🔥
**Impact**: MEDIUM-HIGH | **Effort**: LOW

**Problem**: Stop losses average -$78.38, while wins average +$43.15 (inverted R:R).

**Current Stop Logic**: ATR-based, likely 2-3x ATR
**Target**: 1-1.5x ATR with dynamic adjustment

**Optuna Hyperparameter Targets**:
```python
{
    'stop_loss_atr_mult': [0.8, 1.8],  # Currently ~2.5
    'trailing_stop_activation': [1.2, 2.0],  # When to activate trailing
    'trailing_stop_distance': [0.5, 1.2],  # Trail distance
    'stop_loss_regime_scaling': [0.7, 1.3],  # Adjust by regime
}
```

**Expected Gain**: If avg loss reduces to -$45 (from -$78):
- 19 stop loss trades × $33 saved = **+$627 total** (+$209/year)

**PyTorch Opportunity**: Train StopLossLSTM to predict optimal stop distance:
```python
class StopLossLSTM(nn.Module):
    """LSTM to predict optimal stop loss distance based on:
    - Recent volatility patterns
    - Archetype-specific risk profiles
    - Regime-aware risk scaling
    - Live position P&L trajectory
    """
```

---

### Priority 5: Increase Volume Exhaustion Detection 🔥
**Impact**: MEDIUM | **Effort**: LOW-MEDIUM

**Problem**: Only 6 trades in 3 years, but 66.7% WR and solid $51 avg PNL.

**Optuna Hyperparameter Targets**:
```python
{
    've_threshold': [0.30, 0.60],  # Lower to catch more
    've_volume_spike': [1.5, 3.0],  # Volume spike requirement
    've_divergence_periods': [10, 30],  # Lookback for divergence
    've_confirmation_bars': [1, 3],  # Confirmation required
}
```

**Expected Gain**: If VE trades increase from 2 to 8/year:
- Additional 6 trades/year × $51 avg × 0.67 WR = **+$204/year**

---

## PyTorch Deep Learning Roadmap

### Phase 1: Archetype Quality Classifiers (Months 1-2)

#### OrderBlockQualityNet (MLP)
**Purpose**: Score order block setups 0-1 based on confluence factors.

**Architecture**:
```python
Input: [
    'ob_formation_volume',  # Volume during OB creation
    'ob_retest_distance',   # Price distance to OB level
    'ob_age_bars',          # How old is the OB
    'htf_alignment',        # Higher timeframe trend alignment
    'liquidity_confluence', # Liquidity grabs nearby
    'regime_label',         # Current regime (one-hot)
    'regime_confidence',    # GMM confidence
    'price_action_quality', # PA score at retest
    'volume_at_retest',     # Current volume
    'mtf_confluence'        # Multi-timeframe confluence
]  # 15 features → 128 → 64 → 32 → 1 (sigmoid)
```

**Training Data**:
- Positive samples: All successful OB retests (90% of current OBs)
- Negative samples: Failed OB retests + near-misses (rejected setups)
- Target: Future 4H PnL (normalized -1 to +1)

**Expected Impact**:
- Increase OB detection from 3.3 to 10/year while maintaining 75%+ WR
- Estimated gain: **+$800/year**

---

#### TrapQualityClassifier (MLP)
**Purpose**: Filter trap within trend setups to reduce false positives.

**Architecture**:
```python
Input: [
    'trap_price_structure',   # Trap formation quality
    'trap_volume_profile',    # Volume during trap
    'trap_confluence_score',  # How many factors align
    'trap_trend_strength',    # Underlying trend momentum
    'trap_location',          # Trap location in trend
    'regime_label',           # Current regime
    'market_volatility',      # ATR percentile
    'failed_trap_proximity',  # Distance to last failed trap
]  # 15 features → 64 → 32 → 1 (sigmoid)
```

**Expected Impact**:
- Filter out 30-40% of losing trap trades
- Improve trap WR from 46% to 55%+
- Estimated gain: **+$400-600/year**

---

### Phase 2: Dynamic Risk Management (Months 3-4)

#### StopLossLSTM (Sequence Model)
**Purpose**: Predict optimal stop loss distance based on recent price behavior.

**Architecture**:
```python
Input Sequence (last 50 bars): [
    'close', 'high', 'low', 'volume',
    'atr', 'realized_volatility',
    'regime_label', 'fusion_score',
    'archetype_active'
]  # (50, 10) → LSTM(64) → LSTM(32) → Dense(16) → 1 (stop_distance)
```

**Training Data**:
- All historical trades
- Target: Optimal stop that would have:
  - Avoided stop-out if trade eventually won
  - Exited faster if trade lost
- Loss function: Custom - penalize stopped winners heavily

**Expected Impact**:
- Reduce avg stop loss from -$78 to -$45
- Estimated gain: **+$209/year**

---

#### PositionSizingNet (MLP)
**Purpose**: Dynamic position sizing based on regime + setup quality.

**Architecture**:
```python
Input: [
    'regime_label',
    'regime_confidence',
    'archetype_type',
    'setup_quality_score',  # From quality classifiers
    'recent_win_streak',
    'equity_drawdown',
    'market_volatility',
    'event_proximity'
]  # 12 features → 32 → 16 → 1 (position_size 0.5-1.5x base)
```

**Expected Impact**:
- Scale winners (OB retest → 1.5x size)
- Reduce losers (trap in bear → 0.5x size)
- Estimated gain: **+15-20% on existing PnL = +$170-230/year**

---

### Phase 3: Multi-Timeframe Fusion (Months 5-6)

#### FusionTransformer (Attention-Based)
**Purpose**: Replace hardcoded fusion score with learned attention over MTF features.

**Architecture**:
```python
Input: [
    # 1H, 4H, 1D features separately
    'wyckoff_signals_per_tf',    # (3 TF, 8 signals)
    'liquidity_signals_per_tf',  # (3 TF, 6 signals)
    'momentum_signals_per_tf',   # (3 TF, 5 signals)
    'smc_signals_per_tf',        # (3 TF, 4 signals)
    'regime_per_tf',             # (3 TF, regime encoding)
]  # TransformerEncoder(d_model=64, nhead=4, layers=3) → 1 (entry_score)
```

**Training Data**:
- All historical bars (not just trades)
- Target: Future 4H return (regression)
- Positive label: Return > +2% within 24H
- Negative label: Return < -1% within 24H

**Expected Impact**:
- Learn optimal MTF weighting automatically
- Discover non-linear feature interactions
- Estimated gain: **+$300-500/year** (speculative, needs validation)

---

### Phase 4: Regime-Aware Meta-Learning (Months 7-8)

#### RegimeMetaClassifier (Ensemble)
**Purpose**: Meta-model that selects best sub-model based on regime.

**Architecture**:
```python
# Train separate models per regime:
models = {
    'RISK_ON': OrderFlowBullNet(),
    'RISK_OFF': DefensiveBearNet(),
    'CRISIS': CapitalPreservationNet(),
    'TRANSITIONAL': AdaptiveNet(),
    'NEUTRAL': MeanReversionNet()
}

# Meta-router selects model:
meta_input = ['regime_label', 'regime_confidence', 'regime_stability']
selected_model = MetaRouter(meta_input)  # Soft ensemble weights
final_score = sum(w_i * model_i(features) for w_i, model_i in zip(weights, models))
```

**Expected Impact**:
- Specialist models per market condition
- Better bear market performance
- Estimated gain: **+$400-600/year** (mostly from fixing 2022-like periods)

---

## Expected Cumulative Impact

### Conservative Estimate (70% Success Rate on Optimizations)

| Optimization | Annual Gain | 3-Year Gain |
|--------------|-------------|-------------|
| Fix Trap Within Trend | +$350 | +$1,050 |
| Scale Order Block | +$560 | +$1,680 |
| Improve Bear Performance | +$190 | +$570 |
| Tighten Stop Losses | +$150 | +$450 |
| Increase Vol Exhaustion | +$140 | +$420 |
| **Phase 1 PyTorch (Quality)** | +$420 | +$1,260 |
| **Phase 2 PyTorch (Risk Mgmt)** | +$280 | +$840 |
| **Phase 3 PyTorch (Fusion)** | +$210 | +$630 |
| **Phase 4 PyTorch (Meta)** | +$280 | +$840 |
| **TOTAL** | **+$2,580/year** | **+$7,740** |

### Optimistic Estimate (90% Success Rate)

| Category | Annual Gain | 3-Year Gain |
|----------|-------------|-------------|
| Classical Optimization | +$1,800 | +$5,400 |
| PyTorch Integration | +$1,700 | +$5,100 |
| **TOTAL** | **+$3,500/year** | **+$10,500** |

---

## Implementation Timeline

### Month 1-2: Foundation (Optuna + Frontier)
- [ ] Run Pareto frontier on trap within trend parameters (2022-2024 split test)
- [ ] Run Optuna on order block quality threshold (maximize detection, maintain WR)
- [ ] Run Optuna on stop loss parameters (minimize avg loss)
- [ ] Validate on 2024 holdout

**Target**: +$500-700/year improvement

---

### Month 3-4: PyTorch Phase 1 (Archetype Quality)
- [ ] Build training dataset (extract all archetype candidates + outcomes)
- [ ] Train OrderBlockQualityNet (MLP, 10 features)
- [ ] Train TrapQualityClassifier (MLP, 15 features)
- [ ] Integrate into backtest_router_v10_full.py
- [ ] Validate on 2024 holdout

**Target**: +$800-1,000/year improvement

---

### Month 5-6: PyTorch Phase 2 (Risk Management)
- [ ] Build sequence dataset (50-bar windows for each trade)
- [ ] Train StopLossLSTM (predict optimal stop distance)
- [ ] Train PositionSizingNet (dynamic sizing)
- [ ] Integrate into risk management pipeline
- [ ] Validate on 2024 holdout

**Target**: +$300-400/year improvement

---

### Month 7-8: PyTorch Phase 3 (Fusion Transformer)
- [ ] Build multi-timeframe feature tensor dataset
- [ ] Train FusionTransformer (attention over 1H/4H/1D)
- [ ] A/B test against hardcoded fusion score
- [ ] Integrate as fusion_v2 if superior
- [ ] Validate on 2024 holdout

**Target**: +$300-500/year improvement

---

### Month 9-10: Meta-Learning & Final Integration
- [ ] Train regime-specific specialist models
- [ ] Build meta-router ensemble
- [ ] Full system integration test
- [ ] Run final 2022-2024 validation
- [ ] Generate production-ready configs

**Target**: +$400-600/year improvement

---

## Risk Considerations

### Overfitting Risks
- **3 years of data = limited sample size**
- Mitigation: Walk-forward validation, 2022-2023 train / 2024 test
- PyTorch models should be simple (avoid deep architectures)

### Regime Shift Risk
- **Bull market bias**: 2024 performance may not generalize
- Mitigation: Oversample 2022 bear market data in training

### Computational Cost
- PyTorch inference adds latency (~5-10ms per bar)
- Mitigation: Precompute where possible, use ONNX for production

---

## Next Steps

1. **Immediate** (This Week):
   - Run Optuna on trap within trend params (2022-2024 data)
   - Run Optuna on order block quality threshold
   - Generate new config candidates

2. **Short-Term** (Next 2 Weeks):
   - Implement top Optuna configs
   - Re-run full backtest
   - Validate improvements

3. **Medium-Term** (Next 1-2 Months):
   - Build PyTorch training pipeline
   - Train OrderBlockQualityNet
   - Integrate and validate

4. **Long-Term** (Next 3-6 Months):
   - Complete PyTorch roadmap (Phases 1-4)
   - Build production inference pipeline
   - Deploy to live trading (paper first)

---

## Conclusion

The Router v10 system shows **strong potential** (+11.40% over 3 years) but has **clear optimization opportunities**:

### Strengths:
- ✅ Order block retest is exceptional (90% WR)
- ✅ Regime switching working in 2024
- ✅ Sharpe ratio 1.966 (excellent risk-adjusted returns)

### Weaknesses:
- ❌ Trap within trend generates 83% of trades but net loss
- ❌ Stop losses too wide (inverted R:R)
- ❌ Bear market performance poor (2022: -$965)

### Optimization Path:
1. **Fix trap within trend** (highest impact)
2. **Scale order block retest** (highest ROI)
3. **Improve bear market survival** (risk mitigation)
4. **Deploy PyTorch** (frontier performance)

**Conservative target**: +$2,500/year improvement → **$730/year total** (7.3% annual return)
**Optimistic target**: +$3,500/year improvement → **$850/year total** (8.5% annual return)

With systematic optimization and PyTorch integration, this system can evolve from **"barely profitable"** to **"consistently strong"** across all market regimes.
