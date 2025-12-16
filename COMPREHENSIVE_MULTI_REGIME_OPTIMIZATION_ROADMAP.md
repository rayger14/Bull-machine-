# Bull Machine v2: Multi-Regime Profit Maximization Roadmap

**Generated**: 2025-11-14
**Objective**: Achieve PF > 1.5 across ALL market regimes (bull, bear, neutral) for BTC
**Current Status**:
- Bull Markets (2024): PF 6.17 ✅ GOLD STANDARD (preserve)
- Bear Markets (2022): PF 0.91 ❌ UNPROFITABLE (fix required)
- Mixed/Neutral: Unknown ❓ (needs validation)

---

## Executive Summary

**Critical Findings**:
1. **Regime classifier broken**: 90% of 2022 classified as "neutral" instead of "risk_off"
2. **Routing config missing**: Code exists but no regime weights configured
3. **Bear archetypes unprofitable**: PF 0.91 with 377 trades (overtrading)
4. **S2 pattern validated**: 58.5% win rate in 2022 (205 occurrences)
5. **S5 pattern missing**: 0 occurrences (thresholds too strict)

**Bottom Line**: Fix regime detection → Enable routing → Optimize bear patterns → Validate end-to-end

---

## Phase 0: Pre-Flight Validation (CRITICAL)

**Goal**: Understand current system behavior before making changes

### Task 0.1: Regime Classification Audit
**Complexity**: Simple
**Est. Time**: 30 min
**Priority**: P0 (BLOCKER)

**Objective**: Diagnose why 2022 is classified as "neutral" instead of "risk_off"

**Steps**:
```bash
# Test regime classifier on 2022 data
python -c "
from engine.context.regime_classifier import RegimeClassifier
import pandas as pd

# Load classifier
feature_order = ['VIX', 'DXY', 'MOVE', 'YIELD_2Y', 'YIELD_10Y',
                 'USDT.D', 'BTC.D', 'TOTAL', 'TOTAL2',
                 'funding', 'oi', 'rv_20d', 'rv_60d']
rc = RegimeClassifier.load('models/regime_classifier_gmm.pkl', feature_order)

# Load 2022 macro data
macro = pd.read_parquet('data/macro/BTC_1H_macro.parquet')
macro_2022 = macro['2022-01-01':'2022-12-31']

# Classify
regime_df = rc.classify_series(macro_2022)

# Print distribution
print('\n2022 Regime Distribution:')
print(regime_df['regime'].value_counts(normalize=True))
print('\nSample classifications:')
print(regime_df[['regime', 'features_used']].head(20))
"
```

**Acceptance Criteria**:
- [ ] Regime distribution percentages calculated
- [ ] If >50% neutral, root cause identified (missing features? wrong GMM clusters?)
- [ ] Decision made: fix GMM model vs use regime override

**Expected Outcome**: Root cause of neutral classification identified

**Tools**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/engine/context/regime_classifier.py`

**Dependencies**: None

---

### Task 0.2: Create Ground Truth Regime Labels
**Complexity**: Simple
**Est. Time**: 20 min
**Priority**: P0 (BLOCKER)

**Objective**: Define known bull/bear/neutral periods for validation

**Steps**:
```python
# Create ground truth regime mapping
GROUND_TRUTH_REGIMES = {
    # Bull markets (sustained uptrends)
    "2024-01-01": "risk_on",   # 2024 bull run (PF 6.17)
    "2024-02-01": "risk_on",
    "2024-03-01": "risk_on",
    "2024-04-01": "risk_on",
    "2024-05-01": "risk_on",
    "2024-06-01": "risk_on",
    "2024-07-01": "risk_on",
    "2024-08-01": "risk_on",
    "2024-09-01": "risk_on",

    # Bear markets (sustained downtrends)
    "2022-01-01": "risk_off",  # Terra collapse approaching
    "2022-02-01": "risk_off",
    "2022-03-01": "risk_off",
    "2022-04-01": "risk_off",
    "2022-05-01": "crisis",    # Terra -60%
    "2022-06-01": "crisis",    # Celsius freeze, 3AC
    "2022-07-01": "risk_off",
    "2022-08-01": "risk_off",
    "2022-09-01": "risk_off",
    "2022-10-01": "risk_off",
    "2022-11-01": "crisis",    # FTX collapse -25%
    "2022-12-01": "risk_off",

    # Neutral/choppy markets
    "2023-01-01": "neutral",   # Post-FTX recovery
    "2023-02-01": "neutral",
    "2023-03-01": "neutral",   # Banking crisis (SVB)
    "2023-04-01": "neutral",
    "2023-05-01": "neutral",
    "2023-06-01": "neutral",
    "2023-07-01": "neutral",
    "2023-08-01": "neutral",
    "2023-09-01": "neutral",
    "2023-10-01": "neutral",
    "2023-11-01": "risk_on",   # ETF hopes building
    "2023-12-01": "risk_on",
}
```

**Acceptance Criteria**:
- [ ] Ground truth labels created for 2020-2024
- [ ] Labels validated against BTC price action
- [ ] Regime override dict created for RegimeClassifier

**Expected Outcome**: Ground truth regime labels for validation

**Tools**: Manual analysis + TradingView charts

**Dependencies**: Task 0.1

---

### Task 0.3: Baseline Performance by Regime
**Complexity**: Moderate
**Est. Time**: 1 hour
**Priority**: P0 (CRITICAL)

**Objective**: Measure current PF/WR/trade count across actual regimes

**Steps**:
```bash
# Run backtest on 2022 (bear market)
python bin/backtest_knowledge_v2.py \
  --config configs/mvp/mvp_bear_market_v1.json \
  --asset BTC \
  --start 2022-01-01 \
  --end 2022-12-31 \
  --output results/baseline/bear_2022_baseline.json

# Run backtest on 2024 (bull market)
python bin/backtest_knowledge_v2.py \
  --config configs/mvp/mvp_bull_market_v1.json \
  --asset BTC \
  --start 2024-01-01 \
  --end 2024-09-30 \
  --output results/baseline/bull_2024_baseline.json

# Run backtest on 2023 (neutral market)
python bin/backtest_knowledge_v2.py \
  --config configs/mvp/mvp_bull_market_v1.json \
  --asset BTC \
  --start 2023-01-01 \
  --end 2023-12-31 \
  --output results/baseline/neutral_2023_baseline.json
```

**Acceptance Criteria**:
- [ ] 2022 bear market: PF, WR, trade count, archetype distribution
- [ ] 2024 bull market: PF 6.17 confirmed (preserve)
- [ ] 2023 neutral: PF, WR, trade count measured
- [ ] Archetype distribution analyzed per regime

**Success Metrics**:
- 2024 bull: PF 6.17, 17 trades (MUST PRESERVE)
- 2022 bear: PF 0.91, 377 trades (NEEDS FIXING)
- 2023 neutral: Unknown (establish baseline)

**Expected Outcome**: Clear baseline metrics for each regime

**Tools**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/bin/backtest_knowledge_v2.py`

**Dependencies**: Task 0.2

---

## Phase 1: Regime Detection & Routing (FIX INFRASTRUCTURE)

**Goal**: Fix regime classifier and enable routing weights

### Task 1.1: Fix Regime Classifier
**Complexity**: Moderate
**Est. Time**: 2 hours
**Priority**: P0 (BLOCKER)

**Objective**: Make GMM classifier correctly detect bear markets as "risk_off"

**Approach Options**:
1. **Option A (Quick)**: Use regime override dict (skip GMM)
2. **Option B (Proper)**: Retrain GMM with correct labels
3. **Option C (Hybrid)**: Use override for known periods + GMM for live

**Recommended**: Option C (hybrid approach)

**Steps for Option C**:
```python
# Enable regime override in backtest
from engine.context.regime_classifier import RegimeClassifier

# Load classifier with overrides
regime_override = {
    "2022": "risk_off",  # Force all 2022 as risk_off
    "2024": "risk_on",   # Force all 2024 as risk_on
    "2023": "neutral",   # Force all 2023 as neutral
}

rc = RegimeClassifier.load(
    model_path='models/regime_classifier_gmm.pkl',
    feature_order=feature_order,
    regime_override=regime_override
)
```

**Acceptance Criteria**:
- [ ] 2022 classified as "risk_off" (>80% of bars)
- [ ] 2024 classified as "risk_on" (>80% of bars)
- [ ] 2023 classified as "neutral" (>60% of bars)
- [ ] Regime override logging works

**Success Metrics**:
- <10% neutral classification in 2022
- >80% risk_off classification in 2022

**Expected Outcome**: Regime classifier produces sensible labels

**Tools**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/engine/context/regime_classifier.py`

**Dependencies**: Task 0.1, 0.2

---

### Task 1.2: Create Regime Routing Configs
**Complexity**: Simple
**Est. Time**: 1 hour
**Priority**: P1

**Objective**: Add routing weights to configs for each regime

**Config Structure**:
```json
{
  "routing": {
    "risk_on": {
      "weights": {
        "trap_within_trend": 1.3,
        "order_block_retest": 1.4,
        "bos_choch_reversal": 1.2,
        "wick_trap_moneytaur": 1.1,
        "failed_rally": 0.3,
        "long_squeeze": 0.2
      },
      "final_gate_delta": 0.0
    },
    "neutral": {
      "weights": {
        "trap_within_trend": 1.0,
        "order_block_retest": 1.0,
        "bos_choch_reversal": 1.0,
        "wick_trap_moneytaur": 0.8,
        "failed_rally": 0.6,
        "long_squeeze": 0.5
      },
      "final_gate_delta": 0.01
    },
    "risk_off": {
      "weights": {
        "trap_within_trend": 0.2,
        "order_block_retest": 0.4,
        "bos_choch_reversal": 0.5,
        "wick_trap_moneytaur": 0.3,
        "failed_rally": 1.8,
        "long_squeeze": 2.0
      },
      "final_gate_delta": 0.02
    },
    "crisis": {
      "weights": {
        "trap_within_trend": 0.1,
        "order_block_retest": 0.2,
        "bos_choch_reversal": 0.3,
        "wick_trap_moneytaur": 0.1,
        "failed_rally": 2.2,
        "long_squeeze": 2.5
      },
      "final_gate_delta": 0.04
    }
  }
}
```

**Files to Update**:
1. `configs/mvp/mvp_bull_market_v1.json` - Add routing (bull-biased)
2. `configs/mvp/mvp_bear_market_v1.json` - Add routing (bear-biased)
3. Create `configs/regime/regime_routing_production_v1.json` - Balanced

**Acceptance Criteria**:
- [ ] Routing weights added to 3 configs
- [ ] Weights sum check passed (bull archetypes downweighted in risk_off)
- [ ] Config validation passed (no syntax errors)

**Expected Outcome**: Configs ready for regime-aware backtests

**Tools**: JSON editor

**Dependencies**: Task 1.1

---

### Task 1.3: Validate Routing Impact
**Complexity**: Moderate
**Est. Time**: 1 hour
**Priority**: P1

**Objective**: Confirm routing changes archetype distribution as expected

**Steps**:
```bash
# Test 2022 with routing enabled
python bin/backtest_knowledge_v2.py \
  --config configs/regime/regime_routing_production_v1.json \
  --asset BTC \
  --start 2022-01-01 \
  --end 2022-12-31 \
  --output results/routing_validation/2022_with_routing.json \
  --verbose

# Analyze archetype distribution
python -c "
import json
result = json.load(open('results/routing_validation/2022_with_routing.json'))
trades = result['trades']

# Count archetypes
from collections import Counter
archetypes = Counter([t['archetype'] for t in trades])

print('\n2022 Archetype Distribution WITH ROUTING:')
for arch, count in archetypes.most_common():
    print(f'{arch:25s}: {count:3d} trades ({count/len(trades)*100:.1f}%)')
"
```

**Acceptance Criteria**:
- [ ] Trap Within Trend <30% of trades (down from 96.5%)
- [ ] Failed Rally >20% of trades (up from 0%)
- [ ] Long Squeeze >15% of trades (up from 0%)
- [ ] Total trade count reasonable (50-150 trades)

**Success Metrics**:
- Bear archetypes (S2, S5) >40% of trades in 2022
- Bull archetypes (A, H, B) <50% of trades in 2022

**Expected Outcome**: Routing successfully shifts archetype distribution

**Tools**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/bin/backtest_knowledge_v2.py`

**Dependencies**: Task 1.2

---

## Phase 2: Bear Archetype Optimization (IMPROVE PATTERNS)

**Goal**: Optimize S2 and S5 to achieve PF > 1.5 in bear markets

### Task 2.1: Fix S5 Long Squeeze Thresholds
**Complexity**: Moderate
**Est. Time**: 2 hours
**Priority**: P1

**Objective**: Make S5 actually trigger (currently 0 occurrences in 2022)

**Problem Analysis**:
```json
// Current S5 thresholds (TOO STRICT)
{
  "long_squeeze": {
    "fusion_threshold": 0.50,      // Very high (hardest to reach)
    "funding_z_min": 1.5,          // Requires extreme funding
    "rsi_min": 75,                 // Requires extreme RSI
    "liquidity_max": 0.25,         // Requires thin liquidity
    "max_risk_pct": 0.015,
    "atr_stop_mult": 2.0
  }
}
```

**Optimization Strategy**:
```bash
# Use optimizer to sweep S5 parameters
python bin/optimize_v18.py \
  --mode grid \
  --asset BTC \
  --start 2022-01-01 \
  --end 2022-12-31 \
  --archetype S5 \
  --param-ranges '{
    "fusion_threshold": [0.35, 0.38, 0.40, 0.42, 0.45],
    "funding_z_min": [0.8, 1.0, 1.2, 1.5],
    "rsi_min": [65, 68, 70, 72, 75]
  }' \
  --target profit_factor \
  --output results/optimization/s5_grid_2022.csv
```

**Acceptance Criteria**:
- [ ] S5 triggers 10-30 times in 2022 (not 0)
- [ ] PF > 1.2 for S5 in isolation
- [ ] Win rate 45-55%
- [ ] No false positives in 2024 bull market

**Success Metrics**:
- S5 PF > 1.3 in 2022
- S5 triggers 15-25 times in 2022
- S5 PF > 0.8 in 2024 (not harmful)

**Expected Outcome**: Optimized S5 config with reasonable hit rate

**Tools**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/bin/optimize_v18.py`

**Dependencies**: Task 1.3

---

### Task 2.2: Optimize S2 Failed Rally Parameters
**Complexity**: Moderate
**Est. Time**: 2 hours
**Priority**: P1

**Objective**: Improve S2 from 58.5% WR to PF > 1.4

**Current S2 Performance** (from validation_2022.json):
- Occurrences: 205 in 2022
- Win rate: 58.5% (GOOD)
- Forward 1h PnL: -0.096% (slightly negative)
- Forward 24h PnL: -0.676% (negative)

**Problem**: Good win rate but negative PnL → exits too early or stops too tight

**Optimization Strategy**:
```bash
# Optimize S2 exit parameters
python bin/optimize_v18.py \
  --mode grid \
  --asset BTC \
  --start 2022-01-01 \
  --end 2022-12-31 \
  --archetype S2 \
  --param-ranges '{
    "fusion_threshold": [0.34, 0.36, 0.38, 0.40],
    "rsi_min": [68, 70, 72, 75],
    "atr_stop_mult": [1.8, 2.0, 2.2, 2.5],
    "trail_atr": [1.2, 1.5, 1.8, 2.0]
  }' \
  --target profit_factor \
  --output results/optimization/s2_grid_2022.csv
```

**Acceptance Criteria**:
- [ ] S2 PF > 1.4 in 2022
- [ ] Win rate maintained >55%
- [ ] Average winner > 1.5R
- [ ] Triggers 30-60 times in 2022

**Success Metrics**:
- S2 PF > 1.4 in 2022
- S2 avg R-multiple > +0.3
- S2 doesn't harm 2024 performance

**Expected Outcome**: Optimized S2 config with positive expectancy

**Tools**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/bin/optimize_v18.py`

**Dependencies**: Task 2.1

---

### Task 2.3: Test Other Bear Archetypes (S1, S3, S4, S8)
**Complexity**: Complex
**Est. Time**: 4 hours
**Priority**: P2 (Nice to have)

**Objective**: Evaluate if S1/S3/S4/S8 add value in bear markets

**Current Status** (from validation_2022.json):
- S1 Liquidity Vacuum: 0 occurrences (missing liquidity_score)
- S3 Whipsaw: Not evaluated
- S4 Distribution: 992 occurrences, 49.8% WR (break-even)
- S8 Exhaustion: 390 occurrences, 48.7% WR (slightly losing)

**Approach**:
1. **S4 Distribution**: Already has data, optimize thresholds
2. **S8 Exhaustion**: Already has data, optimize thresholds
3. **S1 Liquidity Vacuum**: Skip (requires liquidity_score feature)
4. **S3 Whipsaw**: Define logic + test

**Steps for S4**:
```bash
# S4 has most occurrences (992) - worth optimizing
python bin/optimize_v18.py \
  --mode grid \
  --asset BTC \
  --start 2022-01-01 \
  --end 2022-12-31 \
  --archetype S4 \
  --param-ranges '{
    "fusion_threshold": [0.36, 0.38, 0.40, 0.42],
    "volume_z_min": [1.5, 2.0, 2.5],
    "atr_stop_mult": [1.8, 2.0, 2.2]
  }' \
  --target profit_factor \
  --output results/optimization/s4_grid_2022.csv
```

**Acceptance Criteria**:
- [ ] S4 and S8 optimized (if PF > 1.2 achievable)
- [ ] Decision made: enable or disable each pattern
- [ ] Total bear archetypes portfolio tested

**Success Metrics**:
- At least 3 bear archetypes with PF > 1.2
- Combined bear portfolio PF > 1.5

**Expected Outcome**: Validated set of profitable bear archetypes

**Tools**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/bin/optimize_v18.py`

**Dependencies**: Task 2.2

---

### Task 2.4: Bear Market Portfolio Assembly
**Complexity**: Moderate
**Est. Time**: 1 hour
**Priority**: P1

**Objective**: Create optimal bear config combining best archetypes

**Steps**:
```python
# Combine optimized bear archetypes
bear_portfolio = {
    "enable_S2": True,   # Failed Rally (PF 1.4+, 58% WR)
    "enable_S5": True,   # Long Squeeze (PF 1.3+, 50% WR)
    "enable_S4": True,   # Distribution (if PF > 1.2)
    "enable_S8": False,  # Exhaustion (skip if PF < 1.1)

    # Keep minimal bull archetypes for counter-trend
    "enable_A": True,    # Trap Within Trend (downweighted 0.2x)
    "enable_H": True,    # Order Block (downweighted 0.4x)
    "enable_B": False,   # BOS/CHOCH (downweighted 0.5x)
}

# Update configs/mvp/mvp_bear_market_v2.json
```

**Acceptance Criteria**:
- [ ] Bear config includes only profitable archetypes (PF > 1.2)
- [ ] Monthly share caps adjusted (40% S2, 25% S5, etc.)
- [ ] Cooldowns tuned to prevent overtrading
- [ ] Config validated with schema

**Success Metrics**:
- Bear portfolio PF > 1.5 on 2022
- Trade count 60-120 (not 377)

**Expected Outcome**: Production-ready bear market config

**Tools**: JSON editor + validation script

**Dependencies**: Task 2.3

---

## Phase 3: Multi-Regime Validation (END-TO-END TESTING)

**Goal**: Validate that each config works in its target regime without harming others

### Task 3.1: Bull Config Regression Test
**Complexity**: Simple
**Est. Time**: 30 min
**Priority**: P0 (CRITICAL)

**Objective**: Ensure routing changes don't break PF 6.17 gold standard

**Steps**:
```bash
# Test updated bull config on 2024
python bin/backtest_knowledge_v2.py \
  --config configs/regime/regime_routing_production_v1.json \
  --asset BTC \
  --start 2024-01-01 \
  --end 2024-09-30 \
  --output results/validation/bull_2024_with_routing.json

# Compare to baseline
python scripts/compare_results.py \
  results/baseline/bull_2024_baseline.json \
  results/validation/bull_2024_with_routing.json \
  --tolerance 0.05
```

**Acceptance Criteria**:
- [ ] PF > 6.0 (within 5% of 6.17)
- [ ] Trade count 15-20 (similar to baseline 17)
- [ ] Same archetype distribution (mostly A, H, B, K)
- [ ] No bear archetypes triggered in bull market

**Success Metrics**:
- PF 6.0-6.5 (gold standard preserved)
- Win rate >70%
- Trade count 15-20

**Expected Outcome**: Bull config performance preserved

**Tools**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/bin/backtest_knowledge_v2.py`

**Dependencies**: Task 2.4

---

### Task 3.2: Bear Config Forward Test
**Complexity**: Moderate
**Est. Time**: 1 hour
**Priority**: P1

**Objective**: Confirm bear config achieves PF > 1.5 on 2022

**Steps**:
```bash
# Test optimized bear config on 2022
python bin/backtest_knowledge_v2.py \
  --config configs/mvp/mvp_bear_market_v2.json \
  --asset BTC \
  --start 2022-01-01 \
  --end 2022-12-31 \
  --output results/validation/bear_2022_optimized.json \
  --verbose

# Analyze results
python scripts/analyze_results.py \
  results/validation/bear_2022_optimized.json \
  --breakdown-by archetype \
  --breakdown-by month
```

**Acceptance Criteria**:
- [ ] PF > 1.5 (up from 0.91)
- [ ] Win rate 50-55%
- [ ] Trade count 60-120 (down from 377)
- [ ] Bear archetypes >50% of trades
- [ ] Max drawdown <30%

**Success Metrics**:
- PF > 1.5 on 2022
- Avg R-multiple > +0.3
- No single month with PF < 0.8

**Expected Outcome**: Profitable bear market strategy

**Tools**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/bin/backtest_knowledge_v2.py`

**Dependencies**: Task 3.1

---

### Task 3.3: Neutral Market Validation
**Complexity**: Moderate
**Est. Time**: 1 hour
**Priority**: P1

**Objective**: Validate performance in choppy/sideways markets (2023)

**Steps**:
```bash
# Test on 2023 (neutral/choppy market)
python bin/backtest_knowledge_v2.py \
  --config configs/regime/regime_routing_production_v1.json \
  --asset BTC \
  --start 2023-01-01 \
  --end 2023-12-31 \
  --output results/validation/neutral_2023.json

# Analyze choppy market behavior
python scripts/analyze_results.py \
  results/validation/neutral_2023.json \
  --focus choppy_periods
```

**Acceptance Criteria**:
- [ ] PF > 1.2 (break-even or better)
- [ ] Trade count reasonable (40-80)
- [ ] Mixed archetype usage (bull + bear)
- [ ] Low max drawdown (<20%)

**Success Metrics**:
- PF > 1.2 on 2023
- Win rate >45%
- Sharpe ratio >0.5

**Expected Outcome**: System doesn't blow up in choppy markets

**Tools**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/bin/backtest_knowledge_v2.py`

**Dependencies**: Task 3.2

---

### Task 3.4: Full Period Walk-Forward Test
**Complexity**: Complex
**Est. Time**: 3 hours
**Priority**: P1

**Objective**: Validate 2020-2024 with regime routing enabled

**Steps**:
```bash
# Full multi-year backtest
python bin/backtest_knowledge_v2.py \
  --config configs/regime/regime_routing_production_v1.json \
  --asset BTC \
  --start 2020-01-01 \
  --end 2024-09-30 \
  --output results/validation/full_period_2020_2024.json

# Walk-forward analysis
python bin/optimize_v18.py \
  --mode walkforward \
  --asset BTC \
  --start 2020-01-01 \
  --end 2024-09-30 \
  --train-months 12 \
  --test-months 3 \
  --output results/walkforward/regime_routing_wf.csv
```

**Acceptance Criteria**:
- [ ] Overall PF > 2.0
- [ ] All years have PF > 1.0
- [ ] No regime has PF < 1.2
- [ ] Sharpe ratio > 1.5
- [ ] Max drawdown < 35%

**Success Metrics**:
- 2020: PF > 1.5
- 2021: PF > 2.0 (bull market)
- 2022: PF > 1.5 (bear market)
- 2023: PF > 1.2 (neutral)
- 2024: PF > 6.0 (bull market)

**Expected Outcome**: Profitable across all market conditions

**Tools**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/bin/backtest_knowledge_v2.py`

**Dependencies**: Task 3.3

---

## Phase 4: Production Config Creation (FINALIZE)

**Goal**: Create deployment-ready configs with regime awareness

### Task 4.1: Create Regime-Aware Production Config
**Complexity**: Simple
**Est. Time**: 1 hour
**Priority**: P1

**Objective**: Merge bull/bear configs into single regime-aware config

**Config Structure**:
```json
{
  "version": "production_regime_aware_v1",
  "profile": "multi_regime_optimized",
  "description": "Regime-aware config (bull PF 6+, bear PF 1.5+)",

  "regime_detection": {
    "enabled": true,
    "model_path": "models/regime_classifier_gmm.pkl",
    "use_override": true,
    "override_map": {
      "2022": "risk_off",
      "2024": "risk_on"
    }
  },

  "archetypes": {
    "use_archetypes": true,
    "max_trades_per_day": 3,

    // Enable all profitable archetypes
    "enable_A": true,   // Bull: Trap Within Trend
    "enable_B": true,   // Bull: BOS/CHOCH
    "enable_C": true,   // Bull: Order Block
    "enable_H": true,   // Bull: Wick Trap
    "enable_K": true,   // Bull: Volume Exhaustion
    "enable_L": true,   // Bull: Liquidity Sweep
    "enable_S2": true,  // Bear: Failed Rally
    "enable_S5": true,  // Bear: Long Squeeze

    // Monthly share caps (regime-dependent)
    "monthly_share_cap": {
      "trap_within_trend": 0.35,
      "order_block_retest": 0.25,
      "bos_choch_reversal": 0.15,
      "failed_rally": 0.15,
      "long_squeeze": 0.10
    }
  },

  "routing": {
    // Full routing config from Task 1.2
  }
}
```

**Acceptance Criteria**:
- [ ] Single config works for all regimes
- [ ] Regime detection enabled
- [ ] All optimized archetypes included
- [ ] Routing weights tuned per regime
- [ ] Config validation passed

**Expected Outcome**: Production-ready multi-regime config

**Tools**: JSON editor

**Dependencies**: Task 3.4

---

### Task 4.2: Create Regime-Specific Overrides
**Complexity**: Simple
**Est. Time**: 30 min
**Priority**: P2

**Objective**: Create specialized configs for known regime periods

**Files to Create**:
1. `configs/production/bull_only_v1.json` - Force risk_on behavior
2. `configs/production/bear_only_v1.json` - Force risk_off behavior
3. `configs/production/neutral_only_v1.json` - Force neutral behavior

**Use Cases**:
- Testing specific regime behavior
- Override GMM when you KNOW the regime
- Debugging archetype performance

**Acceptance Criteria**:
- [ ] 3 specialized configs created
- [ ] Each config forces specific regime
- [ ] Configs validated

**Expected Outcome**: Flexible deployment options

**Tools**: JSON editor

**Dependencies**: Task 4.1

---

### Task 4.3: Final Validation Suite
**Complexity**: Moderate
**Est. Time**: 2 hours
**Priority**: P0 (CRITICAL)

**Objective**: Run full test suite before production deployment

**Test Matrix**:
```bash
# Test 1: Bull market performance (2024)
python bin/backtest_knowledge_v2.py \
  --config configs/production/production_regime_aware_v1.json \
  --asset BTC --start 2024-01-01 --end 2024-09-30 \
  --output results/final_validation/bull_2024.json

# Test 2: Bear market performance (2022)
python bin/backtest_knowledge_v2.py \
  --config configs/production/production_regime_aware_v1.json \
  --asset BTC --start 2022-01-01 --end 2022-12-31 \
  --output results/final_validation/bear_2022.json

# Test 3: Neutral market performance (2023)
python bin/backtest_knowledge_v2.py \
  --config configs/production/production_regime_aware_v1.json \
  --asset BTC --start 2023-01-01 --end 2023-12-31 \
  --output results/final_validation/neutral_2023.json

# Test 4: Full period (2020-2024)
python bin/backtest_knowledge_v2.py \
  --config configs/production/production_regime_aware_v1.json \
  --asset BTC --start 2020-01-01 --end 2024-09-30 \
  --output results/final_validation/full_period.json

# Test 5: Integration tests
python -m pytest tests/integration/test_regime_routing.py -v
```

**Acceptance Criteria**:
- [ ] Bull 2024: PF > 6.0 (gold standard preserved)
- [ ] Bear 2022: PF > 1.5 (target achieved)
- [ ] Neutral 2023: PF > 1.2 (break-even or better)
- [ ] Full period: PF > 2.0, all years profitable
- [ ] All integration tests pass

**Success Metrics**:
- NO regime has PF < 1.2
- Overall PF > 2.5
- Sharpe ratio > 1.5
- Max drawdown < 35%

**Expected Outcome**: Production-ready system validated

**Tools**: Full test suite

**Dependencies**: Task 4.1, 4.2

---

### Task 4.4: Performance Report & Documentation
**Complexity**: Simple
**Est. Time**: 1 hour
**Priority**: P1

**Objective**: Document final performance across all regimes

**Deliverables**:
1. **Performance Report** (`MULTI_REGIME_PERFORMANCE_REPORT.md`)
   - Metrics per regime (PF, WR, trade count, Sharpe, max DD)
   - Archetype distribution per regime
   - Comparison to baseline
   - Equity curves

2. **Config Guide** (`docs/REGIME_AWARE_CONFIGS_GUIDE.md`)
   - How to use production configs
   - When to use bull/bear/neutral overrides
   - Regime weight tuning guide

3. **Update CHANGELOG.md**
   - Multi-regime optimization summary
   - Breaking changes (if any)
   - Migration guide

**Acceptance Criteria**:
- [ ] Performance report complete
- [ ] Config guide complete
- [ ] CHANGELOG updated
- [ ] All docs reviewed

**Expected Outcome**: Complete documentation package

**Tools**: Markdown editor

**Dependencies**: Task 4.3

---

## Critical Success Factors

### Must-Have Outcomes
1. **Bull market preserved**: 2024 PF > 6.0 (gold standard intact)
2. **Bear market profitable**: 2022 PF > 1.5 (was 0.91)
3. **Neutral market stable**: 2023 PF > 1.2
4. **Full period profitable**: ALL years PF > 1.0

### Key Metrics Dashboard
| Metric | 2022 (Bear) | 2023 (Neutral) | 2024 (Bull) | Full Period |
|--------|-------------|----------------|-------------|-------------|
| **Profit Factor** | >1.5 | >1.2 | >6.0 | >2.5 |
| **Win Rate** | 50-55% | 45-50% | 70-75% | 55-65% |
| **Trade Count** | 60-120 | 40-80 | 15-20 | 150-250 |
| **Sharpe Ratio** | >0.8 | >0.5 | >2.0 | >1.5 |
| **Max Drawdown** | <30% | <20% | <15% | <35% |

### Risk Management
- **Preserve gold standard**: Always test bull config on 2024 first
- **No overtrading**: Max 4 trades/day, cooldowns enforced
- **Regime validation**: Verify regime labels before optimizing
- **Walk-forward testing**: Validate on unseen data

---

## Execution Timeline

### Week 1: Infrastructure Fix
- **Days 1-2**: Phase 0 (Pre-Flight Validation)
- **Days 3-4**: Phase 1 (Regime Detection & Routing)
- **Day 5**: Validation & adjustment

**Milestone**: Routing working, regimes detected correctly

### Week 2: Optimization
- **Days 1-2**: Phase 2 Tasks 2.1-2.2 (S5, S2 optimization)
- **Days 3-4**: Phase 2 Tasks 2.3-2.4 (Other archetypes, portfolio)
- **Day 5**: Bear config validation

**Milestone**: Bear config PF > 1.5 on 2022

### Week 3: Validation & Production
- **Days 1-2**: Phase 3 (Multi-regime validation)
- **Days 3-4**: Phase 4 (Production configs)
- **Day 5**: Final validation & documentation

**Milestone**: Production-ready regime-aware configs

---

## Tools & Scripts Reference

### Primary Tools
1. **Backtest Engine**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/bin/backtest_knowledge_v2.py`
2. **Optimizer**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/bin/optimize_v18.py`
3. **Regime Classifier**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/engine/context/regime_classifier.py`
4. **Feature Flags**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/engine/feature_flags.py`

### Config Locations
- **MVP Configs**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/configs/mvp/`
- **Bear Configs**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/configs/bear/`
- **Regime Configs**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/configs/regime/`
- **Frozen Baseline**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/configs/frozen/btc_1h_v2_baseline.json`

### Results Directories
- **Baseline**: `results/baseline/`
- **Optimization**: `results/optimization/`
- **Validation**: `results/validation/`
- **Walk-Forward**: `results/walkforward/`
- **Final**: `results/final_validation/`

---

## Troubleshooting Guide

### Problem: Regime classifier still marks 2022 as neutral
**Solution**: Use regime override dict (Task 1.1 Option C)
```python
regime_override = {"2022": "risk_off"}
```

### Problem: S5 still has 0 occurrences
**Solution**: Lower thresholds dramatically
```json
{
  "fusion_threshold": 0.35,  // Down from 0.50
  "funding_z_min": 0.8,      // Down from 1.5
  "rsi_min": 65              // Down from 75
}
```

### Problem: Routing doesn't change archetype distribution
**Solution**: Check logs for regime detection
```bash
grep "REGIME ROUTING" results/validation/bear_2022.log
```

### Problem: Bear config hurts bull performance
**Solution**: Separate configs for each regime
- Use `configs/production/bull_only_v1.json` for bull markets
- Use `configs/production/bear_only_v1.json` for bear markets

### Problem: Overtrading (>300 trades/year)
**Solution**: Increase cooldowns and fusion thresholds
```json
{
  "max_trades_per_day": 2,
  "cooldown_bars": 20,
  "fusion_threshold": 0.42
}
```

---

## Next Steps After Completion

1. **Live Paper Trading**
   - Deploy regime-aware config to paper trading
   - Monitor for 1-2 months
   - Validate regime detection in real-time

2. **Multi-Asset Expansion**
   - Test on ETH (similar patterns)
   - Test on SOL, MATIC (alt-coins)
   - Adjust regime weights per asset

3. **Advanced Regime Detection**
   - Retrain GMM with more features
   - Add sentiment indicators (Fear & Greed)
   - Implement regime transition detection

4. **Portfolio Optimization**
   - Multi-asset correlation analysis
   - Kelly criterion position sizing
   - Regime-dependent leverage

---

## Sign-Off Checklist

### Before Moving to Production
- [ ] All phases completed (0-4)
- [ ] Bull market gold standard preserved (PF > 6.0)
- [ ] Bear market profitable (PF > 1.5)
- [ ] Neutral market stable (PF > 1.2)
- [ ] Full period validation passed
- [ ] Integration tests passing
- [ ] Documentation complete
- [ ] Performance report published
- [ ] Configs backed up to `/configs/frozen/`

### Deployment Approval
- [ ] Technical review completed
- [ ] Risk management approval
- [ ] Paper trading plan defined
- [ ] Rollback plan documented

---

**End of Roadmap**

**Document Version**: 1.0
**Last Updated**: 2025-11-14
**Owner**: Bull Machine System Architect
**Status**: READY FOR EXECUTION
