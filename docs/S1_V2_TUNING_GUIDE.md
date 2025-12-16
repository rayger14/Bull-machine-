# S1 Liquidity Vacuum V2 - Threshold Tuning Guide

**Version**: V2 Production (2025-11-23)
**Config**: `configs/s1_v2_production.json`
**Target Audience**: Engineers and quantitative analysts optimizing S1 performance

---

## Table of Contents

1. [When to Tune](#1-when-to-tune)
2. [What to Tune](#2-what-to-tune)
3. [How to Tune](#3-how-to-tune)
4. [Tuning Recipes](#4-tuning-recipes)
5. [Validation Framework](#5-validation-framework)
6. [Advanced Topics](#6-advanced-topics)

---

## 1. When to Tune

### Immediate Tuning Triggers (Within 1 Week)

**Trigger**: Major regime change (bear→bull or bull→bear)
- **Why**: S1 thresholds optimized for specific market volatility profiles
- **Action**: Re-optimize exhaustion thresholds for new regime
- **Example**: 2023 bear→bull transition required raising thresholds to reduce false positives in recovery environment

**Trigger**: Trade frequency drifts outside 40-60/year
- **Why**: Too many trades (>100) → low precision, too few (<20) → low recall
- **Action**: Adjust confluence parameters or exhaustion gates
- **Example**: If 150 trades/year, raise `confluence_threshold` from 0.65 to 0.70

**Trigger**: Missing 2+ major capitulation events in sequence
- **Why**: Pattern detection too conservative
- **Action**: Lower confluence requirements or reduce crisis threshold
- **Example**: If missed LUNA + FTX, consider lowering `confluence_min_conditions` from 3 to 2

**Trigger**: Win rate drops below 45% for >1 month
- **Why**: Signal quality degrading, high false positive rate
- **Action**: Tighten confluence or raise exhaustion thresholds
- **Example**: If win rate 35%, raise `volume_climax_3b_min` and `wick_exhaustion_3b_min`

### Scheduled Tuning (Quarterly)

**Quarterly Review Checklist**:
- [ ] Review trade frequency trend (is it drifting?)
- [ ] Calculate false positive ratio (target 10-15:1)
- [ ] Compare captured vs missed events (target 50-70% recall)
- [ ] Analyze win rate by regime (risk_off should be higher than risk_on)
- [ ] Check threshold percentiles (are they still at p90-p95 level?)

**Action**: If 2+ checklist items show degradation, schedule optimization run.

### Opportunistic Tuning (As Needed)

**After New Feature Addition**:
- New V2 features become available (e.g., enhanced liquidity metrics)
- Re-optimize confluence weights to incorporate new signals

**After Market Structure Change**:
- Exchange dominance shifts (e.g., Binance → Coinbase)
- Liquidity dynamics change (e.g., post-ETF approval)
- Re-optimize with recent data window

**After Extended Quiet Period**:
- 3+ months with <5 trades
- May need to relax thresholds or verify regime classifier working

---

## 2. What to Tune

### Parameter Categories

#### Tier 1: High-Impact Parameters (Tune First)

**Confluence Parameters** (directly control precision/recall tradeoff):

| Parameter | Current | Range | Impact on Trades | Impact on Quality |
|-----------|---------|-------|------------------|-------------------|
| `confluence_threshold` | 0.65 | 0.50-0.80 | ±50-100 trades/year per 0.05 change | High (strong filter) |
| `confluence_min_conditions` | 3 | 2-4 | ±30-50 trades/year per unit | Very high (hard gate) |

**Exhaustion Gates** (control sensitivity to microstructure):

| Parameter | Current | Range | Impact on Trades | Impact on Quality |
|-----------|---------|-------|------------------|-------------------|
| `volume_climax_3b_min` | 0.50 | 0.30-0.70 | ±20-40 trades/year per 0.10 | Medium (noise filter) |
| `wick_exhaustion_3b_min` | 0.60 | 0.40-0.80 | ±15-30 trades/year per 0.10 | Medium (rejection signal) |

#### Tier 2: Medium-Impact Parameters (Tune After Tier 1)

**Hard Gates** (set minimum requirements):

| Parameter | Current | Range | Impact on Trades | Impact on Quality |
|-----------|---------|-------|------------------|-------------------|
| `capitulation_depth_max` | -0.20 | -0.15 to -0.30 | ±10-20 trades/year per 0.05 | High (event filter) |
| `crisis_composite_min` | 0.35 | 0.25-0.50 | ±50-100 trades/year per 0.05 | Very high (regime gate) |

**Regime Filter** (control operating context):

| Parameter | Current | Range | Impact on Trades | Impact on Quality |
|-----------|---------|-------|------------------|-------------------|
| `drawdown_override_pct` | 0.10 | 0.05-0.20 | ±5-15 trades/year per 0.05 | Low (rare trigger) |
| `allowed_regimes` | risk_off, crisis | Any combo | ±20-40 trades/year per regime | High (regime quality) |

#### Tier 3: Low-Impact Parameters (Tune Last or Leave Fixed)

**Confluence Weights** (fine-tuning signal importance):

| Weight Group | Current Total | Range | Impact on Trades | Impact on Quality |
|--------------|---------------|-------|------------------|-------------------|
| Core signals | 0.50 | 0.40-0.60 | ±5-10 trades/year | Low (marginal effect) |
| Liquidity dynamics | 0.25 | 0.15-0.35 | ±2-5 trades/year | Low (marginal effect) |
| Confluence signals | 0.25 | 0.15-0.35 | ±2-5 trades/year | Low (marginal effect) |

**Risk Management** (controls position sizing, not detection):

| Parameter | Current | Range | Notes |
|-----------|---------|-------|-------|
| `max_risk_pct` | 0.02 | 0.01-0.03 | Does NOT affect detection, only position size |
| `atr_stop_mult` | 2.5 | 1.5-3.5 | Does NOT affect detection, only stop placement |
| `cooldown_bars` | 12 | 8-24 | Prevents overtrading same event |

### What NOT to Tune

**Do NOT tune** (keep fixed):
- `use_v2_logic: true` - Required for multi-bar detection
- `direction: "long"` - S1 is long-only by design
- `archetype_weight: 2.5` - Fusion weight (legacy, not used in confluence mode)
- `fusion_threshold: 0.30` - Legacy parameter (confluence threshold used instead)

---

## 3. How to Tune

### Walk-Forward Validation Approach (RECOMMENDED)

Walk-forward validation prevents overfitting by optimizing on in-sample periods and validating on out-of-sample periods.

**Process**:
1. Split data into train/validate/test windows
2. Optimize on train window
3. Validate on next window (out-of-sample)
4. Walk forward and repeat
5. Select configuration that performs consistently across all windows

**Example Timeline** (2022-2024 data):

```
Train 1:  2022-01 to 2022-06  →  Validate 1: 2022-07 to 2022-09
Train 2:  2022-04 to 2022-09  →  Validate 2: 2022-10 to 2022-12
Train 3:  2022-07 to 2022-12  →  Validate 3: 2023-01 to 2023-03
Train 4:  2022-10 to 2023-03  →  Validate 4: 2023-04 to 2023-06
Final Test: 2023-07 to 2024-11 (full out-of-sample)
```

**Implementation**:
```bash
# Run walk-forward validation
python bin/validate_walk_forward.py \
  --config configs/s1_v2_production.json \
  --archetype liquidity_vacuum \
  --start 2022-01-01 \
  --end 2024-11-18 \
  --train-months 6 \
  --validate-months 3 \
  --step-months 3 \
  --output results/s1_walk_forward_validation.json
```

### Optuna Optimization Setup

Optuna is a hyperparameter optimization framework that efficiently searches parameter space.

**Step 1**: Define parameter search space

```python
# bin/optimize_s1_thresholds.py
import optuna

def objective(trial):
    """Define optimization objective for S1."""

    # Tier 1: High-impact parameters
    confluence_threshold = trial.suggest_float('confluence_threshold', 0.55, 0.75, step=0.05)
    confluence_min_conditions = trial.suggest_int('confluence_min_conditions', 2, 4)
    volume_climax_min = trial.suggest_float('volume_climax_3b_min', 0.35, 0.65, step=0.05)
    wick_exhaustion_min = trial.suggest_float('wick_exhaustion_3b_min', 0.45, 0.75, step=0.05)

    # Tier 2: Medium-impact parameters (optional)
    crisis_composite_min = trial.suggest_float('crisis_composite_min', 0.30, 0.45, step=0.05)
    capitulation_depth = trial.suggest_float('capitulation_depth_max', -0.15, -0.25, step=0.05)

    # Build config with trial parameters
    config = base_config.copy()
    config['archetypes']['thresholds']['liquidity_vacuum'].update({
        'confluence_threshold': confluence_threshold,
        'confluence_min_conditions': confluence_min_conditions,
        'volume_climax_3b_min': volume_climax_min,
        'wick_exhaustion_3b_min': wick_exhaustion_min,
        'crisis_composite_min': crisis_composite_min,
        'capitulation_depth_max': capitulation_depth,
    })

    # Run backtest
    results = run_backtest(config, train_start, train_end)

    # Multi-objective: maximize profit factor AND constrain trade frequency
    pf = results['profit_factor']
    trades_per_year = results['trades_per_year']

    # Penalty for trade frequency outside target range (40-60/year)
    if trades_per_year < 40:
        freq_penalty = (40 - trades_per_year) * 0.1  # Too few trades
    elif trades_per_year > 60:
        freq_penalty = (trades_per_year - 60) * 0.05  # Too many trades
    else:
        freq_penalty = 0  # In target range

    # Combined score: PF - frequency penalty
    score = pf - freq_penalty

    return score

# Create study and optimize
study = optuna.create_study(
    direction='maximize',
    study_name='s1_threshold_optimization',
    storage='sqlite:///optuna_s1_thresholds.db'
)

study.optimize(objective, n_trials=100, n_jobs=4)

print(f"Best parameters: {study.best_params}")
print(f"Best score: {study.best_value}")
```

**Step 2**: Run optimization

```bash
# Single-objective optimization (maximize profit factor)
python bin/optimize_s1_thresholds.py \
  --config configs/s1_v2_production.json \
  --start 2022-01-01 \
  --end 2023-12-31 \
  --n-trials 100 \
  --n-jobs 4

# Multi-objective optimization (PF vs trade frequency)
python bin/optimize_s1_thresholds.py \
  --config configs/s1_v2_production.json \
  --start 2022-01-01 \
  --end 2023-12-31 \
  --n-trials 200 \
  --n-jobs 8 \
  --multi-objective \
  --objectives profit_factor trades_per_year
```

**Step 3**: Analyze results

```bash
# View best trials
python bin/analyze_optuna_results.py \
  --study-name s1_threshold_optimization \
  --database sqlite:///optuna_s1_thresholds.db \
  --top-n 10

# Visualize parameter importance
python bin/visualize_optuna_importance.py \
  --study-name s1_threshold_optimization \
  --database sqlite:///optuna_s1_thresholds.db
```

### Pareto Frontier Analysis

For multi-objective optimization (e.g., profit factor vs trade frequency), use Pareto frontier to select tradeoffs.

**Concept**: Pareto frontier shows configurations where improving one objective requires worsening another.

**Process**:
1. Run multi-objective optimization
2. Plot Pareto frontier (PF vs trades/year)
3. Select configuration matching your preferences

**Example**:
```bash
# Generate Pareto frontier visualization
python bin/visualize_pareto_frontier.py \
  --study-name s1_threshold_optimization \
  --database sqlite:///optuna_s1_thresholds.db \
  --objective-x trades_per_year \
  --objective-y profit_factor \
  --output results/s1_pareto_frontier.png
```

**Interpretation**:

```
Profit Factor
    ^
3.0 |     A  (Conservative: 30 trades/year, PF 2.8)
2.5 |   B    (Balanced: 50 trades/year, PF 2.3)
2.0 | C      (Aggressive: 80 trades/year, PF 1.8)
1.5 |
    +-------> Trades per Year
```

- **Point A**: High precision, low recall (conservative)
- **Point B**: Balanced precision/recall (recommended)
- **Point C**: High recall, lower precision (aggressive)

**Selection Criteria**:
- **Risk-averse**: Choose Point A (fewer trades, higher quality)
- **Balanced**: Choose Point B (target 40-60 trades/year)
- **Risk-seeking**: Choose Point C (more opportunities, accept lower quality)

### Out-of-Sample Testing

**CRITICAL**: Always validate optimized parameters on out-of-sample data to avoid overfitting.

**Process**:
1. Optimize on training period (e.g., 2022-01 to 2023-06)
2. Validate on test period (e.g., 2023-07 to 2024-11)
3. Compare performance (in-sample vs out-of-sample)
4. If degradation >30%, re-optimize with more conservative parameters

**Implementation**:
```bash
# Step 1: Optimize on training period
python bin/optimize_s1_thresholds.py \
  --config configs/s1_v2_production.json \
  --start 2022-01-01 \
  --end 2023-06-30 \
  --n-trials 100 \
  --output configs/s1_optimized_train.json

# Step 2: Test on validation period
python bin/backtest.py \
  --config configs/s1_optimized_train.json \
  --start 2023-07-01 \
  --end 2024-11-18 \
  --output results/s1_oos_validation.csv

# Step 3: Compare results
python bin/compare_is_oos_performance.py \
  --in-sample results/s1_train_results.csv \
  --out-of-sample results/s1_oos_validation.csv
```

**Acceptable Degradation**:
- Profit Factor: <30% drop (e.g., 2.5 → 1.75+)
- Win Rate: <10% drop (e.g., 55% → 45%+)
- Trade Frequency: <50% change (e.g., 50 → 25-75)

**If degradation excessive**:
- Reduce search space (fewer parameters)
- Increase training period (more data)
- Add regularization (penalize extreme parameter values)

---

## 4. Tuning Recipes

### Recipe 1: Too Many Trades (>100/year)

**Symptom**: High trade frequency, low win rate (<45%)

**Quick Fix** (5 minutes):
```json
{
  "confluence_threshold": 0.70,  // Raised from 0.65
  "volume_climax_3b_min": 0.60,  // Raised from 0.50
  "wick_exhaustion_3b_min": 0.70  // Raised from 0.60
}
```

**Expected Impact**: Reduce trades by ~30-40%, increase win rate by 5-10%

**Medium Fix** (if quick fix insufficient):
```json
{
  "crisis_composite_min": 0.40,  // Raised from 0.35
  "drawdown_override_pct": 0.15  // Raised from 0.10
}
```

**Expected Impact**: Further reduce trades by ~20%, filter regime edges

**Validation**:
```bash
python bin/backtest.py --config configs/s1_tuned.json --start 2022-01-01 --end 2024-11-18
# Target: 40-60 trades/year, 50%+ win rate
```

### Recipe 2: Missing Events (Low Recall)

**Symptom**: Missed 2+ major capitulation events, trade frequency <30/year

**Quick Fix** (5 minutes):
```json
{
  "confluence_min_conditions": 2,  // Lowered from 3
  "confluence_threshold": 0.60      // Lowered from 0.65
}
```

**Expected Impact**: Increase trades by ~50%, catch more events but accept higher false positives

**Medium Fix** (if still missing events):
```json
{
  "crisis_composite_min": 0.30,     // Lowered from 0.35
  "volume_climax_3b_min": 0.40,     // Lowered from 0.50
  "wick_exhaustion_3b_min": 0.50    // Lowered from 0.60
}
```

**Expected Impact**: Increase trades significantly (~80-100/year), will need to re-tighten confluence

**Iterative Approach**:
1. Lower confluence to catch events
2. Raise exhaustion thresholds to filter noise
3. Iterate until 40-60 trades/year with acceptable recall

### Recipe 3: Wrong Regime Trades

**Symptom**: Trades occurring in risk_on regime, low win rate on those trades

**Quick Fix** (5 minutes):
```json
{
  "allowed_regimes": ["risk_off", "crisis"],  // Remove "neutral" if present
  "require_regime_or_drawdown": true,
  "drawdown_override_pct": 0.15  // Raised from 0.10
}
```

**Expected Impact**: Eliminate risk_on trades, only allow crisis/bear markets + deep crashes

**Validation**:
```bash
# Check regime distribution of trades
python bin/analyze_trade_regimes.py \
  --results results/s1_backtest.csv \
  --regime-file data/regime_labels.csv
```

**Target Distribution**:
- risk_off: 70-80% of trades
- crisis: 15-25% of trades
- neutral: 0-5% of trades (via drawdown override only)
- risk_on: 0% of trades

### Recipe 4: Borderline Signal Quality

**Symptom**: Many trades with confluence scores 0.65-0.67 (barely passing threshold)

**Analysis**:
```bash
# Check confluence score distribution
python bin/analyze_confluence_scores.py \
  --results results/s1_backtest.csv
```

**If scores clustered near threshold (0.65)**:
```json
{
  "confluence_threshold": 0.68  // Raise to create buffer
}
```

**If scores bimodal (two clusters)**:
- Cluster 1: 0.65-0.70 (marginal signals) → likely false positives
- Cluster 2: 0.75-0.85 (strong signals) → likely true events

**Action**: Raise threshold to 0.72 to select only strong signals

### Recipe 5: Regime-Specific Tuning

**Use Case**: Optimize separately for bull vs bear markets

**Approach**: Create regime-specific configs

**Bull Market Config** (`s1_bull_market.json`):
```json
{
  "confluence_threshold": 0.75,     // Very conservative
  "confluence_min_conditions": 4,   // Require all conditions
  "allowed_regimes": ["crisis"],    // Only trade extreme crises
  "drawdown_override_pct": 0.12     // Higher threshold for crashes
}
```

**Bear Market Config** (`s1_bear_market.json`):
```json
{
  "confluence_threshold": 0.60,     // More permissive
  "confluence_min_conditions": 3,   // Standard
  "allowed_regimes": ["risk_off", "crisis"],
  "drawdown_override_pct": 0.08     // Lower threshold
}
```

**Deployment**: Use regime router to switch configs dynamically

---

## 5. Validation Framework

### Validation Metrics

**Primary Metrics** (optimize for these):

| Metric | Target | Acceptable Range | Red Flag |
|--------|--------|------------------|----------|
| Profit Factor | >2.0 | 1.5-3.0 | <1.5 |
| Win Rate | 50-60% | 45-65% | <45% |
| Trades/Year | 40-60 | 30-80 | <20 or >100 |
| Event Recall | 50-70% | 40-80% | <40% |

**Secondary Metrics** (monitor for issues):

| Metric | Target | Red Flag |
|--------|--------|----------|
| Sharpe Ratio | >1.0 | <0.5 |
| Max Drawdown | <40% | >60% |
| Average R:R | 2-3:1 | <1.5:1 |
| False Positive Ratio | 10-15:1 | >25:1 |

### Event-Based Validation

**Critical**: Track whether S1 catches major capitulation events.

**Known Major Events** (2022-2024):
1. LUNA Death Spiral (May 12, 2022) - MUST CATCH
2. LUNA Final Capitulation (Jun 18, 2022) - MUST CATCH
3. FTX Collapse (Nov 9, 2022) - MUST CATCH
4. Japan Carry Unwind (Aug 5, 2024) - SHOULD CATCH

**Validation Process**:
```python
# bin/validate_event_recall.py
major_events = [
    {'date': '2022-05-12', 'name': 'LUNA Death Spiral', 'priority': 'MUST'},
    {'date': '2022-06-18', 'name': 'LUNA Final Cap', 'priority': 'MUST'},
    {'date': '2022-11-09', 'name': 'FTX Collapse', 'priority': 'MUST'},
    {'date': '2024-08-05', 'name': 'Japan Carry', 'priority': 'SHOULD'},
]

# Check if trades occurred within 24h of each event
results_df = pd.read_csv('results/s1_backtest.csv')
for event in major_events:
    event_date = pd.to_datetime(event['date'])
    window_start = event_date - pd.Timedelta(hours=12)
    window_end = event_date + pd.Timedelta(hours=12)

    trades_in_window = results_df[
        (results_df['entry_time'] >= window_start) &
        (results_df['entry_time'] <= window_end)
    ]

    if len(trades_in_window) > 0:
        print(f"✓ CAUGHT: {event['name']}")
    else:
        print(f"✗ MISSED: {event['name']} ({event['priority']})")
```

**Acceptable Miss Rates**:
- MUST events: 0-1 misses (catch 75-100%)
- SHOULD events: 1-2 misses (catch 50-75%)
- MAY events: any (nice to have)

### Statistical Validation

**Bootstrap Confidence Intervals**:

```python
# bin/bootstrap_validation.py
import numpy as np
from scipy import stats

def bootstrap_metric(trades, metric_func, n_bootstrap=1000):
    """Calculate bootstrap confidence interval for metric."""
    bootstrap_samples = []

    for _ in range(n_bootstrap):
        # Resample trades with replacement
        sample = trades.sample(n=len(trades), replace=True)
        metric_value = metric_func(sample)
        bootstrap_samples.append(metric_value)

    # Calculate 95% confidence interval
    ci_lower = np.percentile(bootstrap_samples, 2.5)
    ci_upper = np.percentile(bootstrap_samples, 97.5)

    return ci_lower, ci_upper

# Example: Win rate confidence interval
def calc_win_rate(trades):
    return (trades['pnl'] > 0).mean()

ci_lower, ci_upper = bootstrap_metric(results_df, calc_win_rate)
print(f"Win rate: {calc_win_rate(results_df):.1%} (95% CI: {ci_lower:.1%}-{ci_upper:.1%})")
```

**Interpretation**:
- Narrow CI (e.g., 52-58%) → consistent performance
- Wide CI (e.g., 40-70%) → high variance, need more data

### Monte Carlo Simulation

**Purpose**: Assess robustness to parameter uncertainty

```python
# bin/monte_carlo_validation.py
def monte_carlo_validation(base_config, n_simulations=100):
    """Test robustness by adding noise to parameters."""
    results = []

    for _ in range(n_simulations):
        # Add ±10% noise to key parameters
        config = base_config.copy()
        config['confluence_threshold'] *= np.random.uniform(0.9, 1.1)
        config['volume_climax_3b_min'] *= np.random.uniform(0.9, 1.1)
        config['wick_exhaustion_3b_min'] *= np.random.uniform(0.9, 1.1)

        # Run backtest with noisy parameters
        result = run_backtest(config)
        results.append(result['profit_factor'])

    # Analyze distribution
    pf_mean = np.mean(results)
    pf_std = np.std(results)
    pf_min = np.min(results)

    print(f"Profit Factor: {pf_mean:.2f} ± {pf_std:.2f} (min: {pf_min:.2f})")

    # Check if worst case still acceptable
    if pf_min < 1.0:
        print("WARNING: Some parameter variations produce losing strategies")
```

**Interpretation**:
- Low variance (std <0.3) → robust to parameter changes
- High variance (std >0.5) → sensitive to parameters, may be overfit

---

## 6. Advanced Topics

### Multi-Regime Optimization

**Challenge**: Optimal parameters differ between bull/bear markets.

**Approach 1**: Unified config with regime routing

```json
{
  "routing": {
    "risk_on": {
      "weights": {"liquidity_vacuum": 0.3},
      "confluence_threshold_delta": 0.10  // Raise threshold by 0.10 in risk_on
    },
    "risk_off": {
      "weights": {"liquidity_vacuum": 1.5},
      "confluence_threshold_delta": 0.0  // Baseline threshold
    }
  }
}
```

**Approach 2**: Separate configs per regime

- `s1_bull_config.json` → deployed when regime = risk_on
- `s1_bear_config.json` → deployed when regime = risk_off/crisis

**Implementation**: Regime router automatically switches configs

### Adaptive Thresholds

**Concept**: Adjust thresholds based on recent market behavior

**Example**: Volume climax threshold adapts to market volatility

```python
# Adaptive volume threshold
vix_current = df['VIX'].iloc[-1]
vix_baseline = 20.0

# Lower threshold in high volatility (easier to fire)
volume_threshold = 0.50 * (vix_baseline / vix_current)
volume_threshold = np.clip(volume_threshold, 0.30, 0.70)
```

**Benefits**: Automatically adjusts to regime changes
**Risks**: More complex, harder to debug

### Feature Engineering for Confluence

**Current**: Confluence uses 10 weighted features
**Opportunity**: Engineer new features to improve signal quality

**Example New Features**:

1. **Exchange Stress Index**: Composite of funding + OI + liquidations
   ```python
   exchange_stress = (
       0.4 * funding_z_score +
       0.3 * oi_change_z_score +
       0.3 * liquidation_volume
   )
   ```

2. **Multi-Timeframe Confluence**: Check if capitulation visible across timeframes
   ```python
   mtf_confluence = (
       (capitulation_depth_1h < -0.20).astype(int) +
       (capitulation_depth_4h < -0.25).astype(int) +
       (capitulation_depth_1d < -0.30).astype(int)
   ) / 3.0  # Normalize to [0,1]
   ```

3. **Wyckoff Event Alignment**: Check if Wyckoff selling climax detected
   ```python
   wyckoff_alignment = 1.0 if 'selling_climax' in wyckoff_events else 0.0
   ```

**Process**:
1. Implement new feature in runtime enrichment
2. Add to confluence weights (small weight initially)
3. Re-optimize confluence weights with new feature
4. Validate improvement on out-of-sample data

### Hyperband Pruning

**Problem**: Optuna optimization slow (100 trials x 2 years data = hours)

**Solution**: Use Hyperband pruner to early-stop unpromising trials

```python
# bin/optimize_s1_thresholds_fast.py
import optuna
from optuna.pruners import HyperbandPruner

def objective(trial):
    # ... (parameter sampling)

    # Run backtest in chunks and report intermediate results
    for month in range(1, 25):  # 24 months
        results = run_backtest_partial(config, start_month=1, end_month=month)
        trial.report(results['profit_factor'], step=month)

        # Allow pruner to stop unpromising trials early
        if trial.should_prune():
            raise optuna.TrialPruned()

    return results['profit_factor']

# Create study with Hyperband pruner
study = optuna.create_study(
    direction='maximize',
    pruner=HyperbandPruner()
)

study.optimize(objective, n_trials=200)
```

**Speedup**: 3-5x faster (stops bad trials after 3-6 months instead of running full 24 months)

### Ensemble Methods

**Concept**: Run multiple S1 configs in parallel and combine signals

**Approach 1**: Vote ensemble
- Config A (conservative): `confluence_threshold=0.75`, weight=1.0
- Config B (balanced): `confluence_threshold=0.65`, weight=1.5
- Config C (aggressive): `confluence_threshold=0.55`, weight=0.5

**Signal Logic**:
- If 2+ configs fire → take trade (high confidence)
- If 1 config fires → skip (low confidence)

**Approach 2**: Weighted ensemble
```python
signal_a = config_a.check_entry() * 1.0  # Conservative
signal_b = config_b.check_entry() * 1.5  # Balanced
signal_c = config_c.check_entry() * 0.5  # Aggressive

ensemble_score = (signal_a + signal_b + signal_c) / 3.0

if ensemble_score > 0.7:
    take_trade()
```

**Benefits**: More robust, less sensitive to single config failure
**Risks**: More complex, requires tuning ensemble weights

---

## Summary: Tuning Workflow

### Standard Tuning Workflow (Quarterly)

1. **Assess Need** (15 min)
   - Check trade frequency (target 40-60/year)
   - Check win rate (target 50-60%)
   - Check event recall (target 50-70%)
   - If 2+ metrics off-target → proceed to tuning

2. **Define Scope** (10 min)
   - Tier 1 only (quick tune): 50 trials, 1-2 hours
   - Tier 1+2 (full tune): 100-200 trials, 3-6 hours
   - Multi-regime: 200+ trials, 6-12 hours

3. **Run Optimization** (1-6 hours depending on scope)
   ```bash
   python bin/optimize_s1_thresholds.py \
     --config configs/s1_v2_production.json \
     --n-trials 100 \
     --n-jobs 8
   ```

4. **Analyze Results** (30 min)
   - Review Pareto frontier
   - Check parameter importance
   - Select configuration matching risk tolerance

5. **Validate Out-of-Sample** (1 hour)
   ```bash
   python bin/backtest.py --config configs/s1_optimized.json --start 2023-07-01 --end 2024-11-18
   ```
   - Compare in-sample vs OOS performance
   - Check event recall on OOS period
   - Verify trade frequency in target range

6. **Deploy** (15 min)
   - Copy to production config
   - Update metadata (optimization date, performance)
   - Monitor first week for issues

**Total Time**: 3-9 hours (depending on scope)

---

## Troubleshooting Optimization

### Issue: Optimization Converging to Extreme Parameters

**Symptom**: Best trials have `confluence_threshold=0.50` or `volume_climax_3b_min=0.80`

**Diagnosis**: Overfitting to training data or poor objective function

**Solution**:
1. Add regularization: penalize extreme parameter values
   ```python
   param_penalty = (
       abs(confluence_threshold - 0.65) * 0.1 +
       abs(volume_climax_3b_min - 0.50) * 0.1
   )
   score = profit_factor - param_penalty
   ```

2. Constrain search space: narrow ranges around sensible defaults

3. Use multi-objective: optimize PF AND trade frequency simultaneously

### Issue: Out-of-Sample Performance Collapse

**Symptom**: In-sample PF 2.8, out-of-sample PF 1.2

**Diagnosis**: Severe overfitting

**Solution**:
1. Reduce number of parameters optimized (Tier 1 only)
2. Increase training data (use longer period)
3. Use walk-forward validation (multiple OOS periods)
4. Add regularization (penalize complexity)

### Issue: Optimization Too Slow

**Symptom**: 100 trials taking >12 hours

**Solution**:
1. Use Hyperband pruner (3-5x speedup)
2. Reduce trial count (50 trials sufficient for Tier 1 only)
3. Use shorter training period (12 months instead of 24)
4. Parallelize (increase `n_jobs`)

---

## References

- **Operator Guide**: `docs/S1_V2_OPERATOR_GUIDE.md` - Deployment and monitoring
- **Known Issues**: `docs/S1_V2_KNOWN_ISSUES.md` - Edge cases and limitations
- **Implementation**: `engine/strategies/archetypes/bear/liquidity_vacuum_runtime.py` - Feature calculations
- **Optuna Docs**: https://optuna.readthedocs.io/ - Optimization framework
- **Walk-Forward Validation**: `bin/validate_walk_forward.py` - Validation script
