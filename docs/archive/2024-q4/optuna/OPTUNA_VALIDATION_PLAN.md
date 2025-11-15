# Optuna Trap Optimization - Validation & Critique Plan

**Status**: Optimization 100/200 trials complete (ETA: 4.5 hours)
**Date**: 2025-11-06
**Purpose**: Rigorous validation before accepting "optimized" parameters

---

## 🚨 CRITICAL CONCERNS IDENTIFIED

### 1. **Position Sizing Artifacts**
**Issue**: Optimization ran with `dynamic_kelly_archetype` active.
**Risk**: "Better" parameters might just trigger larger position sizes, not better entries.
**Fix**: Re-run winner with **fixed sizing** to isolate entry quality.

### 2. **Suspicious Train/Val Split**
**Current Best (Trial #0)**:
- Train (2022-2023): PF 0.88, WR 41.4%, PNL -$222 ❌
- Val (2024): PF 3.37, WR 66.0%, PNL +$1,479 ✅

**Concern**: This divergence suggests:
- Regime shift (2024 structure differs from 2022-2023)
- Overfitting to 2024's specific volatility
- Position sizing masking poor entry quality
- Accidental leakage

**Fix**: Rolling OOS validation + regime stratification.

### 3. **Weak Objective Function**
**Current**: `score = PF × WR`
**Problems**:
- Can be gamed by a few fat wins
- Doesn't account for tail risk
- No stability consideration
- Hard constraints (DD>10%, trades<20) prevent gradient learning

**Better Options**:
```python
# Option A: Risk-adjusted expectancy
expectancy = total_pnl / total_trades_risked_R
stability = 1.0 / (1.0 + std_R_per_trade)
score = (expectancy * sqrt(trades)) * stability
score -= 5.0 * max(0, DD - 0.10)  # Soft penalty
score -= 2.0 * max(20 - trades, 0)  # Soft penalty

# Option B: Calmar-like
score = (annual_return / max_drawdown) - lambda * volatility

# Option C: Multi-objective (Pareto)
optimize([↑ PF, ↑ WR, ↓ DD, ↑ trades]) → pick Pareto front
```

### 4. **Single-Period Validation**
**Current**: Train(2022-2023) → Test(2024)
**Problem**: One period can be unrepresentative.

**Fix**: Rolling windows:
- Train: 2022-H1 → Test: 2022-H2
- Train: 2022 → Test: 2023
- Train: 2022-H1:2023-H1 → Test: 2023-H2
- Train: 2022-2023 → Test: 2024

Aggregate: median PF, median expectancy, worst-case DD.

### 5. **No Regime Stratification**
**Problem**: Trap archetype behaves differently in RISK_ON vs RISK_OFF.
**Current**: Optimization sees aggregated metrics across all regimes.

**Fix**:
- Report PF/WR breakdown by regime
- Consider regime-conditioned parameters
- Or minimize across regimes (ensure robust in all conditions)

### 6. **Potential Data Leakage**
**Concerns**:
- ML filter (`btc_trade_quality_filter_v1.pkl`) - No metadata, can't verify ≤2023 training
- Regime GMM (`regime_gmm_v3.1_fixed.pkl`) - No metadata
- Feature availability: Did 2022 have same columns as 2024?

**Action**:
- Verify ML filter was trained on ≤2023 only
- Confirm GMM fitted on ≤2023
- Check feature parity across years

### 7. **Missing Realism Factors**
**Current**: No slippage, no spread costs.
**Risk**: Tight stops look artificially perfect.

**Fix**: Add fixed 2-3bp cost per trade + slippage model.

---

## ✅ VALIDATION CHECKLIST (When Optimization Completes)

### Phase 1: Immediate Sanity Checks

- [ ] **Check Data Leakage**
  - Verify ML filter trained on ≤2023
  - Verify GMM fitted on ≤2023
  - Confirm no future data used

- [ ] **Inspect Best Trial Parameters**
  ```bash
  cat results/optuna_trap_v10_full/best_params.json
  ```
  - Are they sensible? (e.g., stop_mult not pathologically small)
  - Do they make trading sense?

- [ ] **Check Trial Distribution**
  ```python
  import pandas as pd
  df = pd.read_csv('results/optuna_trap_v10_full/trials.csv')

  # How many trials beat baseline?
  baseline_score = 0.364932
  print(f"Trials better than baseline: {(df['value'] > baseline_score).sum()}/200")

  # Check score distribution
  df['value'].hist(bins=50)

  # Look for outliers
  print(df.nlargest(10, 'value'))
  ```

---

### Phase 2: Fixed-Size Validation (CRITICAL)

**Goal**: Isolate entry quality from position sizing artifacts.

**Method**: Re-run best trial with fixed sizing:

```python
# Modify configs
bull_config['position_sizing'] = {
    'mode': 'fixed_fractional',
    'base_risk_per_trade_pct': 0.8,  # Fixed 0.8% risk
    'max_risk_per_trade_pct': 0.8,
    'kelly_fraction': 1.0,
    'confidence_scaling': False,
    'archetype_quality_weight': 0.0  # DISABLE quality sizing
}

# Run on 2022-2024
results_fixed_size = backtest(optimized_params, fixed_sizing=True)

# Compare to baseline with same fixed sizing
results_baseline_fixed = backtest(baseline_params, fixed_sizing=True)
```

**Acceptance Criteria**:
- Optimized params still beat baseline with fixed sizing
- PF improvement ≥ 10%
- WR improvement ≥ 5%
- No degradation in max DD

**If this fails**: Position sizing was masking poor entries. Reject.

---

### Phase 3: Rolling OOS Validation

**Windows**:
1. **2022-H1 (Jan-Jun) → 2022-H2 (Jul-Dec)**
   - Train on H1, test on H2
   - Measure: PF, WR, DD, expectancy

2. **2022 → 2023**
   - Train on 2022, test on 2023
   - Check regime stability

3. **2022-2023-H1 → 2023-H2**
   - Expanding window

4. **2022-2023 → 2024** (already have this)

**Aggregate Metrics**:
```python
# For each window, get OOS metrics
oos_results = {
    '22H1->22H2': run_backtest(...),
    '22->23': run_backtest(...),
    '22-23H1->23H2': run_backtest(...),
    '22-23->24': run_backtest(...)
}

# Calculate robustness
median_pf = np.median([r['pf'] for r in oos_results.values()])
min_pf = np.min([r['pf'] for r in oos_results.values()])
std_pf = np.std([r['pf'] for r in oos_results.values()])

median_expectancy = np.median([r['avg_r'] for r in oos_results.values()])
worst_dd = np.max([r['dd'] for r in oos_results.values()])
```

**Acceptance Criteria**:
- Median PF > 1.3
- Min PF > 1.0 (profitable in all periods)
- Worst DD < 12%
- No single period dominates the aggregate

---

### Phase 4: Regime Stratification

**Goal**: Ensure params work in all market regimes.

**Method**:
```python
# For 2022-2024 backtest with optimized params
trades_df = pd.read_csv('results/.../trades.csv')

# Group by regime
regime_breakdown = trades_df.groupby('regime_at_entry').agg({
    'net_pnl': ['sum', 'mean'],
    'trade_id': 'count',
    'is_win': 'mean'  # win rate
})

# Calculate PF by regime
for regime in ['RISK_ON', 'RISK_OFF', 'NEUTRAL', 'CRISIS', 'TRANSITIONAL']:
    regime_trades = trades_df[trades_df['regime_at_entry'] == regime]
    wins = regime_trades[regime_trades['net_pnl'] > 0]['net_pnl'].sum()
    losses = abs(regime_trades[regime_trades['net_pnl'] < 0]['net_pnl'].sum())
    pf = wins / losses if losses > 0 else np.inf
    wr = (regime_trades['net_pnl'] > 0).mean()

    print(f"{regime}: PF={pf:.2f}, WR={wr:.1%}, trades={len(regime_trades)}")
```

**Acceptance Criteria**:
- PF > 1.0 in at least 4/5 regimes
- No regime with catastrophic losses (>-$500)
- CRISIS/RISK_OFF should not be severely negative

**Red Flags**:
- All profits come from RISK_ON only
- CRISIS regime shows -$800 PNL
- TRANSITIONAL has <5 trades (can't assess)

---

### Phase 5: Trade-Level Diagnostics

**Goal**: Understand *how* the params changed behavior.

**Method**: Compare optimized vs baseline trade-by-trade on same dates.

```python
# Load both trade logs
trades_baseline = pd.read_csv('results/baseline/.../trades.csv')
trades_optimized = pd.read_csv('results/optimized/.../trades.csv')

# Merge on entry time (same bars)
merged = pd.merge(
    trades_baseline,
    trades_optimized,
    on='entry_time',
    suffixes=('_base', '_opt')
)

# Analyze differences
print("Trades taken by both:", len(merged))
print("Only baseline:", len(trades_baseline) - len(merged))
print("Only optimized:", len(trades_optimized) - len(merged))

# For shared trades, compare outcomes
shared = merged[merged['entry_price_base'] == merged['entry_price_opt']]
print("\nShared trade PNL diff:")
print((shared['net_pnl_opt'] - shared['net_pnl_base']).describe())

# Where did improvements come from?
improved = shared[shared['net_pnl_opt'] > shared['net_pnl_base']]
print(f"\nImproved trades: {len(improved)}")
print(f"Improvement came from:")
print(f"  - Tighter stops: {(improved['stop_price_opt'] > improved['stop_price_base']).sum()}")
print(f"  - Longer holds: {(improved['bars_held_opt'] > improved['bars_held_base']).sum()}")
```

**Look For**:
- Are improvements from better entries or better exits?
- Did tighter stops reduce loss size?
- Were any big wins eliminated?

---

### Phase 6: Session & Temporal Analysis

**Goal**: Check for session-specific degradation.

**Method**:
```python
# Add session labels (UTC times)
def get_session(timestamp):
    hour = timestamp.hour
    if 0 <= hour < 8: return 'ASIA'
    elif 8 <= hour < 16: return 'EUROPE'
    else: return 'US'

trades_df['session'] = trades_df['entry_time'].apply(get_session)

# Breakdown by session
session_stats = trades_df.groupby('session').agg({
    'net_pnl': 'sum',
    'is_win': 'mean',
    'trade_id': 'count'
})

print(session_stats)
```

**Red Flags**:
- One session has WR < 30%
- Asia session loses -$600 while others profit
- Weekend vs weekday split shows large divergence

---

### Phase 7: Slippage & Cost Sensitivity

**Goal**: Ensure params survive realistic costs.

**Method**: Add 3bp cost per trade + 1bp slippage on stops.

```python
# Adjust PNL for costs
trades_df['net_pnl_with_costs'] = trades_df['net_pnl'] - (
    trades_df['entry_price'] * trades_df['size'] * 0.0003  # 3bp entry
    + trades_df['exit_price'] * trades_df['size'] * 0.0003  # 3bp exit
    + trades_df['stop_slippage'] * 0.0001  # 1bp stop slippage
)

# Recalculate metrics
pf_with_costs = wins_with_costs / losses_with_costs
wr_with_costs = (trades_df['net_pnl_with_costs'] > 0).mean()
total_pnl_with_costs = trades_df['net_pnl_with_costs'].sum()
```

**Acceptance Criteria**:
- PF with costs > 1.2
- Total PNL with costs > +$800/year
- Improvement over baseline persists

---

## 🔄 NEXT ITERATION IMPROVEMENTS

### For Future Optuna Runs (v2):

#### 1. **Fixed Sizing During Optimization**
```python
# In optimizer config
'position_sizing': {
    'mode': 'fixed_fractional',
    'base_risk_per_trade_pct': 1.0,  # Fixed 1R
    'confidence_scaling': False,
    'archetype_quality_weight': 0.0
}
```

After locking entry params, re-tune sizing separately.

#### 2. **Better Objective Function**
```python
def objective(trial):
    # ... run backtest ...

    # Calculate robust metrics
    expectancy_R = total_pnl / (total_trades * risk_per_trade)
    stability = 1.0 / (1.0 + np.std(R_per_trade))

    # Soft penalties
    dd_penalty = 5.0 * max(0, dd - 0.10)
    trade_penalty = 2.0 * max(20 - total_trades, 0)

    # Score
    score = (expectancy_R * np.sqrt(total_trades)) * stability
    score -= dd_penalty
    score -= trade_penalty

    return score
```

#### 3. **Rolling Windows Built-In**
```python
# Define multiple train/test splits
splits = [
    ('2022-01-01', '2022-06-30', '2022-07-01', '2022-12-31'),
    ('2022-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
    ('2022-01-01', '2023-06-30', '2023-07-01', '2023-12-31'),
    ('2022-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
]

# Optimize for median performance across all splits
for split in splits:
    results = run_backtest(train_start, train_end, test_start, test_end)
    scores.append(results['score'])

return np.median(scores)  # Or min for robustness
```

#### 4. **Regime-Stratified Objective**
```python
# Get regime breakdown
regime_pfs = {}
for regime in ['RISK_ON', 'RISK_OFF', 'NEUTRAL', 'CRISIS']:
    regime_trades = trades[trades['regime'] == regime]
    pf = calculate_pf(regime_trades)
    regime_pfs[regime] = pf

# Weighted or min
score = np.mean(list(regime_pfs.values()))  # Average
# OR
score = np.min(list(regime_pfs.values()))   # Worst-case
```

#### 5. **Pruning for Speed**
```python
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(  # Stop bad trials early
        n_startup_trials=20,
        n_warmup_steps=5
    )
)
```

#### 6. **Reduced Logging**
```python
# In backtest engine
import logging
logging.getLogger('bin.backtest_knowledge_v2').setLevel(logging.WARNING)
```

Only log detailed info for best trials.

#### 7. **Feature/Regime Caching**
```python
# Pre-compute once
regime_labels = regime_detector.classify(df)
df['regime_label'] = regime_labels

# Save
df.to_parquet('data/features_with_regime_2022_2024.parquet')

# In trials, just load pre-computed
df = pd.read_parquet('data/features_with_regime_2022_2024.parquet')
```

Saves ~10-15s per trial.

---

## 🎯 DECISION CRITERIA

### Accept Optimized Parameters If:
- ✅ Fixed-size validation shows >10% PF improvement
- ✅ Rolling OOS shows median PF > 1.3, min PF > 1.0
- ✅ All regimes have PF > 0.9 (or 4/5 > 1.0)
- ✅ Trade diagnostics show improvements from entries/stops, not sizing
- ✅ Slippage test maintains PF > 1.2
- ✅ No single session/period dominates results

### Reject If:
- ❌ Fixed-size validation fails (no improvement)
- ❌ OOS rolling shows one period dominates (e.g., 2024 only)
- ❌ Regime breakdown shows CRISIS/RISK_OFF catastrophic
- ❌ Trade diagnostics reveal sizing artifacts
- ❌ Slippage test drops below baseline

### Conditional Accept (Re-run with Fixes):
- ⚠️ Passes most tests but fails regime robustness
  → Re-run with regime-stratified objective
- ⚠️ Passes but suspicious train/val split
  → Re-run with rolling windows
- ⚠️ Passes but uncertainty about sizing
  → Re-run v2 with fixed sizing

---

## 📝 NOTES FOR FUTURE

**Model Training Metadata**: Always save:
```python
metadata = {
    'train_start': '2020-01-01',
    'train_end': '2023-12-31',
    'features': list(feature_names),
    'model_version': 'v1',
    'timestamp': datetime.now().isoformat()
}

with open('models/btc_trade_quality_filter_v1.pkl.metadata', 'w') as f:
    json.dump(metadata, f, indent=2)
```

This prevents leakage concerns.

---

**Generated**: 2025-11-06
**Current Run**: 100/200 trials (ETA 4.5 hours)
**Next Action**: Wait for completion, then execute this validation plan
