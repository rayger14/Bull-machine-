# Walk-Forward Validation Framework

Comprehensive validation system to ensure trading configs aren't overfit and perform consistently across different market regimes.

## Overview

The walk-forward validation framework performs rigorous out-of-sample testing using three complementary approaches:

1. **Temporal Validation**: Train → Validation → OOS testing with degradation checks
2. **Cross-Regime Analysis**: Performance breakdown by macro regime (risk_on, neutral, risk_off, crisis)
3. **Statistical Testing**: Permutation tests to validate edge significance

## Quick Start

### 1. Test the Framework

```bash
# Create sample configs and verify framework works
python bin/test_validation_framework.py

# Run validation on test configs
python bin/validate_walk_forward.py \
    --configs results/validation/test_configs \
    --output results/validation/test_run/ \
    --asset BTC

# Visualize results
python bin/visualize_validation_results.py \
    --input results/validation/test_run/
```

### 2. Validate Optimization Results

```bash
# Validate Optuna study results
python bin/validate_walk_forward.py \
    --configs results/phase2_optimization/optimization_study.db \
    --study-name bear_phase2_tuning \
    --output results/validation/phase2_validation/ \
    --asset BTC \
    --min-trials 50

# Generate visualizations
python bin/visualize_validation_results.py \
    --input results/validation/phase2_validation/

# Review summary
cat results/validation/phase2_validation/summary_report.md
```

## Validation Methodology

### Walk-Forward Periods

| Period | Date Range | Purpose | Regime |
|--------|-----------|---------|--------|
| Train | 2022-01-01 to 2022-06-30 | Parameter optimization | Bear market (H1) |
| Validation | 2022-07-01 to 2022-12-31 | Overfitting detection | Bear market (H2) |
| OOS | 2023-01-01 to 2023-12-31 | True out-of-sample test | Transition regime |

### Acceptance Criteria

All criteria must be met for a config to pass validation:

| Check | Threshold | Rationale |
|-------|-----------|-----------|
| **Validation Degradation** | Val PF ≥ 0.8 × Train PF | Max 20% degradation acceptable |
| **OOS Edge** | OOS PF ≥ 1.1 | Must maintain profitable edge |
| **OOS Drawdown** | Max DD < 25% | Risk management requirement |
| **Regime Consistency** | All regime PF ≥ 0.8 | Avoid regime-specific overfitting |
| **Sample Size** | ≥ 5 trades per period | Statistical validity |
| **Statistical Significance** | p-value < 0.05 | Edge not due to chance |

### Red Flags (Auto-Reject)

- Train PF = 2.5, Val PF = 0.9 → **Severe overfitting**
- OOS trade count < 5 → **Insufficient sample size**
- Any period with PF < 0.7 → **No edge**
- p-value > 0.05 → **Edge may be spurious**

## Output Structure

```
results/validation/walk_forward/
├── summary_report.md              # Main validation report
├── validation_summary.csv         # All configs in tabular format
├── pf_degradation.png            # Train vs Val/OOS scatter
├── regime_heatmap.png            # Config × regime performance
├── metric_distributions.png       # Passed vs failed distributions
├── period_comparison.png          # Top configs across periods
├── risk_return_scatter.png        # OOS Sharpe vs DD
│
└── config_001/                    # Individual config results
    ├── train_metrics.json
    ├── val_metrics.json
    ├── oos_metrics.json
    └── regime_breakdown.csv
```

## Understanding Results

### Summary Report

The `summary_report.md` contains:

1. **Overview**: Pass/fail counts and rates
2. **Validation Criteria**: Threshold reference table
3. **Top Performers**: Configs that passed all checks
4. **Failure Analysis**: Breakdown of rejection reasons
5. **Recommendations**: Production candidates and next steps

### Metrics Files

Each period (train/val/oos) has a metrics JSON with:

```json
{
  "period_name": "oos",
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "total_trades": 42,
  "win_rate": 0.62,
  "profit_factor": 1.45,
  "sharpe_ratio": 0.82,
  "max_drawdown": 0.18,
  "total_return": 0.23,
  "avg_r_multiple": 0.54,
  "regime_trades": {
    "risk_on": 15,
    "neutral": 20,
    "risk_off": 5,
    "crisis": 2
  },
  "regime_pf": {
    "risk_on": 1.2,
    "neutral": 1.5,
    "risk_off": 1.8,
    "crisis": 0.9
  }
}
```

### Regime Breakdown CSV

Performance by macro regime across all periods:

```csv
regime,train_trades,train_pf,train_wr,val_trades,val_pf,val_wr,oos_trades,oos_pf,oos_wr
risk_on,25,1.3,0.56,20,1.2,0.55,15,1.2,0.60
neutral,40,1.5,0.60,35,1.4,0.58,20,1.5,0.65
risk_off,30,1.7,0.63,28,1.6,0.61,5,1.8,0.80
crisis,5,0.8,0.40,7,0.9,0.43,2,0.9,0.50
```

## Visualization Outputs

### 1. PF Degradation Plots

Two scatter plots showing:
- **Left**: Train PF vs Validation PF
- **Right**: Train PF vs OOS PF

**Interpretation**:
- Points near diagonal (1:1 line) → Stable performance
- Points below 0.8× line → Failed degradation check
- Green points = passed all checks
- Red X = failed validation

### 2. Regime Heatmap

Performance by config × regime (top 20 configs shown).

**Color Scale**:
- Green (PF > 1.3): Strong performance
- Yellow (PF ≈ 1.0): Breakeven
- Red (PF < 0.8): Losing in this regime

**Look for**:
- Configs with consistent green across all regimes
- Configs with red in crisis regime (acceptable if rare)

### 3. Metric Distributions

Histograms comparing passed vs failed configs across 6 metrics.

**Key Insights**:
- Separation between passed/failed distributions
- Which metrics are most discriminative
- Typical ranges for successful configs

### 4. Period Comparison

Bar chart showing top 10 configs' PF across train/val/oos.

**Look for**:
- Consistent bars (similar height across periods)
- Configs where OOS ≥ Val ≥ Train × 0.8

### 5. Risk-Return Scatter

OOS Sharpe Ratio vs OOS Max Drawdown.

**Quadrants**:
- **Upper Left** (low DD, high Sharpe): Ideal zone
- **Upper Right** (high DD, high Sharpe): High risk/reward
- **Lower Left** (low DD, low Sharpe): Low risk/reward
- **Lower Right** (high DD, low Sharpe): Avoid

**Point Size**: Scaled by Profit Factor

## Advanced Usage

### Validate Specific Trials

```bash
# Only validate trials 50-100 from Optuna study
python bin/validate_walk_forward.py \
    --configs results/phase2_optimization/optimization_study.db \
    --study-name bear_phase2_tuning \
    --min-trials 50 \
    --max-configs 50 \
    --output results/validation/trials_50_100/
```

### Custom Time Periods

Edit `bin/validate_walk_forward.py`:

```python
self.periods = {
    'train': ('2022-01-01', '2022-04-30'),   # Q1 2022
    'val': ('2022-05-01', '2022-08-31'),     # Q2 2022
    'oos': ('2022-09-01', '2022-12-31'),     # Q3-Q4 2022
}
```

### Adjust Thresholds

```python
self.thresholds = {
    'val_degradation_max': 0.15,      # Stricter: 15% max degradation
    'oos_pf_min': 1.2,                # Higher bar: 1.2 PF minimum
    'oos_dd_max': 0.20,               # Tighter risk: 20% max DD
    'min_trades_per_period': 10,      # Larger sample: 10 trades
    'permutation_alpha': 0.01,        # Stricter: 1% significance
    'regime_pf_min': 0.9,             # More lenient: 0.9 PF min
}
```

## Interpreting Common Patterns

### Pattern 1: High Train, Low OOS

```
Train PF: 2.5
Val PF: 1.0
OOS PF: 0.8
```

**Diagnosis**: Severe overfitting
**Action**: Reject config, increase regularization

### Pattern 2: Stable Across Periods

```
Train PF: 1.4
Val PF: 1.3
OOS PF: 1.4
```

**Diagnosis**: Robust strategy
**Action**: Strong production candidate

### Pattern 3: Regime-Specific

```
Risk On PF: 2.0
Neutral PF: 1.5
Risk Off PF: 0.6
Crisis PF: 0.5
```

**Diagnosis**: Only works in specific regimes
**Action**: Use with regime filter or reject

### Pattern 4: High Variance

```
OOS PF: 1.8
OOS Trades: 3
p-value: 0.15
```

**Diagnosis**: Too few trades, edge not significant
**Action**: Reject, insufficient evidence

## Production Deployment Workflow

### Step 1: Validate All Candidates

```bash
python bin/validate_walk_forward.py \
    --configs results/phase2_optimization/optimization_study.db \
    --output results/validation/production_candidates/ \
    --asset BTC
```

### Step 2: Review Summary

```bash
cat results/validation/production_candidates/summary_report.md
```

### Step 3: Select Top Configs

```bash
# Extract top 5 passed configs
head -n 6 results/validation/production_candidates/validation_summary.csv \
    | tail -n 5 > results/validation/top_5_configs.csv
```

### Step 4: Deep Dive Analysis

For each top config:

1. Review individual metrics files
2. Check regime breakdown CSV
3. Verify no red flags in specific regimes
4. Compare against existing production strategies

### Step 5: Ensemble Construction

Combine 3-5 diverse configs:

```python
# configs/production_ensemble_v1.json
{
    "ensemble_members": [
        "config_042",  # High PF, conservative
        "config_087",  # Balanced, all-weather
        "config_123",  # Active trading, good Sharpe
    ],
    "weights": [0.4, 0.4, 0.2],  # Based on OOS Sharpe
    "rebalance_frequency": "monthly"
}
```

### Step 6: Final Validation

Run ensemble through validation framework:

```bash
python bin/validate_ensemble.py \
    --configs configs/production_ensemble_v1.json \
    --output results/validation/ensemble_final/
```

## Troubleshooting

### Issue: No configs passed

**Possible causes**:
1. Optimization overfitted to training period
2. Thresholds too strict
3. Market regime shifted significantly

**Solutions**:
- Re-run optimization with regularization
- Relax thresholds slightly (e.g., 15% degradation)
- Use ensemble of near-pass configs

### Issue: Low p-values (edge not significant)

**Possible causes**:
1. Too few trades
2. Edge is weak
3. High variance in outcomes

**Solutions**:
- Increase sample size (longer periods)
- Improve signal quality (tighter filters)
- Use portfolio-level validation

### Issue: Regime inconsistency

**Possible causes**:
1. Strategy optimized for specific regime
2. Insufficient data in some regimes
3. Regime classifier issues

**Solutions**:
- Use regime-specific strategies
- Aggregate rare regimes (crisis + risk_off)
- Validate regime labels

## Performance Benchmarks

Expected runtime on standard hardware:

| Configs | Periods | Trades/Config | Runtime |
|---------|---------|---------------|---------|
| 10 | 3 | 50 | 5 min |
| 50 | 3 | 50 | 25 min |
| 100 | 3 | 50 | 50 min |
| 500 | 3 | 50 | 4 hours |

**Optimization tips**:
- Use `--max-configs` to limit validation
- Run in parallel (split configs across processes)
- Cache feature store in memory

## Best Practices

### 1. Always Run Full Validation

Don't skip validation to save time. Overfitting is costly in live trading.

### 2. Validate Multiple Assets

If trading multiple assets, validate on each separately:

```bash
for asset in BTC ETH SOL; do
    python bin/validate_walk_forward.py \
        --configs results/optimization/${asset}_study.db \
        --asset $asset \
        --output results/validation/${asset}/
done
```

### 3. Document Decisions

Record why configs were selected/rejected:

```bash
# results/validation/selection_log.md
Config 042: SELECTED - Strong OOS PF (1.45), low DD (18%)
Config 087: SELECTED - All-regime consistency
Config 123: REJECTED - Failed p-value test (0.08)
```

### 4. Monitor Live vs Validation

Track actual production performance against validation metrics:

```python
validation_oos_pf = 1.45
live_pf_30d = 1.38  # Within 5% → Good

if live_pf_30d < validation_oos_pf * 0.8:
    alert("Strategy degrading, re-evaluate")
```

### 5. Re-validate Periodically

Re-run validation quarterly with updated data:

```bash
# Q1 2024 re-validation
python bin/validate_walk_forward.py \
    --configs configs/production_ensemble_v1.json \
    --output results/validation/2024_Q1_revalidation/ \
    --start-date 2023-01-01 \
    --end-date 2024-03-31
```

## Related Documentation

- **Optimization Guide**: `docs/OPTIMIZATION_GUIDE.md`
- **Backtest Engine**: `docs/BACKTEST_ENGINE_V2.md`
- **Regime Classification**: `docs/REGIME_CLASSIFIER_GUIDE.md`
- **Production Deployment**: `docs/PRODUCTION_DEPLOYMENT.md`

## Support

For issues or questions:

1. Check troubleshooting section above
2. Review example outputs in `results/validation/examples/`
3. Consult optimization documentation
4. Open issue with validation logs

---

**Framework Version**: 1.0
**Last Updated**: 2025-11-20
**Maintainer**: Bull Machine V2 Team
