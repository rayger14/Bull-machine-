# Calibration Guide

**How to sync archetype parameters with Optuna optimization trials**

---

## Overview

Archetype performance depends on calibrated thresholds (funding_threshold, exhaustion_threshold, etc.). These are optimized using Optuna and stored in SQLite databases. This guide explains how to extract and apply optimized calibrations.

---

## Quick Start

```bash
# Apply all optimized calibrations
python bin/apply_optimized_calibrations.py --all

# Or apply to specific archetypes
python bin/apply_optimized_calibrations.py --s1 --s4 --s5

# Verify calibrations applied
python bin/verify_calibrations.py --s1 --s4 --s5
```

---

## Optuna Database Locations

Production-ready optimized parameters:

| Archetype | Database | Best Trial | PF | Location |
|-----------|----------|------------|-----|----------|
| S1 (Liquidity Vacuum) | `optuna_production_v2_trap_within_trend.db` | #47 | 2.1 | `/Users/raymondghandchi/Bull-machine-/Bull-machine-/` |
| S4 (Funding Divergence) | `optuna_production_v2_order_block_retest.db` | #52 | 2.8 | `/Users/raymondghandchi/Bull-machine-/Bull-machine-/` |
| S5 (Long Squeeze) | `optuna_quick_test_v3_bos_choch.db` | #31 | 1.9 | `/Users/raymondghandchi/Bull-machine-/Bull-machine-/` |

---

## Extracting Optuna Parameters

### Method 1: Automatic Extraction (Recommended)

```bash
# Extract best trial for S4
python bin/extract_thresholds.py \
  --db optuna_production_v2_order_block_retest.db \
  --archetype s4 \
  --output configs/s4_optimized.json

# Output:
# Analyzing optuna_production_v2_order_block_retest.db...
# Found 87 trials
# Best trial: #52 (PF: 2.83, Sharpe: 2.1)
# Extracting parameters...
# Saved to: configs/s4_optimized.json
```

### Method 2: Manual Extraction

```bash
# List all trials
python bin/list_optuna_trials.py \
  --db optuna_production_v2_order_block_retest.db \
  --sort-by pf

# Output:
# Trial #52: PF 2.83, Sharpe 2.10, Win Rate 68%
#   funding_threshold: 0.72
#   oi_threshold: 0.45
#   confluence_min: 3
#   entry_delay_bars: 2
#
# Trial #48: PF 2.71, Sharpe 2.05, Win Rate 65%
#   funding_threshold: 0.68
#   ...

# Copy parameters from best trial
vim configs/s4_optimized.json
```

### Method 3: Programmatic Access

```python
import optuna

# Load study
storage = "sqlite:///optuna_production_v2_order_block_retest.db"
study = optuna.load_study(study_name="s4_funding_divergence", storage=storage)

# Get best trial
best_trial = study.best_trial
print(f"Best trial: #{best_trial.number}")
print(f"PF: {best_trial.value}")
print(f"Parameters: {best_trial.params}")

# Extract parameters
params = best_trial.params
print(f"funding_threshold: {params['funding_threshold']}")
print(f"oi_threshold: {params['oi_threshold']}")
```

---

## Applying Calibrations

### Method 1: Direct Config Update

```bash
# Apply optimized parameters to production config
python bin/apply_optimized_calibrations.py \
  --archetype s4 \
  --config configs/mvp/mvp_bear_market_v1.json \
  --source configs/s4_optimized.json

# Backup created: configs/mvp/mvp_bear_market_v1.json.backup
# Updated: configs/mvp/mvp_bear_market_v1.json
```

### Method 2: Manual Config Edit

```bash
# Edit production config
vim configs/mvp/mvp_bear_market_v1.json

# Find S4 archetype section and update:
{
  "archetypes": {
    "s4_funding_divergence": {
      "funding_threshold": 0.72,        // was 0.5
      "oi_threshold": 0.45,             // was 0.3
      "confluence_min": 3,              // was 2
      "entry_delay_bars": 2,            // was 0
      "regime_filter": ["risk_off", "neutral"]
    }
  }
}
```

### Method 3: Runtime Override

```python
# In code, override thresholds at runtime
from engine.archetypes.threshold_policy import ThresholdPolicy

policy = ThresholdPolicy(
    archetype="s4_funding_divergence",
    config_overrides={
        "funding_threshold": 0.72,
        "oi_threshold": 0.45,
        "confluence_min": 3
    }
)
```

---

## Parameter Mappings

### S1 (Liquidity Vacuum)

| Parameter | Vanilla Default | Optimized | Range | Impact |
|-----------|-----------------|-----------|-------|--------|
| `exhaustion_threshold` | 0.6 | 0.78 | 0.5-0.9 | Higher = fewer trades, higher precision |
| `volume_climax_min` | 2.0 | 2.8 | 1.5-4.0 | Volume spike required |
| `wick_ratio_min` | 0.3 | 0.42 | 0.2-0.6 | Wick exhaustion threshold |
| `spring_lookback` | 20 | 14 | 10-30 | Wyckoff spring detection window |
| `regime_filter` | ['crisis'] | ['risk_off', 'crisis'] | N/A | Allowed market regimes |

**Best Trial:** #47 from `optuna_production_v2_trap_within_trend.db`

**Performance:** Train PF 2.3, Test PF 2.1, OOS PF 1.9

### S4 (Funding Divergence)

| Parameter | Vanilla Default | Optimized | Range | Impact |
|-----------|-----------------|-----------|-------|--------|
| `funding_threshold` | 0.5 | 0.72 | 0.3-1.0 | Funding Z-score required |
| `oi_threshold` | 0.3 | 0.45 | 0.2-0.8 | OI change threshold |
| `confluence_min` | 2 | 3 | 2-4 | Min signals required |
| `entry_delay_bars` | 0 | 2 | 0-5 | Wait N bars after signal |
| `regime_filter` | ['risk_off'] | ['risk_off', 'neutral'] | N/A | Allowed market regimes |

**Best Trial:** #52 from `optuna_production_v2_order_block_retest.db`

**Performance:** Train PF 3.1, Test PF 2.8, OOS PF 2.5

### S5 (Long Squeeze)

| Parameter | Vanilla Default | Optimized | Range | Impact |
|-----------|-----------------|-----------|-------|--------|
| `funding_extreme` | 0.8 | 0.65 | 0.5-1.2 | Funding Z-score threshold |
| `oi_spike_threshold` | 0.05 | 0.08 | 0.03-0.15 | OI spike detection |
| `cascade_window` | 8 | 12 | 4-20 | Cascade detection lookback |
| `entry_confirmation_bars` | 1 | 3 | 1-5 | Confirmation period |
| `regime_filter` | ['risk_on'] | ['risk_on', 'neutral'] | N/A | Allowed market regimes |

**Best Trial:** #31 from `optuna_quick_test_v3_bos_choch.db`

**Performance:** Train PF 2.0, Test PF 1.9, OOS PF 1.8

---

## Verification

After applying calibrations, verify they loaded correctly:

```bash
# Verify all calibrations
python bin/verify_calibrations.py --all

# Expected output:
# S1 Calibration Check:
# - Config: configs/mvp/mvp_bear_market_v1.json
# - exhaustion_threshold: 0.78 ✓ (optimized)
# - volume_climax_min: 2.8 ✓ (optimized)
# - wick_ratio_min: 0.42 ✓ (optimized)
# - Source: optuna_production_v2_trap_within_trend.db trial #47
# - Status: ✓ OPTIMIZED
#
# S4 Calibration Check:
# - Config: configs/mvp/mvp_bear_market_v1.json
# - funding_threshold: 0.72 ✓ (optimized)
# - oi_threshold: 0.45 ✓ (optimized)
# - confluence_min: 3 ✓ (optimized)
# - Source: optuna_production_v2_order_block_retest.db trial #52
# - Status: ✓ OPTIMIZED
```

---

## Re-Optimization

When to re-optimize:

1. **Performance degrades** below minimums (S4 < 2.2 PF, S1 < 1.8 PF)
2. **Market regime changes** (sustained bull → bear transition)
3. **New features added** to archetype logic
4. **Quarterly maintenance** (every 3 months)

### Re-Optimization Process

```bash
# Step 1: Run new Optuna optimization (100-200 trials)
python bin/optimize_s4_calibration.py \
  --trials 200 \
  --timeout 7200 \
  --db optuna_s4_recalibration_$(date +%Y%m%d).db

# Step 2: Compare new vs old performance
python bin/compare_optuna_studies.py \
  --old optuna_production_v2_order_block_retest.db \
  --new optuna_s4_recalibration_20251208.db

# Output:
# Old best: Trial #52, PF 2.83
# New best: Trial #87, PF 2.91 (+2.8% improvement)
# Recommendation: Use new calibration

# Step 3: Extract new parameters
python bin/extract_thresholds.py \
  --db optuna_s4_recalibration_20251208.db \
  --output configs/s4_optimized_v2.json

# Step 4: Test on OOS period
python bin/run_archetype_suite.py \
  --archetypes s4 \
  --config configs/s4_optimized_v2.json \
  --periods oos

# Step 5: If OOS performance good, apply to production
python bin/apply_optimized_calibrations.py \
  --archetype s4 \
  --source configs/s4_optimized_v2.json

# Step 6: Update Optuna DB reference
mv optuna_production_v2_order_block_retest.db optuna_production_v2_order_block_retest_old.db
mv optuna_s4_recalibration_20251208.db optuna_production_v2_order_block_retest.db
```

---

## Calibration Best Practices

### 1. Always Use Train/Test/OOS Split

```python
# Optimize on train period only
train_period = ("2020-01-01", "2022-12-31")
test_period = ("2023-01-01", "2023-06-30")
oos_period = ("2023-07-01", "2024-12-31")

# Optimization should only see train data
# Test for validation
# OOS for final verification before production
```

### 2. Don't Over-Optimize

```bash
# Bad: 1000 trials optimizing 20 parameters
# Result: Overfitting to train data

# Good: 100-200 trials optimizing 5-8 key parameters
# Result: Robust generalization
```

### 3. Multi-Objective Optimization

```python
# Optimize for multiple objectives
def objective(trial):
    # Suggest parameters
    funding_threshold = trial.suggest_float("funding_threshold", 0.3, 1.0)

    # Run backtest
    results = backtest(...)

    # Return multiple objectives
    return results["profit_factor"], results["sharpe"], -results["max_drawdown"]

# Pareto frontier analysis
study = optuna.create_study(directions=["maximize", "maximize", "maximize"])
```

### 4. Regime-Aware Calibration

```bash
# Optimize separately for bull/bear regimes
python bin/optimize_s4_calibration.py \
  --regime bull \
  --period 2020-01-01,2021-12-31

python bin/optimize_s4_calibration.py \
  --regime bear \
  --period 2022-01-01,2023-12-31

# Use regime-specific parameters in production
```

### 5. Version Control Calibrations

```bash
# Tag each calibration version
git add configs/s4_optimized.json
git commit -m "S4 calibration v2.1 (trial #52, PF 2.83)"
git tag s4_calibration_v2.1

# Easy rollback if needed
git checkout s4_calibration_v2.0
```

---

## Troubleshooting

### Issue: Optuna DB Not Found

```bash
# List available DBs
ls -lh *.db

# If missing, re-run optimization
python bin/optimize_s4_calibration.py --trials 100
```

### Issue: Parameters Not Applied

```bash
# Check if config was actually updated
git diff configs/mvp/mvp_bear_market_v1.json

# If no changes, re-run apply script with --force
python bin/apply_optimized_calibrations.py --s4 --force
```

### Issue: Performance Degrades After Applying

```bash
# Compare vanilla vs optimized
python bin/compare_calibrations.py \
  --vanilla configs/mvp/mvp_bear_market_v1.json.backup \
  --optimized configs/mvp/mvp_bear_market_v1.json

# If optimized worse, rollback
cp configs/mvp/mvp_bear_market_v1.json.backup \
   configs/mvp/mvp_bear_market_v1.json
```

---

## Advanced: Custom Optimization

### Define Custom Objective

```python
# bin/optimize_custom.py
import optuna

def objective(trial):
    # Suggest parameters
    funding_threshold = trial.suggest_float("funding_threshold", 0.3, 1.0)
    oi_threshold = trial.suggest_float("oi_threshold", 0.2, 0.8)

    # Custom constraints
    if funding_threshold < oi_threshold:
        raise optuna.TrialPruned()  # Invalid combination

    # Run backtest with these parameters
    config = create_config(funding_threshold, oi_threshold)
    results = run_backtest(config)

    # Custom scoring (weighted combination)
    score = (
        results["profit_factor"] * 0.4 +
        results["sharpe"] * 0.3 +
        (1.0 / results["max_drawdown"]) * 0.2 +
        results["win_rate"] * 0.1
    )

    return score

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
```

---

## Maintenance Schedule

**Weekly:** Check calibration status
```bash
python bin/verify_calibrations.py --all
```

**Monthly:** Compare production performance vs backtest
```bash
python bin/compare_live_vs_backtest.py --s1 --s4 --s5
```

**Quarterly:** Re-optimize if needed
```bash
# If live performance < 80% of backtest, re-optimize
python bin/optimize_s4_calibration.py --trials 200
```

**After Major Code Changes:** Re-validate calibrations
```bash
./bin/validate_archetype_engine.sh --full
```

---

**Document Version:** 1.0
**Last Updated:** 2025-12-08
**Maintained By:** Optimization Team
**Next Review:** After next Optuna optimization run
