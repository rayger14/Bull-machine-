# Comprehensive Validation Framework Guide

**Version:** 1.0
**Created:** 2025-11-20
**Status:** PRODUCTION READY

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Framework Architecture](#framework-architecture)
3. [Validation Components](#validation-components)
4. [Quick Start Guide](#quick-start-guide)
5. [Usage Examples](#usage-examples)
6. [Interpreting Results](#interpreting-results)
7. [Production Deployment Criteria](#production-deployment-criteria)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)

---

## Executive Summary

### Purpose

The Comprehensive Validation Framework ensures that optimized trading configs:
1. **Don't overfit** - Maintain performance on unseen data
2. **Work across regimes** - Perform appropriately in different market conditions
3. **Have statistical edge** - Performance is not due to random chance

### Three-Pillar Validation Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                  VALIDATION FRAMEWORK                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Pillar 1: WALK-FORWARD        Pillar 2: CROSS-REGIME      │
│  ─────────────────────        ─────────────────────        │
│  • Train (2022 H1)             • Risk-on performance        │
│  • Validate (2022 H2)          • Risk-off performance       │
│  • Test (2023 H1)              • Regime routing checks      │
│  • Degradation analysis        • Forbidden trade detection  │
│                                                              │
│  Pillar 3: STATISTICAL                                      │
│  ──────────────────────                                     │
│  • Bootstrap resampling (1000x)                             │
│  • Permutation tests                                        │
│  • Confidence intervals                                     │
│  • P-value < 0.05 requirement                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Acceptance Criteria (All Must Pass)

A config is **production ready** if and only if:

1. **Walk-Forward:**
   - ✅ PF_validate ≥ 0.7 × PF_train (max 30% degradation)
   - ✅ PF_test ≥ 1.0 (edge maintained OOS)
   - ✅ DD_test ≤ 25% (risk control)

2. **Cross-Regime:**
   - ✅ Bull archetypes excel in risk_on (PF > 1.5)
   - ✅ Bear archetypes excel in risk_off (PF > 1.3)
   - ✅ < 30% trades in "forbidden" regime

3. **Statistical:**
   - ✅ Bootstrap PF 95% CI lower bound > 1.2
   - ✅ Permutation test p-value < 0.05
   - ✅ Sample size ≥ 20 trades

---

## Framework Architecture

### Directory Structure

```
bin/
├── validate_walkforward.py              # Pillar 1: Walk-forward validator
├── validate_cross_regime.py             # Pillar 2: Cross-regime validator
├── validate_statistical_significance.py # Pillar 3: Statistical validator
└── run_full_validation.sh               # Automated pipeline

results/validation/
└── {timestamp}/
    └── {config_name}/
        ├── walkforward/
        │   ├── train_metrics.json
        │   ├── validate_metrics.json
        │   ├── test_metrics.json
        │   ├── regime_breakdown.csv
        │   └── validation_summary.json
        ├── cross_regime/
        │   ├── regime_breakdown.json
        │   ├── regime_performance.csv
        │   ├── regime_distribution.png
        │   ├── regime_pf_chart.png
        │   └── anomaly_report.txt
        ├── statistical/
        │   ├── statistical_summary.json
        │   ├── bootstrap_distribution.png
        │   ├── permutation_test.png
        │   ├── confidence_intervals.csv
        │   └── regime_comparison.csv
        └── VALIDATION_STATUS               # PASSED or FAILED
```

### Data Flow

```
┌──────────────┐
│ Config JSON  │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ Feature Store        │
│ (2022-2023 data)     │
└──────┬───────────────┘
       │
       ├────────────────┐
       │                │
       ▼                ▼
┌──────────────┐  ┌──────────────┐
│ Backtest     │  │ Backtest     │  ...
│ Engine       │  │ Engine       │
└──────┬───────┘  └──────┬───────┘
       │                │
       ▼                ▼
┌──────────────┐  ┌──────────────┐
│ Trades DF    │  │ Trades DF    │
└──────┬───────┘  └──────┬───────┘
       │                │
       ├────────────────┼────────────────┐
       │                │                │
       ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Walk-Forward │  │ Cross-Regime │  │ Statistical  │
│ Validator    │  │ Validator    │  │ Validator    │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                │                │
       └────────────────┴────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Validation      │
              │ Summary Report  │
              └─────────────────┘
```

---

## Validation Components

### 1. Walk-Forward Validation (`validate_walkforward.py`)

**Purpose:** Detect temporal overfitting by testing on unseen future data.

**Periods:**
```
Train:    2022-01-01 to 2022-06-30  (6 months) - Parameter optimization
Validate: 2022-07-01 to 2022-12-31  (6 months) - Hyperparameter selection
Test:     2023-01-01 to 2023-06-30  (6 months) - Out-of-sample verification
```

**Metrics Computed:**
- Profit Factor (PF)
- Win Rate (WR)
- Sharpe Ratio
- Max Drawdown (DD)
- Trade Count
- R-Multiples
- Consecutive Losses
- Regime Breakdown (PF/WR/DD per regime)

**Key Checks:**
1. **Validation Degradation:** `(PF_train - PF_val) / PF_train < 30%`
2. **Test Collapse:** `PF_test ≥ 1.0`
3. **DD Explosion:** `DD_test ≤ 25%`
4. **Sample Size:** `trades ≥ 5 per period`
5. **Regime Consistency:** At least one regime with PF > 1.3

**Example:**
```bash
python bin/validate_walkforward.py \
    --config configs/mvp/mvp_bear_market_v1.json \
    --output results/validation/mvp_bear/
```

**Output Interpretation:**
```json
{
  "train": {"pf": 1.58, "wr": 0.47, "trades": 15},
  "validate": {"pf": 1.42, "wr": 0.45, "trades": 12},
  "test": {"pf": 1.31, "wr": 0.43, "trades": 10},
  "val_pf_degradation": 0.10,    // 10% - GOOD
  "test_pf_degradation": 0.17,   // 17% - ACCEPTABLE
  "production_ready": true        // ✅ PASS
}
```

---

### 2. Cross-Regime Validation (`validate_cross_regime.py`)

**Purpose:** Ensure configs perform appropriately across different market regimes.

**Regime Requirements:**

| Archetype Type | risk_on       | neutral      | risk_off     | crisis       |
|----------------|---------------|--------------|--------------|--------------|
| **Bull**       | PF > 1.5 ✅   | PF > 1.2 ✅  | Disabled ❌  | Muted ❌     |
| **Bear**       | Disabled ❌   | PF > 1.0 ⚠️  | PF > 1.3 ✅  | PF > 1.2 ✅  |

**Key Checks:**
1. **Bull Regime Performance:** Bull archetypes excel in risk_on
2. **Bear Regime Performance:** Bear archetypes excel in risk_off/crisis
3. **Forbidden Trades:** < 30% trades in wrong regime
4. **Regime Routing:** Config respects regime-specific weights

**Anomaly Detection:**
- S5 (bear) trading 40% in risk_on → Regime misclassification
- Bull trap active in crisis → Routing broken

**Example:**
```bash
python bin/validate_cross_regime.py \
    --config configs/optimized/s5_balanced.json \
    --output results/validation/s5_regime/
```

**Output Interpretation:**
```
REGIME BREAKDOWN:
Regime       Trades  Trade%  PF     WR     Max DD
risk_on      2       10.0%   0.85   0.50   5.2%    ⚠️ Low trades (expected)
neutral      5       25.0%   1.12   0.60   7.3%    ✅ Moderate
risk_off     10      50.0%   1.78   0.70   8.1%    ✅ Strong
crisis       3       15.0%   1.65   0.67   12.4%   ✅ Strong

VALIDATION: PASS ✅
- Bear archetype performing as expected
- Most trades in risk_off/crisis (65%)
- PF > 1.3 in intended regimes
```

---

### 3. Statistical Significance Validation (`validate_statistical_significance.py`)

**Purpose:** Prove that observed edge is statistically significant, not random chance.

**Bootstrap Resampling:**
```python
# Pseudo-code
for i in range(1000):
    sample_trades = resample(trades, with_replacement=True)
    pf_distribution.append(compute_pf(sample_trades))

ci_lower = percentile(pf_distribution, 2.5)   # 95% CI
ci_upper = percentile(pf_distribution, 97.5)

# Test: Is CI lower bound > 1.2?
significant = ci_lower > 1.2
```

**Permutation Test:**
```python
# Null hypothesis: PF = 1.0 (no edge)
observed_pf = compute_pf(trades)

for i in range(1000):
    shuffled_trades = shuffle(trades['pnl'])
    perm_pf.append(compute_pf(shuffled_trades))

# p-value: How often random shuffling beats observed?
p_value = (perm_pf >= observed_pf).mean()

# Significant if p < 0.05 (< 5% chance of being random)
significant = p_value < 0.05
```

**Example:**
```bash
python bin/validate_statistical_significance.py \
    --config configs/optimized/s2_aggressive.json \
    --n-bootstrap 1000 \
    --n-permutations 1000 \
    --output results/validation/s2_stats/
```

**Output Interpretation:**
```
BOOTSTRAP CONFIDENCE INTERVALS (95%):
  Profit Factor:  1.58 [1.32, 1.89] ✅  (CI lower > 1.2)
  Win Rate:       0.47 [0.38, 0.56] ✅
  Sharpe Ratio:   1.23 [0.87, 1.61] ✅

PERMUTATION TESTS:
  Edge Significance: p=0.0180 ✅ SIGNIFICANT
    (Only 1.8% chance this edge is random)

  Regime Comparisons:
    risk_off vs risk_on: p=0.0023 ✅ (regimes differ significantly)

STATISTICALLY SIGNIFICANT: YES ✅
```

---

## Quick Start Guide

### Minimal Example (Single Config)

```bash
# 1. Validate a single config
python bin/validate_walkforward.py \
    --config configs/mvp/mvp_bear_market_v1.json \
    --output results/validation/test/

# 2. Check results
cat results/validation/test/mvp_bear_market_v1/validation_summary.json

# 3. If production_ready: true, config is good to deploy
```

### Full Pipeline (Automated)

```bash
# Run all 3 validation pillars automatically
./bin/run_full_validation.sh configs/mvp/mvp_bear_market_v1.json

# Or validate entire directory
./bin/run_full_validation.sh configs/mvp/

# Check consolidated report
cat results/validation/*/VALIDATION_SUMMARY_REPORT.md
```

---

## Usage Examples

### Example 1: Validate Optimized S5 Config

```bash
# Context: S5 (Long Squeeze) was optimized via Optuna
# Need to validate it maintains edge OOS

# Run full validation pipeline
./bin/run_full_validation.sh \
    configs/optimized/s5_balanced.json \
    results/validation/s5_production/

# Expected output structure:
# results/validation/s5_production/
#   s5_balanced/
#     walkforward/        → PF test: 1.45 ✅
#     cross_regime/       → PF risk_off: 1.78 ✅
#     statistical/        → p-value: 0.0089 ✅
#     VALIDATION_STATUS   → PASSED ✅

# Deploy to production if PASSED
```

### Example 2: Batch Validate MVP Configs

```bash
# Validate all MVP configs at once
./bin/run_full_validation.sh configs/mvp/

# Review consolidated report
cat results/validation/*/VALIDATION_SUMMARY_REPORT.md

# Example output:
# | Config                  | Walk-Forward | Cross-Regime | Statistical | Overall |
# |-------------------------|--------------|--------------|-------------|---------|
# | mvp_bear_market_v1      | ✅           | ✅           | ✅          | ✅      |
# | mvp_bull_market_v1      | ✅           | ⚠️           | ✅          | ⚠️      |
# | mvp_bull_wyckoff_v1     | ❌           | ✅           | ❌          | ❌      |

# Deploy only mvp_bear_market_v1 (full pass)
```

### Example 3: Validate with Custom Parameters

```bash
# Higher bootstrap iterations for more robust CI
python bin/validate_statistical_significance.py \
    --config configs/critical/production_candidate.json \
    --n-bootstrap 5000 \
    --n-permutations 2000 \
    --alpha 0.01 \
    --output results/validation/critical/

# More stringent alpha = 0.01 (99% confidence required)
```

### Example 4: Validate Specific Period

```bash
# Test on 2022 bear market only
python bin/validate_cross_regime.py \
    --config configs/bear_market_2022_test.json \
    --start-date 2022-01-01 \
    --end-date 2022-12-31 \
    --output results/validation/2022_only/
```

---

## Interpreting Results

### Walk-Forward Results

**✅ Production Ready:**
```json
{
  "train": {"pf": 1.85, "trades": 18, "dd": 0.12},
  "validate": {"pf": 1.62, "trades": 14, "dd": 0.15},
  "test": {"pf": 1.48, "trades": 11, "dd": 0.18},
  "val_pf_degradation": 0.12,    // 12% - Excellent
  "test_pf_degradation": 0.20,   // 20% - Good
  "overfit_detected": false,
  "collapse_detected": false,
  "production_ready": true       // ✅ DEPLOY
}
```

**❌ Overfitted:**
```json
{
  "train": {"pf": 3.25, "trades": 22, "dd": 0.08},
  "validate": {"pf": 1.42, "trades": 18, "dd": 0.22},
  "test": {"pf": 0.88, "trades": 15, "dd": 0.31},
  "val_pf_degradation": 0.56,    // 56% - TERRIBLE
  "test_pf_degradation": 0.73,   // 73% - TERRIBLE
  "overfit_detected": true,      // ❌ OVERFIT
  "collapse_detected": true,     // ❌ NO EDGE OOS
  "production_ready": false      // ❌ REJECT
}
```

**Interpretation:**
- **Train PF >> Test PF:** Config fit to in-sample noise
- **High DD in test:** Risk management failed OOS
- **Trade count dropping:** Pattern becoming less frequent (market adapting?)

---

### Cross-Regime Results

**✅ Proper Bear Archetype:**
```
REGIME BREAKDOWN:
risk_on:   2 trades (10%), PF: 0.75   ✅ Expected failure
neutral:   5 trades (25%), PF: 1.08   ✅ Marginal
risk_off:  10 trades (50%), PF: 1.82  ✅ STRONG
crisis:    3 trades (15%), PF: 1.68   ✅ STRONG

VALIDATION: PASS ✅
- Bear pattern performs in bear regimes
- Properly disabled in risk_on
- 65% trades in intended regimes (risk_off + crisis)
```

**❌ Broken Regime Routing:**
```
REGIME BREAKDOWN:
risk_on:   15 trades (60%), PF: 1.85  ❌ ANOMALY
neutral:   5 trades (20%), PF: 1.22   ✅
risk_off:  3 trades (12%), PF: 0.88   ❌ SHOULD BE STRONG
crisis:    2 trades (8%), PF: 1.01    ❌ LOW

ANOMALIES:
- risk_on: 60% of trades in forbidden regime (max 30%)
- Bear archetype not performing in risk_off (PF < 1.3)

VALIDATION: FAIL ❌
- Regime routing broken (trading in wrong regime)
- Bear pattern not working in bear market
```

---

### Statistical Results

**✅ Statistically Significant:**
```
BOOTSTRAP 95% CI:
  PF: 1.62 [1.38, 1.91]  ✅ Lower bound > 1.2

PERMUTATION TEST:
  p-value: 0.0123  ✅ < 0.05
  Interpretation: Only 1.23% chance edge is random

SIGNIFICANT: YES ✅
```

**❌ Not Statistically Significant:**
```
BOOTSTRAP 95% CI:
  PF: 1.18 [0.92, 1.48]  ❌ Lower bound < 1.2
  (CI includes values < 1.0 → no confidence in edge)

PERMUTATION TEST:
  p-value: 0.1842  ❌ > 0.05
  Interpretation: 18.42% chance this is random luck

SIGNIFICANT: NO ❌
```

**Sample Size Warning:**
```
WARNINGS:
- Small sample size (12 trades < 20) - results may be unstable
- Consider extending validation period or lowering thresholds
```

---

## Production Deployment Criteria

### Tier 1: Production Ready (Immediate Deployment)

**Requirements:**
- ✅ Walk-forward: PF_test ≥ 1.5, DD_test ≤ 20%, degradation < 25%
- ✅ Cross-regime: Correct regime performance (PF > 1.5 in intended regime)
- ✅ Statistical: p < 0.01 (99% confidence), CI lower > 1.3
- ✅ Sample size: ≥ 30 trades in test period

**Example Configs:**
- `mvp_bear_market_v1` (S5 only, PF 1.86 in risk_off)
- `optimized_bull_v2_production` (Multiple bull archetypes, PF 2.15 in risk_on)

---

### Tier 2: Shadow Mode (Monitor Before Deploying)

**Requirements:**
- ⚠️ Walk-forward: PF_test ≥ 1.2, DD_test ≤ 25%, degradation < 30%
- ✅ Cross-regime: Correct regime performance (PF > 1.2 in intended regime)
- ✅ Statistical: p < 0.05 (95% confidence), CI lower > 1.1
- ⚠️ Sample size: 20-29 trades

**Action:**
- Deploy in shadow mode (log signals, no trades)
- Monitor for 30 days
- Promote to Tier 1 if live performance matches backtest

---

### Tier 3: Rejected (Do Not Deploy)

**Reasons for Rejection:**
- ❌ Walk-forward: Overfit (degradation > 30%) or collapse (PF_test < 1.0)
- ❌ Cross-regime: Trading in forbidden regime (> 30% trades)
- ❌ Statistical: Not significant (p > 0.05) or CI includes < 1.0
- ❌ Sample size: < 10 trades (insufficient data)

**Action:**
- Re-run optimization with different parameter ranges
- Extend training period
- Consider different archetypes
- Mark as PERMANENTLY DISABLED if repeated failures (like S2)

---

## Troubleshooting

### Issue 1: "No trades generated in validation"

**Symptom:**
```
WARNING: No trades generated in validate period
ValidationResult: production_ready=False
```

**Causes:**
1. Thresholds too high (fusion_threshold = 0.55 too strict)
2. Regime override forcing wrong regime
3. Archetype disabled in config
4. Feature store missing required columns

**Solution:**
```bash
# Check config thresholds
cat configs/problem_config.json | grep threshold

# Verify regime labels exist
python -c "
import pandas as pd
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31_backup.parquet')
print(df['macro_regime'].value_counts())
"

# Lower thresholds temporarily for testing
# Edit config: fusion_threshold: 0.35 → 0.30
```

---

### Issue 2: "Statistical test fails despite good PF"

**Symptom:**
```
PF: 1.65 (looks good)
p-value: 0.1234 (NOT significant)
Bootstrap CI: [0.88, 2.42] (wide, includes < 1.0)
```

**Causes:**
1. **Small sample size** - Only 8 trades, high variance
2. **Few large wins** - 2 huge wins skewing results
3. **High variance** - Inconsistent trade outcomes

**Solution:**
```bash
# Extend validation period for more trades
python bin/validate_statistical_significance.py \
    --config configs/problem.json \
    --start-date 2022-01-01 \
    --end-date 2024-12-31 \  # Longer period
    --output results/extended/

# Or: Lower significance requirements (not recommended for production)
python bin/validate_statistical_significance.py \
    --alpha 0.10 \  # 90% confidence instead of 95%
    --output results/relaxed/
```

---

### Issue 3: "Regime routing broken (trading in wrong regime)"

**Symptom:**
```
ANOMALY: risk_on: 65% of trades in forbidden regime
Bear archetype (S5) trading heavily in risk_on
```

**Causes:**
1. Regime classifier not loaded
2. `regime_override` forcing wrong regime
3. Routing weights not applied
4. `fusion_adapt.enable = false` (routing disabled)

**Solution:**
```bash
# Check regime override
cat configs/problem.json | grep regime_override

# Verify fusion_adapt enabled
cat configs/problem.json | grep -A 5 fusion_adapt

# Fix: Enable adaptive fusion
{
  "fusion_adapt": {
    "enable": true,  // ← MUST BE TRUE
    "ema_alpha": 0.2
  },
  "regime_classifier": {
    "regime_override": {}  // ← REMOVE OVERRIDES
  }
}
```

---

## API Reference

### `validate_walkforward.py`

```bash
python bin/validate_walkforward.py \
    --config <path_to_config.json> \
    --output <output_directory> \
    --feature-store <path_to_feature_store.parquet>
```

**Arguments:**
- `--config` (required): Path to config JSON file
- `--output` (default: `results/validation/walkforward`): Output directory
- `--feature-store` (default: `data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31_backup.parquet`): Feature store path

**Output Files:**
- `train_metrics.json` - Training period (2022 H1) metrics
- `validate_metrics.json` - Validation period (2022 H2) metrics
- `test_metrics.json` - Test period (2023 H1) metrics
- `regime_breakdown.csv` - Performance by regime across all periods
- `validation_summary.json` - Overall validation status

**Exit Codes:**
- `0`: Validation passed (production ready)
- `1`: Validation failed

---

### `validate_cross_regime.py`

```bash
python bin/validate_cross_regime.py \
    --config <path_to_config.json> \
    --output <output_directory> \
    --feature-store <path_to_feature_store.parquet> \
    --start-date <YYYY-MM-DD> \
    --end-date <YYYY-MM-DD>
```

**Arguments:**
- `--config` (required): Path to config JSON file
- `--output` (default: `results/validation/cross_regime`): Output directory
- `--feature-store` (default: see above): Feature store path
- `--start-date` (default: `2022-01-01`): Validation start date
- `--end-date` (default: `2023-06-30`): Validation end date

**Output Files:**
- `regime_breakdown.json` - Detailed regime-specific metrics
- `regime_performance.csv` - Metrics by regime in tabular format
- `regime_distribution.png` - Trade distribution visualization
- `regime_pf_chart.png` - PF by regime bar chart
- `trades_by_regime.csv` - All trades grouped by regime
- `anomaly_report.txt` - Detected anomalies and warnings

---

### `validate_statistical_significance.py`

```bash
python bin/validate_statistical_significance.py \
    --config <path_to_config.json> \
    --output <output_directory> \
    --feature-store <path_to_feature_store.parquet> \
    --n-bootstrap <iterations> \
    --n-permutations <iterations> \
    --alpha <significance_level> \
    --start-date <YYYY-MM-DD> \
    --end-date <YYYY-MM-DD>
```

**Arguments:**
- `--config` (required): Path to config JSON file
- `--output` (default: `results/validation/statistical`): Output directory
- `--n-bootstrap` (default: `1000`): Bootstrap resampling iterations
- `--n-permutations` (default: `1000`): Permutation test iterations
- `--alpha` (default: `0.05`): Significance level (0.05 = 95% confidence)
- `--start-date` (default: `2022-01-01`): Validation start date
- `--end-date` (default: `2023-06-30`): Validation end date

**Output Files:**
- `statistical_summary.json` - All test results and p-values
- `bootstrap_distribution.png` - PF distribution from bootstrap
- `permutation_test.png` - Permutation test visualization
- `confidence_intervals.csv` - 95% CI for all metrics
- `regime_comparison.csv` - Regime-vs-regime statistical tests

---

### `run_full_validation.sh`

```bash
./bin/run_full_validation.sh <config_or_directory> [output_directory]
```

**Arguments:**
- `config_or_directory` (required): Path to config JSON or directory of configs
- `output_directory` (optional): Custom output directory (default: `results/validation/{timestamp}`)

**Environment Variables:**
- `FEATURE_STORE`: Path to feature store parquet
- `N_BOOTSTRAP`: Bootstrap iterations (default: 1000)
- `N_PERMUTATIONS`: Permutation iterations (default: 1000)
- `ALPHA`: Significance level (default: 0.05)

**Output Files:**
- `VALIDATION_SUMMARY_REPORT.md` - Consolidated report for all configs
- `{config_name}/walkforward/` - Walk-forward results
- `{config_name}/cross_regime/` - Cross-regime results
- `{config_name}/statistical/` - Statistical results
- `{config_name}/VALIDATION_STATUS` - PASSED or FAILED

---

## Best Practices

### 1. Always Validate Before Production

```bash
# ❌ BAD: Deploy optimized config directly
cp results/optimization/best_trial.json configs/production.json

# ✅ GOOD: Validate first
./bin/run_full_validation.sh results/optimization/best_trial.json
# Only deploy if validation passes
```

---

### 2. Use Conservative Thresholds for Critical Configs

```bash
# For production-critical configs, require 99% confidence
python bin/validate_statistical_significance.py \
    --config configs/production_candidate.json \
    --alpha 0.01 \
    --n-bootstrap 5000
```

---

### 3. Monitor Validation Metrics Over Time

```bash
# Re-validate production configs monthly
for config in configs/production/*.json; do
    ./bin/run_full_validation.sh "$config" \
        "results/validation/monthly_$(date +%Y%m)/"
done

# Alert if PF degrades > 15%
```

---

### 4. Document Validation Results in Git

```bash
# Commit validation results with config
git add configs/optimized/s5_balanced.json
git add results/validation/s5_balanced/
git commit -m "Add S5 balanced config (validation: PF 1.86, p=0.0089)"
```

---

## Conclusion

The Comprehensive Validation Framework provides rigorous, multi-dimensional validation to ensure configs are production-ready. By combining walk-forward testing, regime analysis, and statistical significance tests, we can confidently deploy configs that maintain edge in live trading.

**Remember:**
- **All three pillars must pass** for production deployment
- **Small sample sizes** (< 20 trades) require extra caution
- **Re-validate regularly** as market conditions change
- **When in doubt, use shadow mode** before live deployment

---

**Version History:**
- v1.0 (2025-11-20): Initial framework documentation

**Related Documents:**
- `OPTIMIZATION_REQUIREMENTS_SPEC.md` - Optimization requirements
- `WALK_FORWARD_VALIDATION_GUIDE.md` - Detailed walk-forward methodology
- `REGIME_GROUND_TRUTH_USAGE.md` - Regime classification guide
