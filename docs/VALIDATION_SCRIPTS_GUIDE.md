# Validation Scripts Guide

Comprehensive guide for Phase 1 Quick Validation and Phase 4 OOS Validation scripts.

---

## Overview

Three scripts support the validation pipeline:

1. **Phase 1 Quick Validation Runner** (Bash) - Fast config testing on 2022 data
2. **Phase 4 OOS Validation** (Python) - Out-of-sample testing on 2024 data
3. **Quick Test Results Analyzer** (Python) - Visualization and recommendations

---

## Script 1: Phase 1 Quick Validation Runner

**File**: `/bin/run_phase1_quick_validation.sh`

### Purpose

Quickly test multiple configurations on 2022 bear market data to identify which configs are in the optimal trade count range (25-40 trades) before investing time in full optimization.

### Usage

```bash
./bin/run_phase1_quick_validation.sh
```

No arguments required - the script auto-discovers configs and runs all tests.

### What It Does

1. Finds all `quick_test_*.json` configs in `configs/`
2. Runs each on BTC 2022-01-01 to 2022-12-31
3. Extracts key metrics:
   - Total trades
   - Profit factor (PF)
   - Win rate (WR%)
   - Max drawdown (DD%)
   - Sharpe ratio
   - Total return
4. Prints color-coded comparison table
5. Identifies configs in target range (25-40 trades)
6. Saves summary report

### Output Files

```
results/phase1_quick_validation/
├── summary_2022_bear_market.txt               # Quick summary
├── quick_test_optimized_2022_bear_market.log  # Individual logs
├── quick_test_optimized_2022_bear_market.csv  # Individual trades
└── ... (other configs)
```

### Interpretation

**Color Codes:**
- **Green**: In target range (25-40 trades), good metrics
- **Yellow**: Outside target range, needs threshold adjustment
- **Red**: Failed, error, or very poor metrics

**Target Criteria:**
- Trade Count: 25-40 (statistical significance without overtrading)
- Profit Factor: >1.5 (ideally >3.0)
- Win Rate: >40%
- Max Drawdown: <25%

### Example Output

```
================================================================================
Results Summary
================================================================================

Config                                    Trades       PF       WR%      DD%   Sharpe  Return%
--------------------------------------------------------------------------------
quick_test_optimized                          32     3.45     52.0     18.2     1.23     45.6
quick_test_optimized_v2                       28     3.12     48.5     21.4     1.08     38.2
quick_validation_fixed                        15     2.85     55.0     12.8     0.92     22.1
mvp_bull_market_v1                            42     2.98     46.2     24.5     0.88     35.4
================================================================================

Recommendations:

  ✓ quick_test_optimized: 32 trades (PF: 3.45) - IN TARGET RANGE
  ✓ quick_test_optimized_v2: 28 trades (PF: 3.12) - IN TARGET RANGE
  ○ quick_validation_fixed: 15 trades - Too few (need 25-40)
  ○ mvp_bull_market_v1: 42 trades - Too many (need 25-40)

Found 2 config(s) in target range
Best performer: quick_test_optimized (PF: 3.45)

Recommended for Phase 2 optimization starting point:
  Config: configs/quick_test_optimized.json
  Baseline PF: 3.45
```

### Timeout Handling

Each backtest has a 10-minute timeout. If exceeded:
- Marked as "ERR" in results
- Log file saved with partial output
- Script continues to next config

---

## Script 2: Phase 4 OOS Validation

**File**: `/bin/run_phase4_oos_validation.py`

### Purpose

Validate optimized configurations on unseen 2024 data to detect overfitting and confirm production readiness.

### Usage

**Basic:**
```bash
python3 bin/run_phase4_oos_validation.py
```

**Advanced:**
```bash
python3 bin/run_phase4_oos_validation.py \
  --optuna-db sqlite:///optuna_archetypes_v2.db \
  --top-n 5 \
  --study-name archetype_trap_within_trend_v2 \
  --base-config configs/mvp_bull_market_v1.json \
  --output-dir results/phase4_oos_validation
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--optuna-db` | `sqlite:///optuna_archetypes_v2.db` | Path to Optuna database |
| `--top-n` | `5` | Number of top trials to validate |
| `--study-name` | Auto-select | Specific Optuna study to analyze |
| `--base-config` | `configs/mvp_bull_market_v1.json` | Base config template |
| `--output-dir` | `results/phase4_oos_validation` | Output directory |

### Workflow

1. **Load Pareto Frontier** from Optuna database (Phase 2 optimization results)
2. **Select Diverse Trials**:
   - Conservative: Highest PF, lower risk
   - Balanced: Middle ground
   - Aggressive: More trades, acceptable PF
3. **Test Each Trial**:
   - In-sample: 2022-2023 (verify optimization)
   - Out-of-sample: 2024 (detect overfitting)
4. **Compare Metrics**:
   - PF degradation
   - Sharpe degradation
   - Trade count consistency
5. **Flag Overfitting**: >30% degradation
6. **Recommend Best Config** for production

### Output Files

```
results/phase4_oos_validation/
├── oos_validation_report.md              # Main report
├── oos_validation_summary.csv            # Metrics table
├── trial_5/
│   ├── trial_5_config.json              # Generated config
│   ├── backtest_2022-01-01_2023-12-31.log
│   ├── trades_2022-01-01_2023-12-31.csv
│   ├── backtest_2024-01-01_2024-09-30.log
│   └── trades_2024-01-01_2024-09-30.csv
└── ... (other trial directories)
```

### Overfitting Detection

**Criteria:**
- Profit Factor degrades >30% from in-sample to OOS
- Sharpe Ratio degrades >30% from in-sample to OOS

**Formula:**
```
Degradation = (InSample - OutOfSample) / InSample
```

**Example:**
```
Trial 5 (Conservative):
  In-Sample PF:  4.20
  OOS PF:        3.85
  Degradation:   8.3%   ✓ Stable (< 30%)

Trial 12 (Aggressive):
  In-Sample PF:  6.50
  OOS PF:        2.80
  Degradation:   56.9%  ⚠️ OVERFITTING
```

### Example Report Excerpt

```markdown
## Summary

| Trial | Profile      | IS PF | OOS PF | Degradation | Overfitting? |
|-------|-------------|-------|--------|-------------|--------------|
| 5     | conservative | 4.20  | 3.85   | 8.3%       | ✓ no         |
| 12    | balanced     | 5.10  | 3.20   | 37.3%      | ⚠️ YES       |
| 18    | aggressive   | 6.50  | 2.80   | 56.9%      | ⚠️ YES       |

## Recommendations

**Recommended for Production**: Trial 5 (conservative)

- OOS Profit Factor: 3.85
- OOS Sharpe Ratio: 1.42
- PF Degradation: 8.3%
- Config: `results/phase4_oos_validation/trial_5/trial_5_config.json`
```

---

## Script 3: Quick Test Results Analyzer

**File**: `/bin/analyze_quick_test_results.py`

### Purpose

Analyze Phase 1 backtest logs, create visualizations, and generate detailed recommendations for Phase 2 optimization.

### Usage

**Basic:**
```bash
python3 bin/analyze_quick_test_results.py
```

**Advanced:**
```bash
python3 bin/analyze_quick_test_results.py \
  --input-dir results/phase1_quick_validation \
  --output-dir results/phase1_quick_validation \
  --target-trades-min 25 \
  --target-trades-max 40
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input-dir` | `results/phase1_quick_validation` | Directory with backtest logs |
| `--output-dir` | `results/phase1_quick_validation` | Output directory |
| `--target-trades-min` | `25` | Minimum trades for target range |
| `--target-trades-max` | `40` | Maximum trades for target range |

### Features

1. **Metric Extraction**
   - Parses all log files automatically
   - Extracts comprehensive metrics
   - Handles errors gracefully

2. **Statistical Analysis**
   - Identifies top performers
   - Ranks by multiple criteria
   - Calculates risk-adjusted returns

3. **Visualizations**
   - **Scatter Plot**: Trade count vs Profit Factor
   - **Heatmap**: Normalized performance metrics
   - Highlights target range
   - Annotated with config names

4. **Recommendations**
   - Best config in target range
   - Alternative configs if none in range
   - Threshold adjustment suggestions
   - Next steps for Phase 2

### Output Files

```
results/phase1_quick_validation/
├── analysis_summary.csv           # Complete metrics table
├── trade_count_vs_pf.png         # Scatter plot
├── performance_heatmap.png       # Metrics heatmap
└── recommendations.md            # Detailed analysis
```

### Example Visualization

**Scatter Plot** (`trade_count_vs_pf.png`):
- X-axis: Trade count
- Y-axis: Profit factor
- Green shaded area: Target range (25-40 trades)
- Green points: Configs in target range
- Gray points: Configs outside target range
- Reference lines: PF=1.5 (min), PF=3.0 (good)

**Heatmap** (`performance_heatmap.png`):
- Rows: Metrics (trades, PF, WR, Sharpe, risk-adjusted return)
- Columns: Configs
- Colors: Green (good) → Yellow → Red (poor)
- Normalized for cross-metric comparison

### Example Recommendations

```markdown
# Phase 1 Quick Test Analysis

**Date**: 2025-11-19 23:30:00

**Target Trade Range**: 25-40 trades

---

## Summary Statistics

- Total configs tested: 5
- Successful runs: 5
- In target range: 2

## Top Performers

### In Target Range (25-40 trades)

| Rank | Config | Trades | PF | WR% | Sharpe | Return% |
|------|--------|--------|----|----|--------|---------|
| 1 | quick_test_optimized | 32 | 3.45 | 52.0 | 1.23 | 45.6 |
| 2 | quick_test_optimized_v2 | 28 | 3.12 | 48.5 | 1.08 | 38.2 |

## Recommendations for Phase 2

**Primary Recommendation**: `quick_test_optimized`

- Trades: 32 (in target range)
- Profit Factor: 3.45
- Sharpe Ratio: 1.23
- Win Rate: 52.0%

**Action**: Use this config as the base for Phase 2 optimization

---

## Next Steps

1. Review top performers in detail
2. Select base config for Phase 2 Optuna optimization
3. Define parameter search space
4. Run multi-objective optimization (PF, Sharpe, Drawdown)
5. Validate on out-of-sample data (2024)
```

---

## Complete Validation Workflow

### Phase 1: Quick Validation

```bash
# Step 1: Run quick tests on all configs
./bin/run_phase1_quick_validation.sh

# Step 2: Analyze results and generate recommendations
python3 bin/analyze_quick_test_results.py

# Step 3: Review recommendations
cat results/phase1_quick_validation/recommendations.md

# Step 4: View visualizations
open results/phase1_quick_validation/trade_count_vs_pf.png
open results/phase1_quick_validation/performance_heatmap.png
```

### Phase 2: Optimization

```bash
# Use recommended config from Phase 1
python3 bin/optuna_parallel_archetypes_v2.py \
  --base-config configs/quick_test_optimized.json \
  --groups trap_within_trend order_block_retest bos_choch \
  --trials 100 \
  --output optuna_archetypes_v2.db
```

### Phase 3: Pareto Analysis

```bash
# Analyze Pareto frontier from optimization
python3 bin/analyze_pareto_frontier.py \
  --optuna-db optuna_archetypes_v2.db \
  --study-name archetype_trap_within_trend_v2
```

### Phase 4: OOS Validation

```bash
# Test top trials on unseen 2024 data
python3 bin/run_phase4_oos_validation.py \
  --optuna-db optuna_archetypes_v2.db \
  --top-n 5

# Review OOS validation report
cat results/phase4_oos_validation/oos_validation_report.md
```

### Production Deployment

```bash
# Copy recommended config to production
cp results/phase4_oos_validation/trial_5/trial_5_config.json \
   configs/production/btc_production_v1.json

# Run full historical backtest (2022-2024)
python3 bin/backtest_knowledge_v2.py \
  --asset BTC \
  --start 2022-01-01 \
  --end 2024-09-30 \
  --config configs/production/btc_production_v1.json

# Begin paper trading
# (Monitor for 2-4 weeks before live deployment)
```

---

## Troubleshooting

### Phase 1: No Configs in Target Range

**Problem**: All configs have <25 or >40 trades

**Solution**:
1. Review entry thresholds in configs
2. Adjust `fusion_threshold` or `archetype.thresholds.fusion_threshold`
3. Lower thresholds → more trades
4. Raise thresholds → fewer trades
5. Re-run validation

### Phase 1: Script Timeouts

**Problem**: Backtests exceed 10-minute timeout

**Possible Causes**:
- Database size too large
- Inefficient feature calculations
- Infinite loops in logic

**Solution**:
1. Check database size: `ls -lh data/btc_1h_mtf.csv`
2. Profile slow code: `python3 -m cProfile bin/backtest_knowledge_v2.py ...`
3. Optimize bottlenecks
4. Increase timeout in script if necessary

### Phase 4: All Trials Show Overfitting

**Problem**: >30% degradation for all trials

**Possible Causes**:
1. Over-optimization in Phase 2
2. Market regime mismatch (2022-2023 bear, 2024 bull)
3. Lookahead bias in features

**Solution**:
1. Reduce Phase 2 trial count (less optimization)
2. Use regime-aware optimization
3. Audit feature pipeline for lookahead bias
4. Consider walk-forward validation

### Missing Dependencies

**Problem**: `ModuleNotFoundError` when running scripts

**Solution**:
```bash
# Install required packages
pip3 install pandas numpy matplotlib seaborn optuna

# Verify installation
python3 -c "import pandas, matplotlib, seaborn, optuna; print('OK')"
```

---

## File Locations Reference

```
Bull-machine-/
├── bin/
│   ├── run_phase1_quick_validation.sh      # Script 1: Quick validation
│   ├── run_phase4_oos_validation.py        # Script 2: OOS validation
│   ├── analyze_quick_test_results.py       # Script 3: Results analysis
│   └── backtest_knowledge_v2.py            # Backtest engine (used by scripts)
├── configs/
│   ├── quick_test_optimized.json           # Quick test config 1
│   ├── quick_test_optimized_v2.json        # Quick test config 2
│   ├── quick_validation_fixed.json         # Quick test config 3
│   ├── mvp_bull_market_v1.json            # MVP config
│   └── production/                         # Production configs (after Phase 4)
├── results/
│   ├── phase1_quick_validation/
│   │   ├── README.md                       # Phase 1 guide
│   │   ├── summary_2022_bear_market.txt
│   │   ├── analysis_summary.csv
│   │   ├── recommendations.md
│   │   ├── trade_count_vs_pf.png
│   │   └── performance_heatmap.png
│   └── phase4_oos_validation/
│       ├── README.md                       # Phase 4 guide
│       ├── oos_validation_report.md
│       ├── oos_validation_summary.csv
│       └── trial_{N}/                      # Per-trial results
└── docs/
    └── VALIDATION_SCRIPTS_GUIDE.md         # This file
```

---

## Best Practices

1. **Always run Phase 1 first** - Don't skip quick validation
2. **Review plots visually** - Numbers don't tell the whole story
3. **Check for overfitting early** - Don't over-optimize in Phase 2
4. **Use diverse trials** - Test conservative, balanced, and aggressive profiles
5. **Document decisions** - Keep notes on why you selected certain configs
6. **Version control configs** - Commit production configs to git
7. **Monitor degradation** - Track in-sample vs OOS metrics over time

---

## Advanced Usage

### Custom Target Ranges

If your strategy naturally trades more or less:

```bash
python3 bin/analyze_quick_test_results.py \
  --target-trades-min 15 \
  --target-trades-max 30
```

### Multiple Validation Periods

Test on different bear market periods:

```bash
# 2022 bear market
./bin/run_phase1_quick_validation.sh

# Edit script to test on 2018 bear market
# Change START_DATE="2018-01-01" and END_DATE="2018-12-31"
./bin/run_phase1_quick_validation.sh
```

### Walk-Forward OOS Validation

Test multiple OOS periods for robustness:

```bash
# Q1 2024
python3 bin/run_phase4_oos_validation.py \
  --output-dir results/phase4_oos_validation/q1_2024
# (Edit script to change OOS_START/OOS_END)

# Q2 2024
python3 bin/run_phase4_oos_validation.py \
  --output-dir results/phase4_oos_validation/q2_2024

# Q3 2024
python3 bin/run_phase4_oos_validation.py \
  --output-dir results/phase4_oos_validation/q3_2024
```

Consistent performance across all quarters → High confidence

---

## Support

For issues or questions:
1. Check this guide first
2. Review README files in results directories
3. Examine log files for error details
4. Consult codebase documentation in `docs/`
