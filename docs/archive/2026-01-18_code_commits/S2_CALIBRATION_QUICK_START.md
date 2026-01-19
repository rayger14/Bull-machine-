# S2 Calibration Quick Start Guide

**Goal:** Fix S2 overtrading (418 trades → 10 trades/year) using data-driven thresholds

## Quick Test (30 minutes)

### Step 1: Distribution Analysis (1 minute)
```bash
python3 bin/analyze_s2_distribution.py
```

**Output:**
- Shows current baseline is 90th percentile (too low)
- Recommends search ranges for top 1-3% of signals
- Saves percentiles to `results/s2_calibration/fusion_percentiles_2022.json`

**Expected:** Current 0.55 threshold → should target ~0.75-0.80 for 10 trades/year

### Step 2: Quick Optimization (20 minutes)
```bash
# Test with 10 trials (faster)
python3 bin/optimize_s2_calibration.py --trials 10
```

**What it does:**
- Tests 10 parameter combinations
- Each trial runs 3 backtests (2022 H1, H2, 2023 H1)
- Finds Pareto frontier (non-dominated solutions)

**Expected:** 3-5 Pareto solutions with PF > 1.5, trades 8-12/year

### Step 3: Generate Configs (5 seconds)
```bash
python3 bin/generate_s2_configs.py
```

**Output:** 3 production configs
- `configs/optimized/s2_conservative.json` - Highest PF
- `configs/optimized/s2_balanced.json` - Best tradeoff (recommended)
- `configs/optimized/s2_aggressive.json` - Most trades

### Step 4: Validate (2 minutes)
```bash
# Test balanced config
python3 bin/backtest_knowledge_v2.py \
    --asset BTC \
    --start 2022-01-01 \
    --end 2022-12-31 \
    --config configs/optimized/s2_balanced.json
```

**Expected:**
- Total Trades: 8-12 (vs. baseline 418)
- Win Rate: > 55%
- Profit Factor: > 1.5
- Max Drawdown: < 10%

## Production Run (2-3 hours)

### Full Optimization
```bash
# 50 trials for robust Pareto frontier
python3 bin/optimize_s2_calibration.py --trials 50 --timeout 7200
```

**Time:** 2-3 hours (50 trials × 3 folds × 2 min/backtest)

### Resume Capability
```bash
# If interrupted, resume from where you left off
python3 bin/optimize_s2_calibration.py --trials 50 --resume
```

## Key Insights from Distribution Analysis

### Current Problem
- **Threshold:** 0.55 (90th percentile)
- **Meaning:** Trading top 10% of bars
- **Result:** 418 trades/year (way too many)

### Target Solution
- **Threshold:** ~0.75-0.80 (99th-99.5th percentile)
- **Meaning:** Trading top 0.5-1% of bars only
- **Result:** 8-12 trades/year (high-conviction only)

### Why 44% Higher Threshold?
- Current: Catches noise and weak signals
- Target: Only extreme failed rallies with strong confluence

## Understanding the Configs

### Conservative (Risk-Averse)
- Highest profit factor
- Fewer trades (8-10/year)
- Lowest drawdown
- **Use when:** Prioritize quality over quantity

### Balanced (Recommended)
- Best PF/trade tradeoff
- Moderate trades (10-12/year)
- Good Sharpe ratio
- **Use when:** Want balanced approach

### Aggressive (Active)
- Most trades (12-15/year)
- Still maintains PF > 1.3
- Higher drawdown tolerance
- **Use when:** Want more signals, accept more risk

## Files Generated

```
results/s2_calibration/
├── fusion_distribution_2022.csv       # All bar scores
├── fusion_percentiles_2022.json       # Percentile summary
├── optuna_s2_calibration.db           # Optimization database
└── pareto_frontier_top10.csv          # Best solutions

configs/optimized/
├── s2_conservative.json
├── s2_balanced.json
└── s2_aggressive.json
```

## Troubleshooting

### "No Pareto solutions found"
- Search space too restrictive
- Check `fusion_percentiles_2022.json` for actual ranges
- Widen ranges or reduce pruning

### "All trials pruned"
- Thresholds too aggressive
- Lower `fusion_threshold` minimum (e.g., 0.60 instead of 0.68)
- Check if 2022 data has enough signals

### Config has too few trades (< 5/year)
- Threshold too high
- Lower `fusion_threshold` range
- Increase `TARGET_ANNUAL_TRADES` in optimizer

### Config has too many trades (> 20/year)
- Threshold too low
- Raise `fusion_threshold` range
- Decrease `TARGET_ANNUAL_TRADES` in optimizer

## Next Steps After Calibration

1. **Validate OOS:** Test on 2023-2024 data (not used in optimization)
2. **Compare to Baseline:** Show 418 trades → 10 trades improvement
3. **Paper Trade:** Deploy balanced config to paper account
4. **Monitor:** Track realized vs. expected performance
5. **Re-calibrate:** Every 6 months with new data

## Complete Documentation

See `S2_CALIBRATION_RESULTS.md` for:
- Full methodology
- Architecture details
- Expected results
- Troubleshooting guide
- Production deployment plan

---

**Time Investment:** 30 min (quick test) or 3 hours (production)
**Expected Outcome:** 97% reduction in trades, 36% improvement in PF
**Status:** Ready to test
