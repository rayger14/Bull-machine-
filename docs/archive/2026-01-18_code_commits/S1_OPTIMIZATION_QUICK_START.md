# S1 Liquidity Vacuum Re-Optimization - Quick Start Guide

**Author:** Claude Code (Performance Engineer)
**Date:** 2026-01-16
**Status:** 🔄 Optimization in progress (32% complete)

---

## Overview

This guide covers the complete S1 re-optimization workflow using institutional-grade walk-forward validation with multi-objective optimization.

**Problem:** S1 current PF is 1.0 (needs >1.4), optimized only on 2022 data
**Solution:** Re-optimize on full 7-year dataset (2018-2024) with strict validation

---

## Files Created

### 1. Optimization Scripts

```bash
bin/optimize_s1_walkforward.py          # Main optimization script
bin/validate_s1_optimized_results.py    # Validation script
```

### 2. Output Files

```bash
configs/s1_optimized_2018_2024.json     # New optimized config
S1_REOPTIMIZATION_REPORT.md             # Detailed optimization report
S1_VALIDATION_REPORT.md                 # Constraint validation report
s1_optimization_log.txt                 # Full optimization log
```

### 3. Documentation

```bash
WALK_FORWARD_OPTIMIZATION_BEST_PRACTICES.md   # Detailed methodology
S1_OPTIMIZATION_QUICK_START.md                # This file
```

---

## Quick Commands

### Run Optimization (50 trials, ~30-40 minutes)

```bash
python3 bin/optimize_s1_walkforward.py \
  --data data/features_2018_2024_UPDATED.parquet \
  --regimes risk_off crisis \
  --n-trials 50 \
  --output-dir configs
```

### Run Optimization (100 trials for production, ~60-80 minutes)

```bash
python3 bin/optimize_s1_walkforward.py \
  --data data/features_2018_2024_UPDATED.parquet \
  --regimes risk_off crisis \
  --n-trials 100 \
  --output-dir configs
```

### Validate Results

```bash
python3 bin/validate_s1_optimized_results.py \
  --config configs/s1_optimized_2018_2024.json \
  --old-config configs/s1_multi_objective_production.json \
  --output S1_VALIDATION_REPORT.md
```

### Monitor Progress

```bash
# Watch optimization progress
tail -f s1_optimization_log.txt

# Check current trial count
grep "Trial" s1_optimization_log.txt | tail -5

# View latest results
tail -50 s1_optimization_log.txt
```

---

## Current Status

**Optimization Progress:**
- Status: 🔄 Running
- Progress: 16/50 trials (32%)
- Estimated time remaining: ~25 minutes
- Average trial time: ~40 seconds (improving)

**Dataset:**
- File: `data/features_2018_2024_UPDATED.parquet`
- Size: 61,277 hourly bars (7 years)
- Date range: 2018-01-01 to 2024-12-31
- Train: 35,033 bars (2018-2021)
- Test: 26,236 bars (2022-2024)

**Regime Distribution:**
- Risk-on: 32,044 bars (52%)
- Crisis: 18,996 bars (31%)
- Neutral: 7,299 bars (12%)
- Risk-off: 2,938 bars (5%)

---

## Optimization Methodology

### Walk-Forward Validation

```
Training Window: 2018-2021 (4 years)
├── Purpose: Learn patterns across multiple market cycles
├── Regimes: All (risk_on, crisis, neutral, risk_off)
└── Trials: 50 multi-objective optimizations

Test Window: 2022-2024 (3 years)
├── Purpose: Validate on completely unseen data
├── Constraint: OOS degradation < 20%
└── Metric targets: PF > 1.4, Sharpe > 0.5, MaxDD < 25%
```

### Multi-Objective Optimization

```python
objectives = [
    'maximize sharpe_ratio',      # Risk-adjusted returns
    'maximize profit_factor',     # Win/loss ratio (primary)
    'minimize max_drawdown'       # Risk management
]

# NSGA-II finds Pareto-optimal solutions
# Best solution selected by composite score:
score = (PF × 2.0) + Sharpe - (MaxDD / 25.0)
```

### Parameters Being Optimized

| Parameter | Current | Range | Description |
|-----------|---------|-------|-------------|
| fusion_threshold | 0.400 | 0.30-0.60 | Minimum signal confidence |
| liquidity_max | 0.192 | 0.10-0.30 | Max liquidity score (drain threshold) |
| volume_z_min | 1.695 | 1.0-3.0 | Minimum volume spike (panic threshold) |
| wick_lower_min | 0.351 | 0.20-0.50 | Minimum lower wick ratio (rejection) |

---

## Validation Constraints

### Hard Constraints (Must Pass)

| Constraint | Target | Importance |
|------------|--------|------------|
| OOS Degradation | < 20% | HIGH |
| Test Profit Factor | > 1.4 | HIGH |
| Test Sharpe Ratio | > 0.5 | HIGH |
| Test Max Drawdown | < 25% | HIGH |
| Min Trades (Test) | >= 100 | MEDIUM |

### Soft Constraints (Monitored)

| Constraint | Target | Importance |
|------------|--------|------------|
| Regime Profitability | PF > 1.2 in 3/4 regimes | MEDIUM |
| Regime Dominance | No regime > 60% trades | MEDIUM |
| Yearly Distribution | Trades in all years | MEDIUM |
| Parameter Stability | No params at boundaries | LOW |

---

## Expected Results

### Conservative Scenario
- Sharpe: 0.3-0.5
- PF: 1.2-1.4
- Max DD: 20-25%
- Trades: 8-12/year
- OOS Degradation: 15-20%

### Target Scenario ✅
- Sharpe: 0.5-0.8
- PF: 1.4-1.6
- Max DD: 15-20%
- Trades: 12-15/year
- OOS Degradation: 10-15%

### Optimistic Scenario
- Sharpe: 0.8-1.2
- PF: 1.6-2.0
- Max DD: 10-15%
- Trades: 15-20/year
- OOS Degradation: < 10%

---

## Post-Optimization Workflow

### 1. Wait for Completion

```bash
# Monitor until complete
tail -f s1_optimization_log.txt

# Look for:
# - "OPTIMIZATION COMPLETE"
# - Config saved to: configs/s1_optimized_2018_2024.json
# - Report saved to: S1_REOPTIMIZATION_REPORT.md
```

### 2. Review Reports

```bash
# Main optimization report
cat S1_REOPTIMIZATION_REPORT.md

# Check:
# - OOS degradation percentage
# - Test PF and Sharpe
# - Regime breakdown
# - Parameter changes vs old config
```

### 3. Validate Constraints

```bash
python3 bin/validate_s1_optimized_results.py

# Expected output:
# ✅ All constraints passed
# OR
# ⚠️ Some constraints need review
# OR
# ❌ Failed - re-optimization needed
```

### 4. Visual Inspection

```bash
# Review optimization report sections:
# 1. Executive Summary
# 2. Training vs Test Metrics
# 3. Optimized Parameters (old vs new)
# 4. Performance by Regime
# 5. Constraint Validation
# 6. Next Steps
```

### 5. Decision Tree

```
All constraints passed?
├─ YES → Deploy to paper trading
│   ├─ Monitor 2-4 weeks
│   ├─ Validate live vs backtest
│   └─ Promote to production if validated
│
└─ NO → Review failures
    ├─ OOS degradation > 20%? → Increase trials to 100
    ├─ PF < 1.4? → Adjust search space, add features
    ├─ Sharpe < 0.5? → Review risk management
    └─ Regime issues? → Adjust regime weights
```

---

## Troubleshooting

### Issue: Optimization Taking Too Long

**Symptoms:** > 2 hours for 50 trials

**Solutions:**
```bash
# 1. Use fewer trials for initial test
python3 bin/optimize_s1_walkforward.py --n-trials 30

# 2. Check system resources
top -l 1 | grep -E "CPU|PhysMem"

# 3. Restart if hung
pkill -f optimize_s1_walkforward
python3 bin/optimize_s1_walkforward.py --n-trials 50
```

### Issue: Constraints Not Met

**Symptoms:** OOS degradation > 20% or PF < 1.4

**Solutions:**
```bash
# 1. Increase trials (more exploration)
python3 bin/optimize_s1_walkforward.py --n-trials 100

# 2. Adjust search space (tighter bounds)
# Edit bin/optimize_s1_walkforward.py:
# S1_SEARCH_SPACE = {
#   'fusion_threshold': (0.35, 0.55),  # Narrower
#   ...
# }

# 3. Reduce parameters (simpler model)
# Fix liquidity_max = 0.20
# Optimize only 3 parameters
```

### Issue: Config File Missing

**Symptoms:** validate script can't find config

**Solution:**
```bash
# Check optimization completed
tail -50 s1_optimization_log.txt | grep "COMPLETE"

# Check config exists
ls -lh configs/s1_optimized_2018_2024.json

# If missing, re-run optimization
python3 bin/optimize_s1_walkforward.py --n-trials 50
```

---

## Advanced Usage

### Custom Regime Filters

```bash
# Test with different regime combinations
python3 bin/optimize_s1_walkforward.py \
  --regimes crisis risk_off neutral \
  --n-trials 50

# Crisis only (conservative)
python3 bin/optimize_s1_walkforward.py \
  --regimes crisis \
  --n-trials 50
```

### Custom Data Range

```bash
# Optimize on different period
# Edit script to change window_configs:
window_configs = [
    {
        'train_start': '2019-01-01',
        'train_end': '2022-12-31',
        'test_start': '2023-01-01',
        'test_end': '2024-12-31',
    }
]
```

### Parallel Optimization

```bash
# Run multiple optimizations with different settings
python3 bin/optimize_s1_walkforward.py --n-trials 50 --regimes crisis risk_off &
python3 bin/optimize_s1_walkforward.py --n-trials 50 --regimes crisis &

# Compare results
diff S1_REOPTIMIZATION_REPORT.md S1_REOPTIMIZATION_REPORT_crisis_only.md
```

---

## Best Practices

### ✅ Do's

1. **Wait for completion** - Don't interrupt optimization mid-run
2. **Review all reports** - Check optimization + validation reports
3. **Validate constraints** - All high-priority must pass
4. **Paper trade first** - 2-4 weeks before production
5. **Monitor continuously** - Track live vs backtest metrics
6. **Re-optimize quarterly** - Market dynamics change

### ❌ Don'ts

1. **Don't skip validation** - Always run validation script
2. **Don't cherry-pick metrics** - Must pass all constraints
3. **Don't over-optimize** - 50-100 trials sufficient
4. **Don't ignore regime breakdown** - Must work across regimes
5. **Don't deploy blindly** - Paper trade first
6. **Don't forget to monitor** - Set up alerting

---

## Performance Metrics Glossary

**Sharpe Ratio:**
- Risk-adjusted return metric
- Higher is better
- > 0.5 = acceptable, > 1.0 = good, > 2.0 = excellent

**Profit Factor (PF):**
- Gross profit / gross loss
- > 1.0 = profitable, > 1.5 = good, > 2.0 = excellent
- Primary optimization target for S1

**Max Drawdown (DD):**
- Largest peak-to-trough decline
- Lower is better
- < 25% = acceptable, < 15% = good, < 10% = excellent

**OOS Degradation:**
- (Train metric - Test metric) / Train metric
- Measures overfitting
- < 10% = excellent, < 20% = acceptable, > 30% = overfit

**Win Rate:**
- % of trades that are profitable
- > 50% = profitable, > 60% = good, > 70% = excellent
- Less important than PF for rare-event strategies

---

## Next Steps After Optimization

### 1. Immediate (Today)

- [ ] Wait for optimization completion
- [ ] Review S1_REOPTIMIZATION_REPORT.md
- [ ] Run validation script
- [ ] Check all constraints passed

### 2. Short-term (This Week)

- [ ] Deploy to paper trading environment
- [ ] Set up monitoring dashboard
- [ ] Document baseline metrics
- [ ] Create alerting rules

### 3. Medium-term (2-4 Weeks)

- [ ] Monitor paper trading performance
- [ ] Compare live vs backtest metrics
- [ ] Track slippage and execution quality
- [ ] Validate regime behavior

### 4. Long-term (Monthly/Quarterly)

- [ ] Review monthly performance
- [ ] Re-optimize if metrics degrade
- [ ] Update documentation
- [ ] Refine monitoring thresholds

---

## Support

**Issues?** Check:
1. This guide (troubleshooting section)
2. WALK_FORWARD_OPTIMIZATION_BEST_PRACTICES.md (detailed methodology)
3. S1_REOPTIMIZATION_REPORT.md (optimization results)
4. S1_VALIDATION_REPORT.md (constraint validation)

**Questions about:**
- Methodology → See WALK_FORWARD_OPTIMIZATION_BEST_PRACTICES.md
- Results → See S1_REOPTIMIZATION_REPORT.md
- Validation → See S1_VALIDATION_REPORT.md
- Configuration → See configs/s1_optimized_2018_2024.json

---

## Summary

✅ **Optimization Script:** Created and running
✅ **Validation Script:** Created and ready
✅ **Dataset:** 7 years, 61K bars, all features present
✅ **Methodology:** Walk-forward, multi-objective, strict constraints
✅ **Documentation:** Comprehensive guides and best practices

⏳ **Current Status:** 32% complete, ~25 minutes remaining

🎯 **Expected Outcome:** PF 1.4+, Sharpe 0.5+, OOS degradation < 20%

---

**Last Updated:** 2026-01-16 11:30 AM
**Optimization ETA:** ~11:45 AM (15 minutes)
