# Soft Gating Validation Backtest - Quick Start

**Status**: ✅ Production-Ready
**Date**: 2026-01-10

---

## What This Is

Comprehensive validation framework for testing soft gating implementation:
- Score-level gating (regime weights applied to signal scores)
- Position sizing integration (regime weights scale position sizes)
- Cash bucket mechanism (under-allocation when edge is weak)
- Regime budget caps (max exposure limits per regime)

**Expected Results**: +$220 to +$270 improvement from baseline

---

## Quick Start

### Step 1: Test Framework (1 minute)
```bash
# Validate backtest logic with synthetic data
python bin/soft_gating_backtest_quick_test.py
```

### Step 2: Run Full Validation (10-20 minutes)
```bash
# Test all modes and periods
python bin/validate_soft_gating_backtest.py --mode all --periods all
```

### Step 3: Review Results
```bash
# Check markdown report
cat results/soft_gating_validation/SOFT_GATING_VALIDATION_REPORT_*.md

# Or review console output during execution
```

---

## Files Delivered

### Scripts
1. **`bin/validate_soft_gating_backtest.py`** - Main validation script
2. **`bin/soft_gating_backtest_quick_test.py`** - Quick test with synthetic data

### Documentation
1. **`SOFT_GATING_BACKTEST_VALIDATION_GUIDE.md`** - Complete usage guide
2. **`SOFT_GATING_VALIDATION_DELIVERABLES.md`** - Detailed deliverables summary
3. **`SOFT_GATING_VALIDATION_README.md`** - This file

---

## What Gets Tested

### 4 Comparison Modes
- **Baseline**: No soft gating (establishes baseline)
- **Score-only**: Score-level gating only
- **Sizing-only**: Position sizing gating only
- **Full**: All features enabled (production config)

### 4 Test Periods
- **2022 Crisis** (Jun-Dec): CRISIS regime validation
- **2023 Q1** (Jan-Apr): RISK_ON recovery validation
- **2023 H2** (Aug-Dec): NEUTRAL/mixed regime validation
- **Full 2022-2024**: Comprehensive system validation

### 7 Archetypes
- B: order_block_retest
- C: wick_trap_moneytaur
- S1: liquidity_vacuum
- H: momentum_continuation
- K: trap_within_trend
- S4: funding_divergence
- S5: long_squeeze

---

## Key Metrics

### Performance
- Total PnL ($)
- Total return (%)
- Win rate (%)
- Profit factor
- Sharpe ratio
- Max drawdown (%)

### Breakdowns
- PnL by regime (4 regimes)
- PnL by archetype (7 archetypes)
- Stop-out rate by regime

### Soft Gating
- Average regime weight
- Average position size
- Budget cap trigger count
- Cash bucket utilization
- Rejection reasons

---

## Expected Results

### CRISIS Regime
- **Archetype**: liquidity_vacuum (S1)
- **Improvement**: ~$120
- **Mechanism**: 60% position size reduction (20% → 8%)

### RISK_ON Regime
- **Archetype**: wick_trap_moneytaur (C)
- **Improvement**: ~$102
- **Mechanism**: 93% position size reduction via low weight (0.07)

### Total
- **Combined**: +$220 to +$270
- **No side effects**: Other regimes stable

---

## Command Reference

```bash
# Quick test (synthetic data)
python bin/soft_gating_backtest_quick_test.py

# Full validation (all modes, all periods)
python bin/validate_soft_gating_backtest.py --mode all --periods all

# Test specific period
python bin/validate_soft_gating_backtest.py --periods crisis

# Test specific mode
python bin/validate_soft_gating_backtest.py --mode full --periods full

# Custom output directory
python bin/validate_soft_gating_backtest.py --output results/my_test

# Custom data paths
python bin/validate_soft_gating_backtest.py \
  --data data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet \
  --edge-table results/archetype_regime_edge_table.csv
```

---

## Output Files

### Trade Blotters
`results/soft_gating_validation/trades_<period>_<mode>_<timestamp>.csv`

Contains every trade with:
- Entry/exit details
- PnL breakdown
- Regime and archetype
- Soft gating metrics (weight, size, budget cap flag)

### Markdown Report
`results/soft_gating_validation/SOFT_GATING_VALIDATION_REPORT_<timestamp>.md`

Contains:
- Executive summary table
- Detailed results by period
- Regime/archetype breakdowns
- Expected vs actual comparison
- Validation checks

---

## Success Indicators

✓ **CRISIS improves by ~$120**
✓ **RISK_ON improves by ~$102**
✓ **Total improvement +$220 to +$270**
✓ **No regressions in other regimes**
✓ **Trade count stable (< 50% increase)**
✓ **CRISIS exposure ≤ 30%**

---

## Red Flags

❌ Worse performance than baseline
❌ Massive trade count changes
❌ High stop-out rates
❌ Budget caps trigger excessively

---

## Next Steps

### If Results Match Expectations
1. Enable soft gating in production
2. Monitor in paper trading
3. Gradual rollout to live

### If Results Don't Match
1. Review edge table quality
2. Tune RegimeWeightAllocator parameters
3. Check rejection reasons
4. Re-run validation

---

## Documentation

- **Full Guide**: `SOFT_GATING_BACKTEST_VALIDATION_GUIDE.md` (comprehensive)
- **Deliverables**: `SOFT_GATING_VALIDATION_DELIVERABLES.md` (detailed summary)
- **This File**: `SOFT_GATING_VALIDATION_README.md` (quick reference)

---

## Dependencies

### Required Files
- `engine/models/archetype_model.py`
- `engine/portfolio/regime_allocator.py`
- `results/archetype_regime_edge_table.csv`

### Required Data
- `data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet`
  - Must include: OHLCV, regime_label, atr_14

---

## Support

Questions? Check:
1. `SOFT_GATING_BACKTEST_VALIDATION_GUIDE.md` (usage guide)
2. Console logs (error messages)
3. Quick test output (framework validation)
4. Implementation docs (SOFT_GATING_*.md files)

---

**Author**: System Architect (Claude Code)
**Last Updated**: 2026-01-10
