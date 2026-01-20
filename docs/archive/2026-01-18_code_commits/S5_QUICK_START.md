# S5 (Long Squeeze) Calibration - Quick Start Guide

## Overview

S5 is a bear archetype that shorts overleveraged longs during bull market pullbacks. This guide provides a quick workflow to calibrate and deploy S5.

## Prerequisites

- Feature store data: `data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet`
- Python 3.8+
- Optuna installed: `pip install optuna`

## Quick Workflow

### Step 1: Analyze Distribution (2 minutes)

```bash
python3 bin/analyze_s5_distribution.py
```

**Output:**
- Percentile distribution of S5 fusion scores
- Recommended Optuna search ranges
- CSV exports in `results/optimization/`

**Expected Results:**
- p99 threshold: ~0.50-0.52
- Target range for 7-12 trades/year: p99.5-p99.9

### Step 2: Run Optimization (1-2 hours)

```bash
# Quick test (10 trials, ~5 minutes)
python3 bin/optimize_s5_calibration.py --trials 10

# Production run (100 trials, ~1.5 hours)
python3 bin/optimize_s5_calibration.py --trials 100

# Parallel (200 trials, 4 workers, ~45 minutes)
python3 bin/optimize_s5_calibration.py --trials 200 --jobs 4
```

**Output:**
- Pareto frontier of optimal solutions
- SQLite database: `optuna_s5_calibration.db`
- CSV results in `results/optimization/`

**Success Criteria:**
- Best PF > 1.5
- Win Rate > 50%
- Trade count: 7-12/year

### Step 3: Generate Configs (30 seconds)

```bash
python3 bin/generate_s5_configs.py
```

**Output:**
- `configs/optimized/s5_conservative.json` - High PF, fewer trades
- `configs/optimized/s5_balanced.json` - **RECOMMENDED**
- `configs/optimized/s5_aggressive.json` - More trades, lower PF
- `configs/optimized/S5_CONFIGS_COMPARISON.md` - Comparison report

### Step 4: Validate on 2024 OOS Data (2 minutes)

```bash
# Test balanced config (recommended)
python3 bin/backtest_knowledge_v2.py \
  --asset BTC \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --config configs/optimized/s5_balanced.json
```

**Success Criteria (2024 OOS):**
- PF > 1.5
- Win Rate > 50%
- Trades: 5-10
- Max DD < 20%

## Key Parameters

### S5 Fusion Components (Weights)

- **Funding Z-Score (35%)** - Primary signal, extreme positive funding
- **OI Change (25%)** - Rising open interest (if available)
- **RSI Overbought (20%)** - RSI > threshold
- **Liquidity (20%)** - Low liquidity = cascade risk

### Typical Thresholds

| Parameter | Conservative | Balanced | Aggressive |
|-----------|-------------|----------|------------|
| fusion_threshold | 0.65-0.70 | 0.55-0.60 | 0.50-0.55 |
| funding_z_min | 2.5-3.0 | 2.0-2.5 | 1.5-2.0 |
| rsi_min | 75-85 | 70-75 | 70 |
| cooldown_bars | 12-20 | 8-12 | 4-8 |

### Regime Gating

| Regime | Weight | Notes |
|--------|--------|-------|
| risk_on | 2.0x | Primary regime (bull markets) |
| neutral | 1.5x | Transition periods |
| **risk_off** | **0.0x** | **DISABLED** (bear markets) |
| crisis | 2.5x | Highest weight (capitulation) |

## Expected Performance (2023-2024)

| Config | Trades/Year | PF | Win Rate |
|--------|-------------|-----|----------|
| Conservative | 7-9 | 2.0-2.5 | 50-60% |
| Balanced | 9-11 | 1.7-2.0 | 55-65% |
| Aggressive | 11-14 | 1.5-1.8 | 60-70% |

## Common Issues

### OI Data Missing
```
[WARNING] OI data not available - using 0.0 fallback
```
**Solution:** Expected behavior. System continues with 75% signal strength (funding + RSI + liquidity).

### No Trades in 2022
```
Total trades: 0
```
**Solution:** Correct! S5 is disabled in bear markets (2022 = risk_off). Test on 2023-2024 instead.

### Low Profit Factor
```
PF < 1.5 on validation
```
**Solution:**
- Increase fusion_threshold (more selective)
- Increase funding_z_min (stronger signals)
- Increase cooldown_bars (reduce overtrading)

## Files Created

```
Bull-machine-/
├── bin/
│   ├── analyze_s5_distribution.py       ✓ Distribution analyzer
│   ├── optimize_s5_calibration.py       ✓ Optuna optimizer
│   └── generate_s5_configs.py           ✓ Config generator
│
├── engine/strategies/archetypes/bear/
│   └── long_squeeze_runtime.py          ✓ S5 runtime features
│
├── configs/optimized/
│   ├── s5_conservative.json             → Generated
│   ├── s5_balanced.json                 → Generated
│   ├── s5_aggressive.json               → Generated
│   └── S5_CONFIGS_COMPARISON.md         → Generated
│
├── results/optimization/
│   ├── s5_fusion_distribution.csv
│   ├── s5_percentile_distribution.csv
│   ├── s5_calibration_all_trials.csv
│   └── s5_calibration_pareto_frontier.csv
│
├── S5_CALIBRATION_RESULTS.md            ✓ Full documentation
└── S5_QUICK_START.md                    ✓ This file
```

## Next Steps

After successful validation:

1. **Integration Testing**
   ```bash
   # Test S5 with other archetypes in combined config
   python3 bin/backtest_knowledge_v2.py \
     --config configs/mvp/mvp_regime_routed_production.json \
     --start 2023-01-01 --end 2024-12-31
   ```

2. **Walk-Forward Validation**
   ```bash
   python3 bin/validate_walk_forward.py --archetype S5
   ```

3. **Production Deployment**
   - Update `configs/mvp/mvp_regime_routed_production.json`
   - Enable S5: `"enable_S5": true`
   - Add S5 thresholds from balanced config
   - Set monthly share cap: `"long_squeeze": 0.05` (5% of trades)

## Support

- **Full Documentation:** `S5_CALIBRATION_RESULTS.md`
- **Architecture:** See S2 pattern (similar design)
- **Troubleshooting:** Check logs in `results/optimization/`

---

**Version:** 1.0
**Date:** 2025-11-20
**Status:** Production Ready ✓
