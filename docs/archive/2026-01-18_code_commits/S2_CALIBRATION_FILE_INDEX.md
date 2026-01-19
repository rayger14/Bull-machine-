# S2 Calibration System - Complete File Index

## Execution Scripts (in order)

### 1. Phase 1: Distribution Analysis
```bash
bin/analyze_s2_distribution.py
```
- **Size:** 443 lines
- **Runtime:** 30-60 seconds
- **Input:** `data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet`
- **Output:**
  - `results/s2_calibration/fusion_distribution_2022.csv`
  - `results/s2_calibration/fusion_percentiles_2022.json`
- **Purpose:** Compute S2 fusion scores for ALL 2022 bars, identify percentile thresholds

### 2. Phase 2: Multi-Objective Optimization
```bash
bin/optimize_s2_calibration.py --trials 50 --timeout 7200
```
- **Size:** 682 lines
- **Runtime:** 2-3 hours (50 trials × 3 folds)
- **Input:** `results/s2_calibration/fusion_percentiles_2022.json` (search ranges)
- **Output:**
  - `results/s2_calibration/optuna_s2_calibration.db`
  - `results/s2_calibration/pareto_frontier_top10.csv`
- **Purpose:** Find Pareto-optimal S2 thresholds via multi-objective Optuna

### 3. Phase 3: Config Generation
```bash
bin/generate_s2_configs.py
```
- **Size:** 417 lines
- **Runtime:** < 5 seconds
- **Input:** `results/s2_calibration/optuna_s2_calibration.db`
- **Output:**
  - `configs/optimized/s2_conservative.json`
  - `configs/optimized/s2_balanced.json`
  - `configs/optimized/s2_aggressive.json`
- **Purpose:** Generate production configs from Pareto frontier

## Documentation Files

### Quick Start Guide
```
S2_CALIBRATION_QUICK_START.md (~200 lines)
```
- 30-minute quick test instructions
- Expected outputs at each step
- Troubleshooting tips

### Complete Methodology
```
S2_CALIBRATION_RESULTS.md (~500 lines)
```
- Full problem statement
- Phase-by-phase methodology
- Architecture diagrams
- Expected results and comparisons
- Troubleshooting guide
- Production deployment plan

### Implementation Summary
```
S2_CALIBRATION_IMPLEMENTATION_SUMMARY.md (~350 lines)
```
- High-level overview
- Deliverables summary
- Code quality notes
- Success criteria
- Next steps

### This File
```
S2_CALIBRATION_FILE_INDEX.md
```
- Quick reference for all files
- File sizes and purposes
- Execution order

## Output Files (after running)

### Distribution Analysis Outputs
```
results/s2_calibration/fusion_distribution_2022.csv
```
- All 2022 bars with computed S2 fusion scores
- Columns: timestamp, ob_retest, wick_rejection, rsi_signal, volume_fade, tf4h_confirm, fusion_score

```
results/s2_calibration/fusion_percentiles_2022.json
```
- Percentile summary (p50-p99.9)
- Statistics (mean, std, min, max)
- Recommended search ranges for Optuna

### Optimization Outputs
```
results/s2_calibration/optuna_s2_calibration.db
```
- SQLite database with all trials
- Pareto frontier solutions
- Per-fold metrics for each trial
- Resume capability

```
results/s2_calibration/pareto_frontier_top10.csv
```
- Top 10 non-dominated solutions
- Columns: trial_number, harmonic_pf, annual_trades, max_drawdown, win_rate, sharpe_ratio, parameters...

### Production Configs
```
configs/optimized/s2_conservative.json
```
- Highest profit factor config
- >= 8 trades/year
- Lowest drawdown
- For risk-averse trading

```
configs/optimized/s2_balanced.json
```
- Best PF/trade tradeoff
- ~10 trades/year
- Recommended default

```
configs/optimized/s2_aggressive.json
```
- Most trades config
- PF >= 1.3 maintained
- For active trading

## Reference Files (existing)

### Baseline Config
```
configs/s2_baseline.json
```
- Current S2 config (fusion_threshold = 0.55)
- Produces 418 trades/year (too many)

### Runtime Features
```
engine/strategies/archetypes/bear/failed_rally_runtime.py
```
- S2 feature enrichment (wick ratios, volume fade, etc.)
- Used by distribution analyzer

### Backtest Engine
```
bin/backtest_knowledge_v2.py
```
- Used by optimizer for all backtests
- Validates final configs

## Quick Commands

### Full Pipeline (first time)
```bash
# 1. Distribution analysis (1 min)
python3 bin/analyze_s2_distribution.py

# 2. Quick test (10 trials, 30 min)
python3 bin/optimize_s2_calibration.py --trials 10

# 3. Generate configs (5 sec)
python3 bin/generate_s2_configs.py

# 4. Validate balanced (2 min)
python3 bin/backtest_knowledge_v2.py \
    --asset BTC \
    --start 2022-01-01 \
    --end 2022-12-31 \
    --config configs/optimized/s2_balanced.json
```

### Production Run (after quick test)
```bash
# Full 50-trial optimization (2-3 hours)
python3 bin/optimize_s2_calibration.py --trials 50 --timeout 7200

# Generate configs
python3 bin/generate_s2_configs.py

# Validate on 2023 (out-of-sample)
python3 bin/backtest_knowledge_v2.py \
    --asset BTC \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --config configs/optimized/s2_balanced.json
```

### Resume Interrupted Optimization
```bash
python3 bin/optimize_s2_calibration.py --trials 50 --resume
```

## File Size Summary

| File | Lines | Type |
|------|-------|------|
| `bin/analyze_s2_distribution.py` | 443 | Script |
| `bin/optimize_s2_calibration.py` | 682 | Script |
| `bin/generate_s2_configs.py` | 417 | Script |
| `S2_CALIBRATION_RESULTS.md` | ~500 | Doc |
| `S2_CALIBRATION_QUICK_START.md` | ~200 | Doc |
| `S2_CALIBRATION_IMPLEMENTATION_SUMMARY.md` | ~350 | Doc |
| **Total** | **~2,600** | **All** |

## Dependencies

### Python Libraries (already in project)
- pandas
- numpy
- optuna
- json
- logging
- subprocess
- tempfile

### Data Requirements
- `data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet` (exists)
- Columns needed: OHLCV, rsi_14, volume_zscore, wick features (or computed at runtime)

### System Integration
- Uses `bin/backtest_knowledge_v2.py` for all backtests
- Leverages `engine/strategies/archetypes/bear/failed_rally_runtime.py` for features
- Integrates with `engine/context/regime_classifier.py` for regime gating

## Expected Timeline

| Phase | Quick Test | Production |
|-------|-----------|-----------|
| Distribution Analysis | 1 min | 1 min |
| Optimization | 30 min (10 trials) | 2-3 hours (50 trials) |
| Config Generation | 5 sec | 5 sec |
| Validation | 2 min | 5 min (multiple periods) |
| **Total** | **33 min** | **2-3 hours** |

## Success Indicators

After running the pipeline, you should see:

1. **Distribution Analysis:**
   - Current baseline (0.55) at ~90th percentile
   - Target threshold (0.75-0.80) at ~99.5th percentile
   - 418 bars above 0.55 → ~10 bars above 0.79

2. **Optimization:**
   - 5-10 Pareto solutions found
   - Top solution: PF > 1.5, trades 8-12/year, DD < 10%
   - Harmonic PF > 1.3 across all top solutions

3. **Config Generation:**
   - 3 distinct configs created
   - Conservative < Balanced < Aggressive (trade count)
   - Conservative > Balanced > Aggressive (profit factor)

4. **Validation:**
   - Baseline: 418 trades, PF ~1.1, WR ~48%
   - Balanced: ~10 trades, PF > 1.5, WR > 55%
   - 97% reduction in trades, 36% improvement in PF

## Troubleshooting Index

### Issue: Distribution analysis fails
**Check:** `data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet` exists
**Fix:** Verify data path, ensure 2022 data present

### Issue: No Pareto solutions found
**Check:** `results/s2_calibration/fusion_percentiles_2022.json` for search ranges
**Fix:** Widen search ranges or reduce pruning thresholds

### Issue: All trials pruned
**Check:** Search space too restrictive
**Fix:** Lower fusion_threshold minimum (e.g., 0.60 instead of 0.68)

### Issue: Configs have wrong trade count
**Adjust:** `TARGET_ANNUAL_TRADES` in `optimize_s2_calibration.py`
**Re-run:** Optimization with new target

## Contact / Support

For issues or questions:
1. Check `S2_CALIBRATION_RESULTS.md` troubleshooting section
2. Review console logs for specific error messages
3. Verify data files exist and have required columns
4. Check that `bin/backtest_knowledge_v2.py` works standalone

---

**Status:** ✅ Complete
**Author:** Claude Code (Backend Architect)
**Date:** 2025-11-20
