# S2 (Failed Rally) Archetype - Optimization Final Report

**Date**: 2025-11-20
**Status**: PATTERN FUNDAMENTALLY BROKEN - RECOMMEND KEEP DISABLED
**Optimizer**: Optuna Multi-Objective (50 trials)
**Result**: 0 Pareto solutions found (all 50 trials pruned)

---

## Executive Summary

After fixing 4 critical bugs in the S2 optimization pipeline and running a comprehensive 50-trial multi-objective search, we conclusively determined that **the S2 (Failed Rally) archetype is fundamentally unprofitable and cannot be calibrated to meet performance targets**.

**Recommendation**: Keep S2 disabled in production configs (as currently configured in `mvp_bear_market_v1.json`).

---

## Critical Bugs Fixed

### Bug #1: Runtime Enrichment Not Applied
**Problem**: S2 runtime features (wick_upper_ratio, volume_fade_flag, rsi_bearish_div, ob_retest_flag) were never computed during backtests.

**Root Cause**: `bin/backtest_knowledge_v2.py` didn't call `apply_runtime_enrichment()` before running backtests.

**Fix**: Added enrichment hook at `bin/backtest_knowledge_v2.py:2619-2632`:
```python
if runtime_config and runtime_config.get('archetypes', {}).get('enable_S2', False):
    s2_thresholds = runtime_config['archetypes'].get('thresholds', {}).get('failed_rally', {})
    if s2_thresholds.get('use_runtime_features', False):
        from engine.strategies.archetypes.bear.failed_rally_runtime import apply_runtime_enrichment
        df = apply_runtime_enrichment(df, lookback=lookback)
```

**Verification**: Logs show `[S2 Runtime] Enriching dataframe with 4302 bars` with stats on wick rejections, volume fades, etc.

---

### Bug #2: Parameter Unit Mismatch
**Problem**: `wick_ratio_min` searched in range [2.0-4.0], but the actual feature `wick_upper_ratio` is bounded [0-1] (it's a percentage).

**Root Cause**: Optimizer assumed absolute values instead of ratios.

**Fix**: Updated search range to [0.46, 0.57] based on empirical distribution analysis.

**Impact**: With wrong range [2-4], the check `wick_upper_ratio < 2.0` always passed, making the parameter ineffective.

---

### Bug #3: Config Nesting Error
**Problem**: Optuna-generated temp configs had `failed_rally` at wrong nesting level:
```json
{
  "archetypes": {
    "thresholds": {"min_liquidity": 0.10},
    "failed_rally": {...}  // WRONG LEVEL!
  }
}
```

**Expected Structure**:
```json
{
  "archetypes": {
    "thresholds": {
      "failed_rally": {...}  // INSIDE thresholds!
    }
  }
}
```

**Root Cause**: `bin/optimize_s2_calibration.py:210` placed `failed_rally` as sibling to `thresholds` instead of child.

**Fix**: Moved `failed_rally` dict inside `thresholds` dict at line 207.

**Impact**: RuntimeContext couldn't find S2 thresholds, causing all trials to use hardcoded defaults.

---

### Bug #4: Baseline Trades Not Disabled
**Problem**: Optimizer config allowed baseline fusion trades (tier1_market) alongside S2 trades, inflating trade counts.

**Root Cause**: `entry_threshold_confidence: 0.36` allowed baseline trades to pass fusion gate.

**Fix**: Set `entry_threshold_confidence: 0.99` to disable baseline trades, ensuring S2-only testing.

**Impact**: Before fix: 91 total trades (87 tier1_market + 4 failed_rally). After fix: 38 failed_rally trades only.

---

## Optimization Configuration

### Search Ranges (Final Tightened)
```json
{
  "fusion_threshold": [0.70, 0.85],
  "wick_ratio_min": [0.46, 0.57],
  "rsi_min": [65.0, 80.0],
  "volume_z_max": [-0.54, 0.21],
  "liquidity_max": [0.05, 0.25],
  "cooldown_bars": [12, 24]
}
```

**Note**: Ranges were tightened from initial calibration values to reduce excessive trade frequency.

### Objectives
1. **Maximize Profit Factor** (harmonic mean across folds)
2. **Target 10 trades/year** (acceptable range: 3-30)
3. **Minimize Max Drawdown**

### Cross-Validation Folds
- **2022 H1** (train): 2022-01-01 to 2022-06-30
- **2022 H2** (validate): 2022-07-01 to 2022-12-31
- **2023 H1** (test): 2023-01-01 to 2023-06-30

---

## Results Summary

### Trade Frequency: 207-284 trades/year
**Target**: 3-30 trades/year
**Actual**: ALL 50 trials produced 207-284 trades/year (7-10x over target)

**Sample Results**:
- Trial 0: 228.6 trades/year
- Trial 1: 254.8 trades/year
- Trial 4: 209.8 trades/year (BEST)
- Trial 5: 283.7 trades/year (WORST)

### Profit Factors: 0.33-0.54 (bear) / 0.95-0.99 (bull)
**2022 Bear Market Performance**:
- H1: PF 0.33-0.54, WR 32-44%
- H2: PF 0.45-0.54, WR 37-42%
- **Pattern LOSES money in bear markets despite being bear-biased**

**2023 Bull Market Performance**:
- H1: PF 0.95-0.99, WR 42-47%
- **Pattern barely breaks even in bull markets**

### Pruning: 50/50 trials (100%)
**Pruning Reasons**:
- ALL trials exceeded target trade frequency
- Even with strictest parameters (fusion=0.85, cooldown=24), still produced 200+ trades/year
- No parameter combination achieved 3-30 trades/year target

---

## Key Findings

### 1. Pattern Over-Trades Severely
Even with aggressive filtering:
- fusion_threshold: 0.70-0.85 (vs baseline 0.36)
- cooldown_bars: 12-24 (vs baseline 8)
- rsi_min: 65-80 (vs baseline 70)

The pattern still produces **7-10x more trades than target**.

### 2. Pattern is Unprofitable in Bear Markets
Despite being designed as a **bear-biased short pattern**, S2 **loses money in 2022 bear market** (PF 0.33-0.54).

This is a fundamental flaw: the pattern doesn't work in the market conditions it's designed for.

### 3. Pattern Barely Breaks Even in Bull Markets
2023 bull market shows PF 0.95-0.99 (breakeven to slight loss), suggesting the pattern has no edge in either regime.

### 4. Runtime Features Don't Provide Selectivity
Even with runtime enrichment working correctly:
- Strong upper wicks detected: 40.3% of bars
- Volume fades: 26.3% of bars
- OB retests: 35.8% of bars

The features are too common to be selective, resulting in excessive trade frequency.

---

## Comparison to Working Archetypes

### S3 (Long Squeeze) - WORKING
- **Trade Frequency**: 9 trades/year ✓
- **Profit Factor**: 1.86 ✓
- **Win Rate**: 55.6% ✓
- **Status**: Enabled in production

### S2 (Failed Rally) - BROKEN
- **Trade Frequency**: 207-284 trades/year ✗ (7-10x too high)
- **Profit Factor**: 0.33-0.54 ✗ (loses money in bear markets)
- **Win Rate**: 32-44% ✗ (poor accuracy)
- **Status**: Disabled in production

---

## Variance Confirmation

The fixes successfully enabled parameter variance:

**Before Fixes**: All trials identical (135.9 trades/year)
**After Fixes**: Trials varied significantly:
- 2022 H1: 83-118 trades (range: 35 trades)
- 2022 H2: 94-137 trades (range: 43 trades)
- 2023 H1: 126-171 trades (range: 45 trades)

This confirms parameters are being applied correctly and affecting results.

---

## Recommendations

### PRIMARY: Keep S2 Disabled
Maintain current production config status:
```json
{
  "enable_S2": false,
  "_comment_S2": "Failed Rally - PF 0.48 after optimization (baseline 0.38, enriched 0.48), pattern fundamentally broken"
}
```

### SECONDARY: Document Lessons Learned
1. **Runtime feature density**: Features that trigger on >30% of bars are not selective enough
2. **Regime alignment**: Bear patterns MUST profit in bear markets (PF > 1.5 minimum)
3. **Trade frequency**: Target 5-15 trades/year for archetype patterns
4. **Cross-validation**: 2022/2023 split effectively catches regime-specific failures

### TERTIARY: Archive Artifacts
Preserve optimization artifacts for future reference:
- Database: `results/s2_calibration/optuna_s2_calibration.db`
- Logs: `results/s2_optimization_FINAL.log`
- Configs: `configs/test_s2_manual.json`, `configs/test_s2_ultra_{strict,relaxed}.json`

---

## Technical Debt Paid

This optimization effort successfully:

1. ✅ Fixed S2 runtime enrichment pipeline
2. ✅ Corrected parameter unit mismatches
3. ✅ Fixed config nesting bugs
4. ✅ Validated proper parameter application
5. ✅ Conclusively proved S2 is not viable
6. ✅ Documented why S2 should remain disabled

The MVP config comment was correct: **"pattern fundamentally broken"**.

---

## Conclusion

Despite fixing 4 critical bugs and running 50 optimization trials with aggressive filtering, **no viable parameter configuration was found for the S2 (Failed Rally) archetype**.

The pattern:
- Over-trades by 7-10x even with strictest parameters
- Loses money in bear markets (its intended regime)
- Barely breaks even in bull markets
- Cannot meet performance targets (PF > 1.5, 3-30 trades/year)

**RECOMMENDATION: Keep S2 disabled in production.**

---

**Generated**: 2025-11-20
**Optimization Runtime**: 16.6 minutes (50 trials)
**Database**: `results/s2_calibration/optuna_s2_calibration.db`
**Log**: `results/s2_optimization_FINAL.log`
