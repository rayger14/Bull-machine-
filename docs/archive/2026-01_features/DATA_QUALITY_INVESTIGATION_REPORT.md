# Data Quality & Pipeline Integrity Report
## 2018-2024 Dataset Validation

**Investigation Date:** 2026-01-15
**Priority:** P0 - CRITICAL
**Status:** ✗ MAJOR ISSUES IDENTIFIED

---

## Executive Summary

### Critical Finding
**The 2018-2021 historical data has ZERO domain-specific features computed.**

- **Temporal Coverage:** ✓ Excellent - 61,277 hourly bars (2018-2024)
- **Feature Engineering:** ✗ **BROKEN** - Only 2022-2024 has features
- **Impact:** Training on 2018-2024 is training on ~43% features (2022-2024) and ~57% zeros/nulls (2018-2021)
- **Root Cause:** Backfill scripts only computed basic macro features, not domain features

### Data Completeness by Period

| Period | Rows | Wyckoff | SMC | Liquidity | Volume Features | Regime |
|--------|------|---------|-----|-----------|-----------------|--------|
| 2018 | 8,752 | **0%** | **0%** | **0%** | **0%** | **0%** |
| 2019 | 8,760 | **0%** | **0%** | **0%** | **0%** | **0%** |
| 2020 | 8,784 | **0%** | **0%** | **0%** | **0%** | **0%** |
| 2021 | 8,745 | **0%** | **0%** | **0%** | **0%** | **0%** |
| 2022 | 8,741 | ✓ 100% | ✓ 100% | ✓ 100% | ✓ 100% | ✓ 100% |
| 2023 | 8,734 | ✓ 100% | ✓ 100% | ✓ 100% | ✓ 100% | ✓ 100% |
| 2024 | 8,761 | ✓ 100% | ✓ 100% | ✓ 100% | ✓ 100% | ✓ 100% |

**Overall:** 42.8% complete (26,236 / 61,277 rows have features)

---

## 1. File Integrity Analysis

### Available Dataset Files

| File | Size | Last Modified | Status |
|------|------|---------------|--------|
| `features_2018_2024_complete.parquet` | 22.2 MB | 2026-01-14 19:39 | ✓ Current |
| `features_2018_2024_combined.parquet` | 17.0 MB | 2026-01-14 00:30 | ⚠ Older |
| `features_2018_2021_backfilled.parquet` | 5.2 MB | 2026-01-14 19:06 | ✗ **Incomplete** |
| `features_2022_2024_MTF_with_signals.parquet` | 15.0 MB | 2026-01-13 18:00 | ✓ Complete |
| `features_2022_COMPLETE.parquet` | 3.9 MB | 2025-12-11 22:49 | ✓ Reference |
| `features_2023_MTF_with_signals.parquet` | 4.9 MB | 2026-01-13 06:49 | ✓ Complete |
| `features_2024_MTF_with_signals.parquet` | 6.0 MB | 2026-01-13 17:59 | ✓ Complete |

### ✓ Good News
- All files exist and are readable
- Temporal coverage is excellent (99-100% hourly data per year)
- Only 3 small gaps (< 1 day each)
- File timestamps are recent (within 2 days)

### ✗ Bad News
- `features_2018_2021_backfilled.parquet` only has **24 basic columns**
- `features_2018_2024_complete.parquet` has **241 columns**, but 2018-2021 rows are mostly null
- No domain features were backfilled for 2018-2021

---

## 2. Feature Completeness Analysis

### Critical S1 (Liquidity Vacuum) Features

| Feature | Overall | 2018-2021 | 2022-2024 | Status |
|---------|---------|-----------|-----------|--------|
| `wyckoff_spring` | **MISSING** | - | - | ✗ Critical |
| `wyckoff_utad` | 42.8% | 0% | 100% | ✗ Critical |
| `macro_regime` | 42.8% | 0% | 100% | ✗ Critical |
| `regime_label` | 42.8% | 0% | 100% | ✗ Critical |
| `smc_score` | 42.8% | 0% | 100% | ✗ Critical |
| `tf1h_bos_bearish` | 42.8% | 0% | 100% | ✗ Critical |
| `tf4h_bos_bearish` | 42.8% | 0% | 100% | ✗ Critical |
| `volume_climax` | **MISSING** | - | - | ✗ Critical |
| `liquidation_ratio` | **MISSING** | - | - | ✗ Critical |

**Finding:** `wyckoff_spring_a` exists (alternate column name) but still 0% for 2018-2021.

### Important Supporting Features

| Feature | Overall | 2018-2021 | 2022-2024 | Status |
|---------|---------|-----------|-----------|--------|
| `liquidity_score` | 42.8% | 0% | 100% | ✗ Critical |
| `liquidation_cluster` | **MISSING** | - | - | ⚠ Medium |
| `swing_failure` | **MISSING** | - | - | ⚠ Medium |
| `fvg_15m` | **MISSING** | - | - | ⚠ Medium |
| `order_block_retest` | **MISSING** | - | - | ⚠ Medium |
| `momentum_divergence` | **MISSING** | - | - | ⚠ Medium |
| `funding_rate` | 42.8% | 0% | 100% | ✗ Critical |
| `open_interest` | **MISSING** | - | - | ⚠ Medium |
| `cvd` | **MISSING** | - | - | ⚠ Medium |

### Features That DO Exist (Available in 2022-2024)

✓ **Wyckoff features (partial):** `wyckoff_spring_a`, `wyckoff_utad`, `wyckoff_sc`, `wyckoff_st`, `wyckoff_ar`, `wyckoff_bc`, etc. (30+ wyckoff columns)
✓ **Volume features:** `volume_climax_last_3b`, `volume_panic`, `volume_spike`, `volume_z`, `volume_ratio`
✓ **SMC features:** `smc_score`, `smc_bos`, `smc_choch`, `smc_demand_zone`, `smc_supply_zone`, `smc_liquidity_sweep`
✓ **Liquidity features:** `liquidity_score`, `liquidity_drain_pct`, `liquidity_persistence`, `liquidity_velocity`, `liquidity_vacuum_score`
✓ **FVG features:** `tf1h_fvg_high`, `tf1h_fvg_low`, `tf1h_fvg_present`, `tf4h_fvg_present`
✓ **Swing features:** `is_swing_high`, `is_swing_low`

**But all of these are 0% complete for 2018-2021.**

---

## 3. Backfilled Features in 2018-2021 File

The `features_2018_2021_backfilled.parquet` file contains **only 24 columns**:

```
Basic OHLCV:
- open, high, low, close, volume

Macro Features (from backfill script):
- BTC.D, USDT.D, DXY, DXY_Z
- YC_SPREAD
- RV_7, RV_30
- funding_Z (proxy)

Basic Volume/Momentum:
- volume_24h_mean, volume_ratio_24h, volume_spike_score, volume_z_7d
- returns_24h, returns_72h, returns_168h

Crisis Features:
- aftershock_score, crash_frequency_7d, crisis_persistence, drawdown_persistence
```

### Missing from Backfill

**Domain-specific features NOT backfilled:**
- ✗ All Wyckoff features (30+ columns)
- ✗ All SMC features (smc_score, bos, choch, zones, etc.)
- ✗ All liquidity features (liquidity_score, drain_pct, velocity, etc.)
- ✗ All order block features
- ✗ All FVG features
- ✗ All swing features
- ✗ Regime labels (macro_regime, regime_label)

---

## 4. Feature Distribution Analysis

### Comparison: 2018-2021 vs 2022-2024

| Feature | 2018-2021 Mean | 2022-2024 Mean | Drift % | Status |
|---------|----------------|----------------|---------|--------|
| `wyckoff_utad` | NaN | 0.0000 | N/A | No data |
| `smc_score` | NaN | 0.4465 | N/A | No data |
| `tf1h_bos_bearish` | NaN | 0.6400 | N/A | No data |
| `tf4h_bos_bearish` | NaN | 0.0361 | N/A | No data |
| `liquidity_score` | NaN | 0.3850 | N/A | No data |
| `funding_rate` | NaN | -0.0002 | N/A | No data |

**Conclusion:** Cannot compare distributions because 2018-2021 has NO feature data.

---

## 5. Known-Good 2022 Reference Check

### 2022 Feature Validation

The 2022 COMPLETE dataset has **185 columns** with 100% completeness:

✓ **Present and Complete (100%):**
- wyckoff_utad, wyckoff_spring_a, wyckoff_sc, wyckoff_st, etc.
- macro_regime, regime_label
- smc_score, tf1h_bos_bearish
- liquidity_score
- funding_rate

✗ **Missing from ALL Datasets:**
- wyckoff_spring (old name - replaced by wyckoff_spring_a)
- volume_climax (replaced by volume_climax_last_3b)
- liquidation_ratio
- liquidation_cluster
- swing_failure
- fvg_15m (replaced by tf1h_fvg_high, tf1h_fvg_low)
- order_block_retest (may be computed in archetype logic, not stored)
- momentum_divergence
- open_interest
- cvd

**Assessment:** Some features were renamed or replaced. Need to verify archetype code for actual feature names.

---

## 6. Gap Analysis: What Archetypes Need

### S1 Liquidity Vacuum Requirements

From `engine/strategies/archetypes/bear/liquidity_vacuum.py`:

**Required Features:**
- `liquidity_drain_pct` - ✗ NOT in 2018-2021
- `liquidity_score` - ✗ NOT in 2018-2021
- `volume_z` or `volume_zscore` - ✗ NOT in 2018-2021
- `wick_lower_ratio` or wick calculation from OHLC - ⚠ Can compute
- `crisis_persistence` - ✓ In backfilled file (but may be wrong)
- `regime_label` - ✗ NOT in 2018-2021

**Verdict:** S1 cannot generate signals on 2018-2021 data without feature backfill.

### S4 Funding Divergence Requirements

From archetype configuration:

**Required Features:**
- `funding_rate` - ✗ NOT in 2018-2021 (proxy exists but untested)
- `open_interest` or `oi_delta` - ✗ MISSING entirely
- `momentum_divergence` - ✗ MISSING entirely
- `regime_label` - ✗ NOT in 2018-2021

**Verdict:** S4 cannot generate signals on 2018-2021 data.

### Other Archetypes (K, B, H, etc.)

All archetypes depend on:
- SMC features (smc_score, bos, choch)
- Wyckoff features (spring, utad, sc, st)
- Regime labels

**Verdict:** NO archetype can generate realistic signals on 2018-2021 data.

---

## 7. Pipeline Status

### Feature Engineering Scripts Available

**Basic Feature Engineering:**
- ✓ `bin/engineer_all_features.py` - Only creates 6 momentum/volume features
- ✓ `bin/backfill_historical_features.py` - Only creates basic macro/crisis features (24 columns)

**Domain Feature Engineering:**
- ⚠ `bin/backfill_domain_features.py` - EXISTS but not run on 2018-2021
- ⚠ `bin/backfill_domain_features_fast.py` - EXISTS but not run on 2018-2021
- ⚠ `bin/backfill_liquidity_score.py` - EXISTS but not run
- ⚠ `bin/backfill_wyckoff_events.py` - EXISTS but not run

**Signal Generation:**
- ✓ `bin/generate_archetype_signals_2022.py` - Works for 2022
- ✓ `bin/generate_archetype_signals_2023.py` - Works for 2023
- ✓ `bin/generate_archetype_signals_2024.py` - Works for 2024
- ✗ No script for 2018-2021 (because features don't exist)

### Which Scripts Were Run?

**Evidence from file timestamps:**

1. ✓ `engineer_all_features.py` - Ran on 2026-01-14 (created 6 features)
2. ✓ `backfill_historical_features.py` - Ran on 2026-01-14 (created 24 features)
3. ✗ `backfill_domain_features.py` - **NEVER RUN on 2018-2021**
4. ✗ `backfill_liquidity_score.py` - **NEVER RUN on 2018-2021**
5. ✗ `backfill_wyckoff_events.py` - **NEVER RUN on 2018-2021**

**Conclusion:** Domain feature backfill was never executed for 2018-2021 period.

---

## 8. Root Cause Analysis

### Why Features Are Missing

1. **Incomplete Pipeline Execution**
   - Basic backfill scripts were run (macro, crisis, momentum)
   - Domain backfill scripts were NOT run (wyckoff, smc, liquidity)
   - Files were combined before domain features were added

2. **Feature Store Design**
   - 2022-2024 data was processed with full feature pipeline
   - 2018-2021 data was processed with minimal pipeline
   - Assumption: Domain features would be backfilled later (never happened)

3. **Combining Logic**
   - `features_2018_2024_complete.parquet` concatenates:
     - 2018-2021: Only 24 basic features
     - 2022-2024: Full 241 features
   - Result: 2018-2021 rows have NaN/0 for 217 columns

### Why This Breaks Training

1. **Feature Nulls Create Bias**
   - Model learns "old data = all features are zero"
   - Temporal leak: Model can identify 2018-2021 vs 2022-2024 by feature presence
   - Overfitting: Model memorizes which period has features vs zeros

2. **Impossible Signal Generation**
   - Archetypes compute fusion scores from domain features
   - If `liquidity_score = NaN`, fusion score = 0
   - Zero signals generated for 2018-2021 → no training examples from that period

3. **Data Leakage via Temporal Pattern**
   - Training: "If features exist → use regime prediction"
   - Training: "If features missing → default to neutral"
   - Test: 2018-2021 has different feature distribution → model breaks

---

## 9. Impact Assessment

### On Overfitting Investigation

**Why models train well but test poorly on 2018-2021:**

1. **Training Distribution Mismatch**
   - Training primarily on 2022-2024 (features present)
   - Testing on 2018-2021 (features missing → zeros)
   - Model never learned to handle 2018-2021 market conditions

2. **Temporal Leakage**
   - Model learns "feature presence = post-2022"
   - Cheats by using feature nulls as temporal indicator
   - Collapses when asked to predict on different temporal pattern

3. **Zero Signal Denominator**
   - Backtests on 2018-2021 generate zero archetype signals
   - Performance metrics divide by zero or show "no trades"
   - Cannot validate if model would have worked

### On Production Deployment

**Risk Level: CRITICAL**

- Production receives live data with full 241 features
- Model was trained on mixed data (57% zeros, 43% features)
- Unknown behavior: Will it ignore old learnings? Underperform?

**Estimated Performance Impact:**
- Training R²: Claims to use 2018-2024 but actually only 2022-2024
- Effective training data: ~26k bars (not 61k bars)
- OOS validation on 2018-2021: **INVALID** (testing on nulls)

---

## 10. Recommendations

### Priority 1: Stop Using 2018-2021 Until Fixed (P0)

**Immediate Actions:**

1. **Acknowledge Data Quality Issue**
   - Document that 2018-2024 training is actually 2022-2024 training
   - Any validation on 2018-2021 is invalid (testing on zeros)
   - Performance metrics on 2018-2021 are meaningless

2. **Revert to 2022-2024 Only**
   - Use `features_2022_2024_MTF_with_signals.parquet` (14.95 MB)
   - Train v4 regime model on 2022-2024 with proper features
   - Accept reduced training data but higher quality

3. **Update All Scripts**
   - Any script loading `features_2018_2024_complete.parquet` should add warning
   - Filter to 2022-2024 only: `df = df[df.index >= '2022-01-01']`

### Priority 2: Backfill Domain Features (P0 - Long Term)

**Estimated Time: 2-4 hours of compute**

**Scripts to Run (in order):**

1. `bin/backfill_domain_features_fast.py`
   - Computes: wyckoff events, SMC features, liquidity scores
   - Input: `features_2018_2021_backfilled.parquet`
   - Output: `features_2018_2021_domain_complete.parquet`
   - Est. time: 30-60 minutes

2. `bin/backfill_liquidity_score.py` (if not in above)
   - Computes: liquidity_score, liquidity_drain_pct, liquidity_velocity
   - Est. time: 15 minutes

3. `bin/backfill_wyckoff_events.py` (if not in above)
   - Computes: wyckoff event detection (spring, utad, sc, st)
   - Est. time: 20 minutes

4. **NEW: Generate regime labels for 2018-2021**
   - Train regime model on 2022-2024
   - Predict on 2018-2021
   - Or: Use rule-based regime for 2018-2021
   - Est. time: 30 minutes

5. **Combine and Validate**
   - Merge 2018-2021 (complete) + 2022-2024 (complete)
   - Validate feature completeness > 95% across all years
   - Test archetype signal generation on 2018-2021 sample
   - Est. time: 20 minutes

**Total Time Investment: 2-3 hours**

### Priority 3: Create Validation Gate (P1)

**Prevent This from Happening Again**

Create: `bin/validate_dataset_quality.py`

```python
def validate_feature_completeness(df, min_threshold=0.90):
    """Validate all critical features are >90% complete."""
    critical_features = [
        'liquidity_score', 'smc_score', 'wyckoff_utad',
        'regime_label', 'funding_rate', 'volume_z'
    ]

    for feature in critical_features:
        completeness = df[feature].notna().mean()
        assert completeness >= min_threshold, \
            f"{feature} only {completeness*100:.1f}% complete"
```

Add to all training scripts:
- `bin/train_logistic_regime_v4.py`
- `bin/train_continuous_risk_score_v2.py`
- Any backtest script

### Priority 4: Documentation (P1)

**Update README/docs:**

1. Document current dataset status:
   - 2022-2024: Complete, ready for training
   - 2018-2021: Incomplete, DO NOT USE until backfilled

2. Create dataset preparation guide:
   - Which scripts to run
   - In what order
   - How to validate quality

3. Add data quality checks to CI/CD:
   - Before training: Validate feature completeness
   - Before backtest: Validate signal generation
   - Before deployment: Validate feature parity with production

---

## 11. Quick Fix Option: Train on 2022-2024 Only

**If backfill is too expensive, use this:**

### Modify Training Scripts

```python
# Add to bin/train_logistic_regime_v4.py

def load_data():
    df = pd.read_parquet('data/features_2018_2024_complete.parquet')

    # CRITICAL: Filter to 2022-2024 only (features exist)
    df = df[df.index >= '2022-01-01']

    print(f"⚠️  WARNING: Using 2022-2024 only ({len(df)} bars)")
    print(f"    2018-2021 excluded due to missing features")

    return df
```

### Benefits
- ✓ Immediate fix (5 minutes)
- ✓ Honest about training data
- ✓ No risk of training on zeros
- ✓ Model learns from complete features

### Drawbacks
- ✗ Less training data (26k vs 61k bars)
- ✗ Less diversity (no 2018-2019 alt season, no 2020 COVID crash)
- ✗ May not generalize to different market regimes

---

## 12. Validation Checklist

### Before Training Any Model

- [ ] Load dataset
- [ ] Check feature completeness by year
- [ ] Verify critical features > 90% complete
- [ ] Filter to years with complete features OR backfill missing
- [ ] Generate test signals (archetype.detect()) on sample
- [ ] Verify signals > 0 for all years
- [ ] Document training period and feature completeness in model metadata

### Before Backtesting

- [ ] Verify features exist for backtest period
- [ ] Generate archetype signals for full period
- [ ] Verify signal counts match expectations (not zero)
- [ ] Check regime labels present and reasonable
- [ ] Validate no nulls in critical features

### Before Deployment

- [ ] Compare production feature schema to training feature schema
- [ ] Verify no feature drift (same calculation methods)
- [ ] Test model on recent live data
- [ ] Validate predictions are non-zero
- [ ] Check regime transitions are reasonable (not constant)

---

## 13. Summary

### Data Quality: ✗ CRITICAL ISSUES

**Files:** ✓ All files exist, reasonable sizes
**Coverage:** ✓ 99-100% hourly bars per year
**Features:** ✗ **57% of data has zero domain features**
**Usability:** ✗ **Cannot train on 2018-2021 without backfill**

### Issues Found: 2 Critical, 1 Warning

1. **[P0] 2018-2021 missing domain features**
   - Impact: Training on this data is invalid
   - Action: Backfill domain features OR filter to 2022-2024 only
   - Est. time: 2-3 hours (backfill) OR 5 minutes (filter)

2. **[P0] Feature completeness check missing from pipeline**
   - Impact: Could happen again with future data
   - Action: Add validation gate to training scripts
   - Est. time: 30 minutes

3. **[P1] 1 file older than 7 days (`features_2022_COMPLETE.parquet`)**
   - Impact: May not have latest feature engineering logic
   - Action: Consider regenerating for consistency
   - Est. time: 10 minutes

### Recommended Path Forward

**Option A: Quick Fix (5 minutes)**
- Filter all training to 2022-2024 only
- Document reduced training period
- Accept less diversity for higher quality

**Option B: Proper Fix (2-3 hours)**
- Run domain feature backfill scripts on 2018-2021
- Validate completeness > 95%
- Regenerate full 2018-2024 combined dataset
- Resume training on full period

**Recommendation: Start with Option A today, schedule Option B for tomorrow.**

---

## Appendices

### A. Full Feature List (241 columns)

See: `bin/validate_data_quality_2018_2024.py` output for complete column list.

### B. Backfill Scripts Located

```
bin/backfill_domain_features.py
bin/backfill_domain_features_fast.py
bin/backfill_liquidity_score.py
bin/backfill_liquidity_score_optimized.py
bin/backfill_wyckoff_events.py
bin/backfill_historical_features.py (already run)
bin/engineer_all_features.py (already run)
```

### C. Dataset Combination Logic

Current file appears to be created by:
1. Load `features_2018_2021_backfilled.parquet` (24 columns)
2. Load `features_2022_2024_MTF_with_signals.parquet` (241 columns)
3. Concatenate rows
4. Result: 2018-2021 rows have NaN for 217 columns

Should be:
1. Backfill domain features for 2018-2021 (output: 241 columns)
2. Load 2022-2024 (241 columns)
3. Concatenate rows
4. Result: All rows have 241 columns with >95% completeness

---

**Report Generated:** 2026-01-15
**Tool:** `bin/validate_data_quality_2018_2024.py`
**Dataset:** `data/features_2018_2024_complete.parquet`
**Status:** Ready for remediation
