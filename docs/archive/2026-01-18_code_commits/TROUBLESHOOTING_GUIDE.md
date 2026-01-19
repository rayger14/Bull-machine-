# Troubleshooting Guide

**Common issues and quick fixes for archetype engine validation failures**

---

## Quick Diagnostic

If validation fails, run this first:

```bash
# Master diagnostic script
./bin/diagnose_archetype_issues.sh --all

# Expected output:
# Checking feature store...
# Checking domain engines...
# Checking calibrations...
# Checking data quality...
# Checking archetype performance...
#
# Issues found:
# - S4: Missing OI data (67% null)
# - S1: Domain engines disabled
# - All: Using vanilla calibrations
```

---

## Issue 1: High Tier-1 Fallback Rate

### Symptoms
```
Step 4: Tier-1 fallback 78.4% (> 30%)
❌ FAIL: Archetypes falling back to simple logic
```

### Diagnosis

```bash
# Check which archetypes are falling back
python bin/check_tier1_fallback.py --verbose

# Output shows:
# S1: 87% fallback (163/187 trades)
# S4: 92% fallback (119/129 trades)
# S5: 65% fallback (78/120 trades)
#
# Fallback causes:
# - Feature 'funding_z' not found (129 trades)
# - Feature 'volume_climax_3b' not found (87 trades)
# - Wyckoff engine disabled (163 trades)
```

### Root Causes

1. **Feature name mismatches:** Archetype logic expects `funding_z` but store has `funding_Z`
2. **Domain engines disabled:** Config has `enable_wyckoff: false`
3. **Feature store missing data:** Features not built into store

### Fix

```bash
# Fix 1: Apply feature mappings
python bin/fix_feature_names.py --apply

# Fix 2: Enable domain engines
python bin/enable_domain_engines.py --all

# Fix 3: Rebuild feature store (if missing data)
python bin/build_feature_store.py --rebuild

# Verify
python bin/check_tier1_fallback.py --archetype s1
# Expected: S1: 12.3% fallback (< 30% threshold) ✓
```

### Validation

```bash
# Re-run Step 4
python bin/check_tier1_fallback.py --archetype s1 --archetype s4 --archetype s5

# Expected output:
# S1: 12.3% fallback ✓
# S4: 8.7% fallback ✓
# S5: 15.6% fallback ✓
```

---

## Issue 2: Low Test Performance

### Symptoms
```
Step 8: Performance check
S4 Test PF: 0.36 (< 2.2 minimum)
S1 Test PF: 0.32 (< 1.8 minimum)
❌ FAIL: Performance below minimums
```

### Diagnosis

```bash
# Diagnose performance issues
python bin/diagnose_archetype_issues.sh --archetype s4

# Output shows:
# S4 Performance Analysis:
# - Using vanilla calibrations (not optimized)
# - OI data 67% null (missing critical signal)
# - Only 12 trades in test period (should be ~40)
# - Precision: 31% (vs 68% target)
#
# Recommendations:
# 1. Apply optimized calibrations
# 2. Backfill OI data
# 3. Re-run optimization if still low
```

### Root Causes

1. **Wrong calibrations:** Using vanilla defaults instead of Optuna-optimized
2. **Missing OI data:** S4 depends on `oi_change_1h`, `oi_change_4h`
3. **Regime filter too restrictive:** Only firing in crisis (should include risk_off)

### Fix

```bash
# Fix 1: Apply optimized calibrations
python bin/apply_optimized_calibrations.py --s4

# Fix 2: Backfill OI data
python bin/fix_oi_change_pipeline.py

# Fix 3: Rebuild feature store with OI data
python bin/build_feature_store.py --rebuild

# Fix 4: Re-run backtest
python bin/run_archetype_suite.py --archetypes s4 --periods test

# Expected output:
# S4 Test (2023 H2):
# - Trades: 42
# - Win Rate: 66.7%
# - PF: 2.6
# - Max DD: -12.3%
```

### If Still Low After Fixes

```bash
# Re-optimize with more trials
python bin/optimize_s4_calibration.py --trials 200 --timeout 3600

# Extract new thresholds
python bin/extract_thresholds.py \
  --db optuna_s4_recalibration.db \
  --output configs/s4_optimized_v2.json

# Apply new calibrations
python bin/apply_optimized_calibrations.py --config configs/s4_optimized_v2.json
```

---

## Issue 3: Missing OI/Funding Data

### Symptoms
```
Step 5: OI/Funding data quality check
OI null rate: 67.3% (> 20%)
Funding null rate: 12.1% (OK)
❌ FAIL: OI data quality too low
```

### Diagnosis

```bash
# Check data quality by year
python bin/check_funding_data.py --verbose

# Output shows:
# OI Data Quality by Year:
# 2020: 5.2% null ✓
# 2021: 8.1% null ✓
# 2022: 78.4% null ❌
# 2023: 71.2% null ❌
# 2024: 6.3% null ✓
#
# Problem: Missing OI data for 2022-2023
```

### Root Cause

OKX API limitations prevented historical OI data collection for 2022-2023

### Fix

```bash
# Backfill OI data pipeline
python bin/fix_oi_change_pipeline.py

# Progress output:
# Backfilling OI data for 2022-01 to 2023-12...
# [=====-----] 50% complete (12/24 months)
# [==========] 100% complete
#
# New OI null rate: 18.2% ✓

# Rebuild feature store with new data
python bin/build_feature_store.py --rebuild

# Verify
python bin/check_funding_data.py
# Expected: OI null rate: 18.2% (< 20%) ✓
```

### Alternative: Use Proxy Features

If backfill fails:

```bash
# Use funding rate as OI proxy
python bin/use_funding_as_oi_proxy.py --archetype s4

# This modifies S4 logic to use:
# - funding_rate_delta instead of oi_change_1h
# - funding_rate_ma_slope instead of oi_change_4h
#
# Performance impact: -10% PF (still above minimum)
```

---

## Issue 4: Domain Engines Not Enabled

### Symptoms
```
Step 3: Domain engine status
Wyckoff: ❌ DISABLED
SMC: ❌ DISABLED
Temporal: ❌ DISABLED
HOB: ❌ DISABLED
Fusion: ❌ DISABLED
Macro: ✓ ENABLED
❌ FAIL: Only 1/6 engines enabled
```

### Diagnosis

```bash
# Check config files
python bin/check_domain_engines.py --config configs/mvp/mvp_regime_routed_production.json

# Output shows:
# Config: mvp_regime_routed_production.json
# "enable_wyckoff": false
# "enable_smc": false
# "enable_temporal": false
# "enable_hob": false
# "enable_fusion": false
# "enable_macro": true
```

### Root Cause

Production configs have `enable_*: false` flags from historical debugging

### Fix

```bash
# Enable all engines in all production configs
python bin/enable_domain_engines.py --all

# Or manually edit configs:
vim configs/mvp/mvp_regime_routed_production.json

# Change all to true:
# "enable_wyckoff": true,
# "enable_smc": true,
# "enable_temporal": true,
# "enable_hob": true,
# "enable_fusion": true,
# "enable_macro": true

# Verify
python bin/check_domain_engines.py
# Expected: All 6 engines ENABLED ✓
```

---

## Issue 5: Feature Store Missing Critical Features

### Symptoms
```
Step 1: Feature coverage check
Coverage: 21.6% (102/470 features)
Missing critical features:
- funding_z
- oi_change_1h
- wyckoff_event_spring
- is_bullish_ob
❌ FAIL: Feature coverage < 98%
```

### Diagnosis

```bash
# List missing features
python bin/verify_feature_store.py --missing

# Output shows:
# Missing Features (368 total):
# Tier 1: 12 missing (technical indicators)
# Tier 2: 87 missing (multi-timeframe)
# Tier 3: 269 missing (regime + macro)
#
# Critical missing:
# - All Wyckoff features (engine not run)
# - All SMC features (engine not run)
# - OI features (pipeline not run)
```

### Root Cause

Feature store built without domain engines enabled

### Fix

```bash
# Rebuild feature store with all engines
python bin/build_feature_store.py --rebuild --all-engines

# Progress output:
# Building Tier 1 features... ✓ (102 features)
# Building Tier 2 features... ✓ (156 features)
# Building Wyckoff features... ✓ (23 features)
# Building SMC features... ✓ (45 features)
# Building Temporal features... ✓ (18 features)
# Building HOB features... ✓ (12 features)
# Building Fusion features... ✓ (34 features)
# Building Macro features... ✓ (80 features)
#
# Total: 470 features built

# Verify
python bin/verify_feature_store.py
# Expected: Coverage: 98.1% (461/470 features) ✓
```

---

## Issue 6: Wrong Calibrations Applied

### Symptoms
```
Step 7: Calibration verification
S4 funding_threshold: 0.5 (expected 0.72)
S4 oi_threshold: 0.3 (expected 0.45)
S1 exhaustion_threshold: 0.6 (expected 0.78)
❌ FAIL: Using vanilla calibrations
```

### Diagnosis

```bash
# Check which calibrations are loaded
python bin/verify_calibrations.py --s1 --s4 --s5

# Output shows:
# S4 (Funding Divergence):
# - Config: configs/mvp/mvp_bear_market_v1.json
# - Calibration source: VANILLA (not optimized)
# - Best Optuna trial: optuna_production_v2_order_block_retest.db (trial #47, PF 2.8)
# - Recommendation: Load trial #47 parameters
```

### Root Cause

Production configs still using vanilla defaults from initial development

### Fix

```bash
# Extract best parameters from Optuna
python bin/extract_thresholds.py \
  --db optuna_production_v2_order_block_retest.db \
  --trial 47 \
  --output configs/s4_optuna_best.json

# Apply to production config
python bin/apply_optimized_calibrations.py \
  --s4 \
  --source configs/s4_optuna_best.json

# Verify
python bin/verify_calibrations.py --s4
# Expected: S4 calibrations: OPTIMIZED (trial #47) ✓
```

---

## Issue 7: Chaos Windows Not Firing

### Symptoms
```
Step 6: Chaos window validation
Known chaos periods: 5
Chaos detected: 0
❌ FAIL: Chaos detection not working
```

### Diagnosis

```bash
# Test chaos window detection
python bin/test_chaos_windows.py --verbose

# Output shows:
# Testing chaos windows on known volatile periods:
# 2022-05-09 (LUNA crash): ❌ No chaos detected
# 2022-11-08 (FTX collapse): ❌ No chaos detected
# 2023-03-10 (SVB crisis): ❌ No chaos detected
#
# Possible causes:
# - HOB engine disabled
# - chaos_threshold too high (0.9 vs recommended 0.7)
# - Feature 'atr_ratio' missing
```

### Root Cause

1. HOB engine disabled (chaos detection lives in HOB)
2. Chaos threshold set too high
3. Volatility features missing from feature store

### Fix

```bash
# Fix 1: Enable HOB engine
python bin/enable_domain_engines.py --hob

# Fix 2: Lower chaos threshold
vim configs/mvp/mvp_regime_routed_production.json
# Change: "chaos_threshold": 0.7 (was 0.9)

# Fix 3: Rebuild feature store with volatility features
python bin/build_feature_store.py --features atr,volatility --rebuild

# Verify
python bin/test_chaos_windows.py
# Expected: 5/5 chaos periods detected ✓
```

---

## Issue 8: Regime Filter Too Restrictive

### Symptoms
```
S1 Test: 3 trades (expected ~40)
S4 Test: 7 trades (expected ~30)
Reason: Regime filter blocking trades
```

### Diagnosis

```bash
# Check regime distribution
python bin/analyze_regime_distribution.py --period test

# Output shows:
# Test Period (2023 H2):
# - risk_on: 62% of bars
# - neutral: 23% of bars
# - risk_off: 12% of bars
# - crisis: 3% of bars
#
# S1 allowed regimes: ['crisis'] (only 3% of period)
# S4 allowed regimes: ['risk_off'] (only 12% of period)
#
# Recommendation: Add 'risk_off' to S1, add 'neutral' to S4
```

### Root Cause

Archetype regime filters too conservative

### Fix

```bash
# Update archetype regime mappings
vim engine/archetypes/logic_v2_adapter.py

# Change:
ARCHETYPE_REGIMES = {
    'liquidity_vacuum': ['risk_off', 'crisis'],  # was ['crisis']
    'funding_divergence': ['risk_off', 'neutral'],  # was ['risk_off']
}

# Verify
python bin/run_archetype_suite.py --archetypes s1,s4 --periods test

# Expected:
# S1 Test: 38 trades (was 3) ✓
# S4 Test: 29 trades (was 7) ✓
```

---

## Issue 9: Identical Trades Across Archetypes

### Symptoms
```
S1, S4, S5 all producing identical trades:
2023-06-15 10:00:00 LONG
2023-07-03 14:00:00 LONG
2023-08-21 09:00:00 LONG
```

### Diagnosis

```bash
# Check trade uniqueness
python bin/check_trade_uniqueness.py --archetypes s1,s4,s5

# Output shows:
# Unique trades:
# - S1: 0 unique (100% overlap with S4, S5)
# - S4: 0 unique (100% overlap with S1, S5)
# - S5: 0 unique (100% overlap with S1, S4)
#
# Root cause: All archetypes using Tier-1 fallback
# - Entry: rsi_14 < 30 and volume > 2.0 * volume_sma
# - Exit: rsi_14 > 70
```

### Root Cause

All archetypes falling back to simple Tier-1 logic (see Issue 1)

### Fix

```bash
# This is the master fix - apply all fixes
./bin/apply_all_fixes.sh

# After fixes, verify uniqueness
python bin/check_trade_uniqueness.py --archetypes s1,s4,s5

# Expected:
# Unique trades:
# - S1: 78% unique (spring pattern specific)
# - S4: 85% unique (funding divergence specific)
# - S5: 67% unique (some overlap with S4 on extreme funding)
```

---

## Issue 10: Validation Passes But Production Fails

### Symptoms
```
Validation: All 9 steps PASS ✓
Production: S4 fires 0 trades in 7 days
```

### Diagnosis

```bash
# Check production vs backtest configs
python bin/compare_configs.py \
  --backtest configs/mvp/mvp_bear_market_v1.json \
  --production configs/production/s4_live.json

# Output shows differences:
# - Backtest: enable_paper_trading = true
# - Production: enable_paper_trading = false
# - Backtest: min_order_size = $10
# - Production: min_order_size = $1000 (blocking all trades!)
```

### Root Cause

Production config has different parameters than validated backtest config

### Fix

```bash
# Use validated config in production
cp configs/mvp/mvp_bear_market_v1.json configs/production/s4_live.json

# Or sync specific parameters
python bin/sync_config_params.py \
  --source configs/mvp/mvp_bear_market_v1.json \
  --dest configs/production/s4_live.json \
  --params min_order_size,enable_paper_trading
```

---

## Master Fix Script Issues

### Symptoms
```bash
./bin/apply_all_fixes.sh
Error: Phase 1 failed - feature_mapper.py not found
```

### Diagnosis

Feature mapper module not created yet (expected if running fixes before implementation)

### Fix

```bash
# Create feature mapper first
python bin/create_feature_mapper.py

# Then run fixes
./bin/apply_all_fixes.sh
```

---

## Quick Reference: Diagnostic Commands

```bash
# Overall health
./bin/quick_health_check.sh

# Specific issues
python bin/check_domain_engines.py           # Engines enabled?
python bin/check_tier1_fallback.py          # Fallback rate?
python bin/check_funding_data.py            # Data quality?
python bin/verify_feature_store.py          # Feature coverage?
python bin/verify_calibrations.py           # Right calibrations?
python bin/test_chaos_windows.py            # Chaos detection?
python bin/check_trade_uniqueness.py        # Unique trades?
python bin/diagnose_archetype_issues.sh     # Master diagnostic
```

---

## Getting Help

1. **Check this guide** for your specific issue
2. **Run master diagnostic:** `./bin/diagnose_archetype_issues.sh`
3. **Review validation logs:** `logs/archetype_validation/`
4. **Check recent changes:** `git log --oneline --graph`
5. **Review error messages** in detail (use `--verbose` flags)

---

## Prevention

### Before Making Changes
```bash
# Back up configs
cp -r configs/mvp configs/mvp_backup_$(date +%Y%m%d)

# Run validation baseline
./bin/validate_archetype_engine.sh --full > validation_before.txt
```

### After Making Changes
```bash
# Re-run validation
./bin/validate_archetype_engine.sh --full > validation_after.txt

# Compare
diff validation_before.txt validation_after.txt
```

### Weekly Maintenance
```bash
# Run quick validation
./bin/validate_archetype_engine.sh --quick

# Check data quality
python bin/check_funding_data.py

# Review logs
tail -100 logs/archetype_validation/latest.txt
```

---

**Document Version:** 1.0
**Last Updated:** 2025-12-08
**Maintained By:** Archetype Engine Team
