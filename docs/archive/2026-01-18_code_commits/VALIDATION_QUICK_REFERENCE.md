# Validation Quick Reference

**1-page checklist for archetype engine validation**

---

## 9-Step Validation Protocol

| Step | Check | Threshold | Command | Pass Criteria |
|------|-------|-----------|---------|---------------|
| **1** | Feature Coverage | ≥ 98% | `python bin/check_domain_engines.py` | Features accessible: 461+/470 |
| **2** | Feature Mapping | 100% | `python bin/test_feature_mapper.py` | All critical features mapped |
| **3** | Domain Engines | 6/6 ON | `python bin/check_domain_engines.py` | All engines enabled |
| **4** | Tier-1 Fallback | < 30% | `python bin/check_tier1_fallback.py --archetype s1` | Fallback rate low |
| **5** | OI/Funding Data | < 20% null | `python bin/check_funding_data.py` | Data quality high |
| **6** | Chaos Windows | Firing | `python bin/test_chaos_windows.py` | Windows detect correctly |
| **7** | Calibrations | Applied | `python bin/verify_calibrations.py` | Optuna params loaded |
| **8** | Test Performance | Meets min | `python bin/run_archetype_suite.py --periods test` | S4>2.2, S1>1.8, S5>1.6 |
| **9** | Baseline Compare | Competitive | `python bin/compare_archetypes_vs_baselines.py` | Within 20% of best |

---

## Quick Validation (5 minutes)

```bash
# Run all 9 steps
./bin/validate_archetype_engine.sh --quick

# Expected output:
# ✓ Step 1: Feature coverage 98.1%
# ✓ Step 2: All features mapped
# ✓ Step 3: 6/6 engines enabled
# ✓ Step 4: Tier-1 fallback 12.3%
# ✓ Step 5: OI null 18.2%
# ✓ Step 6: Chaos windows OK
# ✓ Step 7: Calibrations applied
# ✓ Step 8: Performance meets minimums
# ✓ Step 9: Competitive with baselines
#
# VALIDATION PASSED ✓
```

---

## Full Validation (30 minutes)

```bash
# Run comprehensive validation with detailed reports
./bin/validate_archetype_engine.sh --full

# Generates:
# - logs/archetype_validation/feature_coverage.txt
# - logs/archetype_validation/engine_status.txt
# - logs/archetype_validation/fallback_analysis.txt
# - logs/archetype_validation/data_quality.txt
# - logs/archetype_validation/performance_report.txt
# - logs/archetype_validation/summary.txt
```

---

## Per-Archetype Validation

```bash
# Validate specific archetype
./bin/validate_archetype_engine.sh --archetype s1

# Check steps:
# ✓ Features required by S1: all accessible
# ✓ Wyckoff engine: ENABLED
# ✓ Macro engine: ENABLED
# ✓ Tier-1 fallback: 11.2% (< 30%)
# ✓ Volume/wick exhaustion data: < 5% null
# ✓ Test PF: 2.1 (> 1.8 minimum)
# ✓ Competitive with baseline: within 15%
#
# S1 VALIDATION PASSED ✓
```

---

## Critical Thresholds

### Step 1: Feature Coverage
- **Minimum:** 98% (461/470 features)
- **Acceptable:** 95-98% (some experimental features missing)
- **Fail:** < 95% (missing critical features)

### Step 2: Feature Mapping
- **Pass:** All critical features have mappings
- **Fail:** Any of these missing: `funding_z`, `volume_climax_3b`, `wick_exhaustion_3b`, `btc_d`, `order_block_bull`

### Step 3: Domain Engines
- **Pass:** 6/6 engines enabled
- **Acceptable:** 5/6 (Temporal optional for some archetypes)
- **Fail:** < 5/6 (Macro, Wyckoff, SMC, Fusion, HOB are required)

### Step 4: Tier-1 Fallback
- **Excellent:** < 15%
- **Good:** 15-30%
- **Acceptable:** 30-50%
- **Fail:** > 50% (archetypes not using domain engines)

### Step 5: OI/Funding Data Quality
- **Excellent:** < 10% null
- **Good:** 10-20% null
- **Acceptable:** 20-30% null (for S4/S5 only)
- **Fail:** > 30% null

### Step 6: Chaos Windows
- **Pass:** Detects chaos in known volatile periods
- **Fail:** No chaos detected OR always detecting chaos

### Step 7: Calibrations
- **Pass:** Optuna-optimized parameters loaded
- **Fail:** Using vanilla defaults

### Step 8: Test Performance (Minimum PF)
- **S1:** PF > 1.8
- **S4:** PF > 2.2
- **S5:** PF > 1.6

### Step 9: Baseline Comparison
- **Excellent:** Within 10% of best baseline
- **Good:** Within 20% of best baseline
- **Acceptable:** Within 30% of best baseline
- **Fail:** > 30% below best baseline

---

## Validation Decision Matrix

| Steps Passed | Decision | Action |
|--------------|----------|--------|
| **9/9** | ✅ DEPLOY | Proceed to production |
| **8/9** | ⚠️ REVIEW | Check which step failed, fix if critical |
| **7/9** | ⚠️ FIX | Multiple issues, do not deploy yet |
| **< 7/9** | ❌ HALT | Critical failures, re-run fixes |

**Critical Steps (must pass):**
- Step 3: Domain Engines (6/6)
- Step 5: Data Quality (< 20% null)
- Step 8: Performance (meets minimums)

**If any critical step fails → DO NOT DEPLOY**

---

## Common Failure Patterns

### Pattern 1: High Tier-1 Fallback (Step 4)
**Symptoms:** Step 4 shows > 50% fallback rate

**Diagnosis:**
```bash
python bin/check_tier1_fallback.py --archetype s1 --verbose
```

**Causes:**
1. Feature name mismatches (FeatureMapper not used)
2. Domain engines disabled (check Step 3)
3. Feature store missing data (check Step 1)

**Fix:**
```bash
# Re-run feature mapping fix
python bin/fix_feature_names.py --apply

# Enable engines
python bin/enable_domain_engines.py --all

# Rebuild feature store
python bin/build_feature_store.py --rebuild
```

---

### Pattern 2: Low Performance (Step 8)
**Symptoms:** Step 8 shows Test PF below minimums

**Diagnosis:**
```bash
python bin/diagnose_archetype_issues.sh --archetype s4
```

**Causes:**
1. Wrong calibrations (vanilla defaults)
2. Missing OI data (for S4/S5)
3. Regime filter too restrictive

**Fix:**
```bash
# Apply optimized calibrations
python bin/apply_optimized_calibrations.py --s4

# Backfill OI data
python bin/fix_oi_change_pipeline.py

# Re-run optimization if needed
python bin/optimize_s4_calibration.py --trials 100
```

---

### Pattern 3: Data Quality Issues (Step 5)
**Symptoms:** Step 5 shows > 30% null for OI/Funding

**Diagnosis:**
```bash
python bin/check_funding_data.py --verbose
```

**Causes:**
1. OI pipeline not run for 2022-2023
2. Funding data missing from exchange API
3. Feature store not rebuilt after backfill

**Fix:**
```bash
# Backfill OI data
python bin/fix_oi_change_pipeline.py

# Rebuild feature store
python bin/build_feature_store.py --rebuild

# Verify
python bin/check_funding_data.py
```

---

## Pre-Deployment Checklist

Before deploying to production:

- [ ] All 9 validation steps passed
- [ ] Feature coverage ≥ 98%
- [ ] All 6 domain engines enabled
- [ ] Tier-1 fallback < 30%
- [ ] OI/Funding data < 20% null
- [ ] Test PF meets minimums (S4 > 2.2, S1 > 1.8, S5 > 1.6)
- [ ] Competitive with baselines (within 20%)
- [ ] Logs reviewed for warnings/errors
- [ ] Configs backed up
- [ ] Rollback plan prepared

**Sign-off:** _________________  **Date:** _________

---

## Validation Frequency

| Scenario | Frequency | Command |
|----------|-----------|---------|
| **Development** | Before each PR | `./bin/validate_archetype_engine.sh --quick` |
| **Pre-Deploy** | Before production push | `./bin/validate_archetype_engine.sh --full` |
| **Production** | Weekly | `./bin/validate_archetype_engine.sh --quick` |
| **After Regime Change** | As needed | `./bin/validate_archetype_engine.sh --archetype all` |
| **After Code Changes** | Before merge | `./bin/validate_archetype_engine.sh --full` |

---

## Validation Logs

All validation runs write to:
```
logs/archetype_validation/
├── YYYYMMDD_HHMMSS_summary.txt          # Quick summary
├── YYYYMMDD_HHMMSS_feature_coverage.txt # Step 1 details
├── YYYYMMDD_HHMMSS_engine_status.txt    # Step 3 details
├── YYYYMMDD_HHMMSS_fallback_analysis.txt# Step 4 details
├── YYYYMMDD_HHMMSS_data_quality.txt     # Step 5 details
└── YYYYMMDD_HHMMSS_performance.txt      # Step 8 details
```

**Review logs if any step fails.**

---

## Quick Diagnostic Commands

```bash
# Check overall health
./bin/quick_health_check.sh

# Check specific archetype
python bin/diagnose_archetype_issues.sh --archetype s1

# Check domain engines
python bin/check_domain_engines.py --verbose

# Check feature store
python bin/verify_feature_store.py

# Check calibrations
python bin/verify_calibrations.py --s1 --s4 --s5

# Check OI data
python bin/check_funding_data.py --verbose
```

---

## Support

**For validation failures:** See `TROUBLESHOOTING_GUIDE.md`

**For detailed technical info:** See `ARCHETYPE_ENGINE_FIX_COMPLETE.md`

**For quick fixes:** See `ARCHETYPE_ENGINE_QUICK_START.md`

---

**Document Version:** 1.0
**Last Updated:** 2025-12-08
**Print-Friendly:** Yes
**Shareable:** Yes

## Phase 1: Quick Validation (2022 Bear Market)

### Run All Quick Tests
```bash
./bin/run_phase1_quick_validation.sh
```

**Duration**: ~5-10 minutes per config
**Output**: `results/phase1_quick_validation/`

### Analyze Results
```bash
python3 bin/analyze_quick_test_results.py
```

**Generates**:
- `analysis_summary.csv` - Complete metrics
- `trade_count_vs_pf.png` - Scatter plot
- `performance_heatmap.png` - Metrics heatmap
- `recommendations.md` - Next steps

### View Results
```bash
# Summary table
cat results/phase1_quick_validation/summary_2022_bear_market.txt

# Recommendations
cat results/phase1_quick_validation/recommendations.md

# Plots (macOS)
open results/phase1_quick_validation/*.png
```

---

## Phase 4: OOS Validation (2024 Data)

### Run OOS Validation
```bash
# Basic (auto-selects best study)
python3 bin/run_phase4_oos_validation.py

# Specific study
python3 bin/run_phase4_oos_validation.py \
  --study-name archetype_trap_within_trend_v2

# Custom settings
python3 bin/run_phase4_oos_validation.py \
  --optuna-db optuna_archetypes_v2.db \
  --top-n 5 \
  --base-config configs/mvp_bull_market_v1.json
```

**Duration**: ~15-30 minutes for 5 trials
**Output**: `results/phase4_oos_validation/`

### View Results
```bash
# Main report
cat results/phase4_oos_validation/oos_validation_report.md

# Summary CSV
open results/phase4_oos_validation/oos_validation_summary.csv

# Individual trial results
ls -la results/phase4_oos_validation/trial_*/
```

---

## Complete Pipeline

```bash
# 1. Quick validation
./bin/run_phase1_quick_validation.sh
python3 bin/analyze_quick_test_results.py

# 2. Review recommendations
cat results/phase1_quick_validation/recommendations.md

# 3. Optimize (Phase 2)
python3 bin/optuna_parallel_archetypes_v2.py \
  --base-config configs/quick_test_optimized.json \
  --trials 100

# 4. OOS validation
python3 bin/run_phase4_oos_validation.py

# 5. Deploy to production
cp results/phase4_oos_validation/trial_5/trial_5_config.json \
   configs/production/btc_production_v1.json
```

---

## File Locations

```
bin/
├── run_phase1_quick_validation.sh      # Phase 1 runner
├── analyze_quick_test_results.py       # Phase 1 analyzer
└── run_phase4_oos_validation.py        # Phase 4 OOS validator

results/
├── phase1_quick_validation/
│   ├── README.md                       # Phase 1 guide
│   ├── summary_2022_bear_market.txt    # Quick summary
│   ├── analysis_summary.csv            # Full metrics
│   ├── recommendations.md              # Recommendations
│   └── *.png                           # Plots
└── phase4_oos_validation/
    ├── README.md                       # Phase 4 guide
    ├── oos_validation_report.md        # Main report
    ├── oos_validation_summary.csv      # Summary CSV
    └── trial_{N}/                      # Per-trial results

docs/
└── VALIDATION_SCRIPTS_GUIDE.md         # Complete guide
```

---

## Help Commands

```bash
# Script 2: Phase 4 OOS validation
python3 bin/run_phase4_oos_validation.py --help

# Script 3: Results analyzer
python3 bin/analyze_quick_test_results.py --help
```

---

## Target Metrics

**Phase 1 (Quick Validation)**:
- Trade Count: 25-40 trades
- Profit Factor: >1.5 (ideally >3.0)
- Win Rate: >40%
- Max Drawdown: <25%

**Phase 4 (OOS Validation)**:
- PF Degradation: <30%
- Sharpe Degradation: <30%
- OOS PF: >2.0
- Consistent trade count across periods

---

## Troubleshooting

**No configs in target range**:
```bash
# Adjust thresholds in configs and re-run
# Lower fusion_threshold → more trades
# Raise fusion_threshold → fewer trades
```

**All trials overfitting**:
```bash
# Reduce optimization intensity
# Use wider parameter ranges
# Add regularization
```

**Missing dependencies**:
```bash
pip3 install pandas numpy matplotlib seaborn optuna
```

---

## Documentation

- **Complete Guide**: `docs/VALIDATION_SCRIPTS_GUIDE.md`
- **Phase 1 README**: `results/phase1_quick_validation/README.md`
- **Phase 4 README**: `results/phase4_oos_validation/README.md`
- **Delivery Summary**: `VALIDATION_SCRIPTS_DELIVERY_SUMMARY.md`
