# Validation Protocol Quick Start

**Last Updated:** 2025-12-08

Fast reference for running the 9-step validation protocol.

---

## TL;DR

```bash
# Run everything
bash bin/validate_archetype_engine.sh --full

# Check results
cat validation_report_*.txt
```

**If all green:** Ready for deployment
**If any red:** Fix infrastructure before trusting results

---

## The 9 Steps (30 Second Version)

1. **Feature Coverage** - Do required features exist? (≥98%)
2. **Feature Mapping** - Do config names match store columns?
3. **Engines ON** - Are all 6 domain engines enabled?
4. **No Fallback** - Are trades from archetypes, not Tier1? (<30% fallback)
5. **Data Quality** - Is funding/OI data loaded? (<20% null)
6. **Chaos Windows** - Do archetypes fire on known events?
7. **Calibrations** - Are Optuna-optimized params applied?
8. **Full Validation** - Do archetypes meet minimum PF? (S4≥2.2, S1≥1.8, S5≥1.6)
9. **Baseline Compare** - Do archetypes beat simple strategies?

---

## Command Cheat Sheet

### Run Full Protocol
```bash
bash bin/validate_archetype_engine.sh --full
```

### Run Specific Steps
```bash
# Infrastructure checks (Steps 1-3)
bash bin/validate_archetype_engine.sh --steps 1-3

# Plumbing checks (Steps 4-6)
bash bin/validate_archetype_engine.sh --steps 4-6

# Performance validation (Steps 7-9)
bash bin/validate_archetype_engine.sh --steps 7-9

# Single step
bash bin/validate_archetype_engine.sh --step 4
```

### Individual Script Usage
```bash
# Step 1: Feature coverage
python bin/audit_archetype_pipeline.py

# Step 2: Feature mapping
python bin/verify_feature_mapping.py

# Step 3: Domain engines
python bin/check_domain_engines.py --s1 --s4 --s5

# Step 4: Tier1 fallback
python bin/check_tier1_fallback.py --test-period 2022-05-01:2022-08-01

# Step 5: Data quality
python bin/check_funding_data.py
python bin/check_oi_data.py

# Step 6: Chaos windows
python bin/test_chaos_windows.py --all

# Step 7: Calibrations
python bin/apply_optimized_calibrations.py --s1 --s4 --s5

# Step 8: Full validation
python bin/run_archetype_suite.py --periods train,test,oos

# Step 9: Baseline comparison
python bin/compare_archetypes_vs_baselines.py
```

---

## Quick Diagnostics

### Check if Validation Already Run
```bash
ls -lt validation_report_*.txt | head -1
```

### View Last Validation Results
```bash
cat validation_report_*.txt | grep -E "PASS|FAIL|VERDICT"
```

### Count Passed Steps
```bash
grep -c "PASS" validation_report_*.txt
```

### Find Failed Steps
```bash
grep "FAIL" validation_report_*.txt
```

---

## Common Fixes

### Step 1 Fails (Feature Coverage < 98%)
```bash
# Fix feature names
python bin/fix_feature_names.py --apply

# Re-audit
python bin/audit_archetype_pipeline.py
```

### Step 2 Fails (Feature Mapping Errors)
```bash
# Update feature mapper
vi engine/features/feature_mapper.py  # Add missing mappings

# Re-verify
python bin/verify_feature_mapping.py
```

### Step 3 Fails (Engines Disabled)
```bash
# Enable all domain engines
python bin/enable_domain_engines.py --all

# Re-check
python bin/check_domain_engines.py --s1 --s4 --s5
```

### Step 5 Fails (Missing OI/Funding Data)
```bash
# Backfill data
python bin/fix_oi_change_pipeline.py

# Re-check
python bin/check_funding_data.py
python bin/check_oi_data.py
```

### Step 7 Fails (No Optimized Params)
```bash
# Extract best trial manually
python bin/extract_best_trial.py --archetype s4 --apply

# Re-check
grep "optimized" configs/s4_optimized_oos_test.json
```

---

## Decision Tree

```
Run Step 1 → Feature Coverage ≥ 98%?
  ├─ YES → Proceed to Step 2
  └─ NO → Fix features, STOP

Run Step 2 → All features mapped?
  ├─ YES → Proceed to Step 3
  └─ NO → Update mapper, STOP

Run Step 3 → 18/18 engines enabled?
  ├─ YES → Proceed to Step 4
  └─ NO → Enable engines, STOP

Run Step 4 → Fallback < 30%?
  ├─ YES → Proceed to Step 5
  └─ NO → Review Steps 1-3, STOP

Run Step 5 → Data null < 20%?
  ├─ YES → Proceed to Step 6
  └─ NO → Backfill data, STOP

Run Step 6 → Chaos windows fire?
  ├─ YES → Proceed to Step 7
  └─ NO → Debug plumbing, STOP

Run Step 7 → Configs optimized?
  ├─ YES → Proceed to Step 8
  └─ NO → Apply calibrations, STOP

Run Step 8 → Performance meets minimums?
  ├─ YES → Proceed to Step 9
  └─ NO → Review all steps, PARTIAL

Run Step 9 → Beat or compete with baselines?
  ├─ Scenario A (Beat) → DEPLOY ARCHETYPES
  ├─ Scenario B (Compete) → DEPLOY HYBRID
  └─ Scenario C (Lose) → REWORK OR KILL
```

---

## Pass Criteria Summary

| Step | Metric | Target | Critical |
|------|--------|--------|----------|
| 1 | Feature Coverage | ≥ 98% | YES |
| 2 | Mapped Features | 100% | YES |
| 3 | Engines Enabled | 18/18 | YES |
| 4 | Fallback Trades | < 30% | YES |
| 5 | Data Null % | < 20% | YES |
| 6 | Chaos Trades | > 0 | YES |
| 7 | Optimized Flag | true | YES |
| 8 | S4 Test PF | ≥ 2.2 | NO* |
| 8 | S1 Test PF | ≥ 1.8 | NO* |
| 8 | S5 Test PF | ≥ 1.6 | NO* |
| 9 | vs Baseline | Competitive | NO* |

*Steps 8-9 failures indicate performance issues, not infrastructure problems.

---

## Deployment Decisions (Step 9)

### Scenario A: Clear Winners
- **Criteria:** S4 > 3.24, S1 > 2.10
- **Action:** Deploy archetypes as main engine
- **Commands:**
  ```bash
  python bin/generate_production_configs.py --deploy archetypes
  python bin/deploy_to_paper_trading.py --s4 --s1
  ```

### Scenario B: Competitive
- **Criteria:** S4 2.5-3.2
- **Action:** Deploy hybrid (60% archetypes, 40% baselines)
- **Commands:**
  ```bash
  python bin/generate_hybrid_configs.py
  python bin/deploy_to_paper_trading.py --hybrid
  ```

### Scenario C: Underperformers
- **Criteria:** All < 2.0
- **Action:** Rework or kill archetypes
- **Commands:**
  ```bash
  python bin/analyze_failure_modes.py
  # OR deploy baselines instead:
  python bin/deploy_baseline_strategy.py --strategy SMA50x200
  ```

---

## Validation Report Structure

```
========================================
QUANT LAB VALIDATION PROTOCOL REPORT
Generated: 2025-12-08 14:23:45
========================================

[PASS] STEP 1: Feature coverage 98.2%
[PASS] STEP 2: All features mapped
[PASS] STEP 3: 18/18 engines enabled
[PASS] STEP 4: Fallback 18.2%
[PASS] STEP 5: Data coverage acceptable
[PASS] STEP 6: Chaos windows firing
[PASS] STEP 7: Configs optimized
[PASS] STEP 8: Performance meets minimums
[PASS] STEP 9: Competitive with baselines

VERDICT: 100% VALIDATED
```

---

## File Locations

**Documentation:**
- `/QUANT_LAB_VALIDATION_PROTOCOL.md` - Full protocol (read first)
- `/VALIDATION_INFRASTRUCTURE_COMPLETE.md` - Implementation details
- `/VALIDATION_QUICK_START.md` - This file

**Master Script:**
- `/bin/validate_archetype_engine.sh` - Run this

**Validation Scripts:**
- `/bin/check_domain_engines.py`
- `/bin/check_tier1_fallback.py`
- `/bin/check_funding_data.py`
- `/bin/check_oi_data.py`
- `/bin/test_chaos_windows.py`
- `/bin/verify_feature_mapping.py`
- `/bin/apply_optimized_calibrations.py`
- `/bin/run_archetype_suite.py`
- `/bin/compare_archetypes_vs_baselines.py`

**Outputs:**
- `validation_report_YYYYMMDD_HHMMSS.txt` - Validation results
- `archetype_validation_results.csv` - Performance metrics (if saved)

---

## When to Run

**Required:**
- Before production deployment
- After archetype code changes
- After feature store updates

**Recommended:**
- Monthly sanity check
- After config optimizations
- Before major releases

---

## Troubleshooting

### "Config not found" errors
```bash
# Check configs exist
ls configs/s1_v2_production.json
ls configs/s4_optimized_oos_test.json
ls configs/s5_production.json
```

### "Data file not found" errors
```bash
# Check feature store exists
ls data/features_1h.parquet

# Or update config with correct path
vi configs/s4_optimized_oos_test.json
# Set "data_path": "path/to/your/features.parquet"
```

### "No optimized parameters" warnings
- Normal if Optuna optimization not yet run
- Scripts will check for "optimized: true" flag in configs
- Manually add if configs already have tuned params:
  ```json
  {
    "optimized": true,
    "optuna_trial_id": 999
  }
  ```

### Validation hangs on Step 8
- Full-period validation can take 30-60 minutes
- Check progress in terminal output
- Use `--periods test` for faster runs during debugging

---

## Success Indicators

**Green Lights:**
- All 9 steps show `✓ PASS`
- Final verdict: `100% VALIDATED`
- Validation report shows no `[FAIL]` entries

**Yellow Lights:**
- Steps 1-7 pass, Step 8 below minimums
- Verdict: `PARTIAL`
- Action: Review calibrations, retry optimization

**Red Lights:**
- Any of Steps 1-7 fail
- Verdict: `FAILED`
- Action: Fix infrastructure before proceeding

---

## Example Session

```bash
# Start validation
$ bash bin/validate_archetype_engine.sh --full

========================================
QUANT LAB VALIDATION PROTOCOL
========================================
Report: validation_report_20251208_142345.txt

========================================
STEP 1: Confirm Feature Store Coverage
========================================

Running feature coverage audit...
✓ PASS: Feature coverage: 98.2% (≥ 98% required)

========================================
STEP 2: Validate Feature Name Mapping
========================================

Verifying feature name mappings...
✓ PASS: All feature mappings verified

[... continues through all 9 steps ...]

========================================
VALIDATION SUMMARY
========================================

Steps Passed: 9
Steps Failed: 0
Total Steps:  9

========================================
✓ 100% VALIDATED - READY FOR PRODUCTION
========================================
```

---

**Remember:** If any infrastructure step (1-7) fails, STOP. Fix before proceeding. You cannot trust performance results if infrastructure is broken.

---

**For full details:** See `/QUANT_LAB_VALIDATION_PROTOCOL.md`
