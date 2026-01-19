# Quant Lab Validation Protocol
## The 9-Step Gold Standard for Archetype Testing

This protocol ensures you are testing the REAL Bull Machine archetype engine with 100% of its intelligence active, not a handicapped version.

---

## Overview

**If all 9 steps pass → You are testing correctly**
**If any step fails → Stop and fix before proceeding**

Total time: ~4 hours (first run), ~30 minutes (subsequent runs)

---

## STEP 1: Confirm Feature Store Coverage

**Purpose:** Verify all required domain features exist and are accessible

**Command:**
```bash
python bin/audit_archetype_pipeline.py
```

**PASS CRITERIA:**
- ✅ ≥ 98% of required features exist (36/37 or better)
- ✅ Missing features < 3
- ✅ Null percentage < 5% (except macro series which can be higher)
- ✅ Funding & OI present for S4/S5 test periods
- ✅ Crisis, climax, exhaustion features exist with correct names

**Expected Output:**
```
Domain Coverage:
  Wyckoff:   █████████░ 90% (9/10 features)
  SMC:       ██████████ 100% (10/10 features)
  Temporal:  ████████░░ 80% (8/10 features)
  Macro:     ██████████ 100% (5/5 features)
  Funding:   ████████░░ 80% (4/5 features)

Overall: 98.0% coverage ✓ PASS
```

**If FAIL:**
```bash
# Fix feature mappings
python bin/fix_feature_names.py --apply

# Re-run audit
python bin/audit_archetype_pipeline.py
```

---

## STEP 2: Validate Feature Name Mapping

**Purpose:** Ensure configs reference exact column names from feature store

**Command:**
```bash
python bin/verify_feature_mapping.py
```

**PASS CRITERIA:**
- ✅ All config-expected features map to actual store columns
- ✅ No "feature not found" errors
- ✅ FeatureMapper canonical mappings verified

**Expected Mappings:**
| Config Expects | Feature Store Has | Status |
|----------------|-------------------|--------|
| funding_z | funding_Z | ✓ Mapped |
| volume_climax_3b | volume_climax_last_3b | ✓ Mapped |
| wick_exhaustion_3b | wick_exhaustion_last_3b | ✓ Mapped |
| btc_d | BTC.D | ✓ Mapped |
| usdt_d | USDT.D | ✓ Mapped |
| order_block_bull | is_bullish_ob | ✓ Mapped |

**If FAIL:**
```bash
# Update feature mapper with missing mappings
vi engine/features/feature_mapper.py  # Add missing mappings

# Re-run verification
python bin/verify_feature_mapping.py
```

---

## STEP 3: Confirm Domain Engines Are ON

**Purpose:** Verify all 6 domain engines are enabled in configs

**Command:**
```bash
python bin/check_domain_engines.py --s1 --s4 --s5
```

**PASS CRITERIA:**
- ✅ All 6 engines enabled: Wyckoff, SMC, Temporal, HOB, Fusion, Macro
- ✅ Backtest logs show activation messages
- ✅ Fusion score is non-zero on trades

**Expected Output:**
```
S1 (configs/s1_v2_production.json):
  Wyckoff:  ✓ ENABLED
  SMC:      ✓ ENABLED
  Temporal: ✓ ENABLED
  HOB:      ✓ ENABLED
  Fusion:   ✓ ENABLED
  Macro:    ✓ ENABLED

S4 (configs/s4_optimized_oos_test.json):
  [Same checks...]

Overall: 18/18 engines enabled ✓ PASS
```

**If FAIL:**
```bash
# Enable all domain engines
python bin/enable_domain_engines.py --all

# Re-run check
python bin/check_domain_engines.py --s1 --s4 --s5
```

---

## STEP 4: Confirm Archetype NOT Falling Back to Tier1

**Purpose:** Verify trades originate from archetype logic, not fallback

**Command:**
```bash
python bin/check_tier1_fallback.py --test-period 2022-05-01:2022-08-01
```

**PASS CRITERIA:**
- ✅ ≥ 70% of trades from S1/S4/S5 archetype logic
- ✅ < 30% fallback trades
- ✅ Logs show "Fusion Score" and "Liquidity Score" on trades

**Expected Log Pattern (CORRECT):**
```
Trade Triggered: S4 Funding Divergence Signal
  Fusion Score: 0.41
  Liquidity Score: 0.62
  Confluence: 75%
  Regime: risk_off
```

**Wrong Log Pattern (INCORRECT):**
```
Trade Triggered: Tier1 Liquidity → FALLBACK MODE
```

**If FAIL:**
- Indicates feature access issues or thresholds too strict
- Check Step 1-2 again (features accessible?)
- Review calibrations (thresholds preventing signals?)

---

## STEP 5: Confirm OI/Funding Are Loaded Properly

**Purpose:** Verify funding and OI data available for S4/S5

**Commands:**
```bash
python bin/check_funding_data.py
python bin/check_oi_data.py
```

**PASS CRITERIA:**
- ✅ Funding rate < 20% null for 2022-2024
- ✅ OI data < 20% null for 2022-2024
- ✅ No NaN warnings in S4/S5 backtest logs
- ✅ S4 trade count > 20 in 2022-2023 test

**If FAIL:**
```bash
# Backfill OI data
python bin/fix_oi_change_pipeline.py

# Re-check
python bin/check_oi_data.py
```

---

## STEP 6: Reproduce Short-Window Behavior (Plumbing Sanity)

**Purpose:** Verify archetypes fire on known chaos events

**Chaos Windows to Test:**
- Terra (LUNA) collapse: 2022-05-01 to 2022-05-31
- FTX collapse: 2022-11-01 to 2022-11-30
- CPI shock: 2022-06-10 to 2022-06-17

**Command:**
```bash
python bin/test_chaos_windows.py --s4
```

**PASS CRITERIA:**
- ✅ Non-zero trades in each chaos window
- ✅ Fusion/RuntimeContext features show realistic values (not all zeros)
- ✅ S1/S4/S5 signals NOT identical (different archetypes fire differently)

**Expected:**
```
Terra Collapse (2022-05-01 to 2022-05-31):
  S4: 12 trades, PF 1.8 ✓
  S1: 8 trades, PF 2.1 ✓
  S5: 5 trades, PF 1.2 ✓
  Signal Correlation: 0.32 ✓ (low = good, means different signals)

FTX Collapse (2022-11-01 to 2022-11-30):
  S4: 15 trades, PF 2.3 ✓
  S1: 11 trades, PF 1.9 ✓
  S5: 7 trades, PF 1.5 ✓
```

**If FAIL:**
- 0 trades → Feature access or threshold issues
- Identical signals → Tier1 fallback dominating
- All zeros for fusion scores → Domain engines not activated

---

## STEP 7: Apply OPTIMIZED CALIBRATIONS

**Purpose:** Use Optuna-derived thresholds, not vanilla defaults

**Command:**
```bash
python bin/apply_optimized_calibrations.py --s1 --s4 --s5
```

**PASS CRITERIA:**
- ✅ Every config JSON has Optuna-derived thresholds
- ✅ Backtest logs show "Loaded optimized params from Optuna trial X"
- ✅ Config files have "optimized: true" flag

**Verification:**
```bash
# Check configs have optimized params
grep -A 5 "optimized" configs/s4_optimized_oos_test.json

# Should show:
# "optimized": true,
# "optuna_trial_id": 42,
# "fusion_threshold": 0.7824,  # NOT vanilla 0.5
```

**If FAIL:**
```bash
# Manually load best trial parameters
python bin/extract_best_trial.py --archetype s4 --apply
```

---

## STEP 8: Full-Period Validation (The REAL Test)

**Purpose:** Validate on full train/test/OOS periods

**Command:**
```bash
python bin/run_archetype_suite.py --periods train,test,oos
```

**Periods:**
- Train: 2020-01-01 to 2022-12-31
- Test: 2023-01-01 to 2023-12-31
- OOS: 2024-01-01 to 2024-12-31

**PASS CRITERIA (Minimum Acceptable Performance):**

| Archetype | Test PF | Min Acceptable | Trades | Overfit | Status |
|-----------|---------|----------------|--------|---------|--------|
| S4 | ? | **≥ 2.2** | > 40 | < 0.5 | ✓ / ✗ |
| S1 | ? | **≥ 1.8** | > 40 | < 0.5 | ✓ / ✗ |
| S5 | ? | **≥ 1.6** | > 30 | < 0.5 | ✓ / ✗ |

**If PASS:**
- Proceed to Step 9 (baseline comparison)

**If FAIL:**
- Review Steps 1-7 (are all fixes applied?)
- Check logs for errors/warnings
- Investigate regime classification (is macro_regime correct?)

---

## STEP 9: Compare Against Baselines (Final Truth)

**Purpose:** Determine if archetypes beat simple baselines

**Command:**
```bash
python bin/compare_archetypes_vs_baselines.py
```

**Baselines to Beat:**
- SMA50x200 Crossover: Test PF 3.24
- VolTarget Trend: Test PF 2.10
- RSI Mean Reversion: Test PF 1.70

**PASS CRITERIA:**

**Scenario A: Clear Winners** (Deploy archetypes as main engine)
- S4 Test PF > 3.24 (beats best baseline)
- S1 Test PF > 2.10 (beats second baseline)
- At least one archetype in top 3 overall

**Scenario B: Competitive** (Deploy hybrid)
- S4 Test PF 2.5-3.2 (close to baseline)
- Worth deploying for diversification
- Different correlation profile

**Scenario C: Underperformers** (Rework or kill)
- All archetypes < 2.0 PF
- Worse than simple baselines
- Requires redesign or kill

**If PASS (Scenario A or B):**
```bash
# Generate production deployment configs
python bin/generate_production_configs.py --deploy archetypes

# Begin paper trading
python bin/deploy_to_paper_trading.py --s4 --s1
```

**If FAIL (Scenario C):**
- Document failure modes
- Consider: Fix temporal domain, regime classification, or redesign

---

## Quick Validation Command

Run all 9 steps in one command:

```bash
bash bin/validate_archetype_engine.sh --full
```

This will:
1. Run all 9 validation steps sequentially
2. Stop at first failure with diagnostic info
3. Generate comprehensive validation report
4. Provide clear PASS/FAIL verdict

---

## Success Criteria Summary

**100% VALIDATED** if:
- ✅ Steps 1-3: Feature access, mapping, engines enabled
- ✅ Steps 4-6: Plumbing correct, non-fallback trades, data loaded
- ✅ Step 7: Optimized calibrations applied
- ✅ Step 8: Test PF meets minimums (S4 > 2.2, S1 > 1.8, S5 > 1.6)
- ✅ Step 9: Competitive with or beats baselines

**PARTIAL** if:
- ⚠️ Steps 1-7 pass but Step 8 below minimums → Rework needed
- ⚠️ Steps 1-8 pass but Step 9 loses to baselines → Redesign or kill

**FAIL** if:
- ✗ Any of Steps 1-7 fail → Testing environment broken, cannot trust results

---

## Maintenance

Run validation:
- **Before any production deployment**
- **After any code changes to archetype engine**
- **After feature store updates**
- **Monthly** as sanity check

Keep this protocol version-controlled and updated as archetypes evolve.

---

## Appendix: Feature Domain Requirements

### Wyckoff Domain (10 features)
- spring_detected
- upthrust_detected
- volume_climax_last_3b
- wick_exhaustion_last_3b
- supply_test
- demand_test
- cause_effect_bars
- composite_operator_activity
- wyckoff_phase (A/B/C/D/E)
- trend_strength

### SMC Domain (10 features)
- is_bullish_ob
- is_bearish_ob
- fvg_bull
- fvg_bear
- bos_detected
- choch_detected
- premium_zone
- discount_zone
- liquidity_sweep_bull
- liquidity_sweep_bear

### Temporal Domain (10 features)
- fib_time_cluster
- wisdom_time_score
- temporal_confluence
- pattern_maturity
- cycle_alignment
- time_fractal_score
- session_volatility
- momentum_regime
- trend_age
- pattern_age

### Macro Domain (5 features)
- BTC.D
- USDT.D
- TOTAL
- TOTAL2
- OTHERS.D

### Funding Domain (5 features)
- funding_rate
- funding_Z
- oi_change_1h
- oi_change_4h
- oi_change_24h

**Total: 40 canonical features**
