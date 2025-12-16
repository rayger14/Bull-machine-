# ARCHETYPE VALIDATION BACKTEST RESULTS

**Date:** 2025-12-08
**Validation Protocol:** Full Backtest Suite (Steps 4-6)
**Archetypes Tested:** S4 (Funding Divergence), S1 (Liquidity Vacuum), S5 (Long Squeeze)

---

## EXECUTIVE SUMMARY

### Critical Finding: OI Data Gap

**The validation uncovered a critical data infrastructure issue that prevents proper archetype evaluation for 2022-2023:**

- **Funding Data:** ✅ PASS - 0.0% null across all periods
- **OI Data:** ❌ FAIL - 67.1% null overall
  - 2022: 100.0% null
  - 2023: 100.0% null
  - 2024: 1.4% null

**Impact:** Archetypes S1, S4, and S5 all depend on OI data for confluence scoring. Without OI data, they fall back to Tier-1 generic signals at high rates (70-100% fallback in 2022-2023).

---

## STEP 4: CHAOS WINDOW VALIDATION

**Objective:** Verify archetypes fire correctly during known market chaos events.

### Results Summary

| Archetype | Event | Period | Trades | PF | WR | Archetype Fires | Tier1 Fallback | Fallback % |
|-----------|-------|--------|--------|----|----|----------------|----------------|-----------|
| S4 | Terra Collapse | May 2022 | 15 | 0.26 | 26.7% | 2 | 13 | 87% |
| S1 | FTX Collapse | Nov 2022 | 10 | 0.14 | 30.0% | 0 | 10 | 100% |
| S5 | CPI Shock | Jun 2022 | 5 | 0.00 | 0.0% | 0 | 5 | 100% |

### Analysis

**Fusion Scores Present:** ✅ YES
Domain engines are active and generating fusion scores (0.15-0.77 range observed in logs).

**Archetype Specificity:** ❌ FAIL
- S4: Only 2/15 trades (13%) used archetype-specific logic
- S1: 0/10 trades (0%) used archetype logic
- S5: 0/5 trades (0%) used archetype logic

**Root Cause:**
Archetypes require OI data to meet confidence thresholds. Without OI:
- S4 `fusion_threshold=0.756` too high for funding-only scoring
- S1 liquidity vacuum pattern cannot detect without OI deltas
- S5 long squeeze requires OI + funding confluence

**Verdict:** ❌ FAIL STEP 4
Archetypes revert to Tier-1 fallback at unacceptable rates (>30% threshold).

---

## STEP 5: DATA COVERAGE VERIFICATION

### Funding Data Coverage

```
Overall null: 0.0%
2022: 0.0% null
2023: 0.0% null
2024: 0.0% null
```

**Status:** ✅ PASS (< 20% null threshold)

### OI Data Coverage

```
Overall null: 67.1%
2022: 100.0% null
2023: 100.0% null
2024: 1.4% null
```

**Status:** ❌ FAIL (>> 20% null threshold)

### OI Columns Available

1. `oi_change_24h` - 67.1% null
2. `oi_change_pct_24h` - 67.1% null
3. `oi_z` - 67.1% null

### Impact Assessment

**Why This Matters:**

1. **S4 (Funding Divergence):** Uses `oi_z` in fusion layer. Without OI, falls back to funding + price resilience only (~40% of signal strength).

2. **S1 (Liquidity Vacuum):** Core pattern requires `oi_change_pct_24h < -5%` (OI drawdown). Pattern cannot trigger without OI data.

3. **S5 (Long Squeeze):** Requires `funding_z > +1.5` + `oi_increasing` to detect overleveraged longs. OI provides critical confirmation.

**Verdict:** ❌ FAIL STEP 5
Historical validation impossible for 2022-2023. Only 2024 period usable.

---

## STEP 6: FULL PERIOD VALIDATION

### S4 (Funding Divergence) - 2024 OOS Only

| Period | Trades | PF | WR | MaxDD | Sharpe | Archetype | Tier1 | Fallback % |
|--------|--------|----|----|-------|--------|-----------|-------|-----------|
| 2024 OOS | 238 | 1.12 | 51.7% | 5.8% | 0.04 | unknown | unknown | unknown* |

*Note: Full trade-by-trade analysis required to determine fallback rate.

**Analysis:**
- **PF 1.12:** Below target (2.2+) but above break-even
- **WR 51.7%:** Acceptable for bear archetype
- **Sharpe 0.04:** Near-zero risk-adjusted return
- **Trades 238:** High volume (likely Tier-1 dominated)

**Interpretation:**
System generates trades in 2024 (OI data available) but performance suggests heavy Tier-1 fallback. True archetype performance cannot be isolated without trade-level breakdown.

### S1 (Liquidity Vacuum) - NOT TESTED

**Reason:** Requires OI drawdown data. Cannot validate on 2022-2023 (100% null OI).

### S5 (Long Squeeze) - NOT TESTED

**Reason:** Requires OI + funding confluence. Cannot validate on 2022-2023 (100% null OI).

**Verdict:** ⚠️ PARTIAL STEP 6
Only S4 2024 tested. Cannot assess train/test/OOS stability without 2020-2023 data.

---

## ROOT CAUSE ANALYSIS

### Why Archetypes Aren't Performing

1. **Data Infrastructure Gap**
   - OI data missing for 2020-2023 period
   - Archetypes calibrated expecting OI availability
   - Fallback logic engages when OI unavailable

2. **Threshold Calibration Mismatch**
   - S4 `fusion_threshold=0.756` optimized for OI+funding
   - Without OI, typical fusion scores: 0.15-0.60 (below threshold)
   - Result: Tier-1 fallback dominates

3. **Feature Engineering Incomplete**
   - Domain engines compute scores, but don't adapt to missing OI
   - No "degraded mode" thresholds for OI-absent scenarios
   - Binary outcome: Full archetype signal OR full fallback

### Why This Wasn't Caught Earlier

1. **87% Feature Coverage Metric Misleading**
   - Measured feature *column presence*, not data *completeness*
   - OI columns exist (3/3), but are 67% null

2. **Optimization on 2024 Data**
   - Recent calibrations used 2024 period (OI available)
   - Thresholds tuned to OI-rich environment
   - Historical degradation not tested

3. **Unit Tests vs Integration Tests**
   - Unit tests verified archetype logic fires with mock data
   - Integration tests didn't validate historical data quality
   - Gap between "can fire" and "does fire on real data"

---

## PASS/FAIL DETERMINATION

### Against Stated Criteria

**STEP 4-5 (Plumbing):**
- ❌ All chaos windows generate trades: YES (but Tier-1 dominated)
- ✅ Fusion scores present: YES (engines active)
- ❌ Fallback < 30%: FAIL (87-100% fallback observed)
- ✅ Funding null < 20%: PASS (0% null)
- ❌ OI null < 20%: FAIL (67% null)

**STEP 6 (Performance):**
- ❌ S4 Test PF ≥ 2.2: Cannot test (no 2023 data)
- ❌ S1 Test PF ≥ 1.8: Cannot test (requires OI)
- ❌ S5 Test PF ≥ 1.6: Cannot test (requires OI)
- ⚠️ S4 2024 PF 1.12: Below target but positive
- ❌ Trades ≥ 40 per archetype: Unknown (fallback contaminated)
- ❌ Overfit < 0.5: Cannot calculate (no train/test)

### Overall Verdict

**STATUS: ❌ VALIDATION FAILED**

**Blocker Issues:**
1. OI data infrastructure incomplete (2020-2023 missing)
2. Archetype thresholds not adapted for OI-absent scenarios
3. Historical validation impossible with current data quality

**What Works:**
1. ✅ Domain engines operational (fusion scores present)
2. ✅ Funding data complete and reliable
3. ✅ Threshold policy loading calibrated parameters
4. ✅ 2024 period generates trades (albeit fallback-heavy)

**What Doesn't Work:**
1. ❌ Archetypes revert to Tier-1 at 70-100% rates pre-2024
2. ❌ Cannot validate historical performance claims
3. ❌ Cannot assess train/test/OOS stability
4. ❌ Thresholds too strict for OI-absent operation

---

## RECOMMENDED PATH FORWARD

### Option A: Backfill OI Data (2-3 Days)

**Approach:** Complete OI pipeline for 2020-2023.

**Pros:**
- Enables true historical validation
- Unlocks archetype potential
- Validates optimization claims

**Cons:**
- Requires OKX API historical data pull
- May hit rate limits / data availability issues
- 2-3 day delay to production

**Verdict:** RECOMMENDED if production timeline allows.

### Option B: Adaptive Thresholds (1 Day)

**Approach:** Create "degraded mode" thresholds for OI-absent scenarios.

**Implementation:**
```python
# In threshold_policy.py
def get_archetype_threshold(name, param, has_oi=True):
    base = thresholds[name][param]
    if not has_oi and param == 'fusion_threshold':
        # Relax fusion requirement when OI unavailable
        return base * 0.65  # e.g., 0.756 → 0.491
    return base
```

**Pros:**
- Quick fix (1 day)
- Enables archetype firing on historical data
- Graceful degradation vs hard fallback

**Cons:**
- Performance uncertain without OI
- May increase false positives
- Still cannot validate "true" archetype performance

**Verdict:** ACCEPTABLE as temporary bridge to production.

### Option C: 2024-Only Deployment (Immediate)

**Approach:** Deploy with explicit constraint: "Archetypes active 2024+ only."

**Implementation:**
```python
# In backtest_knowledge_v2.py
if df.index[0].year < 2024:
    logger.warning("OI data unavailable pre-2024. Archetypes disabled.")
    config['use_archetypes'] = False
```

**Pros:**
- Honest about limitations
- Focuses on data-rich period
- Avoids misleading historical claims

**Cons:**
- Limited track record (11 months)
- Cannot prove multi-year stability
- Marketing challenge ("only works since 2024?")

**Verdict:** ACCEPTABLE if timeline critical.

---

## HONEST ASSESSMENT

### What This Validation Proves

1. **Infrastructure Works** ✅
   - Domain engines compute fusion scores
   - Threshold policy routes parameters correctly
   - Fallback logic prevents crashes

2. **Data Quality Matters** ❌
   - 87% feature coverage ≠ 87% data completeness
   - Archetypes optimized on 2024 don't transfer to 2022
   - Missing OI renders bear archetypes ineffective

3. **Integration Gaps Remain** ⚠️
   - Unit tests passed, integration failed
   - Need end-to-end validation with real historical data
   - "Can fire" ≠ "does fire in practice"

### What We Cannot Claim

1. **Historical Performance** ❌
   Cannot validate S4 PF 2.2+ on 2022 data (OI missing).

2. **Train/Test/OOS Stability** ❌
   Cannot assess overfit without multi-period comparison.

3. **Archetype Superiority** ❌
   2024 PF 1.12 does not prove archetype advantage (likely Tier-1 dominated).

### What We Can Claim

1. **2024 Performance** ✅
   S4 achieved PF 1.12, 51.7% WR on 2024 data (OI available).

2. **Robust Fallback** ✅
   System gracefully degrades to Tier-1 when archetypes cannot fire.

3. **Production-Ready Plumbing** ✅
   No crashes, clean logs, parameter routing verified.

---

## FILES DELIVERED

1. **results/validation/s4_terra_collapse.log**
   Chaos window test (May 2022 Terra collapse)

2. **results/validation/s1_ftx_collapse.log**
   Chaos window test (Nov 2022 FTX collapse)

3. **results/validation/s5_cpi_shock.log**
   Chaos window test (Jun 2022 CPI shock)

4. **results/validation/s4_oos_2024.log**
   Full-year 2024 OOS validation for S4

5. **results/validation/data_coverage_check.log**
   OI/funding data quality assessment

6. **bin/check_data_coverage.py**
   Automated data quality validation script

7. **bin/extract_validation_metrics.py**
   Automated metric extraction from logs

8. **THIS FILE: VALIDATION_BACKTEST_RESULTS.md**
   Comprehensive validation report

---

## NEXT ACTIONS

### Immediate (Today)

1. **Decide Path Forward:**
   - Option A (backfill OI): 2-3 days
   - Option B (adaptive thresholds): 1 day
   - Option C (2024-only): immediate

2. **Update Documentation:**
   - Add OI data requirement to README
   - Document degraded mode behavior
   - Set expectations for historical validation

### Short-Term (This Week)

1. **If Option A:** Complete OI backfill pipeline
2. **If Option B:** Implement adaptive threshold logic
3. **If Option C:** Add pre-2024 safety check

4. **Re-run Validation:**
   - Execute full train/test/OOS suite
   - Calculate true fallback rates
   - Assess overfit with proper data

### Medium-Term (Next Sprint)

1. **Integration Test Suite:**
   - Add data quality pre-checks to CI/CD
   - Validate feature completeness, not just presence
   - Test degraded mode paths explicitly

2. **Monitoring Dashboard:**
   - Track fallback rate in production
   - Alert on OI data staleness
   - Compare archetype vs Tier-1 performance live

---

## CONCLUSION

**The good news:** The archetype system architecture is sound. Domain engines work, threshold policy routes correctly, and the system degrades gracefully.

**The bad news:** We cannot validate the core performance claims (S4 PF 2.2+, S1 PF 1.8+) on historical data due to missing OI.

**The path forward:** Choose between:
1. Backfill OI (best validation, 2-3 day delay)
2. Adaptive thresholds (quick fix, uncertain performance)
3. 2024-only deployment (honest scope, limited track record)

**Recommendation:** Option A (backfill OI) if timeline permits. Option B (adaptive thresholds) as fallback for urgent deployment.

The system is production-ready *architecturally*, but needs either better data or adjusted expectations to validate *performance claims*.
