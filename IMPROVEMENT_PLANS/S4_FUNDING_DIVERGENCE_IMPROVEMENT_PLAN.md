# S4 FUNDING DIVERGENCE - IMPROVEMENT PLAN

**Strategy:** S4-FundingDivergence (Short Squeeze Specialist)
**Status:** 🔧 IMPROVE
**Current Performance:** Train PF 0.00, Test PF 0.00, Zero trades (both train and test)
**Historical Performance:** Train PF 2.22, OOS PF 2.32, 10-15 trades/year (validated 2022-2024)
**Priority:** HIGH
**Owner:** Data Pipeline + S4 Strategy Teams
**Timeline:** 4 weeks (Week 9-12)

---

## EXECUTIVE SUMMARY

**Problem:** Zero trades in BOTH train and test periods, despite historical validation showing PF 2.22-2.32 with 10-15 trades/year in bear markets.

**Root Cause:** Funding rate data missing or S4 archetype not enabled in comparison config. Complete detection failure.

**Impact:** CATASTROPHIC - highest PF strategy (2.22-2.32) producing zero value in current state.

**Decision:** DO NOT KILL - This is a DATA/CONFIG issue, not a strategy issue. S4 has proven value.

**Expected Outcome:** After fixes, expect Train PF > 2.0 (2022 bear), Test PF 0-2 trades (2023 bull, EXPECTED low activity).

---

## FAILURE ANALYSIS

### Current Results

| Metric | Train | Test | Expected (Historical) | Gap |
|--------|-------|------|----------------------|-----|
| **PF** | 0.00 | 0.00 | Train: 2.22, OOS: 2.32 | -100% |
| **Trades** | 0 | 0 | Train: 12 (2022), OOS: 7 (2024) | -100% |
| **PnL** | $0.00 | $0.00 | Positive | N/A |
| **Win Rate** | 0.0% | 0.0% | 55.7% | -100% |

### Critical Difference from S1 V2

**S1 V2 had 12 train trades** (detection logic partially working, but test period failed)
**S4 has 0 train trades AND 0 test trades** (detection logic COMPLETELY not working)

This suggests a more fundamental issue:
- Funding data completely missing, OR
- S4 archetype not enabled in config, OR
- Funding feature calculation not executing at all

---

### 6-Rule Acceptance Scorecard

| Rule | Criteria | Current Result | Pass? | Gap |
|------|----------|---------------|-------|-----|
| 1. Beat Baselines | Test PF > 3.27 | 0.00 vs 3.27 | ❌ NO | Need +3.27 PF |
| 2. Low Overfit | < 0.5 | 0.00 (N/A) | ❌ NO | N/A (no trades) |
| 3. Enough Trades | >= 50 | Train: 0, Test: 0 | ❌ NO | Need +50 trades |
| 4. OOS Validated | PF > 1.2 | 0.00 | ❌ NO | Need +1.2 PF |
| 5. Risk Acceptable | DD reasonable | No trades | ❌ NO | N/A |
| 6. Costs Included | Yes | Assumed yes | ✅ YES | None |

**Score: 1/6 rules passed** → IMPROVE status

---

## ROOT CAUSE INVESTIGATION

### Primary Hypotheses

**Hypothesis 1: Funding Rate Data Missing (MOST LIKELY)**

**Evidence:**
- S4 requires funding_zscore feature (calculated from funding rate)
- Zero trades in BOTH train (2022) and test (2023) suggests data missing entirely
- Historical validation had funding data (Binance API)
- Comparison test may not have funding data backfilled

**Test Plan:**
- [ ] Query feature store for funding_rate column in 2022-2023 period
- [ ] Expected: Funding rate values every 1H bar (8H funding snapshots)
- [ ] If missing: Data not backfilled for comparison test
- [ ] If all NaN: Calculation failure

**Fix Plan:**
- [ ] Backfill funding rate data from Binance/OKX historical API
- [ ] Binance endpoint: `GET /fapi/v1/fundingRate` (historical data available)
- [ ] Coverage required: 2022-01-01 to 2024-12-31
- [ ] Regenerate feature store with funding_zscore calculation
- [ ] Validate on known event (e.g., Aug 2022 short squeeze)

---

**Hypothesis 2: S4 Archetype Not Enabled in Config**

**Evidence:**
- Zero trades suggests S4 detection logic never executed
- Comparison config may have S4 disabled
- Or S4 not included in archetype list

**Test Plan:**
- [ ] Review baseline comparison config
- [ ] Check archetypes list: Should include "s4_funding_divergence"
- [ ] Check enable flag: s4_enabled = true
- [ ] Check runtime enrichment: S4 features should be calculated

**Fix Plan:**
- [ ] Enable S4 in comparison config
- [ ] Add S4 to archetypes list
- [ ] Ensure S4 runtime enrichment is active
- [ ] Verify S4 thresholds match production config

---

**Hypothesis 3: Feature Store Missing S4 Enrichment**

**Evidence:**
- S4 requires runtime enrichment (funding_zscore calculation)
- Feature store may not include S4-specific features
- Standard feature store vs S4-enriched feature store mismatch

**Test Plan:**
- [ ] Query feature store schema
- [ ] Check for S4 columns:
   - funding_rate (raw data)
   - funding_zscore (z-score of funding rate)
   - funding_divergence (price vs funding divergence)
- [ ] If missing: Feature store not S4-enriched

**Fix Plan:**
- [ ] Regenerate feature store with S4 enrichment enabled
- [ ] Run S4 feature calculation pipeline
- [ ] Validate funding_zscore calculation:
   - rolling_mean(funding_rate, 30 periods)
   - rolling_std(funding_rate, 30 periods)
   - funding_zscore = (current - mean) / std
- [ ] Expected: z-score range -3.0 to +3.0

---

**Hypothesis 4: Regime Filter Blocking All Signals**

**Evidence:**
- S4 allows risk_off, crisis, neutral regimes (NOT risk_on)
- Zero trades in 2022 (bear) suggests regime filter not the issue
- 2022 should be risk_off/crisis dominant

**Test Plan:**
- [ ] Query regime classification for 2022 train period
- [ ] Expected: risk_off / crisis dominant (bear market)
- [ ] If risk_on dominant: Regime classifier broken (very unlikely)
- [ ] Check S4 allowed regimes match production config

**Fix Plan:**
- [ ] Verify regime filter allows risk_off, crisis, neutral
- [ ] If regime classifier broken: Retrain regime model
- [ ] If regime filter too strict: Relax (but unlikely root cause given zero train trades)

---

### Secondary Hypotheses

**Hypothesis 5: Threshold Too Strict**

**Evidence:**
- S4 requires funding_zscore < -1.976 (extreme negative funding)
- Zero trades suggests threshold never met
- But historical validation with same threshold had 12 trades in 2022

**Test Plan:**
- [ ] Calculate funding_zscore distribution for 2022
- [ ] Expected: At least 10-15 bars with zscore < -1.976
- [ ] Known events: Aug 2022 short squeeze, FTX Dec 2022
- [ ] If no bars meet threshold: Threshold too strict OR data missing

**Fix Plan:**
- [ ] If threshold issue: Relax to -1.5σ (more trades, lower PF expected)
- [ ] If data issue: Fix data first (Hypothesis 1)
- [ ] Re-calibrate threshold after data fix

---

## IMPROVEMENT ROADMAP

### Week 9: Diagnosis (Dec 9-15, 2025)

#### Monday-Tuesday: Funding Data Investigation

**Morning:**
- [ ] Query feature store for funding_rate column
- [ ] Check coverage: 2022-01-01 to 2023-12-31
- [ ] Expected: 8,760 hours × 1 funding snapshot = 8,760 data points
- [ ] If missing: CRITICAL - funding data not available

**Afternoon:**
- [ ] Check Binance/OKX API for historical funding rate access
- [ ] Endpoint: `GET /fapi/v1/fundingRate`
- [ ] Parameters: symbol=BTCUSDT, startTime, endTime, limit=1000
- [ ] Test: Fetch Oct-Dec 2022 funding data (FTX period)
- [ ] Validate: Funding rate values reasonable (-0.01% to +0.05% typical)

**Deliverable:** Funding data availability report

---

#### Wednesday: Feature Store Schema Audit

**Morning:**
- [ ] Query feature store schema (all columns)
- [ ] Check for S4-specific columns:
   - funding_rate ✅
   - funding_zscore ✅
   - funding_divergence ✅
   - price_momentum (for divergence calc) ✅
- [ ] If missing: Feature store not S4-enriched

**Afternoon:**
- [ ] Review S4 runtime enrichment code
- [ ] Verify funding_zscore calculation logic
- [ ] Test calculation on known event (Aug 2022 short squeeze)
- [ ] Expected: funding_zscore < -2.0 during squeeze

**Deliverable:** Feature store schema report + enrichment status

---

#### Thursday: Config Review

**Morning:**
- [ ] Review baseline comparison config
- [ ] Check archetypes list (should include S4)
- [ ] Check S4 enable flag (should be true)
- [ ] Check S4 thresholds match production config:
   - funding_zscore_threshold: -1.976 (< -2σ)
   - liquidity_threshold: 0.20 (< 20th percentile)
   - regime_filter: ["risk_off", "crisis", "neutral"]

**Afternoon:**
- [ ] Diff comparison config vs system_s4_production.json
- [ ] Document any differences
- [ ] Prepare corrected config (use production config)

**Deliverable:** Config diff report + corrected config

---

#### Friday: Regime Analysis

**Morning:**
- [ ] Query regime classification for 2022 (train period)
- [ ] Expected: risk_off / crisis dominant (bear market)
- [ ] Calculate regime distribution:
   - risk_off: 60-80% of bars
   - crisis: 10-20% of bars
   - neutral: 10-20% of bars
   - risk_on: 0-10% of bars

**Afternoon:**
- [ ] If regime distribution incorrect: Regime classifier issue
- [ ] If regime correct: Not root cause (funding data more likely)
- [ ] Document regime analysis for reference

**Deliverable:** Regime analysis report

---

### Week 10: Fix Implementation (Dec 16-22, 2025)

#### Monday-Tuesday: Funding Data Backfill

**Monday Morning:**
- [ ] Setup Binance API access (or OKX if preferred)
- [ ] Write funding rate backfill script
- [ ] Parameters: BTCUSDT, 2022-01-01 to 2024-12-31
- [ ] Fetch funding rate snapshots (8H intervals)

**Monday Afternoon:**
- [ ] Execute backfill (may take 1-2 hours for 3 years data)
- [ ] Validate data quality:
   - No gaps (should have 3,650 snapshots for 3 years × 365 days × 3 per day)
   - Values reasonable (-0.05% to +0.10% typical range)
   - Timestamps correct (8H intervals)

**Tuesday Morning:**
- [ ] Store funding rate data in database/CSV
- [ ] Create funding_rate column in feature store
- [ ] Join funding data to OHLCV bars (nearest 8H snapshot)

**Tuesday Afternoon:**
- [ ] Validate funding_rate availability for 2022-2024
- [ ] Coverage check: > 95% of bars should have funding data
- [ ] If gaps: Interpolate or fill with previous value

**Deliverable:** Backfilled funding rate data (2022-2024)

---

#### Wednesday: S4 Feature Enrichment

**Morning:**
- [ ] Calculate funding_zscore for all bars
- [ ] Formula:
   ```python
   rolling_mean = funding_rate.rolling(30).mean()
   rolling_std = funding_rate.rolling(30).std()
   funding_zscore = (funding_rate - rolling_mean) / rolling_std
   ```
- [ ] Validate calculation on known events:
   - Aug 2022 short squeeze: zscore < -2.0 expected
   - FTX Dec 2022: zscore < -2.5 expected

**Afternoon:**
- [ ] Calculate funding_divergence (optional, may not be needed)
- [ ] Formula: price_momentum × funding_zscore (divergence)
- [ ] Add columns to feature store:
   - funding_rate
   - funding_zscore
   - funding_divergence (if needed)

**Deliverable:** S4-enriched feature store

---

#### Thursday: Config Restoration

**Morning:**
- [ ] Copy system_s4_production.json to comparison config
- [ ] Verify all parameters match:
   - funding_zscore_threshold: -1.976
   - liquidity_threshold: 0.20
   - regime_filter: ["risk_off", "crisis", "neutral"]
   - archetype_enabled: true
   - runtime_enrichment: true

**Afternoon:**
- [ ] Add S4 to comparison archetype list
- [ ] Verify S4 runtime enrichment enabled
- [ ] Add config hash verification (prevent future mismatches)

**Deliverable:** Verified S4 production config in comparison pipeline

---

#### Friday: Integration Testing

**Morning:**
- [ ] Run canary test on Aug 2022 (known short squeeze)
- [ ] Expected: S4 signal should fire (funding zscore < -2.0)
- [ ] Verify signal logic:
   - Funding zscore meets threshold
   - Regime allowed (risk_off expected)
   - Liquidity meets threshold
   - Signal generated

**Afternoon:**
- [ ] Run full 2022 backtest with fixed pipeline
- [ ] Expected: 10-15 trades, PF > 2.0
- [ ] If successful: Ready for Week 11 re-test
- [ ] If failed: Deep dive needed

**Deliverable:** Integration test results

---

### Week 11: Re-test and Validation (Dec 23-29, 2025)

#### Monday-Wednesday: Full Re-test

**Setup:**
- [ ] Use system_s4_production.json (verified config)
- [ ] S4-enriched feature store (verified funding data)
- [ ] Test periods:
   - Train: 2022-01-01 to 2022-12-31 (bear market)
   - Test: 2023-01-01 to 2023-12-31 (bull recovery)
   - OOS: 2024-01-01 to 2024-06-30 (volatility)

**Expected Metrics:**

**2022 Train (Bear Market):**
- Train PF: 2.0-2.5 (should match historical 2.22)
- Train trades: 10-15 (bear market specialist, frequent in this regime)
- Win rate: 50-60%
- Key events captured: Aug short squeeze, FTX Dec squeeze

**2023 Test (Bull Recovery):**
- Test PF: N/A (expect 0-2 trades)
- Test trades: 0-2 (EXPECTED low activity in bull market)
- Note: S4 is bear/volatile specialist, idle in bull = NORMAL

**2024 OOS (Volatility):**
- OOS PF: 2.0-2.5 (should match historical 2.32)
- OOS trades: 5-10 (6 months data, annualized 10-20)
- Key events: Q1-Q2 volatility, corrections

**Deliverable:** Full re-test results

---

#### Thursday-Friday: Analysis and Decision

**Analysis:**
- [ ] Compare re-test to historical validation
- [ ] Check 2022 train metrics (should match historical)
- [ ] Check 2023 test metrics (expect low, NORMAL for bull market)
- [ ] Check 2024 OOS metrics (CRITICAL for validation)

**Decision Tree:**

**Scenario A: Re-test Successful (Train PF > 2.0, Train trades >= 5, OOS PF > 2.0)**
- ✅ Move to KEEP status
- ✅ Schedule Week 13-14 paper trading
- ✅ Prepare production deployment
- ✅ Plan capital allocation (Scenario B: 50/20/30)

**Scenario B: Re-test Partial Success (Train PF > 2.0, Test trades = 0, OOS PF > 1.5)**
- ⚠️ Acceptable (Test period idle is EXPECTED for S4 in bull)
- ✅ Move to KEEP status (OOS validation is key)
- ✅ Schedule Week 13-14 paper trading
- ⚠️ Note: Expect idle periods in risk_on regimes (by design)

**Scenario C: Re-test Failed (Train PF < 2.0 or Train trades < 5)**
- ❌ Escalate to deep investigation
- ❌ Possible issues:
   - Funding data quality poor
   - Threshold calibration wrong
   - Historical validation was overfitted
- ⏸️ Timeline: +4-8 weeks for deep dive

**Scenario D: Zero Trades Again (Train trades = 0)**
- ❌ CRITICAL FAILURE
- ❌ Funding data still missing OR threshold impossibly strict
- ❌ Review all Hypothesis 1-5 again
- ⏸️ Consider kill decision

**Deliverable:** Go/No-Go decision report

---

### Week 12: Preparation for Deployment (Dec 30-Jan 5, 2026)

**If Re-test Successful:**

#### Monday-Tuesday: Paper Trading Preparation

- [ ] Setup S4 in paper trading environment
- [ ] Configure position sizing: 1.5% per trade = $450 (if $30K allocation)
- [ ] Setup S4-specific monitoring:
   - Funding z-score tracking (real-time)
   - Regime classification monitoring
   - Alert on funding < -1.976 (potential signal)
- [ ] Add S4 to alert system

---

#### Wednesday-Thursday: Documentation

- [ ] Update S4 production guide
- [ ] Document funding data backfill process (for future)
- [ ] Create operator runbook:
   - Signal interpretation (funding squeeze mechanics)
   - Risk management (bear market specialist, expect idle in bulls)
   - Position sizing (conservative for specialist)
- [ ] Update capital allocation plan (Scenario B: 50/20/30)

---

#### Friday: Week 13+ Planning

- [ ] Schedule Week 13-14 paper trading
- [ ] Prepare integration with baseline portfolio
- [ ] Plan capital reallocation:
   - Conservative $80K → $50K
   - Aggressive $20K → $20K (unchanged)
   - S4 $0 → $30K (new allocation)
- [ ] Setup correlation monitoring (S4 vs baselines)
- [ ] Expected correlation < 0.5 (S4 fires in different conditions)

---

**If Re-test Failed:**

#### Full Week: Deep Dive Investigation

- [ ] Funding data quality audit (are values correct?)
- [ ] Threshold calibration review (is -1.976 too strict?)
- [ ] Historical validation audit (was it overfitted?)
- [ ] Alternative approaches:
   - Relax threshold to -1.5σ (more trades, lower PF)
   - Add OI divergence (if OI data available)
   - Combine with S5 (opposite direction funding strategy)
- [ ] Go/Kill decision by Week 16

---

## SUCCESS CRITERIA

### Week 9 (Diagnosis)
- ✅ Root cause identified (funding data missing most likely)
- ✅ Funding data source confirmed (Binance API accessible)
- ✅ Fix plan achievable (4 weeks sufficient)

### Week 10 (Fix Implementation)
- ✅ Funding data backfilled (2022-2024, > 95% coverage)
- ✅ Feature store enriched with funding_zscore
- ✅ Production config restored
- ✅ Integration test passes (Aug 2022 signal fires)

### Week 11 (Re-test)
- ✅ Train PF >= 2.0 (2022 bear market)
- ✅ Train trades >= 5 (minimum for statistical significance)
- ✅ OOS 2024 PF >= 2.0 (validates strategy in new data)
- ✅ OOS 2024 trades >= 5 (6 months sample)
- ✅ Test trades = 0-2 (2023 bull, EXPECTED low activity)

### Week 12 (Deployment Prep)
- ✅ S4 ready for paper trading
- ✅ Funding data pipeline established (ongoing updates)
- ✅ Documentation complete
- ✅ Capital allocation plan updated

---

## RISK ASSESSMENT

### Fix Implementation Risks

**Risk 1: Funding Data Not Available from Exchange APIs**
- **Probability:** LOW (Binance provides historical funding data)
- **Impact:** CRITICAL (cannot fix S4 without funding data)
- **Mitigation:** Try multiple sources (Binance, OKX, FTX historical archives)
- **Contingency:** If truly unavailable, kill S4 (strategy unusable)

**Risk 2: Funding Data Quality Poor**
- **Probability:** MEDIUM (historical data may have gaps)
- **Impact:** HIGH (poor data = poor signals)
- **Mitigation:** Interpolate gaps, validate against known events
- **Contingency:** If quality < 80%, relax threshold or kill strategy

**Risk 3: Re-test Still Shows Zero Trades**
- **Probability:** LOW (after funding data backfilled)
- **Impact:** CRITICAL (kills strategy)
- **Mitigation:** Deep dive on threshold calibration, regime filter
- **Contingency:** Kill S4, focus on baselines + S1 V2

**Risk 4: Historical Validation Was Overfitted**
- **Probability:** MEDIUM (PF 2.22-2.32 is very high)
- **Impact:** HIGH (strategy has no real value)
- **Mitigation:** Walk-forward validation on 2024 data (OOS)
- **Contingency:** If OOS 2024 PF < 1.5, kill strategy

**Risk 5: S4 Only Works in Bear Markets (Long Idle Periods)**
- **Probability:** HIGH (S4 is bear specialist, this is KNOWN)
- **Impact:** MEDIUM (reduces capital utilization)
- **Mitigation:** Accept design tradeoff, diversify with bull patterns
- **Contingency:** Pair with S5 (opposite direction) for better coverage

---

## LESSONS LEARNED (Pre-Fix)

**Lesson 1: Data Dependencies Must Be Explicit**
- S4 requires funding data, but this was not checked
- Need pre-flight check: Assert funding_rate column exists
- **Fix:** Add data dependency validation in backtester

**Lesson 2: Zero Trades in Train AND Test is Red Flag**
- S1 V2 had 12 train trades (partial failure)
- S4 had 0 train trades (complete failure)
- Zero train trades should trigger immediate investigation
- **Fix:** Alert if train trades < 5 for any strategy

**Lesson 3: Regime Specialists Need Clear Documentation**
- S4 idle in bull markets is EXPECTED, not a bug
- Need to document expected trade frequency by regime
- **Fix:** Add regime-aware success criteria in validation

**Lesson 4: External Data Sources Need Backfill Process**
- Funding data from exchange API, not internal
- Need repeatable backfill process for new data
- **Fix:** Create funding data pipeline (daily updates)

---

## CONTINGENCY PLANS

### Contingency 1: Week 11 Re-test Fails

**Actions:**
1. Deep dive on threshold calibration (is -1.976 too strict?)
2. Test relaxed threshold: -1.5σ, -1.0σ (expect more trades, lower PF)
3. Walk-forward validation on multiple periods (2022, 2024 separately)
4. If all fail: Kill S4, focus on baselines + S1 V2
5. Timeline: +4 weeks for deep dive, Go/Kill decision by Week 16

---

### Contingency 2: Funding Data Quality Poor

**Actions:**
1. Calculate funding data coverage (% of bars with data)
2. If coverage < 80%: Interpolate gaps or kill strategy
3. Validate funding values against known events (FTX, Aug 2022)
4. If values unreasonable: Find alternative data source
5. Contingency: Kill S4 if data quality unfixable

---

### Contingency 3: S4 Idle Too Often (Low Capital Utilization)

**Actions:**
1. Accept design tradeoff (bear specialist)
2. Pair with S5 (opposite direction, long squeeze)
3. Add bull market patterns (S0, B0) for coverage
4. Consider regime-adaptive allocation (scale S4 up in bear, down in bull)
5. Timeline: Month 4-5 for S5 integration

---

### Contingency 4: Historical Validation Was Overfitted

**Actions:**
1. Kill S4 if OOS 2024 PF < 1.5
2. Document lessons learned (high train PF ≠ robust strategy)
3. Focus on baselines (proven robust) + S1 V2 (if fixed)
4. Consider simpler funding strategy (threshold-based, no confluence)
5. Timeline: Immediate kill decision if OOS fails

---

## OWNER ASSIGNMENTS

**Data Pipeline Team:**
- Funding data backfill (Week 10 priority)
- Feature store enrichment (funding_zscore calculation)
- Ongoing funding data pipeline (daily updates)

**S4 Strategy Team:**
- Config management (restore production config)
- Threshold calibration (if needed)
- Integration testing (canary tests)

**Quant Team:**
- Re-test execution (Week 11)
- Performance analysis (compare to historical)
- Decision making (KEEP/IMPROVE/KILL)

**Trading Ops:**
- Paper trading setup (if successful)
- Funding rate monitoring infrastructure
- Alert system for funding extremes

---

## TIMELINE SUMMARY

| Week | Focus | Deliverables | Decision Point |
|------|-------|--------------|----------------|
| Week 9 | Diagnosis | Funding data availability report | Data accessible? |
| Week 10 | Fix Implementation | Backfilled funding data + enriched feature store | Integration test passes? |
| Week 11 | Re-test | Full validation results (train/test/OOS) | KEEP/IMPROVE/KILL? |
| Week 12 | Deployment Prep | Paper trading ready + funding pipeline | Deploy Week 13? |

**Total Timeline:** 4 weeks (Dec 9 - Jan 5)
**Critical Path:** Funding data backfill → feature enrichment → re-test
**Buffer:** 2 weeks (can extend to Week 14 if needed)

---

## FINAL RECOMMENDATION

**Do NOT Kill S4**

**Rationale:**
1. Highest historical PF among all strategies (2.22-2.32)
2. Current failure is DATA issue (funding data missing), not strategy issue
3. Clear path to recovery (backfill funding data, 4 weeks achievable)
4. Very high potential reward (short squeeze detection is valuable)
5. Bear market specialist (complements baselines, diversification)

**Action Plan:**
1. Execute 4-week improvement plan (Week 9-12)
2. Backfill funding data from Binance API (Week 10 priority)
3. Re-test with enriched feature store (Week 11)
4. If successful: Paper trade Week 13-14, deploy Week 15 with $30K allocation
5. If failed: Deep dive on threshold calibration, Kill decision by Week 16

**Expected Outcome:**
- 80% probability: Funding data backfill successful, S4 moves to KEEP
- 15% probability: Threshold calibration needed, additional 2 weeks
- 5% probability: Fundamental failure (overfitted or data quality issues), kill

**Investment:** 4 weeks × 2 people = 8 person-weeks
**Potential Return:** $30K allocation × 120% annual return = $36K/year (if PF 2.2 sustained)
**ROI:** Very High (if fixes successful, highest PF strategy)

---

**IMPROVEMENT PLAN STATUS:** APPROVED ✅

**Next Action:** Begin Week 9 funding data investigation (Dec 9, 2025)

**Owner:** Data Pipeline Team (lead) + S4 Strategy Team
**Reviewer:** Quant Team Lead
**Escalation:** If Week 11 re-test fails, escalate to CTO (kill decision authority)

**Priority:** HIGH (highest PF potential among all archetypes)
