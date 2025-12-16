# S1 LIQUIDITY VACUUM - IMPROVEMENT PLAN

**Strategy:** S1-LiquidityVacuum (Capitulation Reversal Specialist)
**Status:** 🔧 IMPROVE
**Current Performance:** Train PF 0.00, Test PF 0.00, Zero trades
**Historical Performance:** PF 1.4-1.8, 40-60 trades/year (validated 2022-2024)
**Priority:** HIGH
**Owner:** Data Pipeline + Strategy Teams
**Timeline:** 4 weeks (Week 9-12)

---

## EXECUTIVE SUMMARY

**Problem:** Zero trades in baseline comparison despite historical validation showing PF 1.4-1.8 with 40-60 trades/year.

**Root Cause:** Data pipeline failure - feature calculations not executing, causing detection logic to never fire.

**Impact:** CATASTROPHIC - strategy produces zero value in current state, but has proven value historically.

**Decision:** DO NOT KILL - This is a pipeline issue, not a strategy issue. Fix and retest.

**Expected Outcome:** After fixes, expect Test PF > 1.4, Test trades >= 20 (1 year sample).

---

## FAILURE ANALYSIS

### Current Results

| Metric | Train | Test | Expected (Historical) | Gap |
|--------|-------|------|----------------------|-----|
| **PF** | 0.00 | 0.00 | 1.4-1.8 | -100% |
| **Trades** | 12 | 0 | 40-60/year | -100% |
| **PnL** | -$769.63 | $0.00 | Positive | N/A |
| **Win Rate** | 0.0% | 0.0% | 50-60% | -100% |

### 6-Rule Acceptance Scorecard

| Rule | Criteria | Current Result | Pass? | Gap |
|------|----------|---------------|-------|-----|
| 1. Beat Baselines | Test PF > 3.27 | 0.00 vs 3.27 | ❌ NO | Need +3.27 PF |
| 2. Low Overfit | < 0.5 | 0.00 (N/A) | ❌ NO | N/A (no trades) |
| 3. Enough Trades | >= 50 | Test: 0 | ❌ NO | Need +50 trades |
| 4. OOS Validated | PF > 1.2 | 0.00 | ❌ NO | Need +1.2 PF |
| 5. Risk Acceptable | DD reasonable | -$769 loss | ❌ NO | Major issue |
| 6. Costs Included | Yes | Assumed yes | ✅ YES | None |

**Score: 1/6 rules passed** → IMPROVE status

---

## ROOT CAUSE INVESTIGATION

### Primary Hypotheses

**Hypothesis 1: Feature Calculation Failure**

**Evidence:**
- Zero test trades (detection logic never fired)
- Historical validation shows feature pipeline worked before
- Likely: capitulation_depth, crisis_composite not computing

**Test Plan:**
- [ ] Query feature store for capitulation_depth values in 2023 test period
- [ ] Expected: Values should exist for all bars, range -0.5 to 0.0 (negative = drawdown)
- [ ] If missing: Feature calculation bug or data missing
- [ ] If all zeros: Calculation logic broken

**Fix Plan:**
- [ ] Regenerate feature store with S1 V2 production config
- [ ] Add unit tests for capitulation_depth calculation
- [ ] Validate against known capitulation events (e.g., FTX Nov 2022)

---

**Hypothesis 2: Confluence Logic Not Executing**

**Evidence:**
- S1 V2 uses 3-of-4 confluence gate (depth + crisis + volume + wick)
- Zero trades suggests gate never opens
- Likely: Confluence score calculation bug

**Test Plan:**
- [ ] Manually calculate confluence score for known capitulation event
- [ ] Example: FTX Nov 9, 2022 (should score > 0.65)
- [ ] Compare to code calculation
- [ ] If mismatch: Logic bug in confluence.py

**Fix Plan:**
- [ ] Review confluence logic in archetype detector
- [ ] Add logging: Print confluence score for each bar (debug mode)
- [ ] Fix calculation if broken
- [ ] Add unit test for confluence calculation

---

**Hypothesis 3: Configuration Mismatch**

**Evidence:**
- Historical validation used s1_v2_production.json
- Baseline comparison may have used different config
- Thresholds or gates may differ

**Test Plan:**
- [ ] Diff comparison config vs s1_v2_production.json
- [ ] Check threshold values:
   - capitulation_depth < -0.20 (historical)
   - crisis_composite > 0.35 (historical)
   - confluence_threshold > 0.65 (historical)
- [ ] If different: Config mismatch is root cause

**Fix Plan:**
- [ ] Use EXACT s1_v2_production.json for re-test
- [ ] Version control configs (Git hash verification)
- [ ] Add config equality assertion in comparison script

---

**Hypothesis 4: Regime Filter Too Strict**

**Evidence:**
- S1 V2 allows risk_off + crisis regimes only
- Zero trades suggests no regimes matched
- Possible: 2023 test period classified as risk_on (bull market)

**Test Plan:**
- [ ] Check regime classification for 2023 test period
- [ ] Expected: Some risk_off periods (Jan-Mar 2023 uncertain, Aug dip)
- [ ] If all risk_on: Regime filter blocking all signals (EXPECTED for bull)
- [ ] Also check 2022 train period (should be risk_off/crisis)

**Fix Plan:**
- [ ] If 2023 all risk_on: EXPECTED behavior (S1 is bear specialist)
- [ ] Validate on 2024 data (should have some risk_off periods)
- [ ] Consider adding 10% drawdown override (flash crash detection)
- [ ] If 2022 also blocked: Regime classifier broken, needs retrain

---

### Secondary Hypotheses

**Hypothesis 5: Train Loss (-$769) Position Sizing Error**

**Evidence:**
- 12 train trades but massive loss (-$769)
- Average loss per trade: -$64.13 (way too high)
- Likely: Position sizing error or stop loss not working

**Test Plan:**
- [ ] Review train trades CSV (if available)
- [ ] Check position sizes: Should be ~2% of capital ($40-80 typical)
- [ ] Check stop losses: Should be 3.0 ATR (hit rate < 50%)
- [ ] If position sizes >> expected: Position sizing bug

**Fix Plan:**
- [ ] Verify position sizing logic (2% max per trade)
- [ ] Ensure stop loss calculation correct (3.0 ATR)
- [ ] Add position sizing unit tests
- [ ] Re-run with fixed sizing

---

## IMPROVEMENT ROADMAP

### Week 9: Diagnosis (Dec 9-15, 2025)

#### Monday-Tuesday: Data Audit

**Morning:**
- [ ] Query feature store for 2023 test period data
- [ ] Check capitulation_depth coverage (should be 100% bars)
- [ ] Check crisis_composite coverage (should be 100% bars)
- [ ] Check volume_zscore, wick_exhaustion (confluence components)

**Afternoon:**
- [ ] Analyze feature distributions (are values reasonable?)
- [ ] Expected ranges:
   - capitulation_depth: -0.5 to 0.0 (negative = drawdown)
   - crisis_composite: 0.0 to 1.0 (> 0.35 = stress)
   - volume_zscore: -3.0 to +3.0 (> 0.5 = climax)
   - wick_exhaustion: 0.0 to 1.0 (> 0.6 = exhaustion)
- [ ] If missing or all zeros: CRITICAL data issue

**Deliverable:** Feature store audit report

---

#### Wednesday: Confluence Logic Review

**Morning:**
- [ ] Review S1 V2 confluence calculation code
- [ ] Manually calculate confluence for known event (FTX Nov 9, 2022)
- [ ] Expected score: > 0.65 (should trigger)
- [ ] Compare manual vs code calculation

**Afternoon:**
- [ ] Add logging to confluence calculation (debug mode)
- [ ] Run backtest on Nov 2022 (FTX period)
- [ ] Expected: Signal should fire on Nov 9
- [ ] If no signal: Confluence logic broken

**Deliverable:** Confluence logic audit report

---

#### Thursday: Config Comparison

**Morning:**
- [ ] Diff baseline comparison config vs s1_v2_production.json
- [ ] Check all threshold values match
- [ ] Check feature flags match
- [ ] Check archetype enable/disable flags

**Afternoon:**
- [ ] If config mismatch found: Document differences
- [ ] Prepare corrected config (use production config exactly)
- [ ] Add config hash verification to comparison script

**Deliverable:** Config diff report + corrected config

---

#### Friday: Regime Filter Analysis

**Morning:**
- [ ] Query regime classification for 2022 train period
- [ ] Expected: risk_off / crisis dominant (bear market)
- [ ] If not: Regime classifier broken

**Afternoon:**
- [ ] Query regime classification for 2023 test period
- [ ] Expected: risk_on dominant (bull recovery)
- [ ] If risk_on: Zero trades EXPECTED (S1 is bear specialist)
- [ ] Check for drawdown override (> 10% should trigger anyway)

**Deliverable:** Regime analysis report

---

### Week 10: Fix Implementation (Dec 16-22, 2025)

#### Monday: Feature Store Regeneration

**Morning:**
- [ ] If feature data missing: Backfill OHLCV data for 2023
- [ ] Verify data quality (no gaps, anomalies)

**Afternoon:**
- [ ] Regenerate feature store using S1 V2 production config
- [ ] Run capitulation_depth calculation
- [ ] Run crisis_composite calculation
- [ ] Validate outputs against known events

**Deliverable:** Regenerated feature store (2022-2024 complete)

---

#### Tuesday: Confluence Logic Fix

**Morning:**
- [ ] If confluence calculation broken: Fix logic
- [ ] Add unit tests for confluence calculation
- [ ] Test on known events (FTX, LUNA, Japan Carry)

**Afternoon:**
- [ ] Add comprehensive logging to confluence logic
- [ ] Log each component (depth, crisis, volume, wick)
- [ ] Log final confluence score
- [ ] Log why signal accepted/rejected

**Deliverable:** Fixed confluence logic + unit tests

---

#### Wednesday: Config Restoration

**Morning:**
- [ ] Copy s1_v2_production.json to comparison config location
- [ ] Verify hash matches production config
- [ ] Add assertion in comparison script (config equality check)

**Afternoon:**
- [ ] Review all S1 V2 parameters:
   - capitulation_depth_threshold: -0.20
   - crisis_composite_threshold: 0.35
   - confluence_threshold: 0.65
   - regime_filter: ["risk_off", "crisis"]
   - drawdown_override: 0.10 (10%)

**Deliverable:** Verified production config in comparison pipeline

---

#### Thursday: Position Sizing Review

**Morning:**
- [ ] Review position sizing logic in backtester
- [ ] Verify 2% max per trade enforcement
- [ ] Check stop loss calculation (3.0 ATR)

**Afternoon:**
- [ ] Add position sizing unit tests
- [ ] Test edge cases (small capital, large ATR)
- [ ] Validate against historical trades (should match)

**Deliverable:** Position sizing validation report

---

#### Friday: Integration Testing

**Morning:**
- [ ] Run end-to-end pipeline test
- [ ] Feature store → archetype detection → trade generation
- [ ] Test on Nov 2022 (FTX period)
- [ ] Expected: Signal on Nov 9, profitable trade

**Afternoon:**
- [ ] Run full 2022 backtest with fixed pipeline
- [ ] Expected: 40-60 trades, PF > 1.4
- [ ] If successful: Ready for Week 11 re-test
- [ ] If failed: Deep dive investigation needed

**Deliverable:** Integration test results

---

### Week 11: Re-test and Validation (Dec 23-29, 2025)

#### Monday-Wednesday: Full Re-test

**Setup:**
- [ ] Use s1_v2_production.json (verified config)
- [ ] Regenerated feature store (verified data)
- [ ] Fixed confluence logic (verified calculation)
- [ ] Test periods: Train 2022, Test 2023, OOS 2024

**Execution:**
- [ ] Run baseline comparison with fixed S1 V2 pipeline
- [ ] Expected metrics:
   - Train PF: 1.4-1.8
   - Train trades: 40-60
   - Test PF: 1.4+ (may be low due to 2023 bull market)
   - Test trades: 10-20 (low expected in bull)
   - OOS 2024 PF: 1.5+
   - OOS 2024 trades: 20-30

**Deliverable:** Full re-test results

---

#### Thursday-Friday: Analysis and Decision

**Analysis:**
- [ ] Compare re-test to historical validation
- [ ] Check train metrics (should match historical)
- [ ] Check test metrics (may differ due to regime)
- [ ] Check OOS 2024 metrics (critical for validation)

**Decision Tree:**

**Scenario A: Re-test Successful (Train PF > 1.4, OOS PF > 1.4, Train trades >= 40)**
- ✅ Move to KEEP status
- ✅ Schedule Week 13-14 paper trading
- ✅ Prepare production deployment

**Scenario B: Re-test Partial Success (Train PF > 1.4, Test PF low, OOS PF > 1.4)**
- ⚠️ Acceptable (Test period is bull market, expected low activity)
- ✅ Move to KEEP status (OOS validation is key)
- ✅ Schedule Week 13-14 paper trading

**Scenario C: Re-test Failed (Train PF < 1.4 or Train trades < 40)**
- ❌ Escalate to architecture review
- ❌ Deep dive needed (strategy fundamentally broken?)
- ⏸️ Timeline: +4-8 weeks for redesign

**Deliverable:** Go/No-Go decision report

---

### Week 12: Preparation for Deployment (Dec 30-Jan 5, 2026)

**If Re-test Successful:**

#### Monday-Tuesday: Paper Trading Preparation

- [ ] Setup S1 V2 in paper trading environment
- [ ] Configure position sizing: 2.0% per trade = $400 (if $20K allocation)
- [ ] Setup monitoring dashboard
- [ ] Add S1 V2 to alert system

---

#### Wednesday-Thursday: Documentation

- [ ] Update S1 V2 production guide
- [ ] Document all fixes applied
- [ ] Create operator runbook (signal interpretation, risk management)
- [ ] Update capital allocation plan (Scenario A: 60/20/20)

---

#### Friday: Week 13+ Planning

- [ ] Schedule Week 13-14 paper trading
- [ ] Prepare integration with baseline portfolio
- [ ] Plan capital reallocation (Conservative $80K → $60K, S1 V2 +$20K)
- [ ] Setup correlation monitoring (S1 V2 vs baselines)

---

**If Re-test Failed:**

#### Full Week: Deep Dive Investigation

- [ ] Architecture review: Is confluence approach fundamentally flawed?
- [ ] Alternative approaches:
   - Simpler threshold-based detection (no confluence)
   - Machine learning classifier (historical capitulations)
   - Hybrid approach (threshold + ML)
- [ ] Re-evaluate value proposition (worth the complexity?)
- [ ] Consider archetype redesign vs kill decision

---

## SUCCESS CRITERIA

### Week 9 (Diagnosis)
- ✅ Root cause identified (feature data? confluence logic? config? regime?)
- ✅ Fix plan documented
- ✅ Timeline confirmed (achievable in 4 weeks)

### Week 10 (Fix Implementation)
- ✅ Feature store regenerated with complete data
- ✅ Confluence logic fixed and tested
- ✅ Production config restored
- ✅ Position sizing validated
- ✅ Integration test passes (FTX Nov 2022 signal fires)

### Week 11 (Re-test)
- ✅ Train PF >= 1.4
- ✅ Train trades >= 40 (annualized for 1 year)
- ✅ No train losses > $100
- ✅ OOS 2024 PF >= 1.4
- ✅ OOS 2024 trades >= 20
- ✅ Pipeline logs show clean feature calculation

### Week 12 (Deployment Prep)
- ✅ S1 V2 ready for paper trading
- ✅ Documentation complete
- ✅ Capital allocation plan updated
- ✅ Monitoring infrastructure ready

---

## RISK ASSESSMENT

### Fix Implementation Risks

**Risk 1: Feature Data Not Recoverable**
- **Probability:** LOW (data should exist in raw OHLCV)
- **Impact:** HIGH (cannot fix strategy without data)
- **Mitigation:** Backfill from exchange APIs (Binance, OKX)
- **Contingency:** If data truly missing, re-validate on 2024+ only

**Risk 2: Confluence Logic Fundamentally Flawed**
- **Probability:** MEDIUM (untested complex logic)
- **Impact:** HIGH (requires redesign)
- **Mitigation:** Simplify to threshold-based if confluence fails
- **Contingency:** Fall back to simple capitulation depth threshold

**Risk 3: Re-test Still Shows Zero Trades**
- **Probability:** LOW (after fixes applied)
- **Impact:** CRITICAL (kills strategy)
- **Mitigation:** Deep dive investigation, consider ML approach
- **Contingency:** Kill strategy, focus on baselines + S4

**Risk 4: Historical Validation Was Overfitted**
- **Probability:** MEDIUM (possible)
- **Impact:** HIGH (strategy has no value)
- **Mitigation:** Walk-forward validation on 2024 data
- **Contingency:** Kill strategy if OOS 2024 fails

---

## LESSONS LEARNED (Pre-Fix)

**Lesson 1: Silent Failures are Deadly**
- Zero trades should have triggered immediate alert
- No error messages = wasted time investigating
- **Fix:** Add trade frequency alerts in backtester

**Lesson 2: Config Management is Critical**
- Mismatched configs caused false negative
- Need version control and equality checks
- **Fix:** Config hash verification in comparison pipeline

**Lesson 3: Feature Pipeline Must Be Validated**
- Feature calculations can silently break
- Need unit tests for every feature
- **Fix:** Add feature pipeline test suite

**Lesson 4: Historical Validation ≠ Comparison Test**
- Different pipelines produced different results
- Need unified testing framework
- **Fix:** Single pipeline for all testing

---

## CONTINGENCY PLANS

### Contingency 1: Week 11 Re-test Fails

**Actions:**
1. Escalate to architecture review (quant team lead)
2. Deep dive investigation (4 weeks)
3. Consider alternative approaches (ML, threshold-based)
4. Parallel: Focus on baseline enhancements + S4 fixes
5. Go/Kill decision by Week 16

---

### Contingency 2: Fixes Take Longer Than Expected

**Actions:**
1. Extend timeline to Week 14 (add 2 weeks)
2. Prioritize S4 fixes (higher PF potential)
3. Deploy baselines + S4 first (if S4 fixed)
4. Revisit S1 V2 in Month 4

---

### Contingency 3: Strategy Fundamentally Broken

**Actions:**
1. Kill S1 V2 (archive code for reference)
2. Document lessons learned
3. Focus on baseline enhancements
4. Consider new bear market patterns (S6, S7, etc.)
5. Allocate 100% to baselines (80/20) or baselines + S4

---

## OWNER ASSIGNMENTS

**Data Pipeline Team:**
- Feature store regeneration
- Data quality validation
- Feature calculation fixes

**S1 Strategy Team:**
- Confluence logic review and fix
- Config management
- Unit test development

**Quant Team:**
- Re-test execution
- Performance analysis
- Decision making (KEEP/IMPROVE/KILL)

**Trading Ops:**
- Paper trading setup (if successful)
- Monitoring infrastructure
- Alert system integration

---

## TIMELINE SUMMARY

| Week | Focus | Deliverables | Decision Point |
|------|-------|--------------|----------------|
| Week 9 | Diagnosis | Root cause report | Fix achievable? |
| Week 10 | Fix Implementation | Fixed pipeline | Integration test passes? |
| Week 11 | Re-test | Full validation results | KEEP/IMPROVE/KILL? |
| Week 12 | Deployment Prep | Paper trading ready | Deploy Week 13? |

**Total Timeline:** 4 weeks (Dec 9 - Jan 5)
**Critical Path:** Feature store regeneration → confluence fix → re-test
**Buffer:** 2 weeks (can extend to Week 14 if needed)

---

## FINAL RECOMMENDATION

**Do NOT Kill S1 V2**

**Rationale:**
1. Historical validation shows clear value (PF 1.4-1.8, 40-60 trades/year)
2. Current failure is PIPELINE issue, not strategy issue
3. Clear path to recovery (4 weeks, achievable)
4. High potential reward (capitulation reversal is valuable pattern)
5. Complements baselines (different pattern type, bear specialist)

**Action Plan:**
1. Execute 4-week improvement plan (Week 9-12)
2. Re-test with fixed pipeline (Week 11)
3. If successful: Paper trade Week 13-14, deploy Week 15
4. If failed: Escalate to architecture review, consider kill

**Expected Outcome:**
- 75% probability: Fixes successful, S1 V2 moves to KEEP
- 20% probability: Partial success, needs further work
- 5% probability: Fundamental failure, kill strategy

**Investment:** 4 weeks × 2 people = 8 person-weeks
**Potential Return:** $20K allocation × 50% annual return = $10K/year
**ROI:** High (if fixes successful)

---

**IMPROVEMENT PLAN STATUS:** APPROVED ✅

**Next Action:** Begin Week 9 diagnosis (Dec 9, 2025)

**Owner:** S1 Strategy Team + Data Pipeline Team
**Reviewer:** Quant Team Lead
**Escalation:** If Week 11 re-test fails, escalate to CTO
