# ARCHETYPE DECISION MATRIX

**Date:** 2025-12-07
**Purpose:** Decision framework based on archetype diagnostic results
**Status:** Diagnosis complete, user decision required

---

## DIAGNOSTIC RESULTS SUMMARY

**Finding:** STRATEGY ISSUE (Not Plumbing)

**Evidence:**
- ✅ Pipeline functional (complete data, correct configs)
- ✅ Wrapper fixed (matches native engine exactly)
- ✅ Archetypes work mechanically (detect signals, execute trades)
- ❌ **Archetypes have no edge** (52% worse than simple baseline)

**Performance Gap:**
- Best Baseline: SMA50x200 (Test PF 3.24)
- Best Archetype: S1/S5 (Test PF 1.55)
- Underperformance: -1.69 (-52%)

---

## DECISION TREE

```
START: Archetype Diagnosis Complete
  |
  ├─> Is this a plumbing issue? (wrapper/pipeline broken)
  |   ├─> YES → Fix plumbing, retest
  |   └─> NO → Continue to strategy assessment ✅ [WE ARE HERE]
  |
  ├─> Do archetypes beat baselines?
  |   ├─> YES (PF > 3.34) → Deploy archetypes
  |   └─> NO (PF 1.55 << 3.34) → Continue to gap analysis ✅ [WE ARE HERE]
  |
  ├─> Is the gap fixable with tuning?
  |   ├─> YES (gap < 10%) → Tune parameters (1-2 weeks)
  |   └─> NO (gap = 52%) → Continue to strategic options ✅ [WE ARE HERE]
  |
  └─> Strategic Options:
      ├─> Option A: Deploy Baselines (RECOMMENDED)
      ├─> Option B: Fundamental Redesign (OPTIONAL, 4 months)
      └─> Option C: Abandon Trading System (NOT RECOMMENDED)
```

---

## OPTION A: DEPLOY BASELINES (RECOMMENDED)

### Description
Abandon archetypes, deploy proven baseline strategies immediately.

### Strategy
- **Primary (80%):** SMA50x200 Crossover (Test PF 3.24, OOS PF 2.62)
- **Diversifier (20%):** VolTarget 2% (Test PF 1.45, OOS PF 1.94)

### Expected Performance
- Weighted Portfolio PF: 2.88
- Annual trades: ~20-25
- Sharpe: 1.15-1.44
- Max DD: <10%

### Timeline
| Week | Phase | Action | Go/No-Go Criteria |
|------|-------|--------|-------------------|
| 1-2 | Paper Trading | Setup and run paper | Generate signals correctly |
| 3-4 | Validation | Analyze paper results | PF >= 80% of backtest (2.59) |
| 5-6 | Live (Small) | 10% capital allocation | Match paper performance |
| 7-8 | Live (Full) | 100% capital allocation | Maintain expected PF |

### Pros
- ✅ Works NOW (proven PF 3.24)
- ✅ Simple to implement and maintain
- ✅ Regime-robust (works in bear/bull/mixed)
- ✅ Low risk (4.86% max DD)
- ✅ Immediate revenue potential
- ✅ Can research new strategies in parallel

### Cons
- ❌ Low trade count (17 test trades = high variance)
- ❌ Gives up on archetype research (6 months sunk cost)
- ❌ May miss opportunities in specific regimes
- ❌ Simple strategies may not adapt to changing markets

### When to Choose This Option
- **You want revenue NOW:** Can't wait 4 months for redesign
- **You trust simple strategies:** Believe trend-following has lasting edge
- **You want to cut losses:** Sunk cost fallacy avoided
- **You're resource-constrained:** No time/budget for 4-month redesign

### Risk Assessment
| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Low trade count variance | High | Medium | Add VolTarget diversifier (more trades) |
| Regime shift | Medium | High | Monitor monthly, rebalance if needed |
| Baseline degradation | Low | Medium | Quarterly walk-forward validation |
| Opportunity cost (miss archetype edge) | Low | Low | Archetypes have no proven edge |

### Success Criteria
- [ ] Paper PF >= 2.59 (80% of backtest 3.24)
- [ ] Live (small) matches paper performance
- [ ] No execution issues or data quality problems
- [ ] Portfolio Sharpe >= 1.0

### Hard Stop Criteria
- Paper PF < 1.5 for 4 consecutive weeks → Investigate discrepancy
- Live losses exceed 10% of capital → Halt and review
- Data quality issues > 1% of time → Fix infrastructure first

---

## OPTION B: FUNDAMENTAL REDESIGN (OPTIONAL)

### Description
4-month intensive effort to rebuild archetypes from scratch with strict acceptance criteria.

### Redesign Principles
1. **Start Simple:** Single alpha source (liquidity OR funding OR OI, not all)
2. **Test in Bear:** Must be profitable in 2022 bear market
3. **Beat Baseline:** Must exceed PF 3.34 before adding complexity
4. **Prove Independence:** Correlation < 0.5 with other strategies
5. **Regime Agnostic:** Must work across bear/bull/mixed regimes

### Timeline
| Month | Phase | Deliverable | Success Criteria |
|-------|-------|-------------|------------------|
| 1 | Research | Single-alpha prototypes | Each alpha source beats PF 1.5 in 2022 bear |
| 2 | Development | Best alpha integrated | Prototype beats PF 2.5 in walk-forward |
| 3 | Optimization | Multi-regime tuning | Prototype beats PF 3.34 in test |
| 4 | Validation | Walk-forward + OOS | Maintain PF > 3.0 in OOS, deploy if pass |

### Hard Acceptance Criteria
- [ ] Test PF > 3.34 (beat SMA50x200 + 0.1 margin)
- [ ] Train PF > 1.0 (profitable in 2022 bear market)
- [ ] OOS degradation < 20% (robust generalization)
- [ ] Correlation < 0.5 with baseline (independent alpha)
- [ ] Sharpe > 1.0 (acceptable risk-adjusted returns)
- [ ] Trade count >= 30 (statistical significance)

### Pros
- ✅ Salvages 6 months of archetype research
- ✅ May discover fundamental insights about market structure
- ✅ Builds competitive moat if successful (complex edge)
- ✅ Team learning and skill development
- ✅ Clear success/failure criteria

### Cons
- ❌ 4-month delay to production (zero revenue)
- ❌ High probability of failure (52% gap is large)
- ❌ Opportunity cost (could be improving baselines)
- ❌ Sunk cost risk (may fail after 4 months)
- ❌ No guarantee of success

### When to Choose This Option
- **You believe in archetype potential:** Despite current failures
- **You have time and budget:** Can afford 4-month delay
- **You want competitive edge:** Simple baselines too commoditized
- **You're research-focused:** Value learning over immediate revenue

### Risk Assessment
| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Redesign fails after 4 months | High (70%) | High | Hard deadline (kill if fail) |
| Team burnout | Medium | Medium | Clear milestones, celebrate progress |
| Opportunity cost | High | Medium | Run baselines in parallel for revenue |
| Market conditions change | Medium | High | Monthly regime checks |

### Success Milestones
**Month 1 Checkpoint:**
- [ ] Liquidity-only archetype: PF > 1.5 in 2022 bear
- [ ] Funding-only archetype: PF > 1.5 in 2022 bear
- [ ] OI-only archetype: PF > 1.5 in 2022 bear
- **If FAIL:** Kill redesign, deploy baselines

**Month 2 Checkpoint:**
- [ ] Best single-alpha prototype: PF > 2.5 in walk-forward
- [ ] Proven edge over baseline in at least one regime
- **If FAIL:** Kill redesign, deploy baselines

**Month 3 Checkpoint:**
- [ ] Prototype beats PF 3.34 in test period
- [ ] OOS degradation < 20%
- **If FAIL:** Kill redesign, deploy baselines

**Month 4 Final:**
- [ ] All hard acceptance criteria met
- **If PASS:** Deploy archetype + baseline ensemble
- **If FAIL:** Deploy baselines only, abandon archetypes permanently

### Hard Stop Criteria
- Any monthly checkpoint fails → Kill redesign immediately
- Budget exceeds $X → Reassess cost/benefit
- Key team member leaves → Reassess feasibility

---

## OPTION C: ABANDON TRADING SYSTEM (NOT RECOMMENDED)

### Description
Shut down the entire trading system project, neither archetypes nor baselines.

### When to Choose This Option
- **Baselines also fail in paper trading:** PF < 1.5
- **Data quality issues unfixable:** Cannot trust results
- **Regulatory/legal barriers:** Cannot deploy
- **Fundamental business pivot:** Trading no longer strategic priority

### Pros
- ✅ Cuts all losses immediately
- ✅ Frees up team for other projects
- ✅ Avoids risk of live trading losses

### Cons
- ❌ Wastes 6+ months of research
- ❌ Abandons proven baseline (PF 3.24)
- ❌ No ROI on infrastructure investment
- ❌ Competitive disadvantage (if trading is core business)

### When NOT to Choose This Option (Current Situation)
- ❌ We have a proven baseline (SMA50x200 PF 3.24)
- ❌ Data quality is good (26K bars, <2% null)
- ❌ Infrastructure is functional (backtesting framework works)
- ❌ No regulatory barriers identified

**Recommendation:** Do NOT choose this option unless baselines also fail.

---

## DECISION CRITERIA COMPARISON

| Criterion | Option A (Baselines) | Option B (Redesign) | Option C (Abandon) |
|-----------|---------------------|-------------------|------------------|
| **Time to Revenue** | 5-8 weeks | 4+ months | Never |
| **Probability of Success** | High (80%) | Low (30%) | N/A |
| **Expected PF** | 2.88 (proven) | 3.5+ (if successful) | 0.00 |
| **Implementation Risk** | Low | High | None |
| **Opportunity Cost** | Low | High (4 months) | Total |
| **Learning Value** | Low | High | None |
| **Competitive Edge** | Low (simple) | High (if successful) | None |
| **Team Morale** | Mixed (kill archetypes) | Motivating (second chance) | Demoralizing |
| **Financial Risk** | Low (<10% DD) | Medium (delay + risk) | Sunk cost only |

---

## RECOMMENDED PATH

### Primary Recommendation: OPTION A (Deploy Baselines)

**Reasoning:**
1. **Proven Edge:** SMA50x200 has demonstrated PF 3.24 (excellent)
2. **Low Risk:** 4.86% max DD, robust across regimes
3. **Immediate Revenue:** Can deploy in 5-8 weeks
4. **Clear Fallback:** If fails, still have Option B/C
5. **Archetype Gap Too Large:** 52% underperformance not fixable with tuning

**Action Plan:**
- Week 1-2: Setup paper trading (SMA50x200 + VolTarget)
- Week 3-4: Validate paper results (target PF >= 2.59)
- Week 5-6: Live deployment (10% capital)
- Week 7-8: Scale to full allocation (100% capital)

**Parallel Research (Optional):**
If team wants to pursue archetype redesign, do it in parallel:
- 80% effort: Baseline production deployment and monitoring
- 20% effort: Archetype redesign research (no deadline pressure)
- If redesign succeeds in 4 months: Add to portfolio
- If redesign fails: Already have revenue from baselines

---

### Secondary Recommendation: OPTION B (Only if Specific Conditions Met)

**Required Conditions:**
1. ✅ User has 4-month runway (can afford zero revenue)
2. ✅ Team is motivated (not burned out)
3. ✅ Clear budget allocated for redesign
4. ✅ Willing to kill after 4 months if fail
5. ✅ Strategic value in complex edge (not commoditized baselines)

**Modified Approach (Lower Risk):**
- **Week 1-2:** Deploy baselines to paper (parallel track for revenue)
- **Month 1-4:** Archetype redesign (strict monthly checkpoints)
- **If Redesign Succeeds:** Add to portfolio (baseline + archetype ensemble)
- **If Redesign Fails:** Already have baselines in production

**Key Principle:** Don't put all eggs in redesign basket. Run baselines in parallel for insurance.

---

### Not Recommended: OPTION C (Never Choose This)

**Unless:**
- Baselines also fail in paper trading (PF < 1.5)
- Data quality unfixable (>10% null, gaps)
- Regulatory shutdown
- Business pivot away from trading

**Current Reality:** None of these conditions exist. We have proven baseline (PF 3.24).

---

## HYBRID APPROACH (BEST OF BOTH WORLDS)

### Strategy
1. **Deploy baselines immediately** (Option A)
2. **Research archetypes in parallel** (Option B, but lower risk)

### Resource Allocation
- **80% Team Effort:** Baseline production (deployment, monitoring, improvement)
- **20% Team Effort:** Archetype redesign (research, no deadline pressure)

### Timeline
| Week | Baseline Track (80% effort) | Archetype Track (20% effort) |
|------|----------------------------|------------------------------|
| 1-2 | Paper trading setup | Research single-alpha prototypes |
| 3-4 | Paper validation | Test prototypes in 2022 bear |
| 5-6 | Live deployment (small) | Integrate best alpha source |
| 7-8 | Live deployment (full) | Optimize multi-regime |
| 9-12 | Production monitoring | Walk-forward validation |
| 13-16 | Improvement cycle | Final OOS test |

### Decision Points
**Week 4:** If paper PF < 2.59, investigate before live
**Week 8:** If live matches paper, scale to full. If not, halt.
**Week 16:** If archetype redesign succeeds (PF > 3.34), add to portfolio. If fails, continue baselines only.

### Pros
- ✅ Immediate revenue (baselines)
- ✅ Research optionality (archetypes)
- ✅ Lower risk (baselines proven)
- ✅ Team morale (second chance for archetypes, but not dependent on it)
- ✅ Competitive edge if redesign succeeds

### Cons
- ❌ Split focus (80/20 instead of 100%)
- ❌ Slower archetype progress (20% effort)
- ❌ May still fail after 4 months

**Verdict:** This is the BEST approach for most situations.

---

## FINAL RECOMMENDATION SUMMARY

### For Most Users: HYBRID APPROACH
- Deploy baselines immediately (80% effort)
- Research archetypes in parallel (20% effort, no pressure)
- If archetypes succeed: Add to portfolio
- If archetypes fail: Already have revenue from baselines

### For Revenue-Focused Users: OPTION A ONLY
- Deploy baselines immediately (100% effort)
- Abandon archetypes permanently
- Focus on improving baselines (optimization, multi-asset, regime-specific)

### For Research-Focused Users: OPTION B ONLY
- 4-month archetype redesign (100% effort)
- Strict monthly checkpoints
- Kill if fail, deploy if succeed
- High risk, high reward

### For Users Who Hate Trading: OPTION C
- Abandon everything
- Not recommended given proven baseline exists

---

## DECISION WORKSHEET

**User: Complete this worksheet to determine your path**

### Question 1: What is your primary goal?
- [ ] Revenue NOW (next 2 months) → Option A or Hybrid
- [ ] Competitive edge (complex strategies) → Option B or Hybrid
- [ ] Learning and research → Option B
- [ ] Exit trading → Option C

### Question 2: What is your risk tolerance?
- [ ] Low (proven strategies only) → Option A
- [ ] Medium (proven + research) → Hybrid
- [ ] High (research, accept failure risk) → Option B

### Question 3: What is your timeline?
- [ ] Need revenue in 2 months → Option A or Hybrid
- [ ] Can wait 4-6 months → Option B or Hybrid
- [ ] No timeline (research-focused) → Option B

### Question 4: What is your team capacity?
- [ ] Small team (1-2 people) → Option A (focus)
- [ ] Medium team (3-5 people) → Hybrid (split effort)
- [ ] Large team (6+ people) → Option B (can afford full redesign)

### Question 5: What is your budget?
- [ ] Limited ($0-50K) → Option A (low cost)
- [ ] Medium ($50-200K) → Hybrid (baseline + research)
- [ ] Large ($200K+) → Option B (full redesign)

### Scoring:
- **Mostly Option A:** Deploy baselines only
- **Mostly Hybrid:** Deploy baselines + research archetypes (recommended)
- **Mostly Option B:** Attempt redesign (high risk)
- **Mostly Option C:** Abandon (only if baselines fail)

---

## NEXT STEPS AFTER DECISION

### If Option A (Deploy Baselines):
1. [ ] Review SMA50x200 + VolTarget configs
2. [ ] Setup paper trading environment
3. [ ] Begin Week 1-2 paper trading
4. [ ] Archive archetype code (do not delete, for reference)
5. [ ] Document lessons learned from archetype failure

### If Option B (Redesign):
1. [ ] Create detailed redesign plan (4-month roadmap)
2. [ ] Establish monthly checkpoints with hard criteria
3. [ ] Allocate budget and team resources
4. [ ] Begin Month 1: Single-alpha prototypes
5. [ ] Setup kill switch (auto-abort if checkpoint fails)

### If Hybrid (Recommended):
1. [ ] Split team: 80% baselines, 20% research
2. [ ] Begin baseline paper trading (Week 1-2)
3. [ ] Begin archetype research (Month 1-4, parallel)
4. [ ] Setup dual-track monitoring
5. [ ] Decision point at Week 16: Add archetypes or continue baselines only

### If Option C (Abandon):
1. [ ] Final retrospective (what went wrong?)
2. [ ] Archive all code and documentation
3. [ ] Reallocate team to other projects
4. [ ] Document lessons learned
5. [ ] Exit trading space

---

**Decision Required:** Which path will you choose?

**Recommended:** Hybrid Approach (80% baselines + 20% research)

**Date Decision Made:** _____________
**Signature:** _____________
