# Dual-System Deployment Roadmap: B0 + Archetypes

**Version:** 1.0.0
**Date:** 2025-12-03
**Status:** ACTIVE DEPLOYMENT PLAN
**Timeline:** 5 phases over 8-12 weeks
**Risk Level:** Phased approach minimizes risk

---

## Executive Summary

This roadmap defines a staged deployment strategy for running two independent trading systems in parallel, with clear risk gates and decision points.

**Strategy:** Start conservative, validate with data, scale based on performance.

**Risk Mitigation:** Each phase has exit criteria and rollback procedures.

---

## Phase Overview

```
Timeline:

Week 1:    Phase 1 (B0 Paper Trading)
Week 2:    Phase 2 (S4/S5 Paper Trading)
Week 3-4:  Phase 3 (Combined Paper Trading)
Week 5-8:  Phase 4 (Staged Live Deployment)
Week 9+:   Phase 5 (Full Production)

Risk Gates:
  ✓ Gate 1: B0 paper PF > 2.5
  ✓ Gate 2: S4/S5 paper PF > 1.5
  ✓ Gate 3: Combined portfolio PF > 2.0
  ✓ Gate 4: 4 weeks live success (no catastrophic losses)
```

---

## Phase 1: Deploy B0 to Paper Trading (Week 1)

**Objective:** Validate B0 (Baseline-Conservative) on live data before risking capital.

**Duration:** 7 days

**Capital:** $100k paper money

**Allocation:** 100% B0 (baseline only)

### Tasks

**Day 1: Setup and Deploy**
- [ ] Deploy B0 to paper trading environment
- [ ] Configure monitoring dashboard
- [ ] Test signal generation (manual trigger if needed)
- [ ] Verify order execution (paper fills)
- [ ] Document baseline metrics (starting capital, config)

**Day 2-7: Monitor and Collect Data**
- [ ] Daily check: B0 signals fired
- [ ] Daily check: Paper positions opened/closed
- [ ] Daily check: PnL tracking
- [ ] Daily check: Drawdown vs expectations
- [ ] Log any anomalies (unexpected signals, execution issues)

**Deliverables:**
- B0 paper trading live
- 7 days of paper performance data
- Monitoring dashboard operational
- Incident log (if any issues)

### Success Criteria (Risk Gate 1)

**Minimum Requirements:**
- [ ] B0 fires at least 1 signal in 7 days (if market conditions allow)
- [ ] Paper PF > 2.5 (if trades executed)
- [ ] No critical bugs (signal generation, execution)
- [ ] Latency < 5 seconds (signal to order)

**Expected Outcomes:**
- 0-2 trades in week 1 (low frequency expected)
- If 0 trades: Verify -15% drawdown hasn't occurred (normal)
- If 1-2 trades: PF should be ~2.5-3.5 range

### Risk Assessment

**Risks:**
- B0 doesn't fire (market too stable, no -15% dips)
- Execution latency too high
- Dashboard monitoring fails

**Mitigations:**
- If 0 trades: Extend to week 2, or manually test with historical dip
- If latency high: Investigate, optimize, or adjust expectations
- If monitoring fails: Fix immediately (critical for live trading)

### Decision Point

**Go/No-Go for Phase 2:**

**GO if:**
- B0 executed at least 1 trade with reasonable PF (>2.0)
- OR: No trades but verified signal logic works on manual test
- No critical bugs

**NO-GO if:**
- B0 fires unexpectedly (false signals)
- Execution completely broken
- Critical bugs found

**Rollback:** Not needed (paper trading, no capital at risk)

---

## Phase 2: Deploy S4/S5 to Paper Trading (Week 2)

**Objective:** Validate archetypes (S4 Funding Divergence, S5 Long Squeeze) in parallel with B0.

**Duration:** 7 days

**Capital:** $100k paper money (fresh start)

**Allocation:** 50% B0, 30% S4, 20% S5

### Prerequisites

- [ ] Fix `ArchetypeModel` wrapper (regime gating bug)
- [ ] Validate S4/S5 configs on recent data (2024)
- [ ] Implement runtime enrichment in paper environment
- [ ] Test regime classification (GMM model)

### Tasks

**Day 1: Deploy Archetypes**
- [ ] Deploy S4 and S5 to paper trading
- [ ] Configure archetype-specific monitoring
- [ ] Test regime classification (verify not stuck on 'neutral')
- [ ] Test runtime enrichment (liquidity_score, funding_z, etc.)
- [ ] Verify conflict resolution (if B0 and S4 both signal)

**Day 2-7: Monitor Multi-System Performance**
- [ ] Daily check: Regime classification (risk_on/neutral/risk_off/crisis)
- [ ] Daily check: S4/S5 signal generation
- [ ] Daily check: B0 still working (parallel operation)
- [ ] Daily check: Combined portfolio PnL
- [ ] Log correlation: Do B0 and S4/S5 trade at same times?

**Deliverables:**
- S4 and S5 paper trading live
- 7 days of archetype performance data
- Regime classification logs
- Correlation analysis (B0 vs S4 vs S5)

### Success Criteria (Risk Gate 2)

**Minimum Requirements:**
- [ ] S4 fires at least 1 signal in 7 days (if risk_off/neutral regime occurs)
- [ ] S5 fires at least 1 signal in 7 days (if risk_on regime occurs)
- [ ] Paper PF > 1.5 for each archetype (if trades executed)
- [ ] Regime classification working (not stuck on 'neutral')
- [ ] No wrapper bugs (0 trade issue resolved)

**Expected Outcomes:**
- S4: 0-2 trades (regime-dependent)
- S5: 0-2 trades (regime-dependent)
- If 0 trades for both: Check regime distribution (may be normal if neutral all week)

### Risk Assessment

**Risks:**
- Archetypes fire 0 trades (regime gating or config too strict)
- Wrapper still broken (0 trades despite fixing)
- Regime classifier stuck on 'neutral'
- Runtime enrichment fails (missing features)

**Mitigations:**
- If 0 trades: Relax thresholds by 10-15%, re-test
- If wrapper broken: Emergency fix (1 day), re-deploy
- If regime stuck: Force regime to test (risk_off for S4, risk_on for S5)
- If enrichment fails: Add fallbacks (use 0.0 for missing features)

### Decision Point

**Go/No-Go for Phase 3:**

**GO if:**
- At least ONE archetype (S4 or S5) executed trade with PF > 1.5
- OR: No trades but verified signal logic works on manual test
- Regime classification not stuck
- Wrapper bugs resolved

**NO-GO if:**
- Both archetypes fire 0 trades due to bugs (not regime)
- Wrapper still broken
- Critical execution issues

**Rollback:**
- Disable archetypes (keep B0 only)
- Investigate root cause (1-2 days)
- Fix and re-deploy to Phase 2

---

## Phase 3: Combined Paper Trading Evaluation (Week 3-4)

**Objective:** Evaluate combined portfolio performance and determine live deployment allocation.

**Duration:** 14 days (2 weeks)

**Capital:** $100k paper money (fresh start)

**Allocation:** 50% B0, 25% S4, 25% S5 (balanced for evaluation)

### Tasks

**Week 3: Performance Collection**
- [ ] Run combined portfolio (B0 + S4 + S5)
- [ ] Daily monitoring: Portfolio PnL, drawdown, exposure
- [ ] Daily monitoring: System-level performance (B0 vs S4 vs S5)
- [ ] Log regime transitions and system activity
- [ ] Track conflict resolution events (when multiple systems signal)

**Week 4: Analysis and Decision**
- [ ] Calculate 14-day metrics:
  - Portfolio PF, WR, trade count
  - Per-system PF, WR, trade count
  - Correlation matrix (B0 vs S4 vs S5)
  - Diversification benefit (portfolio DD vs individual DD)
- [ ] Compare to backtest expectations
- [ ] Identify winner(s): B0 vs Archetypes
- [ ] Draft live deployment allocation

**Deliverables:**
- 14 days of combined paper performance
- Comparison report: Paper vs Backtest
- Decision document: Live allocation strategy
- Risk assessment for live deployment

### Success Criteria (Risk Gate 3)

**Minimum Requirements:**
- [ ] Portfolio PF > 2.0 (combined)
- [ ] At least 3 trades executed across all systems
- [ ] No catastrophic losses (DD < 20%)
- [ ] Systems operated independently (no cross-system bugs)
- [ ] Conflict resolution worked (if applicable)

**Comparison to Backtest:**
- B0 paper PF within 20% of backtest (3.17 ± 0.6)
- S4 paper PF within 30% of backtest (2.22 ± 0.7)
- S5 paper PF within 30% of backtest (1.86 ± 0.6)

**Expected Outcomes:**
- 2-5 total trades in 2 weeks
- Portfolio PF in 2.0-3.0 range
- B0 likely best performer (highest backtest PF)

### Risk Assessment

**Risks:**
- Portfolio PF < 2.0 (underperformance)
- Systems highly correlated (no diversification benefit)
- Frequent conflicts (too much overlap)
- One system dominates (others idle)

**Mitigations:**
- If underperformance: Investigate (market regime? Config issue? Slippage?)
- If high correlation: Reduce allocation to redundant system
- If frequent conflicts: Review entry logic, add deconfliction
- If one system idle: Normal (regime-dependent), evaluate regime distribution

### Decision Point

**Live Deployment Allocation Decision:**

**Scenario A: B0 Dominates (B0 PF > S4 PF * 1.5)**
```
Live Allocation:
  B0:  70%
  S4:  15%
  S5:  15%

Rationale: B0 proven superior, keep archetypes for regime specialization
```

**Scenario B: Archetypes Competitive (S4/S5 PF within 20% of B0)**
```
Live Allocation:
  B0:  50%
  S4:  25%
  S5:  25%

Rationale: Balanced portfolio, all systems contributing
```

**Scenario C: Archetypes Outperform (S4/S5 PF > B0 PF)**
```
Live Allocation:
  B0:  30%
  S4:  35%
  S5:  35%

Rationale: Archetypes proven, B0 as safety net
```

**Scenario D: Archetypes Fail (S4/S5 PF < 1.5)**
```
Live Allocation:
  B0:  90%
  S4:  5%
  S5:  5%

Rationale: Deploy B0 only, minimal archetype exposure
```

---

## Phase 4: Staged Live Deployment (Week 5-8)

**Objective:** Deploy to live trading with real capital, starting small and scaling up.

**Duration:** 4 weeks

**Capital:** Start with $10k, scale to $100k over 4 weeks

**Allocation:** Per decision from Phase 3 (default: 50% B0, 25% S4, 25% S5)

### Week 5: Initial Live Deployment ($10k)

**Risk Level:** LOW (10% of target capital)

**Allocation:**
- Total: $10k live
- B0: $5k
- S4: $2.5k
- S5: $2.5k

**Tasks:**
- [ ] Deploy to live exchange (Binance, OKX, etc.)
- [ ] Verify API connectivity
- [ ] Test live order execution (market/limit orders)
- [ ] Monitor latency (signal to execution)
- [ ] Track slippage (paper vs live fills)
- [ ] Daily monitoring: Verify signals match paper trading

**Success Criteria:**
- [ ] At least 1 live trade executed successfully
- [ ] Slippage < 5 bps (acceptable)
- [ ] Latency < 10 seconds
- [ ] No API errors
- [ ] PnL tracking accurate

**Risks:**
- API connectivity issues
- Execution latency too high
- Slippage higher than expected
- Order rejection (insufficient balance, API limits)

**Rollback:**
- If critical issues: Stop live trading immediately
- Revert to paper trading
- Investigate and fix (1-3 days)
- Re-deploy to Week 5

### Week 6: Scale to $25k

**Risk Level:** LOW-MEDIUM (25% of target capital)

**Allocation:**
- Total: $25k live
- B0: $12.5k
- S4: $6.25k
- S5: $6.25k

**Tasks:**
- [ ] Increase capital allocation (2.5x)
- [ ] Verify position sizing scales correctly
- [ ] Monitor for any execution issues at higher size
- [ ] Compare week 6 vs week 5 performance (consistency check)
- [ ] Weekly review: PF, WR, trade count vs expectations

**Success Criteria:**
- [ ] Week 6 PF within 30% of week 5 (consistent performance)
- [ ] No new execution issues at higher capital
- [ ] At least 1-2 trades executed
- [ ] Drawdown < 10%

**Risks:**
- Performance degrades at higher capital (liquidity issues?)
- Execution quality worsens (slippage increases)

**Rollback:**
- If performance degrades significantly: Reduce back to $10k
- If execution issues: Pause, investigate, fix

### Week 7: Scale to $50k

**Risk Level:** MEDIUM (50% of target capital)

**Allocation:**
- Total: $50k live
- B0: $25k
- S4: $12.5k
- S5: $12.5k

**Tasks:**
- [ ] Increase capital allocation (2x)
- [ ] Review cumulative performance (week 5-7)
- [ ] Evaluate system-level performance (B0 vs S4 vs S5)
- [ ] Check if allocation should be adjusted (based on live data)
- [ ] Prepare for final scale-up decision

**Success Criteria:**
- [ ] Cumulative PF (week 5-7) > 2.0
- [ ] Consistent execution quality
- [ ] At least 3-5 trades total
- [ ] Drawdown < 12%

**Risks:**
- Extended drawdown period (no recovery)
- Market regime shift (systems stop firing)

**Rollback:**
- If cumulative PF < 1.5: Pause scale-up, investigate
- If catastrophic loss (DD > 20%): Stop immediately, full review

### Week 8: Scale to $100k (Full Production)

**Risk Level:** MEDIUM-HIGH (100% of target capital)

**Allocation:**
- Total: $100k live
- B0: $50k (if balanced allocation)
- S4: $25k
- S5: $25k

**Tasks:**
- [ ] Final capital scale-up (2x)
- [ ] Full production deployment
- [ ] Enable automated rebalancing (monthly)
- [ ] Weekly performance reports
- [ ] Quarterly strategy review scheduled

**Success Criteria:**
- [ ] Week 8 performance consistent with weeks 5-7
- [ ] Cumulative PF (week 5-8) > 2.0
- [ ] No catastrophic events
- [ ] All systems operating smoothly

### Risk Gate 4: Full Production Approval

**Approve if:**
- [ ] 4-week live performance PF > 2.0
- [ ] No catastrophic losses (DD < 20%)
- [ ] Execution quality acceptable (slippage < 5 bps)
- [ ] Operational stability (no critical bugs)
- [ ] Performance matches paper trading (within 30%)

**Reject if:**
- Cumulative PF < 1.5
- Catastrophic loss occurred
- Persistent execution issues
- Performance significantly worse than paper

**Rollback Plan:**
- If rejected: Scale back to $25k or paper trading
- Full post-mortem (1 week)
- Identify and fix root cause
- Re-start Phase 4 from Week 5

---

## Phase 5: Full Production and Optimization (Week 9+)

**Objective:** Ongoing operation with continuous monitoring and optimization.

**Duration:** Ongoing (long-term)

**Capital:** $100k+ (compound returns or add capital)

**Allocation:** Dynamic (adjust monthly based on performance)

### Week 9-12: Stabilization

**Focus:** Ensure stable operation, collect data, tune parameters.

**Tasks:**
- [ ] Daily monitoring (automated dashboard)
- [ ] Weekly performance review
- [ ] Monthly rebalancing (adjust allocation based on performance)
- [ ] Collect 30 days of live performance data
- [ ] Prepare for first quarterly review

**Monitoring:**
- Portfolio PF, WR, trade count (rolling 30-day)
- Per-system performance
- Regime distribution (actual vs expected)
- Correlation matrix (B0 vs S4 vs S5)
- Drawdown and recovery times

### Month 4-6: Optimization

**Focus:** Fine-tune based on live performance data.

**Tasks:**
- [ ] Analyze 90 days of live data
- [ ] Identify underperforming systems
- [ ] Adjust allocation (increase winners, reduce losers)
- [ ] Test parameter adjustments (if needed):
  - B0: Adjust drawdown threshold (-15% → -12% or -18%?)
  - S4/S5: Adjust fusion thresholds (tighter or looser?)
- [ ] Consider adding S1 (if fixed and validated)

**Rebalancing:**
- If B0 outperforms: Increase B0 allocation by 10-20%
- If S4/S5 outperform: Increase archetype allocation by 10-20%
- If one system consistently underperforms: Reduce to 5-10% (don't remove entirely)

### Month 7+: Long-Term Operations

**Focus:** Automated operation with quarterly reviews.

**Tasks:**
- [ ] Quarterly strategy review
  - Performance vs targets
  - Regime distribution analysis
  - Parameter stability
  - Consider re-optimization if market dynamics change
- [ ] Annual deep review
  - Full backtest refresh (new data)
  - Re-optimize all systems
  - Evaluate new archetypes (if developed)
- [ ] Continuous monitoring
  - Automated alerts (DD > 15%, PF < 1.5, etc.)
  - Daily dashboard check
  - Weekly reports

**Growth:**
- Compound returns monthly (increase capital allocation)
- Or: Withdraw profits quarterly (keep base capital constant)
- Add new systems (if validated)
- Expand to other assets (ETH, SOL, etc.)

---

## Risk Management Throughout Deployment

### Daily Risk Checks (Automated)

```python
def daily_risk_check():
    # Portfolio level
    if portfolio_dd() > 0.15:
        alert("WARNING: Portfolio DD > 15%")

    if portfolio_dd() > 0.25:
        alert("CRITICAL: Portfolio DD > 25% - KILL SWITCH ACTIVATED")
        pause_all_systems()

    # Per-system level
    for system in ['B0', 'S4', 'S5']:
        if system_dd(system) > 0.20:
            alert(f"WARNING: {system} DD > 20%")
            reduce_allocation(system, by=0.20)

    # Exposure check
    if total_exposure() > 0.25:
        alert("WARNING: Total exposure > 25% of capital")

    # Correlation check (weekly)
    if correlation(B0, S4) > 0.8:
        alert("WARNING: B0 and S4 highly correlated (>0.8)")
```

### Weekly Risk Review (Manual)

- [ ] Review max drawdown (daily, weekly)
- [ ] Review stop loss hits (count, severity)
- [ ] Review position sizing (any violations?)
- [ ] Review exposure limits (any breaches?)
- [ ] Review execution quality (slippage, latency)

### Monthly Risk Review (Strategic)

- [ ] Full performance review (PF, WR, Sharpe, DD)
- [ ] Allocation decision (rebalance or maintain)
- [ ] Risk limit evaluation (should limits be adjusted?)
- [ ] System health (any degradation?)
- [ ] Operational issues (bugs, API failures, etc.)

---

## Exit Criteria and Rollback Procedures

### Phase 1 Exit

**Rollback if:**
- B0 fires false signals (unexpected entries)
- Execution completely broken

**Procedure:**
1. Stop B0 deployment
2. Investigate (1 day)
3. Fix and re-test
4. Re-deploy Phase 1

**Impact:** Low (paper trading, no capital at risk)

### Phase 2 Exit

**Rollback if:**
- Archetypes fire 0 trades due to bugs (not regime)
- Wrapper broken
- Critical execution issues

**Procedure:**
1. Disable S4/S5
2. Keep B0 running (if working)
3. Emergency fix (1-3 days)
4. Re-deploy Phase 2

**Impact:** Low (paper trading, B0 unaffected)

### Phase 3 Exit

**Rollback if:**
- Portfolio PF < 1.5 (underperformance)
- Systems not operating independently

**Procedure:**
1. Full review (1 week)
2. Identify root cause
3. Fix and re-test
4. Re-deploy Phase 3 or decide to deploy B0 only

**Impact:** Low-Medium (delays live deployment)

### Phase 4 Exit

**Rollback if:**
- Live PF < 1.0 (losing money)
- Catastrophic loss (DD > 30%)
- Persistent execution issues

**Procedure:**
1. IMMEDIATE: Stop all live trading
2. Close all positions (market orders)
3. Revert to paper trading
4. Full post-mortem (1 week)
5. Identify root cause (slippage? Market regime? Config?)
6. Fix and re-test in paper (2 weeks)
7. Re-deploy Phase 4 from Week 5 OR cancel live deployment

**Impact:** HIGH (real capital at risk, potential significant loss)

### Phase 5 Exit

**Pause if:**
- Monthly PF < 1.0 for 2 consecutive months
- Drawdown > 20% sustained for 30 days

**Procedure:**
1. Reduce allocation by 50% (hold cash)
2. Investigate root cause (1-2 weeks)
3. Fix or adjust strategy
4. Resume gradually (restart Phase 4)

**Impact:** HIGH (long-term strategy viability questioned)

---

## Success Metrics by Phase

### Phase 1: B0 Paper Trading

```
Metrics:
  - Signals fired: 0-2 (expected)
  - Paper PF: >2.5 (target)
  - Latency: <5s
  - Bugs: 0 critical

Success = B0 working as expected
```

### Phase 2: Archetype Paper Trading

```
Metrics:
  - S4 signals: 0-2 (regime-dependent)
  - S5 signals: 0-2 (regime-dependent)
  - Paper PF: >1.5 per archetype
  - Wrapper bugs: 0
  - Regime classifier: Working

Success = At least one archetype validated
```

### Phase 3: Combined Paper Evaluation

```
Metrics:
  - Portfolio PF: >2.0
  - Total trades: >3
  - Diversification: Correlation <0.6
  - Consistency: Performance matches backtest (±30%)

Success = Portfolio outperforms individual systems
```

### Phase 4: Staged Live Deployment

```
Metrics:
  - Live PF (4 weeks): >2.0
  - Max DD: <20%
  - Slippage: <5 bps
  - Execution quality: Good
  - Performance vs paper: Within 30%

Success = Smooth scale from $10k → $100k
```

### Phase 5: Full Production

```
Metrics (Quarterly):
  - PF: >2.0
  - Sharpe: >1.0
  - Max DD: <15%
  - Uptime: >99%
  - Operational incidents: <2 per quarter

Success = Stable long-term operation
```

---

## Timeline Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DEPLOYMENT TIMELINE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Week 1:    Phase 1 - B0 Paper Trading                              │
│             ├─ Deploy B0                                            │
│             ├─ Monitor 7 days                                       │
│             └─ Gate 1: B0 PF >2.5                                   │
│                                                                     │
│  Week 2:    Phase 2 - Archetype Paper Trading                      │
│             ├─ Fix wrapper bugs                                     │
│             ├─ Deploy S4/S5                                         │
│             ├─ Monitor 7 days                                       │
│             └─ Gate 2: S4/S5 PF >1.5                                │
│                                                                     │
│  Week 3-4:  Phase 3 - Combined Evaluation                          │
│             ├─ Run portfolio (B0+S4+S5)                             │
│             ├─ Collect 14 days data                                 │
│             ├─ Analyze and decide allocation                        │
│             └─ Gate 3: Portfolio PF >2.0                            │
│                                                                     │
│  Week 5:    Phase 4 - Initial Live ($10k)                          │
│             ├─ Deploy to live exchange                              │
│             ├─ Test execution                                       │
│             └─ Monitor 7 days                                       │
│                                                                     │
│  Week 6:    Phase 4 - Scale to $25k                                │
│  Week 7:    Phase 4 - Scale to $50k                                │
│  Week 8:    Phase 4 - Scale to $100k                               │
│             └─ Gate 4: 4-week live PF >2.0                          │
│                                                                     │
│  Week 9+:   Phase 5 - Full Production                              │
│             ├─ Automated monitoring                                 │
│             ├─ Monthly rebalancing                                  │
│             └─ Quarterly reviews                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Critical Path: 8 weeks (Phase 1-4)
Full Production: Week 9+
Risk Gates: 4 (B0, S4/S5, Portfolio, Live)
```

---

## Key Milestones

| Milestone | Week | Deliverable | Owner |
|-----------|------|-------------|-------|
| **M1: B0 Paper Validated** | Week 1 | B0 working in paper | DevOps |
| **M2: Archetypes Paper Validated** | Week 2 | S4/S5 working, wrapper fixed | Backend |
| **M3: Combined Portfolio Evaluated** | Week 4 | Allocation decision made | Quant Team |
| **M4: Initial Live Success** | Week 5 | $10k deployed successfully | All |
| **M5: Full Production Deployed** | Week 8 | $100k live, all systems operational | All |
| **M6: First Month Live Complete** | Week 12 | 30 days of live data collected | All |
| **M7: First Quarter Complete** | Week 20 | 90 days of live data, full review | All |

---

## Resource Requirements

### Phase 1-3 (Paper Trading)

**Team:**
- 1 DevOps engineer (setup, monitoring)
- 1 Backend developer (wrapper fixes)
- 1 Quant analyst (performance analysis)

**Infrastructure:**
- Paper trading API access (Binance Testnet, OKX Demo)
- Monitoring dashboard (Grafana or custom)
- Data storage (logs, performance metrics)

**Budget:**
- $0 capital (paper trading)
- $500/month infrastructure (VPS, monitoring)

### Phase 4-5 (Live Trading)

**Team:**
- 1 DevOps engineer (24/7 on-call)
- 1 Backend developer (bug fixes, maintenance)
- 1 Quant analyst (performance monitoring, rebalancing)

**Infrastructure:**
- Live exchange API access (Binance, OKX)
- Production monitoring (uptime alerts, PnL tracking)
- Secure wallet / API key management
- Backup systems (redundancy)

**Budget:**
- $100k capital (initial)
- $1000/month infrastructure (production VPS, monitoring, backup)
- $2000/month team (on-call support)

---

## Communication Plan

### Daily (During Phase 4-5)

**Audience:** Deployment team
**Format:** Slack message
**Content:**
- Daily PnL
- Trades executed
- Any alerts
- System health status

### Weekly (All Phases)

**Audience:** Stakeholders
**Format:** Email report
**Content:**
- Weekly performance summary
- Progress vs plan
- Risk assessment
- Next week actions

### Monthly (Phase 5+)

**Audience:** Executives
**Format:** Presentation
**Content:**
- Monthly performance review
- Allocation decision (rebalance?)
- Risk assessment
- Quarterly outlook

### Quarterly (Phase 5+)

**Audience:** All stakeholders
**Format:** Full review meeting
**Content:**
- 90-day performance analysis
- Strategy review
- Parameter optimization recommendations
- Annual plan

---

## Summary and Recommendations

### Recommended Approach

1. **Start with B0 only** (Week 1) - proven, simple, low risk
2. **Add archetypes incrementally** (Week 2) - validate before scaling
3. **Evaluate combined** (Week 3-4) - data-driven allocation decision
4. **Scale gradually** (Week 5-8) - $10k → $100k over 4 weeks
5. **Optimize continuously** (Week 9+) - monthly rebalancing

### Risk Mitigation

- **Phased deployment:** Start small, scale based on success
- **Clear risk gates:** Must pass each gate before proceeding
- **Rollback procedures:** Defined exit criteria and procedures at each phase
- **Real-time monitoring:** Automated alerts for anomalies
- **Manual reviews:** Weekly and monthly human oversight

### Expected Outcomes

**Conservative Estimate:**
- Portfolio PF: 2.0-2.5
- Trade frequency: 12-20/year
- Max drawdown: 12-15%
- Success rate: 70% (30% chance need to rollback to B0 only)

**Optimistic Estimate:**
- Portfolio PF: 2.5-3.0
- Trade frequency: 15-25/year
- Max drawdown: 10-12%
- Success rate: 90% (all systems work well)

**Realistic Target:**
- Portfolio PF: 2.2-2.7
- Trade frequency: 12-20/year
- Max drawdown: 12-15%
- Success rate: 80%

---

**Document Owner:** Deployment Team
**Last Updated:** 2025-12-03
**Next Review:** End of Phase 1 (Week 1)
