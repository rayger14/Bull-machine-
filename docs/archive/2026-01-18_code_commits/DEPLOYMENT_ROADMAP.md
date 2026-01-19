# DEPLOYMENT ROADMAP - 8-WEEK TIMELINE

**Version:** 1.0
**Date:** 2025-12-07
**Status:** APPROVED - READY FOR EXECUTION
**Target Start:** Week of 2025-12-09

---

## OVERVIEW

**Objective:** Deploy Baseline-Conservative and Baseline-Aggressive strategies to production over 8 weeks using phased approach (paper → limited capital → full deployment).

**Strategies Deploying:**
- Baseline-Conservative: 80% allocation ($80,000)
- Baseline-Aggressive: 20% allocation ($20,000)

**Strategies Pending:**
- S1-LiquidityVacuum: Fix pipeline (Week 9-12)
- S4-FundingDivergence: Fix pipeline (Week 9-12)

**Total Capital:** $100,000

---

## WEEK-BY-WEEK TIMELINE

### Week 1: Dec 9-15, 2025 - Paper Trading Setup

**Objective:** Deploy both baselines to paper trading environment, validate signal generation

#### Monday Dec 9 - Infrastructure Setup

**Morning (9:00-12:00):**
- [ ] Provision paper trading accounts (Binance/OKX testnet)
- [ ] Deploy baseline configs to trading server
- [ ] Configure data feeds (OHLCV, funding rates)
- [ ] Setup monitoring dashboard (Grafana/custom)

**Afternoon (13:00-17:00):**
- [ ] Test signal generation (historical replay on Dec 8 data)
- [ ] Validate execution engine (order placement, fills)
- [ ] Configure alert system (Discord/Telegram/email)
- [ ] Document baseline parameters in production log

**Deliverables:**
- Paper trading environment live and tested
- Both baselines generating signals correctly
- Monitoring dashboard showing real-time status

---

#### Tuesday Dec 10 - Signal Validation

**Morning:**
- [ ] Review overnight signals (if any generated)
- [ ] Compare signals to backtest expectations
- [ ] Validate feature calculations (log actual vs expected values)
- [ ] Check data feed quality (gaps, anomalies)

**Afternoon:**
- [ ] Backtest baseline configs on Dec 1-8 (smoke test)
- [ ] Expected: Signals should match paper trading signals
- [ ] If mismatch: Debug signal generation logic
- [ ] Document any discrepancies

**Deliverables:**
- Signal validation report
- Any bugs identified and fixed
- Baseline signal generation confirmed accurate

---

#### Wednesday Dec 11 - Position Sizing & Risk Config

**Morning:**
- [ ] Configure position sizing:
  - Conservative: 2.0% per trade = $1,600 (paper mode: log only)
  - Aggressive: 1.5% per trade = $300 (paper mode: log only)
- [ ] Set stop loss levels:
  - Conservative: 3.0 ATR
  - Aggressive: 2.5 ATR
- [ ] Configure take profit targets:
  - Conservative: 2.5R
  - Aggressive: 1.8R

**Afternoon:**
- [ ] Test risk management:
  - Daily max loss: -$500 (paper mode)
  - Weekly max loss: -$1,000 (paper mode)
  - Portfolio max loss: -$1,500 (paper mode)
- [ ] Validate stop loss execution (paper fills)
- [ ] Test circuit breaker logic

**Deliverables:**
- Risk management parameters configured
- Position sizing validated
- Stop loss logic tested

---

#### Thursday Dec 12-Friday Dec 13 - Paper Trading Observation

**Daily Routine:**
- [ ] 9:00 AM: Review overnight activity
- [ ] Check signal generation log
- [ ] Monitor open positions (if any)
- [ ] Validate data feed quality
- [ ] 5:00 PM: Daily summary report

**Metrics to Track:**
- Signals generated (Conservative + Aggressive)
- Paper fills (entry price, slippage)
- Stop loss hits (if any)
- Data quality (gaps, delays)

**Expected Activity:**
- Conservative: 0-1 signals (low frequency)
- Aggressive: 0-2 signals (moderate frequency)
- Paper fills should execute within 1 minute

**Deliverables:**
- 2 days of paper trading data
- Signal generation log
- Execution quality report

---

#### Weekend Dec 14-15 - Week 1 Review

**Saturday Analysis:**
- [ ] Review full week paper trading data
- [ ] Compare signal frequency to backtest expectations
- [ ] Analyze execution quality (slippage, timing)
- [ ] Check data feed reliability (uptime, accuracy)

**Decisions:**
- [ ] Continue to Week 2? (if signals match expectations)
- [ ] Extend Week 1? (if issues found)
- [ ] Abort deployment? (if critical issues)

**Week 1 Success Criteria:**
- ✅ Paper trading infrastructure stable (>99% uptime)
- ✅ Signal generation matches backtest logic
- ✅ No data feed issues
- ✅ Execution quality acceptable (slippage < 5 bps)

**Deliverables:**
- Week 1 summary report
- Go/No-Go decision for Week 2
- Issue tracker (if any bugs found)

---

### Week 2: Dec 16-22, 2025 - Paper Trading Validation

**Objective:** Collect 2 weeks of paper trading data, validate performance vs backtest

#### Monday Dec 16 - Week 2 Kickoff

**Morning:**
- [ ] Review Week 1 results
- [ ] Address any issues from Week 1
- [ ] Confirm continued paper trading
- [ ] Update monitoring dashboard (add Week 2 metrics)

**Afternoon:**
- [ ] Continue paper trading observation
- [ ] Begin trade journaling (document each signal)
- [ ] Start PnL tracking (paper trades)

**Metrics to Track (Week 2 focus):**
- Paper PF vs backtest PF
- Win rate vs backtest WR
- Trade frequency vs expectations
- Correlation between Conservative and Aggressive

---

#### Tuesday Dec 17 - Trade Analysis Begins

**For each signal generated:**
- [ ] Document entry conditions (which features triggered)
- [ ] Record entry price and slippage
- [ ] Track stop loss and take profit levels
- [ ] Monitor trade progression (unrealized PnL)

**Analysis:**
- [ ] Compare to backtest equivalent period (Dec 2023?)
- [ ] Assess if signal quality matches expectations
- [ ] Review false positive rate (if applicable)

---

#### Wednesday Dec 18-Friday Dec 20 - Continued Monitoring

**Daily Routine (same as Week 1):**
- [ ] 9:00 AM: Overnight review
- [ ] Monitor signals and positions
- [ ] Update trade journal
- [ ] Check data quality
- [ ] 5:00 PM: Daily summary

**Additional Tasks:**
- [ ] Mid-week performance review (Wednesday PM)
- [ ] Correlation analysis (Conservative vs Aggressive)
- [ ] Execution quality trending

---

#### Weekend Dec 21-22 - 2-Week Validation

**Saturday Analysis:**
- [ ] Compile 2-week paper trading results
- [ ] Calculate actual PF, WR, trade count
- [ ] Compare to backtest expectations:
  - Conservative: Expected PF 3.17, WR 42.9%
  - Aggressive: Expected PF 2.10, WR 33.3%

**Sunday Decision Making:**

**Scenario A: Paper PF >= 80% of Backtest PF**
- ✅ **Proceed to Week 3 (Live Deployment Small Size)**
- Rationale: Performance acceptable, begin limited capital deployment

**Scenario B: Paper PF 50-80% of Backtest PF**
- ⚠️ **Investigate and Extend Paper Trading 2 More Weeks**
- Identify why underperforming (regime mismatch, execution issues, data quality)
- Fix issues before proceeding to live

**Scenario C: Paper PF < 50% of Backtest PF**
- ❌ **Halt Deployment, Full Investigation**
- Critical issues present (data pipeline, strategy logic, regime mismatch)
- Do not proceed to live until root cause identified and fixed

**Week 2 Success Criteria:**
- ✅ Conservative paper PF >= 2.5 (80% of 3.17)
- ✅ Aggressive paper PF >= 1.7 (80% of 2.10)
- ✅ Trade frequency within expectations (±50%)
- ✅ No systemic execution issues
- ✅ Data feed reliability > 99%

**Deliverables:**
- 2-week paper trading report
- Performance comparison table (paper vs backtest)
- Go/No-Go decision for live deployment
- Risk manager sign-off

---

### Week 3-4: Dec 23-Jan 5, 2026 - Live Deployment (10% Capital)

**Objective:** Deploy to live trading with limited capital, monitor closely

**Note:** Week 3 includes Christmas/New Year holidays - adjust as needed

#### Dec 23-27 - Initial Live Deployment

**Monday Dec 23 - Live Go-Live (10% Capital):**

**Morning (Pre-Market):**
- [ ] Final risk manager approval
- [ ] Transfer capital to live exchange accounts:
  - Conservative: $8,000 (10% of $80K target)
  - Aggressive: $2,000 (10% of $20K target)
- [ ] Switch from paper to live execution (CRITICAL - verify config)
- [ ] Triple-check risk limits active:
  - Daily max loss: -$500
  - Weekly max loss: -$1,000
  - Max position size: Conservative $160, Aggressive $30

**Afternoon (Post-Go-Live):**
- [ ] Monitor first live signal (if generated)
- [ ] Validate live execution (order fills, fees, slippage)
- [ ] Check API connectivity and stability
- [ ] Document first live trade in detail

**CRITICAL ALERTS:**
- Any trade > $200 (position size error)
- Daily loss > -$500 (circuit breaker)
- Data feed disruption > 5 minutes
- API errors or failed orders

---

**Dec 24-27 (Holiday Week):**

**Reduced Activity Expected:**
- Lower trading volume (holidays)
- Potential for fewer signals
- Keep monitoring active (24/7)

**Daily Checks:**
- [ ] Morning review (9:00 AM)
- [ ] Evening review (9:00 PM)
- [ ] On-call for critical alerts

**Holiday Adjustments:**
- Consider pausing live trading Dec 25-26 (Christmas)
- Resume Dec 27 with normal monitoring

---

#### Dec 28-Jan 5 - Live Trading Week 2

**Post-Holiday Resumption:**
- [ ] Review holiday week activity (likely minimal)
- [ ] Resume full monitoring schedule
- [ ] Collect live trading data for analysis

**Daily Routine:**
- [ ] 9:00 AM: Overnight review + live positions
- [ ] Monitor real-time execution quality
- [ ] Track actual slippage vs paper trading
- [ ] Compare live PnL to paper PnL expectations
- [ ] 5:00 PM: Daily summary + risk review

**Trade Analysis (for each live trade):**
- [ ] Entry: Price, slippage, fees
- [ ] Execution: Time from signal to fill
- [ ] Risk: Stop loss placement, position size
- [ ] PnL: Unrealized and realized
- [ ] Post-mortem: Compare to backtest expectation

---

#### Weekend Jan 4-5, 2026 - 2-Week Live Review

**Saturday Analysis:**
- [ ] Compile 2 weeks of live trading results (limited data due to holidays)
- [ ] Calculate live PF, WR, trade count
- [ ] Compare live vs paper performance
- [ ] Assess execution quality (slippage, fees, timing)

**Sunday Decision Making:**

**Scenario A: Live PF Matches Paper PF (±20%)**
- ✅ **Proceed to Week 5 (Scale to Full Capital)**
- Rationale: Live execution matches expectations, ready for full deployment

**Scenario B: Live PF Underperforms Paper by 20-50%**
- ⚠️ **Extend 10% Capital Deployment 2 More Weeks**
- Investigate execution issues (slippage higher than expected? fees?)
- May be holiday anomaly (low volume, wide spreads)
- Collect more data before scaling

**Scenario C: Live PF Underperforms Paper by >50% or Loss**
- ❌ **Reduce to Paper Trading, Full Investigation**
- Critical execution issues (live environment differs from paper)
- Do not scale capital until issues resolved
- Consider third-party execution audit

**Week 3-4 Success Criteria:**
- ✅ Live PF >= 80% of paper PF (account for holidays)
- ✅ No risk limit breaches
- ✅ Execution quality acceptable (slippage < 10 bps)
- ✅ Psychological comfort confirmed (no panic exits)
- ✅ API stability > 99.5%

**Deliverables:**
- 2-week live trading report (10% capital)
- Live vs paper comparison table
- Execution quality analysis
- Go/No-Go decision for full capital deployment
- Trading ops sign-off

---

### Week 5-6: Jan 6-19, 2026 - Scale to Full Capital

**Objective:** Scale from 10% to 100% capital allocation, monitor portfolio risk

#### Monday Jan 6 - Capital Scale-Up

**Morning (Pre-Market):**
- [ ] Final approval to scale to full capital
- [ ] Transfer remaining capital to live accounts:
  - Conservative: +$72,000 (total $80,000)
  - Aggressive: +$18,000 (total $20,000)
- [ ] Update position sizing:
  - Conservative: 2.0% = $1,600 per trade
  - Aggressive: 1.5% = $300 per trade
- [ ] Update risk limits:
  - Daily max loss: -$2,000 (2% of $100K)
  - Weekly max loss: -$5,000 (5% of $100K)
  - Monthly max loss: -$10,000 (10% of $100K)

**Afternoon:**
- [ ] Monitor first full-size trade (if generated)
- [ ] Validate larger position execution (no slippage spike?)
- [ ] Psychological check-in (comfortable with full capital at risk?)

---

#### Jan 7-12 - Full Capital Week 1

**Daily Routine:**
- [ ] 9:00 AM: Overnight review
- [ ] Monitor portfolio exposure (max 6% = $6,000 across 3 positions)
- [ ] Track correlation between Conservative and Aggressive
- [ ] Real-time risk monitoring (drawdown, exposure)
- [ ] 5:00 PM: Daily portfolio summary

**Risk Management Focus:**
- [ ] Max concurrent positions: 3 (across both strategies)
- [ ] Max single position: $1,600 (Conservative)
- [ ] Portfolio correlation < 0.7 target
- [ ] No risk limit breaches

**Weekly Checkpoint (Friday Jan 12):**
- [ ] Review first week full capital performance
- [ ] Compare to backtest expectations
- [ ] Assess risk metrics (drawdown, exposure)
- [ ] Adjust if needed (reduce size if uncomfortable)

---

#### Jan 13-19 - Full Capital Week 2

**Continued Monitoring:**
- [ ] Same daily routine as Week 1
- [ ] Begin portfolio-level analysis (diversification benefit)
- [ ] Track rebalancing triggers (allocation drift > 10%?)

**Portfolio Analysis:**
- [ ] Calculate realized correlation (Conservative vs Aggressive)
- [ ] Measure diversification benefit (actual variance reduction)
- [ ] Compare portfolio PF to individual strategy PFs

---

#### Weekend Jan 18-19 - 2-Week Full Capital Review

**Saturday Analysis:**
- [ ] Compile 2 weeks full capital results
- [ ] Calculate portfolio metrics:
  - Portfolio PF (weighted: 0.8 × Conservative + 0.2 × Aggressive)
  - Portfolio Sharpe (if enough data)
  - Correlation (Conservative vs Aggressive)
  - Diversification benefit (variance reduction)

**Sunday Decision Making:**

**Scenario A: Portfolio PF >= Backtest Expectations**
- ✅ **Declare Deployment SUCCESSFUL**
- Transition to ongoing monitoring (Week 7+)
- Quarterly review schedule established

**Scenario B: Portfolio PF 70-100% of Expectations**
- ✅ **Continue Deployment, Monthly Review**
- Some degradation acceptable (regime differences)
- Monitor closely for further deterioration

**Scenario C: Portfolio PF < 70% of Expectations**
- ❌ **Reduce Capital, Investigate Issues**
- Scale back to 50% capital while investigating
- Root cause analysis (regime mismatch? execution quality? data issues?)

**Week 5-6 Success Criteria:**
- ✅ Portfolio PF >= 2.1 (70% of expected 2.96)
- ✅ No risk limit breaches (daily/weekly/monthly)
- ✅ Correlation < 0.7 (diversification working)
- ✅ Trade frequency matches expectations (±50%)
- ✅ Psychological comfort maintained

**Deliverables:**
- 2-week full capital report
- Portfolio performance analysis
- Risk metrics summary
- Deployment success declaration or adjustment plan
- Portfolio manager sign-off

---

### Week 7-8: Jan 20-Feb 2, 2026 - Ongoing Monitoring & Optimization

**Objective:** Establish steady-state operations, prepare for quarterly review

#### Week 7: Jan 20-26 - Steady-State Operations

**Transition to BAU (Business As Usual):**

**Daily Monitoring (Reduced Intensity):**
- [ ] 9:00 AM: Quick status check (signals, positions, PnL)
- [ ] Real-time alerts for critical events only (loss limits, API errors)
- [ ] 9:00 PM: End-of-day summary

**Weekly Tasks:**
- [ ] Monday: Review prior week performance
- [ ] Wednesday: Mid-week risk check
- [ ] Friday: Weekly summary report + rebalancing check

**Rebalancing Assessment:**
- [ ] Check allocation drift (target: 80% Conservative, 20% Aggressive)
- [ ] If drift > 10% (e.g., 85/15): Rebalance via position sizing adjustment
- [ ] If correlation > 0.8: Reduce Aggressive to 10%, investigate

---

#### Week 8: Jan 27-Feb 2 - Pre-Quarterly Review

**Prepare for Quarterly Review (Month 3):**
- [ ] Compile 8 weeks of performance data
- [ ] Full portfolio analysis:
  - Realized PF vs backtest PF (by strategy and portfolio)
  - Win rate vs backtest WR
  - Trade frequency vs expectations
  - Risk metrics (max drawdown, Sharpe, Sortino)
  - Correlation and diversification benefit
  - Execution quality (slippage, fees, timing)

**Archetype Recovery Check:**
- [ ] Review S1 V2 fix progress (should be complete by now)
- [ ] Review S4 fix progress (should be complete by now)
- [ ] Assess if archetypes ready for paper trading (Week 9+)

**Next Experiments Planning:**
- [ ] Review next experiments queue (from FINAL_DECISION_REPORT)
- [ ] Prioritize based on live trading learnings
- [ ] Prepare experiment plans for Month 3-4

---

#### Weekend Feb 1-2 - 8-Week Deployment Complete

**Saturday Comprehensive Review:**
- [ ] Full 8-week performance analysis
- [ ] Compare to deployment roadmap milestones
- [ ] Document lessons learned
- [ ] Identify optimizations for next cycle

**Sunday Planning:**
- [ ] Finalize quarterly review agenda (Month 3)
- [ ] Schedule archetype paper trading (if fixes successful)
- [ ] Plan next experiments (baseline enhancements, multi-asset, etc.)
- [ ] Update capital allocation strategy (if needed)

**Week 7-8 Success Criteria:**
- ✅ Steady-state operations established (minimal manual intervention)
- ✅ No risk limit breaches in 2 weeks
- ✅ Portfolio performance stable (PF not declining)
- ✅ Rebalancing process tested (if needed)
- ✅ Quarterly review prepared

**Deliverables:**
- 8-week deployment complete report
- Lessons learned document
- Quarterly review agenda
- Next phase roadmap (Week 9-20)

---

## WEEK 9-12: ARCHETYPE RECOVERY (PARALLEL TRACK)

**Objective:** Fix S1 V2 and S4 data pipelines while baselines run in production

### Week 9-10: Diagnosis + Fix Implementation

#### S1-LiquidityVacuum Recovery

**Week 9 Tasks:**
- [ ] Audit feature store: Verify capitulation_depth calculation
- [ ] Check confluence logic: Review 3-of-4 gate implementation
- [ ] Compare configs: S1 V2 production vs comparison config
- [ ] Review regime filter: Ensure risk_off/crisis regimes present in 2023 data
- [ ] Inspect raw OHLCV data quality for 2023 period

**Week 10 Tasks:**
- [ ] Fix identified issues (feature calculations, config mismatches)
- [ ] Restore S1 V2 production config parameters
- [ ] Add logging: Capture why signals blocked (which gate failed)
- [ ] Review position sizing: Ensure 2% max per trade
- [ ] Run unit tests on confluence logic

---

#### S4-FundingDivergence Recovery

**Week 9 Tasks:**
- [ ] Check funding rate data availability for 2022-2023
- [ ] Inspect feature store: Confirm funding_zscore column exists
- [ ] Review S4 config: Verify archetype enabled in baseline comparison
- [ ] Check regime routing: Ensure S4 allowed in bear market regime
- [ ] Identify data gaps or missing dependencies

**Week 10 Tasks:**
- [ ] Backfill funding data if missing (Binance/OKX historical API)
- [ ] Regenerate feature store with S4 runtime enrichment
- [ ] Enable S4 archetype in comparison config
- [ ] Add detection logging: Why S4 not firing (which threshold blocking)
- [ ] Run canary test on 2022-06 (known bear market period)

---

### Week 11: Re-test and Validation

#### S1-LiquidityVacuum Re-test

**Monday-Wednesday:**
- [ ] Re-run baseline comparison with fixed S1 V2 pipeline
- [ ] Test period: 2022-2024 (same as original comparison)
- [ ] Expected: Test PF > 1.4, Test trades >= 20

**Thursday-Friday:**
- [ ] Analyze re-test results
- [ ] Compare to historical validation (PF 1.4-1.8)
- [ ] If successful: Prepare for Week 12 decision
- [ ] If failed: Root cause analysis, escalate to architecture review

**Success Criteria:**
- ✅ Test PF > 1.4
- ✅ Test trades >= 20 (1 year sample)
- ✅ No train losses > $100
- ✅ Pipeline logs show clean feature calculation
- ✅ Signals match historical validation

---

#### S4-FundingDivergence Re-test

**Monday-Wednesday:**
- [ ] Re-run baseline comparison with fixed S4 pipeline
- [ ] Test period: 2022-2024 (same as original comparison)
- [ ] Expected: Train PF > 2.0, Train trades >= 5

**Thursday-Friday:**
- [ ] Analyze re-test results
- [ ] Compare to historical validation (PF 2.22-2.32)
- [ ] Expected: Test trades = 0-2 (2023 bull market, NORMAL for S4)
- [ ] Validate 2024 OOS: Should have 7+ trades, PF > 2.0

**Success Criteria:**
- ✅ Train PF >= 2.0 (2022 bear market)
- ✅ Train trades >= 5
- ✅ Test trades = 0-2 (2023 bull, expected low)
- ✅ 2024 OOS PF > 2.0, trades >= 5
- ✅ Funding data coverage > 95%

---

### Week 12: Decision and Next Steps

#### Monday-Tuesday: Decision Making

**For S1 V2:**

**If re-test SUCCESSFUL (PF > 1.4, trades >= 20):**
- ✅ Move to KEEP status
- Schedule Week 13-14 paper trading
- Prepare production config
- Plan capital allocation adjustment (reduce baselines to 60% + 20%, add S1 V2 20%)

**If re-test FAILED (PF < 1.4 or trades < 20):**
- ❌ Escalate to architecture review
- Consider major redesign (confluence logic, thresholds)
- Timeline: 4-8 weeks for redesign + re-validation

---

**For S4:**

**If re-test SUCCESSFUL (train PF > 2.0, OOS PF > 2.0):**
- ✅ Move to KEEP status
- Schedule Week 13-14 paper trading
- Prepare production config
- Plan capital allocation adjustment (reduce baselines, add S4 20-30%)

**If re-test FAILED (train PF < 2.0 or zero trades again):**
- ❌ Deep investigation required
- Potential issues: Funding data quality, threshold calibration, regime routing
- Timeline: 4-8 weeks for deep dive + fixes

---

#### Wednesday-Friday: Week 13+ Planning

**If Both Archetypes Fixed:**
- [ ] Plan phased paper trading (S1 V2 first, then S4)
- [ ] Prepare capital reallocation strategy
- [ ] Design multi-strategy portfolio monitoring
- [ ] Schedule correlation analysis (baselines + archetypes)

**If One Archetype Fixed:**
- [ ] Deploy fixed archetype to paper trading (Week 13-14)
- [ ] Continue working on failed archetype
- [ ] Adjust portfolio allocation (baselines + 1 archetype)

**If Both Archetypes Failed:**
- [ ] Continue baseline-only portfolio
- [ ] Deep dive investigation (architecture issues?)
- [ ] Consider alternative archetype approaches
- [ ] Re-evaluate archetype value proposition

---

## MONITORING AND CHECKPOINTS

### Daily Monitoring Checklist

**Every Trading Day (9:00 AM):**
- [ ] Check overnight signals (Conservative + Aggressive)
- [ ] Review open positions (entry price, unrealized PnL, stop levels)
- [ ] Validate data feed quality (OHLCV, funding rates, features)
- [ ] Check alert log (any critical alerts overnight?)
- [ ] Confirm API connectivity and health

**Every Trading Day (5:00 PM):**
- [ ] Review day's activity (signals, fills, PnL)
- [ ] Update trade journal (document each trade)
- [ ] Check risk limits (daily loss, exposure)
- [ ] Review execution quality (slippage, fees, timing)
- [ ] Prepare daily summary report

---

### Weekly Checkpoints

**Every Monday:**
- [ ] Review prior week performance (PF, WR, trade count)
- [ ] Compare to backtest expectations
- [ ] Check allocation drift (80/20 target)
- [ ] Review risk metrics (max drawdown, exposure)
- [ ] Plan week ahead (expected signals, regime check)

**Every Friday:**
- [ ] Compile weekly summary report
- [ ] Assess rebalancing needs (allocation drift > 10%?)
- [ ] Review correlation (Conservative vs Aggressive)
- [ ] Check data quality trends (any deterioration?)
- [ ] Update project status (share with team)

---

### Monthly Reviews

**End of Month 1 (Week 4):**
- [ ] Full performance analysis (1 month live data)
- [ ] Compare to backtest expectations
- [ ] Assess risk-adjusted returns (Sharpe, Sortino)
- [ ] Review execution quality trends
- [ ] Identify optimizations

**End of Month 2 (Week 8):**
- [ ] Comprehensive 2-month review
- [ ] Prepare quarterly review agenda
- [ ] Archetype recovery status check
- [ ] Next experiments prioritization
- [ ] Capital allocation strategy review

**End of Month 3 (Week 12):**
- [ ] QUARTERLY REVIEW (full analysis)
- [ ] Walk-forward validation on latest data
- [ ] Regime analysis (are we in backtest-like conditions?)
- [ ] Consider parameter adjustments
- [ ] Plan next quarter experiments

---

## RISK MANAGEMENT FRAMEWORK

### Position Limits

**Per-Strategy Limits:**
- Conservative max position: $1,600 (2% of $80K)
- Aggressive max position: $300 (1.5% of $20K)
- Max concurrent positions per strategy: 2
- Max total portfolio positions: 3

**Portfolio Limits:**
- Max portfolio exposure: 6% ($6,000 across all positions)
- Max single position: $1,600 (Conservative)
- Max correlation: 0.7 (between strategies)

---

### Loss Limits

**Daily Limits:**
- Daily max loss: -$2,000 (-2% of portfolio)
- Action: Halt trading for day, investigate

**Weekly Limits:**
- Weekly max loss: -$5,000 (-5% of portfolio)
- Action: Reduce position sizes 50%, investigate

**Monthly Limits:**
- Monthly max loss: -$10,000 (-10% of portfolio)
- Action: Reduce to 50% capital, full review required

---

### Circuit Breakers

**Automatic Trading Halt Triggers:**
1. Daily loss exceeds -$2,000
2. Single position loss > -$1,000 (position sizing error)
3. Data feed disruption > 15 minutes
4. API errors > 5 in 1 hour
5. Manual halt (psychological discomfort)

**Resume Trading Requirements:**
- Root cause identified and documented
- Fix implemented (if technical issue)
- Risk manager approval
- Monitoring enhanced (if needed)

---

## CONTINGENCY PLANS

### Scenario 1: Paper Trading Underperforms

**Trigger:** Paper PF < 50% of backtest PF after 2 weeks

**Actions:**
1. Halt deployment immediately
2. Compare paper signals to backtest signals (data quality issue?)
3. Review execution quality (slippage, fees, timing)
4. Check for regime mismatch (market conditions changed?)
5. Fix identified issues
6. Restart paper trading from Week 1

**Timeline:** +2-4 weeks delay

---

### Scenario 2: Live Trading Underperforms

**Trigger:** Live PF < 50% of paper PF after 2 weeks (Week 3-4)

**Actions:**
1. Reduce to paper trading immediately
2. Investigate execution differences (live vs paper)
3. Review exchange API (order types, fees, slippage)
4. Consider third-party execution audit
5. Fix execution issues
6. Restart live deployment from Week 3 (10% capital)

**Timeline:** +2-4 weeks delay

---

### Scenario 3: Risk Limit Breach

**Trigger:** Daily/weekly/monthly loss limit hit

**Actions:**
1. Halt trading automatically (circuit breaker)
2. Review all open positions (close if needed)
3. Root cause analysis (strategy failure? execution error? market anomaly?)
4. Document incident in risk log
5. Adjust position sizes if needed
6. Risk manager approval to resume

**Timeline:** 1-3 days halt

---

### Scenario 4: Data Feed Failure

**Trigger:** OHLCV or feature data unavailable > 15 minutes

**Actions:**
1. Halt trading automatically
2. Switch to backup data feed (if available)
3. Contact data provider (Binance/OKX/custom)
4. Review impact on open positions
5. Resume trading only when data quality confirmed

**Timeline:** 15 minutes to 24 hours (depends on issue)

---

### Scenario 5: Archetype Fixes Fail

**Trigger:** Week 11 re-test shows S1/S4 still broken (PF < target or zero trades)

**Actions:**
1. Continue baseline-only portfolio (already in production)
2. Escalate to architecture review (deep dive needed)
3. Consider alternative archetype approaches
4. Re-evaluate archetype value proposition (worth the effort?)
5. Focus on baseline enhancements instead (Priority 2 experiments)

**Timeline:** 4-8 weeks for deep investigation

---

## SUCCESS METRICS

### Week 2 (Paper Trading)
- ✅ Conservative paper PF >= 2.5 (80% of 3.17)
- ✅ Aggressive paper PF >= 1.7 (80% of 2.10)
- ✅ Data feed reliability > 99%
- ✅ Trade frequency within ±50% of expectations

### Week 4 (Live 10% Capital)
- ✅ Live PF >= 80% of paper PF
- ✅ No risk limit breaches
- ✅ Execution quality: slippage < 10 bps
- ✅ Psychological comfort confirmed

### Week 6 (Full Capital)
- ✅ Portfolio PF >= 2.1 (70% of expected 2.96)
- ✅ Correlation < 0.7
- ✅ Diversification benefit: 15-20% variance reduction
- ✅ Trade frequency matches expectations (±50%)

### Week 8 (Deployment Complete)
- ✅ 8 weeks stable operations
- ✅ Portfolio PF stable (not declining)
- ✅ Rebalancing process validated
- ✅ Quarterly review prepared

### Week 12 (Archetype Recovery)
- ✅ S1 V2 re-test: PF > 1.4, trades >= 20 (if successful)
- ✅ S4 re-test: Train PF > 2.0, OOS PF > 2.0 (if successful)
- ✅ At least one archetype ready for paper trading (Week 13+)

---

## DELIVERABLES BY WEEK

### Week 1
- [ ] Paper trading infrastructure deployed
- [ ] Week 1 summary report
- [ ] Go/No-Go decision for Week 2

### Week 2
- [ ] 2-week paper trading report
- [ ] Paper vs backtest comparison
- [ ] Go/No-Go decision for live deployment

### Week 4
- [ ] 2-week live (10% capital) report
- [ ] Live vs paper comparison
- [ ] Go/No-Go decision for full capital

### Week 6
- [ ] 2-week full capital report
- [ ] Portfolio performance analysis
- [ ] Deployment success declaration

### Week 8
- [ ] 8-week deployment complete report
- [ ] Lessons learned
- [ ] Quarterly review agenda

### Week 12
- [ ] S1 V2 re-test report
- [ ] S4 re-test report
- [ ] Archetype decision (KEEP/IMPROVE/KILL)
- [ ] Week 13+ roadmap

---

## APPENDIX

### Key Personnel

**Risk Manager:**
- Approves capital deployment
- Reviews risk limits and breaches
- Signs off on rebalancing

**Data Team:**
- Ensures data feed reliability
- Fixes feature pipeline issues
- Supports archetype recovery

**Trading Ops:**
- Manages paper/live trading infrastructure
- Handles execution quality monitoring
- Responds to circuit breaker events

**Portfolio Manager:**
- Approves capital allocation
- Reviews monthly performance
- Makes rebalancing decisions

**Quant Team:**
- Analyzes performance vs backtest
- Investigates strategy failures
- Plans next experiments

---

### Contact Information

**Critical Alerts (24/7):**
- Daily loss > -$500
- Data feed down > 15 min
- API errors spiking
- Manual circuit breaker needed

**Non-Critical (Business Hours):**
- Weekly summary reports
- Performance reviews
- Rebalancing decisions

---

### Version History

**v1.0 (2025-12-07):**
- Initial deployment roadmap
- 8-week baseline deployment plan
- Week 9-12 archetype recovery plan
- Risk management framework
- Contingency plans

---

**DEPLOYMENT ROADMAP STATUS: APPROVED ✅**

**Next Action:** Begin Week 1 execution (Dec 9, 2025)

**Sign-off Required:**
- [ ] Risk Manager
- [ ] Trading Ops
- [ ] Portfolio Manager
- [ ] Data Team

**Once all sign-offs received, proceed to Week 1 execution.**
