# Hyperparameter Optimization Validation Checklist
## Quality Gates for Production Deployment

**Purpose:** Ensure optimized parameters are robust and won't overfit to historical data.
**Date Created:** 2026-01-16
**Review Frequency:** After each optimization run

---

## Pre-Optimization Checklist

### Data Quality
- [ ] **Data Coverage:** Full 7-year dataset (2018-2024) loaded
- [ ] **Missing Data:** < 1% missing bars
- [ ] **Regime Labels:** All bars have regime_label column
- [ ] **Regime Distribution:** All 4 regimes present (risk_on, risk_off, transition, crisis)
- [ ] **Feature Completeness:** All archetype features calculated (fusion, liquidity, volume_z, etc.)

### Configuration Review
- [ ] **Window Size:** 365-day train / 90-day test configured
- [ ] **Purge/Embargo:** 48h purge, 2% embargo enabled
- [ ] **Trial Count:** 150-250 trials per window
- [ ] **Search Space:** Reasonable parameter ranges defined
- [ ] **Objectives:** 4 objectives configured (Sortino, Calmar, MaxDD, Regime Diversity)

---

## Walk-Forward Validation (Per Window)

### Window Quality
- [ ] **Minimum Bars:** Train window has ≥ 500 regime-filtered bars
- [ ] **Test Coverage:** Test window has ≥ 100 regime-filtered bars
- [ ] **Regime Balance:** At least 2 allowed regimes present in train data
- [ ] **Temporal Integrity:** No data leakage between train/test

### Optimization Quality
- [ ] **Convergence:** Study converged (no improvement in last 30 trials)
- [ ] **Pareto Size:** At least 10 solutions on Pareto frontier
- [ ] **Parameter Stability:** Top 5 solutions have similar parameters (CV < 30%)
- [ ] **Constraint Satisfaction:** Best solution passes all constraints

### Performance Metrics (Train)
- [ ] **Minimum Trades:** ≥ 15 trades in train window
- [ ] **Profit Factor:** ≥ 1.3
- [ ] **Win Rate:** 35% - 75%
- [ ] **Sharpe Ratio:** ≥ 0.5
- [ ] **Max Drawdown:** ≤ 25%

### Performance Metrics (Test - OOS)
- [ ] **Minimum Trades:** ≥ 5 trades in test window
- [ ] **Profit Factor:** ≥ 1.2
- [ ] **Win Rate:** 30% - 80%
- [ ] **Sharpe Ratio:** ≥ 0.3
- [ ] **Max Drawdown:** ≤ 30%

### Overfitting Detection
- [ ] **Sharpe Degradation:** Test Sharpe ≥ 50% of Train Sharpe
- [ ] **PF Degradation:** Test PF ≥ 70% of Train PF
- [ ] **Win Rate Stability:** |Train WR - Test WR| ≤ 15%
- [ ] **Trade Frequency:** Test trades within 50-150% of train frequency

---

## Aggregate Walk-Forward Validation

### Cross-Window Consistency
- [ ] **Number of Windows:** ≥ 18 windows completed (7 years with quarterly rolling)
- [ ] **Valid Windows:** ≥ 80% of windows passed individual checks
- [ ] **OOS Consistency:** Correlation(train_sharpe, test_sharpe) ≥ 0.6
- [ ] **Performance Stability:** Std(test_sharpe) ≤ 0.4

### Aggregate Performance
- [ ] **Mean Test Sharpe:** ≥ 0.8 (mean-reversion) or ≥ 0.6 (trend-following)
- [ ] **Mean Test PF:** ≥ 1.4
- [ ] **Mean Test Win Rate:** ≥ 45%
- [ ] **Min Test Sharpe:** ≥ 0.0 (no catastrophic windows)
- [ ] **Total Trades:** ≥ 30 across all test windows

### Parameter Stability
- [ ] **Parameter Convergence:** Top parameters stable across last 6 windows
- [ ] **Parameter CV:** Coefficient of variation < 25% for critical params
- [ ] **No Drift:** Parameters not trending monotonically (sign of overfitting)

---

## Regime-Aware Validation

### Regime Coverage
- [ ] **Regimes Tested:** All allowed regimes appear in test windows
- [ ] **Minimum Regime Bars:** Each regime has ≥ 200 bars across all test windows
- [ ] **Balanced Testing:** No single regime > 70% of test data

### Regime Performance
- [ ] **Multi-Regime Success:** ≥ 3 regimes with Sharpe > 0.5
- [ ] **Regime Diversity:** Diversity score ≥ 0.6 (1 - Gini coefficient)
- [ ] **No Single Regime Dependency:** No regime accounts for > 60% of total returns
- [ ] **Crisis Performance:** If applicable, positive or neutral Sharpe during crisis regime

### Regime Transitions
- [ ] **Transition Testing:** Strategy tested on ≥ 5 regime transitions
- [ ] **Transition Sharpe:** Average Sharpe during transitions ≥ 0.0
- [ ] **Transition Drawdown:** Max DD during transitions ≤ 35%
- [ ] **No Whipsaw:** Win rate during transitions ≥ 30%

---

## CPCV Validation (Final Production Check)

### CPCV Configuration
- [ ] **Groups:** Data split into 6 chronological groups (2018, 2019, ..., 2024)
- [ ] **Test Groups:** 2 groups used as test in each combination
- [ ] **Combinations:** All C(6,2) = 15 combinations evaluated
- [ ] **Purge/Embargo:** Applied to all CPCV folds

### CPCV Performance Distribution
- [ ] **Mean Sharpe:** ≥ 1.0
- [ ] **Median Sharpe:** ≥ 0.9
- [ ] **5th Percentile Sharpe:** ≥ 0.5 (worst case still viable)
- [ ] **95th Percentile Sharpe:** ≥ 1.5 (best case not unrealistic)
- [ ] **Sharpe CV:** ≤ 0.5 (consistency across paths)

### Overfitting Metrics
- [ ] **Deflated Sharpe Ratio (DSR):** ≥ 0.95
- [ ] **Probability of Backtest Overfitting (PBO):** ≤ 0.5
- [ ] **Stochastic Dominance:** ≥ 70% of CPCV paths beat random
- [ ] **Consistency:** ≥ 80% of CPCV paths have positive Sharpe

---

## Stress Testing

### Historical Crisis Periods
- [ ] **2022 Crypto Winter:** Sharpe ≥ 0.3 or MaxDD ≤ 15%
- [ ] **2020 COVID Crash:** Sharpe ≥ 0.0 or MaxDD ≤ 25%
- [ ] **2018 Bear Market:** Sharpe ≥ 0.0 or positive PF

### Monte Carlo Simulation
- [ ] **Randomized Returns:** 1000 simulations with shuffled trades
- [ ] **95% CI Sharpe:** Lower bound ≥ 0.4
- [ ] **95% CI MaxDD:** Upper bound ≤ 30%
- [ ] **Ruin Probability:** P(total loss) < 1%

### Sensitivity Analysis
- [ ] **Parameter Perturbation:** ±10% change in params → Sharpe change ≤ 20%
- [ ] **Fee Sensitivity:** 2x fees → Sharpe remains positive
- [ ] **Slippage Sensitivity:** 5 bps slippage → PF ≥ 1.2
- [ ] **Position Size:** 50% size → Sharpe degrades < 15%

---

## Production Readiness

### Documentation
- [ ] **Parameters Documented:** Final parameters in JSON with rationale
- [ ] **Backtest Report:** Full walk-forward results documented
- [ ] **CPCV Report:** Distribution statistics documented
- [ ] **Risk Limits:** Max position size, daily loss limits defined

### Monitoring Setup
- [ ] **Live Metrics Dashboard:** Real-time Sharpe, DD, trade count
- [ ] **Alerting:** Email/SMS on drawdown > 10%, 0 signals > 7 days
- [ ] **Performance Tracking:** Daily comparison to backtest expectations
- [ ] **Regime Monitoring:** Current regime vs expected regime performance

### Risk Controls
- [ ] **Circuit Breaker:** Auto-disable if DD > 15%
- [ ] **Position Limits:** Max 10% capital per trade
- [ ] **Daily Loss Limit:** -3% daily equity stop
- [ ] **Correlation Limits:** Max 0.7 correlation with other archetypes

### Rollback Plan
- [ ] **Trigger Conditions:** Sharpe < 50% of backtest after 30 trades
- [ ] **Fallback Parameters:** Previous production parameters documented
- [ ] **Rollback SOP:** Step-by-step rollback procedure written
- [ ] **Notification:** Team notified of rollback triggers

---

## Archetype-Specific Criteria

### Mean-Reversion Archetypes (S1, S4, S5, S8)
- [ ] **Target Sharpe:** ≥ 1.2
- [ ] **Target Win Rate:** ≥ 55%
- [ ] **Target PF:** ≥ 1.6
- [ ] **Target MaxDD:** ≤ 15%
- [ ] **Target Trades/Year:** 8-20
- [ ] **Regime:** Perform best in risk_off/crisis

### Trend-Following Archetypes (A, G, H, K)
- [ ] **Target Sharpe:** ≥ 0.8
- [ ] **Target Win Rate:** ≥ 45%
- [ ] **Target PF:** ≥ 1.8
- [ ] **Target MaxDD:** ≤ 20%
- [ ] **Target Trades/Year:** 4-12
- [ ] **Regime:** Perform best in risk_on/transition

### Structure Archetypes (B, C, L)
- [ ] **Target Sharpe:** ≥ 1.0
- [ ] **Target Win Rate:** ≥ 50%
- [ ] **Target PF:** ≥ 1.5
- [ ] **Target MaxDD:** ≤ 18%
- [ ] **Target Trades/Year:** 6-15
- [ ] **Regime:** Perform across all regimes

---

## Red Flags: Immediate Rejection Criteria

### Critical Failures (Do Not Deploy)
- ❌ **OOS Consistency < 0.4:** Severe overfitting
- ❌ **PBO > 0.7:** Backtest overfitting highly likely
- ❌ **DSR < 0.5:** Performance not statistically significant
- ❌ **Worst CPCV Window Sharpe < -0.5:** Catastrophic failure mode exists
- ❌ **Single Regime > 80% Returns:** Over-optimized to one regime
- ❌ **Zero Trades in 2022:** Will fail in next crisis
- ❌ **Train/Test Sharpe Ratio > 3:** Extreme overfitting
- ❌ **Parameter Instability:** Top param CV > 50%

### Warning Signs (Investigate Further)
- ⚠️ **OOS Consistency 0.4-0.6:** Marginal generalization
- ⚠️ **PBO 0.5-0.7:** Moderate overfitting risk
- ⚠️ **Sharpe Degradation > 40%:** Significant OOS decay
- ⚠️ **Win Rate Swing > 20%:** Train/test behavioral change
- ⚠️ **Regime Diversity < 0.5:** Limited regime coverage
- ⚠️ **Trades < 25 over 7 years:** Insufficient sample size
- ⚠️ **MaxDD in test > 25%:** High drawdown risk

---

## Sign-Off

### Development Team
- [ ] **Quant Lead:** Reviewed optimization methodology
- [ ] **Backtest Engineer:** Verified walk-forward implementation
- [ ] **Risk Manager:** Approved risk limits and circuit breakers

### Final Approval
- [ ] **All Phase 1 Checks Passed:** Walk-forward validation complete
- [ ] **All Phase 2 Checks Passed:** CPCV validation complete
- [ ] **All Phase 3 Checks Passed:** Regime validation complete
- [ ] **All Phase 4 Checks Passed:** Stress testing complete
- [ ] **Production Readiness Confirmed:** Monitoring and controls in place

**Archetype:** ______________________

**Optimized Date:** ______________________

**Approved By:** ______________________

**Production Deploy Date:** ______________________

---

## Appendix: Metric Formulas

### OOS Consistency
```
OOS_Consistency = Pearson_Correlation(Train_Sharpes, Test_Sharpes)
```

### Deflated Sharpe Ratio
```
DSR = Observed_Sharpe / sqrt(1 + ((N_Trials - 1) / N_Trials) * (Sharpe_Std / Observed_Sharpe)^2)
```

### Probability of Backtest Overfitting
```
PBO = P(Test_Performance < Median(Train_Performance))
```

### Regime Diversity Score
```
Gini = (Sum of absolute differences) / (2 * N * Sum)
Diversity = 1 - Gini
```

### Coefficient of Variation
```
CV = Standard_Deviation / Mean
```

---

**Document Version:** 1.0
**Last Updated:** 2026-01-16
**Next Review:** After each optimization run
