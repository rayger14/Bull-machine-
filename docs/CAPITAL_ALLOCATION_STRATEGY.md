# Capital Allocation Strategy: Dual-System Framework

**Version:** 1.0.0
**Date:** 2025-12-03
**Status:** ACTIVE FRAMEWORK
**Applies To:** System B0 + System ARCHETYPES (S4/S5/S1)

---

## Executive Summary

This document defines how to split capital between two independent trading systems with different performance characteristics and regime dependencies.

**Core Principle:** Allocate based on risk-adjusted performance, regime suitability, and diversification benefits.

**Key Insight:** B0 and archetypes are NOT perfect substitutes - they have different strengths and should run in parallel with dynamic rebalancing.

---

## System Performance Characteristics

### System B0 (Baseline-Conservative)

```
Performance:
  Test PF:           3.17
  Win Rate:          42.9%
  Trades/Year:       7
  Max Drawdown:      TBD
  Overfit:           -1.89 (excellent)

Characteristics:
  Regime:            All-weather (works in bull, bear, neutral)
  Frequency:         Low (7 trades/year)
  Complexity:        Low (easy to debug)
  Maintenance:       Low
  Feature Deps:      Minimal (OHLCV, ATR)
  Risk Profile:      Conservative (waits for -15% dips)

Strengths:
  + Proven performance (PF 3.17)
  + Regime-agnostic (always active)
  + Simple logic (low operational risk)
  + Excellent generalization (negative overfit)

Weaknesses:
  - Low frequency (may miss opportunities)
  - No pattern recognition (leaves alpha on table)
  - No regime optimization
```

### System ARCHETYPES (S4/S5/S1)

```
Performance:
  S4 (Funding Divergence):
    Train PF:        2.22 (2022 bear)
    OOS PF:          2.32 (2024 volatile)
    Win Rate:        55.7% (train), 42.9% (OOS)
    Trades/Year:     12-14 (in active regimes)
    Regime:          Bear/Volatile specialist

  S5 (Long Squeeze):
    Train PF:        1.86 (2022)
    Win Rate:        ~60%
    Trades/Year:     9
    Regime:          Risk_on/Crisis specialist

  S1 (Liquidity Vacuum):
    Status:          Not yet validated (regime blocking)
    Expected PF:     >2.5
    Regime:          Crisis/Risk_off specialist

Characteristics:
  Regime:            Specialists (regime-dependent)
  Frequency:         Medium-High (12-14/year per archetype)
  Complexity:        High (hard to debug)
  Maintenance:       High
  Feature Deps:      Heavy (80+ features)
  Risk Profile:      Aggressive (pattern-specific)

Strengths:
  + Higher frequency (12-14 trades/archetype)
  + Regime-optimized (excel in target conditions)
  + Pattern recognition (catches specialized opportunities)
  + Diversification (different signal sources than B0)

Weaknesses:
  - Regime-dependent (idle in wrong conditions)
  - Complex maintenance
  - Lower PF than B0 (2.2/1.86 < 3.17)
  - High feature dependencies
```

---

## Risk Budgeting Framework

### Total Capital Definition

```
Total Capital:     $100,000 (example)

Risk Budget:       5% max drawdown per system
                   10% total portfolio max drawdown

Position Sizing:
  B0:              15% of allocated capital per trade
  Archetypes:      10% of allocated capital per trade (higher frequency)

Leverage:          1.0x (no leverage)
```

### Risk Metrics

**System B0:**
- Trades/year: 7
- Expected annual drawdown: ~8-12% (estimated from -15% entry)
- Sharpe Ratio: TBD (need more data)
- Correlation with BTC: High (long-only)

**System ARCHETYPES:**
- Trades/year: 12-14 per archetype (S4/S5 active)
- Expected annual drawdown: ~10-15% (more frequent entries)
- Sharpe Ratio: TBD (need more data)
- Correlation with BTC: High (long-only)
- Correlation with B0: Medium (different entry logic)

**Portfolio-Level:**
- Expected correlation B0 vs Archetypes: 0.4-0.6 (different signals)
- Diversification benefit: 10-20% DD reduction (estimated)

---

## Allocation Scenarios

### Scenario 1: Conservative (70% B0, 30% Archetypes)

```
Allocation:
  B0:              $70,000
  Archetypes:      $30,000
    - S4:          $15,000 (bear/volatile specialist)
    - S5:          $10,000 (risk_on specialist)
    - S1:          $5,000  (crisis specialist, conservative due to unknown performance)

Rationale:
  - Proven performer (B0) gets majority of capital
  - Archetypes provide upside with limited exposure
  - Conservative given archetype validation incomplete

Expected Performance:
  Portfolio PF:    ~2.8-3.0 (weighted avg)
  Portfolio WR:    ~45-50%
  Trades/Year:     ~12-15 total
  Max Drawdown:    ~8-10%

Risk Profile:
  - Low risk (majority in proven B0)
  - Limited downside from archetype experimentation
  - Good for initial deployment

When to Use:
  - Paper trading phase
  - First 1-3 months of live trading
  - Risk-averse capital
  - Validation period for archetypes
```

### Scenario 2: Balanced (50% B0, 50% Archetypes)

```
Allocation:
  B0:              $50,000
  Archetypes:      $50,000
    - S4:          $20,000 (bear/volatile specialist)
    - S5:          $20,000 (risk_on specialist)
    - S1:          $10,000 (crisis specialist)

Rationale:
  - Equal weight to proven baseline and specialized patterns
  - Maximizes diversification benefit
  - Assumes archetypes validated in paper trading

Expected Performance:
  Portfolio PF:    ~2.5-2.7
  Portfolio WR:    ~48-52%
  Trades/Year:     ~20-25 total
  Max Drawdown:    ~10-12%

Risk Profile:
  - Moderate risk (balanced exposure)
  - Higher frequency than conservative
  - Regime diversification (B0 all-weather, archetypes specialists)

When to Use:
  - After successful paper trading (1-3 months)
  - Moderate risk tolerance
  - Seeking regime diversification
  - Confidence in archetype performance
```

### Scenario 3: Aggressive (30% B0, 70% Archetypes)

```
Allocation:
  B0:              $30,000
  Archetypes:      $70,000
    - S4:          $25,000 (bear/volatile specialist)
    - S5:          $25,000 (risk_on specialist)
    - S1:          $20,000 (crisis specialist)

Rationale:
  - Maximize upside from pattern recognition
  - Higher trade frequency
  - Assumes archetypes consistently outperform in live trading

Expected Performance:
  Portfolio PF:    ~2.2-2.5
  Portfolio WR:    ~50-55%
  Trades/Year:     ~30-40 total
  Max Drawdown:    ~12-15%

Risk Profile:
  - High risk (majority in complex archetypes)
  - Higher frequency (more execution risk)
  - Regime-dependent (performance varies with market conditions)

When to Use:
  - After 6+ months of validated live performance
  - Archetypes consistently beat B0 in live conditions
  - High risk tolerance
  - Confidence in maintenance capabilities

Warning:
  - Higher complexity = higher operational risk
  - Performance may degrade if market dynamics change
  - Requires robust monitoring
```

---

## Dynamic Rebalancing Rules

### Performance-Based Adjustments

**Rule 1: Monthly Performance Review**

```python
# Pseudocode for rebalancing logic

monthly_review():
    pf_b0 = calculate_pf(system='B0', period='last_30d')
    pf_arch = calculate_pf(system='Archetypes', period='last_30d')

    # Rebalance if performance diverges significantly
    if pf_b0 > pf_arch * 1.5:
        # B0 significantly outperforming
        increase_b0_allocation(by=10%)
        decrease_arch_allocation(by=10%)
        log("Rebalanced: B0 outperforming")

    elif pf_arch > pf_b0 * 1.3:
        # Archetypes outperforming
        increase_arch_allocation(by=10%)
        decrease_b0_allocation(by=10%)
        log("Rebalanced: Archetypes outperforming")

    else:
        # Performance similar, maintain allocation
        log("No rebalance needed")
```

**Rule 2: Drawdown Protection**

```python
drawdown_check():
    dd_b0 = calculate_drawdown(system='B0', period='current')
    dd_arch = calculate_drawdown(system='Archetypes', period='current')

    # Reduce allocation if drawdown exceeds threshold
    if dd_b0 > 0.15:  # 15% drawdown
        reduce_b0_allocation(by=20%)
        log("Alert: B0 drawdown exceeded 15%, reducing allocation")

    if dd_arch > 0.20:  # 20% drawdown
        reduce_arch_allocation(by=20%)
        log("Alert: Archetype drawdown exceeded 20%, reducing allocation")

    # Kill switch: pause system if catastrophic loss
    if dd_b0 > 0.30 or dd_arch > 0.30:
        pause_system()
        alert_operator()
        log("CRITICAL: System paused due to excessive drawdown")
```

**Rule 3: Regime-Based Adjustment**

```python
regime_adjustment():
    current_regime = get_current_regime()  # risk_on/neutral/risk_off/crisis

    if current_regime == 'crisis':
        # Favor S1 (crisis specialist)
        arch_weights = {'S1': 0.50, 'S4': 0.30, 'S5': 0.20}

    elif current_regime == 'risk_off':
        # Favor S4 (bear specialist)
        arch_weights = {'S4': 0.50, 'S1': 0.30, 'S5': 0.20}

    elif current_regime == 'risk_on':
        # Favor S5 (bull pullback specialist)
        arch_weights = {'S5': 0.50, 'S4': 0.30, 'S1': 0.20}

    else:  # neutral
        # Equal weight
        arch_weights = {'S4': 0.40, 'S5': 0.40, 'S1': 0.20}

    apply_archetype_weights(arch_weights)
    log(f"Regime adjustment: {current_regime}, weights: {arch_weights}")
```

### Rebalancing Frequency

| Trigger | Frequency | Action |
|---------|-----------|--------|
| **Routine Review** | Monthly | Check performance, adjust by ±10% |
| **Drawdown Threshold** | Real-time | Reduce allocation by 20%, alert operator |
| **Regime Change** | Daily | Adjust archetype sub-allocation |
| **Quarterly Review** | Quarterly | Full strategy reassessment, major rebalancing allowed |
| **Kill Switch** | Real-time | Pause system if >30% DD |

---

## Conflict Resolution

### What if Both Systems Signal at Once?

**Case 1: B0 and S4 both signal long entry**
```
Action:
  - Take BOTH positions (systems are independent)
  - B0 position: 15% of B0 allocation
  - S4 position: 10% of S4 allocation
  - Total exposure: ~12.5% of total capital (if 50/50 allocation)

Rationale:
  - Different signal sources (drawdown vs funding divergence)
  - Diversification benefit
  - Both systems independent

Risk Management:
  - Ensure total exposure < 25% of capital
  - If combined exposure > 25%, reduce both position sizes proportionally
```

**Case 2: B0 signals entry, but S4 already has open position**
```
Action:
  - Take B0 position (independent systems)
  - Monitor combined exposure

Risk Check:
  - If total exposure > 25%, skip B0 entry
  - Log: "B0 signal skipped due to exposure limit"
```

**Case 3: Multiple archetypes signal at once (S4 + S5 + S1)**
```
Action:
  - Prioritize by regime fit:
    - Crisis: S1 > S4 > S5
    - Risk_off: S4 > S1 > S5
    - Risk_on: S5 > S4 > S1
  - Take top 2 signals (avoid over-concentration)

Position Sizing:
  - Split archetype allocation: 60% to primary, 40% to secondary

Example:
  - Regime = risk_off
  - S4 and S1 both signal
  - S4 gets 60% of archetype allocation
  - S1 gets 40% of archetype allocation
```

### Exit Priority

**Case 1: B0 hits stop loss, but S4 still profitable**
```
Action:
  - Exit B0 position (independent stops)
  - Keep S4 position open
  - Re-evaluate total exposure
```

**Case 2: S4 hits take profit, but B0 still holding**
```
Action:
  - Exit S4 position
  - Keep B0 position open
  - Systems operate independently
```

**Case 3: Both systems in drawdown**
```
Action:
  - Exit both if combined DD > 20%
  - Pause new entries until DD recovers
  - Review allocation (may have over-allocated to correlated strategies)
```

---

## Risk Limits

### Per-System Limits

```
System B0:
  Max position size:         15% of B0 allocation per trade
  Max concurrent trades:     1 (by design, low frequency)
  Max drawdown:              15% (alert), 25% (kill switch)
  Stop loss:                 2.5x ATR
  Position hold time:        Until TP or SL

System Archetypes:
  Max position size:         10% of archetype allocation per trade
  Max concurrent trades:     3 (one per S1/S4/S5)
  Max drawdown:              20% (alert), 30% (kill switch)
  Stop loss:                 Varies by archetype (1.5-3.0x ATR)
  Position hold time:        Until TP or SL or cooldown
```

### Portfolio-Level Limits

```
Total Exposure Limit:      25% of total capital at any time
Max Drawdown (Portfolio):  15% (alert), 25% (kill switch)
Max Leverage:              1.0x (no leverage)
Correlation Threshold:     If B0 vs Archetypes correlation > 0.8 for 30 days,
                           reduce allocation to one system (too correlated)
```

---

## Example Allocations (Capital: $100k)

### Month 1-3: Paper Trading Phase (Conservative)

```
Total Capital:     $100,000 (paper money)

B0 Allocation:     $70,000 (70%)
  - Max position:  $10,500 per trade (15% of $70k)
  - Expected trades: 1-2 in 3 months

Archetype Allocation: $30,000 (30%)
  - S4: $15,000
    - Max position: $1,500 per trade (10% of $15k)
    - Expected trades: 3-4 in 3 months
  - S5: $10,000
    - Max position: $1,000 per trade (10% of $10k)
    - Expected trades: 2-3 in 3 months
  - S1: $5,000
    - Max position: $500 per trade (10% of $5k)
    - Expected trades: 0-1 in 3 months (crisis is rare)

Expected Total Trades: 6-10 in 3 months
Expected Portfolio PF: ~2.8-3.0
Expected Max Exposure: ~15-20% of capital
```

### Month 4-6: Early Live Phase (Balanced)

```
Total Capital:     $100,000 (live money)

B0 Allocation:     $50,000 (50%)
  - Max position:  $7,500 per trade
  - Expected trades: 1-2 in 3 months

Archetype Allocation: $50,000 (50%)
  - S4: $20,000
    - Max position: $2,000 per trade
    - Expected trades: 3-4 in 3 months
  - S5: $20,000
    - Max position: $2,000 per trade
    - Expected trades: 2-3 in 3 months
  - S1: $10,000
    - Max position: $1,000 per trade
    - Expected trades: 0-1 in 3 months

Expected Total Trades: 6-10 in 3 months
Expected Portfolio PF: ~2.5-2.7
Expected Max Exposure: ~18-22% of capital

Rebalancing:
  - Review monthly
  - Adjust ±10% based on performance
```

### Month 7+: Mature Phase (Performance-Dependent)

**If B0 outperforms:**
```
B0 Allocation:     $70,000 (70%)
Archetype Allocation: $30,000 (30%)
  - Increase B0 due to proven superiority
```

**If Archetypes outperform:**
```
B0 Allocation:     $30,000 (30%)
Archetype Allocation: $70,000 (70%)
  - Increase archetypes due to live validation
```

**If performance similar:**
```
B0 Allocation:     $50,000 (50%)
Archetype Allocation: $50,000 (50%)
  - Maintain balanced allocation (diversification)
```

---

## Performance-Based Decision Tree

```
After 3 months of live trading:

                    ┌─────────────────────┐
                    │  Evaluate Performance│
                    └──────────┬───────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │ B0 PF > Archetypes PF * 1.5?  │
               └───────────┬───────────────────┘
                           │
          ┌────────────────┴────────────────┐
          │                                 │
          ▼ YES                             ▼ NO
   ┌─────────────────┐           ┌─────────────────────┐
   │ Increase B0 to  │           │ Archetypes PF >     │
   │ 70% allocation  │           │ B0 PF * 1.3?        │
   └─────────────────┘           └──────────┬──────────┘
                                            │
                               ┌────────────┴────────────┐
                               │                         │
                               ▼ YES                     ▼ NO
                     ┌──────────────────┐       ┌────────────────┐
                     │ Increase Archetypes│     │ Maintain 50/50 │
                     │ to 70% allocation  │     │ (diversification)│
                     └──────────────────┘       └────────────────┘
```

---

## Monitoring Metrics

### Daily Metrics (Automated)

```
System B0:
  - Open positions
  - Current PnL
  - Drawdown (current)
  - Signals fired today
  - Stop loss breaches

System Archetypes:
  - Open positions (per archetype)
  - Current PnL (per archetype)
  - Drawdown (current, per archetype)
  - Signals fired today (per archetype)
  - Regime classification (current)

Portfolio:
  - Total exposure (% of capital)
  - Combined PnL
  - Portfolio drawdown
  - Correlation B0 vs Archetypes (30-day rolling)
```

### Weekly Metrics (Manual Review)

```
Performance:
  - Weekly PF (B0, Archetypes, Portfolio)
  - Weekly WR (B0, Archetypes, Portfolio)
  - Weekly trade count
  - Average trade duration

Risk:
  - Max drawdown this week
  - Stop losses hit (count)
  - Position size violations (if any)
  - Exposure limit breaches (if any)

Allocation:
  - Current allocation (B0 vs Archetypes)
  - Recommended adjustment (if any)
  - Rationale for adjustment
```

### Monthly Metrics (Rebalancing Decision)

```
Performance Analysis:
  - Monthly PF (by system)
  - Cumulative PF (since start)
  - Sharpe Ratio (rolling 90-day)
  - Max drawdown (monthly)
  - Recovery time from drawdowns

Allocation Decision:
  - Current: X% B0, Y% Archetypes
  - Recommended: New allocation based on performance
  - Rationale: Data-driven justification
  - Action: Rebalance or maintain

Risk Assessment:
  - Are risk limits being respected?
  - Any anomalies (unexpected correlations, regime failures)?
  - Operational issues (bugs, execution failures)?
```

---

## Implementation Checklist

### Setup (Before Deployment)

- [ ] Define total capital amount
- [ ] Choose initial allocation scenario (conservative/balanced/aggressive)
- [ ] Set per-system risk limits
- [ ] Set portfolio-level risk limits
- [ ] Implement conflict resolution logic
- [ ] Implement rebalancing logic
- [ ] Set up monitoring dashboard
- [ ] Define alert thresholds
- [ ] Test kill switch mechanism

### Paper Trading Phase (Month 1-3)

- [ ] Deploy with conservative allocation (70% B0, 30% Archetypes)
- [ ] Monitor daily performance
- [ ] Validate signal generation (both systems firing)
- [ ] Validate conflict resolution (when both signal)
- [ ] Measure correlation between systems
- [ ] Review weekly
- [ ] Document anomalies

### Transition to Live (Month 4)

- [ ] Review paper trading results
- [ ] Adjust allocation based on paper performance
- [ ] Deploy live with chosen allocation
- [ ] Monitor hourly for first week
- [ ] Monitor daily for first month
- [ ] Review weekly

### Ongoing Operations (Month 4+)

- [ ] Daily monitoring
- [ ] Weekly performance review
- [ ] Monthly rebalancing decision
- [ ] Quarterly strategy review
- [ ] Update allocation scenarios based on live data

---

## Risk Scenarios and Responses

### Scenario 1: Both Systems Lose Money for 30 Days

**Response:**
1. Reduce both allocations by 30%
2. Hold remaining 30% in cash
3. Investigate root cause:
   - Is market regime unusual?
   - Are signals broken?
   - Is execution failing?
4. Resume allocations only after root cause resolved

### Scenario 2: B0 Works, Archetypes Fail

**Response:**
1. Reduce archetype allocation to 20%
2. Increase B0 allocation to 80%
3. Investigate archetypes:
   - Regime gating issue?
   - Config too strict?
   - Feature store problem?
4. Fix issues before increasing archetype allocation

### Scenario 3: Archetypes Work, B0 Fails

**Response:**
1. Reduce B0 allocation to 20%
2. Increase archetype allocation to 80%
3. Investigate B0:
   - Is market regime unusual (no deep dips)?
   - Are exits too tight?
4. Keep B0 alive as safety net (don't go to 0%)

### Scenario 4: Extreme Correlation (Both Lose Together)

**Response:**
1. Systems are too correlated (not diversifying)
2. Reduce total allocation to 60% of capital
3. Hold 40% in cash
4. Investigate:
   - Are both systems long-only BTC? (Yes, this is expected)
   - Need to add short strategies or uncorrelated assets
5. Consider adding bear strategies or hedging

### Scenario 5: Flash Crash (BTC -30% in 1 hour)

**Response:**
1. Kill switch activates (if DD > 30%)
2. Close all positions immediately
3. Assess damage
4. Review why systems didn't protect:
   - Were stop losses too wide?
   - Was execution delayed?
5. Resume only after post-mortem and fixes

---

## Capital Growth Strategy

### Compounding Rule

```python
# Monthly compounding
def update_capital_allocation(current_capital, monthly_return):
    new_capital = current_capital * (1 + monthly_return)

    # Maintain allocation percentages
    if current_allocation == 'conservative':
        b0_capital = new_capital * 0.70
        arch_capital = new_capital * 0.30
    elif current_allocation == 'balanced':
        b0_capital = new_capital * 0.50
        arch_capital = new_capital * 0.50
    elif current_allocation == 'aggressive':
        b0_capital = new_capital * 0.30
        arch_capital = new_capital * 0.70

    return b0_capital, arch_capital

# Example:
# Start: $100k (50% B0, 50% Archetypes)
# Month 1: +5% return → $105k ($52.5k B0, $52.5k Archetypes)
# Month 2: +3% return → $108.15k ($54.08k B0, $54.08k Archetypes)
```

### Withdrawal Rule

```
Rule: Withdraw profits quarterly, keep base capital constant

Example:
  Base Capital:    $100,000
  After Q1:        $115,000 (+15%)
  Withdraw:        $15,000 (profit)
  Keep Trading:    $100,000 (base)

Rationale:
  - Protects profits (withdraw regularly)
  - Keeps risk constant (same base capital)
  - Prevents overconfidence (don't let winners run too long)
```

---

## Summary Table

| Allocation | B0 | Archetypes | Expected PF | Risk Level | Use Case |
|------------|-------|------------|-------------|------------|----------|
| **Conservative** | 70% | 30% | 2.8-3.0 | Low | Paper trading, validation |
| **Balanced** | 50% | 50% | 2.5-2.7 | Medium | Standard live trading |
| **Aggressive** | 30% | 70% | 2.2-2.5 | High | After archetypes proven |

---

## Key Takeaways

1. **Start conservative:** 70% B0, 30% Archetypes
2. **Rebalance monthly:** Based on performance
3. **Respect risk limits:** 15% system DD → alert, 25% DD → kill switch
4. **Monitor correlation:** If B0 vs Archetypes > 0.8, reduce one
5. **Regime awareness:** Adjust archetype sub-allocation by regime
6. **Conflict resolution:** Take both positions if independent signals
7. **Compound returns:** Maintain allocation percentages as capital grows
8. **Withdraw profits:** Quarterly to lock in gains

**Philosophy:** Let data guide allocation, not emotions.

---

**Document Owner:** Risk Management Team
**Last Updated:** 2025-12-03
**Next Review:** After 30 days of live trading
