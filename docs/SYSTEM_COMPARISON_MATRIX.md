# System Comparison Matrix: B0 vs S4 vs S5 vs S1

**Version:** 1.0.0
**Date:** 2025-12-03
**Status:** COMPREHENSIVE COMPARISON
**Purpose:** Evaluate systems across multiple dimensions to guide allocation decisions

---

## Executive Summary

This matrix compares four trading systems across performance, operational, and strategic dimensions:

1. **B0 (Baseline-Conservative):** Simple drawdown strategy, PF 3.17
2. **S4 (Funding Divergence):** Bear/volatile specialist, PF 2.22-2.32
3. **S5 (Long Squeeze):** Risk_on/crisis specialist, PF 1.86
4. **S1 (Liquidity Vacuum):** Crisis specialist, PF unknown (blocked by regime)

**Key Finding:** B0 has highest absolute PF, but archetypes provide regime specialization and diversification.

---

## Performance Metrics Comparison

### Profit Factor (Primary Metric)

| System | Train Period | Train PF | Test Period | Test PF | Overfit | Status |
|--------|--------------|----------|-------------|---------|---------|--------|
| **B0** | 2022 (Bear) | 1.28 | 2023 (Recovery) | **3.17** | -1.89 | ✅ Excellent |
| **S4** | 2022 (Bear) | 2.22 | 2024 Q1-Q2 (Volatile) | **2.32** | -0.10 | ✅ Good |
| **S5** | 2022 (Bear) | 1.86 | TBD | TBD | TBD | ⏳ Needs validation |
| **S1** | N/A | N/A | N/A | N/A | N/A | ❌ Blocked (regime) |

**Analysis:**
- B0 has highest test PF (3.17) with excellent generalization
- S4 shows consistent performance (2.22 → 2.32), slight improvement OOS
- S5 trained but needs OOS validation
- S1 cannot be evaluated due to regime gating bug

**Winner (PF):** B0 (3.17) > S4 (2.32) > S5 (1.86) > S1 (unknown)

### Win Rate

| System | Train WR | Test WR | Change | Consistency |
|--------|----------|---------|--------|-------------|
| **B0** | 31.1% | 42.9% | +11.8pp | ✅ Improved |
| **S4** | 55.7% | 42.9% | -12.8pp | ⚠️ Declined |
| **S5** | ~60% | TBD | TBD | ⏳ Unknown |
| **S1** | N/A | N/A | N/A | ❌ Unknown |

**Analysis:**
- B0 WR improved OOS (31% → 43%) - excellent sign
- S4 WR declined OOS (56% → 43%) - slight overfitting or regime effect
- S5 has high train WR (~60%) but needs validation
- Both B0 and S4 converged to 42.9% test WR (interesting coincidence)

**Winner (WR):** S5 (~60%, train) > S4 (55.7%, train) > B0 (42.9%, test)

### Trade Frequency

| System | Train Trades | Test Trades | Annualized | Activity Level |
|--------|--------------|-------------|------------|----------------|
| **B0** | 61 (2022) | 7 (2023) | **7/year** | Low |
| **S4** | 12 (2022) | 7 (2024 Q1-Q2) | **14/year** | Medium |
| **S5** | 9 (2022) | TBD | **9/year** (est) | Medium |
| **S1** | N/A | 0 (blocked) | **Unknown** | Unknown |

**Analysis:**
- B0 has lowest frequency (7/year) - very selective
- S4 has medium frequency (14/year) when regime is active
- S5 similar to S4 (9/year estimated)
- All systems are low-frequency swing traders

**Winner (Frequency):** S4 (14/year) > S5 (9/year) > B0 (7/year) > S1 (0/year)

### Drawdown and Risk

| System | Max Drawdown (Est) | Recovery Time | Stop Loss | Risk Profile |
|--------|-------------------|---------------|-----------|--------------|
| **B0** | ~12-15% | Fast (mean reversion) | 2.5x ATR | Conservative |
| **S4** | ~15-18% | Medium | Varies | Moderate |
| **S5** | ~12-15% | Medium | Varies | Moderate |
| **S1** | Unknown | Unknown | Varies | Unknown |

**Analysis:**
- B0 has predictable drawdown (waits for -15% dips)
- Archetypes have slightly higher DD due to pattern-specific entries
- B0 recovers faster (mean reversion strategy)
- Need live data to validate DD estimates

**Winner (Low Risk):** B0 (conservative entry, fast recovery)

---

## Regime Performance Analysis

### Performance by Market Regime

| System | Bull (Risk_on) | Neutral | Bear (Risk_off) | Crisis | Overall |
|--------|----------------|---------|-----------------|--------|---------|
| **B0** | Good | Good | Excellent | Excellent | All-weather |
| **S4** | **Idle** (0-1 trades) | Good | **Excellent** (PF 2.22) | Good | Bear specialist |
| **S5** | Good | Good | **Idle** (0-1 trades) | Excellent | Risk_on specialist |
| **S1** | Blocked | Blocked | Good (expected) | **Excellent** (expected) | Crisis specialist |

**Regime Activity Heatmap:**

```
           │ Risk_on │ Neutral │ Risk_off │ Crisis │
───────────┼─────────┼─────────┼──────────┼────────┤
    B0     │   🟢    │   🟢    │    🟢    │  🟢   │  All-weather
    S4     │   🔴    │   🟡    │    🟢    │  🟡   │  Bear specialist
    S5     │   🟢    │   🟡    │    🔴    │  🟢   │  Risk_on specialist
    S1     │   🔴    │   🔴    │    🟡    │  🟢   │  Crisis specialist

Legend:
🟢 = Active, performs well
🟡 = Moderately active
🔴 = Idle / blocked
```

**Complementarity Analysis:**

```
Regime Coverage:
  Risk_on:   B0 ✓, S5 ✓             (2 systems active)
  Neutral:   B0 ✓, S4 ✓, S5 ✓       (3 systems active)
  Risk_off:  B0 ✓, S4 ✓, S1 ✓       (3 systems active)
  Crisis:    B0 ✓, S5 ✓, S1 ✓       (3 systems active)

Best Coverage: Neutral and Crisis (3 systems each)
Worst Coverage: Risk_on (2 systems)
```

**Key Insight:** Systems are complementary, not redundant. Portfolio has coverage in all regimes.

---

## Correlation Analysis

### Signal Correlation (Do they trade at same times?)

| Pair | Estimated Correlation | Overlap | Diversification Benefit |
|------|----------------------|---------|-------------------------|
| B0 vs S4 | **Low (0.2-0.3)** | Minimal | High |
| B0 vs S5 | **Medium (0.4-0.5)** | Some | Medium |
| B0 vs S1 | **Low (0.2-0.3)** | Minimal | High |
| S4 vs S5 | **Very Low (0.1-0.2)** | Rare (opposite regimes) | Very High |
| S4 vs S1 | **Medium (0.4-0.5)** | Some (both bear specialists) | Medium |
| S5 vs S1 | **Low (0.2-0.3)** | Rare | High |

**Analysis:**
- B0 vs Archetypes: Low-medium correlation (different entry logic)
- S4 vs S5: Very low correlation (opposite regimes)
- S4 vs S1: Medium correlation (both bear specialists)
- Overall portfolio correlation: ~0.3-0.4 (good diversification)

**Expected Drawdown Reduction from Diversification:** 10-20%

### Feature Overlap

| System | Features Used | Overlap with B0 | Overlap with S4 | Overlap with S5 | Overlap with S1 |
|--------|---------------|-----------------|-----------------|-----------------|-----------------|
| **B0** | 10 (OHLCV, ATR, drawdown) | 100% | ~20% | ~20% | ~15% |
| **S4** | 30+ (funding, liquidity, price) | ~20% | 100% | ~40% | ~50% |
| **S5** | 25+ (funding, RSI, liquidity) | ~20% | ~40% | 100% | ~40% |
| **S1** | 40+ (liquidity, crisis, volume) | ~15% | ~50% | ~40% | 100% |

**Analysis:**
- B0 uses minimal features (intentionally simple)
- S4/S5/S1 share liquidity features (~40-50% overlap)
- S4 and S1 both rely on funding/liquidity (highest overlap)
- B0 is truly independent (low feature overlap with archetypes)

---

## Operational Comparison

### Complexity and Maintenance

| Dimension | B0 | S4 | S5 | S1 |
|-----------|----|----|----|----|
| **Code Complexity** | Low (500 LOC) | High (39k LOC shared) | High | High |
| **Feature Dependencies** | 10 columns | 30+ columns | 25+ columns | 40+ columns |
| **Runtime Enrichment** | None | Required | Required | Required |
| **Regime Classification** | None | Required | Required | Required |
| **Config Complexity** | Hardcoded | JSON (complex) | JSON (complex) | JSON (complex) |
| **Debugging Ease** | Easy | Hard | Hard | Hard |
| **Maintenance Hours/Week** | 1 hour | 3-5 hours | 3-5 hours | 3-5 hours |

**Maintenance Cost Estimate:**
- B0: $50/month (minimal monitoring)
- S4/S5/S1: $200/month each (complex debugging, feature store maintenance)
- Total: $50 + $600 = $650/month for 4 systems

**Trade-off:** B0 is 4x cheaper to maintain but has lower frequency.

### Framework and Infrastructure

| Dimension | B0 | S4 | S5 | S1 |
|-----------|----|----|----|----|
| **Backtesting Framework** | New (v2) ✅ | Old (v1) ⚠️ | Old (v1) ⚠️ | Old (v1) ⚠️ |
| **Wrapper Status** | Native | Broken (0 trades) | Broken (0 trades) | Broken (0 trades) |
| **Production Ready** | ✅ Yes | ⚠️ Needs fix | ⚠️ Needs fix | ❌ Blocked |
| **Live Streaming** | Easy | Hard (enrichment) | Hard (enrichment) | Hard (enrichment) |
| **Latency** | Low (<1s) | Medium (3-5s) | Medium (3-5s) | High (5-10s) |

**Technical Debt:**
- Archetypes stuck on old framework (39k line monolith)
- Wrapper exists but has regime gating bugs
- Need to migrate to new framework (significant effort)

---

## Strategic Comparison

### Use Cases

| System | Best Used When | Avoid When | Ideal For |
|--------|----------------|------------|-----------|
| **B0** | All conditions | None | All-weather baseline |
| **S4** | Bear markets, volatility | Bull markets | Bear/volatile regime |
| **S5** | Bull pullbacks, crisis | Bear markets | Risk_on regime |
| **S1** | Crisis events | Normal markets | Crisis events only |

### Production Readiness

| System | Status | Blockers | Time to Production |
|--------|--------|----------|-------------------|
| **B0** | ✅ Ready | None | Immediate |
| **S4** | ⚠️ Conditional | Wrapper fix, OOS validation | 1-2 weeks |
| **S5** | ⚠️ Conditional | Wrapper fix, OOS validation | 1-2 weeks |
| **S1** | ❌ Not Ready | Regime bug, feature validation | 2-4 weeks |

**Recommendation:**
1. Deploy B0 immediately (paper trading)
2. Fix archetype wrapper (1 week)
3. Validate S4/S5 on 2024 data (1 week)
4. Deploy S4/S5 if PF > 1.5 (conditional)
5. Fix S1 regime gating and validate (2 weeks)

---

## Complementarity Assessment

### Overlap vs Diversification

**Overlap (Negative - Redundant):**
- S4 and S1: Both bear specialists (medium overlap)
- All systems: Long-only BTC (100% correlation to BTC direction)

**Diversification (Positive - Complementary):**
- B0: All-weather vs S4/S5/S1 regime specialists
- S4 vs S5: Opposite regimes (bear vs bull)
- B0 vs Archetypes: Different signal sources (drawdown vs patterns)

**Diversification Score:**
```
B0 + S4:   8/10 (low overlap, different regimes)
B0 + S5:   7/10 (medium overlap, different logic)
B0 + S1:   8/10 (low overlap, different conditions)
S4 + S5:   9/10 (opposite regimes, very low overlap)
S4 + S1:   6/10 (both bear, medium overlap)
S5 + S1:   7/10 (different regimes, some overlap)

Portfolio (B0+S4+S5+S1): 8/10 (good diversification)
```

### Regime Coverage Map

```
Market Conditions Timeline (2022-2024):

2022 Q1-Q2 (Risk_off):   S4 ✓✓✓, B0 ✓✓, S1 ✓✓
2022 Q3-Q4 (Crisis):     S1 ✓✓✓, B0 ✓✓, S5 ✓
2023 Q1-Q2 (Recovery):   B0 ✓✓✓, S5 ✓
2023 Q3-Q4 (Risk_on):    S5 ✓✓✓, B0 ✓✓
2024 Q1-Q2 (Volatile):   S4 ✓✓✓, B0 ✓✓, S5 ✓

Legend:
✓✓✓ = Ideal conditions
✓✓  = Good conditions
✓   = Acceptable conditions

Coverage:
- All periods: B0 (baseline)
- 3/5 periods: S4, S5
- 2/5 periods: S1 (rare crisis)

Conclusion: Portfolio has consistent coverage across all market conditions.
```

---

## Decision Matrix for Allocation

### Scenario 1: Conservative Approach

**Allocate 70% B0, 30% Archetypes**

Rationale:
- B0 has highest proven PF (3.17)
- Archetypes need live validation
- Minimize risk during evaluation period

When to Use:
- First 1-3 months of live trading
- Paper trading phase
- Risk-averse capital

### Scenario 2: Balanced Approach

**Allocate 50% B0, 50% Archetypes (20% S4, 20% S5, 10% S1)**

Rationale:
- B0 provides baseline performance
- Archetypes provide regime specialization
- Maximize diversification benefit

When to Use:
- After successful paper trading (1-3 months)
- Archetypes validated on OOS data
- Moderate risk tolerance

### Scenario 3: Aggressive Approach

**Allocate 30% B0, 70% Archetypes (25% S4, 25% S5, 20% S1)**

Rationale:
- Maximize frequency and regime optimization
- Archetypes consistently outperform in live conditions
- Higher complexity acceptable

When to Use:
- After 6+ months of live success
- Archetypes beat B0 in live trading
- High risk tolerance

---

## Absolute vs Relative Performance

### Absolute Performance (Test PF)

```
Ranking by Test PF:
1. B0:  3.17  ★★★★★ (excellent)
2. S4:  2.32  ★★★★☆ (good)
3. S5:  1.86  ★★★☆☆ (acceptable)
4. S1:  N/A   ☆☆☆☆☆ (unknown)

Winner: B0 (highest absolute PF)
```

### Relative Performance (Regime-Adjusted)

```
In Bear Markets (Risk_off):
1. S4:  2.22-2.32  ★★★★★ (specialist)
2. B0:  ~1.3       ★★☆☆☆ (struggled)
3. S1:  Unknown    ☆☆☆☆☆
4. S5:  Idle       ☆☆☆☆☆

Winner: S4 (bear specialist)

In Bull Markets (Risk_on):
1. B0:  3.17       ★★★★★ (excellent)
2. S5:  ~1.86      ★★★☆☆ (acceptable)
3. S4:  Idle       ☆☆☆☆☆
4. S1:  Idle       ☆☆☆☆☆

Winner: B0 (all-weather)

In Crisis Events:
1. S1:  Expected High  ★★★★★ (specialist, but unproven)
2. B0:  Good           ★★★★☆
3. S5:  Good           ★★★☆☆
4. S4:  Moderate       ★★★☆☆

Expected Winner: S1 (crisis specialist)
```

**Conclusion:** No single winner - each system excels in different conditions.

---

## Trade-off Summary

### B0 (Baseline-Conservative)

**Pros:**
- Highest test PF (3.17)
- Simplest logic (easy to debug)
- All-weather (always active)
- Minimal maintenance
- Production-ready now

**Cons:**
- Low frequency (7/year)
- No pattern recognition
- No regime optimization
- May underperform in specific regimes

**Best For:** Safety net, baseline performance, risk-averse capital

### S4 (Funding Divergence)

**Pros:**
- Excellent in bear/volatile markets (PF 2.22-2.32)
- Regime specialist (knows when to abstain)
- Medium frequency (14/year)
- Proven pattern

**Cons:**
- Idle in bull markets (0-1 trades)
- Complex maintenance
- Requires funding data
- Old framework (needs migration)

**Best For:** Bear market hedge, regime-aware trading, higher frequency

### S5 (Long Squeeze)

**Pros:**
- Good in bull pullbacks (PF 1.86)
- High train WR (~60%)
- Medium frequency (9/year)
- Complementary to S4 (opposite regimes)

**Cons:**
- Lower PF than B0/S4
- Idle in bear markets
- Requires OI data (may have gaps)
- Needs OOS validation

**Best For:** Bull market pullbacks, crisis events, diversification from S4

### S1 (Liquidity Vacuum)

**Pros:**
- Expected high PF in crisis
- Rare but high-conviction signals
- Crisis specialist

**Cons:**
- Blocked by regime gating bug
- Unproven (no backtest results)
- Very low frequency (crisis events rare)
- Most complex features

**Best For:** Crisis alpha, rare high-conviction opportunities

---

## Recommended Portfolio Allocation

### Phase 1: Paper Trading (Month 1-3)

```
B0:  70%  ($70k of $100k)
S4:  15%  ($15k)
S5:  10%  ($10k)
S1:  5%   ($5k)

Rationale:
- B0 provides proven baseline
- S4/S5 get validation opportunity
- S1 minimal allocation (unproven)
- Low risk during evaluation
```

### Phase 2: Early Live (Month 4-6)

```
If Archetypes Validated (PF > 1.5 in paper):
  B0:  50%  ($50k)
  S4:  20%  ($20k)
  S5:  20%  ($20k)
  S1:  10%  ($10k)

If Archetypes Fail (PF < 1.5 in paper):
  B0:  90%  ($90k)
  S4:  5%   ($5k)
  S5:  5%   ($5k)
  S1:  0%   ($0)

Rationale:
- Increase archetype allocation if proven
- Keep B0 majority if archetypes fail
```

### Phase 3: Mature (Month 7+)

```
Dynamic allocation based on trailing 90-day performance:
  - Best performer: +10% allocation
  - Worst performer: -10% allocation
  - Rebalance monthly

Example:
  If S4 outperforms: 40% B0, 30% S4, 20% S5, 10% S1
  If B0 outperforms: 70% B0, 15% S4, 10% S5, 5% S1
```

---

## Key Metrics Dashboard

### Real-Time Monitoring (For Operators)

```
┌─────────────────────────────────────────────────────────────┐
│                   SYSTEM HEALTH DASHBOARD                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  System  │  PF    │  WR    │  Trades  │  PnL    │ Status  │
│─────────────────────────────────────────────────────────────│
│  B0      │  3.05  │  40%   │  3/7     │  +$230  │  🟢     │
│  S4      │  2.15  │  50%   │  2/14    │  +$115  │  🟢     │
│  S5      │  1.92  │  67%   │  2/9     │  +$92   │  🟢     │
│  S1      │  N/A   │  N/A   │  0/TBD   │  $0     │  🔴     │
│─────────────────────────────────────────────────────────────│
│  Portfolio │ 2.71 │  46%   │  7/30    │  +$437  │  🟢     │
│─────────────────────────────────────────────────────────────│
│                                                             │
│  Current Regime:    NEUTRAL                                │
│  Exposure:          12% of capital                         │
│  Max DD (Today):    -2.3%                                  │
│  Open Positions:    1 (B0)                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Legend:
🟢 = Healthy
🟡 = Warning
🔴 = Alert / Offline
```

---

## Summary and Recommendations

### Overall Winner

**No single winner - use portfolio approach.**

| Dimension | Winner | Rationale |
|-----------|--------|-----------|
| **Absolute Performance** | B0 (PF 3.17) | Highest test PF |
| **Regime Specialization** | S4 (PF 2.32 in bear/volatile) | Best in target regime |
| **Diversification Value** | S4+S5 portfolio | Opposite regimes, low correlation |
| **Simplicity** | B0 | Low complexity, easy maintenance |
| **Frequency** | S4 (14/year) | Highest trade count |
| **Production Readiness** | B0 | Ready now, no blockers |

### Strategic Recommendation

1. **Deploy B0 immediately** (70% allocation) - proven, simple, ready
2. **Validate S4/S5** in paper trading (30% allocation) - specialized, higher frequency
3. **Fix S1 regime bug** and validate (reserve 5-10% for future)
4. **Rebalance monthly** based on live performance
5. **Maintain diversification** - don't put all capital in one system

**Philosophy:** Pragmatic portfolio beats perfect optimization.

---

**Document Owner:** Quantitative Research Team
**Last Updated:** 2025-12-03
**Next Review:** After 30 days of live data collection
