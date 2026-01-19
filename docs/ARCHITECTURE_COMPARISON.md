# Architecture Comparison: Dual-System vs Alternatives

**Version:** 1.0.0
**Date:** 2025-12-04
**Purpose:** Compare dual-system architecture against alternative approaches
**Audience:** System architects, technical decision makers

---

## Architecture Options Compared

We evaluated three architectural approaches for production deployment:

1. **Single Best System** (Deploy only B0 or only best archetype)
2. **Merged Unified System** (Combine B0 + archetypes into one codebase)
3. **Dual-System with Router** (Current design - run both independently)

---

## Option 1: Single Best System

### Architecture

```
┌──────────────────────────────────────┐
│         SINGLE SYSTEM                │
│                                      │
│  Either:                             │
│  - B0 only (PF 3.17)                │
│  OR                                  │
│  - Best archetype only (S4 PF 2.32) │
│                                      │
│  100% capital allocated              │
│  No diversification                  │
│  Simple operations                   │
└──────────────────────────────────────┘
```

### Pros

- **Maximum Simplicity:** One codebase, one strategy, one set of configs
- **Lowest Maintenance:** ~4-6 hours/month (if B0) or 12-16 hours/month (if S4)
- **Easiest to Debug:** No cross-system complexity
- **Clear Performance Attribution:** No ambiguity about what's working

### Cons

- **Regime Risk:** S4 idles in bull markets, B0 underperforms in bear
- **No Diversification:** Single point of failure
- **Missed Opportunities:** B0 low frequency (7/year), S4 misses bull setups
- **Lower Risk-Adjusted Returns:** No correlation benefit

### Performance Estimate

**If B0 Only:**
```
Portfolio PF:     3.17 (bull), 1.28 (bear), 2.2 (average)
Portfolio WR:     42.9%
Trades per Year:  7
Max Drawdown:     12-15%
Sharpe Ratio:     ~1.3-1.5

Issue: Underperforms in bear markets
```

**If S4 Only:**
```
Portfolio PF:     2.32 (bear), 0 (bull idle), 1.2 (average if 50% bull)
Portfolio WR:     50%
Trades per Year:  14 (but only in bear/volatile)
Max Drawdown:     15-18%
Sharpe Ratio:     ~1.0-1.2

Issue: Misses 50% of time (bull markets)
```

### Decision

**Reject Option 1**

Reasons:
- Leaves performance on table (regime specialization unused)
- Higher drawdown (no diversification buffer)
- Misses opportunities (B0 low freq, S4 regime-gated)

---

## Option 2: Merged Unified System

### Architecture

```
┌────────────────────────────────────────────────────┐
│         UNIFIED MERGED SYSTEM                      │
│                                                    │
│  ┌──────────────────────────────────────────┐    │
│  │  Combined Entry Logic:                    │    │
│  │  - B0 drawdown trigger (simple)          │    │
│  │  - S4 funding divergence (complex)       │    │
│  │  - S5 long squeeze (complex)             │    │
│  │                                           │    │
│  │  Fusion: Weighted combination of all     │    │
│  │  signals with learned weights             │    │
│  └──────────────────────────────────────────┘    │
│                                                    │
│  Single framework (v2 new)                        │
│  Single backtest engine                           │
│  Single config file                               │
│  100% capital in one strategy                     │
└────────────────────────────────────────────────────┘
```

### Implementation Approaches

**Approach A: Port Archetypes to v2 Framework**

```python
class UnifiedStrategy(BaseModel):
    """
    Merged B0 + S4 + S5 logic in one model.
    """

    def generate_signal(self, ctx: RuntimeContext) -> Signal:
        # Calculate all signals
        b0_signal = self._b0_drawdown_logic(ctx)
        s4_signal = self._s4_funding_divergence_logic(ctx)
        s5_signal = self._s5_long_squeeze_logic(ctx)

        # Weighted combination
        combined_score = (
            b0_signal.confidence * 0.5 +
            s4_signal.confidence * 0.3 +
            s5_signal.confidence * 0.2
        )

        if combined_score > self.threshold:
            return Signal(direction='LONG', confidence=combined_score)
        else:
            return Signal(direction='NEUTRAL')
```

**Approach B: Ensemble with Learned Weights**

```python
class EnsembleStrategy(BaseModel):
    """
    ML-learned combination of B0, S4, S5 signals.
    """

    def __init__(self):
        self.b0_model = BuyHoldSellClassifier()
        self.s4_model = S4FundingDivergence()
        self.s5_model = S5LongSqueeze()
        self.ensemble_weights = self._load_ml_weights()  # Trained XGBoost

    def generate_signal(self, ctx: RuntimeContext) -> Signal:
        # Get all sub-signals
        b0_sig = self.b0_model.generate_signal(ctx)
        s4_sig = self.s4_model.generate_signal(ctx)
        s5_sig = self.s5_model.generate_signal(ctx)

        # ML ensemble decision
        feature_vec = [
            b0_sig.confidence,
            s4_sig.confidence,
            s5_sig.confidence,
            ctx.regime,  # risk_on/off/neutral
            ctx.atr_percentile
        ]

        ensemble_confidence = self.ensemble_weights.predict_proba(feature_vec)
        return Signal(direction='LONG', confidence=ensemble_confidence)
```

### Pros

- **Single Codebase:** Easier to maintain (one framework)
- **Optimal Signal Combination:** ML learns best weights
- **Unified Backtesting:** One engine, consistent metrics
- **Potentially Higher Returns:** If ensemble learns non-obvious patterns

### Cons

- **Complexity Explosion:** B0 simplicity lost (500 LOC → 40k LOC merged)
- **Debugging Nightmare:** When signals fail, unclear which component broke
- **Loss of Modularity:** Can't disable B0 or S4 independently
- **Validation Burden:** Entire merged system needs re-validation from scratch
- **Migration Risk:** 3+ months to port archetypes, high bug risk
- **Loses Proven Baselines:** B0 PF 3.17 validated on v2 framework - merging throws away this validation

### Performance Estimate

**Best Case (Ensemble Works):**
```
Portfolio PF:     2.8-3.2 (10-15% improvement over best single)
Portfolio WR:     55-60%
Trades per Year:  15-20
Max Drawdown:     10-12%
Sharpe Ratio:     1.8-2.0

If: ML ensemble successfully learns signal combination
```

**Worst Case (Ensemble Fails):**
```
Portfolio PF:     1.5-2.0 (worse than any single system)
Portfolio WR:     45-50%
Max Drawdown:     15-20%

If: Merged complexity introduces bugs, ML overfits, or signal conflicts cancel out
```

### Decision

**Defer Option 2 Until Future**

Reasons:
- High migration risk (3+ months, bugs likely)
- Loses B0 simplicity advantage (main selling point)
- Needs 200+ trades for ML ensemble (only have ~130)
- Can revisit after 12 months if dual-system proves limiting

Possible Future Trigger:
- After 12 months dual-system operation
- If maintenance overhead becomes unsustainable
- If ML ensemble shows >20% improvement in backtest

---

## Option 3: Dual-System with Router (CHOSEN)

### Architecture

```
┌────────────────────────────────────────────────────────────┐
│              DUAL-SYSTEM ARCHITECTURE                      │
└────────────────────────────────────────────────────────────┘

┌──────────────────────┐         ┌──────────────────────┐
│   SYSTEM B0          │         │   SYSTEM ARCHETYPES  │
│   (v2 Framework)     │         │   (v1 Framework)     │
├──────────────────────┤         ├──────────────────────┤
│                      │         │                      │
│ Simple drawdown      │         │ Pattern recognition  │
│ 500 LOC              │         │ 39k LOC              │
│ PF 3.17              │         │ PF 2.2-2.3           │
│ All-weather          │         │ Regime specialists   │
│ 7 trades/year        │         │ 12-14 trades/year    │
│                      │         │                      │
└──────────┬───────────┘         └──────────┬───────────┘
           │                                │
           └────────────┬───────────────────┘
                        ▼
              ┌───────────────────┐
              │  CAPITAL ROUTER   │
              │  (Allocation)     │
              ├───────────────────┤
              │                   │
              │ Performance-based │
              │ Regime-aware      │
              │ Correlation-adj   │
              │                   │
              │ Rebalance monthly │
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │   MARKET          │
              └───────────────────┘
```

### Pros

- **Modularity:** Systems operate independently (failure isolation)
- **Simplicity Preserved:** B0 stays simple (500 LOC), proven
- **Low Migration Risk:** No need to port archetypes to v2
- **Flexibility:** Can adjust allocation monthly, disable systems independently
- **Proven Baselines:** Both B0 (PF 3.17) and S4 (PF 2.32) validated
- **Optionality:** Can merge later if desired (Option 2), or add meta-layer (ML ensemble)
- **Diversification:** Low correlation (0.2-0.3) reduces drawdown 10-20%

### Cons

- **Duplicate Infrastructure:** Two backtesting frameworks (v2 for B0, v1 for archetypes)
- **Higher Maintenance:** Need to maintain both systems (20-30 hours/month vs 12-16)
- **Manual Allocation:** Capital router is algorithmic but not ML-optimized
- **Suboptimal Signal Combination:** No learned ensemble (just parallel operation)

### Performance Estimate

**Conservative Allocation (70% B0, 30% Arch):**
```
Portfolio PF:     2.8-3.0
Portfolio WR:     47-50%
Trades per Year:  10-15
Max Drawdown:     8-10%
Sharpe Ratio:     1.5-1.7

Expected: B0 dominates performance, archetypes provide diversification
```

**Balanced Allocation (50% B0, 50% Arch):**
```
Portfolio PF:     2.5-2.7
Portfolio WR:     50-52%
Trades per Year:  15-20
Max Drawdown:     10-12%
Sharpe Ratio:     1.6-1.8

Expected: Both systems contribute roughly equally
```

**Aggressive Allocation (30% B0, 70% Arch):**
```
Portfolio PF:     2.2-2.5
Portfolio WR:     52-55%
Trades per Year:  20-25
Max Drawdown:     12-15%
Sharpe Ratio:     1.4-1.6

Expected: Archetypes dominate, higher frequency, more regime-dependent
```

### Decision

**CHOOSE Option 3 (Dual-System)**

Reasons:
1. **Lowest Risk:** No migration needed, both systems already validated
2. **Pragmatic:** Keeps B0 simplicity, adds archetype specialization
3. **Flexible:** Can adjust allocation monthly based on performance
4. **Preserves Optionality:** Can merge later (Option 2) or add ML ensemble if data supports
5. **Proven:** B0 PF 3.17, S4 PF 2.32 - both validated independently

---

## Comparison Matrix

| Dimension | Single System | Merged System | Dual-System (CHOSEN) |
|-----------|---------------|---------------|----------------------|
| **Complexity** | Low | Very High | Medium |
| **Lines of Code** | 500 (B0) or 39k (S4) | ~45k merged | 40k total (separate) |
| **Maintenance Hours/Month** | 4-6 (B0) or 12-16 (S4) | 30-40 | 20-30 |
| **Migration Risk** | None | High (3+ months) | Low (wrapper fix only) |
| **Debugging Ease** | Easy | Hard | Medium |
| **Regime Coverage** | Poor (gaps) | Good | Excellent |
| **Diversification** | None (single strategy) | N/A (merged) | Excellent (low corr) |
| **Expected PF** | 2.2 (avg across regimes) | 2.8-3.2 (if works) | 2.5-2.8 |
| **Expected Sharpe** | 1.3-1.5 | 1.8-2.0 (if works) | 1.6-1.8 |
| **Max Drawdown** | 12-18% | 10-12% (if works) | 10-12% |
| **Flexibility** | Low (one strategy only) | Low (merged, hard to change) | High (adjust allocation) |
| **Optionality** | None | Locked in | Can merge later if desired |
| **Production Readiness** | Immediate (B0 only) | 3-6 months away | Immediate (B0), 2 weeks (S4/S5) |
| **Risk Level** | Medium | High | Low-Medium |

---

## Decision Criteria Framework

### Choose Single System If:

- [ ] Only one system performs well (PF > 2.5, others < 1.5)
- [ ] Team bandwidth extremely limited (<10 hours/month)
- [ ] Want maximum simplicity (ok with regime gaps)
- [ ] Short-term deployment (<3 months)

**Not Recommended:** Leaves performance on table

---

### Choose Merged System If:

- [ ] Both systems validated for 12+ months
- [ ] Have 200+ trades for ML ensemble training
- [ ] ML ensemble shows >20% PF improvement in backtest
- [ ] Team has 4-6 months for migration project
- [ ] Dual-system maintenance overhead unsustainable

**Not Recommended Now:** Too risky, insufficient data for ML

**Revisit:** After 12 months dual-system operation

---

### Choose Dual-System If:

- [x] Want to validate both systems before committing to merge
- [x] Value B0 simplicity (don't want to lose it)
- [x] Want operational flexibility (adjust allocation monthly)
- [x] Both systems show promise (B0 PF 3.17, S4 PF 2.32)
- [x] Accept moderate maintenance overhead (20-30 hours/month)
- [x] Want diversification benefit (low correlation)

**RECOMMENDED:** Start here, migrate to merged later if data supports

---

## Evolution Path

### Year 1: Dual-System Operation

```
Month 1-3:  Paper trading → validate both systems
Month 4-6:  Live trading → conservative allocation (70/30)
Month 7-9:  Optimization → balanced allocation (50/50)
Month 10-12: Evaluation → collect data, review integration options
```

### Year 2: Integration Decision

```
Evaluate after 12 months:

IF both systems PF > 2.5 AND have 200+ trades:
  → Consider Option 2 (Merged System)
  → Build ML ensemble, backtest
  → If ensemble PF > best single system by 20%: Migrate
  → Else: Stay on dual-system

IF one system clearly dominates (PF > 3.0, other < 2.0):
  → Move to Option 1 (Single System)
  → Keep losing system at 10% allocation (hedge)

IF both systems performing well (PF 2.0-2.5):
  → STAY on Option 3 (Dual-System)
  → Optimize capital router (add ML to allocation)
  → Keep systems separate (proven to work)
```

**Default Recommendation:** STAY on dual-system unless strong evidence for migration

Reason: Dual-system works, low risk, preserves optionality

---

## Risk Assessment by Option

### Single System Risk

**Low Technical Risk:**
- Simple to operate
- Easy to debug
- Well-validated (B0 PF 3.17)

**High Business Risk:**
- Regime gaps (underperforms in some conditions)
- Missed opportunities (low frequency or idle periods)
- No diversification (single point of failure)

**Overall: MEDIUM RISK** (technical low, business high)

---

### Merged System Risk

**High Technical Risk:**
- Complex migration (3+ months)
- Many integration points (bugs likely)
- Validation burden (entire system from scratch)
- Debugging difficult (merged complexity)

**High Business Risk:**
- Loses B0 simplicity (main advantage)
- ML ensemble may not work (overfitting risk)
- Locked in (hard to revert after merge)

**Overall: HIGH RISK** (avoid until proven necessary)

---

### Dual-System Risk

**Low Technical Risk:**
- No migration needed (both systems already work)
- Isolated failure domains (B0 fails, S4 keeps running)
- Easy to debug (per-system monitoring)

**Low Business Risk:**
- Diversification reduces drawdown 10-20%
- Can adjust allocation monthly (adaptive)
- Preserves optionality (can merge later)

**Medium Operational Risk:**
- Higher maintenance (two systems)
- Requires disciplined rebalancing

**Overall: LOW-MEDIUM RISK** (best risk/reward)

---

## Final Recommendation

**START with Dual-System (Option 3)**

**Reasons:**
1. Lowest risk (both systems validated)
2. Best diversification (low correlation)
3. Operational flexibility (adjust monthly)
4. Preserves B0 simplicity (main selling point)
5. Keeps option to merge later (if data supports)

**DEFER Merged System (Option 2) until:**
- 12 months dual-system operation
- 200+ trades collected
- ML ensemble shows >20% improvement in backtest
- Team has 4-6 months bandwidth

**AVOID Single System (Option 1) unless:**
- One system clearly fails (PF < 1.5 for 6 months)
- Extreme bandwidth constraints (<10 hours/month)

**Default Path: Dual-System for at least Year 1, reevaluate at 12 months**

---

**Document Owner:** System Architect
**Last Updated:** 2025-12-04
**Next Review:** Month 12 (Year 1 complete)

**Decision Status:** ✅ APPROVED - Deploy Dual-System Architecture
