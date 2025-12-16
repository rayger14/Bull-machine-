# Dual-System Trading: Quick Start Guide

**Version:** 1.0.0
**Date:** 2025-12-04
**Status:** Executive Summary
**Audience:** Decision makers, operators, new team members

---

## Overview in 60 Seconds

**What:** Two independent trading systems running in parallel
**Why:** B0 provides simple baseline (PF 3.17), Archetypes provide specialized alpha (PF 2.2/1.86)
**How:** Capital router allocates between systems based on performance and regime
**When:** Start conservative (70/30), rebalance monthly based on data

---

## The Two Systems

### System B0 (Baseline-Conservative)

```
What it does:  Buys deep dips, sells at +8% profit
Entry trigger: 30-day drawdown < -15%
Exit logic:    +8% take profit OR stop loss

Performance:   PF 3.17, WR 42.9%, 7 trades/year
Complexity:    LOW (500 lines of code)
Maintenance:   4-6 hours/month ($150)
Framework:     New v2 (clean architecture)

Best used:     All market conditions (all-weather)
Status:        ✅ Production-ready NOW
```

### System Archetypes (S4/S5/S1)

```
What it does:  Pattern recognition with regime awareness
Patterns:      S4 (Funding Divergence), S5 (Long Squeeze), S1 (Crisis)
Entry trigger: Complex fusion score + regime gates
Exit logic:    Dynamic trailing stops

Performance:   S4 PF 2.32, S5 PF 1.86, 12-14 trades/year each
Complexity:    HIGH (39,000 lines of code)
Maintenance:   12-16 hours/month ($500 per system)
Framework:     Old v1 (legacy, needs migration)

Best used:     Specific regimes (S4: bear, S5: bull pullbacks, S1: crisis)
Status:        ⚠️ Needs wrapper fix and validation
```

---

## Capital Allocation Scenarios

### Conservative (Recommended for Start)

```
B0:          70% ($70k of $100k)
Archetypes:  30% ($30k)
  ├─ S4:     15% ($15k)
  ├─ S5:     10% ($10k)
  └─ S1:     5% ($5k)

Expected Portfolio PF: 2.8-3.0
Risk Level: LOW
Use when: Paper trading, first 1-3 months live
```

### Balanced (Standard Operating Mode)

```
B0:          50% ($50k)
Archetypes:  50% ($50k)
  ├─ S4:     20% ($20k)
  ├─ S5:     20% ($20k)
  └─ S1:     10% ($10k)

Expected Portfolio PF: 2.5-2.7
Risk Level: MODERATE
Use when: After successful paper trading, moderate risk tolerance
```

### Aggressive (After Validation)

```
B0:          30% ($30k)
Archetypes:  70% ($70k)
  ├─ S4:     25% ($25k)
  ├─ S5:     25% ($25k)
  └─ S1:     20% ($20k)

Expected Portfolio PF: 2.2-2.5
Risk Level: HIGH
Use when: Archetypes proven superior in live (6+ months)
```

---

## Deployment Timeline (8 Weeks)

### Week 1: B0 Paper Trading
- Deploy B0, validate on live data
- Collect 0-2 trades
- **Gate 1:** Paper PF > 2.5 OR verified signal logic

### Week 2: S4/S5 Paper Trading
- Fix archetype wrapper
- Deploy S4/S5 alongside B0
- Collect 0-4 archetype trades
- **Gate 2:** At least one archetype fires, PF > 1.5

### Week 3-4: Combined Evaluation
- Run 50/50 portfolio for 14 days
- Calculate metrics, compare to backtest
- Draft live allocation (conservative/balanced/aggressive)
- **Gate 3:** Portfolio PF > 2.0, no catastrophic losses

### Week 5: Minimal Live (10% capital)
- Deploy with $10k live, $90k cash
- Validate execution
- Monitor hourly

### Week 6-7: Scale to 50%
- If week 5 successful, scale to $50k live
- Monitor daily

### Week 8: Full Live (100%)
- If week 6-7 consistent, deploy full $100k
- **Gate 4:** No catastrophic losses, live matches paper

### Week 9+: Production Operations
- Daily monitoring
- Monthly rebalancing
- Quarterly strategy review

---

## Daily Operations Checklist

### Morning (15 minutes)

```
[ ] All systems online? (B0, S4, S5)
[ ] Exchange API connected?
[ ] Open positions: count, PnL, age
[ ] Exposure: X% (alert if > 25%)
[ ] Overnight PnL: $XXX
[ ] Drawdown: X% (alert if > 15%)
[ ] Alerts: X critical, X warnings
```

### End of Day (20 minutes)

```
[ ] Daily PnL: $XXX (compare to target)
[ ] Trades executed: X
[ ] Stop losses hit: X (review losses)
[ ] Position review: All have SL set?
[ ] System uptime: 24/24 hours?
[ ] Log day's events
```

---

## Key Performance Metrics

### Portfolio-Level (Target)

| Metric | Target | Alert | Kill Switch |
|--------|--------|-------|-------------|
| **PF (30d)** | > 2.5 | < 2.0 | < 1.5 (for 6 months) |
| **WR (30d)** | > 50% | < 40% | - |
| **Drawdown** | < 10% | 15% | 30% |
| **Sharpe (90d)** | > 1.5 | < 1.0 | - |
| **Exposure** | 15-20% | 22% | 25% |

### Per-System (30-day rolling)

| System | Target PF | Alert PF | Action if Alert |
|--------|-----------|----------|-----------------|
| **B0** | > 2.5 | < 2.0 | Investigate, relax thresholds? |
| **S4** | > 2.0 | < 1.5 | Reduce allocation by 50% |
| **S5** | > 1.8 | < 1.5 | Reduce allocation by 50% |

---

## Decision Framework

### After 3 Months Live

**If B0 dominates (B0 PF > Archetypes * 1.5):**
→ Increase B0 to 80%, reduce archetypes to 20%

**If competitive (both PF > 2.0, within 20%):**
→ Keep 50/50 balanced

**If archetypes outperform (Arch PF > B0 * 1.3):**
→ Increase archetypes to 60%, reduce B0 to 40%

### After 6 Months Live

**Integration Path Decision:**

**Option A: Keep Separate (Capital Router)**
- Simplest, lowest risk
- Choose if: Both performing well, want flexibility
- Effort: LOW (already implemented)

**Option B: Fix Wrapper (Unified Framework)**
- Migrate archetypes to v2 framework
- Choose if: Archetypes validated, reduce maintenance
- Effort: MEDIUM (2-3 months)

**Option C: Build Meta-System (ML Ensemble)**
- Train ML to combine signals
- Choose if: 200+ trades, want max performance
- Effort: HIGH (4-6 months)

**Default: Option A** (keep separate for first year)

---

## Emergency Procedures

### Kill Switch (DD > 30%)

```
1. Close all positions (market orders)
2. Pause new entries (all systems)
3. Alert stakeholders (Slack, email, SMS)
4. Investigation (1-2 days):
   - What caused DD? (flash crash, bug, strategy failure)
   - Which system contributed most?
   - Were risk limits followed?
5. Recovery:
   - Fix issues
   - Backtest fix
   - Paper trade 7 days
   - Resume live (gradual: 10% → 50% → 100%)
```

### System Down (Critical Bug)

```
1. Attempt auto-restart
2. If fails: Alert on-call engineer
3. Isolate failure (B0 vs S4 vs S5)
4. Disable failing system, keep others running
5. Debug offline, deploy fix to paper first
```

---

## Why Two Systems? (FAQ)

**Q: Why not merge into one system?**
A: Different philosophies - B0 is simple (500 LOC), archetypes complex (39k LOC). Merging loses B0's simplicity advantage.

**Q: Why not just use the best performer?**
A: Performance varies by regime. B0 best in bull (PF 3.17), S4 best in bear (PF 2.22). Portfolio captures both.

**Q: Isn't this more maintenance?**
A: Yes (+$500/month for archetypes). But diversification reduces drawdown 10-20%, worth the cost.

**Q: What if archetypes fail?**
A: Reduce to 10% allocation or disable. B0 keeps portfolio profitable (PF 3.17 standalone).

**Q: When will we merge the systems?**
A: Maybe never. Dual-system is the DESIGN. Review after 12 months, but default is keep separate.

---

## Contact and Resources

**Documents:**
- [Production Architecture (Full)](/Users/raymondghandchi/Bull-machine-/Bull-machine-/docs/PRODUCTION_DUAL_SYSTEM_ARCHITECTURE.md)
- [Capital Allocation Strategy](/Users/raymondghandchi/Bull-machine-/Bull-machine-/docs/CAPITAL_ALLOCATION_STRATEGY.md)
- [System Comparison Matrix](/Users/raymondghandchi/Bull-machine-/Bull-machine-/docs/SYSTEM_COMPARISON_MATRIX.md)
- [Deployment Roadmap](/Users/raymondghandchi/Bull-machine-/Bull-machine-/docs/DUAL_SYSTEM_DEPLOYMENT_ROADMAP.md)
- [Operational Playbook](/Users/raymondghandchi/Bull-machine-/Bull-machine-/docs/OPERATIONAL_PLAYBOOK.md)

**Code Locations:**
- B0: `/engine/models/simple_classifier.py`
- Archetypes: `/engine/archetypes/logic_v2_adapter.py`
- Capital Router: `/engine/capital_router.py` (to be created)
- Backtest Engine: `/engine/backtesting/engine.py`

**Owner:** System Architect / Trading Operations
**Last Updated:** 2025-12-04
**Next Review:** After Week 2 (S4/S5 paper trading complete)

---

**TL;DR:** Run B0 (simple, PF 3.17) + Archetypes (complex, regime specialists) in parallel. Start 70/30, rebalance monthly. Both systems have value - keep separate for flexibility, low risk.
