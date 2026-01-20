# Bull Machine Optimization Execution Playbook

**Created:** 2025-11-20
**Status:** READY FOR EXECUTION
**Audience:** Engineering teams implementing optimization initiative

---

## What This Playbook Is

A **step-by-step execution guide** for optimizing the Bull Machine's S2 and S5 trading archetypes using multi-objective Optuna and walk-forward validation.

**Problem:** S2 currently produces 418 trades/year (target: 5-10) with poor profitability. Thresholds need data-driven calibration.

**Solution:** Use Optuna to find optimal parameter combinations that hit target trade frequency while maximizing profit.

**Outcome:** Production-ready configs with Pareto frontier analysis and validation proof.

---

## Quick Navigation

### I'm Starting Now
**Read this:** `OPTIMIZATION_EXECUTION_PLAYBOOK.md` (main document)
- Complete step-by-step walkthrough
- All code included (copy-paste ready)
- Troubleshooting guide
- Success checklist

**Time:** 1 hour to read, 6-15 hours to execute

### I Need Just the Commands
**Read this:** `OPTIMIZATION_QUICK_REFERENCE.md`
- Copy-paste commands for each phase
- Decision matrices
- Common fixes
- Performance benchmarks

**Time:** 10 minutes to scan

### I Need a Visual Flowchart
**Read this:** `OPTIMIZATION_PHASE_FLOWCHART.md`
- ASCII phase diagrams
- Decision trees
- Troubleshooting flowchart
- Red flags & abort criteria

**Time:** 15 minutes to review

### I Need Detailed Specifications
**Read this:** `OPTIMIZATION_REQUIREMENTS_SPEC.md`
- Complete system specifications
- All workstreams detailed
- Risk analysis
- Open questions

**Time:** 2 hours (reference document)

### I Need a Quick Overview
**Read this:** `OPTIMIZATION_PLAYBOOK_SUMMARY.md` (this section)
- 5-minute overview
- Document map
- Success metrics
- Getting help

**Time:** 10 minutes

---

## The Documents

| Document | Purpose | Read When | Length |
|----------|---------|-----------|--------|
| **OPTIMIZATION_EXECUTION_PLAYBOOK.md** | Main step-by-step guide | Starting execution | 22 KB |
| **OPTIMIZATION_QUICK_REFERENCE.md** | Command cheat sheet | During execution | 9 KB |
| **OPTIMIZATION_PHASE_FLOWCHART.md** | Visual flowcharts & diagrams | Stuck or confused | 14 KB |
| **OPTIMIZATION_REQUIREMENTS_SPEC.md** | Detailed specifications | Understanding context | 39 KB |
| **OPTIMIZATION_PLAYBOOK_SUMMARY.md** | Overview & decision guide | Quick reference | 12 KB |
| **README_OPTIMIZATION_PLAYBOOK.md** | This file | Navigation | 5 KB |

---

## Choose Your Path

### MVP Track: 6 Hours (Conservative)
```
Phases: 0 → 1 → 2 → 4
Scope:  S2 only, full validation
Result: S2 production config + Pareto frontier

Best for:
├─ First-time optimization
├─ Limited time availability
├─ Risk-averse teams
└─ OI data still broken (can skip S5)
```

**Schedule:**
- Day 1 AM: Phase 0 (setup, 30 min) + Phase 1 (distribution, 45 min)
- Day 1 PM: Phase 2 (Optuna, 3.5 hours - runs in background)
- Day 2 AM: Phase 4 (validation, 1 hour)

### Full Track: 15 Hours (Ambitious)
```
Phases: 0 → 1 → 2 → 3 → 4 → 5 (optional)
Scope:  S2 + S5 + optional engine weights
Result: Both archetypes production-ready

Best for:
├─ Complete optimization
├─ Have 2-3 days available
├─ OI data fixed
└─ Want S2 + S5 ready
```

**Schedule:**
- Day 1: Phase 0 + Phase 1
- Day 2: Phase 2 (S2) + Phase 3 (S5) in parallel
- Day 3: Phase 4 (validation) + Phase 5 optional

---

## Success = This Checklist

### MVP Success (6 hours)
- [ ] S2 produces 5-10 trades/year on 2022 H1
- [ ] S2 Profit Factor > 1.3 on 2022 H2 validation
- [ ] S2 Profit Factor > 1.1 on 2023 H1 test
- [ ] All 3 walk-forward tiers pass
- [ ] At least 1 Pareto solution exists

### Full Success (15 hours)
- All MVP items PLUS:
- [ ] S5 produces 7-12 trades/year
- [ ] S5 Profit Factor > 1.5 on validation
- [ ] Regime gating verified (S2 = 0 in risk_on)
- [ ] Both configs deployed
- [ ] 15+ Pareto solutions across both archetypes

---

## Critical Decisions

### Decision 1: Which Track? (Day 1)
```
Question: How much time do we have?
├─ < 1 day available → MVP
└─ 2-3 days available → Full

Question: Is OI_CHANGE data working?
├─ No → Can skip S5 with MVP
└─ Yes → Can do Full track
```

### Decision 2: Kill S2? (Day 2, mid-Phase 2)
```
After 50 Optuna trials, best PF is:
├─ >= 1.5 → KEEP, proceed confidently
├─ 1.2-1.5 → KEEP, proceed normally
├─ 1.0-1.2 → MARGINAL, proceed cautiously
└─ < 1.0 → KILL, abandon S2
```

### Decision 3: Which Config? (After Phase 2)
```
From Pareto frontier, choose based on risk:
├─ Conservative (low risk) → Solution #1-2
├─ Balanced (medium risk) → Solution #3-5
└─ Aggressive (high risk) → Solution #6-10
```

### Decision 4: Deploy? (End of Phase 4)
```
Walk-forward validation test result:
├─ Test PF > 1.1 → YES, proceed to deployment
├─ Test PF 0.9-1.1 → MARGINAL, use conservative config
└─ Test PF < 0.9 → NO, re-run with different params
```

---

## How to Read These Documents

### Scenario 1: "I'm executing right now"
1. Open `OPTIMIZATION_EXECUTION_PLAYBOOK.md`
2. Start with "Phase 0: Prerequisites"
3. Follow each step exactly
4. Keep `OPTIMIZATION_QUICK_REFERENCE.md` open for commands
5. Use `OPTIMIZATION_PHASE_FLOWCHART.md` if you get stuck

### Scenario 2: "Phase 2 is running, what do I do?"
1. While Optuna runs (3-5 hours), read:
   - Decision matrices in `OPTIMIZATION_QUICK_REFERENCE.md`
   - What "Pareto frontier" means in `OPTIMIZATION_EXECUTION_PLAYBOOK.md` Phase 2
   - Validation logic in `OPTIMIZATION_PHASE_FLOWCHART.md`
2. Every 30 min, check progress:
   ```bash
   python3 -c "import pandas as pd; print(f'Trials: {len(pd.read_csv(\"results/s2_optimization/pareto_trials.csv\"))}/50')"
   ```

### Scenario 3: "Something went wrong"
1. Find your issue in `OPTIMIZATION_EXECUTION_PLAYBOOK.md` → "Troubleshooting Guide"
2. Follow the diagnosis steps
3. Check `OPTIMIZATION_QUICK_REFERENCE.md` for one-line fixes
4. Review `OPTIMIZATION_PHASE_FLOWCHART.md` for red flags

### Scenario 4: "I need context / background"
1. Read `OPTIMIZATION_REQUIREMENTS_SPEC.md` Executive Summary
2. Check `docs/OPTIMIZATION_ARCHITECTURE_SUMMARY.md` for design
3. Review decision matrices in `OPTIMIZATION_QUICK_REFERENCE.md`

---

## Files You'll Create

### Must Create (MVP)
```
results/distributions/s2_fusion_percentiles_2022.csv
results/s2_optimization/pareto_frontier.csv
results/s2_optimization/pareto_trials.csv
configs/optimized/s2_best.json
results/validation/validation_report.json
```

### Optional (Full Track)
```
results/distributions/s5_fusion_percentiles_2022.csv
results/s5_optimization/pareto_frontier.csv
configs/optimized/s5_best.json
```

---

## Estimated Timeline

### MVP Track (6 hours work, 1.5 days wall time)
```
Phase 0: Prerequisites          30 min
Phase 1: Distribution           45 min
Phase 2: S2 Optimization        3.5 hours (Optuna runs in background)
Phase 4: Validation             1 hour
─────────────────────────────────────
TOTAL:                          6 hours
```

### Full Track (15 hours work, 3 days wall time)
```
Phase 0: Prerequisites          30 min
Phase 1: Distribution           1.5 hours
Phase 2: S2 Optimization        3.5 hours (parallel start)
Phase 3: S5 Optimization        3.5 hours (parallel with Phase 2)
Phase 4: Validation             2 hours
Phase 5: Engine Weights (opt)   4 hours (optional)
─────────────────────────────────────
TOTAL:                          15 hours
```

---

## Troubleshooting Quick Links

**Issue: "No trades generated"**
- Root cause: Search space too tight or no bear signals
- Solution: See `OPTIMIZATION_EXECUTION_PLAYBOOK.md` → "Troubleshooting" → Issue 1

**Issue: "Best PF is only 0.8"**
- Root cause: Pattern fundamentally weak or constraints too strict
- Solution: See `OPTIMIZATION_EXECUTION_PLAYBOOK.md` → "Troubleshooting" → Issue 2

**Issue: "Test PF collapsed (0.5 vs 1.6 train)"**
- Root cause: Overfitting to 2022 or market regime changed
- Solution: See `OPTIMIZATION_EXECUTION_PLAYBOOK.md` → "Troubleshooting" → Issue 3

**Issue: "Optuna crashes"**
- Root cause: Backtest engine broken
- Solution: See `OPTIMIZATION_QUICK_REFERENCE.md` → Common Issues table

**Full troubleshooting:** `OPTIMIZATION_EXECUTION_PLAYBOOK.md` has detailed flowchart in Phase 3.

---

## Key Metrics

### Must Achieve
- S2 trades: 5-10/year on 2022 training
- S2 PF: > 1.3 on 2022 validation
- S2 PF: > 1.1 on 2023 test
- Validation: All 3 tiers pass

### Should Achieve
- S5 trades: 7-12/year
- S5 PF: > 1.5 on validation
- Regime gating: S2 = 0 trades in risk_on
- Pareto solutions: 10+

### Nice to Have
- Sharpe Ratio > 1.0
- Win Rate > 55%
- Generalization to ETH (PF > 1.0)
- Engine weight improvement > 10%

---

## Pre-Execution Checklist

Before starting, confirm:

- [ ] You've chosen MVP or Full track
- [ ] You have 6+ hours (MVP) or 15+ hours (Full) available
- [ ] You have `OPTIMIZATION_EXECUTION_PLAYBOOK.md` bookmarked
- [ ] You understand what "Pareto frontier" means
- [ ] You understand "walk-forward validation"
- [ ] You know the 4 critical decisions (above)
- [ ] You have `OPTIMIZATION_QUICK_REFERENCE.md` handy for commands
- [ ] You've read Phase 0 of the main playbook

If unsure on any point, read the relevant section in `OPTIMIZATION_EXECUTION_PLAYBOOK.md`.

---

## Support & Getting Help

### "I'm confused about what to do"
→ Read `OPTIMIZATION_EXECUTION_PLAYBOOK.md` Phase 0-2 (30 min)

### "I need the command for Phase X"
→ Open `OPTIMIZATION_QUICK_REFERENCE.md` (command cheat sheet)

### "I'm stuck and need visual help"
→ Check `OPTIMIZATION_PHASE_FLOWCHART.md` (flowcharts)

### "I want to understand the why"
→ Read `OPTIMIZATION_REQUIREMENTS_SPEC.md` (executive summary)

### "I need a quick overview"
→ This file or `OPTIMIZATION_PLAYBOOK_SUMMARY.md` (5 min read)

### "Something went wrong"
→ See `OPTIMIZATION_EXECUTION_PLAYBOOK.md` section 3 (troubleshooting)

---

## Next Step

Open `OPTIMIZATION_EXECUTION_PLAYBOOK.md` and start Phase 0.

It's copy-paste ready. You've got this.

---

**Version:** 1.0
**Created:** 2025-11-20
**Status:** READY FOR EXECUTION

