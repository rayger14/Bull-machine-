# Pareto Frontier Exploration - Live Status

**Started**: 2025-11-10 21:40:15
**Process**: PID 59895
**Status**: Running Frontier 1 (Unconstrained)

## Why Frontier Exploration?

The constrained optimization (PF ≥ 1.2) was failing because we were **guessing constraints**. Instead, we're now systematically mapping the **Pareto frontier** to discover what's ACTUALLY achievable.

## 5 Frontiers Running (30 trials each)

### Frontier 1: Unconstrained (CURRENT)
**Objective**: Pure PF maximization (no constraints)
**Purpose**: Find the absolute ceiling - what's the max PF we can get?
**Status**: Trial 1/30 complete
- Best PF so far: **0.94** (median across windows)
- ETA: ~1.6 hours for this frontier

### Frontier 2: DD ≤ 30% (PENDING)
**Objective**: Max PF with relaxed drawdown constraint
**Purpose**: Can we get PF 1.2+ if we tolerate 30% DD?

### Frontier 3: DD ≤ 25% (PENDING)
**Objective**: Max PF with our original target DD
**Purpose**: Is PF 1.2 achievable at DD 25%?

### Frontier 4: DD ≤ 20% (PENDING)
**Objective**: Max PF with conservative DD
**Purpose**: What's achievable with strict risk management?

### Frontier 5: No Bleeders (PENDING)
**Objective**: Max PF while ensuring all archetypes have PF ≥ 1.0
**Purpose**: Can we fix Trap/VE bleeding without killing overall performance?

## Expected Timeline

- **Per trial**: ~3.5 minutes (4 windows × 50 seconds each)
- **Per frontier**: 30 trials × 3.5min = ~1.75 hours
- **Total**: 5 frontiers × 1.75h = **~8.75 hours**
- **ETA**: ~6:30 AM (2025-11-11)

## Early Insights (Trial 1)

**Unconstrained Frontier 1, Trial 0**:
- Median PF: **0.94**
- This matches our baseline (PF 0.92-0.94)

**This suggests**:
- Even without constraints, first trial achieved PF 0.94
- Baseline may already be near the ceiling (concerning!)
- OR TPE sampler needs more trials to find better regions (hopeful!)

## What We'll Learn

After completion, we'll have a **Pareto frontier map**:

```
PF vs DD Frontier:
PF 1.5 ─────────────┐
                     │ Unconstrained
PF 1.3 ──────────────┤
                     │ DD ≤ 30%
PF 1.2 ──────────────┤ ← Original target
                     │ DD ≤ 25%
PF 1.1 ──────────────┤
                     │ DD ≤ 20%
PF 1.0 ──────────────┤
                     │ No Bleeders
PF 0.9 ──────────────┴───────────────
      10%  15%  20%  25%  30%  35%  40% DD
```

This tells us:
1. **What's achievable**: Max PF at each DD level
2. **Where the constraints bind**: Which constraint is hardest (bleeders vs DD vs PF)
3. **Best realistic target**: What constraint to use for final optimization

## Monitoring Progress

```bash
# Check live progress
tail -f results/frontier_exploration.log

# Check process status
ps aux | grep optuna_frontier

# Quick summary (when frontiers complete)
ls -lh results/frontier_exploration/frontier_*_summary.txt
```

## Database Locations

Each frontier saves to its own SQLite database:
- `results/frontier_exploration/frontier_unconstrained.db`
- `results/frontier_exploration/frontier_dd30.db`
- `results/frontier_exploration/frontier_dd25.db`
- `results/frontier_exploration/frontier_dd20.db`
- `results/frontier_exploration/frontier_no_bleeders.db`

## Next Steps

When exploration completes:

1. **Analyze Pareto frontier**
   - Which frontier achieved highest PF?
   - What's the PF/DD tradeoff curve?
   - Can we fix bleeding archetypes?

2. **Pick best constraint**
   - If DD ≤ 25% achieves PF 1.15: Relax to DD ≤ 30%
   - If No Bleeders achieves PF 1.10: Accept it and focus on consistency
   - If Unconstrained only hits PF 1.05: Major strategy pivot needed

3. **Final focused optimization**
   - Run 100-200 trials with the best constraint
   - Deploy winning config

4. **Fallback strategies** (if all frontiers fail to hit PF 1.2):
   - Option A: OB-only strategy (9 trades @ PF 43.45)
   - Option B: Add bear archetypes (breakdown-below-support)
   - Option C: Accept PF 1.0-1.1 and focus on Sharpe/DD instead
   - Option D: PyTorch meta-learning for context-aware parameters

## Key Files

- Script: `bin/optuna_frontier_exploration.py`
- Log: `results/frontier_exploration.log`
- Results: `results/frontier_exploration/`
- Status: `FRONTIER_EXPLORATION_STATUS.md` (this file)
