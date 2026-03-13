---
name: optuna-optimize
description: Run Optuna hyperparameter optimization with walk-forward validation and CPCV. Enforces best practices to prevent overfitting.
user_invocable: true
---

# Optuna Gate Optimization — Best Practice Protocol

You are about to run hyperparameter optimization on trading strategy gate thresholds. Follow this protocol EXACTLY to avoid overfitting.

## MANDATORY Rules (Never Violate)

1. **NEVER optimize on the full date range and call it done.** Always validate out-of-sample.
2. **NEVER accept parameters from a single train/test split.** Use walk-forward with 2+ windows or CPCV.
3. **NEVER use WFE (Walk-Forward Efficiency) as the only validation.** WFE > 70% is necessary but NOT sufficient — also check that PF > 1.2 in EVERY OOS window.
4. **NEVER optimize more than 15 parameters at once.** Split into groups of 5-8 parameters. TPE works best with 4-8x the parameter count in trials (i.e., 30-60 trials for 7 params).
5. **NEVER set n_startup_trials below 10.** TPE needs random exploration before exploitation. Use `min(15, trials // 3)`.
6. **NEVER deploy parameters without comparing to baseline.** Always run the baseline (current YAML values) first.
7. **NEVER ignore trade count drops.** If optimized params reduce trades by >50%, the optimizer found a niche overfit, not a better filter.

## Validation Protocol: Anchored Walk-Forward + CPCV

### Option A: Anchored Walk-Forward (simpler, recommended for <100 trials)

```
Window 1: Train 2020-2022, Test 2023
Window 2: Train 2020-2023, Test 2024
```

- Run 40 TPE trials per window
- For each window, record best parameters AND their OOS performance
- **ACCEPT only if**: PF > 1.2 in BOTH OOS windows AND trade count within 30% of baseline
- **REJECT if**: PF > 2.0 in one window but < 1.0 in another (classic overfit signature)

### Option B: CPCV — Combinatorial Purged Cross-Validation (rigorous, for production)

Split data into k=6 groups (~1 year each for 2019-2024), use p=2 test groups.
This gives C(6,2) = 15 train/test paths.

```
For each of 15 combinations:
  - Train on 4 groups, test on 2 groups
  - Purge: remove bars within 48h of train/test boundary (prevents leakage)
  - Embargo: remove 24 additional bars after each purge zone
```

- Run 30 TPE trials on each of 3 representative paths (not all 15 — too expensive)
- Validate final parameters on ALL 15 paths
- **Stitch equity curves** into phi(k,p) = 5 non-overlapping backtest paths
- **Compute PBO** (Probability of Backtest Overfitting) = fraction of stitched paths with Sharpe < 0
- **ACCEPT only if**: median OOS PF > 1.2 across all 15 paths AND no path has PF < 0.8 AND PBO < 50%

### Purge & Embargo Logic (for financial time series)

```
purge_bars = 48   # 48 hours at 1H frequency — removes autocorrelation leakage
embargo_bars = 24  # Additional 24h buffer after purge

For each train/test boundary:
  - Remove from train: any bar within purge_bars of the test period start/end
  - Remove from train: embargo_bars additional bars after each purge zone
```

## Scoring Function

```python
# Primary: Profit Factor with mild drawdown penalty
dd_penalty = max(0, (max_dd_pct - 8.0)) * 0.02
score = pf - dd_penalty if pf > 1.0 else pf - 1.0

# Hard constraints (return -inf):
if trades < min_trades: return -1e9      # min_trades = 50 for Group A, 20 for Group B
if max_dd_pct > 15.0: return -1e9        # Absolute DD ceiling
```

## Optuna Configuration

```python
sampler = TPESampler(
    seed=42,
    n_startup_trials=min(15, n_trials // 3),  # Random exploration phase
    multivariate=True,                          # Model parameter correlations
)

# Optional: pruning for 2x trial efficiency
pruner = MedianPruner(
    n_startup_trials=10,
    n_warmup_steps=1,   # Allow 1 intermediate report before pruning
)
```

## Parameter Importance Analysis

After optimization, ALWAYS run:
```python
importance = optuna.importance.get_param_importances(study)
```

- If one parameter accounts for >80% of importance, the others are noise — consider fixing them at defaults.
- If all parameters have <5% importance, the gates don't matter — the signal quality comes from structural checks, not thresholds.

## Anti-Overfit Checklist (verify before deploying)

- [ ] Optimized on multiple time windows (not just one period)
- [ ] OOS PF > 1.2 in every validation window
- [ ] Trade count within 30% of baseline in every window
- [ ] No parameter hit its boundary (if it did, widen the range and re-run)
- [ ] Parameter importance makes intuitive sense
- [ ] MaxDD in OOS <= 1.5x MaxDD in training
- [ ] Win rate in OOS within 10 percentage points of training

## Running the Optimizer

```bash
# Walk-Forward mode (recommended)
python3 bin/optuna_wfo.py --group A --trials 40 --mode wfo

# CPCV mode (rigorous)
python3 bin/optuna_wfo.py --group A --trials 30 --mode cpcv

# Quick single-window (for testing only, NOT for production)
python3 bin/optuna_optimize_gates.py --group A --trials 60
```

## What NOT To Do

- Do NOT run Optuna on the full 2020-2024 range without WFO/CPCV validation
- Do NOT increase trials beyond 200 hoping for better results (diminishing returns after 60-80 for 7 params)
- Do NOT optimize fusion weights (lesson #54: fusion has negative predictive power)
- Do NOT add bonuses to scoring (lesson #1: only penalties and gates)
- Do NOT optimize multiple groups simultaneously (they interact — optimize A first, freeze, then B)
