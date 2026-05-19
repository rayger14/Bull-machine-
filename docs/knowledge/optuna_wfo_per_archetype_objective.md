# Optuna WFO — Per-Archetype Objective (Rule 7 Enforcement)

**Date**: 2026-05-18
**Branch**: `feat/optuna-per-archetype-objective`

## What changed

`bin/optuna_wfo.py` now supports a `--target-archetype` CLI flag that:

1. Computes baseline per-archetype OOS PnL on the unmodified config before optimization starts
2. Tracks per-window target-archetype PnL/PF/trade-count for every trial
3. Applies a **0.5× score multiplier penalty** to trials where the target archetype's OOS PnL regresses by > 20% vs baseline (or drops by > $500 if baseline was negative)
4. Stores per-archetype results in `trial.user_attrs` for downstream analysis

This is the infrastructure equivalent of codifying **Quant-Analyst Rule 7** (target-archetype-must-improve auto-reject) into the sweeper. Previously the sweeper optimized only system-wide PF, so dedup-reshuffling false signals could slip through. Now the sweeper actively penalizes them.

## Why

Per the May 18 quant-analyst Rule 7 codification: when a parameter change makes system PnL improve but the target archetype's PnL worsens, it's dedup-reshuffling — bars get re-routed to other archetypes without the change doing what we intended. Three+ studies this cycle hit this false-signal pattern; manual per-archetype inspection caught them. The sweeper should catch them automatically going forward.

## Usage

```bash
# Single target archetype
python3 bin/optuna_wfo.py --group A --trials 40 --mode wfo \
    --target-archetype liquidity_compression

# Multiple target archetypes
python3 bin/optuna_wfo.py --group A --trials 40 --mode wfo \
    --target-archetype long_squeeze,oi_divergence
```

## What you'll see during a run

```
Target archetypes (Rule 7 dedup-reshuffling guard): ['liquidity_compression']
Computing baseline per-archetype PnL across 2 OOS windows...
  baseline OOS liquidity_compression: $5,832

Starting WFO optimization (40 trials)...
  Trial   0 | W0: PF=1.45 (52 tr, WFE=98%) | W1: PF=1.62 (48 tr, WFE=105%) | min_PF=1.45 | score=1.43 *** BEST | 91s
  Trial   1 | W0: PF=1.48 (50 tr, WFE=99%) | W1: PF=1.60 (46 tr, WFE=104%) | min_PF=1.48 | score=0.74 | 89s   <-- penalty applied (target LC OOS PnL dropped to $3,200)
  ...
```

The score on a penalized trial halves, making it virtually impossible to beat clean trials. Best-trial selection naturally filters out dedup-reshuffling false signals.

## Trial attrs (per-archetype reporting)

For each trial, `trial.user_attrs` now contains (per target archetype `<arch>`):

- `target_<arch>_oos_pnl` (summed across WFO windows)
- `target_<arch>_oos_trades` (summed)
- `target_<arch>_oos_pf_mean` (mean of per-window PFs)

Plus, when penalty applied:
- `target_archetype_penalty: True`
- `target_archetype_penalty_reasons: [str, ...]`

These enable post-hoc analysis in `study.trials_dataframe()` or the results.json output.

## Smoke test result (May 18 validation run)

Ran `python3 bin/optuna_wfo.py --group A --trials 2 --mode wfo --target-archetype liquidity_compression`:
- `target_liquidity_compression_oos_pnl: $8,288.66` ✓
- `target_liquidity_compression_oos_trades: 38` ✓
- `target_liquidity_compression_oos_pf_mean: 2.77` ✓

The trial attrs populate correctly. Penalty path wasn't exercised in 2 trials (no regression). Future deeper runs will exercise it.

## Files

- `bin/optuna_wfo.py` — modified
- This documentation: `docs/knowledge/optuna_wfo_per_archetype_objective.md`

## When to use the flag

**Required** for any study that targets a specific archetype's improvement (e.g., "tune long_squeeze gates"). Without the flag, the sweeper might find params that improve system PF via dedup-shifting without actually fixing the target archetype — exactly the May 18 LC + oi_change_24h failure mode.

**Optional** for system-wide tuning (e.g., "tune all of Group A jointly with no target preference"). The classical objective is still available by omitting `--target-archetype`.

## Limitations

- Penalty is a hard 0.5× score multiplier, not graduated. A trial that improves the target archetype hugely AND regresses system PnL slightly might still pass; a trial that improves system PnL hugely AND regresses target slightly will likely fail. That's the intended bias per Rule 7.
- Baseline is computed ONCE at startup on the unmodified config. If `--config` is overridden, baseline reflects that override (not main production config).
- The `CPCVObjective` (separate class around line 384) does NOT yet have per-archetype tracking. Adding it is a 1-2 hour follow-up.
