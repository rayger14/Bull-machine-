# wick_trap CPCV — PASS (15/15): The Edge Is Not Era-Luck

**Date**: 2026-07-09. **Subject**: standalone wick_trap_v14rq (0.432, unstacked — the config the
stack-validation verdict ranked #1). **Artifact**: `results/champion_v14_cpcv/`, `scripts/champion/cpcv_wick_trap.py`

## Method
k=6 contiguous time groups over 2018-01-01..2026-06-10; all C(6,2)=15 two-group test
combinations evaluated (strategy is FIXED — no parameters fitted, so group-level backtests
aggregate exactly). Pre-registered bar: >=10/15 combos positive, median PF >= 1.2, worst > -$8K.

## Result: PASS on every criterion, with margin
- **15/15 combinations positive** (bar: 10)
- **Median combination PF 1.41** (bar: 1.2)
- **Worst combination +$2,421** (bar: > -$8K) — the worst alternate history still MAKES money
- Only one negative group: g4 (2022-03..2023-08, the deep bear) at -$1,916 — modest, and every
  pairing containing it is still net positive.

## What this means
wick_trap's 8.5-year record is NOT carried by a lucky era. Combined with the prior gates —
per-year battery, pristine 2025-26 holdout (PF 1.43, n=70), stacking adjudication (survived
attempts to "improve" it) — **wick_trap standalone has now cleared every offline validation
this project knows how to run.** The remaining evidence frontier is LIVE (it is already
unlocked in the live full book at 0.43; its live trades will accrue).

## Honest limits (unchanged)
Satellite strategy, not a system: ~55-95 trades/1.4yr, bleeds ~$2-9K in deep-bear stretches,
full-period MaxDD 14.3%. No overlay tested to date improves it (stack verdict 2026-07-08).

## Cross-references
[[stack_validation_verdict_2026_07_08]] · [[v14_champion_hunt_2026_06_11]] ·
[[breakeven_study_verdict_2026_06_29]] (RETRACTED)
