# Unified Strategy Verdict — Level Anchoring Fails the Bar; the Unified Strategy IS wick_trap

**Date**: 2026-07-13. **Build**: V14L store (V14 + 17 level features, no-lookahead), live parity by
construction (shared module now emits level features hourly). **Grid**: pre-registered, 3 level-gate
variants on repaired wick_trap vs baseline. **Artifacts**: `results/champion_unified/`, `scripts/champion/unified_grid.py`

## Results (V14L, thresholds enforced)
| variant | train PF | holdout PF | hold PnL | hold n | full PF | full PnL | 2022 |
|---|---|---|---|---|---|---|---|
| baseline (repaired wick_trap) | 1.37 | 1.43 | +$6,764 | 70 | 1.50 | $51,148 | −$8,667 |
| U1 sweep-required | 2.08 | 0.58 | −$214 | **3** | 1.85 | $7,651 | −$431 |
| U2 level-quality ≥0.5 | **1.26** | 1.53 | +$7,754 | 67 | 1.44 | $44,169 | −$7,704 |
| U3 near-support ≤2 ATR | 1.96 | 1.39 | +$1,833 | **23** | 1.71 | $23,195 | **+$244** |

**All three FAIL the pre-registered bar** (train AND holdout PF ≥ baseline, holdout n ≥ 30):
- U1: starves the strategy (3 holdout trades) — requiring a literal level sweep removes ~90% of
  wick_trap's trades; the shiny train PF is small-n mirage. The full ES recipe again fails at 1H.
- U2: holdout up (1.53) but TRAIN DOWN (1.26 < 1.37) — Rule 9 co-move violation; the holdout gain
  is not trustworthy (single regime window).
- U3: strong shape (train 1.96, 2022 flips positive) but n=23 < 30 — under-sampled. **Watch item**:
  re-test when live level-feature data accumulates; NOT deployable.

## The verdict
**Level anchoring does not robustly improve wick_trap at 1H granularity. The unified strategy is
wick_trap itself — by proof, not default.** Four months of knowledge (trigger design, exit ladder,
threshold re-quantiling, repair batch) already converged into the one validated instrument:
repaired wick_trap — battery ✓, pristine holdout PF 1.43 ✓, CPCV 15/15 ✓, repair re-baseline ✓,
live and trading. The 16 other archetypes remain research scaffolding / data collection.

## What survives for the future
- V14L + live level features: permanent infrastructure (any future level hypothesis is now a
  config test, not a build).
- U3 near-support: the one intriguing near-miss (2022 positive!) — pre-registered watch item,
  revisit at n≥30 holdout equivalent.
- Confirmed again: adding conditions to a validated edge usually subtracts (stack verdict, now this).

## Cross-references
[[audit1_repair_batch_2026_07_13]] · [[wick_trap_cpcv_2026_07_09]] · [[strategy_book_review_2026_07_10]] (H1 resolved: tested, failed) · [[trader_knowledge_failed_breakdown_es_transfer_2026_06_11]]
