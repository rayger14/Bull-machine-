# Audit-1 Repair Batch — Fixes + Re-Baseline Verdict

**Date**: 2026-07-13. **Scope**: the engine-integrity audit's critical findings, repaired and re-validated.

## Repairs shipped
1. **BOS/CHoCH emitters fixed** (`live_feature_computer._smc_features`): `BreakOfStructure` has no
   `direction` attr — `getattr(sb,'direction','')` returned '' forever → flags 0 since inception.
   Now reads `bos_type.value` (the SMC engine's own convention), restricted to CURRENT-BAR breaks
   (sparse events; verified 4% fire rate vs 0% before, direction-sensible). CHoCH derived from
   `new_trend != previous_trend` (the module has no CHoCH concept; the old "'CHOCH' in name" test
   could never fire). NOTE: V14's STORED BOS columns remain 0 — benefits live + next store rebuild.
2. **Impossible thresholds rescaled to documented intent** (both config dirs):
   `wick_exhaustion_last_3b` 1.4→0.47 (sum-vs-mean scale); `wick_lower_ratio` 1.3→0.5
   (wick/body intent on wick/range scale).
3. **Never-validated live-only gates REMOVED** (`rsi_divergence` ×5 archetypes,
   `wyckoff_bullish_score` ×2): absent from V12+V14 (inert in all validation) but binding live
   (96-98% block) — restores live=validated parity. `adx_14` nan_policy→skip (store-replay parity).
4. **`instability`→`instability_score`** (wick_trap, both dirs): gate ACTIVATED for the first time.
5. **`ls_ratio_extreme` live formula** → true 7d rolling z-score (was (ls−1.1)/0.3, never negative;
   funding_divergence's ≤−0.5 gate passed 0% live vs 26% in store).
6. Backtester-dir `liquidity_threshold` 0.72→0.43 (dir divergence); stale champion_paper note fixed.

## Re-baseline verdict (V14, thresholds enforced, repaired configs)
- **wick_trap: VALIDATION HOLDS.** Holdout **identical** (PF 1.43, +$6,764, n=70); full slightly
  improved (PF 1.46→1.50, $49.1K→$51.1K) — the newly-active instability gate trims noise trades.
  The champion survives the repaired engine unchanged. CPCV/battery conclusions stand.
- **liquidity_sweep: DOWNGRADED — its edge was an artifact of the broken gate.** Removing the
  impossible-gate −50% penalty makes it WORSE: holdout PF 1.10→**0.61** (−$8,257). The accidental
  penalty was accidentally protective; the "holdout-positive #2 archetype" status is retracted.
  wick_trap is now the book's ONLY validated edge, unambiguously.
- liquidity_vacuum: still 0 trades everywhere post-rescale (other binding constraints) — remains dead.

## Unified-theory implication
All archetypes are variants of one trade (buy the BTC flush at a level in a favorable regime).
With liquidity_sweep's demotion, the consolidation target is even clearer: ONE canonical
level-anchored flush-reversal built from wick_trap's trigger + level features + regime/cascade
awareness + the validated exit knowledge (H1 in [[strategy_book_review_2026_07_10]]).

## Cross-references
[[engine_integrity_audit_2026_07_10]] · [[wick_trap_cpcv_2026_07_09]] (still valid) ·
[[stack_validation_verdict_2026_07_08]]
