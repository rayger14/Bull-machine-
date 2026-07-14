# Trailing-Start Sweep — REJECTED (Monotone Overfit Gradient); Exit Frontier Closed

**Date**: 2026-07-13. **Question**: does starting wick_trap's trailing stop earlier (protecting the
+0.4R→1R giveback window) improve it? **Artifacts**: `results/trailing_sweep2.log`

## Dead-config finding first (new entry for the epidemic list)
YAML `exit_logic.trailing_start_r` is DEAD — three YAML variants produced bit-identical backtests.
It is read into ArchetypeInstance metadata (isolated_archetype_engine:347) but never reaches
ExitLogic's rules. The operative value comes from ExitLogic defaults/create_default_exit_config
(behaviorally 1.0R for wick_trap). This silently corrects the stack-verdict adjudicator's premise
("the v14rq YAML sets trailing_start_r: 1.0") — right value, wrong mechanism. The WORKING override
path is the user-JSON `exit_logic.<archetype>.<param>` (fixed 2026-07-07).

## Sweep results (working path, V14L, thresholds enforced)
| trailing_start_r | train PF | mid PF | holdout PF | holdout PnL |
|---|---|---|---|---|
| 1.0 (operative baseline) | 1.37 | 1.70 | **1.43** | +$6,764 |
| 0.75 | 1.48 | 1.76 | 1.31 | +$4,803 |
| 0.5 | 1.56 | 1.75 | 1.24 | +$3,638 |
| 0.25 | 1.76 | 2.09 | **0.96** | −$620 |

**Monotone overfit gradient**: every step earlier improves train (1.37→1.76) and degrades the
pristine holdout (1.43→0.96) — the identical signature as the wider-stops trap
([[risk_overlay_verdict_2026_06_27]]). Earlier protection clips winners that dip before running;
in-sample the clipped losses dominate, out-of-sample the clipped winners do. **REJECTED. Keep 1.0R.**

## The bigger conclusion: the optimization frontier is CLOSED
Every evidence-motivated lever has now been honestly tested against the holdout:
entry filters (0/8 + 3 fusion inversions) · entry sizing (0/13) · stop width (overfit) ·
breakeven (no-op) · level anchoring (0/3) · edge-within screens (4×, nothing) ·
regime skip (kills the edge's own winners) · trailing start (this — overfit gradient).
**wick_trap exactly as configured is a local maximum in every direction we know how to test.**
The remaining sources of new information: (1) LIVE evidence accumulating (~1/wk), (2) the U3
near-support watch item at n≥30, (3) a V15 store rebuild backfilling the repaired BOS emitters
(may revive SMC-dependent archetypes), (4) structurally NEW instruments (short side) — not tuning.

## Cross-references
[[wick_trap_edge_within_2026_07_13]] · [[unified_strategy_verdict_2026_07_13]] · [[risk_overlay_verdict_2026_06_27]]
