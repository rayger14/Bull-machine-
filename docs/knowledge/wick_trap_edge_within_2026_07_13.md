# Edge-Within-the-Edge Hunt (wick_trap) — No Exploitable Entry-State Discriminator

**Date**: 2026-07-13. **Question (user)**: within wick_trap, what distinguishes winners from losers —
is there another edge inside the edge?
**Method**: repaired wick_trap trade logs (train 2018-22 / mid 2023-24 / holdout 2025-26; 81/61/30
positions), 25 pre-registered features across 4 families (levels, regime, market, wick geometry)
incl. the NEW level features; rank-AUC per window with 3-window sign-consistency required.
**Artifacts**: `results/wick_trap_eww/`

## Findings
1. **The strong train discriminators evaporate OOS — again.** wick_sc (0.653 train → 0.52 hold),
   atr_percentile (0.64 → 0.48), threshold_margin (0.62 → 0.43), dd_score (0.60 → 0.36),
   risk_temp (0.58 → 0.34). Third independent confirmation of the same law.
2. **Level features do not discriminate** (quality 0.37 train — inverted; dist_to_support flat 0.47;
   eq_low_pool and swing_touches FLIP across windows: train inverted, holdout elevated — untrustworthy).
3. **The only 3-window-consistent pair: LOW chop + HIGH adx** ("clean-trend context"), which the
   last-50 LIVE forensic independently flagged (winners adx 27.9 vs 17.8, chop 0.44 vs 0.65).
   Convergent — but weak (holdout AUC ~0.49-0.51, effect →0 OOS).
4. **Pre-registered sizing test of that candidate (×1.5 when adx>25 & chop<0.5): REJECTED.**
   Context covers 73% of bars (not selective); train PF 1.37→1.32 DOWN, holdout 1.43→1.46 up —
   Rule 9 co-move violation, the untrustworthy signature.

## Verdict
**There is no exploitable edge-within-the-edge in observable entry state.** wick_trap's winners and
losers are decided by which flushes bounce — irreducible at entry (now shown on 4 independent
windows with the fullest feature set we possess). The remaining "within" dimensions are:
- **Exit-side**: winners run 48-67h vs losers 10-16h; the +1R-giveback pool; the trailing_start_r
  sweep {0.5/0.75/1.0} remains the one open, cheap, evidence-motivated exit study.
- **U3 near-support** ([[unified_strategy_verdict_2026_07_13]]): watch item at holdout n≥30.

## Cross-references
[[winner_loser_forensic_2026_06_28]] · [[live_50_forensic_2026_07_10]] · [[unified_strategy_verdict_2026_07_13]]
