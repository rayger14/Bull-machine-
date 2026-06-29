# Per-Archetype Winner-vs-Loser Forensic — "What Makes Each Archetype Win?"

**Date**: 2026-06-28
**Goal (user)**: make each archetype a better trigger / win more — not cut the book.
**Method**: for each archetype, split trades into winners/losers, measure how well
each entry-state feature separates them (rank-AUC; 0.5 = no signal), on train
(2018-2024) with sign-consistency required on the pristine holdout (2025-26).
**Artifact**: `results/champion_v14/winner_loser_forensic.json`, `scripts/champion/winner_loser_forensic.py`

## Headline findings

### 1. The archetypes ALREADY trigger good trades (high win rates)
Per-archetype train win rate: exhaustion_reversal 78%, wick_trap_v14rq 78%,
liquidity_sweep 76%, trap_within_trend 75%, retest_cluster 73%, spring 65%.
**These patterns win 65-78% of the time individually.** Trigger quality is NOT
the problem. The problem (PF < 1 despite high WR) is that the few losers are
bigger than the wins.

### 2. Entry features barely predict win vs loss — and most "signals" flip OOS
Across all archetypes, entry-state feature AUCs cluster at 0.45-0.60 (≈ coin
flip), and the strong-looking train discriminators almost all REVERSE sign on
the holdout ("OOS flip"). This is Lesson #54 in the data: you cannot tighten
entries on these features to "win more" — the separating power isn't there, and
chasing it produces in-sample illusions (the documented 0/8 filter failures).
fusion_score is explicitly weak/negative except in one archetype.

### 3. The ONE robust, OOS-consistent discriminator is EXIT-side: duration
Winners are held materially longer than losers, OOS-consistently:
- spring: winners 32.1h vs losers 20.2h (AUC 0.658 train / 0.653 holdout)
- retest_cluster: winners 34.8h vs losers 21.2h (0.625 / 0.545)
Duration is determined AFTER entry — it's reverse-causal (winners run, losers
stop out fast). So the difference between a winning and losing trade is made
mostly by MANAGEMENT, not by the trigger. **The lever for "win more" is exits
(let winners run, cut losers faster), not entry selection.**

### 4. A few narrow, OOS-real archetype-specific entry edges (boost candidates)
These survived the train+holdout sign check and are worth WFO boost tests
(Rule 8 — size up where they hold, never raw filters):
- **trap_within_trend**: wins with HIGHER trend_align (0.56 vs 0.45, AUC
  0.587/0.625) and FURTHER from support (dist_to_support_atr 2.51 vs 1.54,
  0.589/0.619) → "enter with trend, not right at support."
- **liquidity_sweep**: fusion_score actually positive here (0.59/0.616) and
  winners have FEWER concurrent sweep events (cleaner setups).
- **spring**: higher threshold_margin + wyckoff_score (mild, 0.58/0.53).
Level-quality (level_quality_low) did NOT broadly discriminate — but archetypes
don't currently condition on levels, so this needs build-and-WFO, not observation.

## What this means for "make archetypes win more"
The data says the win/loss outcome is ~70% determined by trade MANAGEMENT, not by
the entry trigger (which is already 65-78% accurate). Three honest avenues:
1. **Exit / winner-capture overhaul (highest leverage)** — let winners run (later
   scale-outs, runner-stop tightening after first scale-out), cut losers. Now
   DOUBLY evidenced (duration finding + the PF<1-despite-67%-WR live signature).
2. **Boost the 2-3 OOS-real entry edges** above, via WFO (small, additive).
3. **Build level-conditioned entries and WFO them** (the ES "level quality" thesis
   — untestable by observation since archetypes ignore levels today).

## Honest caveats
- Holdout n per archetype is 39-95 (directional, not conclusive); funding_divergence
  skipped (n=27 < 40).
- duration's signal is reverse-causal — it diagnoses the exit lever, it is not an
  entry feature to be used predictively.
- All entry-edge candidates are hypothesis-generating; survive only if WFO boosts confirm.

## Cross-references
[[risk_overlay_verdict_2026_06_27]] (cut-the-book + overfit traps) ·
[[v14_champion_hunt_2026_06_11]] · [[trader_knowledge_failed_breakdown_es_transfer_2026_06_11]] (level quality)
