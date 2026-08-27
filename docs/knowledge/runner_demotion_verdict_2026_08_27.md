# Runner-Slot Demotion + 75/15/10 Final Closure — 2026-08-27

**Build**: config-gated demotion (positions ≤25% of original stop counting toward max_pos) in backtester + shadow runner (branch feat/runner-slot-demotion, NOT merged). Off-switch penny-verified vs package baseline ($304,487.82). Live note discovered en route: bypass mode already skips position limits live — slots bind in BACKTEST only (a fidelity asymmetry worth remembering).

**Stage 1 — demotion alone (current exits): REJECT.** +124 trades admitted, −$17.1K, PF 1.45→1.39, DD −15.8→−17.9%. The 3-slot limit is an accidental quality throttle — marginal trades admitted by freed slots lose net. Third independent confirmation of the scarcity-is-alpha law (Mancini frozen bar_idx −$15.1K when unfrozen; trap concurrency 2+ → PF 0.42).

**Stage 2 — 75/15/10 + demotion: REJECT, and CLOSES 75/15/10 PERMANENTLY.** 2,692 trades (−821 — identical penalty to the no-demotion test), $271.1K vs $304.5K. Root cause finally understood: the turnover loss was never runner tails — it's the SLOW FIRST RUNG. Under 75/15/10 positions hold 100% size until 1R; full-size positions can't be demoted (nor should be). The current ladder's 0.5R first rung is the slot-recycling mechanism. Long-runner exits are intrinsically incompatible with this book's turnover edge.

**Affirmative conclusion**: the current exit ladder (10/30/50 @ 0.5/1/2R) is now positively validated against the full long-runner family, with mechanism. "Donating 11-23R trends" is the accepted cost of a turnover-edge book; trend upside is harvested via the dial's 1.25x post-ignition sizing instead. Do not reopen: 75/15/10 (any variant), runner demotion (standalone or paired), long-runner exits generally — unless the book's edge model changes fundamentally.
