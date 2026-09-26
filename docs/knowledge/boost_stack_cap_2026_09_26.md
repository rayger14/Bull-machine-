# Boost-stack cap — PRE-REGISTRATION (locked 2026-09-26, before any run)

**Defect:** 7 sizing boosts multiply with no ceiling (1.25 x 1.5 x 1.25^5 = 5.7x max; capex_mult up to 3.05x lifts
the per-position cap). Each boost was validated ALONE; the stack was never tested. All 7 are enabled live
(champion_paper.json). All recent silo studies ran with boosts OFF.

**Change under test:** config key `boost_stack_cap` (float). After all boosts: if multiplier > cap, rescale
allocated_size_pct by cap/multiplier, set multiplier = cap, and clip capex_mult to cap. Unset = current behavior (byte-identical).

**Setup:** full live book = configs/champion_paper.json (all boosts ON, live archetype dir incl. the PR #84 LC gate),
V23 store, 2020-01-01 -> 2026-08-30, commission 0.0002, slippage 3bps. Runs: baseline (no cap), PRIMARY cap = 1.5,
secondary cap = 2.0 (reported, not used for the decision). Eras: 2020-21, 2022-23, 2024-Aug26.

**PASS (primary cap 1.5) requires ALL:**
1. Full-period return/|MaxDD| better than baseline.
2. Return/|MaxDD| better in >= 2 of 3 eras (era DD measured on the equity curve inside the era window).
3. Full-period PnL not lower than baseline by more than 15%.
Also reported: the multiplier census (how many trades at each stack level and their PnL).
No re-rolls; the cap value is fixed a priori.
