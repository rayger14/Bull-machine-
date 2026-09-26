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

---
## RESULT (run once, 2026-09-26): **REJECTED — the stack is not a leak**

Full live engine, all boosts on, 2020-01 -> 2026-08, $2M wallet, 1,071 positions:

| Run | PnL | MaxDD | Return/DD | 2020-21 | 2022-23 | 2024-Aug26 |
|---|---|---|---|---|---|---|
| Baseline (no cap) | $114,952 | -3.10% | 37,110 | $85,677 | $53,436 | **-$24,161** |
| Cap 1.5x (PRIMARY) | $108,364 | -3.05% | 35,569 | $76,267 | $56,504 | -$24,407 |
| Cap 2.0x (secondary) | $114,500 | -3.07% | 37,328 | $84,308 | $54,950 | -$24,758 |

Cap 1.5: (1) full ratio WORSE, (2) better in 1/3 eras, (3) PnL 94% of base -> **FAIL**. Cap 2.0 is a no-op (1/3 eras).
The cap code was removed (no dead switches); the boost_mult trade-log column is kept.

**Why it failed, from the census:** stacked trades are mostly the GOOD trades.
| Stack | n | PnL | avg | win% |
|---|---|---|---|---|
| 1x (no boost) | 369 | **-$50,513** | -$137 | 37% |
| 1.25x | 528 | +$129,930 | +$246 | 47% |
| ~1.5x | 117 | +$32,029 | +$274 | 50% |
| 1.5-2x | 35 | +$27,972 | +$799 | 46% |
| 2-2.5x | 18 | -$11,075 | -$615 | 39% |
| >2.5x (max 3.66x) | 4 | +$14,954 | +$3,739 | 100% |
The "most-confirmed defect" (Sept 1.25x-boosted losses) was anecdote: across 6.7 years, boosts pick winners.
RETRACTED as a priority.

**Two bigger findings (post-hoc, need their own pre-registration):**
1. **Unboosted longs lose in EVERY era**: -$36.4K / -$8.7K / -$5.4K (n=117/78/174). The boosts behave like the
   engine's real quality selector; the trades that earn no boost are net losers. This meets the "negative every era"
   bar in-sample, but it was found looking, so it is a lead, not a ship.
2. **The live engine lost -$24K over 2024-Aug 2026.** Positive all three eras: liquidity_compression (+$16.8K/+$3.9K/+$17.0K)
   and wick_trap (+$32.0K/+$5.6K/+$4.3K). liquidity_sweep is the biggest earner (+$66.8K) but -$10.4K in 24-26;
   trap_within_trend +$14.3K total but -$19.6K in 24-26. Seven archetypes are net negative over the full period
   (confluence_breakout -$10.8K, spring -$7.1K, retest_cluster -$3.2K, liquidity_vacuum -$2.9K, oi_divergence -$2.4K,
   exhaustion_reversal -$1.5K, funding_divergence -$1.4K). The user decided on 2026-07-23 to keep the junk book at full size for data.
