# Fresh-Data Battery — 2026-08-30

**Infrastructure (permanent):** store extended to 2018→2026-08 (74,436 bars):
Coinbase candles (0.000% divergence vs V12 lineage on 1,441-bar overlap), 4
parity shards via production LiveFeatureComputer (live-restart warmup semantics),
witness splice from Binance Vision **bit-exact vs trusted 2024 values**
(agent-verified; key gotcha: Binance re-labeled metrics archives by +5min;
store formulas beat live formulas where they disagree — backfill script is the
provenance). Path: `data/features_mtf/BTC_1H_FEATURES_V23_PARITY_2018_2026.parquet`.
Flagged separately: FEAR_GREED / fear_greed_norm / funding_Z are frozen
placeholder constants (V12 legacy) — do not consume.

**The gate:** 2025-01 → 2026-08 = 20 months with ZERO role in any rule's
discovery. Assertions pre-registered before each run.

## Verdicts

1. **LC volume floor 1.5→2.0 (PR #76): PASS, third independent confirmation.**
   Witness-blind: −$7.6K→+$1.7K. Witness-filled: −$6.4K→+$2.2K (PF 0.76→1.11).
   The weak-volume class loses on unseen data exactly as on 2020-24.
2. **wick_trap buyer-aggressed boost: FAIL — dead.** Frozen threshold from
   discovery years; boosted class on fresh = 25% WR, −$8,235 (worse than rest).
   Survived tercile bar AND CPCV 12/15, then died on fresh data = selection
   bias caught. No sensor build. One re-test allowed ONLY if witness splice had
   materially changed the trade list (it didn't: 55→53 positions).
3. **Graduate re-validation: the book is regime-dependent.** 7-graduate book
   on fresh 20 months = **−$32,735** (witness-blind and witness-filled nearly
   identical — sensors were not the story). Shape: 2025 = −$34.6K bleed;
   2026 = +$1.9K flat-green (wick_trap +$3.2K, FC +$2.2K, LC-floored +$5.7K,
   oi_div +$1.4K). Edges made +$214K on 2020-24, bled through 2025, partially
   returned 2026. CB is the worst decay (PF 0.47 fresh vs 1.70 grad). 
   oi_divergence honestly measured: PF 1.01 flat (witness-blind PF 3.77 was
   INVALID — OI gate skipped on NaN).
   **This is the fresh-data confirmation of the Aug root-cause: the missing
   piece is a regime stand-down dial, not better entries.**
4. **Loss-triggered stand-down K=2/48h→24h pause (book-level): +$6,165 on
   fresh** (cuts 52 trades worth −$6.2K), second independent positive dataset
   (live cohort +$4,977). Trims, does not cure (2025 still −$30K). K=3 variant
   NEGATIVE (skips winners) — effect is specific, not "any throttle."
   Exploratory 48h-pause variant +$9,952 — NOT pre-registered, flagged only.
   Caveat: trade-level sim (no sequencing), book-level pooling of silos.

## Decision queue
- PR #76: evidence complete (3 confirmations incl. fresh) — user call.
- Stand-down: promote from sim to real engine A/B (needs a book-level
  loss-tracker in the backtester + runner; shadow first).
- CB: fresh PF 0.47 — graduation status in question; propose review.
- Next fold discipline: 2025-26 is now PARTIALLY SPENT for anything tested
  above; the golden-master replay and live shadow remain the only untouched
  validators.
