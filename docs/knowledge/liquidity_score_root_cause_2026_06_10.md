# Root Cause: liquidity_score Scale Mismatch Makes wick_trap + trap_within_trend Structurally Dead in Live

**Date**: 2026-06-10
**Status**: ROOT CAUSE CONFIRMED via instrumented backtests. Fix requires re-quantiled thresholds on live-path features (user approval needed for YAML changes).

## The mechanism (fully traced)

`ArchetypeInstance.detect()` has a liquidity cliff AFTER the fusion threshold
(`archetype_instance.py:862`): `if liquidity < self.config.min_liquidity: return None`.
`_get_liquidity_score()` prefers the precomputed `liquidity_score` column.

The two pipelines produce **different scales** for that column (Aug–Oct 2024, same market):

| | V12 store (old builder) | Live path (segment) |
|---|---|---|
| mean | 0.492 | 0.221 |
| q90 | 0.721 | 0.437 |
| max | 1.000 | **0.675** |
| P(≥ 0.72) | 10.2% | **0.00%** |

`wick_trap` and `trap_within_trend` both set `liquidity_threshold: 0.72` (≈ V12's q90).
On live-path features the score **never reaches 0.72 — the threshold is unreachable by
construction**, in 2024 H2 and in all of 2025–26.

## Confirmation chain

1. Instrumented backtest (Aug–Oct 2024): on LIVE features wick_trap computes fusion 286×
   (gates pass MORE often than V12's 158×) yet the engine returns **0 signals**; on V12 it
   returns 9 → 17 trades. The drop point is the min_liquidity check.
2. The only two archetypes with `liquidity_threshold > 0.675` (wick_trap,
   trap_within_trend) are **exactly the two archetypes absent from all 158 real live
   trades**. Perfect correspondence.
3. These are the #2/#3 backtest earners ($52K trap_within_trend, $39K wick_trap,
   2020-24 standalone) — i.e. a large share of backtest PnL was never expressible live.

## Why the scales differ

The live `liquidity_score` (live_feature_computer.py update() step H2) =
0.35·vol_score + 0.25·atr_percentile + 0.20·tf1h_fvg + 0.20·oi_score, where components
are systematically lower than the V12 builder's:
- `atr_percentile` ranks against the 1000-bar live buffer (vs longer window in V12): mean 0.43 vs 0.53
- `volume_zscore` from Binance volume (vs TradingView/Coinbase volume in V12): corr only 0.57
- The deleted V12 builder's exact formula is unknown (bin/build_mtf_feature_store.py, removed)

Precedent: live_feature_computer.py itself documents an earlier instance of this exact
bug class ("This mismatch caused fusion scores to be ~0.19 points too low in live").
It also explains the broader fusion shift (wick_trap fusion mean 0.47 V12 vs 0.27 live)
since liquidity is 40% of wick_trap's fusion weight.

## The fix (principled, in order)

1. **Full-history live-path rebuild (2018→2026)** — now PROVEN FEASIBLE with
   `scripts/rebuild/` (klines from binance.vision 2017+, derivatives 2020-09+, macro
   yfinance; 8-way parallel ≈ 6–17 h compute). Produces a store where train and live
   are the same feature space. This unblocks everything else.
2. **Re-express every threshold as a quantile of the live-path distribution.**
   0.72 was ≈ V12 q90 → live-path q90 ≈ 0.44. Applies to liquidity thresholds,
   fusion entry thresholds, and any gate values that were V12-quantile-derived.
3. **Re-run the champion battery natively on the live-path store** with fresh
   train/holdout splits (candidates that demonstrably fire+win live: liquidity_sweep,
   funding_divergence, exhaustion_reversal — plus wick_trap/trap_within_trend once
   their thresholds are reachable again).
4. Per standing orders: YAML threshold changes need explicit user approval and WFO
   validation before production.

## Do NOT
- Do not "fix" live to match V12 (V12's builder is deleted; live is production truth).
- Do not hand-tune 0.72 → 0.44 and ship without re-validating: the whole threshold
  was calibrated against a distribution that no longer exists; it must be re-learned
  on live-path data via the existing WFO machinery.

## Cross-references
- [[holdout_verdict_2026_06_10]] — the failure this explains
- [[champion_strategy_pair_2026_06_10]] — the V12-space battery results
