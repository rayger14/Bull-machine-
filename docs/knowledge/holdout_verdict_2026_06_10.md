# Holdout Verdict: Champion Pair FAILS Promotion — Pipeline Divergence Is the Root Cause

**Date**: 2026-06-10
**Status**: FINAL for this iteration. Champion NOT promoted to live shadow.
**Artifacts**: `results/champion_overlay/wick_trap__exhaustion_reversal/pair_k025/{holdout,bridge_v12,bridge_seg}/`, `results/rebuild/`

## The result

wick_trap + exhaustion_reversal pair_k025 on the pristine 2025-01-01 → 2026-06-10 holdout
(V13_EXTENDED store, live-pipeline features, data never touched by any optimization):

| Gate (pre-registered) | Required | Actual | Verdict |
|---|---|---|---|
| PnL | > 0 | **−$315** | FAIL |
| PF | ≥ 1.3 | **0.93** | FAIL |
| n | ≥ 30 | 43 | pass |
| MaxDD | ≤ 10% | 2.1% | pass |
| No leg < −3% capital | — | ER −0.3% | pass |

Per year: 2025 +$1,630 (38 trades); 2026 −$1,946 (5 trades, all stop-outs in the bear onset).

## The decisive finding: wick_trap is DEAD on the live feature pipeline

- Holdout: **0 of 43 trades** were wick_trap — the "pair" ran on one leg.
- Bridge test (same 2024 H2 market, two pipelines): V12 features → wick_trap fires 17 trades; live-path features → **0 trades**.
- **Live confirmation**: wick_trap appears in 0 of 158 real live trades (Mar–Jun 2026).

So the replay faithfully reproduces live behavior. **V12-based backtests overstate what the live engine can express** — wick_trap's PF 2.36 backtest record is unreachable in production because the entry path (fusion score vs threshold, NOT the hard gates — wick_anomaly fires at 57.9% on live features) never clears on live-path features. Root cause not yet isolated; candidates: liquidity_score components (atr_percentile mean 0.43 live-path vs 0.53 V12), fusion_smc (0.81 vs frozen 0.5), dynamic-threshold inputs (chop_score 0.30 vs 0.72, adx 37 vs 28).

## Pipeline divergence quantified (2024 H2 overlap, 4,393 bars)

| Feature | corr | gate-level agreement |
|---|---|---|
| close | 1.000 | — |
| oi/taker/LS (derivatives) | 1.000 | — |
| atr_percentile | 0.931 | ER gate ≥0.5: 81.5% (fires 39% vs 55%) |
| rsi_14 | 0.873 | ER rsi-extreme: 79.1% (fires 38% vs 22%) |
| volume_zscore | 0.568 (exchange volume: Binance vs TradingView) | WT gate ≥0: 75.4% |
| wick_anomaly (derived) | — | 96.8% |
| adx | 0.715 (means 37.4 vs 28.3) | ≥14 gate: 91.2% |
| chop_score | 0.645 (means 0.30 vs 0.72) | known divergence |

## What this means (honest)

1. **The champion as backtested does not exist in production.** Its best leg can't fire. The 2020–2024 battery pass was real on V12 features but V12 features are not what live sees.
2. **ER alone is ~breakeven on 2025–26** — capital-preserving (2.1% MaxDD vs the live system's −27% over the same period) but not a winner.
3. **The rebuild itself succeeded and is the durable asset**: `BTC_1H_FEATURES_V13_EXTENDED.parquet` (73,930 bars, 2018→2026-06) with 2025+ on the live path, plus a reusable replay pipeline (`scripts/rebuild/`: klines download, 8-way parallel LFC replay, stitcher, parity gate, merger). Future strategy validation MUST use live-path features for any go-live claim.
4. **Implication for ALL V12-validated results**: every backtest claim (including per-archetype PnL tables in MEMORY.md) describes the V12 feature space. Transfer to live is unproven per archetype until re-validated on live-path features. wick_trap is the proven first casualty; liquidity_sweep and funding_divergence are live-positive (PF ~1.5 each in real live trades), making them the natural next candidates.

## Pre-registered next step (not yet executed)

Re-run the champion battery (standalone, thresholds enforced) on live-path features: candidates = the archetypes that demonstrably fire AND win in live (liquidity_sweep, funding_divergence, exhaustion_reversal) over the full segment + bridge windows. Diagnose wick_trap's silent fusion path only if it's cheap (<half day) — compare fusion components bar-by-bar on a V12-fired timestamp.

## Cross-references
- [[champion_strategy_pair_2026_06_10]] — battery results (V12 space; status now: FAILED holdout)
- [[composite_boost_wfo_2026_06_03]] — earlier documentation of live/backtest distribution divergence
