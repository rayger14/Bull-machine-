# V14 Champion Hunt: Full Results — No Passer Yet; wick_trap Resurrected as Best Honest Asset

**Date**: 2026-06-11
**Substrate**: `BTC_1H_FEATURES_V14_FULL_LIVE_PATH.parquet` (73,829 bars, 2018→2026-06, entirely live-path — first hunt where backtest = live feature space).
**Battery**: 6 candidates × 10 windows (incl. NEW y2018/y2019 + 2025-26 holdout) + 2 re-quantiled variants + the pair overlay. Criteria C1–C5 with C1 now covering ALL years 2018–2024.
**Artifacts**: `results/champion_v14/`, `results/champion_v14_overlay/`

## Verdict: 0 configurations pass all tests on honest data

### Wave 1 — live-proven candidates, standalone (all FAIL)

| Candidate | Full 18-24 | Holdout 25-26 | Fatal flaws |
|---|---|---|---|
| liquidity_sweep | +$11.7K, PF 1.13 | **+$2.9K, PF 1.32** | loses 2021-2024 individually; 18% DD. V12-era $91K/1.59 was substantially feature-scale inflation |
| funding_divergence | +$6.1K, PF 2.23 | −$0.3K | ~4 trades/yr — garnish, not a strategy |
| exhaustion_reversal | +$22.5K, PF 1.23, n=443 | −$1.0K, PF 0.85 | bleeds 2018/2022 bears + holdout |
| retest_cluster | −$0.3K | −$7.6K, PF 0.38 | dead |
| spring | +$12.3K, PF 1.06 | −$0.02K | 27% DD; dead |
| trap_within_trend | **0 trades in every window** | 0 | total lockout confirmed (liquidity_threshold 0.72 unreachable) |

### Wave 2 — re-quantiled (0.72 → 0.432 = identical percentile on live-path scale)

**wick_trap_v14rq — the hunt's big win.** Resurrected from 0 trades to:
- Full 2018–24: **+$49.1K, PF 1.46, n=329**
- **Holdout 2025–26: +$6.8K, PF 1.43, n=70** — profitable on pristine data in the live feature space
- Monster years: 2020 PF 4.13 (+$31.7K), 2019 PF 2.13, 2023 PF 2.02
- Still fails strict: bear years negative (2018 −$8.1K, 2021 −$7.5K, 2022 −$8.7K), MaxDD 14.3%
- **The original wick_trap edge was real — production just couldn't express it.**

trap_within_trend_v14rq: fires again (653 trades) but 2023/2024/holdout all negative — the edge did NOT survive re-quantiling. Likely V12-artifact. Deprioritize.

### Pair overlay (the V12-era champion construction) — FAILS on V14, and the reason matters

Clean run (after catching a stale-cache contamination that initially produced a FALSE PASS — 7/10 windows had silently reused V12 results; isolated output root fixed it):

2018 −$3.7K | 2021 −$10.2K | 2022 −$8.8K | holdout +$1.8K (PF 1.09) | MaxDD 16.4% → FAIL.

**Root cause (Rule 7 split): the legs' regime personalities FLIPPED between feature spaces.**
- V12 2021: WT +$5.5K / ER −$5.3K → overlay (shrink ER in bull) rescued the year.
- V14 2021: WT **−$10.4K** / ER +$0.1K → same overlay now shrinks the winner and keeps the loser.
- V14 2022: ER −$4.5K (its V12-era 2022 strength +$3.3K did not reproduce) — ER is no longer the bear-year complement. **The pair thesis itself was partly a feature-pipeline artifact.**
- Holdout: WT +$3.5K / ER −$1.7K — ER drags wick_trap's clean holdout down.

## State of the hunt

1. **Best honest asset: wick_trap_v14rq standalone.** Holdout-positive, big bull-year engine, known weakness (macro-bear years) with a known treatment class (sizing by regime/day-type — NOT yet validated on V14; the V12-validated overlay does NOT transfer).
2. **exhaustion_reversal**: breadth (n=443) but no honest complementarity claim survives. Re-evaluate only with V14-derived regime profiles.
3. **Methodology lesson (codify)**: regime-complementarity claims are feature-space-dependent. Any pairing/overlay must be derived AND validated on the same store generation. Cache hygiene: overlay/battery output roots must be store-specific (fixed: results/champion_v14_overlay).

## Pre-registered next studies (small, on V14 only)

- S1: wick_trap_v14rq + macro-200d ×k sizing sweep (k ∈ {0.25, 0.5}, 150/200/250d) — does the bear-year bleed compress without damaging holdout? (Single-leg, no pair assumption.)
- S2: wick_trap_v14rq + day_type ×0.25 counter-trend sizing (the level_features classifier built today — H3 retest on entries that HAVE positive expectancy now).
- S3: wick_trap_v14rq + level-quality sizing (level_features.level_quality_low terciles — H2 retest).
- Acceptance for any: train AND holdout co-move, per-leg/per-year reporting, MaxDD ≤ 10%, no new parameters beyond the pre-registered grids.

## Cross-references
[[liquidity_score_root_cause_2026_06_10]] (the lockout this fixes) · [[holdout_verdict_2026_06_10]] (V13 era) · [[champion_strategy_pair_2026_06_10]] (V12-era pair — now known not to transfer) · [[industry_study_backtest_live_parity_2026_06_11]] (cache-contamination incident = exactly the predicted failure class)
