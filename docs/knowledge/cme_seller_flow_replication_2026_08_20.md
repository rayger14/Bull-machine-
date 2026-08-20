# CME Seller-Flow Replication Study — 2026-08-20

**Data**: CME BTC futures via Databento (~$11 of $40 credit): 1m bars 2021-01→2026-08 (`data/databento/btc_fut_1m_2021_2026.parquet`), 875K tick trades w/ aggressor side 2026-02→08 (`btc_fut_trades_2026.parquet`). Basis vs our spot store: mean +0.41%, std 0.29% — data real/aligned.
**Question**: does wick_trap's seller-flow boost (1.25× on taker_imbalance≤0 flushes, validated once on OKX/Binance-Vision data) replicate on independent institutional flow?

## Q1 — Does the live signal match real flow? NO, and the reason is a LIVE BUG
Live `taker_imbalance` vs CME hourly aggressor imbalance (1,255 overlap hours, Jun–Aug 2026):
**pearson r = 0.017 (p=0.56), sign agreement 52% = coin flip.** No lag (−3..+3h) or smoothing (4/8/24h) fixes it.
**Root cause found** (`live_feature_computer.py:2395`): live taker_imbalance = `(ratio−1)/max(ratio,0.01)` on OKX `fetch_all_current` — a once-per-hour POINT SNAPSHOT. autocorr(1h)=0.007 (white noise), std 0.27.
The VALIDATED store feature is different: Binance Vision `sum_taker_long_short_vol_ratio`, a true 1h aggregate — autocorr 0.125, std 0.075, 28% of bars ≤0 (`derivatives_data_backfill_method.md`).
**⇒ The live boost condition fires on ~white noise that is NOT the variable the boost was validated on.** Same class of live/backtest divergence as audit #7 (A2). Sensor repair needed: aggregate OKX taker volume over the full hour (or persist+integrate snapshots) to match the store definition.

## Q2 — Does the THESIS replicate on institutional flow? DIRECTIONALLY YES (underpowered)
CME 1h bars Feb–Aug 2026, wick-trap-shaped flushes (lower_wick≥0.35, vol_z≥1.5, 20-bar low probe): n=37.
| cohort | n | fwd 72h | fwd 168h |
|---|---|---|---|
| flush + CME SELLER aggression | 25 | **+0.29%** | **+0.41%** |
| flush + CME buyer aggression | 12 | −0.52% | −1.73% |
| baseline (all bars) | 3510 | −0.30% | — |
Binary split matches the boost thesis on a venue and months never used in any prior study. Caveats: n small; graded version flat (spearman 0.04, p=0.80) — it's a sign effect, not a dose-response.

## Q3 — Live wick_trap entries vs CME flow (n=7, anecdotal)
Winners (n=2) mean CME imbalance **−0.088** (sellers capitulating); losers (n=5) **+0.137**. The worst entry (08-16, −$602) had CME imb **+0.89** — extreme buyer aggression at the "flush" = trap signature. Direction consistent with Q2 and the original study.

## Verdicts
1. **Thesis: REPLICATES directionally** on fresh institutional data (Q2+Q3 agree with the OKX-era validation). The seller-flow idea is market structure, not a venue artifact.
2. **Live implementation: BROKEN** — the boost keys on a snapshot that doesn't measure hourly flow. The boost's live effect to date is essentially random sizing noise on wick_trap. Fix is a sensor repair (hourly aggregation), then the validated condition actually binds live.
3. Boost-shaped follow-up (after repair): none needed immediately; re-validate the boost's live hit-rate once the repaired feature accumulates ~4-8 weeks.
4. NOT proposed: any filter/gate (Rules 7-10), any CME live feed (cost; revisit only if the repaired OKX aggregate still diverges from CME on a longer overlap).

*Method note: all tests on fresh data (CME Feb–Aug 2026 + live logs); zero spent folds reused. Study data retained in data/databento/ (purchased, reusable); scratch deleted.*
