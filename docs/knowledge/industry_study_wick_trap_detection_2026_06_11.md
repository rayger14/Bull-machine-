# Industry Study: How Real Systems Detect Wick Traps / Liquidity Sweeps / Stop Hunts

**Date**: 2026-06-11
**Method**: Deep-research workflow — 103 agents, 21 sources, 99 claims extracted, 25 adversarially verified (22 confirmed, 3 refuted 0-3).
**Purpose**: Rebuild the wick_trap detector from first principles after [[liquidity_score_root_cause_2026_06_10]] invalidated its calibration.

## Headline: our detector is built backwards relative to the field

Our current wick_trap = wick-shape primary (35% of range) + volume gate + liquidity gate, with **no location condition**. Every surviving real-world implementation does the opposite:

| | Our detector | Field standard (smartmoneyconcepts lib, LuxAlgo ×2) |
|---|---|---|
| Primary trigger | wick ≥ 35% of candle range, ANYWHERE | price trades beyond a PRIOR LIQUIDITY LEVEL and closes back inside |
| Wick-ratio threshold | hard gate at fixed 35% | **none anywhere** — wick shape is a byproduct, not a condition |
| Location | none | required (swing pivot, equal-high/low pool, level) |
| Confirmation | none (enter on signal bar) | close-based break-of-structure / follow-through |

No published implementation or study provides ANY evidence-backed wick-ratio constant — there is no industry anchor to recalibrate 35% against. The field treats ratios as tunable strictness dials, never gates.

## Verified findings

1. **Classic candle geometry alone ≈ no edge** (3-0). TA-Lib's pattern functions are pure OHLC with no context; the Zorro platform ships all ~61 patterns while stating "no serious tests found any predictive value in any of the patterns." A vendor disrecommending its own feature.
2. **Field-standard sweep primitives** (3-0, from source code): swing highs/lows via symmetric pivot windows (smartmoneyconcepts default 50 bars — NOTE: 50-bar lag/repaint, engineer around it); equal-high/low pools clustered by tolerance (1% of range, or ATR-scaled margin in LuxAlgo); sweep = trade-through of the pool; BOS/CHoCH confirmation defaults to CLOSE-based breaks, not wick pokes.
3. **The best academic evidence inverts the naive trade** (Osler, JIMF 2005, 9,655 real dealer orders; 3-0): stops cluster predictably past round numbers (14.3% vs 6.9% asymmetry); when price hits a stop cluster the move ACCELERATES into a cascade lasting HOURS (continuation, not reversal); reversals occur reliably at take-profit clusters exactly ON levels (59.3% vs 54.8%, p<0.001). **Design consequence: never enter on the sweep bar — require cascade-exhaustion confirmation. The reversal edge lives at the level; the continuation risk lives in the cascade.** (Caveat: 1999-2000 FX data; crypto liquidation cascades are the analogous mechanism, by extrapolation.)
4. **Candle structure DOES carry signal on crypto 1H** — as a feature, not a trigger (medium confidence, sources conflict by timeframe): 2026 IREF study (hourly, ~400 cryptos, ~200M obs, data-snooping-corrected) finds Bullish/Bearish Harami, Hikkake, Hanging Man retain significant predictive power; IEEE 2021 study on DAILY bars finds 68 patterns "of little use." Hourly evidence ends Jan 2022 — regime persistence unproven.
5. **Volume confirmation: UNANSWERED.** No order-flow/liquidation-cascade claims survived verification, and the one claim that the hourly edge was volume-independent was REFUTED 0-3. Our `volume_zscore >= 0` gate has neither support nor refutation — must be validated in-house (as a soft feature, per Rule 8).

## Recommended detector design (BTC 1H) — to be WFO/CPCV-validated on the live-path store

1. **Level inventory**: confirmed swing highs/lows (pivot window well below 50 bars to cut lag; provisional pivots flagged), equal-high/low pools (ATR-scaled tolerance), round numbers (Osler), optionally FRVP POC/value-area edges (already computed in our store).
2. **Sweep event**: bar's extreme trades beyond a level intrabar AND closes back inside (LuxAlgo criterion). Second archetype optional: close-based outbreak → later retest with opposite wick.
3. **Confirmation (mandatory, the anti-cascade guard)**: do NOT enter on the sweep bar. Require a follow-through bar closing back beyond the swept level, or close-based BOS in the reversal direction. Trend context: sweep-of-lows only at end of a decline.
4. **Invalidation**: close beyond the sweep extreme (stop past wick tip) + time-stop if no displacement within N bars.
5. **Soft scoring features (never hard gates)**: wick-to-body and wick-to-opposite-wick ratios (two dials, per TradingFinder pattern), volume z, taker delta, OI flush — all empirically tested through the boost/WFO machinery.

## Refuted claims (do not cite)
- "Real systems detect pin bars via TA-Lib named patterns" (0-3 — they use level logic instead)
- "TA-Lib exposes no tunable thresholds on single-candle patterns" (0-3)
- "The hourly crypto candlestick edge is volume-independent" (0-3)

## Open questions
- Does liquidation/OI-flush data separate genuine stop hunts from noise on BTC 1H? (Our whale features may answer this in-house.)
- Optimal confirmation lag on 1H given hours-long cascades — bars of follow-through vs missed entry cost.
- Post-2022 persistence of the hourly candle edges; net-of-costs viability.
- Session/time-of-day concentration (ICT "killzones") — no verified evidence either way.

## Cross-references
- [[liquidity_score_root_cause_2026_06_10]] — why the old detector died
- [[industry_study_backtest_live_parity_2026_06_11]] — the parity discipline any new detector must be built under
- [[champion_strategy_pair_2026_06_10]] — wick_trap's V12-space record (PF 2.36) that motivated saving the archetype
