# Trader Knowledge: Failed-Breakdown Methodology (ES Engine) → Bull Machine Transfer Map

**Date**: 2026-06-11
**Source**: User's ES (E-mini S&P) engine knowledge — Mancini-lineage failed-breakdown methodology, with the user's own transfer notes for BTC.
**Status**: DESIGN PILLARS for the wick_trap rebuild + system-level changes. Independently triangulated by [[industry_study_wick_trap_detection_2026_06_11]] (implementations + Osler microstructure evidence).

## The knowledge (condensed)

1. Highest-probability long = **failed breakdown**: sweep below an obvious well-tested support (where stops pool) → liquidity taken → **reclaim**. Trade the reclaim, never the break.
2. **Acceptance** distinguishes real trap-reversal from dead-cat pause: time spent holding back above the level.
3. **Levels are earned by structure**: prior session extremes, multi-touch shelves, repeatedly defended swings. **Level quality IS the edge** — same pattern at a weak level loses. Most system intelligence should be in refusing trades.
4. **Classify the day before trading it**: assume range day (fade sweeps) until the market proves trend day behaviorally (levels break and STAY broken; dips bought / rips sold instantly; counter-trade destroyed on first attempt). On confirmed trend days, mean-reversion entries kill accounts: quarter size, stop fading, wait for the failed breakdown at the END of the move.
5. Exits in scales; runner with structural trail. **Stops beyond the swept level**, not fixed distances. Never chase far above the reclaimed level — that gap is where you're the liquidity.
6. User's transfer notes: mechanism transfers to BTC *better* than ES (liquidation cascades = the same trap, leveraged); all MAGNITUDES must be re-derived ATR/vol-relative; session anatomy needs a substitute clock (UTC daily/weekly opens, funding timestamps 00/08/16 UTC, US-equity overlap, weekend holes); shadow-instrument first.

## Triangulation (three independent sources, same machine)

| Principle | ES knowledge | Deep-research finding |
|---|---|---|
| Trade the reclaim, not the break | core rule | LuxAlgo: wick-beyond + close-back-inside; smartmoneyconcepts: close-based BOS |
| Acceptance / don't enter the sweep bar | "time holding above the level" | Osler: stop cascades persist for HOURS; reversal edge is AT levels |
| Level quality is the edge | "levels are earned" | every surviving implementation is level-anchored; no wick-ratio gates exist |
| Range-until-proven-trend | day classification | (our own data: all 2022 + May-Jun 2026 losses = fading trend days/regimes) |
| Structural stops past swept level | stop placement rule | research spec: invalidation = close beyond sweep extreme |

## Concrete application to Bull Machine

1. **wick_trap rebuild detector** (already spec'd, now reinforced): level-anchored sweep + reclaim + ACCEPTANCE confirmation (N bars holding the level — new, from this knowledge) + no-chase guard (max entry distance from reclaimed level, ATR-relative).
2. **Level-quality scoring — the missing edge dimension.** No archetype currently scores its level. Build from existing features: FRVP POC/value-area edges (multi-touch acceptance zones — already computed), swing points + touch counts, prior UTC-day/week high-low (NEW features to add to live computer + store), round numbers. Quality score modulates SIZE (house Rule 8), while presence-of-a-level is detector identity (structural condition, not a post-hoc filter — consistent with Lesson #54, which rejected fusion-score filters, not identity conditions).
3. **Day-type classification (range vs trend) at the daily timescale** — the missing INTERMEDIATE regime between CMI (hours) and macro-200d (months). Behavioral tests per the knowledge: did prior-day level break and HOLD broken? Were counter-moves instantly absorbed? Output: trend-day ⇒ counter-trend archetypes sized ×0.25 (sizing, never a filter). This is precisely the failure mode of: 2022 backtest losses (long-only archetypes fading a trend year), May–Jun 2026 live bleed (buying dips in a confirmed downtrend), and the 2026 holdout losses (5 ER stop-outs in the bear onset).
4. **Structural stops**: wick_trap currently uses atr_stop_mult 4.9 (fixed-distance). Replace with stop-beyond-sweep-extreme + ATR buffer in the rebuild. Live forensics support: stop_loss exits = −$34.3K while everything else was +$17.7K — stops are placed where the market can reach them without invalidating the thesis.
5. **Session anatomy for BTC** (new feature candidates, post feature-logging): UTC daily open/close, weekly open, prior-UTC-day high/low, funding timestamps (00/08/16 UTC), US-equity-hours overlap flag, weekend flag. All concurrent-state (Lesson #41 compliant).
6. **Scale-outs**: already validated in our system (live scale-outs 100% green) — keep.
7. **Shadow-first**: matches the parity study Layer 3 and our standing practice going forward.

## Reconciliation with house rules

- "Most intelligence should be in refusing trades" vs "boosts over filters (0/8)": NOT a contradiction. The 0/8 failures were post-hoc statistical filters on existing archetypes (fusion thresholds, regime gates, CMI tuning). The ES principle is about DETECTOR IDENTITY — a sweep at a strong level IS the pattern; a wick at no level is NOT a degraded version of the pattern, it's a different (worthless) event. Tighten identity at detection; modulate exposure by sizing at the portfolio level.

## Pre-registered hypotheses to test (WFO, on the V14 live-path store once built)

- H1: level-anchored wick_trap (reclaim + acceptance) beats the old 35%-wick detector on live-path features, train AND OOS.
- H2: level-quality score is monotonic with per-trade PnL (tercile analysis).
- H3: trend-day ×0.25 counter-trend sizing improves bear-regime years without damaging bull years (the crisis-as-sizing pattern at daily scale).
- H4: structural stops (beyond sweep extreme) reduce the stop_loss exit bleed vs atr_stop_mult 4.9 with no win-rate cost.

## Cross-references
[[industry_study_wick_trap_detection_2026_06_11]] · [[liquidity_score_root_cause_2026_06_10]] · [[holdout_verdict_2026_06_10]] · [[champion_strategy_pair_2026_06_10]] (200-hour vs 200-day regime gap — the monthly-scale cousin of day-type classification)
