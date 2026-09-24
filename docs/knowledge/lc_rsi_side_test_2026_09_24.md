# LC RSI-side test — PRE-REGISTRATION (locked 2026-09-24, before any run)

**Origin:** post-hoc finding while autopsying the Sept-23 LC loss. LC's hard gate `rsi_extreme_65` passes RSI >65 OR <35.
On 2020→Aug 2026 (the discovery data): RSI>65 side n=92 +$43.7K (6/7 yrs+); RSI<35 side n=31 −$2.2K (1/6 yrs+).
History: the gate was added 2026-03-13 (commit f1d4f9d, "E identity rewrite") as a direction-agnostic
"exhaustion reversal" confirmation. The two sides were never evaluated separately.

**Hypothesis:** the RSI<35 (downside-flush) side of LC has no edge; the RSI>65 (upside-climax) side carries it.

**Test data (untouched by the discovery):** BTC 1H V23 store, 2018-03-01 → 2019-12-31. LC-only silo, LIVE config dir
(configs/champion/archetypes_v14rq/), commission 0.0002, slippage 3bps, same engine as the discovery run.

**Variant:** baseline LC + one extra hard gate `rsi_14 min 50` (removes the RSI<35 side, keeps the RSI>65 side).

**PASS requires ALL of:**
1. On 2018-19 baseline: the RSI<35 side's total PnL < 0.
2. On 2018-19 baseline: RSI>65 side's avg PnL/trade > RSI<35 side's avg PnL/trade.
3. Engine A/B on 2018-19: variant total PnL > baseline total PnL.
4. Sanity on 2020→Aug 2026: variant total PnL ≥ baseline (−$0 tolerance) AND variant MaxDD no worse by >1pp.

**If 1 or 2 fails:** the split is a discovery-data fluke → BURIED, no live change.
**If 1-3 pass but 4 fails:** interaction effect (cooldown/capacity) → report, no ship.
**If all pass:** recommendation = ship the gate to BOTH config dirs (user's decision); live-shadow optional.
No re-rolls, no threshold tuning (50 is fixed; 35/65 are the existing gate's own bounds).
