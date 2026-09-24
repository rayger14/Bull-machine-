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

---
## RESULT (run 2026-09-24, once, no re-rolls): **PASS 4/4**

| Check | Result |
|---|---|
| 1. 2018-19 RSI<35 side total < 0 | **−$2,349** (n=12, 33% WR; 2018 −$4,336, 2019 +$1,987) ✅ |
| 2. RSI>65 avg > RSI<35 avg | **+$651 vs −$196** (RSI>65 n=17, 65% WR, both years positive) ✅ |
| 3. Variant beats base on 2018-19 | **$11,273 vs $8,711** (+$2,562); MaxDD −2.17% vs −4.01% ✅ |
| 4. 2020→Aug 2026 sanity | **$41,560 vs $41,512** (+$48, a wash); MaxDD −3.53% vs −4.68% (1.1pp better) ✅ |

Honest reading: on untouched data the capitulation side lost and blocking it added PnL and halved drawdown. On the
discovery period the engine A/B is a PnL WASH (+$48, not the +$2.2K the side-split implied): blocking changes
cooldown timing, and 2024, the one year the capitulation side made money, gave back $2.8K. The durable benefit is
**same PnL with ~21-40% fewer trades and a smaller drawdown**, not extra profit. Doctrine tension: the class is not
negative in EVERY year (2019 and 2024 were positive), so this is a pre-registered pass, not a "negative every era" cut.
n=12 on the test side is small.

---
## Literature check (2026-09-24, research agent; paywalled sources = search snippets only)
**LC after the fix is a recognized setup:** a volume-confirmed volatility-squeeze breakout with momentum-chosen direction.
Closest relatives: Bollinger Squeeze (Method I) + %B/MFI thrust (Method II), TTM Squeeze (Carter) with a volume rule,
Minervini VCP, Darvas box, Donchian/TSMOM (CTA). Every squeeze author picks direction by breakout/momentum; none fades it.
The removed RSI<35 side was a Connors-style dip-buy missing Connors' two essentials: a 200-period trend filter and a fast exit.
No published study tests "high volume + overbought continues, high volume + oversold keeps falling" on crypto 1H; our
pre-registered result is as strong as anything found.

**Pro rules LC lacks, not yet tested (cheapest first):**
1. Relative squeeze: BB width in the bottom X% of its own ~6-month history (Bollinger's definition) instead of a fixed 0.06.
2. Range-high trigger: the spike bar must CLOSE above the squeeze range high (Darvas/Minervini/Bollinger IV), same bar, not a wait.
3. Keltner squeeze (TTM): BB inside Keltner(20,1.5).  4. OBV/CMF rising during the squeeze.  5. Volume dry-up before the spike (VCP).
**Collide with buried lenses:** 200-MA/stage-2 trend filter (HTF/regime gating), multi-timeframe firing (MTF sequencing),
wait-for-pullback entry (confirmation bars, buried 3x), 8-10 bar momentum-fade exit (exit mods 0-for-5; kills the runner).
**Binding constraint:** ~17 LC trades per 2 years. Filters starve the sample; test #1 and #2 as SIZING BOOSTS, pre-registered,
and judge them on live accumulation, since 2018-19 is now spent on the RSI-side test.
