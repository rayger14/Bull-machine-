# LC setup comparison: revealed winners and losers

September14,2026. Descriptive follow-up to the [gate diagnostic](lc_pattern_diagnostic_2026_09_14.md).

## Result

The saved examples show why **direction, structural location and entry sequence
must be considered together**. They do not yet supply a reliable winning rule.
Upside expansion wins in one small study and loses in another. A minute rebound
appears before a winning downside-exhaustion case, but also before a losing
upper-range rejection. No new threshold, entry simulation or model assessment
was run. These are old outcomes joined to their original decision-time evidence.

Completed deliverable: a source-linked table of11 reconstructed hourly LC cases
across three separate studies, plus37 recorded live LC groups. The live groups
are not certified closed positions and cannot currently be joined reliably to
full nested-structure inputs. Do not combine these into48 independent trades.

The private machine-readable table is
`results/lc_pattern_table_2026_09_14/table.json`. It includes source hashes,
raw annotation operands, separate cohort summaries, data gaps and the already
completed April immediate-versus-wait comparison. No frozen input was modified.

## Cohort separation

- **Layered March/May/July:** six selected cases, same Binance1m reconstruction,
  immediate minute open, independent50knotional/$60cost, fixed stop/2R/24h.
  Historical saved-native comparator selects five; L09 is false there, but its
  diagnostic still contains a pre-winner LC signal. Final selection and native
  detection are different stages; preserve the old comparator's meaning.
- **April:** three native pre-winner LC candidates, same Binance-minute source,
  90s processing, continuous isolated books, fixed stop/2R/24h,50k/$60.
- **June mini:** two selected V23/hourly-stream cases, not the same Binance
  minute reconstruction; zero modeled processing delay. C1 is non-selected and
  fails the saved numeric gate baseline. Do not merge its result into a native
  LC strategy or a same-source minute analysis.
- **Live:**37explicit-ID recorded-exit groups with incomplete closure/cost/source
  provenance. Scores and metadata are recorded; nested structure is not verified.

All these historical periods have been researched already. This is discovery,
not a holdout or a new walk-forward/CPCV result.

## 1. Six layered cases

| Case | Decision UTC,2026 | RSI | Hourly close | 4H parent | Daily parent | Final5m candle | Saved outcome |
|---|---|---:|---|---|---|---|---|
| L01 | 03-06 14:00 | 28.42 | Below prior low | Intact | Absent at strict bind | down | stop / −$1295.95 |
| L02 | 03-15 23:00 | 72.05 | Above prior high | Intact | Intact | up | deadline / +$1317.49 |
| L05 | 05-01 13:00 | 72.95 | Above prior high | Intact | Intact | down | deadline / +$147.14 |
| L06 | 05-23 08:00 | 19.04 | Below prior low | Absent at strict bind | Absent at strict bind | up | target / +$1251.37 |
| L09 | 07-03 21:00 | 73.45 | Above prior high | Absent at strict bind | Intact | down | deadline / +$298.11 |
| L10 | 07-15 14:00 | 65.20 | Back inside prior range | Absent at strict bind | Intact | up | stop / −$841.21 |

All6 pass the study's numeric selection. All6 old assessors rejected. Numeric
cases total+$876.95 (4positive/2negative); saved-native five total+$578.84
(3positive/2negative). These are sums of independent cases, not a portfolio.

“Absent at strict bind” means no algorithmic parent strictly before setup;
it does not mean missing candles or no range could form later. L01 daily and
L09 4H have new lineages at the setup boundary; L09's new4H subsequently breaks
up by decision. Old bound-parent geometry must not be attached to a later range.
L10 is a prior-high sweep with a close back inside, not an inside bar and not
an upside close expansion despite RSI>65.

### A useful within-pattern contrast, and its counterexample

- **L01, downside expansion, loss:** final5m sells from69,932.4 to69,189.6,
  finishing only10.2 above its low. Its bound4H range is intact.
- **L06, downside expansion, win:** penultimate5m flushes to74,203.6; final5m
  makes a higher low74,458 and rises74,481.2→74,573.1. Both algorithmic parents
  are forming, yet the frozen immediate entry later reaches2R.
- **L10, loss:** final5m also rises,65,004.4→65,090.2, but the hourly candle
  rejects the prior high and closes near the bound daily ceiling.

Thus “last5m green” is not a demonstrated filter. The hypothesis is whether
recovery at an identified level, in the appropriate larger structure, matters.
That level/trigger still needs an exact prospective definition; these examples
were compared after outcomes were known.

Parent ceiling distance is likewise not a universal veto. L02 has only0.971R
to its bound4H ceiling and1.119R to daily; L05 has0.026R to4H and1.601R to daily.
Both finish positive at the24h deadline. L01 has2.840R to its4H ceiling and
still stops. R here uses original indicative close minus frozen stop; it is a
geometric distance, not proof the path is unobstructed or the ceiling a target.
A strategy with profitable deadline exits is not equivalent to “must hit2R.”

## 2. April is a counterexample to choosing upside expansion alone

| Case | Decision UTC,2026 | RSI | Hourly close | 4H parent | Daily parent | Final5m candle | Saved outcome |
|---|---|---:|---|---|---|---|---|
| 2026-04-04 | 04-04 16:00 | 66.70 | Above prior high | Intact | Not extracted | Not extracted | stop / −$468.83 |
| 2026-04-08 | 04-08 00:00 | 77.17 | Above prior high | Broken up | Intact | down | deadline / −$670.36 |
| 2026-04-29 | 04-29 19:00 | 32.97 | Below prior low | Broken down | Intact | up | deadline / +$475.83 |

The two upside expansions lose−$1,139.20 combined; the downside expansion
finishes+$475.83. This challenges the tempting3-for-3 upside result in the
layered numeric subset. April and layered execution assumptions differ, so
this is not a controlled regime comparison or a pooled win-rate estimate.
April4 daily/minute fields are not extracted in this bounded table; this does
not assert the underlying archive lacks them.

April8 is a concrete multi-level observation: the4H parent has broken upward,
but the daily range remains intact with ceiling71,999.9. At the simulated
entry71,916.1, that ceiling is just83.8 above entry, roughly0.045 of actual
entry-to-stop price distance. The case loses. This is a source-backed example to
investigate, not a fitted room threshold or proof the ceiling caused the loss.

### The already-run minute-wait comparison

| April decision UTC | Immediate net | Fixed5m-high wait | Difference |
|---|---:|---:|---:|
| Apr4 16:00 | −$468.83 | Expired; no trade | Avoided this observed loss |
| Apr8 00:00 | −$670.36 | −$771.87 | Worse by$101.50 |
| Apr29 19:00 | +$475.83 | +$488.75 | Better by$12.92 |
| Separate book total | −$663.37 | −$283.12 | +$380.25, still negative |

These values are copied from the locked old result, not rerun or newly optimized.
“Fixed5m-high” uses the latest completed5m candle's high as a frozen level.
A fully closed post-arm1m candle must close strictly above it; entry is at
the next minute open. The existing policy assumes90seconds processing (arm
rounded up to a minute boundary), zero routing delay and an exclusive15minute
entry expiry. This is not a newly tested full5m retest sequence.
Waiting is not automatically better: most improvement here
comes from one expired loser, and the remaining loser gets a worse fill.

## 3. June mini examples stay separate

| Case | Decision UTC,2026 | RSI | Hourly close | 4H parent | Daily parent | Final5m candle | Saved outcome |
|---|---|---:|---|---|---|---|---|
| C1 | 06-11 13:00 | 62.01 | Below prior low | Intact | Not supplied | Not extracted | deadline / +$296.81 |
| C2 | 06-14 22:00 | 71.43 | Above prior high | Broken up | Not supplied | Not extracted | target / +$1323.50 |

C2 is a known successful upward break of its bound4H ceiling; the older agent,
code and numeric baseline all selected it. C1 is a profitable counterfactual
outside that baseline (RSI62.01 and chop0.5261). Neither proves an agent edge or
that failed gates should be removed. No minute confirmation outcome was measured
for these hourly-stream cases.

## 4. Live group appendix: what is known versus missing

All rows below have `completion_certified=false`. Net-like dollar values are
recorded exit subtotals, not fully net trade profits: complete original quantities,
entry commissions and position funding are missing. Exit-leg count is an outcome,
not an entry-time feature. Nested-parent state and minute trigger are unknown
for this ledger join; they were not filled from a different venue or guessed
from approximate factor attribution. Regime labels are stored labels, not newly
validated Wyckoff phases or historical source-version attestations.

| Entry UTC | Stored regime | Fusion | Margin | Exit legs | Recorded subtotal |
|---|---|---:|---:|---:|---:|
| 2026-03-06 20:00 | bear | 0.1965 | -0.2816 | 1 | −$1371.02 |
| 2026-03-27 14:00 | bear | 0.4900 | 0.0158 | 3 | +$668.45 |
| 2026-03-29 22:00 | bear | 0.3877 | -0.1275 | 3 | +$763.31 |
| 2026-04-05 22:00 | neutral | 0.3950 | -0.0325 | 3 | +$580.66 |
| 2026-04-14 13:00 | neutral | 0.2665 | -0.1193 | 1 | −$930.51 |
| 2026-04-28 13:00 | neutral | 0.3019 | -0.1364 | 3 | +$621.52 |
| 2026-04-29 18:00 | neutral | 0.3134 | -0.1419 | 4 | +$1459.33 |
| 2026-05-01 12:00 | bull | 0.2860 | -0.0766 | 4 | +$1006.68 |
| 2026-05-05 12:00 | bull | 0.4137 | 0.1324 | 3 | +$334.70 |
| 2026-05-10 15:00 | bull | 0.2959 | 0.0455 | 1 | −$416.46 |
| 2026-05-12 13:00 | bull | 0.3033 | -0.0401 | 2 | +$211.43 |
| 2026-05-14 14:00 | bull | 0.2799 | -0.0601 | 2 | −$28.04 |
| 2026-05-18 00:00 | neutral | 0.6141 | 0.1878 | 1 | −$704.40 |
| 2026-05-22 19:00 | neutral | 0.4734 | 0.0263 | 1 | −$782.91 |
| 2026-05-23 20:00 | bull | 0.4348 | 0.0921 | 1 | −$787.72 |
| 2026-05-28 13:00 | neutral | 0.3430 | -0.0950 | 2 | −$761.72 |
| 2026-06-01 12:00 | bear | 0.3224 | -0.1454 | 1 | −$798.84 |
| 2026-06-02 13:00 | bear | 0.5152 | 0.0603 | 1 | −$1022.03 |
| 2026-06-11 17:00 | neutral | 0.2339 | -0.2321 | 4 | +$1818.19 |
| 2026-06-23 13:00 | bear | 0.2682 | -0.2960 | 1 | −$1017.79 |
| 2026-06-24 15:00 | bear | 0.3647 | -0.2277 | 3 | +$798.53 |
| 2026-06-30 12:00 | bear | 0.4999 | -0.0430 | 4 | +$1724.07 |
| 2026-07-02 14:00 | bear | 0.4227 | -0.0697 | 3 | +$909.05 |
| 2026-07-04 15:00 | neutral | 0.4043 | -0.0113 | 3 | −$118.98 |
| 2026-07-08 14:00 | bear | 0.3902 | -0.1417 | 4 | +$1611.02 |
| 2026-07-10 13:00 | neutral | 0.3725 | -0.0593 | 1 | −$847.80 |
| 2026-07-15 12:00 | neutral | 0.3980 | 0.0205 | 1 | −$811.95 |
| 2026-07-21 13:00 | bull | 0.5277 | 0.1652 | 1 | −$787.94 |
| 2026-07-24 15:00 | neutral | 0.2478 | -0.2282 | 3 | +$595.91 |
| 2026-07-26 22:00 | neutral | 0.5397 | 0.1218 | 1 | −$444.34 |
| 2026-07-28 13:00 | neutral | 0.2667 | -0.2063 | 3 | +$777.61 |
| 2026-08-11 15:00 | neutral | 0.4050 | -0.0266 | 2 | −$478.46 |
| 2026-08-14 13:00 | neutral | 0.2608 | -0.1736 | 4 | +$1035.24 |
| 2026-08-17 14:00 | bull | 0.2302 | -0.1333 | 3 | +$374.76 |
| 2026-08-18 14:00 | bull | 0.3353 | 0.0351 | 4 | +$3720.92 |
| 2026-08-19 12:00 | bull | 0.3740 | 0.0577 | 4 | +$3546.49 |
| 2026-09-08 13:00 | bull | 0.2651 | -0.0668 | 3 | +$721.73 |

37groups:20positive/17negative,+$11,168.69. FiveAugustgroups contribute+$8,198.95.
Median fusion is0.32435 for positives and0.398 for negatives. This supports
questioning a single score as the explanation, not choosing a lower cutoff.
No hidden outcome join was manufactured from the rolling signal log.

## Next executable experiment to specify

Prioritize **context-conditioned confirmation**, not another universal score:

1. Keep LC candidate detection fixed and retain upside expansion, downside
   expansion and upper-rejection identities separately. Preserve stage ownership.
2. Separate reliable data from context. Break direction, parent lifecycle,
   identified local level and larger opposing boundary are facts/annotations;
   absent parents need explicit alternative thesis or abstention, not invented
   geometry. Do not automatically mark every upward parent break adverse.
3. Define one level-based reclaim/retest trigger for the applicable subtype,
   with fixed confirmation clock, expiry, executable fill, invalidation and
   common economics. Compare with immediate entry in independent chronological
   books, including expiries, losers and missed winners.
4. Give a small outcome-hidden agent comparison the same facts and legal plans.
   Code verifies the evidence; the agent judges which supported structural
   interpretation applies. A rule allowing only reject cannot measure that.

This report recommends the next specification; it does not implement those new
rules, select an optimized threshold or claim validation. Trial list and later
chronological test need freezing before new economic comparisons. Original Q1
policy, sources and all old responses/outcomes remain unchanged. No newly
revealed Q1 outcomes, model calls, dependencies, live edits, push or PR.

## Verification

Source-derived annotations checked candle cadence/completion, OHLC sanity and
parent availability/update cutoffs. Group counts are6/2/3, with no duplicate
within-cohort IDs. Root matched previously reported mini inputs/outcomes,
April population/results and layered outcomes hashes. The independent layered
analysis additionally matched all six packet/response hashes, lock/manifest/
hidden references and259 non-price manifest expectations. Raw price archives
were not reread/rehashed in this pass; no execution re-simulation was done.
Final independent quant review approved the assembled table for descriptive
research. It verified all14 source hashes, separate cohort counts and arithmetic,
causal parent labels, saved-native selection-stage distinctions and April wait
comparisons. Two nonblocking wording clarifications were incorporated: the1m
confirmation clock above the frozen5m high, and entry-to-stop price distance.
This approval does not certify trading rules, statistical validity, the unfinished
Q1 harness or profitability; no raw archive rehash or execution rerun was performed.
