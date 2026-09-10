# H3 fixed-event parent permission: bounded research

## Contract and limits

This work asks whether a frozen child setup had a valid parent range, not whether the setup was profitable. It is independent of fusion, native entry selection, exits, sizing and portfolio allocation. The [design](../superpowers/specs/2026-09-10-h3-parent-permission-design.md) freezes two policies:

- Hourly LC: its low breaches the pre-bound parent floor and its close returns strictly inside that same range.
- Minute equal-low sweep: its child level lies inclusively in the pre-bound parent's lower half and its reclaim close lies strictly inside the range.

Parent availability must strictly precede first-sweep bar open. The decision clock is derived from reclaim-bar close after UTC normalization, not trusted from a caller's arbitrary later timestamp. Tightening preserves frozen geometry; a source break of the bound lineage rejects, even if that break used a newer tightened floor. Reformation cannot rescue an already-bound event. This is lineage-lifecycle permission, not a frozen-old-floor survival counterfactual.

Unknown evidence never grants permission. Exact instrument and stream identity are required. All four 4H/1D × N3/N5 parent hypotheses are reported, with no winner selected. No new parent exits, RSI filter or universal higher-timeframe directional veto are included. Source receipts and teacher fidelity remain uncertified.

## Frozen hourly population

Source: the existing 240-hour current-native replay, June 10–20, 2026, and existing June 1–July 1 V23 parent 720-hour ledgers. The **38 events** are earliest native LC `structural.passed is True` records with `reason=structural_passed`, before cooldown, gates, fusion, dedup or entry selection. There were no structural errors in those 240 rows. This is not 38 independent positions.

Every replay OHLCV row exactly matched its V23 source row. The original 720-hour parent inputs were rehashed from V23 and matched each ledger's declared input hash. All four manifests use instrument `BTC`, stream `V23_PARITY_saved_hourly`, and input hash `e437bdcf10d5c6d30b80b282d03355ba7e858715663732264c8309504783d1a5`.

The child feature prehistory starts June 10; parent prehistory starts June 1. Parent ATR is the saved V23 column with **unverified formula/receipt lineage**. This mismatch is disclosed; exact OHLCV identity does not imply equal feature warmup or native ATR parity.

Private frozen-event artifact: `results/research_validation_2026_09_10/h3_parent_permission/june_hourly_frozen_events.json`, SHA256 `6e888f51d3205f54015fc0a03fe577f3a061eb5d34a829215f645990703ee374`.

## Frozen minute population

Arm: `minute_same_stream_talib_atr14_parent_reference_v1`. The **76 events** have reclaim-bar opens in June 10–20 and come from the unchanged frozen `minute_next_open/minute_replay.json` event population, already after 60-minute selector spacing. No event detection or four-hour busy simulation was rerun to select these events.

The full 2,979,360-row minute parquet SHA256 matched `5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035`. Baseline replay SHA256 matched `6ed3caee375d8a023dd29173d92e3863b6e355dbd5591a7f8a9899fbbe74ea8a`. Its recorded script SHA256 `d716d68efba33ad88b2ad32b81445961960ad920b0d6b018719903a47884fb44` was recovered exactly from commit `4ce121b`; `detect_events` AST is identical to the present selector. Pivot/confirmation/sweep/reclaim indices and source values were checked, including first sweep/reclaim conditions for the selected events.

Parent inputs aggregate **43,200 contiguous UTC minutes into 720 complete hours**, June 1–July 1, from that exact minute stream. Each hour contains 60 minutes. Instrument `BTC`, stream `btc_1m_2021_2026_saved_5b8a4533f70b8ccd`; the V23 hourly stream is never substituted.

ATR was frozen before permission counts as `talib.ATR(high,low,close,timeperiod=14)`, package 0.7.1, underlying library `0.7.1 (Jul 16 2026 18:35:07)`, available at hour close. The first 14 ATR values remain NaN, with no filling or rounding. June 1 is an explicit cold-start boundary; Wilder initialization is path-dependent. This is a separately declared ATR reference, **not native LFC parity**. Its permission rate must not be compared with the hourly arm as if ATR/prehistory were held constant.

All four parents consume identical hourly/ATR arrays. Private reference artifact: `results/research_validation_2026_09_10/h3_parent_permission/june_minute_frozen_reference.json`, SHA256 `9f9587253250b56d547d2a5af17a2d02bc1a8035e86d5f962eae0b7e0f01ff20`.

## What these populations cannot establish

No pass/reject count is a profitability result. These histories have already been used in research; they are not pristine holdout. Overlapping events are not independent trades. A fixed-event filter does not reproduce native cooldown/dedup/book selection, and the minute population cannot recover candidates already removed by its 60-minute selector. No live setting or production code changed.

Before any strategy conclusion, complete consumed-field provenance, then separately replay selection/lockout displacement with costs, position grouping, causal fills and frozen risk treatment. Retain genuinely new forward evidence after the rule freeze.

## Verified annotation results

Implementation commits `c0929a4` and `2bd6b70`; 46 focused policy tests. Independent task review caught and closed three defects: binding outside the validated causal prefix, duplicate consumed transition IDs, and malformed-event batch exceptions. Root's fresh full research suite after fixes: **265 passed**, one pre-existing urllib3/LibreSSL warning, 11.99 seconds. Production `engine/`, `bin/`, and `configs/` diff is empty for this phase.

June 10–20 fixed event population, parent histories beginning June 1:

| Parent hypothesis | Hourly LC pass / reject (38 events) | Minute sweep pass / reject (76 events) |
|---|---:|---:|
| 4H, N3 | 1 / 37 | 16 / 60 |
| 4H, N5 | 0 / 38 | 19 / 57 |
| 1D, N3 | 0 / 38 | 8 / 68 |
| 1D, N5 | 0 / 38 | 0 / 76 |

All **456 annotations** returned a known rule result; zero `unknown`. This means the supplied evidence satisfied this evaluator's contract, **not** that its receipts, ATR formula or economic validity became certified. There is no starting equity, risk allocation, cost model or PnL in this annotation-only run.

Rejection reasons:

- Hourly 4H/N3: 21 geometry, 5 bound-lineage breaks, 11 absent parents; 4H/N5: 25 geometry, 5 breaks, 8 absent parents. Both daily variants lacked a usable bound parent for every selected hourly event.
- Minute 4H/N3: 36 geometry and 24 absent parents; 4H/N5: 35 geometry and 22 absent parents; daily N3: 68 absent parents; daily N5: 76 absent parents. No bound-lineage break occurred within these selected minute sweep-to-reclaim intervals.

Full 720-hour parent histories and **freshly rebuilt 456-hour prefixes ending June 20** produced exactly equal full annotation records, including IDs, for both arms. Fresh copied-input evaluation also matched; inputs remained unchanged. These are fixed-event/evaluator checks, not native engine selection/restart certification.

Private result: `results/research_validation_2026_09_10/h3_parent_permission/june_fixed_event_annotations.json`, SHA256 `a60cc953a80c8239cb90e136b2758c5460df9d60b3cebdce54882ebd483acec6`. The artifact records policy source hash, frozen input hashes, every annotation, reason counts, and checks.

Interpretation: this particular parent-floor-reclaim subtype is **very restrictive for hourly LC** in the inspected population. It is not ready to become a universal LC gate. Daily coverage is strongly dependent on bounded prehistory and anchor confirmation; an absent parent is not evidence that the trade would lose. The minute variants admit different subsets, but a larger admitted count does not identify the better strategy. Do not loosen thresholds or choose an anchor because these counts look attractive. Preserve baseline archetypes and advance only with separate coverage and economic validation.
