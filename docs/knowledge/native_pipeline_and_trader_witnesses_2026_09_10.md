# Native pipeline and trader-intent witnesses — September 10, 2026

## Accomplished

Implemented the previously separate feature/signal/book layers as one guarded offline source reference. Quant reviewers approved the design and implementation. This remains on the local research branch; production code/configuration, original data and live trading are unchanged.

Verification: **140 research tests pass**, including 12 native-pipeline tests, nine report tests, two execution-assumption witnesses and 11 trader-intent witnesses added at this checkpoint. The existing urllib3/LibreSSL warning remains; no live network path was exercised. Source/configuration diff against checkpoint `b3a4df1` is empty under `engine/`, `bin/` and `configs/`.

`scripts/research/native_pipeline_replay.py` owns one actual `LiveFeatureComputer` and one runner-owned `IsolatedArchetypeEngine`. The adapter advances on each base candle; the actual LFC and book advance only on completed hours. Minute candles are not passed to the hourly indicators as if they were hours.

`NativeSignalBook` observes the runner's original `get_signals` call in place. It returns the original list and signal objects, and copies diagnostics before downstream threshold/fusion/sizing mutations. A second independent signal engine is not constructed. Native exits, maker endpoint, phantom exits, adaptive threshold, detection, filtering and entries retain their order. All 17 archetypes remain represented.

Prehistory advances every layer, including positions and exits. `emit_from` suppresses output, not trading. Restart replays the identical original prehistory into a fresh processor. It does not restore a partial serialized snapshot or reset the book at the reporting boundary.

`scripts/research/native_pipeline_report.py` requires explicit initial cash, commission and slippage. It reports emitted event counts separately from final book state including prehistory. Exit events are not called independent completed positions. It preserves source/input/configuration identities, checks prefixes/restarts, and limits output to repository research-results directories. Source fills and strategy certification remain false even when every comparison passes.

The common observer was extracted into `engine_signal_replay.py`; the standalone adapter retains its own clock/buffer and allocation prohibition. A small manifest traversal correction permits non-file descriptive metadata alongside nested file inventories.

## Verified behavior and safety boundaries

The native fixture was compared over 20 positive synthetic wick hours with both standalone signal observations and an unobserved native book. Selected signals, observed archetype state and checked native book state match; the book records three native exit events. This is a synthetic behavior test, not a return estimate.

Additional tests cover pre-runner diagnostic copies, same-object return, native call order, soft-gate rejection into a phantom followed by an exit, hidden warmup positions, minute carry/observation lineage, changed cash/fee/slippage checkpoint rejection, and failures that prohibit reuse of partially advanced state.

A fresh-process test exposed a dependency import detail hidden by preloaded modules: urllib3 probes local IPv6 support with a socket/bind during import. The offline adapter disables only that optional transport capability probe during guarded LFC initialization, restores the process flag afterward, and binds this policy into its manifest. Actual socket attempts remain prohibited. This does not change any trading formula. The instrumentation is a Python guard, not an OS sandbox; no production CSV, maker ledger, saved state, exchange or live host loop is run.

## Material execution findings

These are source-backed limitations, not a claim that future prices enter the feature computation:

1. The Coinbase host waits until at least 90 seconds after hourly close before fetching and computing (`bin/live/coinbase_runner.py:3187`). Its eventual call passes the candle's hour-open label (`:3352`). The V11 runner's alternative host loop has a different minute-only wait check (`bin/live/v11_shadow_runner.py:2678`); do not describe 90 seconds as guaranteed across all host variants.
2. Native detection uses the completed candle's close as its reference entry (`engine/archetypes/archetype_instance.py:874`). The paper book applies fixed slippage and stores the hour-open timestamp (`v11_shadow_runner.py:1795`, `:1810`). A reference entry at 100.05 labeled 00:00 can therefore rely on information available no earlier than 01:00, with real host processing later still.
3. Existing exits precede new entries. The new position is not retrospectively stopped on the signal candle's earlier extreme. However, a delayed real entry in the following hour could still be evaluated against that hour's entire OHLC, including the pre-entry interval; hourly bars cannot resolve this ordering.
4. The actual-detector gap witness is explicit: native liquidity-sweep stop 94, reference entry 100.05, followed by OHLC 80/82/79/81 produces a native stop exit at **93.953**, above the entire bar's high of 82. A separate controlled-signal witness with stop 95 exits at 94.9525 above a following high of 92. This characterizes the source stop-level fill convention; it is not an executable gap fill.

The pipeline therefore retains `source_fill_not_executable`, `backdated_source_labels` and `host_delay_not_replayed`. Replacing these assumptions requires a separately versioned execution candidate, not a silent reference repair.

## Trader-intent tests: what rules actually distinguish

Eleven additional native-outcome characterization tests are in `tests/research/test_trader_intent_witnesses.py`. LC tests stop at identity and configured gate evaluation; minute tests stop at event selection. A further OI gate witness distinguishes positive OI change from zero/missing inputs. Synthetic combinations do not prove that a real LFC row has those values or that a full book would enter.

| Witness | Current source result | Research question |
|---|---|---|
| Quiet coil; low volume and midrange RSI | LC identity rejects | The name does not mean quiet-coil breakout eligibility. |
| Absorption but volume Z below the configured minimum | Identity passes, hard volume gate rejects | Identity and entry permission are different layers. |
| RSI 25 versus 75, other LC gate inputs fixed | Both pass under a long-only configuration | Is the intended thesis directional capitulation or a broader exhaustion proxy? |
| Same LC terminal vector after compression versus expansion | Same identity/gate outcome | A prior compression sequence is not required by this check. |
| Same LC predicate inputs with close above versus below a parent range low | Same outcome | Parent-level reclaim geometry is not consumed by these predicates. |
| Climax flag plus NaN volume/RSI/BB/chop gate inputs | Identity and gates pass | Missing data is not positive confirmation: skipped gates and derived RSI defaulting must remain visible. |
| Minute first touch outside cluster tolerance | No event | The selector uses the identity of the swept level. |
| Sweep before pivot confirmation | No baseline event | An eventually visible pivot cannot authorize an earlier entry. |
| Wick above level but closes at/below it | No event | Wick penetration is not a body-close reclaim. |
| First close above the level at 30 versus 31 minutes after first sweep | Accepted versus rejected | The first-sweep lifetime is explicit; a new wick does not reset it. |
| Identical minute OHLC with bullish versus bearish parent annotations | Same event | The current minute selector has no parent-context permission rule. |

Source anchors: LC identity `engine/archetypes/logic.py:662`; champion gates `configs/champion/archetypes_v14rq/liquidity_compression.yaml:21`; missing-value/derived RSI behavior `engine/archetypes/archetype_instance.py:32`, `:53`, `:727`; minute selector `scripts/research/minute_sweep_validation.py:36`.

The local founding inventory describes lost level clusters, reclaim speed, location and ordering (`founding_knowledge_archaeology_2026_07_17.md:17`). Its later provenance correction is more precise: fixed structural objects and body-close/wick-break distinctions are documented concepts, while numerical magnitudes are project hypotheses; the Moneytaur corpus is thin (`wyckoff_audit.md:905`). None of these tests establishes teacher-authored numeric thresholds, a mandatory HTF veto, or profitability. Earlier positive minute-study claims remain superseded by the negative causal diagnostic in `research_validation_results_2026_09_10.md`.

## Historical exercises

Both use $100,000 initial cash, 0.0004 commission rate (4 bps) and 5 bps native slippage. Average initial stop risk is not calculated by this diagnostic; no risk-adjusted performance claim is made. Candle availability is assumed at exact close, venue labels are explicitly unverified, external observations are absent, and local model/calibrator fallbacks remain declared. These reused research windows are not pristine holdouts.

- Hourly: June 10 00:00–June 20 00:00 UTC, end exclusive, 240 bars; emit from June 17 after 168 hours of full prehistory. All row/state/contract checks pass at 168 and 239 bars. The first prefix compares zero emitted rows (the prehistory boundary); the second compares 71. `--require-certification` exits 2 deliberately.
- Minute: June 19 00:00–02:00 UTC, end exclusive, 120 bars; two completed hourly book updates. All prefix/restart row, state and contract checks pass at 59, 60 and 119 minutes. The 59-minute prefix contains 59 clock rows but no completed hourly event. `--require-certification` exits 2 deliberately.

Local ignored artifacts: `results/research_validation_2026_09_10/native_pipeline/`. No raw input data or computed artifacts are added to public Git history.

Hourly artifact `hourly_240h.json` SHA-256: `c50fb4831c785c76f354739073ec817dde258349ddd9c6a98a8fb934adfbbb52`. Minute artifact `minute_120m.json` SHA-256: `ba14b27e620afc0d2237585e5873cbcf20eeded71f13e137fbd6f163c521b24b`.

### Decision trace, not historical live attribution

The hourly emitted window contains 1,224 archetype evaluations (72×17), five pre-dedup candidates, three selected signals, two virtual entries and one rejected signal. One emitted exit event is separate from those entry counts. Final full-prehistory book state has one open position, 14 exit events and nine phantom exit events; these are not 14 independent trades. Native cash is $98,426.38 and marked equity $100,211.60, including prehistory and source fee/fill assumptions—not an executable return estimate. Minute input produces 34 archetype evaluations, no selected signal or entry, and unchanged $100,000 native cash/equity.

| Decision available at UTC | Native result | Evidence |
|---|---|---|
| June 17 17:00 | OI divergence rejected into phantom | RSI 56.01 fails its oversold gate; below-threshold bypass enforcement blocks it. |
| June 17 19:00 | Liquidity sweep entered through collection bypass | Adjusted fusion 0.251183 < per-archetype threshold 0.280417; gates pass. Wick trap and trap-within-trend also produce candidates but lose dedup and still arm cooldown. |
| June 18 17:00 | OI divergence entered through collection bypass | Adjusted fusion 0.303824 < per-archetype threshold 0.311265; gates pass although derivatives observations were absent. |

The OI entry exposes a concrete data-quality distinction: the reference defaults OI changes and taker imbalance to zero when its snapshot is missing. Champion OI gates use inclusive `max: 0.0`, so those zero OI values pass; missing direct values also skip under `nan_policy: skip`. The additional predicate test verifies zero/missing OI passes while positive 4h OI change fails. The research report flags absence correctly, but the native entry does not require affirmative observed declining OI. This is not proof that those historical live trades had missing feeds; these are current-source diagnostic decisions on a deliberately OHLCV-only replay.

For decision arithmetic, use the acted entry's per-archetype `threshold` and `threshold_margin`. Native narrative/last-bar fields can instead show the global threshold (0.3704 versus the sweep's 0.2804, and 0.3413 versus OI's 0.3113). Narrative text alone is not the operative gate trace. No source behavior was changed to reconcile it.

### Fixed scheduling-floor price-basis diagnostic

A separate read-only calculation uses only the two fixed entry decisions above. It applies the current Coinbase host's 90-second scheduling floor and assumes zero additional fetch/compute time, then samples the first exact minute-open timestamp at or after that time. Consequently the sampled observation is at +120 seconds, with 30 seconds of grid rounding. It requires every intervening minute; it neither fills gaps nor simulates an order.

| Archetype | Native unslipped hourly reference | Last closed minute close | Sampled +120s minute open | Initial dataset basis | Subsequent minute-price movement |
|---|---:|---:|---:|---:|---:|
| Liquidity sweep | 65,453.18 | 65,481.30 | 65,564.60 | +4.2962 bps | +12.7212 bps |
| OI divergence | 62,603.71 | 62,664.20 | 62,592.00 | +9.6624 bps | −11.5217 bps |

Dataset basis is `(last_minute_close / native_reference - 1) * 10000`; subsequent movement is `(sampled_open / last_minute_close - 1) * 10000`. Total cross-dataset reference differences are +17.0229 and −1.8705 bps respectively. These cannot be attributed solely to latency: the hourly and minute references already differ, venue/instrument pairing is unverified, actual receipt/processing times are unknown, and a minute open is not an executable quote. No fill, P&L, outcome or parameter selection is inferred from two observations.

Ignored output `entry_delay_90s.json` stores the method, exact observation times, input/source hashes and limitations. SHA-256: `b70c3dfa11b6da9c32f20aa5686513992cccb2e3bd13af94729d84c097e56ca4`.

## Next steps

1. Resolve instrument/venue pairing, then define a separate causal execution contract: signal availability, host/compute latency, pending order, first eligible price observation, post-entry stop/target ordering and adverse gap treatment. A same-timestamp next-open convention is an assumption, not a measured execution receipt.
2. Compare that candidate with the frozen native reference on the same decisions. Keep signal changes separate from fill/accounting changes; do not optimize thresholds using this comparison.
3. Turn trader-intent gaps into named hypotheses with explicit level identity, parent/child structure, direction, prerequisite sequence, invalidation and data-quality requirements. Preserve all 17 native archetypes as independent reference identities; no wholesale fusion or disabling.
4. Recover or replace with an explicitly versioned model/input contract the missing artifacts and release-time provenance. Only then run a predeclared out-of-sample strategy test, keeping reused data and genuinely new evidence separate.

No strategy is declared safe, consistently profitable or ready for live capital by this checkpoint.
