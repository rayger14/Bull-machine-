# Actual feature and signal replay — September 10, 2026

## Scope and delegated decisions

Ray authorized quant agents to make routine research proceed decisions while he sleeps. Two independent read-only quant reviewers approved the bounded feature adapter, its corrections, the signal-generation adapter, and controlled historical exercises. This did not authorize production changes, orders, deployment, public pushes, optimizer runs or declaring a strategy profitable.

The implemented reference executes current local source, not an assertion about the software or retained state running on the historical server. Commit `e249f3c` contains the reviewed adapters and 24 new tests (15 feature and nine signal). The subsequent report hardening (six tests) and synthetic native virtual-book fixture (13 tests) also received independent quant-review GO. The full research suite passes 106 tests.

## Delivered adapters

- `scripts/research/live_feature_replay.py`: isolated actual `LiveFeatureComputer` globals, replay-time derivatives cache clock, recorded transport provider, real funding/L/S/taker transformations, separate macro snapshots, strict hourly or minute-aggregation cadence, observable state and source/dependency identity.
- `scripts/research/engine_signal_replay.py`: actual `IsolatedArchetypeEngine.get_signals`, with the runner's live structural mode, causal 500-feature buffer, all 17 archetypes, native regime service, and observers around the existing single detection/gate/fusion/structural calls. It retains native cooldown and deduplication behavior.
- `run_signal_replay` composes both retained engines on the original observation clock, including hidden warmup. Minute carry rows emit no new signal-engine event.
- `scripts/research/live_feature_replay_report.py`: reproducible CLI with optional signal composition, emitted-only counts for all 17 archetypes, explicit comparison cardinalities, prefix/restart contract checks, input/configuration/source hash checks and fail-closed certification status.
- `scripts/research/virtual_book_replay.py`: actual `V11ShadowRunner` constructor and `process_bar`, with controlled copied signals and in-memory log/maker endpoints. Native sizing, thresholds, bypass branches, entry/exit calculations and equity methods remain unchanged. This is a separate synthetic fixture, not yet a composed feature-to-book historical replay.

Network calls are denied around import/construction/execution. This is a Python socket guard, not an operating-system sandbox. The signal adapter denies allocation calls. No runner or exchange execution object is constructed by these two adapters.

## What the tests actually establish

Feature fixtures cover funding's ten-sample threshold, L/S's 24-sample threshold, nonzero derivatives funding versus candle funding (including two samples when both channels are supplied), retained NaN representation, missing/defaulted inputs, hourly/minute aggregation, completed/developing/incomplete buckets, observation lineage, future append and restart, network denial, snapshot limitations and changed dependency rejection.

Signal fixtures use real gates and fusion. In the positive wick fixture, bar 1 selects `liquidity_sweep` while a deduplicated `wick_trap` still arms its cooldown. Bar 18 emits no signal; bar 19 selects `wick_trap`. This is a synthetic characterization, not a market-performance result. A structural detector exception remains permissive in the source but becomes an explicit diagnostic blocker.

Independent review additionally compared 20 fixture hours with an unwrapped native engine: outputs and checked retained state matched. A 181-minute composition exercise with delayed observations, hidden warmup and six restart cuts preserved rows and final state. A separate 502-hour synthetic replay retained 500 feature rows and 48 regime-history entries, performed 8,534 structural checks with zero structural errors, and matched a second fresh replay's state. No unsupported state types were observed in that fixture.

## Initial real-data observations

A cold-start V23 OHLCV-only window covering June 17–19 (72 hourly bars) emitted 24 June 19 rows. The final vector contained 224 fields. All optional LFC detector imports were available; no unsupported state type appeared. This short window does not establish sufficient warmup for every detector.

Recovered minute data for June 19 00:00–01:59 UTC (120 bars) produced two completed hourly LFC updates. No hourly feature existed at minute 59. Comparison against independently pandas-aggregated hourly candles found four exact floating-point differences: `volume` and `volume_ma_20` in each hour, with maximum absolute difference approximately `4.55e-13`. All other compared fields matched exactly. Do not call this bit-exact aggregation parity or silently select a tolerance after observing it; bind any future numerical tolerance explicitly. The final cold-start vector contained 161 fields.

Minute input SHA-256: `5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035`. Output: ignored `results/research_validation_2026_09_10/full_feature/minute_native_equivalence.json`.

The 1,002-hour V23 buffer-boundary experiment completed: May 9 00:00 through June 19 18:00 UTC (end exclusive), emitting 18 June 19 decisions and 228 final feature fields. At the 1,000-bar cut, prefix rows, restarted rows and final retained state matched the full replay. `--require-certification` returned exit code 2 as intended. This was feature-only, not a profit backtest; starting equity and average risk per trade are not applicable.

Its ignored output is `results/research_validation_2026_09_10/full_feature/hourly_buffer_boundary.json`, SHA-256 `6b500d8b8ae58242b010db0d9e36136b704129e8a2de8b8bc96dc4a03e8ddd2d`. It executed the older report CLI pinned at commit `1523ce5ca98ffb547481e9b14623ea5c9e6d114d`; reported CLI SHA-256 `08c08fdfec33459ef1cd94d0fa3a213d5fa69454457389159a65f528b5137e40` matches that commit exactly. The process completed before report hardening was applied. Do not attribute the new CLI guards to this older run.

A subsequent 72-hour combined feature/signal exercise emitted the 24 June 19 decisions, with zero pre-dedup or selected signals across all 17. Across the entire supplied prehistory it made 1,224 structural checks: 271 passed, 953 rejected, zero errors. This short cold-start, missing-external-data scenario is not grounds for tuning or disabling an archetype.

The hardened CLI reproduced this combined exercise with cuts at 48 and 71 hours. All row/state/contract checks passed; the warmup-boundary prefix compared zero emitted rows, explicitly reported as such, while the second prefix compared 23. Saved output `results/research_validation_2026_09_10/full_feature/hourly_signal_72h.json` has SHA-256 `c33798c7e91463ac6f3bb5c0bd88f81539d9bf88ecb6498fb642011c018b77d4`.

The same CLI exercised 120 minute candles on June 19 00:00–02:00 UTC (end exclusive), with cuts at 59, 60 and 119 minutes. It advanced the signal engine exactly twice, evaluated each archetype twice, emitted zero native/selected candidates and preserved every tested prefix/restart row, final state and contract identity. Output `results/research_validation_2026_09_10/full_feature/minute_signal_120m.json` has SHA-256 `35e839ead37cb03d1afa2d09f9983cfed3a66c473faf004df10afcbe9a4638e9`. Both commands deliberately exited 2 under `--require-certification`. These are feature/signal diagnostics without positions: starting equity and average risk per trade are not applicable.

Related cache inventory was checked directly: `../data/cache/derivatives_hourly_full.parquet` has 50,617 rows through June 11 00:00 UTC, with eight already-transformed fields and no metadata attributes. It contains `ls_ratio_extreme`, not the raw `ls_ratio` needed to reproduce live sampling. The smaller derivatives cache has 19,201 rows through June 10. `macro_daily_history.parquet` has 2,274 daily rows through June 11 with naive date labels, raw/z-scored macro fields, and no metadata attributes. These are useful historical inputs, but neither actual release/receipt provenance nor the missing raw L/S history can be inferred from them. The raw OHLCV cache has 75,864 rows through August 30; it is distinct from the 74,436-row V23 feature store.

## Source-backed constraints for the all-seeing-eye design

1. LFC is hourly. Direct minute calls redefine row-count lookbacks, funding sampling and annualized volatility. Minute-native structure needs a separately versioned detector contract, not renamed hourly columns.
2. Native LFC higher-timeframe resampling includes developing final buckets and potentially incomplete first buckets. Reference features retain that behavior; the adapter's completed/developing/incomplete context is separate.
3. Deep daily history is off by default in the host and unsupported in this initial adapter. Before any later daily ingestion, filter by actual availability: the source trims to its last 320 rows before its later historical date filter, so future rows can otherwise discard required past history.
4. Wyckoff reconstructs a state machine over the rolling buffer and applies current higher-timeframe context to historical events. A historical event listed today is not necessarily its original detection-time record. Do not backdate entries to those reconstructed event timestamps.
5. The LFC constructs but does not call its probabilistic regime detector in the current CMI path. The signal engine does call its own regime service and changes its retained history/hysteresis. These are different state paths.

## Remaining certification blockers

The configured crisis model and confidence calibrator were not found in the three related checkout levels; an indexed exact-name search also found no crisis model. The engine retains the actual mock fallback and reports it. A model file that exists but fails to load is also flagged. This is not proof that the deployed server lacks its model.

The adapters do not reconstruct historical receipt times, actual live startup, macro-provider calculations, deep daily context, host funding injection/costs, daily downtrend/tape updates, the runner's adaptive entry filters, allocation, positions or fills. Every top-level and row-level strategy certificate remains false; the generic clock result is separately labeled.

The separate synthetic virtual-book fixture characterizes those native downstream runner methods without supplying historical detector signals. Its four-hour fixture uses $100,000 initial cash, 4 bps commission and 5 bps slippage; no market-performance or average-risk estimate is claimed. Tests establish exit-before-entry ordering, stop-first handling when stop and target both touch, real versus phantom rejection, and the source's fee accounting (trade PnL subtracts exit commission while entry costs reside in cash). Champion bypass naturally skips allocation/spacing; the fixture does not force unexecuted branches. Native paper-margin and Series-entry-metadata quirks are retained. Maker calculations, CSV persistence and host additions remain excluded. Sticky filesystem/network/subprocess guards are scoped Python instrumentation, not an OS sandbox.

## Next bounded research step

Compose feature/signal replay with the reviewed native-book reference without a second detector pass, first on controlled positive fixtures. Bind the availability/fill convention explicitly before historical execution: retaining the runner's hour-open labels does not establish that a close-known signal could have traded at that open. Then add properly timestamped external observations and model artifacts if recovered. No gate optimization or strategy promotion should precede these checks and a fresh out-of-sample protocol.

No archetype was disabled or tuned. No raw data, engine, runner or production config was edited. No profitability estimate was produced by this work.
