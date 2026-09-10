# Actual feature and signal replay — September 10, 2026

## Scope and delegated decisions

Ray authorized quant agents to make routine research proceed decisions while he sleeps. Two independent read-only quant reviewers approved the bounded feature adapter, its corrections, the signal-generation adapter, and controlled historical exercises. This did not authorize production changes, orders, deployment, public pushes, optimizer runs or declaring a strategy profitable.

The implemented reference executes current local source, not an assertion about the software or retained state running on the historical server. Commit `e249f3c` contains the reviewed adapters and 24 new tests (15 feature and nine signal). Research report hardening and the longer historical exercise are still in progress at this document's initial creation.

## Delivered adapters

- `scripts/research/live_feature_replay.py`: isolated actual `LiveFeatureComputer` globals, replay-time derivatives cache clock, recorded transport provider, real funding/L/S/taker transformations, separate macro snapshots, strict hourly or minute-aggregation cadence, observable state and source/dependency identity.
- `scripts/research/engine_signal_replay.py`: actual `IsolatedArchetypeEngine.get_signals`, with the runner's live structural mode, causal 500-feature buffer, all 17 archetypes, native regime service, and observers around the existing single detection/gate/fusion/structural calls. It retains native cooldown and deduplication behavior.
- `run_signal_replay` composes both retained engines on the original observation clock, including hidden warmup. Minute carry rows emit no new signal-engine event.

Network calls are denied around import/construction/execution. This is a Python socket guard, not an operating-system sandbox. The signal adapter denies allocation calls. No runner or exchange execution object is constructed by these two adapters.

## What the tests actually establish

Feature fixtures cover funding's ten-sample threshold, L/S's 24-sample threshold, nonzero derivatives funding versus candle funding (including two samples when both channels are supplied), retained NaN representation, missing/defaulted inputs, hourly/minute aggregation, completed/developing/incomplete buckets, observation lineage, future append and restart, network denial, snapshot limitations and changed dependency rejection.

Signal fixtures use real gates and fusion. In the positive wick fixture, bar 1 selects `liquidity_sweep` while a deduplicated `wick_trap` still arms its cooldown. Bar 18 emits no signal; bar 19 selects `wick_trap`. This is a synthetic characterization, not a market-performance result. A structural detector exception remains permissive in the source but becomes an explicit diagnostic blocker.

Independent review additionally compared 20 fixture hours with an unwrapped native engine: outputs and checked retained state matched. A 181-minute composition exercise with delayed observations, hidden warmup and six restart cuts preserved rows and final state. A separate 502-hour synthetic replay retained 500 feature rows and 48 regime-history entries, performed 8,534 structural checks with zero structural errors, and matched a second fresh replay's state. No unsupported state types were observed in that fixture.

## Initial real-data observations

A cold-start V23 OHLCV-only window covering June 17–19 (72 hourly bars) emitted 24 June 19 rows. The final vector contained 224 fields. All optional LFC detector imports were available; no unsupported state type appeared. This short window does not establish sufficient warmup for every detector.

Recovered minute data for June 19 00:00–01:59 UTC (120 bars) produced two completed hourly LFC updates. No hourly feature existed at minute 59. Comparison against independently pandas-aggregated hourly candles found four exact floating-point differences: `volume` and `volume_ma_20` in each hour, with maximum absolute difference approximately `4.55e-13`. All other compared fields matched exactly. Do not call this bit-exact aggregation parity or silently select a tolerance after observing it; bind any future numerical tolerance explicitly. The final cold-start vector contained 161 fields.

Minute input SHA-256: `5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035`. Output: ignored `results/research_validation_2026_09_10/full_feature/minute_native_equivalence.json`.

The 1,002-hour V23 buffer-boundary experiment is running; its result is not yet claimed here.

## Source-backed constraints for the all-seeing-eye design

1. LFC is hourly. Direct minute calls redefine row-count lookbacks, funding sampling and annualized volatility. Minute-native structure needs a separately versioned detector contract, not renamed hourly columns.
2. Native LFC higher-timeframe resampling includes developing final buckets and potentially incomplete first buckets. Reference features retain that behavior; the adapter's completed/developing/incomplete context is separate.
3. Deep daily history is off by default in the host and unsupported in this initial adapter. Before any later daily ingestion, filter by actual availability: the source trims to its last 320 rows before its later historical date filter, so future rows can otherwise discard required past history.
4. Wyckoff reconstructs a state machine over the rolling buffer and applies current higher-timeframe context to historical events. A historical event listed today is not necessarily its original detection-time record. Do not backdate entries to those reconstructed event timestamps.
5. The LFC constructs but does not call its probabilistic regime detector in the current CMI path. The signal engine does call its own regime service and changes its retained history/hysteresis. These are different state paths.

## Remaining certification blockers

The configured crisis model and confidence calibrator were not found in the three related checkout levels; an indexed exact-name search also found no crisis model. The engine retains the actual mock fallback and reports it. A model file that exists but fails to load is also flagged. This is not proof that the deployed server lacks its model.

The adapters do not reconstruct historical receipt times, actual live startup, macro-provider calculations, deep daily context, host funding injection/costs, daily downtrend/tape updates, the runner's adaptive entry filters, allocation, positions or fills. Every top-level and row-level strategy certificate remains false; the generic clock result is separately labeled.

No archetype was disabled or tuned. No raw data, engine, runner or production config was edited. No profitability estimate was produced by this work.
