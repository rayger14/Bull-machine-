# Wyckoff evidence integrity repair design

## Purpose and authority

Repair the first correctness batch from the October 4 M1/M2 audit. The user
delegated open questions to a quant agent and requested continuous execution.
The quant reviewer approved the boundaries and selected completed-only Wyckoff
inputs, identity-preserving delayed confirmation, and direction-aware fallback.
This is a cross-module repair to existing behavior, not a new strategy or agent.

Work in the existing quant branch. No deployment, orders, configuration changes,
threshold/weight tuning, M2 activation, V2 promotion, installations, paid model
experiments, economic study, commit, push or PR. Preserve frozen artifacts and
unrelated dirty files. Changed source hashes make earlier receipts historical;
do not rewrite those receipts to bind new code.

## Delayed events

Keep raw detector tuple interfaces. Add optional `event_metadata` to
`WyckoffStateMachine.process_bar`; omitting it preserves deliberate pretagged
callers. The batch adapter always supplies a mapping, including when empty.
Malformed or missing metadata in that route rejects a proposed delayed event.

`DelayedEventEvidence` carries event type, candidate/confirmation indices,
candidate extreme, prior swept boundary, candidate parent identity/status,
candidate/confirmation timestamps, and availability at confirmation close.
The adapter captures the parent active immediately before candidate processing.
Parent generations change on reset/new SC/new BC. Confirmation must still refer
to the same established, correctly directed parent; no reattachment to a new
generation or demotion of a lost parent into no-context fallback. A no-parent
candidate can remain unstructured only if no parent has arisen in the meantime.
An incomplete parent is not an established one. UTAD shares UT candidate identity.

Preserve existing offsets, geometry thresholds and invalidation ordering.
Spring B retains its existing zero-delay convention when recovery_bars <= 1.
Only accepted spring events may cause a spring-state transition; independent
valid events/invalidation still take effect. Expose row-level provenance without
backfilling earlier rows. Explicit timeframe controls close timestamps in the
live route. Compatibility inputs can infer interval from prior timestamps only;
missing/irregular temporal provenance cannot become delayed confirmation.

## Direction and availability

Add per-timeframe status/reason/source/last input close/available-at fields using
prefixes `wyckoff_`, `tf4h_wyckoff_`, `tf1d_wyckoff_`. Status `available` means
detector inputs were admissible, not that a setup was confirmed. Zero scores
with available status mean no supporting evidence; unavailable is distinct.

Keep same-direction weights and event-confidence fallback. Any directional
score or directional-confidence key, including opposite-only/zero/NaN, disables
generic and binary compatibility fallback. Explicit proxy/unavailable/error
status suppresses that timeframe's stale values. Other available timeframes
remain usable. Legacy generic/binary fallback is allowed only with no directional
schema and no explicit invalid/proxy status; inputs must be finite/nonnegative.
EMA fallback uses its own proxy field and unavailable Wyckoff status, not a
positive Wyckoff phase score. No weight redistribution beyond existing behavior.

## Candle integrity and idempotence

Add a pure Wyckoff-specific input preparer with explicit UTC `as_of`. Source
hourly rows are start-stamped completed observations. A 4H/day requires exactly
4/24 unique aligned hours, finite valid OHLCV, and closure by as_of. Missing or
duplicate constituents cannot produce confirmation. Use the latest contiguous
valid segment; never compress a gap into sequential bars. An invalid latest
expected closed bin makes that timeframe unavailable. A still-open current bin
does not invalidate the preceding complete bin. Existing non-Wyckoff resampling
remains unchanged.

Native daily inputs have separate provenance: closed, unique, UTC-aligned native
daily observations, not inspected 24-hour constituents. The Coinbase adapter
already documents/excludes the current day. Validate closure/continuity again
at the research decision cutoff; do not splice future/current-day observations.
Do not silently fill a known invalid hourly-derived day from another source.

Identical warmup/poll candles do not append a second buffer row. The first poll
may still compute its first decision. An already processed duplicate returns
the saved feature vector without advancing stateful histories; a conflicting or
older update is explicitly rejected. Leave runner decision deduplication intact.

## Verification and limits

Use failing deterministic regressions before each implementation. Cover raw
delayed recognition plus sequencer adapter, same-level replacement, reset,
expiry, no-parent and partial-parent cases, candidate/confirmation clock and
prefix invariance; both trade directions and per-timeframe availability;
partial/duplicate/missing/invalid candles, native daily cutoff and duplicate
update histories. Exercise actual offline LiveFeatureComputer with transport
boundaries disabled; never construct a live exchange runner.

Baseline expanded selection: 53 pass/8 fail. Five old raw-event fixtures and
three archetype expectations already fail; preserve/report them instead of
changing strategy to satisfy them. Run new regressions, existing causal/M2/V2
tests, the offline actual-source adapter suite, and the full repository command.
Known suite collection blockers must be named, not presented as passing.

One independent software review follows implementation. Completion means these
contracts are implemented and locally tested, with remaining audit findings
explicit. It does not certify chart recognition, full Wyckoff semantics or edge.
