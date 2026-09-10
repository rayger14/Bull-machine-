# Causal parent ledger: guarded research reference

## Approval and scope

Ray approved proceeding toward shared multi-timeframe context and delegated routine research design decisions to quant reviewers. The quant context reviewer approved the frozen-source adapter approach on September 10. This specification implements that first subsystem only: parent annotations, not signals, gates, sizing, fills or deployment. H1/H2 remain separate audit policies.

Alternatives considered: (A) guarded recovered source with an append-only ledger; (B) a copied incremental implementation; (C) a redesigned range constructor. Choose A for attributable parity. B duplicates logic before a reference exists; C mixes a new strategy hypothesis with clock/identity work.

No production `engine/`, `bin/`, `configs/`, original data or sibling files are edited. No network, optimizer, public push or live order is permitted. All outputs remain uncertified strategy evidence.

## Frozen source and assumptions

Call the recovered `htf_pivots.py` and `structural_range.py` functions loaded by explicit paths and expected SHA-256 under the existing research side-effect guard. Never execute main blocks. Reference source hashes: HTF `0c37d254cba93f40638ffeb2422d53649ea326adecee0a0690ed417b2f492f0b`; range `22e3e1ab21d8db585d89569c3d1d4a3fe3c74ce17ca72c5080a0ca7cf0caa8ca`.

Support explicit 4H/1D anchor timeframe and N=3/5, with no default winner. These are project hypotheses. Preserve source width constants 1.5/0.75 ATR, one-hour body-close breaks with zero buffer, floor tightening and old-anchor reformation. Anchor timeframe does not change the range machine's hourly invalidation clock.

Use batch evaluation capped at 2,048 input hours, with prefix/restart comparison as the initial reference. Exceeding the cap rejects rather than silently truncating. The caller supplies ATR and its formula/source contract; do not calculate a substitute.

## Input contract

`build_parent_ledger(bars, *, instrument, data_stream_id, anchor_timeframe, pivot_n, atr_contract, source_paths, expected_hashes)` consumes an hourly DataFrame with OHLCV and `atr_14`, plus explicit ATR contract metadata. `pivot_n` is an integer in {3,5}, not a boolean. `source_paths`/`expected_hashes` map `htf` and `range`; stream and instrument identities are nonempty strings. The ATR contract identifies source, version/formula and availability policy.

Use existing `replay_clock.validate_bars(bars, '1h')`: aware UTC-normalizable, unique, sorted, exactly hourly-aligned contiguous OHLCV, valid finite envelope and nonnegative volume. Naive input, gaps, duplicates, offsets and invalid OHLCV reject. ATR NaN/zero may represent unavailable/warmup or flat-range input; preserve values and disclose unusable ATR, never fill backward. Reject nonnumeric/infinite/negative ATR. This first version assumes all hourly values available exactly at close and explicitly rejects any supplied availability column inconsistent with that assumption. It does not certify real receipts or revisions.

Only complete aligned 4/24-hour buckets enter pivot detection. Exclude a leading partial bucket and retain a trailing partial bucket as a developing diagnostic, not a pivot constituent. Missing interior hours reject. Recovered aggregation must match independently aggregated complete buckets on valid inputs.

Canonicalize external timestamps to UTC. For the source-only compatibility copy, make UTC timestamps naive, execute the unchanged source functions, then explicitly restore UTC. Check epoch identity. This bridge solves the recovered dtype mismatch without resampling in local time or accepting naive external timestamps.

## Ledger and identities

Return `certified=False`, contract/source/runtime manifest, input manifest, `pivots`, `versions`, `transitions`, and incomplete/developing bucket diagnostics. Caller-owned input and source are immutable.

Pivot records include stable ID, side, level, pivot occurrence/open/close, confirming close/availability, supporting evidence identity, anchor timeframe and N. Equal-priced distinct pivots have different IDs. Preserve source broadcast: confirmation must be no later than the source hourly row's OPEN.

Each formation starts a lineage and version. Each floor tightening creates a new immutable version in the same lineage. Version records include predecessor, creation reason, fixed bounds, adopted anchor IDs, formation hour and availability. A range with unchanged numerical bounds does not silently adopt a newer equal-priced pivot ID. Reformation may reuse old anchors and still starts a new lineage.

Each hourly transition records source hour, availability at close, pre-state/bounds/lineage/version, source break/sweep flags with their evaluated pre-version, post-state/bounds/lineage/version, latest observed pivot IDs, adopted anchor IDs and quality flags. Preserve source order: old-bound break, old-bound sweep, possible tightening. Broken geometry is diagnostic only; active parent becomes absent.

Do not retrospectively write an invalidation/retirement timestamp into an older pivot/version. Append a later transition. IDs use contract + stream + causal event content/predecessor, never the future full-run input hash. Full-run hashes belong only in the run manifest.

## Consumer contract

`parent_asof(ledger, decision_time, strict=False)` chooses the latest transition available at or before decision time and returns its active version, otherwise None. `strict=True` uses `<`.

Query times must be timezone-aware and UTC-normalizable. Carry is valid during intervening minutes, but at or after `last_processed_close + 1h` raise an explicit `out_of_coverage` error; do not extrapolate an active range indefinitely after a missing scheduled update. This bound also applies to bindings.

`bind_parent(ledger, *, child_event_id, child_timeframe, first_sweep_open)` returns a new immutable binding or an explicit absent-parent rejection. Use strict `<` for these candidate bindings. Store the chosen lineage/version, fixed geometry, parent availability and binding policy. Later tightening or a break of the current lineage must not rewrite an old binding or be described automatically as a break of an older frozen floor.

Hourly and minute consumers use the same ledger. Minute inputs first produce 60-complete-minute hourly OHLCV using the existing clock contract; no separate minute parent engine. Same-data aggregated hours must have identical IDs and state to direct hourly input. This does not establish equivalence between different historical venues.

## Verification and limits

Require source parity for all four timeframe/N combinations, prefix equality of historical records, fresh full-history restart equality, and immutable binding checks. Pin incomplete buckets, UTC bridge, right-side pivot confirmation, equal-level pivot identities, delayed range availability, floor tightening and reused-anchor reformation. Hash source/helper/runtime/ATR/parameter contracts. Changing any must change the contract and invalidate an expected-contract comparison.

Prefix comparisons filter pivots, versions and transitions by `available_at <= cutoff`, not pivot occurrence time: a later-confirmed old pivot is not earlier knowledge.

The first adapter does not restore a serialized partial checkpoint; a fresh full-history rebuild is the supported restart. It exposes enough identities for later replay integration without claiming that integration is done. No P&L test or selected parent rule is part of this subsystem.
