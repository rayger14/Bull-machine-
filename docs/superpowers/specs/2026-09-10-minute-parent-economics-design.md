# Minute parent-context economics: independent conditional sleeves

## Intent and authority

Ray approved proceeding with setup-specific context experiments and delegated routine research review to quant agents. Independent quant design recommends this next executable unit because its existing minute simulator has a frozen economic contract. This is not a new strategy graduation. Hourly LC remains equally important but cannot inherit minute management without becoming another hypothesis.

No production `engine/`, `bin/`, or `configs/` changes, live settings/orders/deployments, public pushes, optimizer, or dependency installation. Keep the existing local research branch; preserve unrelated graph outputs. New code is a pure research comparison wrapper around the unchanged `minute_sweep_validation.simulate_events`.

## Frozen experiment

Use the 76 already frozen minute events whose reclaim-bar opens lie in `[2026-06-10T00:00:00Z, 2026-06-20T00:00:00Z)`. Child history and 60-minute selector spacing stay unchanged. Parent history starts June 1 with the separately frozen same-stream TA-Lib ATR14 reference. Compare baseline and all four existing parent policies: `4H:3`, `4H:5`, `1D:3`, `1D:5`. No variant is selected or modified from counts or outcomes.

All five sleeves start flat with no lockout at June 10. This is a declared bounded experiment, not a reproduction of inherited multi-year book state. Apply permission BEFORE each sleeve's independently replayed 240-minute lockout. Rejection/unknown permission does not arm lockout. Keep the upstream 60-minute selector fixed; this experiment cannot recover candidates it removed.

Economics are unchanged: long, first exact minute open after reclaim close, zero additional delay, $50,000 fixed notional, stop `sweep_low * 0.9985`, deadline open 240 minutes after actual entry, fixed lockout until that deadline even after an early stop. Reject stop >= entry. Include entry-bar stops; gap-stop fill `min(stop, bar_open)`. No profit target, scale-out, trailing or parent exit. Charge $30 per side (12bps round trip on fixed notional). Initial stop risk is `(entry-stop)*quantity`. Starting equity is unspecified: these are unfunded reference sleeves, not account returns. Funding, impact, additional spread/slippage and empirical receipt latency remain unmodeled.

Use same-source outcome bars through June 20 04:00 inclusive, admitting no new event outside the window. Missing tails retain open/censored or unfilled states; never manufacture closure. Input bars may include prehistory for original event indices, but only supplied in-window events may enter.

## Pure comparison interface

`compare_minute_parent_arms(bars, events, permissions, *, window_start, window_end)`:

- `bars`: the existing simulator's contiguous aware minute OHLC frame; read-only.
- `events`: list of raw selector dictionaries with local indices into `bars`, including `pivot_idx`, `confirmed_idx`, `sweep_idx`, `reclaim_idx`, `level`, `sweep_low`, `touches`. Preserve exact content and order; unique increasing reclaims and valid source indices. Every reclaim must lie in the declared half-open window. Event ID is the simulator's UTC `reclaim:<ISO>|pivot:<ISO>` identity.
- `permissions`: dictionary with EXACT four variant keys above. Each value maps EVERY event ID exactly once to strict `True`, strict `False`, or `None` (unknown). Reject missing/extra IDs or variants, numeric/string truthiness, duplicates in derived IDs, invalid clocks/indices and malformed containers with ValueError. No partial or guessed joins.
- The adapter consumes precomputed decisions, NOT authenticated parent evidence. The historical caller must separately rehash frozen sources, verify annotation policy/config/decision clocks/event identity, and preserve original annotation hashes in the artifact. This wrapper does not reconstruct parents or certify their truth.
- Validate/copy inputs before use. Empty event lists with four empty permission maps are valid. Keep fixed constants; no tuning arguments or generalized strategy engine.

Output is deterministic JSON-safe and `certified=False`. Include fixed parameters/limitations, normalized window, event count, and all five arms. Each arm contains the unmodified simulator return under `simulation`; a complete joined `event_ledger` in original event order (simulator status for permitted events, `permission_rejected` or `permission_unknown` otherwise); and a summary with the existing simulator summary plus completed fees, gross PnL (net + fees), mean/median net PnL divided by initial stop risk. Do not call ratios fully net R beyond the specified fee model. Unrepresentable/nonfinite output fails ValueError rather than silently clipping.

Each arm also reports comparison against baseline: shared entered IDs, baseline-only entered IDs, arm-only entered IDs, and full status-transition counts over all input IDs. Entered means `completed` or `open_censored`; `unfilled` is not entered. This distinguishes direct filtering from later opportunity displacement. Retain full trades/open records so duration/exposure analysis can respect ambiguous intrabar stop timestamps; do not invent an exact exposure statistic from bar labels.

## Source freeze and historical run checks

- Minute source: SHA256 `5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035`, physical path recorded in `results/research_validation_2026_09_10/minute_next_open/minute_replay.json`.
- Frozen minute parent reference: `results/research_validation_2026_09_10/h3_parent_permission/june_minute_frozen_reference.json`, SHA256 `9f9587253250b56d547d2a5af17a2d02bc1a8035e86d5f962eae0b7e0f01ff20`.
- Frozen annotations: `results/research_validation_2026_09_10/h3_parent_permission/june_fixed_event_annotations.json`, SHA256 `77ff36592b862a43033a15e48b8ca43159b4a455136f209af62a2db661e10d21`.
- Recover original raw events from baseline artifact SHA256 `6ed3caee375d8a023dd29173d92e3863b6e355dbd5591a7f8a9899fbbe74ea8a`. Join all 76 by simulator identity; rebase all index fields by the same input-slice offset, preserving timestamp/value identity. Validate child level, sweep low, reclaim close and sweep/decision clocks against the frozen child events and actual bars. Never regenerate/rerank candidates to increase coverage.
- Independently compare baseline against a direct unchanged-simulator call on the same bounded input/flat state. For each permission arm independently call that simulator on the corresponding filtered candidate list, NOT filtered trades, and match whole results.
- Verify all 76 IDs appear once in every ledger, all four annotation populations match the prior 16/19/8/0 permissions, no inputs changed, copied-input rerun equality and strict JSON. Preserve full source/code hashes and all results privately under `results/research_validation_2026_09_10/minute_parent_economics/`.

## Verification and interpretation

TDD hand-derived witnesses: all-permit exact baseline identity; rejection before lockout exposes a later otherwise-busy candidate; unknown never enters; early-stop lockout unchanged; permission counts not trade counts; correct $60 fee and risk ratio; same-entry-bar/gap stops inherited unchanged; tail censorship/unfilled and no out-of-window admission; missing/extra IDs and malformed permission values fail closed; nonmutation/repeatability; strict JSON. Existing simulator tests remain unchanged.

Only after implementation review run the frozen economic comparison. Report all arms, fees, net/gross dollars, completed/open/unfilled/busy counts, average stop risk, risk ratios, diagnostic marked-dollar drawdown and entry-set displacement. No inferential intervals, winner selection, cross-timeframe profit ranking, strategy-readiness or capital recommendation: ten days/76 preselected events are insufficient and have already been examined. Test passes establish software contracts, not predictive edge.

Hourly next-step diagnosis is a separate read-only deliverable: identify the exact outer score stage and why changing native bypass also changes portfolio limits, spacing and allocation. No hourly economic results are implied by this minute experiment.
