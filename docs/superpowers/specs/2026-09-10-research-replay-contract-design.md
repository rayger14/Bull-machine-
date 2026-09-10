# Research replay contract — design for review

Status: proposed implementation; not a production behavior change. Evidence: `docs/knowledge/feature_parity_root_causes_2026_09_10.md`.

## Objective and scope

Build the smallest offline contract that can distinguish wrong inputs from wrong decisions before any gate tuning or multi-timeframe strategy synthesis. Keep all 17 archetypes visible, preserve production/configs/data, and work on the existing local research branch. No public push, network access from the harness, optimizer, strategy graduation or orders.

This is a new research interface, so implementation should follow review of this design. The first delivery covers point-in-time observations, deterministic state replay, selected feature dependencies, and the existing threshold boundary. Full portfolio, fills and exits are subsequent acceptance layers, not implied by this delivery.

## Alternatives considered

1. Patch individual store columns. Fast but insufficient for stateful funding, stale downstream features, NaN behavior, and restart defects.
2. Rebuild all history immediately. Expensive and risks recreating the same defects without a verified clock and state contract.
3. **Recommended: small deterministic replay contract first.** Certify causality/state behavior on fixtures and short historical windows, then make any large rebuild use that contract.

## Two separate tracks

**Reference:** execute frozen production source behavior exactly, including existing defaults and known NaN behavior. Source/config hashes define identity. A mismatch is reported, not silently reconciled. Archived old-version snapshots and current-source replays are distinct baselines.

**Correction candidate:** explicit policy/version plus a list of permitted changes. Initial candidates are finite FVG presence and recomputation of liquidity after historical OI is available. Funding reconstruction stays unavailable unless observation timing, sampling cadence, venue, lookback and restart state are specified. Candidate output never replaces the reference or original parquet.

Passing reference parity does not prove the behavior is correct or profitable. Passing correction tests does not authorize production deployment.

## Observation and clock contract

Every input observation carries:

- Instrument and venue/source; units; feature schema/formula version.
- `event_time`, `available_at`, validity status, and value. `received_at` is optional when genuinely captured, otherwise unknown.
- For candles: open time, close time, timeframe and complete/developing state.
- For derived values: source observation IDs and computation version.

The replay may consume an observation only if its recorded `available_at` is at or before the decision time. Unknown availability produces an explicit uncertified input; it is not zero and does not certify an as-of replay. Configured legacy defaults remain available only in the labeled reference track.

Stable tie order for equal timestamps must be explicit: incorporate available observations, close/update context, evaluate trigger and risk, then schedule an order. A next-open research fill is a declared reference assumption, never a claim of a guaranteed executable price.

Hourly and minute triggers use the same availability rule. Completed higher-timeframe bars become available at close; developing context is built only from already-closed lower-timeframe bars and marked partial. Never forward-fill a completed higher-timeframe value into earlier constituent bars.

## Deterministic replay and dependency contract

Warmup is strictly before the first emitted bar. Timestamps must be unique and increasing. Missing bars are explicit gaps, not synthesized observations.

State includes retained OHLCV, funding/long-short histories, detector memory and replay cursor. Resume must either restore the complete state or re-run the identical original prehistory in timestamp order, discarding already-emitted outputs. Do not seed checkpoint-near candles and then replay earlier history.

A historical overlay invalidates every derived descendant, not just its immediate score. The initial dependency inventory must include funding→funding history/Z, OI→liquidity, liquidity→fusion descendants, and derivatives→context where enabled. Distinguish unavailable, genuine zero, invalid and stale observations. Preserve the original raw representation for reproduction.

## Decision contract

Reuse actual-source execution from `decision_boundary_parity.py`; freeze current local live behavior as the boundary reference. Record gate predicate outcome, missing/defaulted inputs, gate mode/penalty, pre/post-adjustment score, threshold, global/per-archetype bypass and enforcement status, acceptance and rejection stage.

Do not change global bypass or disable any archetype. Compare backtest behavior against the reference and report differences. The known 48/192 disagreement is an initial witness, not a requirement to edit live policy. Per-archetype structural checks, cooldown, dedup, sizing and exits must remain explicitly outside the initial boundary certificate.

## Initial acceptance tests

1. Appending future observations cannot change any earlier emitted feature or decision.
2. At a 1H/4H/day boundary, completed context is unavailable early; developing context uses only elapsed constituents.
3. Warmup does not duplicate the first emitted candle.
4. Uninterrupted and checkpoint-resumed fixture replays have identical emitted outputs and final state; test multiple cut points.
5. Unknown release times and out-of-order/duplicate input fail certification, with actionable errors.
6. Reference reproduces NaN `any_fvg=True`; the named correction candidate does not treat NaN as positive evidence.
7. Historical OI is injected before dependent liquidity calculation; unchanged price inputs plus changed OI recompute the declared descendants.
8. Identical raw derivatives under differing live/backfill definitions produce an explicit formula-version mismatch, never a silently averaged value.
9. All 17 archetypes remain represented, including no-signal/no-data states. An empty test window cannot produce a passing certificate.
10. Existing 35 research regressions remain green. Source/config/input hashes, limitations and every diagnostic failure appear in the output manifest.

## Initial real-data exercise and stopping rules

Use the 526 unambiguous June/July timestamps for a version-aware paired-input report, not as a current-live golden master. Use controlled observations to isolate formula/state behavior; test a short contiguous historical window only after required prehistory and availability assumptions are established. No P&L-based threshold selection in this stage.

Stop certification when a required source version, release time, observation history or configuration cannot be reconstructed. Continue inventory/diagnostics if useful, but do not label a provisional replay as proven equivalent.

After this layer passes, propose the bounded full signal/book parity layer. Only after measurement is sound should the hourly-context/minute-trigger hypothesis enter a preregistered strategy test, with reused data labeled as reused and future shadow evidence reserved.
