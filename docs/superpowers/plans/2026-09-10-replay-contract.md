# Research Replay Contract Implementation Plan

> **For agentic workers:** Execute inline with executing-plans and test-driven-development. Request an independent read-only code review before completion. Existing separate checkout and local research branch are approved; preserve them.

**Goal:** Implement the approved offline observation/replay contract and selected feature/boundary diagnostics without altering trading behavior.

**Architecture:** A strict observation clock feeds a freshly constructed processor in chronological order. Resume replays identical prehistory, not partially reconstructed detector state. A selected-feature processor preserves a raw reference and emits a named correction candidate; a CLI profiles archived inputs without falsely certifying their unknown availability/version history.

**Tech Stack:** Existing Python, pandas, numpy, unittest; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-10-research-replay-contract-design.md` (user approved).

## Global constraints

- No production/config/data edits, deployment, orders, optimizer or public push.
- All 17 archetypes remain represented. Empty evidence cannot certify parity.
- This certificate covers the clock and selected feature/threshold interfaces, not the full live feature computer, detector/book/fill/exit path.
- Unknown availability/source version stays uncertified. Never infer it from a bar label.
- Completed versus developing context is explicit. Inputs must be unique and ordered; gaps are errors, not synthesized candles.
- Keep reference/correction separate; source hashes anchor the reference. Corrections: finite FVG presence and OI-aware liquidity/fusion recomputation. Unimplemented dependent context is explicitly invalidated, never labeled refreshed.

## Task 1 — Observation clock and restart contract

Files: `scripts/research/replay_clock.py`, `tests/research/test_replay_clock.py`.

Interfaces: `Observation` (id, feature, value, instrument, source, units, version, event_time, available_at, status, received_at, valid_until); `replay(bars, observations, factory, instrument, timeframe, emit_from=None, checkpoint=None, expected_versions=None)`; `context_at(bars, decision_time, base_timeframe, target_timeframe)`.

- [x] Write tests first for delayed release, unknown release, duplicates/out-of-order, formula mismatch, stale/invalid values, missing bars, empty input, and closed/developing context at exact boundaries.
- [x] Test prefix/restart invariance using a stateful processor: `update` accumulates every candle's close and observation funding readings; literal sum verifies warmup includes each candle exactly once. Check multiple resume cuts and changed-prehistory rejection.
- [x] Run `python3 -m unittest discover -s tests/research -p test_replay_clock.py -v`; fail before implementation.
- [x] Implement UTC validation, availability-first stable observation order, exact instrument/version contracts, strict candle envelope/grid/continuity, and prefix-bound checkpoints. Fresh `factory()` on every call rebuilds processor state by replaying every prior candle. Emit only post-checkpoint rows.
- [x] Implement context aggregation on base candles whose close is at/before decision time. Output complete and developing buckets separately with constituent completeness.
- [x] Run tests and commit independently.

## Task 2 — Source-faithful reference and correction candidate

Files: `scripts/research/replay_features.py`, `tests/research/test_replay_features.py`.

Interfaces: `compare_features(features)` and `decision_probe(features, cfg, case, root)`; processor `SelectedFeatureProcessor.update(candle, observations)` for the clock.

- [x] Tests first: reference retains NaN FVG pass; correction rejects it; OI change .02 increases liquidity by .04 on otherwise equal inputs; fusion descendants refresh; missing dependencies invalidate candidate descendants; inputs remain unchanged.
- [x] Tests first: gate audit records combined actual evaluator penalty/mode, reference/candidate pre/post gate score, actual live/backtest acceptance, and narrow scope. A known global-bypass case still disagrees.
- [x] Run tests to observe expected missing-implementation failure.
- [x] Extract trusted pure helpers `_liquidity_score_from` and `_fusion_scores` by exact AST from local source; execute actual `DERIVED_FEATURES`/gate evaluator and existing boundary probe. Do not construct network clients or live runners.
- [x] Correction recomputes declared liquidity→fusion descendants; derivative-dependent regime context is flagged invalidated. Funding reconstruction remains unavailable absent its full observation contract.
- [x] Run all research tests and commit.

## Task 3 — Version-aware historical CLI and certificate

Files: `scripts/research/replay_contract_report.py`, `tests/research/test_replay_contract_report.py`.

Interface: `build_report(store, live_records, configs, hashes)` and CLI `--store --live-jsonl --config --out`.

- [x] Tests first: excludes all ambiguous duplicate rows; preserves 17/no-data archetypes; missing version/availability blocks certification; genuine source metadata mismatch reported; JSON finite-safe; empty evidence fails; returns decision-boundary diagnostic separately from paired-input drift.
- [x] Run failing tests. Implement report using existing gate/config loaders, selected-feature probes and boundary matrix. Record hashes, coverage and all blockers. Provide exit 2 with `--require-certification` for incomplete evidence.
- [x] Run actual 526-row paired archive exercise; emit candidate predicate/feature changes, not hypothetical P&L or actual entry counts.
- [x] Run all research tests, CLI expected-failure checks, independent source review, and production-diff checks. Fix review findings with failing regressions. Commit report/docs locally.

## Literal acceptance examples

```python
# At 01:00, only the hourly candle opened at 00:00 is completed.
assert context_at(two_hour_bars, '2026-01-01T01:00:00Z', '1h', '4h')['developing'][0]['volume'] == 1
# NaN is retained only in reference behavior, not positive correction evidence.
assert compare_features({'tf1h_fvg_present': 0, 'tf4h_fvg_present': 0, 'fvg_present': float('nan')})['reference']['any_fvg'] is True
assert compare_features({'tf1h_fvg_present': 0, 'tf4h_fvg_present': 0, 'fvg_present': float('nan')})['candidate']['any_fvg'] is False
```

Full-LFC reconstruction cannot be certified from the recovered archives alone: they lack availability/version/state provenance. The implemented interface and diagnostics must make this limitation explicit rather than fabricate a complete golden master.

## Execution record

Initial scoped interfaces implemented and reviewed; 63 research tests pass. The historical CLI processed 526 unambiguous paired rows and correctly exits 2 when certification is required. Full-LFC and strategy/book certification is not implemented or claimed. Review regressions, exact outcomes and reproduction commands are in `docs/knowledge/replay_contract_build_2026_09_10.md`. Production/configs/data remain unchanged; all work stays on the local research branch.
