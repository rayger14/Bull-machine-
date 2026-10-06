# Wyckoff Evidence Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan inline, task by task. User delegated design questions to a quant reviewer; continue without approval pauses.

**Goal:** Repair event identity, directional availability and candle integrity in the selected local Wyckoff path.

**Architecture:** Preserve current detectors and numerical policies. Extend the existing sequencer boundary with delayed-event provenance, add one pure candle-preparation module, and wire explicit availability into the current feature/scoring path. No parallel replacement engine.

**Tech Stack:** Existing Python, pandas, numpy, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-04-wyckoff-evidence-integrity-design.md`

## Global Constraints

- Existing quant branch; preserve unrelated dirty files and frozen artifacts.
- No deployment, orders, configuration changes, threshold/weight tuning, M2 activation, V2 promotion, installations, paid model experiments, economic study, commit, push or PR.
- Keep invalidation/z-score semantics, numeric thresholds, offsets and other feature families' resampling unchanged.
- Eight expanded-baseline failures remain separately reported; do not loosen detectors to make them pass.
- One final independent software review; completion requires controller verification.

## Review Focus

- Same-level parent replacements must not inherit older candidates.
- Prefix calculations must not depend on future-inferred cadence or timestamps.
- An unavailable timeframe cannot leak through generic, directional or binary fallbacks.
- Missing closed bins must not turn older evidence into apparently current confirmation.
- Duplicate first polling after warmup must preserve one decision without double history ingestion.

### Task 1: Preserve delayed event identity

**Files:** Modify `engine/wyckoff/events.py`; create `tests/test_wyckoff_evidence_integrity.py`.

**Interfaces:** `DelayedEventEvidence` immutable record; optional `event_metadata` mapping on `process_bar`. `_apply_state_machine_validation` supplies it and emits provenance columns for delayed events. Existing detector tuple returns unchanged.

- [ ] Write failure witnesses for rejected-spring phase mutation and actual delayed A/B/UT geometry. Include wrong/missing metadata, no-parent fallback, generation replacement/reset/timeout, same-bar independent events, and full-versus-prefix provenance.

```python
sm = established_accumulation()
valid, _ = sm.process_bar(9, candle(low=100.5), {'spring_a': True})
assert valid['spring_a'] is False
assert sm.get_phase_dir() == 'A_accum'
```

- [ ] Run `python3 -m pytest -o addopts='' -q tests/test_wyckoff_evidence_integrity.py --tb=short`. Expected: behavioral failures against old code, not fixture/import errors.
- [ ] Add immutable candidate metadata and monotonic parent identity; snapshot before each candidate, validate before mutation, compare original extreme to the surviving parent, and stamp confirmation availability only on that row.

```python
# Compatibility only when metadata was deliberately omitted.
if event_metadata is not None:
    evidence = event_metadata.get(event_type)
    # Missing/invalid evidence or a different parent generation rejects.
    # Candidate geometry replaces current-row geometry only after validation.
if validated.get('spring_a') or validated.get('spring_b'):
    self.state = WyckoffState.ACCUM_SPRING
```

- [ ] Run the new file plus `tests/test_wyckoff_m2_sequence.py`, `tests/test_wyckoff_causality.py`, `tests/test_wyckoff_v2_climax.py`. Expected: all pass; record exact output in ledger. No commit.

### Task 2: Enforce directional evidence availability

**Files:** Modify `engine/archetypes/archetype_instance.py`; create `tests/test_wyckoff_directional_evidence.py`.

**Interfaces:** Existing `_get_wyckoff_score(features)` returns float. It consumes optional per-TF `wyckoff_evidence_status` and `wyckoff_evidence_source` fields, with `tf4h_`/`tf1d_` prefixes. Legacy no-status directional payloads remain supported.

- [ ] Write long/short table cases for opposite-side-only, explicit zero, NaN/invalid values, proxy/unavailable values and binary fallback. Pin unchanged valid same-direction scores and genuinely legacy-only compatibility.

```python
features = {'wyckoff_bullish_score': 0.0, 'wyckoff_bearish_score': .8,
            'wyckoff_event_confidence': .8}
assert long_instance._get_wyckoff_score(features) == 0.0
```

- [ ] Run `python3 -m pytest -o addopts='' -q tests/test_wyckoff_directional_evidence.py --tb=short`. Expected: wrong-direction/status cases fail against old code.
- [ ] Zero only disallowed sources before existing weighted calculation; check key presence before generic and binary fallbacks; sanitize finite legacy values without tuning weights.

```python
directional_schema = any(key in features for key in directional_keys)
if directional_schema:
    return 0.0  # after valid same-side score/confidence opportunities
```

- [ ] Rerun this file and task 1 selection. Expected: all new/previously green selected tests pass. No commit.

### Task 3: Prepare completed Wyckoff candles and make updates idempotent

**Files:** Create `engine/wyckoff/candle_integrity.py`, `tests/test_wyckoff_candle_integrity.py`; modify `bin/live/live_feature_computer.py`.

**Interfaces:** `prepare_wyckoff_bars(hourly, timeframe, as_of, min_bars=1, native_daily=None)` returns `WyckoffInput` with frame, status, reason, source, last input close and available_at. Live features expose the task 2 status/source fields; event calls receive explicit timeframe in cfg. Existing generic resampler and runner decision guard unchanged.

- [ ] Test literal complete OHLCV aggregates and rejection of missing/duplicate/unclosed/nonfinite/invalid/alignment cases. Test latest contiguous suffix, unavailable latest closed bin, native daily provenance/cutoff, and future append invariance.

```python
result = prepare_wyckoff_bars(hours_00_02_03, '4h', '2026-01-01T04:00Z')
assert result.status == 'unavailable'
assert result.frame.empty
```

- [ ] Run `python3 -m pytest -o addopts='' -q tests/test_wyckoff_candle_integrity.py --tb=short`. Expected: explicit missing-preparer assertion fails before adding the module.
- [ ] Implement pure UTC bin validation and latest-contiguous-segment selection; preserve native-daily provenance and exclude current/future observations. Add live producer status, separate EMA proxy, and per-TF errors/insufficient-history status.
- [ ] Add failing actual-source offline tests for identical warmup/poll, duplicate processed poll, conflicting duplicates and history counts; implement cached duplicate results and unique buffer append without suppressing first computation.

```python
if ts == last_feature_ts and identical_ohlcv:
    return last_feature_vector.copy(deep=True)
# Warmup-only identical tail skips append but still computes its first result.
```

- [ ] Run the new file, all task 1/2 files and `tests/research/test_live_feature_replay.py`. Expected: all pass without any network attempts. No commit.

### Task 4: Review and handoff

**Files:** Create `docs/knowledge/wyckoff_evidence_repair_2026_10_04.md`; update `PROJECT.md`, `docs/knowledge/MEMORY.md` and this plan's progress ledger only.

**Interfaces:** Review all task 1–3 changes against spec and Review Focus; root verifies reviewer claims before acting.

- [ ] Run full new/previously green selected suite, expanded baseline suite and `python3 -m pytest --tb=short`. Expected: new selection green; known baseline failures/collection blockers explicitly recorded, any new regression investigated.
- [ ] Request independent software review on actual uncommitted changes, including new files. Fix important defects with red-to-green regressions; record smaller unresolved items and scope limitations.
- [ ] Verify diff/whitespace, source hashes, untouched selected configs/runner/frozen artifacts and no running jobs. Write concise results and next semantic-validation action; do not claim profitability or deployability.
- [ ] Preserve local work and ledger for another CLI; no automatic commit/push/PR or deletion of research artifacts.
