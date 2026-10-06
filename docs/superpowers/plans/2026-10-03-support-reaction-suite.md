# Support-Reaction Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify one evidence-linked support-reaction policy and its research acceptance suite.

**Architecture:** New support_reaction modules consume original spring origins and causal minute prices. They separate observations/decisions, fixed execution transport, and source/semantic/economic study boundaries. Reuse the frozen replay, event split, hashing and bounded output utilities without editing them.

**Tech Stack:** Existing Python, pandas, numpy, pytest and standard library.

**Spec:** [Approved quant-reviewed contract](../specs/2026-10-03-support-reaction-suite-design.md).

## Global Constraints

- Existing quant checkout; preserve all frozen/native/live/config/fusion files.
- No paid API calls, installs, commits, push or PR. Retain local progress/artifacts.
- Numeric rules are project hypotheses; phase stays unclassified.
- ALL183 raw source episodes, never old17 support survivors. Exposed history is development data.
- Root implements inline to limit contexts; delegated quant approves technical design and one fresh software reviewer checks whole addition.
- Source/benchmark launch only: one process,600seconds,256MiB aggregate output, exclusive directory, no retry. Natural economics needs a separate source/semantic review.

## Review Focus

1. Missing volume must not poison price-only B; missing required price must not become neutral or a no-trade zero.
2. Future preflight cancellation must preserve pending capacity before discovery.
3. Current opening may be visible while its future candle extremes are not.
4. Missing normalization intervals may not be replaced by older bars; pivots need delayed causal availability.
5. Source identity, full denominator and negative/ambiguous semantic evidence may not vanish during adapters or splits.

## Task 1: Causal evidence and decisions

**Files:** Create `scripts/research/support_reaction.py`, `tests/research/support_reaction_fixtures.py`, `tests/research/test_support_reaction.py`.

**Interfaces:** `origin_record(old_packet)` strips future tails into a signed new origin record; `assess(origin, minutes, *, as_of)` returns a signed record containing source observations, range/phase facts, recovery/support/child anchors, two decisions B/C and causal unknown/invalid clocks. `evidence(candle, prior20, role)` produces known/unknown ratios and supportive/neutral/adverse classifications. `policy()` is the immutable v1 contract. `verify_record(record)` rejects tampering or foreign policy.

- [ ] Build literal synthetic candles and an old packet fixture. Write tests using delayed import so absent implementation is a behavioral failure:

```python
def test_local_recovery_and_real_minute_structure():
    raw, bars = fixture()
    api = import_api()
    out = api.assess(api.origin_record(raw), bars, as_of='2024-01-02T06:10Z')
    assert out['decisions']['B']['at'] == '2024-01-02T06:09:00+00:00'
    assert out['phase']['state'] == 'unclassified'
    assert out['decisions']['C']['status'] == 'intent'
```

- [ ] Run `python3 -m pytest -o addopts='' -q tests/research/test_support_reaction.py`; expect failures for missing module before implementation.
- [ ] Implement consumed-prefix hourly/minute observations,20-interval evidence and ordered recovery/support/high/low/trigger with distinct terminal decisions. Cite all operands; stable decision IDs cannot contain future totals/cutoffs.
- [ ] Exercise supportive+neutral in both directions, adverse/unknown alternatives, price gaps, boundary clocks, earlier failed support, first pivot locks, same-bar prohibition, parent/source corruption and future append/mutation. Runtime/price errors must not be softened into profitable rejections.
- [ ] Run focused new and old sequence/source tests; expect all passing. Record actual result, no commit.

## Task 2: Fixed execution transport and chronological comparison

**Files:** Create `scripts/research/support_reaction_replay.py`, `tests/research/test_support_reaction_replay.py`.

**Interfaces:** `replay(records, old_packets, minutes, *, arm, execution=None, capacity=False, as_of=None)` returns a signed new-policy book wrapping original fixed runtime outputs and explicit admission evidence. A is old-thesis identity; B/C are tagged runtime transports. `compare(records, old_packets, minutes, *, execution=None, as_of=None)` produces independent arms and paired attribution. `folds(origins)` calls existing event-label splitter using fixed blocks and maximum7d horizon.

- [ ] Write real fixture replay tests before implementation:

```python
def test_unknown_volume_is_not_an_avoided_loss():
    raw, bars = fixture(missing_volume=True)
    record = assess(origin_record(raw), bars, as_of=raw['deadline'])
    out = replay([record], [raw], bars, arm='C')
    assert out['rows'][0]['net'] is None
```

- [ ] Run the new replay tests; expect absent module failures.
- [ ] Verify record/origin/arm bindings before creating a legacy runtime envelope. Retain pending intent; annotate cancellation/unknown at discovery. Recheck room and child support on the delayed opening without consuming its future extrema. Keep original stop,2R,7d,$100 risk and common costs.
- [ ] Tests cover A exact runtime parity, B/C delayed admission, room boundary, gaps, unknowns, second-episode pending occupancy, prefix replay, distinct policy/runtime identities and all raw IDs. Never sum independent case PnL as portfolio return.
- [ ] Compare three arms under same fixed execution. Keep original candidate denominator, explicit known subtotal/complete totals and paired winner/loser dispositions. Event-label folds use original T0+7d, show excluded IDs and no model fitting.
- [ ] Run new replay plus existing execution/walk-forward tests; expect all passing. Record actual result, no commit.

## Task 3: Benchmark, bounded source runner and final verification

**Files:** Create `scripts/research/support_reaction_study.py`, `scripts/research/run_support_reaction_study.py`, `tests/research/test_support_reaction_study.py`, `docs/knowledge/support_reaction_suite_2026_10_03.md`. Update PROJECT/MEMORY.

**Interfaces:** `benchmark_roster(origins)` deterministically selects3 lowest ID hashes per fixed block; `blind_packet(record)` exposes only predecision source and rubric; `validate_annotation(packet, annotation)` checks source-cited required fields and unknown/disputed states, not agreement with our code; `run_source(output)` writes signed source, summary, benchmark and receipt under fixed caps. CLI accepts `source --output`; no automatic natural economics launch.

- [ ] Add output/selection/annotation/CLI guard tests first:

```python
def test_annotation_cannot_cite_future_or_foreign_evidence():
    packet = natural_style_packet()
    with pytest.raises(ValueError):
        validate_annotation(packet, annotation(citations=['future-event']))
```

- [ ] Run the new tests; expect missing functions/modules failures.
- [ ] Implement deterministic source projections and annotation validation. Reuse bounded Output and timeout/cost guards, but new launch/receipt schemas and policy. Bind source, receipt, archive, implementation/tests/spec BEFORE compute; verify afterward. Record failure without success receipt on invalid dependencies.
- [ ] Export all183 source decisions and the12-case outcome-hidden roster. Keep independent annotation explicitly incomplete, no fake expert labels. Source candidate paths include every rejection/unknown with evidence-linked reasons.
- [ ] Fresh software review checks only new files and frozen integration boundary; fix important findings test-first. Root then runs all relevant research tests and bare full-repository pytest; report unrelated legacy collection failures by name.
- [ ] After review, one source/benchmark launch under600s/256MiB; no natural PnL. Verify all artifact/input hashes and A old-count3 parity, source/decision prefix checks and raw denominators. Synthetic replay tests are engineering evidence, not strategy results.
- [ ] Update report, progress, PROJECT and MEMORY with results, runnable commands, what remains and local-only dependencies. No publishing or live changes.

## Self-review and delegated review

All spec interfaces have one owner; Task2 consumes Task1 signed records, Task3
consumes both. No edits to frozen dependencies. Quant review approved the spec
after explicit pending-capacity, baseline-gap, subset and cutoff clarifications.
Full natural economics/independent semantic annotation are separate stages, not
promised completed by exporting packets. Do not silently choose winners later.
