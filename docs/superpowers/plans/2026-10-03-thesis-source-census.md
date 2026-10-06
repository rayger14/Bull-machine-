# Full calendar thesis source census implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Count the unchanged spring sequence across the development calendar, without calculating trade outcomes or tuning rules.

**Architecture:** Add a separate indexed source adapter using the frozen candle, ATR, pivot and episode compiler. Preserve January packets and file bindings. The run calendar is separate from the legacy compiler identity. Root implements inline; the delegated quant reviews design and saved counts; one fresh software reviewer checks the extension.

**Tech Stack:** Existing Python, pandas, pytest and standard library only.

**Spec:** [Frozen thesis and adaptive management contract](../specs/2026-10-02-thesis-management-lab-design.md). The following bounded source-only extension is authorized by the user's October 3 “proceed but explain before”; no full economic campaign is authorized.

## Global constraints and source extension

- Keep the existing quant checkout. Do not modify the frozen thesis modules, January spec, native archetypes, configuration or live orders. No git publication or paid market assessors.
- Aggregate seed: `2023-12-02T00:00:00+00:00`. Origin closes: `[2024-01-01T00:00:00+00:00, 2026-08-24T00:00:00+00:00)`. Source tail exclusive end: `2026-09-01T00:00:00+00:00`.
- Use the archive and parent hashes pinned in the frozen spec. Consume one lineage only at its first qualifying spring in the origin calendar; never reset at month boundaries.
- Retain `seal(protocol())` as the explicitly named legacy compiler/packet identity, including its January run fields. Derive a separate rule fingerprint by removing only calendar/resource/authority fields from that protocol. Calendar dates are campaign metadata, not replacement compiler constants.
- Record every expected four-hour origin close and its mutually exclusive admission disposition. Report cumulative milestones and first entry decision/closure by origin month, not final thesis state after a later invalidation. Unknown evidence is not a rejected setup.
- Source-only means no fills, PnL, returns or economic policy ranking. Packets still contain later source prices. Fib anchors, available review clocks and destination definitions are source observability, not executed actions.
- One process, 600 seconds wall time, 268435456 bytes total output; no automatic retry. Record runtime and peak RSS separately; output size is not a RAM cap. Never overwrite an existing output directory.
- Hash inputs and implementations before and after running. Write a launch record before processing and a failure record on exceptions. Save packets, catalog, raw decisions, monthly summary and receipt under `results/thesis_census_2026_10_03/source_v1/`.
- Stop at reviewed source counts. Intents are only an upper bound for the old 50-fill/12-month research floor. Exposed history is not a holdout; enough intents would not establish enough fills or edge.

## Review focus

1. Equal-time parent transitions must remain excluded under strict as-of, with stable tie order.
2. A lineage spanning a month/year boundary must not originate twice; warmup must not consume it.
3. Leading/trailing source truncation must fail, while interior gaps remain explicit unknowns.
4. An entry or known expiry before later missing/invalid structure must not be retroactively reclassified.
5. January packet parity and closed-prefix invariance must survive indexed source selection.

## Task 1: Indexed census and summaries

**Files:** Create `scripts/research/thesis_census.py`, `tests/research/test_thesis_census.py`.

**Interfaces:** `build_census(minutes, parents, start, end, *, seed, source_end, stream)` returns a signed source record; `summarize_census(source)` returns a finite JSON source-only monthly summary. `ParentIndex(ledger).asof(clock)` implements strict parent selection.

- [ ] Write tests first for strict/tied/invalid parent transitions, old-builder packet/catalog parity, complete literal funnel, interior gaps, short coverage, continuous lineage consumption and first-closure classification. Example behavioral assertions:

```python
assert result['counts']['raw_episodes'] == 1
assert result['counts']['thesis_intents'] == 1
assert result['decisions'][1]['disposition'] == 'raw_episode'
assert summary['totals']['thesis_intents'] == 1
assert summary['economic_outcomes_computed'] is False
```

- [ ] Run `python3 -m pytest -o addopts='' -q tests/research/test_thesis_census.py`. Expected: missing new implementation fails; after implementation all cases pass.
- [ ] Index parent transition clocks and versions once; use bisect-left for strict time. Aggregate source once and slice ordered observations by available time. Reuse unchanged `compile_episode`; construct only last-support minute windows, not millions of event objects.
- [ ] Summarize raw dispositions, stage milestones, first entry status, Fib anchors and review-clock availability, source coverage and monthly distributions. Keep all raw episodes regardless of eligibility.
- [ ] Verify focused old thesis tests and new tests together; record results in the progress ledger. No commit: local checkpoint only.

## Task 2: Bounded launch and verified handoff

**Files:** Create `scripts/research/run_thesis_census.py`, `tests/research/test_thesis_census_run.py`, and a dated report under `docs/knowledge/`. Update `PROJECT.md` and `docs/knowledge/MEMORY.md` after results are verified.

**Interfaces:** `run(output)` uses the fixed calendar, pinned files and `build_census`; CLI accepts only `--output`. No economics command or arbitrary calendar override.

- [ ] Test no-overwrite, changed file rejection, cumulative output cap and failure records before implementing. Use temporary local inputs only for bounded runner tests; the pinned production launcher must not accept unqualified inputs.
- [ ] Run `python3 -m pytest -o addopts='' -q tests/research/test_thesis_census.py tests/research/test_thesis_census_run.py`. Expected: red for missing behaviors, then all pass.
- [ ] Save launch binding, run once after quant clearance, verify output seals/hashes and compare January packets to the saved January artifact. Build a fixed-prefix witness from the same archive without computing returns.
- [ ] Have the quant independently reconcile raw counts, stage decisions and calendar reporting from source evidence. Run a fresh software review, fix important findings test-first, then rerun focused and full repository checks. Report unrelated full-suite failures explicitly.
- [ ] Write a digestible result report and next-action decision. Keep all old artifacts and this plan's verification ledger; no cleanup or publication is authorized.

## Decisions already made

Keep the original hypothesis instead of loosening it after January. If the setup is sparse, report sparsity. Use indexed selection only to reduce repeated scans, with parity tests to guard semantic changes. Existing frozen execution modules are not called for this census.
