# Causal Parent Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline, source-attributable parent structure ledger shared by hourly and minute consumers.

**Architecture:** Guard and hash the recovered pivot and range functions, then wrap their outputs in causal immutable identities and transitions. Keep native signals, execution and all 17 archetypes untouched. Use the existing replay clock for closed-bar validation and aggregation.

**Tech Stack:** Python 3.9, pandas, numpy, unittest/pytest, existing research side-effect guard.

**Spec:** `docs/superpowers/specs/2026-09-10-causal-parent-ledger-design.md`

## Global Constraints

- Research-only: no production `engine/`, `bin/`, `configs/`, sibling or original data edits, network, optimizer, deployment or public push.
- All outputs `certified=False`; ATR and source contracts must be explicit.
- Support explicit `4H`/`1D` and integer `pivot_n` in {3,5}; no optimized/default winner.
- Batch cap 2,048 hours; reject above cap, never silently truncate.
- Preserve recovered 1.5/0.75 ATR widths, zero-buffer one-hour close breaks, tightening and reused-anchor reformation.
- Retain original source unchanged and execute no main blocks. Exact source hashes below are required for the integration tests.
- Fresh full-history restart only; no serialized partial restore claim.
- New code must remain Python 3.9-compatible. Do not commit private raw data or result artifacts.

## File responsibilities

- `scripts/research/causal_parent_ledger.py`: validated source loader, complete-anchor construction, immutable ledger, as-of queries and child bindings. Split a helper into `scripts/research/parent_source_adapter.py` only if needed to keep these responsibilities readable.
- `tests/research/test_causal_parent_ledger.py`: deterministic fixtures, causal invariants, source parity and side-effect tests.
- Existing `replay_clock.py` and `virtual_book_replay.py` are read-only dependencies.

### Task 1: Frozen source adapter and causal ledger

**Files:**
- Create: `scripts/research/causal_parent_ledger.py`
- Optional create: `scripts/research/parent_source_adapter.py`
- Create: `tests/research/test_causal_parent_ledger.py`

**Interfaces:**
- Consume `replay_clock.validate_bars`, `context_at`, `utc`, `digest`, `json_safe`, and `virtual_book_replay.side_effect_guard(records)`.
- Produce `build_parent_ledger(bars, *, instrument, data_stream_id, anchor_timeframe, pivot_n, atr_contract, source_paths, expected_hashes)` returning a serializable dict; `parent_asof(ledger, decision_time, strict=False)` returning a copied active version or None; `bind_parent(ledger, *, child_event_id, child_timeframe, first_sweep_open)` returning copied immutable binding or explicit absent-parent rejection.
- ATR contract keys: nonempty `source`, `formula_id`, `version`, plus `availability_policy='hour_close_assumed'`. Optional `available_at`/`atr_available_at` input columns, when supplied, must equal each row's UTC hour close. No receipt certification.
- `pivots`, `versions`, `transitions` are lists of dictionaries with stable `id` and `available_at`. Include manifest contract ID/source/helper/runtime hashes, input hash, coverage, and incomplete/developing diagnostics. Contract IDs must not depend on full future input content.

Recovered paths (explicit caller configuration, not hidden production dependency):

```python
source_paths = {
    'htf': '/Users/rayghandchi/Bull Machine/one-strategy/idea_lab/htf_pivots.py',
    'range': '/Users/rayghandchi/Bull Machine/one-strategy/idea_lab/structural_range.py',
}
expected_hashes = {
    'htf': '0c37d254cba93f40638ffeb2422d53649ea326adecee0a0690ed417b2f492f0b',
    'range': '22e3e1ab21d8db585d89569c3d1d4a3fe3c74ce17ca72c5080a0ca7cf0caa8ca',
}
```

- [ ] **Step 1: Write deterministic failing tests.** Tests must exercise real source functions, not mock a successful ledger. Skip only source integration tests with an explicit missing-local-source reason on machines without the sibling checkout; validation and query tests must still run there. Build aligned UTC hourly fixtures with actual varying OHLCV and explicit ATR. A seed fixture:

```python
index = pd.date_range('2026-01-01', periods=1200, freq='h', tz='UTC')
x = np.arange(len(index))
center = 110 + 7 * np.sin(x / 35.0) + 2 * np.sin(x / 7.0)
bars = pd.DataFrame({'open': center, 'high': center + 3,
                     'low': center - 3, 'close': center,
                     'volume': np.ones(len(x)), 'atr_14': np.full(len(x), 2.0)}, index=index)
```

Include actual assertions for all four anchor/N combinations: source range columns equal corresponding transition diagnostics; confirmation not before N right-side buckets close; parent available at source hour close; strict binding excludes equal-time confirmation; immutable bindings survive later updates. Add controlled source-output seam tests, if necessary, for the exact old-floor 90/new-floor100/low95/close110 tightening witness and 110→121→110 reused 90/120-anchor reformation. Seam tests supplement, not replace, real-source integration.

Pin invalid grid/NaN OHLCV, negative/infinite/nonnumeric ATR, missing ATR contract, source hash mismatch, negative/bool/invalid N, oversized input, naive query, out-of-coverage query, source side effects, empty complete buckets. NaN/zero ATR must remain unchanged and receive quality flags. A 03:00 start must exclude the incomplete leading 00:00 4H bucket. Input half-hour shift and missing interior hour must reject.

```python
cutoff = bars.index[799] + pd.Timedelta('1h')
for key in ('pivots', 'versions', 'transitions'):
    visible = [r for r in full[key] if pd.Timestamp(r['available_at']) <= cutoff]
    assert visible == prefix[key]
assert rebuilt == full
assert full['certified'] is False
pd.testing.assert_frame_equal(bars, original)
```

Compare minute→hour aggregation using `context_at` to identical hourly data, with identical explicit ATR and data_stream_id. Do not claim different-venue parity. Verify equal-priced pivot occurrences have distinct IDs while unchanged bounds retain originally adopted anchor IDs.

- [ ] **Step 2: Run and record RED.**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -o addopts='' -q -p no:cacheprovider tests/research/test_causal_parent_ledger.py
```

Expected missing-module/function failures before implementation, not silent skips.

- [ ] **Step 3: Implement the guarded adapter and ledger.** Read the exact recovered functions first. Under the existing guard, compile/execute verified source bytes in a non-main module namespace (no bytecode writes). Restrict inputs to complete independently aggregated anchor buckets; convert only the private source copy from UTC-aware to UTC-naive and restore/check UTC epoch identity. Call unchanged `detect_fractal_pivots`, `_broadcast`, and `build_structural_range`; compare recovered aggregation to independent complete buckets. Do not reimplement the range strategy.

Use source output to annotate the sequential state; the core distinction is:

```python
if post_state == 'active' and pre_version is None:
    reason = 'formation'  # new lineage even if same old anchors return
elif post_state == 'active' and post_low != pre_low:
    reason = 'floor_tightening'  # new version, same lineage
else:
    reason = None
```

Formation adopts latest visible low/high pivot IDs. Tightening adopts the new low ID and preserves the ceiling ID. Record sweep/break evaluation against pre-version, not newly emitted post-floor. Append transitions on every hour, retain broken geometry diagnostically, and return no active parent after break. IDs hash source contract, stream identity and causal event/predecessor only. Never mutate an old version with a future retirement time.

Parent lookup uses UTC timestamps with inclusive or strict comparison; before coverage/no active state returns None. At or beyond `last_processed_close + 1h`, raise `ValueError('out_of_coverage')`; carry is allowed before that boundary. Binding uses strict `< first_sweep_open`, with copied fixed geometry and stored lineage/version/availability. Changing the returned record must not mutate the ledger.

- [ ] **Step 4: Run focused tests and full research regression.**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -o addopts='' -q -p no:cacheprovider tests/research/test_causal_parent_ledger.py
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -o addopts='' -q -p no:cacheprovider tests/research
git diff -- engine bin configs
```

Expected all pass, zero production diff. In the report distinguish real-source parity tests from synthetic annotation seam tests and any skipped local integrations.

- [ ] **Step 5: Self-review and commit only owned code/test files.**

```bash
git add scripts/research/causal_parent_ledger.py tests/research/test_causal_parent_ledger.py
git commit -m "Add guarded causal parent structure ledger"
```

If the optional helper was created, add it explicitly. Do not stage other workers' evidence policy or docs. Report exact tests, RED/GREEN evidence, commits, remaining limits and any decisions not covered by this contract. Do not spawn subagents; the controller supplies independent review.
