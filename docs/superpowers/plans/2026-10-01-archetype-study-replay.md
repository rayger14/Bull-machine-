# Archetype Study Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a verified source pilot and a reproducible paired economic comparison for the frozen R1 and R3 ideas, without modifying live archetypes.

**Architecture:** Keep source opportunity construction independent of execution. Add small, versioned offline modules for R3, native/R1 observation, common execution and reporting, reusing qualified clock/parent/source primitives. Preserve the old reference implementations and their artifacts.

**Tech Stack:** Existing Python 3.9.6, pandas 2.3.3, numpy 1.23.5, pytest 8.4.2, pyarrow 21.0.0 and TA-Lib 0.7.1; no installations.

**Spec:** [Seeing eye design](../specs/2026-10-01-seeing-eye-market-map-design.md) and [canonical bounded study](../specs/2026-09-30-archetype-repair-discovery-design.md).

## Global Constraints

- Preserve all 17 production archetypes, their configuration and frozen research artifacts.
- No live orders/config/fusion changes, model market roles, optimizer, new dependencies, commit/push/PR or prospective collector deployment.
- Existing research branch remains in use; preserve all pre-existing dirty work.
- At most three hypothesis slots are permitted: R1, R2 and R3. R1 and R3 are active; R2 is parked.
- The old upside-LC rule stays unchanged; neither room >=2R nor the old minute confirmation is added.
- Use the hash-bound Binance USD-M BTCUSDT minute archive and completed aggregates from that same stream.
- Pilot arm/decision times: `[2024-01-01, 2024-02-01)` UTC; prehistory starts `2023-12-02` UTC.
- Campaign decisions: `[2024-01-01, 2026-08-31)` UTC; reserve final archive day for outcomes.
- Primary execution allowance 12 bps round trip; stress 24 bps. Mechanical processing 5 seconds; stress 65 seconds.
- Initial intended risk $100, entry-notional cap $50,000, fixed 2R target; no compounding or combined funded-portfolio claim.
- Funding stress: 8 bps of entry notional per UTC 00:00/08:00/16:00 settlement with `entry_time < settlement <= exit_time`.
- Bootstrap 5,000 paired calendar-month blocks, seed 20260930; preserve all 32 months and original decision table.
- Development history is exposed; no claim of a new holdout or walk-forward optimization.
- No economics before complete source-pilot, code/parity, funding/trial and resource locks pass.

## Review Focus

1. An invalidation known while the child forms must not become an eligible armed box; pin in Task 2.
2. Reservation end equal to a new candle's open permits that candle, but no older constituent; pin in Task 2.
3. A gap below stop at the deadline pays the adverse opening price before the time exit; pin in Task 4.
4. A signal blocked by pending capacity stays missed, while R1 signal-time cooldown still advances; pin in Tasks 3 and 4.
5. Full-history parent construction must not silently exceed the protected cap or reset every month; pin in Task 6.

## Execution and resource boundaries

User approved the revised backtesting scope and delegated routine next-step guidance
to a quant agent to keep work moving. Use main-agent implementation with one focused
quant/code review, as that reviewer recommends. Record the review of this written
plan before implementation. Ordinary fixture fixes and in-scope offline preparation
can continue under that delegation; it does not authorize live risk, paid market
roles, publication, extra hypotheses or silent protocol changes.

Use `python3 -m pytest -q` on named files, not an unrelated full-suite collection.
Run one source process at a time. The first real launch is January source-only;
record wall time, output bytes and peak memory. Set a 30-minute pilot wall-time
ceiling and 2 GiB output ceiling; save raw completed work on interruption. These
are operational budgets, not trading parameters. Do not relaunch an expensive
source because a projection/report failed. Review measured cost before the full
32-month census; no unsupported duration promise.

No commit steps are included because the campaign explicitly excludes commits.
Code checkpoints are test evidence plus PROJECT/MEMORY updates. Schema or protocol
corrections found before economic reveal get a versioned receipt; findings after
reveal do not silently change a tested hypothesis.

## File structure and interfaces

All new implementation lives under `scripts/research/`, with focused counterparts
under `tests/research/`. No existing engine/config or frozen replay module is edited.

| New file | Responsibility |
|---|---|
| `study_contract.py` | Frozen constants, UTC/finite validation, canonical IDs and protocol manifest |
| `r3_census.py` | Arm-neutral minute-driven R3 opportunities, event ledger and restart state |
| `study_hourly.py` | Non-mutating per-gate diagnostics and isolated native-TWT/R1 decisions |
| `study_execution.py` | Common equal-risk bracket fills, pending/occupied books and funding |
| `study_cases.py` | Minimal as-of evidence packet, validator and readable card |
| `study_parents.py` | Separately guarded continuous parent construction beyond the old cap |
| `study_source.py` | Qualified minute/aggregate/source capture and launch receipts |
| `study_report.py` | Paired denominators, uncertainty, diagnostics and fixed decisions |
| `run_archetype_study.py` | Offline preflight, pilot, census and score commands with explicit locks |

Shared interchange uses JSON-safe dictionaries and UTC ISO timestamps, not custom
classes crossing worker boundaries. Opportunity rows have `id`, `family`,
`origin_time`, `instrument`, `data_stream_id` and immutable setup evidence.
Signals have `opportunity_id`, `family`, `decision_time`, `stop`, `entry_expiry`,
`exit_deadline`, and `parent_lineage_id` (null for R1). Family labels are `R1` and
`R3`; arm labels are `baseline` and `repair`. No outcome fields enter source rows.

Book results retain one row per raw opportunity, with `status`, `reason`,
`net_pnl` (zero for known nonentries, null for unresolved), `position` (null if
unfilled), and `opportunity_id`. Position fields include entry/exit/observed clocks,
quantity, stop, target, immutable initial risk, fees, funding and net R. Per-minute
marks and occupancy events are separate ledgers. Never infer zero from a missing
row. These common interfaces must be frozen in Task 1 tests before integration.

### Task 1: Frozen contract and portable witnesses

**Files:** Create `scripts/research/study_contract.py`,
`tests/research/test_study_contract.py`, `tests/research/study_fixtures.py`.

**Interfaces:** `protocol() -> dict`, `stable_id(kind, fields) -> str`,
`utc_minute(value) -> pandas.Timestamp`, `finite_number(value, positive=False) -> float`.
Reuse `replay_clock.digest` only on qualified finite JSON-safe values. Protocol
manifest lists source assumptions, canonical spec hash, hypotheses and fixed
cost/delay/funding modes; archive hashes do not enter event identity.

- [ ] Write contract tests, including this actual rejection witness:

```python
import pytest
from scripts.research.study_contract import finite_number, protocol, stable_id

def test_frozen_contract_rejects_boolean_prices():
    with pytest.raises(ValueError):
        finite_number(True, positive=True)
    assert protocol()["active_hypotheses"] == ["R1", "R3"]
    assert stable_id("op", {"at": "2024-01-01T00:00:00Z"}) == stable_id(
        "op", {"at": "2024-01-01T00:00:00Z"})
```

- [ ] Run `python3 -m pytest -q tests/research/test_study_contract.py`; expect missing-module failure first.
- [ ] Implement explicit finite/nonboolean checks, UTC-aware aligned clocks and fresh-copy protocol data. Reject duplicate IDs, non-finite values, naive clocks and unsupported family labels in owning validators.
- [ ] Add fixture helpers that construct contiguous UTC minute OHLCV DataFrames and a minimal valid versioned parent ledger. Fixtures are synthetic and contain no real outcomes.
- [ ] Re-run the named tests; record actual results before marking complete.

### Task 2: R3 census and case projection

**Files:** Create `scripts/research/r3_census.py`, `scripts/research/study_cases.py`,
`tests/research/test_r3_census.py`, `tests/research/test_study_cases.py`.

**Interfaces:**
`build_r3_census(minutes, parent_ledger, *, instrument, data_stream_id, emit_from, end_exclusive, checkpoint=None) -> dict`.
Output keys: `opportunities`, `events`, `attempts`, `checkpoint`, `coverage`,
`blockers`, `schema`. Events carry `opportunity_id`, `available_at`, `kind`,
evidence IDs and relevant frozen prices. `compile_r3_signals(census, arm) -> list`
selects breakout for baseline and trigger for repair, never only common fills.
`project_case(census, opportunity_id, as_of) -> dict`,
`validate_case(packet) -> list[str]`, `render_case(packet) -> str`.

- [ ] Write minute fixtures for a six-bar box `[100,102]` in parent `[90,110]`, arm at 00:30, breakout close at 00:35 above 102, first valid retest close at 00:40 above 102, and 1m close above the retest high at 00:41. Assert the exact clocks and stop at retest low.
- [ ] Run `python3 -m pytest -q tests/research/test_r3_census.py tests/research/test_study_cases.py`; observe failure before implementation.
- [ ] Implement chronological updates: process known parent transitions and minute stop touches, completed 5m state changes, trigger, then arm attempts. Read only candles closed by the update clock.

```python
parent = parent_asof(parent_ledger, child_first_open, strict=True)
eligible_geometry = (
    parent is not None
    and parent["range_low"] <= child_low < child_high <= parent["range_high"]
    and child_high - child_low <= (parent["range_high"] - parent["range_low"]) / 4
)
# A down transition is associated with pre_lineage_id, not post_lineage_id.
# Evaluate known construction-time down events before creating a raw opportunity.
```

- [ ] Pin inclusive A+30/B+30/T+5 final closes, strict break/confirm inequalities, first-touch failure, simultaneous stop/trigger cancellation, parent-down availability, upward break annotation and no rebinding.
- [ ] Keep reservations independent of terminal status and books: A+30 without break, B+40 after break; first new child open >= reservation end, arm no earlier than end+30. Test early cancellation, missing bars and independent arms.
- [ ] Treat right-censoring as an as-of report status, not a terminal event in resumable state. January cutoff retains the unfinished stage and reservation; a February suffix must continue identically to an uninterrupted replay.
- [ ] Preserve immutable raw opportunity rows and append events. Checkpoint retains pending box/stage, recent complete/partial buckets, lineage reservations and last processed clock; it must not retain full raw minute history. Resume only a strictly contiguous suffix.
- [ ] Test future-append invariance of prior IDs and as-of packets; restart at arm, breakout, retest and month boundary; malformed/foreign parent refs; stop-first; zero-case and censored outputs.
- [ ] Project only declared fields from the source ledger. Include qualified parent/child/event citations and explicit missing context. Reject PnL/MFE/MAE/outcome fields and future observations. `execution_authorized` is always false.
- [ ] Re-run both focused files. Do not generate real cases until source prerequisites pass.

### Task 3: Native diagnostics and isolated hourly repair

**Files:** Create `scripts/research/study_hourly.py`,
`tests/research/test_study_hourly.py`.

**Interfaces:** `HourlyStudyObserver(signal_engine)`;
`observer.update(features, decision_time, provenance) -> dict` and
`observer.snapshot() -> dict`. Output contains unchanged native diagnostic,
all-17 gate receipts, raw TWT identity record and independent baseline/repair
decisions/signals. Bind source-hour-close to opportunity origin time.

- [ ] Write tests asserting underlying native signal object/output/cooldown parity with and without observation, and exact 17-name diagnostic coverage.
- [ ] Run `python3 -m pytest -q tests/research/test_study_hourly.py`; expect failure before module exists.
- [ ] Wrap the original call once, restoring patched methods in `finally`/`ExitStack`. Capture structural reason/errors, actual derived-gate returns/exceptions, primitive gate inputs, fusion stages, called versus not-evaluated branches, cooldown and pre-dedup/selected status. Do not run identity twice.
- [ ] Instantiate independent TWT instances using the effective native configuration. Reuse the captured native identity result and regime for each arm; preserve original detect/gate/fusion/stop behavior. The repair rejects before cooldown/detect when its single directional predicate fails.

```python
def directional_permission(value, status):
    import math
    from numbers import Real
    return (status == "observed" and not isinstance(value, bool)
            and isinstance(value, Real) and math.isfinite(value) and value >= 1.0)
```

- [ ] Test above-EMA preservation, below-EMA/high-fusion rejection, missing/defaulted/non-finite/boolean evidence, no cooldown on identity rejection, and cooldown on emitted-but-busy signals. Use actual native configs plus portable synthetic feature rows.
- [ ] Preserve baseline permissive behavior in diagnostics, but mark required structural/derived errors as blocking the affected economic cohort. Record fallback ATR separately from observed ATR. Do not score short/neutral or funding/OI families.
- [ ] Re-run focused hourly and existing signal observer tests. Full source rows must be saved before candidate projection; all-17 relative selection is observational only.

### Task 4: Common equal-risk executor

**Files:** Create `scripts/research/study_execution.py`,
`tests/research/test_study_execution.py`.

**Interfaces:** `position_terms(entry, stop, cost_bps) -> dict`;
`replay_book(minutes, opportunities, signals, *, as_of, cost_bps, delay_seconds, funding_mode, parent_down=(), occupied=True) -> dict`.
Funding mode for the first research run is `adverse_stress` unless qualified actual
same-venue funding is bound before reveal; `zero_diagnostic` is never primary.
Actual funding support must reject unqualified input rather than pretend to exist.

- [ ] Write the hand-calculated risk witness and fail it first:

```python
import pytest
from scripts.research.study_execution import position_terms

def test_cost_inclusive_risk():
    p = position_terms(100.0, 99.0, 12)
    assert p["quantity"] == pytest.approx(100 / 1.12)
    assert p["initial_risk"] == pytest.approx(100)
    assert p["target"] == 102.0
```

- [ ] Run `python3 -m pytest -q tests/research/test_study_execution.py`.
- [ ] Implement the exact formula and pin quantity before observing any exit:

```python
c = cost_bps / 10000.0
q = min(100.0 / ((entry - stop) + c * entry), 50000.0 / entry)
initial_risk = q * ((entry - stop) + c * entry)
target = entry + 2.0 * (entry - stop)
```

- [ ] Process old exits/cancellations before new signals at the same clock. Reserve one pending or open position; busy signals do not retry. Ready time is signal+delay, rounded up to an exact minute strictly after signal. Preserve signal-relative expiry and breakout-relative R3 deadline.
- [ ] Before fill, consume known stop touches, same-lineage down events and adverse opening gaps. Baseline and repair use their own stops/signals, common source IDs and independent books. Cancellation of a retest setup must not cancel an already open breakout comparator.
- [ ] For an open long, settle funding first when carried into a settlement. At deadline open apply adverse stop gap first, then timed exit, never that minute's later extrema. At other opens handle gaps, then completed-bar extremes stop-first if ambiguous. Entry-bar extremes count.
- [ ] Test settlement-at-entry exclusion, settlement-at-exit inclusion, multi-settlement holds, unchanged R0, notional cap, losses beyond R0, fees split between entry/exit, gaps/missing bars, exact expiries, pending displacement and future-append invariance.
- [ ] Return known nonentries as zero and unknown/censored positions as null. Record realized events and minute-close liquidation-value marks net of round-trip allowance/funding; do not infer account returns.
- [ ] Test `occupied=False` as explicitly nonportfolio fixed-event diagnostic; changing another arm never changes a book's result. Re-run the focused suite.

### Task 5: Source qualification and bounded January pilot

**Files:** Create `scripts/research/study_source.py`,
`scripts/research/run_archetype_study.py`,
`tests/research/test_study_source.py`, `tests/research/test_archetype_study_cli.py`.

**Interfaces:** `preflight() -> dict`, `build_pilot(output_dir, *, max_seconds=1800, max_bytes=2147483648) -> dict`.
CLI stages start with `preflight` and `pilot`; `census`/`score` refuse launch until
their verified lock receipt exists. Output directories must be new and exclusive.

- [ ] Test missing archive/helper, wrong hashes/runtime, gaps/partial buckets, invalid output destination, duplicate launch and injected source failure. No test may call network or a model.
- [ ] Run `python3 -m pytest -q tests/research/test_study_source.py tests/research/test_archetype_study_cli.py` before implementation.
- [ ] Bind archive SHA `5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035`, current protocol/code/config hashes and recovered helper hashes from the reviewed parent receipt. Qualify the local `LogEvidence` dependency explicitly; do not silently require it on another CLI.
- [ ] Load only the fixed pilot/prehistory slice from parquet. Validate contiguous minute OHLCV, derive complete 5m/hourly bars on UTC boundaries and causal hourly ATR. The existing capped parent builder is valid for this 1,464-hour input.
- [ ] Run actual feature/signal producer once through warmup+January with the observer from Task 3; persist raw feature/diagnostic rows as they become available. Keep restart/prefix evidence and source-only side-effect guards. Flush partial captures on failure.
- [ ] Build R3 census/cases from the same minutes and parent ledger. Preserve exact source availability, all raw opportunities and complete failure/censor records. Produce coverage counts, earliest raw case, manifest, source hashes and resource receipt, no economic outcomes.
- [ ] Run source and new-module focused tests, then preflight. Launch January only if they pass:

```sh
python3 -m scripts.research.run_archetype_study preflight
python3 -m scripts.research.run_archetype_study pilot --output-dir results/archetype_study_2026_10_01/pilot_v1
```

- [ ] Read saved artifacts independently, confirm all17/R1/R3 scope, immutable hashes, raw-capture recovery, no outcome leakage and runtime/size. Source quality failures block affected certification and are reported; they are not ordinary rejections.

### Task 6: Continuous history without monthly resets

**Files:** Create `scripts/research/study_parents.py`,
`tests/research/test_study_parents.py`; extend `study_source.py` and its tests.

**Interfaces:** `build_study_parents(hourly, *, instrument, data_stream_id, atr_contract, source_paths, expected_hashes) -> dict` returns the same parent ledger semantic fields plus a separately versioned manifest. Source replay remains continuous from December2,2023; restart may replay and verify the same full prefix rather than falsely claiming native state deserialization.

- [ ] Pin frozen-builder parity on <=2,048 hours, using real recovered helpers when available and labelled synthetic fixtures otherwise. Test >2,048-hour synthetic history, multiple parent transitions, partial final buckets and future-append/restart equivalence.
- [ ] Run `python3 -m pytest -q tests/research/test_study_parents.py` before implementation.
- [ ] Implement a separately bounded full-prefix adapter using the same guarded recovered aggregation/pivot/range functions and lineage annotation. Cap at the declared full source horizon, retain historical state through a single chronological construction, and never monkeypatch `MAX_INPUT_HOURS` or join independent monthly parents. Separate versioned manifest differences from semantic parity.
- [ ] Preserve native feature/signal state across months in one replay. Deterministic restart rebuilds from the same original seed and verifies the saved prefix before appending; it does not restart a 30-day lookback each month. Flush/checkpoint output without discarding the in-memory state.
- [ ] Re-run parity, boundary/restart, and existing parent tests. A missing helper or inability to qualify continuous behavior blocks the full census; January remains valid within its stated scope.
- [ ] Quant review checks January resource receipt and extrapolated full-run cost. Bind a new bounded run receipt before allowing `census`; do not launch automatically merely because January produced candidates.

### Task 7: Paired report and economic launch lock

**Files:** Create `scripts/research/study_report.py`,
`tests/research/test_study_report.py`; extend CLI and integration tests.

**Interfaces:** `paired_months(opportunities, baseline, repair, months) -> list[dict]`,
`paired_interval(month_rows, *, draws=5000, seed=20260930) -> dict`,
`campaign_decision(summary) -> dict`, `render_report(results) -> str`.
Month rows contain month, common opportunity count, baseline and repair net sums,
fill counts and unresolved counts. Only complete paired outcomes permit inference.

- [ ] Write a denominator witness: one entered opportunity and nine valid nonentries remain ten, not one. Add a boundary trade with next-month exit/funding attributed to its origin month and marks remaining at actual timestamps.
- [ ] Run `python3 -m pytest -q tests/research/test_study_report.py` before implementation.
- [ ] Bootstrap paired monthly vectors, recomputing the ratio from sampled sums:

```python
count = sampled["opportunity_count"].sum()
estimate = (sampled["repair_net"].sum() - sampled["baseline_net"].sum()) / (100 * count) if count else None
```

- [ ] Include every one of the 32 calendar months. Do not redraw undefined zero-denominator samples. More than 1% undefined yields insufficient evidence; otherwise report their count and the declared 0.0083333333/0.9916666667 quantiles.
- [ ] Implement the canonical ordered decision table verbatim, with secondary failure reasons. Test data-blocked, evidence floors, negative primary/incremental, stress failure, period/concentration failure, nonpositive interval bound and forward-eligible states. None means live-ready.
- [ ] Produce net expectancy, common-opportunity value, counts, busy skips, unknowns, exposure, drawdown, worst loss, excursions, top-three concentration and all four cost/delay scenarios with locked primary funding. Excursions are diagnostic only, not retrospective filters.
- [ ] After complete pilot, source/engine/execution parity, trial ledger, funding and resource locks, obtain Task 8's integrated quant/code review receipt before enabling the economic launch. Tests alone cannot create this lock. Then run the frozen full census and score once. Log all artifacts and failures. If blocked, report the precise blocker rather than substitute a partial or selected-trade backtest.

### Task 8: Integrated review and handoff

- [ ] Main self-review checks each spec requirement against a task and exercised test, with no unsupported completion claims.
- [ ] Quant reviewer reviews the written plan before implementation and the integrated result before economic launch. Re-review only material corrections; do not start another teaching audit.
- [ ] Run all named new focused tests plus relevant existing observer/parent/clock regressions. Record exact passes, skips and failures; unrelated full-suite blockers are not called a pass.
- [ ] Recompute protected29 digest and canonical spec hash; verify old artifacts unchanged and `git diff --check` clean.
- [ ] Update PROJECT/MEMORY with implemented modules, actual tests, completed/running stages, source resources, local-only inputs and next permitted action.
- [ ] Deliver the result as a source report, economic comparison, or explicit engineering/data blocker according to the stage actually reached. Never describe plan completion or synthetic test success as a profitable backtest.
