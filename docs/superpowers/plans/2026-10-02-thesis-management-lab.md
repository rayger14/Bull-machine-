# Thesis management lab implementation plan

Goal: Build and qualify a new offline causal thesis and adaptive manager, ending
with a bounded January engineering replay rather than a profit claim.
Architecture: shared evidence and episode sequence; one execution reducer;
independent four-arm books and identical-entry attribution.
Stack: existing Python, pandas, numpy, pytest, PyArrow; no new dependencies.
Spec: `docs/superpowers/specs/2026-10-02-thesis-management-lab-design.md`.

## Global constraints

Preserve all frozen research, all 17 archetypes, live/config/fusion and existing
dirty work. No paid market roles, downloads, installs or git publication. Existing
quant checkout is user-selected. Quant/software delegated reviews replace routine
approval pauses. All formulas are explicit project hypotheses, not trader-certified.
Only the January source and engineering replay are in this milestone.

## Review focus

Source availability and strict parent binding; no same-bar sequence advancement;
first-failed-touch cancellation; immutable anchors; missing-vs-absent semantics;
stop replacement latency and old-stop protection; fee/funding/quantity conservation;
source/arm-bound restart; attribution vs occupied-book distinction; independent
capacity; no end-of-data liquidation or suppressed unknown; no exposed-history
promotion. Inspect literal arithmetic, not only helper-derived expectations.

### Task 1: Contract and causal sequence

Files: new `scripts/research/thesis_contract.py`, `thesis_sequence.py`,
`tests/research/thesis_fixtures.py`, `test_thesis_sequence.py`.
Interfaces: `protocol()`, `seal(value)`, `clock(value)`, `event(...)`,
`compile_episode(base, events) -> packet` with sealed finite JSON output.
Implement the exact spring/test/strength/LPS/minute order and deadlines in spec,
including explicit terminal/expired/unknown state, daily context, Fib and clocks.
Write tests first for full sequence, out-of-order/lookahead, failure/expiry, fixed
anchors, unknowns, gaps, no-context/no-lineage rejection, stable prefix IDs.
Run `python3 -m pytest -o addopts='' -q tests/research/test_thesis_sequence.py`.
Expected RED: missing module/API; GREEN: all cases pass with literal event clocks.
No commits. Record tests and reviewer contract clearance in plan ledger.

### Task 2: Position and occupied-book replay

Files: new `scripts/research/thesis_execution.py`,
`tests/research/test_thesis_execution.py`.
Interfaces consume Task1 packet; `replay_book(packets, minutes, entry, management,
execution=None, checkpoint=None, until=None, capacity=True) -> sealed book`.
Support capacity-free identical entries by `capacity=False`; adaptive actions
never alter entry tapes there. JSON checkpoint resumes a processed minute prefix.
TDD: exact risk/fee arithmetic, partial original fractions, funding reduced qty,
stop-gap and same-bar stop-first, delayed trail, unknown occupancy, no-progress
clock, lower-priority cancellation, foreign checkpoint rejection, prefix/resume,
fixed/adaptive identical entry tapes, independent occupied arms.
Run `python3 -m pytest -o addopts='' -q tests/research/test_thesis_execution.py`.
Expected RED: missing replay; GREEN: exact quantities/clocks and all cases pass.
No production imports for broker/exit logic; no code changes to older executors.

### Task 3: Shared continuous source and locked CLI

Files: new `scripts/research/thesis_source.py`, `thesis_study.py`,
`run_thesis_study.py`; `test_thesis_source.py`, `test_thesis_study.py`.
Interfaces: `build_source(minutes, parents, start, end) -> source`; complete
aggregation once; `run_source(output)`, `run_engineering(source, output, review)`.
Use pinned archive/parent SHA; full source-only provenance receipt with exact
calendar, code/protocol hashes, coverage and candidate counts. No native LC filter.
CLI denies overwrite, full campaign, unreviewed scoring or changed source/code.
Build exactly four books plus capacity-free paired runs, common-ID/null-aware
report, per-position/cashflow audit. Write tests first for synthetic full pipeline,
strict parent, complete candle gaps, seeded ATR, prefix invariance, output guards,
hash tampering, four arms and reconciliation.
Run `python3 -m pytest -o addopts='' -q tests/research/test_thesis_source.py tests/research/test_thesis_study.py`.
Expected RED: missing source/study; GREEN: all tests pass, no unqualified scoring.

### Task 4: Bounded qualification and handoff

Run focused new suite plus protected LC/parent regression; attempt repository suite
once and identify unrelated blockers without repairing them. Freeze protocol/code,
run source-only January pilot under resource cap. Quant checks counts/provenance
without economics; root checks hashes and source citations. If clear, write exact
review receipt and run one four-book January engineering replay. Independently
reconcile saved quantities/fees/funding/net and paired entry equality. Request fresh
whole-diff software review and fix material findings test-first before final claims.
Expected: zero unresolved source/economic paths, deterministic reports; sparse or
zero thesis entries remains an honest engineering result, not permission to relax.
Write `docs/knowledge/thesis_management_checkpoint_2026_10_02.md`, update PROJECT
and newest MEMORY with actual status, commands, artifacts, limits and next action.
No full-development launch, policy tuning, commits/push/PR or live edits.
