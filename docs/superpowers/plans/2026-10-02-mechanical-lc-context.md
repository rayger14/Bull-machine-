# Mechanical LC context implementation and economic test

**Goal:** Determine whether one fixed, contextual LC entry policy improves on
immediate entry and unconditional minute confirmation, without changing live LC.

**Architecture:** Immutable predecision evidence feeds a deterministic scenario
classifier, then an isolated per-subtype bracket simulator and paired report.
Separate source preparation from outcome scoring. Reuse protected arithmetic and
source construction only through tested interfaces; never modify old studies.

**Tech Stack:** Existing Python 3.9, pandas, NumPy, TA-Lib, PyArrow and pytest.
No packages, network services or market-model calls are required.

**Spec:** [Mechanical LC context design](../specs/2026-10-02-mechanical-lc-context-design.md).

**Execution:** Root implements inline with test-driven development. The requested
quant and software reviewers guide source/accounting qualification; a fresh
integrated reviewer checks the final implementation. The user's October 2
instruction delegates routine continuation rather than requiring another approval
prompt per artifact. No commit steps are authorized. Preserve the ledger locally.

## Global constraints

- Work in the existing `quant/archetype-evidence-audit` checkout as requested.
- Only new `lc_context_*` research modules, their tests, runner, report and handoff
  change. Existing code, frozen artifacts and all 17 production archetypes stay intact.
- No live orders/config/fusion changes, publication, paid market assessors,
  training, downloads or dependency installation.
- Bind input bytes, current consumer/helper bytes, archive identity and policy
  before scoring. Output directories must be new; never overwrite a historical run.
- All 146 expected native candidates remain in coverage, including three expected
  unresolved geometries. Reconcile actual receipts rather than selecting 146 IDs.
- Parent source choice, missingness and execution conventions are frozen before
  economic outcomes. Known abstentions are zero; unknown admitted outcomes are null.
- The complete exposed calendar is January 2024 through August 2026. No holdout,
  walk-forward training, CPCV, full Wyckoff or hidden-Fibonacci/Gann claim is made.
- One source run and one economics run after qualification, each at most 600 seconds
  and 512 MiB artifacts, with peak RSS telemetry. No automatic parameter retries.
  An engineering defect can be corrected with a reproducer and an explicit new
  run version; an adverse result cannot trigger tuning.

## Review focus

1. Future leakage through parent selection, partially completed candles or trigger
   processing clocks. Tasks 1, 2 and 4 exercise as-of and prefix invariance.
2. Controls accidentally inheriting parent gates, lost subtype denominators or
   stale context repairing an old setup. Tasks 1, 2 and 5 exercise missingness.
3. Stop/target ambiguity, gap prices, settlement timestamps and capped sizing.
   Task 3 uses literal arithmetic and independent existing-primitives witnesses.
4. Shared books, post-hoc busy filtering and missing-bar paths treated as zero.
   Tasks 3 and 5 exercise causal occupancy, restart and unresolved publication.
5. Byte seals mistaken for source qualification or newly generated rules presented
   as trader-certified. Tasks 4 and 6 reconstruct inputs and audit output claims.

## Interfaces

`protocol()` returns the fixed JSON policy and `seal(value)` hashes canonical JSON.
`prepare_evidence(raw, minutes, parents, provenance)` returns a predecision case:
candidate ID, T/S, subtype, common-source and risk statuses, prior/current hourly
candles, H5, stop, bound 4H/daily versions, acceptance histories, optional annotations
and source/policy hashes. No post-T bar is consumed.

`classify(case)` returns scenario, state, action (`immediate`, `wait`, `none`),
trigger level, frozen parent binding, necessary predicates and exact reasons.
`entry_location_ok(decision, opening)` checks only contextual executable location.

`replay_book(cases, windows, arm, subtype, cost_bps, delay_seconds, funding_mode)`
returns every subtype ID, disposition, position/accounting, transitions, marks,
capacity releases and blockers. `windows[id]` is the same-stream T through T+24h
minute window including the deadline opening. No shared mutable book state.

`prepare(output)` verifies and saves cases, coverage, policy and a source receipt.
`score(output, review)` requires a matching reviewed source/code lock, produces
48 independent books and a comparison, and writes a completion receipt last.
Both commands fail closed on changed bindings, output collisions or resource limits.

### Task 1: Freeze policy and build causal evidence

**Files:** Create `scripts/research/lc_context_contract.py`,
`scripts/research/lc_context_evidence.py`, `tests/research/lc_context_fixtures.py`,
`tests/research/test_lc_context_evidence.py`.

**Produces:** Protocol, canonical seals, timestamp/finite-number validation,
predecision case and parent acceptance stream. **Consumes:** Verified raw native
candidates, minute bars and saved N3 parent ledgers, never outcomes.

1. Write synthetic fixtures with T=2024-01-03T12:00Z, a 4H version known before S,
   frozen L=90/U=110, complete candles and hand-derived current/prior geometry.
   Failing behavior examples:

   ```python
   assert case['subtype'] == 'upside_expansion_candidate'
   assert case['risk_status'] == 'unknown'  # ATR missing, geometry unchanged
   assert acceptance['state'] == 'inside'  # a later 1h wick/break is not a 4H close
   assert select_parent(ledger, S)['bound']['id'] == 'latest-broken-version'
   ```

2. Run `python3 -m pytest -o addopts='' -q tests/research/test_lc_context_evidence.py`.
   Expected: missing implementation behavior fails, not fixture syntax errors.
3. Implement geometry-first subtype, exact two-hour reconstruction, independent
   risk/H5 validity, immutable most-recent version selection with 30-day formation
   rule and strict pre-S availability. Retain historical broken versions.
4. Aggregate each acceptance candle from all required minute constituents; open
   must be at/after version availability and close <= T. Missing coverage is unknown,
   no eligible close is not_established, strict outside/inside/equality are distinct.
   Preserve same-version transitions and source identities.
5. Add prefix/future append, equality, minute gap, nonfinite source, future pivot,
   duplicate identity, absent versus unknown, daily optional and precedence tests.
   Run the task command; expected all green and old evidence tests unchanged.

### Task 2: Implement contextual decisions and explanations

**Files:** Create `scripts/research/lc_context_controller.py` and
`tests/research/test_lc_context_controller.py`.

**Consumes:** Task 1 case. **Produces:** Pure deterministic decision and case card;
no order hook, `execution_authorized=false` on every record.

1. Write a table-driven failing test for the complete spec scenario table:
   accepted expansion => immediate; contained local expansion => wait H5;
   rebound sweep plus inside parent => wait max(L,H5); unconfirmed parent breakout
   => watching/no reservation; accepted-below rebound => invalidated; absent =>
   outside; unknown => insufficient. Literal example:

   ```python
   assert classify(rebound)['trigger_level'] == 103.0
   assert classify(unconfirmed)['action'] == 'none'
   assert classify(unconfirmed)['state'] == 'watching'
   ```

2. Run `python3 -m pytest -o addopts='' -q tests/research/test_lc_context_controller.py`;
   expected new behavior red. Implement named pass/fail/unknown predicates with
   conflict detection, bound-parent identity and readable factual reasons.
3. Test optional-score duplication/removal cannot change permission; same local
   signal with different established parent changes scenario; no fallback parent
   or scenario migration; actual-fill location strictly checks L/U only for context.
4. Run Tasks 1–2 tests; expected green. Record exact limitations in case cards.

### Task 3: Implement independent LC execution and accounting

**Files:** Create `scripts/research/lc_context_execution.py` and
`tests/research/test_lc_context_execution.py`.

**Consumes:** Cases/decisions and independent outcome minute windows. **Produces:**
One occupied book plus event/mark ledger. Reuse `study_execution.position_terms`
only; do not pass LC into its R1/R3-only signal/book interface.

1. Write failing literal-clock tests: 90-second immediate fills at T+2; wait first
   eligible candle opens T+2/closes T+3, fills T+5; 300-second wait fills T+11.
   Entry at T+15 is prohibited; a stop touch during either processing interval
   cancels before a trigger/fill. No use of the entry minute's future extrema.
2. Run `python3 -m pytest -o addopts='' -q tests/research/test_lc_context_execution.py`.
   Implement the pending state machine then long bracket simulation with stop-first
   completed-minute barriers, opening gap handling, T+24h opening deadline and
   coincident funding before exit. Recheck contextual location at actual fill.
3. Test capped and uncapped quantity, total fees on entry notional, price-distance
   target, cost-inclusive R, zero/adverse funding including entry < settlement <= exit.
   Example P=100, D=90, c=.0012 gives q=100/10.12 and normal stop net=-100 before funding.
4. Replay candidates chronologically. Admitted plans reserve at T; watch/reject do
   not. A candidate is busy if its T precedes the existing release clock; at equal
   clocks process prior exit first. Unknown admitted paths poison later eligible
   capacity; a skipped candidate never returns. Keep subtype and arm books separate.
5. Test split/restart equals continuous, future append cannot rewrite a completed
   earlier case, explicit source gaps/right censoring, no stop-to-new-scenario rearm,
   known abstention=0 versus unresolved=null. MTM uses opening/close marks, fees and
   funding with an initial zero equity peak. Run Tasks 1–3 tests; expected green.

### Task 4: Qualify and freeze the complete source population

**Files:** Create `scripts/research/lc_context_source.py`,
`scripts/research/run_lc_context_study.py`, `tests/research/test_lc_context_source.py`.

**Consumes:** Existing 31-month LC input lock plus August source receipt, archive,
N3 helper files and separately qualified parent evidence. **Produces:** Immutable
source-only cases/coverage/receipt and policy/code/input bindings.

1. Write failing provenance tests: changed source bytes, mismatched stream, missing
   monthly coverage, duplicate candidate IDs/times, dropped unavailable candidate,
   month endpoint mismatch, changed policy/code after launch, existing output file.
2. Verify old `results/lc_mechanical_extension_2026_09_22/run_v1/input_lock.json`
   and its source files/cases; reconcile exact IDs independently with raw captures.
   Add all four candidates from `results/lc_room_validation_2026_09_30/run_v2/august_source.json`
   through its source receipt, not the two filtered upside replication cases.
3. Read same-stream archive with bounded column/date predicates. Reconstruct the
   source hourly OHLCV and monthly 30-day-seeded TA-Lib ATR for every candidate,
   retaining native fallback/default provenance. Verify parent constructor helper
   hashes and raw input parity. Both reviewers selected the saved continuous
   December 2, 2023 seed ledger before launch; native candidate ATR stays monthly.
4. Source build reads only predecision slices for case construction and no economics.
   Write case/policy/source seals before allowing the score stage. Coverage includes
   every subtype, scenario, parent absence/unknown and abstention reason.
5. Run task tests plus actual source preflight, then one bounded source-only launch.
   Expected 142+4 reconciled raw IDs, not assumed tradable fills. Audit exact input
   hashes, reconstruction, prefix invariance and source counts before economics.

### Task 5: Implement paired reports and scoring launch guard

**Files:** Create `scripts/research/lc_context_study.py`,
`tests/research/test_lc_context_study.py`; extend the new runner only.

**Consumes:** Qualified sealed cases, reviewed new code, outcome windows and 48
independent books. **Produces:** Comparison, finite verdict and completion receipt.

1. Test all raw subtype IDs occur once in every matching book even when rejected,
   missing parent does not suppress a control, and unresolved geometry remains in
   separate coverage. Unknown economics blocks complete totals/verdict promotion.
2. Test hand-computed PnL/R/calendar sums, MTM drawdown, exposure, frequency, winners
   preserved/missed and losers avoided/introduced against immediate. No-fill net=0
   and verdict inconclusive; negative primary complete net is unsupported.
3. Run `python3 -m pytest -o addopts='' -q tests/research/test_lc_context_study.py`;
   expected red, then implement fixed 32-month paired bootstrap (5000 draws,
   seed20261002, 95% interval), accounting for empty-month draws explicitly.
4. Test every worth_forward_test condition independently: positive primary net,
   positive per-candidate dollar delta/lower CI, >=50 fills and >=12 filled months,
   no negative adverse-funding cost/delay aggregate. No fitted/search branch exists.
5. Score stage validates source/code seals and independent review receipt before
   reading outcomes. Run all focused new/existing regression; expected green.

### Task 6: Run economics, audit and hand off the actual result

**Files:** New ignored `results/lc_context_study_2026_10_02/` run directories,
`docs/knowledge/lc_context_results_2026_10_02.md`, PROJECT.md and MEMORY.md.

1. Request integrated software review of new modules/tests, with Review Focus above;
   quant reviewer checks qualified source counts and execution/accounting witnesses.
   Fix important defects with red-to-green reproducing tests, no outcome-based tuning.
2. Freeze reviewed files and source receipt; launch one bounded economic run. Read
   completion or failure receipt before any retry. Report runtime/output telemetry.
3. Independently recalculate saved fills, quantity, fees, funding, barriers, timing,
   occupancy, calendar totals and paired resamples. At least one saved example per
   admitted scenario and terminal outcome, when available; absence is reported.
4. Run `python3 -m pytest -o addopts='' -q` once for repository compatibility and
   report pre-existing failures by name without unrelated repairs. Focused new and
   dependent suites must be green. Verify protected source bytes and git diff check.
5. Write the results with plain-language verdict per subtype, comparison table,
   what was actually tested and what remains unmodeled. Update PROJECT/MEMORY with
   verification, no running processes, next justified action, local-only dependencies.
   Do not commit/push/PR or claim profitable consistency without evidence.

## Plan self review

The evidence/controller interface is predecision-only; execution alone consumes
future windows. Source seals are necessary but reconstruction is separately tested.
The controls retain independent admissibility, books and missingness. New observation
latency is applied to both wait policies, and exclusive expiry differs intentionally
from the older executor. Empty/no-fill cases cannot pass the forward screen. The
parent source choice is frozen by the prelaunch ruling, not an outcome-driven
choice. No external authority is required for these local steps.
