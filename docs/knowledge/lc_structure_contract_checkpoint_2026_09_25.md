# LC structure contract implementation checkpoint

Updated September 29, 2026; filename follows the September25 plan.

## What is implemented

The [approved plan](../superpowers/plans/2026-09-25-lc-structure-contract.md)
now has three separate offline modules. No existing engine or frozen research
module was edited.

1. `scripts/research/lc_structure_packet.py` converts a validated saved source
   request into a timestamped level/candle catalog. Raw extrema are not promoted
   to confirmed pivots; absent, broken and unknown parents remain distinct.
2. `scripts/research/lc_structure_proposal.py` checks the proposed entry, stop,
   invalidation, destination and contrary evidence against source and policy
   bindings. Invalid output is not silently treated as a profitable rejection.
3. `scripts/research/lc_structure_preentry.py` checks a supplied hypothetical
   entry against delay, first confirmation, expiry, pre-entry invalidation,
   coverage, tick rounding, costs and external risk/price constraints.

Every result remains `execution_authorized: false`. This is not a live service,
new agent trader, exit simulator, or profitability result. The quantity output
is a modeled upper bound, not an executable size or guaranteed maximum loss.

## Verification before independent review

- Task1:18 new packet tests;94 with its legacy regressions, all passed.
- Task2:53 new proposal tests;147 with packet/legacy regressions, all passed.
- Task3:34 new pre-entry tests, all passed.
- Combined focused verification:254 passed in49.64s (105 new,149 existing).
- Entire `tests/research`:1,259 passed, one existing urllib3/LibreSSL warning,
  in1,028.77s. This larger run explains much of the elapsed verification time.
- Bare repository `python3 -m pytest`: collection aborted before a full run.
  Unchanged `tests/test_integration_fixes.py:27` opens missing
  `configs/baseline_wyckoff_test.json`, then calls `sys.exit(1)` at line46.
  Pytest reports five collection errors but its fatal internal traceback does
  not enumerate all five; no whole-repository green claim is made.
- Each module's initial tests failed for its missing API before implementation.
  Two initial nonfinite-policy tests tried to hash NaN/infinity in their own
  setup; existing canonical hashing correctly refused. Tests were corrected
  to submit those invalid policies directly, and the full focused run passed.

Tests use public synthetic inputs, not private market data or model responses.
Source authenticity remains caller-verified historical reconstruction, not
independent archive verification or authenticated live receipt times.

Reproduce the bounded verification with:

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -o addopts='' -q tests/research/test_lc_structure_packet.py tests/research/test_lc_structure_proposal.py tests/research/test_lc_structure_preentry.py tests/research/test_lc_context_assessment.py tests/research/test_lc_published_assessment.py tests/research/test_lc_context_facts.py tests/research/test_conditional_entry.py tests/research/test_lc_single_assessment.py
```

## Concrete synthetic demonstration

At a hypothetical105 entry, stop99 and destination110, the pre-entry checker
reports risk6 and reward5 per unit, before costs; all permission flags stay false.
A gap to107 cancels the same proposal against its fixed106 entry cap. A separate
wait fixture must observe its first strict post-arm close above110 before its
111 hypothetical fill can qualify for the120 destination. These invented prices
demonstrate behavior, not a winning trade or a recommended BTC policy.

## Review and remaining work

One independent software review of `f4ff8e1..3804acf` completed. It independently
passed105 new tests and found two issues, both addressed in one root fix pass:

- Important: arbitrary extension fields on current candles, parent bounds and
  parent updates survived the source projection. Nested dictionaries now use
  explicit allowed fields, preventing these extension fields from entering the
  outcome-hidden view. This is not automatic redaction of prose in intentionally
  supplied teaching/limitation records; those remain a curated source boundary.
- Oversized integer policies could raise an uncaught conversion exception. Root
  graded this Important despite the reviewer's Minor label because invalid input
  must not crash the controller. Numeric checks now reject conversion overflow.

Six additional checks cover those paths and the existing fill-input guard. A
diagnostic loading the pre-fix module in memory produced five expected failures
and one already-passing fill guard. The revised new-module suite passed111 tests
in5.06s. The first failing leak test had slow pytest string-diff rendering and was
interrupted after showing the failure; boolean assertions now avoid that cost.
Final combined verification **passed260 tests in49.23s**:111 new and149 legacy.
The broader1,259-test research run above preceded hardening; it was not rerun
after these fixes. Bare repository pytest was rerun and again aborted during
unchanged integration collection (five reported errors,9.14s).
Scope acceptance: offline contract and pre-entry checker verified; no whole-repo
green, merge, economic or live-readiness certification. No reviewer was dispatched
a second time; root verified the fixes as the approved one-review workflow requires.
Commits so far: `d13f914` packet, `7a8de0b` proposal, `3804acf` pre-entry checks.
No push/PR, market assessor calls, live orders, threshold changes or new outcomes.

After final acceptance, the next separate deliverable is a structural-target
exit replay adapter and a frozen numerical comparison protocol. It must compare
code and agent under the same risk/execution policy before attributing gains to
judgment. All142 prior cases remain exposed development data. A further paid
evaluation needs its own approved scope/budget and eligible unseen/prospective
data; this contract does not authorize it.

## Rulings during execution

- Retained the existing quant branch per the user's explicit preference.
- Continued scoped work despite unrelated whole-repository collection failure;
  cost: verification is research-suite coverage, not a whole-repository green.
- Retained the ignored execution ledger for continuity rather than deleting
  verification evidence; cost: a small local artifact. Durable results live here.
- Upgraded the oversized-number finding and fixed it; cost: a small extra
  regression/fix rather than a deferred controller exception. No deferred minors.
- Declined trading/policy and semantic confirmation judgments remain separate
  evaluations; cost: this work establishes no economic edge.
- Archive authentication and cross-representation source consistency remain
  caller-owned; cost: reconstruction needs separate verification.
- Exits, gap losses, executable sizing and capital accounting remain outside
  this contract; cost: no full backtest or live-readiness claim.
- Stale handoff state is corrected here; unrelated collection failure
  stays outside scope and prevents an integration/whole-repository-green claim.

Local execution logs/ledger: `.superpowers/sdd/2026-09-25-lc-structure-contract/`.
No private archive is required for these tests. Historical research still needs
the private dependencies documented in PROJECT.md; GitHub alone lacks them.
