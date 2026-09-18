# Task 2 — campaign contract report

## Delivered

`scripts/research/lc_campaign_contract.py` provides an offline-only immutable
campaign ledger. It makes no model, network, source-replay, market or outcome
call. The focused test suite is
`tests/research/test_lc_campaign_contract.py`.

Verification: `env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -o
addopts='' -q tests/research/test_lc_campaign_contract.py` passed **11 tests in
9.66 seconds**. `git diff --check` passed.

## Task 4 interface and exact schemas

The controller first selects a frozen ordered roster using
`select_block(candidate_ids, excluded, cap=30)`. It returns the first complete
contiguous unexcluded run of `cap`; without one, it returns the longest run,
with chronology resolving ties. It never deletes excluded interior IDs.

Construct `CampaignLedger(run_dir)` (the optional `clock` and `job_loader`
arguments are test-only injection points). Before dispatch call `freeze` with
exactly:

```python
{
  "schema_version": "lc_campaign_contract_v1",
  "campaign_id": "nonempty string",
  "cases": [{
    "case_id": "nonempty string",
    "job_directory": "published job directory",
    "source_request_sha256": "64 lowercase hex",
    "role_request_sha256": "64 lowercase hex",
  }],  # 0..30, unique case_id
}
```

The default loader reopens `PublishedContextResearchJob`, validates its existing
hash chain and requires its request bundle hashes to equal the manifest pins.
This is the actual-source/grade eligibility path; there is no `eligible` flag.

Immediately before an external dispatch call, invoke
`start_attempt(case_id, "specialist" | "reviewer")`. It atomically reserves the
role slot and persists a 600-second deadline. There are at most 30 attempts per
role and three active attempts globally. An already invoked role is always
rejected after reopen. Reviewer invocation also asks the published job whether
its specialist permits a review, so invalid/missing specialists cannot route to
a reviewer.

After delivery, call `finish_attempt` with exactly one of:

```python
{"kind": "delivered", "job_directory": str,
 "raw_response_sha256": str, "capture_sha256": str}
{"kind": "external_failure", "reason": "nonempty string"}
```

Delivered bindings must equal the immutable published job capture artifacts.
Past-deadline arrivals are recorded as immutable `late_deliveries`; the role
remains a deadline timeout. Missing or failed attempts do not create captures.
Uninvoked roles are explicitly stored as `not_invoked` when terminals lock.

After all cases have a terminal, call `lock_terminals` with a mapping whose keys
are exactly the frozen case IDs. Each terminal is either:

```python
{"kind": "published_grade", "job_directory": str, "grade_sha256": str}
{"kind": "external_failure", "reason": "nonempty string"}
```

A published grade is re-opened/recomputed from `PublishedContextResearchJob`
before storage; state records its recomputed status and plan. An external
failure stays a controller terminal with no invented judgment plan. Only after
that successful lock may Task 4 call `assert_reveal_allowed()`. The empty roster
requires `lock_terminals({})`, representing `no_eligible_cohort`; it never
permits a role attempt.

`ledger.json` is canonical JSON with a SHA-256 over its complete state. All
operations take an advisory exclusive file lock, revalidate the pinned manifest
and published job bindings, and write an fsynced atomic replacement. Unequal
manifest, role-result, terminal, or duplicate-invocation replay is rejected.

## Boundary notes

The existing `PublishedContextResearchJob` stage model was compatible: its
request, capture and grade stages are already immutable and recomputable. No
workaround or stage-model change was needed. Task 3 should consume the stored
terminal’s distinct published-grade versus controller-failure kind rather than
turning controller failure into an agent grade.

## Timing-contract addendum

Commits `5d53fa8` and `b409d7f` add one exact, state-hash-covered
`decision_path` to every case. A specialist start seals wall and monotonic
nanoseconds plus a fresh runtime ID in the same locked write that reserves the
attempt. `finalize_case(case_id, terminal)` recomputes the terminal before
sealing the end observation and binds its canonical hash. `lock_terminals`
requires every supplied terminal to match a finalized path.

For Task 3, consume the full object returned by `CampaignLedger.state()`: the
terminal map is under `terminals`, and each case's timing record is under
`cases[case_id].decision_path`. A measured processing delay is available only
when `timing_valid is True`; convert `elapsed_ns` with
`ceil_elapsed_seconds`. When timing is invalid, S4/S5 are null while S0-S3 and
the original grade remain unchanged. A reopened controller has a fresh runtime
ID, so an unfinished cross-process path finalizes as
`timer_runtime_changed` rather than using wall time.
