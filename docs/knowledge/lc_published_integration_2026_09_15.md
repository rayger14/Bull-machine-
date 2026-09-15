# LC published assessment integration — September 15, 2026

## Purpose

Connect the one-catalog publisher to both assessment and independent review,
then preserve their exact inputs and outputs in restartable research jobs.
The previous run failed because the evidence IDs shown to agents were not the
same IDs accepted by the grader. This work addresses that software boundary;
it does not tune a trading rule or establish that an agent is profitable.

## Current checkpoint

Implementation is committed as `d2f1b0e`, with recovery fix `cfc5214`.
Pre-fix root verification passed **1,028 distinct
tests**: 1,012 public/current-private tests in 118.77s and 16 original-private tests
in 3.72s. The only warning is the existing system LibreSSL/urllib3 compatibility
warning. Independent task review found one recovery defect: direct reviewer
capture could persist a file before rejecting an invalid specialist. Pre-write
validation and regression tests now fix it; scoped re-review approved with no
remaining findings. Post-fix root verification and final integration review
remain pending.

The root source-only integration check passed on all four existing source
requests: **798 IDs** resolve and validate through specialist and critic paths
(203/203/202/190). Material-error critic fixtures always block the plan. Sixteen
temporary jobs cover enter/wait/reject/uncertain through capture, grade, reopen
and reveal ordering, preserving exact original-menu economics and source hashes.
All answers in that check are synthetic; no real model or semantic review was
performed. The completed old experiment's `verify` command also exited 0.

Design: [integration spec](../superpowers/specs/2026-09-15-lc-citation-integration-design.md).
Execution: [implementation plan](../superpowers/plans/2026-09-15-lc-citation-integration.md).
Previous completed results remain in the [LC comparison report](lc_context_adapter_2026_09_15.md).

## Concrete deliverables

1. **One model-facing evidence contract.** A deterministic role request exposes
   one accepted catalog and one current instruction/schema. The original source
   request stays controller-side; it is not accidentally delivered as another
   competing authority. The response copies the visible request seal, not a hash
   the model would have to calculate.
2. **Connected specialist and critic validation.** Both fine evidence IDs and
   larger context-group IDs pass through the same catalog. Unknown IDs are
   contract failures. A real ID with an unsupported claim still needs a material
   critic finding; ID existence alone is not factual or semantic approval.
3. **Saved jobs that recheck their decisions.** Reopen validates the source,
   derived role request, exact captured strings, computed grade and reveal
   binding. Invalid specialists can finish with an explicit critic-not-invoked
   event. Missing critic, failed review and insufficient evidence never become
   a scored profitable rejection.

Code entry points:

- `scripts/research/lc_published_assessment.py`: `build_published_request`,
  `validate_published_request`, specialist/reviewer grading and research-plan gate.
- `scripts/research/lc_published_jobs.py`: `PublishedContextResearchJob.prepare`,
  `role_request`, `capture`, `skip_review`, `lock_grade`, `authorize_reveal`.
- `tests/research/test_lc_published_assessment.py` and
  `test_lc_published_jobs.py`: 62 new tests, including explicit unknown-ID errors,
  raw-string binding, truthful skipped-review provenance and forged-grade refusal.

The job bundle is controller-only. Deliver only `role_request('specialist')` or
`role_request('reviewer')`; do not deliver the bundle, project MEMORY or outcomes.
No API here invokes a model or places an order. Caller-declared transport is not
authenticated model identity, delivery or attention.

The trading menu and economics are unchanged: immediate entry, fixed five-minute
high confirmation, reject or insufficient evidence. No fusion, parent-intactness,
room, RSI, stop or exit tuning has been introduced.

## What happens next

Independent quant review recommends **one new specialist/critic pair first**,
with a fixed stop rule. The first chronological candidate after the prior four
is `hourly-lc:2026-01-19T01:00:00+00:00`. Existing frozen exposure metadata records
no match in its prior-assessment registries for this candidate. That is not proof
of no exposure, and full Q1 code outcomes have already been examined.

The concrete next experiment is an **exposed-development contract-validity
pilot**, not a holdout or profitability test:

- Verify the saved January source and manifests, then assemble this one candidate
  using its existing parent ledgers and only the archive prefix
  `[2026-01-07 01:00, 2026-01-19 01:00)` UTC. No engine rerun is required.
- Reuse the unchanged reviewed five-record curriculum and brief. Freeze the new
  source packet, role request, code hashes, exposure record, delivery envelopes
  and model/role limit under a new case ID and directory.
- Invoke one specialist; invoke one independent critic only if delivery and the
  specialist contract pass. Preserve every raw response. No repair, retry or
  replacement case after failure.
- Report whether the complete real-role path produced a valid reviewed judgment.
  A reviewed reject or uncertain answer is not a failure merely because it does
  not enter. Any invalid/incomplete/material-error result ends this pilot.

No new role packet, role invocation, outcome comparison or execution permission
is created merely by documenting this next experiment. After a usable real-role
path is demonstrated, preregister the economic comparison before outcome scoring.
Profitability requires broader chronological validation, not a one-case success.

## Design decisions made under delegated quant review

- Use an explicit skipped-critic event in the new subclass. This avoids confusing
  a specialist contract failure with an actual failed reviewer invocation. If
  this event design proves wrong, the new storage interface needs rework.
- Bind responses to the visible request-body seal. All evidence and instructions
  remain covered, without requiring an LLM to calculate SHA256. If this binding
  is misunderstood, the new response interface needs revision.
- A skipped critic has no delivery-capture hash; only its empty storage placeholder
  is hashed. If consumers conflate those fields, their audit interpretation needs
  adjustment. No model invocation is being claimed.

None of these decisions changes a frozen historical answer or a live trading rule.

## Reproduction boundary

The reusable modules/tests/design ship in git; source data and private verification
artifacts remain local. A clone alone does not reproduce the market data.
The local `.superpowers/sdd/2026-09-15-lc-citation-integration/` directory contains
the implementation report, review package, ledger and source-only check.
Source-check script SHA256:
`374fa59209dbf541d56ccdabc096f06ef6bc38cc2458bf1677d73f15037b1858`.
Recorded result SHA256:
`1070fbf74d25651318807bb3a784e1a7a98e294d1a8dc5f083d5baa0f74c8482`.

No new market-role run, new PnL result, training, dependency installation,
production/config/fusion change, push or PR occurred in this integration work.
