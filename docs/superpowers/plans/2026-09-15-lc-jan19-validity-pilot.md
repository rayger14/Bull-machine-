# January 19 LC real-role validity pilot

User explicitly approved the one-case assessor/critic test following the reviewed
published-catalog integration. This is an operational execution plan and frozen
protocol, not a new strategy or profitability experiment.

## Frozen protocol

- Case ID `LCV1`; sole candidate `hourly-lc:2026-01-19T01:00:00+00:00`.
- Selection: next chronological Q1 candidate after the previous four. Existing
  exposure metadata has no registry match, not proof unexposed. Q1 code outcomes
  have been examined; this is exposed-development data, not a holdout.
- Assemble only archive prefix `[2026-01-07 01:00, 2026-01-19 01:00)` UTC: exactly
  17,280 consecutive minute bars. No engine/source rerun or post-decision prices.
- Verify January source SHA `6ba371d2ded2fb152176b6e82ea2bd56f4ba5714e982f27ee04f44efaa0ad51e`, its manifests and archive SHA `5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035` before asserting reconstructed provenance.
- Reuse unchanged `results/lc_context_discrimination_2026_09_15/curriculum_v1/`
  memory snapshot, master brief and review. Snapshot ID
  `299193b77af10d1382bc02d537a8f919b1e2cd9a3fe5557543e4bb5184fb727d`;
  snapshot file SHA `e42238cf2f647f672f833c8f91d04c8bcc390bc8a42b79645eea14b7e08d9fd7`;
  brief SHA `c7d63d6675a0903604a5bd124835acf5cb880cef28530affe2ed7c40b53664fc`;
  review SHA `ec87b84a112057f0cb614e0aa271578d28bc39a142708ada6bcd08adce2cc783`.
- Same strict source builder, context request, published request and saved-job
  APIs from integration commits d2f1b0e/cfc5214. No source-code changes.
- Same enter/wait_5m_high/reject/null choices and registered economics. No fitted
  threshold, gate, instruction change, exit tuning, new archetype or live action.
- Exactly one fresh requested `gpt-6-astra`/high specialist; at most one separate
  fresh requested `gpt-6-astra`/high critic, only after valid specialist contract
  and verified captured input bytes. Actual snapshot/attention/billed tokens are
  unknown unless independently observed. Instructional isolation, not filesystem
  or model-memory isolation, is the enforced experimental boundary.
- Deliver only the appropriate `PublishedContextResearchJob.role_request` through
  4096-byte ASCII chunks using the existing envelope/actual-return validator.
  The existing transport helper requires a top-level plan for arithmetic checks,
  so the private transport packet is exactly `{case_id, plan, request}`: `request`
  is the unchanged job role request and `plan` is an identical copy of the
  validated source plan. The inner request is the sole role contract; the outer
  plan is transport metadata, not a second policy. Validate this exact wrapper
  against the job before delivery. Specialists copy the inner request's visible
  seal, not the envelope packet hash. No shared source/helper is modified.
  Persist each actual inner tool return before exposing its output to the role.
  Each role writes one exact raw JSON answer, without grading/repair loops.
- Roles must not read project handoffs, source files, archives, other cases,
  historical answers/outcomes or network. Supplied source references are inert.
  No role delegation, requests for more data, extra tools, order or answer retry.
- Lock protocol/code/source/request/envelope before role launch. Lock exact
  specialist response and delivery captures before constructing critic request;
  lock critic packet before critic launch. Save and recompute terminal grade on
  restart. Every changed/missing frozen file fails closed.
- Invalid specialist: explicit skipped-critic event and null terminal grade.
  Missing actual output/capture or corrupt source: report failure; no replacement.
  Invalid/incomplete/material critic review ends the pilot with null plan. A
  reviewed reject or uncertain answer need not be an entry to demonstrate a
  usable contract. Never score null as a profitable skip.
- No reveal authorization, post-decision price read or PnL scoring in this pilot.
  Success means a real evidence-backed judgment survives an independent review,
  not a profitable trade or proven semantic correctness beyond that review.

## Execution checklist

- [ ] Prepare a small private runner and literal storage/transport tests using
  existing builders, immutable artifact helpers and published jobs; do not edit
  reusable sources or old experiments. Independently inspect its narrow paths.
- [ ] Assemble/verify source and curriculum; freeze new private run under
  `results/lc_jan19_validity_2026_09_15/run_v1/`; audit chronological packet contents
  before launching roles. Do not invoke an old experiment's prepare method.
- [ ] Run one specialist and preserve exact tool returns/answer. Validate delivery
  and contract; either terminal skip or freeze the critic request.
- [ ] If eligible, run one fresh critic and preserve exact returns/answer. Lock
  terminal grade and verify restart. No role retries even if usability fails.
- [ ] Report actual judgments, supporting/contrary evidence, review findings,
  exact validity/transport counts and limitations. Update PROJECT/MEMORY and
  commit public protocol/report locally; leave private data local and branch
  unchanged. No push/PR/live changes.

## Controller handoff

Data-building path: existing January source candidate and parent ledgers ->
`build_lc_packet` -> `build_context_request` ->
`PublishedContextResearchJob.prepare` -> `role_request('specialist')`.
After exact valid specialist capture, `role_request('reviewer')` is the only
critic packet. Use existing `build_envelope`, `validate_runtime_returns`, and
immutable `Experiment.save/lock/verify_lock` utilities only, never the legacy
experiment's role preparation, grading or outcome methods.

The next economic experiment requires its own preregistered sample/controls and
outcome scoring. This pilot neither silently performs it nor proves agent edge.
