# Two-chunk runtime smoke — September 12, 2026

**Delivery passed; the complete smoke failed on citation formatting.** Both actual
captured inner tool returns matched the frozen chunks exactly and the fresh agent
returned all three factual answers correctly. Its citation object did not match
the frozen contract. The original failure is retained, not repaired or retried.

## Implemented and tested

Local commit `ce9fbc0` adds `validate_runtime_returns` to
`scripts/research/assessment_evidence_guard.py`, with 34 new regression tests and
the [pre-call protocol](../superpowers/specs/2026-09-12-two-chunk-runtime-smoke.md).
The helper validates supplied captures in original order, binding role/case/hash/
index, requiring integer-zero exit and no active session, and passing captured
output to the existing exact envelope/count/order/text guard. It never sorts,
deduplicates, reconstructs output from files or authorizes execution.

The initial 29 new cases failed before implementation, then passed. Quant review
recommended five additional regression cases for stale/tampered inputs and missing
fields. All 73 focused tests and all 489 research tests passed; the full suite took
12.42 seconds with the existing LibreSSL warning. Four private harness tests also
passed after an observed failing-first run. No live engine/gate/threshold changes.

## Frozen agent result

One fresh `/root/two_chunk_assessor`, requested `gpt-5.6-sol`/medium with
`fork_turns=none`, read two chunks sequentially. Actual snapshot/billed usage is
unknown. Each invocation persisted the entire inner return before emitting output.
The 4,234-byte canonical synthetic packet split into 4,096 and 138 bytes. The
marker began at byte4090, split as `opal-r` / `idge-62`; ordering operands appeared
in the first chunk and the tail nonce in the second. No real prices, dates,
forecasts, old cases or reference answers were supplied. No reviewer-role model
invocation or retry occurred; quant audit was separate.

The agent correctly returned parent-before-sweep=true, `opal-ridge-62` and
`silver-otter-83`. It also returned an extra `case_id` citation and flat/dotted
paths instead of lists of typed paths. For example, it used
`["a_event.parent_tick","a_event.sweep_tick"]` instead of
`[["a_event","parent_tick"],["a_event","sweep_tick"]]`.

Both raw runtime returns and the original response were locked before validation
and grading. Result: transport valid, `answer_errors=["citations"]`,
`smoke_pass=false`. That field is the harness's whole-section error label: the
three factual values were correct, not three failed factual answers. The saved
response preserves the original JSON text with only a final file newline added.
Repeating grading yielded identical immutable results.

## Diagnosis and future correction

The delivered instruction said "each answer key" and "list of exact typed paths,"
but did not enumerate the three allowed citation keys, explicitly exclude
`case_id`, or demonstrate nested-array syntax. The private reference required
those details. This is a prompt/grader contract gap; it does not establish the
cause of all model mistakes or demonstrate inability to read market structure.

For the next separately frozen packet, make the input contract explicit:

- Supply the exact top-level response-key allowlist as well as field types.
- Supply the exact required citation-key allowlist. State that metadata such as
  `case_id` is excluded; missing and additional keys are rejected.
- Define each citation value as a list of nonempty typed paths. A path is an array
  of separate string object keys and nonnegative integer list indices (booleans
  are not integer indices). A neutral
  shape example is `{"claim_a":[["section","rows",0,"value"]]}`. This example
  illustrates syntax only, not evidence or a correct claim.
- Explicitly forbid dotted-string paths, flat lists of path components where a
  list of paths is required, and extra explanation outside the response schema.
- Supply field types and unknown/abstention representation consistently with the
  registered grader. Do not provide expected fact values, acceptance labels,
  reference citations or future outcomes. Validate the actual response strictly;
  do not silently coerce or repair it.

This documents the future input-contract correction; no agent has tested its
behavioral effectiveness yet. The frozen prompt/reference/response/results here
are unchanged. Independent quant audit reproduced the transport/fact pass,
citation failure and result hash, and approved this prospective correction only.
No additional synthetic scale-up is planned. The next substantive
step is a separately registered two-case historical integration—one hourly and
one minute—with this explicit contract, identical baseline evidence and per-case
assessment/review locks before outcomes. That experiment has not been launched.

## Receipts and limits

Private harness, chunks, captures, response and locks remain ignored/local under
`results/two_chunk_runtime_smoke_2026_09_12/`, not GitHub. Public helper/tests and
protocol are committed locally. No remote push or PR change occurred.

- Final pre-call manifest: `5a0d410b8f1c93a9c990fa8aec714b505ebd15838f8ae2caab9cc83112089a99`.
- Canonical packet: `97182218cfd3097c0dc206fbc61ef0324576d0b546ca7b025bc961389c9e5b48`.
- Response/runtime/manifest lock: `d597c3331e380ba2362fbb5d6719bcd26d4fbb9a5d4c6b8997f30742e3c75d26`.
- Result: `79698c7da84c37e313c26bf499f518eb4a7b0a4ad1855750ae828375c91de48a`.

The pure validator cannot authenticate who produced a capture; integration
instrumentation supplies that provenance. Outer rendering and model attention
remain unverified. Fresh context is workflow separation, not enforced filesystem
isolation. This is one two-chunk sequence with a small second chunk, not a
delivery reliability estimate, full-size historical workload certification,
market-selection result or profitability evidence. Older frozen manifests bind
their original source revisions; the new additive helper is a new code revision,
not permission to overwrite or re-freeze prior study artifacts.
