# Runtime delivery and one-case reviewer smoke — September 12, 2026

The bounded synthetic smoke passed. Two fresh role invocations each captured an
actual inner `exec_command` return before emitting its output. Both captured
outputs exactly matched their separately frozen source packets. The assessor
returned all ten requested facts and all eleven typed citation paths correctly;
the fresh reviewer matched the reference and rejected a labeled incorrect value.
This is delivery/routing evidence, **not market-selection or profitability evidence**.

## What changed

The existing pure guard at `scripts/research/assessment_evidence_guard.py` is
unchanged. A private study harness now exercises it against actual inner tool
returns, rather than reconstructing delivery records from source files. The
[pre-call protocol](../superpowers/specs/2026-09-12-delivery-runtime-smoke.md)
was committed locally as `504f39a` before either smoke invocation.

Private artifacts are under `results/delivery_runtime_smoke_2026_09_12/` and remain
ignored/local, not included in Git or GitHub. They include the harness and tests,
frozen packets/envelopes/call scripts/reference, original response JSONs, captured
runtime objects, locks, configuration and final result. Another checkout can read
this protocol/report and the reusable guard, but cannot reproduce this particular
run without those private artifacts. No production gates, live engine, dependency,
vendor API client, remote branch or PR was changed.

## Frozen experiment and observed result

One deliberately synthetic case, `SYNTHETIC_DELIVERY_01`, contained completed
1m/5m/15m/1h/4h/daily observations, parent-before-sweep timing, an explicit false
full-model-completion flag, indicative economics and a final-tail nonce. No real
symbol, market date, old study packet or future market outcome was supplied.

- Assessor packet: 1,763 canonical ASCII bytes, one chunk. Ten exact fact answers
  correct; ten citation entries contain eleven correct typed paths. Its original
  response was locked before constructing the reviewer packet.
- Reviewer packet: 3,092 canonical ASCII bytes, one chunk. It contained only the
  same case, locked original assessment, expected answers and an explicitly
  labeled incorrect-value control: claimed 1m close 101 versus observed 100.
  The reviewer returned no answer/path errors and `control_supported=false`.
- Each role was instructed to make one bounded source-read invocation, which
  also persisted the full inner return before emitting exactly `result.output`.
  Both output limits were 6,000 tokens. Both captured reads exited with integer
  zero, no unfinished session, and exact source text. Root derived validation
  records from these returned objects, not from a substitute source-file read.
- Fresh contexts: `/root/runtime_smoke_assessor` and
  `/root/runtime_smoke_reviewer`, each requested with `fork_turns=none`,
  `gpt-5.6-sol`, medium reasoning. Configuration was recorded before invocation.
  Actual model snapshot and billed tokens remain unknown. Runtime output-token
  estimates are not billed usage. Exactly two smoke role invocations; quant
  design/final review was separate. No retry or response repair occurred.

Responses were preserved as returned JSON text, with only a final file newline
added. Original responses/runtime objects were hash-locked before grading or
downstream review; repeat `finish` verified immutable artifacts and results.

## Pre-call correction and verification

Quant review caught a mismatch: the protocol required exact fact values, but the
private grader accepted differences below `1e-10`. A regression test first failed
on `100.00000000005`; bool-excluding numeric equality then passed, accepting
numerically equal `100`/`100.0` but rejecting the deviation and booleans. The
original pre-review manifest was preserved and a new manifest frozen before calls.

Ten private tests pass, covering exact output and missing/altered/newline output,
failed or boolean exit codes, unfinished sessions, wrong role/case/hash bindings,
nonobject captures and exact numeric grading. Root also reported twenty additional
single-field answer/path negative controls in pre-call tool output; these do not
have a separate persisted test/result artifact and were not independently
reproduced by the post-run reviewer. The
fresh full research suite passed all 455 tests, with the existing LibreSSL warning.
No market backtest was run in this smoke.

An alternate relative-path import invocation of `freeze` was rejected by the
immutable manifest check because its path key differed. It changed no frozen
artifact; the registered direct CLI invocation repeated with identical hashes.
The private harness is a study tool, not a generalized production runner.

Independent post-run quant review verified both captured outputs, the frozen
manifest files, response/reference/path checks, same-case reviewer input, control
rejection, lock chains and result hash. It approved the narrow result and flagged
the multi-chunk gap below. This final review was not another smoke role invocation.

## Receipts

| Artifact | SHA-256 |
|---|---|
| Preserved pre-review manifest | `0a0d5eb0f62efe9aef10601916678f3b15e07050628aaf272d396fb67844e97c` |
| Final pre-call manifest | `194b9601789ddad4b24fec8d282effde8a253ed63496345fb29edfda25f0edef` |
| Assessor canonical packet | `db2124f5b106721998b184decc135b0a090abd9bf628d81dd949aa2874ac70e5` |
| Assessor lock | `34f9bbff49fd569206866d10cd1e947877e245e292f46de1fd8dc01554510c83` |
| Reviewer canonical packet | `780d2b7039f6003f984a1fadf76f583813035a81ad5e864c293437559698df2b` |
| Reviewer lock | `b8394b535b62a374336e76f7fcaa09d180fcb0a23a0a6a85c1b800b5fb2fd78e` |
| Results | `ebd612457274b019c8c637c5b118486091432d74b4e3a99a3415cab3c6000b84` |
| Requested run configuration | `357e122bcae0114546ff4284d9207bc32b4aa81ffef7767f93efae294668f702` |

## Limits and next step

The supported boundary is **captured inner tool return equals frozen source
chunk**. Outer rendering and model attention remain unverified. Successful fact
recall does not prove comprehension of every input, source authenticity, trader
teaching fidelity, teaching benefit, forecast skill or a delivery reliability
rate. Separate fresh contexts are workflow isolation, not filesystem enforcement
or statistically independent model judgments. Reference-assisted review is
deliberately scaffolded; the planted error was labeled, not a hidden challenge.

The immediate next step recommended by the quant reviewer is to preregister one
fresh **two-chunk, assessor-only synthetic transport test**: an ordering fact,
boundary/tail fact, and exact per-chunk runtime captures. Do not repeat the reviewer
routing smoke. This study covered one chunk per role; earlier historical packets
needed six or seven, so multi-chunk delivery remains untested at the actual runtime
boundary. That next test was neither registered nor launched in this turn.

After that boundary check, return to the substantive question: whether an agent
improves real setup selection beyond same-evidence mechanical rules. A separately
reviewed historical test must use newly frozen
cases, preserve equal hourly/minute standing, completed multi-timeframe evidence,
source-bounded delivery, per-case review and assessment/review locks before
outcomes. No additional historical batch was launched in this smoke.
