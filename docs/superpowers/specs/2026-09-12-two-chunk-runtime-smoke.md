# Two-chunk runtime smoke — pre-call protocol

Bounded follow-up to the single-chunk smoke; one fresh assessor invocation only,
no reviewer role, retries, market prices, real dates, old cases or forecasts.
No live change, vendor client, remote push or dependency installation.

Use existing canonical envelope with 4096-byte chunks. One synthetic ASCII JSON
packet must exceed4096 and be at most8192 bytes, producing exactly two chunks.
An ordering fact is in chunk0, a literal marker straddles the4096 boundary and a
tail nonce is in chunk1. Return exact case_id, parent_before_sweep, boundary_marker,
tail_nonce and citations (three entries with exact typed paths), JSON <=600bytes.
The independent reference is not supplied to the assessor.

Freeze this protocol, harness/tests, research guard/tests, packet/envelope,
reference, requested model configuration and both exact call scripts before the
invocation. Requested model gpt-5.6-sol, medium, fresh context fork_turns=none;
actual snapshot and billed tokens unknown. Two sequential functions.exec calls
in that context: each invokes exactly one exec_command source read, persists its
entire returned object before text(result.output), inner/outer limits6000tokens.
Binding: case_id, role, packet SHA and chunk_index. No other source reads or tools.
If a read fails, preserve failure and stop; no retry or substitute file contents.

Root locks both runtime-return artifacts and original response before grading.
validate_runtime_returns consumes captures in original order, requires complete
integer-zero exits/no active session and exact bindings, then delegates envelope,
count/order/text validation to validate_delivery. Do not sort or deduplicate.
Separately grade every answer and citation against the frozen literal reference;
preserve any failed/malformed answer without substantive repair. Repeat grading
must preserve result hashes. Independent quant review checks the artifacts.

Run focused negative regression tests before the call, then full research tests
before handoff. Failure modes include missing/extra/swapped/duplicate/changed
chunks, wrong bindings, failed/incomplete returns and malformed records. Returned
authority is captured_inner_runtime_returns_only; outer rendering, attention and
capture authenticity are not certified by the pure validator. Fresh context is
workflow isolation, not filesystem enforcement. The observed call demonstrates
one delivery sequence, not a reliability rate or market skill.

After this bounded test, no further synthetic scale-up stages are planned. Next
is a separately frozen and reviewed two-case historical integration, one hourly
and one minute, with assessment/review locked before outcomes; not launched by
this protocol. Failed future packet delivery remains fail-closed.
