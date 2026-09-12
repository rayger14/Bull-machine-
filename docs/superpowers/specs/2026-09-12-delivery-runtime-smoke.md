# Runtime delivery and one-case reviewer smoke — frozen v1

Quant-guided bounded follow-up to the reviewed evidence guard. Exactly two fresh
model calls at most: one assessor, then one reviewer after assessor response and
runtime-return lock. Each receives one deliberately synthetic case, no real
symbol/date/outcome, market forecast or old E01–E08 packet. No retries/repairs.

Each role gets one canonical ASCII JSON packet <=4096bytes and one actual bounded
exec read. The packet has six synthetic timeframe observations, a parent/event
ordering fact, explicit non-completion of a larger model, deterministic indicative
economics, exact typed paths, and an unusual final-tail nonce. The assessor returns
ten exact fact answers and their paths, <=1200 canonical bytes. The reviewer gets
only this case, the locked original response and frozen reference, plus one
explicitly labeled synthetic incorrect value/path control (not an altered agent
answer), and reports answer/path errors and whether that control is supported.

Freeze protocol, private harness/tests, existing guard, assessor packet/envelope,
reference and exact capture-call script before assessor invocation. The reviewer
packet follows the pre-frozen recipe, includes the assessor lock hash and is
separately hashed before its invocation. Source-byte and runtime-return artifacts
are distinct. No production, live runner, dependency or vendor API integration.

At the actual functions.exec boundary, await exec_command, persist its entire
returned object before text(result.output), then emit exactly that output. Both
inner and outer output limits are explicitly6000tokens. Only one source read per
role; artifact persistence is part of that invocation and is not a second read.
Capture metadata binds expected role, case and source SHA. Root derives guard
records from captured output only: completed exit0, exact output text, index0,
case/hash/bytes all match, no unresolved process session. Otherwise retain failure
and do not quietly substitute file contents or retry. No false claim that a
source SHA proves delivery. The assessor's lock precedes reviewer construction.

Report separately: inner_transport_valid; outer_renderer_unverified=true;
model_attention_unverified=true. Exact runtime-return agreement and successful
tail/fact recall establish neither full outer-rendering integrity, comprehension
of all content, market skill, teacher authenticity, teaching benefit nor a
delivery reliability rate. Fresh contexts and an explicit single-case input list
are workflow isolation evidence, not a filesystem security boundary or proof of
statistical reviewer independence.

Run deterministic capture-boundary negative tests before calls: missing/changed
output, unexpected newline, failed/incomplete exit, incorrect role/case/hash, and
nonobject capture. Use the existing guard for exact delivery, economics and path
resolution. Lock original responses and runtime objects without substantive edits.
No economic scoring. After the two-call smoke and review, report the smallest next
step; do not silently launch a new historical batch.
