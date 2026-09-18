# Frozen20-case LC assessment runbook

This is a historical assessment, not live trading. Evidence and curriculum stay
unchanged. No outcomes may be inspected until every case has a locked terminal.

Completion check: full lifecycle integration tests passed; runtime_v2 is prepared
with20 jobs and zero attempts. Root also verified a persistent Python tool
session retains the same controller identity across commands, then closed it
without authorization. Software approval is saved in runtime_v2/software_readiness.json.
No controller is currently left running. Fresh budget remains required.

## Runtime and authority

Use `JudgmentExecution` from `scripts/research/lc_judgment_execution.py`, with
preparation directory `results/lc_consolidated_2026_09_15/judgment_v1` and a new
`runtime_v2` directory inside it. Keep ONE controller object alive in ONE Python
process across reservation, external model work, capture and finalization.
Running a separate one-shot Python process for each phase deliberately loses
measured timing continuity. Never reuse a runtime ID to hide a restart.

`prepare()` clones the20 exact published jobs and freezes runtime dependencies;
it does not authorize model calls. `verify()` and `status()` check this state.
Do not edit any code pinned by an existing runtime lock; preserve prior runs.

Only after independent software-readiness approval and a fresh actual observed
balance meeting the1900-credit START floor may the lead call `authorize_start`.
The observation and approval must be real, not the synthetic values in tests.
Subsequent reservations require fresh positive balance observations. Optional
`max_observed_balance_drop` is a user-specified conservative stop condition;
balance decline includes unrelated usage and is not exact campaign billing.
Neither the API nor the controller guarantees an exact per-call credit price.

## Each case

1. Reserve specialist with `reserve_role`. The returned reservation is consumed
   BEFORE external dispatch. Save the returned paths/attempt ID and launch once.
   Maximum three roles active; each has600 seconds. Never retry uncertain dispatch.
2. Use the frozen `role_launch_instructions.md`, exact wrapper/envelope and fresh
   role context. The specialist reads only assigned chunks, records ACTUAL tool
   returns, and writes one raw response. No local grading/repair or other evidence.
3. Pass exact response bytes, actual runtime-return objects and truthful declared
   metadata to `capture_role`. Do not reconstruct transport returns or invent an
   actual model ID. A reused agent identity invalidates transport.
4. `prepare_reviewer` freezes a critic request only for an eligible on-time
   specialist. An invalid specialist yields an explicit skipped review/null grade.
5. If required, reserve one fresh critic, deliver its exact frozen request,
   capture it and call `finalize_grade`. The same runner measures the entire path.
6. Definite dispatch failure uses `finish_failure`, then `finalize_failure`.
   Interrupted uncertain dispatch uses explicit recovery without a second call.
   Timeouts and late responses cannot become valid on-time grades. Never reopen
   a finalized case or mutate the locked batch with a late capture.

## Completion

Every one of20 cases must have a terminal, including missing/invalid/timeout
cases. `lock_terminals()` binds all results. `assert_reveal_allowed()` must pass
before any future-price read or economic scoring. The existing
`lc_campaign_accounting.score_campaign` consumes the locked ledger snapshot and
the separately frozen accounting manifest. Preserve all registered scenarios
and separate broad A/B from filtered matched A/B/C results.

No retry, extra pilot, threshold tuning, replacement candidate, new curriculum,
live orders or deployment is authorized by this runbook. A successful synthetic
software test is not evidence of profitable agent trading.
