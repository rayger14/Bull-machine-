# LC context checkpoint — September 15, 2026

## Accomplished

Implemented and independently reviewed a facts-only LC context extractor:
[`lc_context_facts.py`](../../scripts/research/lc_context_facts.py).
It separates hourly geometry, independent sweep/reclaim flags, strict-before
parent binding, directional breaks, decision-time state and predecision minute
observations. It does not choose trades, change old gates or read future prices.

The [separate comparison contract](../superpowers/specs/2026-09-15-lc-context-discrimination.md)
is specified and quant reviewed: A immediate LC, B generic five-minute-high wait,
C agent choosing immediate/wait/reject/null using the same execution menu.
Context conditions the agent's decision, not a hidden deterministic gate.
No new room, RSI, fusion or subtype-specific trigger cutoff was fitted.

## Source-only check on the already-selected four cases

| UTC decision, 2026 | Hourly close | Pre-bound 4H lifecycle | Daily before setup → at decision | Last 5m body |
|---|---|---|---|---|
| Jan2 04:00 | Above prior high | Intact | Present → active | Down |
| Jan4 01:00 | Above prior high | Broken up | Present → active | Down |
| Jan5 01:00 | Above prior high | Broken up | Present → active | Up |
| Jan16 16:00 | Below prior low | Broken down | Absent → new active structure | Up |

All four retain native pre-winner long identity and known hourly, 1m and 5m
operands. They are not all the same structural situation. In the last case,
a daily parent forming by decision does not retroactively become a parent that
existed before setup. A predecision five-minute rebound does not prove later
confirmation or profitability. No trade decisions or new outcomes were assessed.

Private source-only output:
`results/lc_context_discrimination_2026_09_15/source_smoke.json`, SHA256
`f462fb6350f3a82cc3fc75af3e0f6067cd1e2c3193e2f273675ed8f57f37fc2e`.
It binds all four original packet hashes and the new extractor hash. The original
archive was not reread or rehashed this turn. Saved packet integrity and caller
reconstruction attestations are not original live receipt authentication.

## Verification and review

Thirty initial literal tests failed because the extractor was absent, then
passed after implementation. A further failing test froze the distinction between
strict-before-bound lifecycle and decision-time state. Added source-stream,
future-tail and sweep-plus-equality checks bring the new tests to34.

Fresh combined suite: **883 passed**,19.61seconds, one existing urllib3/LibreSSL
warning. This includes the unchanged private v1 harness tests; it is software
verification, not strategy validation.

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -o addopts='' -q results/lc_persistent_master_2026_09_14/test_experiment.py tests/research
```

Independent quant/code review approved the bounded facts slice. Follow-up review
accepted the added decision-state fields and the comparison design, conditional
only on replacing abbreviated excluded memory IDs with full IDs. Root matched
those IDs to the saved snapshot and incorporated the correction. No material
review blocker remains for this slice. This does not approve the not-yet-built
new adapter, jobs, role captures or economic results.

## Next concrete deliverable

Implement the separate context-aware request/grader/job adapter and test it with
literal outcomes-free fixtures. Existing v1 instructions, remembered hypothesis
and job grading all enforce the old rejection rule, so changing just a prompt
would not create a valid discrimination test. Preserve that experiment; use a
new namespace, five policy-neutral reviewed memory records and a new reviewed
outcome-free brief. Do not rerun the original source collection or old master.

After adapter review, freeze all four requests, capture four specialists and four
critics, lock every grade, then compare A/B/C. Unknown output must not earn
avoided-loss credit. The Q1 population is exposed development history, not a
pristine holdout. This small pilot checks useful judgment and integration; a
separately frozen chronological validation is still required before edge claims.

No live changes, new market-role calls, newly revealed outcomes, dependencies,
push or PR. Local artifacts remain necessary for same-machine reproduction.
