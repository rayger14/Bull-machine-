# LC practice suite — implementation checkpoint

September 29, 2026. **Software implemented and real inputs prepared. The new
12-case market-agent experiment has NOT run.** User authorized direct
implementation, retaining one software review and separate market-call approval.
No production orders, configurations, fusion logic or existing archetypes changed.

## What is now usable

`python3 -m scripts.research.lc_practice` provides preparation, status, a persistent
capture-owner bridge, terminal locking, scoring and report commands. It does not
silently dispatch models. See the [operator guide](lc_practice_operator_2026_09_29.md).

The five new modules connect the existing structure packet/proposal/pre-entry/
exit contracts. The workflow verifies source candles, freezes each request and
mechanical control, captures one original response per fresh role with measured
latency, then prevents outcome scoring until every case has an immutable terminal.
Failures remain unavailable, not successful rejections. Restarts cannot retry
in-flight attempts. Same-timing mechanical comparisons use the same risk and costs.

The generated report includes all six predecision timeframe panels, a separately
labeled future-minute path, original rationale/citations, entry/stop/target and
outcome records. Rebound and expansion have separate coverage, matched deltas
and winner/loser attribution. HTML embeds charts and needs no server; JSON and
Markdown come from the same locked records. Synthetic end-to-end runs exercised
this report; there is no real-case outcome report yet.

## Verification and independent review

- Final post-fix focused suite: **368 passed in 78.71 seconds**, including 43 new
  practice tests and 325 existing regressions. Fourteen installed matplotlib/
  pyparsing deprecation warnings; no test failures.
- One independent software reviewer inspected the integrated path and ran its
  original 40 tests. Verdict: ready with fixes; no Critical findings.
- Important finding: nullable future prices could crash the whole batch. Root
  reproduced it, preserved bad cells as unknown, and verified missing data before
  an exit makes that case unavailable while missing data after a known exit does
  not erase the resolved result. Both regressions and report rendering pass.
- Important finding: subtype summaries lacked matched coverage and attribution.
  Root added a failing winner/loser/missing-data example, implemented separate
  subtype accounting and visible scorecards, and verified it in the final suite.
- Root also caught and fixed Markdown table spacing with a failing integration
  assertion. The final suite includes that assertion and the HTML scorecards.
- A broader `tests/research` run started before those fixes was deliberately
  interrupted after 478.44 seconds, with 665 tests passed so far. It is **not** a
  completed full-suite verification and is not the final post-fix count.
- Bare repository pytest still aborts during collection in unchanged
  `tests/test_integration_fixes.py` because
  `configs/baseline_wyckoff_test.json` is missing. No repository-wide green claim.
- Synthetic chart inspected visually. Source preparation emitted sandboxed Arrow
  CPU-discovery warnings but completed all parquet reads and candle checks.

The reviewer did not certify profitability, trader-thesis correctness, provider
attestation/billing, host role isolation or funded portfolio execution. Those are
not established by software tests. No further review loop was launched.

Reproduce the final focused check (no private files/model calls required):

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -o addopts='' -q \
  tests/research/test_lc_practice_sources.py \
  tests/research/test_lc_practice_runtime.py \
  tests/research/test_lc_practice_replay.py \
  tests/research/test_lc_practice_integration.py \
  tests/research/test_lc_structure_outcome.py \
  tests/research/test_lc_structure_packet.py \
  tests/research/test_lc_structure_proposal.py \
  tests/research/test_lc_structure_preentry.py \
  tests/research/test_lc_context_assessment.py \
  tests/research/test_lc_published_assessment.py \
  tests/research/test_lc_context_facts.py \
  tests/research/test_conditional_entry.py \
  tests/research/test_lc_single_assessment.py \
  tests/research/test_entry_case_outcome.py --tb=short
```

## Real run prepared without outcome scoring

Local namespace: `results/lc_practice_2026_09_29/run_v1`.
Manifest SHA256:
`1d51c49b68732f821160a2fd22a39bf9f2559826f47d62e5210fb5e75e53a86c`.

First 12 of the original 20-case source roster, January 20–May 3, 2026:
9 downside rebound and 3 upside expansion. All 12 source checks passed,
including 1,320 supplied candles compared with their complete minute intervals.
Original source lock and archive hash reverified. The frozen mechanical rule
produced 10 proposals and 2 no-setups (January 20/25, stop geometry); this is a
source-only rule result, not a count of filled or winning trades. Agent candidates
remain all 12; no cases replaced and no geometry tuned to force participation.

| Decision UTC (2026) | Full request bytes |
|---|---:|
| January 20 06:00 | 156,054 |
| January 25 09:00 | 156,186 |
| January 29 16:00 | 155,907 |
| January 31 15:00 | 154,025 |
| February 23 02:00 | 156,204 |
| February 25 02:00 | 151,836 |
| February 28 07:00 | 153,859 |
| March 7 20:00 | 156,148 |
| March 8 23:00 | 156,227 |
| March 22 22:00 | 153,882 |
| May 2 22:00 | 156,125 |
| May 3 23:00 | 154,150 |

Total 1,860,603 canonical ASCII bytes before host wrapper. Byte counts are not
billed tokens or credits. No current billing/balance tool is exposed; exact
charges, available balance and a hard credit cap remain unknown.

Prepared status: authorization false, 12 pending, zero reservations/terminals,
no terminal lock or `case_results.json`. All processes used for this checkpoint
have stopped. There are no background assessments waiting to finish.

## Next action and limits

Request explicit approval tied to this manifest for **up to 12 Astra/high
assessor calls**, one per case, no critics/retries/replacements, max one in flight,
600 seconds per attempt. This limits calls, not dollar/credit spend. Once approved,
use the persistent owner and fresh packet-only roles, lock all terminals, then
score and deliver the real 12-case report. Do not start another planning phase,
reuse old answers, silently shrink packets or reveal outcomes early.

This is exposed practice, not a held-out estimate of an edge. It excludes raw
funding/OI/fusion/Fibonacci inputs, adaptive position management, the other 16
archetypes and actual execution. Independent hypothetical case sums are not a
funded portfolio. Fixed 12bps costs exclude funding/impact; no live promotion.

Local dependencies for another CLI: this working tree, the original
`results/lc_consolidated_2026_09_15/judgment_v1/evidence/` roster/requests/lock,
`data/recovered_binance_minute_2026_09_14/btc_1m_2021_2026.parquet`, and this new
run namespace. The archive SHA is
`5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035`.
Research-module hashes and absolute local paths are bound; moving machines or
changing code is not a transparent resume. Preserve the original run.

Branch `quant/archetype-evidence-audit`, base HEAD `85923a4`. This implementation
and continuity documentation remain uncommitted. No push or PR. Unrelated graph
outputs and `rebuild_entry_population.py` were preserved, not included as work
for this task. Read `PROJECT.md` first in another development CLI; never give
developer memory or this outcome-capable workflow's reports to market assessors.
