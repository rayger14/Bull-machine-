# Support reaction review and historical accounting

The twelve-case source review and one historical comparison are complete. The
revised structure policy is roughly break-even under primary assumptions and
loses under stress. Adding the proposed volume gate makes performance materially
worse: it excludes five winners and no losers in the independent comparison.
Do not promote either version as dependable. This is evidence against this
specific gate on this sample, not against all volume analysis or the wider vision.

The source review supports the stated mechanical interpretation on these cases,
with an explicit minute-high limitation. Correct explanations and passing tests
did not produce a demonstrated trading edge. The result is not trader
certification, complete Wyckoff phase recognition or native LC/all17 validation.

## Source review

A fresh packet-only assessor processed twelve fixed cases chronologically,
locking each answer before accessing the next case. The assessor did not receive
developer history, code verdicts or future outcomes. One accumulating reviewer
context is a source-fidelity audit, not twelve independent predictive trials.
All packets reconstruct exactly from the shortened views; citations retain their
original source identities and availability times. All twelve parent ranges
predate their origin candles, and all displayed observations were available by
the cutoff.

The assessor marked eleven nonentries and one hypothetical entry. These match
the engine's enter/non-enter decisions when two expiries count as nonentries.
Only one positive case was in this fixed roster; this is not broad positive-path
coverage. All twelve phases remained unclassified.

| Case | Independent decision | Decisive evidence |
| --- | --- | --- |
| 01 | Reject | Valid local sequence but only 1.9387R room |
| 02 | Reject | Original stop crossed before recovery/support |
| 03 | Reject | Recovery occurred, but stop crossed before qualifying support |
| 04 | Reject | Confirmed minute sequence but only 1.6260R room |
| 05 | Reject | Supportive hourly evidence, then child support failed |
| 06 | Reject | First subsequent hour crossed original stop |
| 07 | Hypothetical enter | Confirmed sequence, 2.1350R room, supportive demand and neutral supply |
| 08 | Reject | 1.3267R room; first-high interpretation also needs care |
| 09 | Reject | No breakout before child expiry; room also prohibitive |
| 10 | Reject | No recovered support before original-stop failure |
| 11 | Reject | Recovery level not cleared by the 48-hour deadline |
| 12 | Reject | Original-stop failure, independent of volume label |

Root read every claim, checked cited source facts and recomputed observed
recovery/support volume, spread and close-location ratios from packet candles.
Eleven adjudications found no material discrepancy. Case08 retains a documented
interpretation limitation: its later confirmed low sits above the first locked
high. The frozen rule permits that ordering without replacing the high. It
should not be described as a fresh escape from an ordinary minute range.
The case fails room under either interpretation; no rule was changed mid-study.

Two other examples matter to the broader vision. Case11 has constructive demand
without crossing the required recovery level; a rejected setup does not erase
that demand. Case12 has heavy selling but a close fraction just above one-third,
so the categorical formula is neutral. The label is not the entire market story.
These observations are future hypothesis material, not evidence that different
thresholds would make money.

Review directory: `results/support_reaction_2026_10_03/review_v1`.
Semantic receipt SHA256:
`d998fae5d939a897d1ded8c214205c402c75d1cd20657a4ba4aa7dd6d5885b6e`.
Receipt seal:
`23fbb37130ecccf4135c20d0cd35c258e4bedc5e3ee8602e95dd16d1f7ef3d7d`.
The receipt binds111 source, code, plan and review files. A manually copied
adjudication citation typo was rejected before sealing, then corrected against
the source; locked assessor answers were never changed.

## What the accounting compares

All183 raw spring origins from January2024 through August23,2026 remain in every
arm. These are not all native LC live trades and not all17 archetypes.

- A: unchanged old thesis entry.
- B: revised support-reaction structure, with causal minute timing.
- C: the same B trigger, additionally requiring the specified contextual volume
  evidence. Neutral plus supportive is allowed; universal indicator agreement is
  not required.

Common assumptions: $100 intended risk, $50,000 notional cap, original spring
stop, fixed2R target and seven-day deadline. Normal costs use90-second delay and
6bp each-side fees; stress uses180seconds and12bp. Both assume8bp adverse funding
every eight hours. These are modeled costs, not venue-fill or observed funding
claims. Adaptive management, Fibonacci/Gann and macro/fusion are not tested here.

The unchanged replay reducer still determines fills and cashflows. The new
orchestrator separately finalizes nonoccupying no-intent watchers at their
source-known closure and groups possible positions by prospective seven-day
windows. It never selects groups by realized profit or holding time. All rejected
and unknown rows remain counted. Possible exposure candidates are A3, B15 and
C10, in3/12/9 overlapping components respectively. Occupied unknown state aborts
the run rather than silently resetting capacity. Global portfolio drawdown is
explicitly unavailable, not synthesized from component drawdowns.

Capacity-free comparisons attribute policy differences; their sums are not a
portfolio return. Separate occupied books show admission effects. Three fixed
chronological evaluation blocks purge maximum seven-day labels at boundaries,
and their occupied books reset at each test window. No tuning or fitting occurs;
the exposed calendar is not a pristine holdout.

## Verification and run status

New review and accounting code was built test-first. The sparse scheduler matched
the original replay on primary/stress rows, entries, admissions, positions and
cashflows, pending reservations, equal horizon boundaries and chronological
metrics. Tests also cover unknown occupancy, missing execution candles, full
seven-day holding/funding, altered bindings, timeout and output exclusivity.

Fresh software-agent creation hit a session thread limit. The existing software
reviewer inspected the addition read-only; the market assessor remained fresh
and isolated. The software review found one Important issue: expensive preflight
was outside the wall-clock guard. Root reproduced and fixed it. All preparation
is now inside the same timed, failure-recorded boundary. No other material
software findings were reported; source semantics and natural accounting were
separate responsibilities, not delegated completion claims.

Final focused command:

```sh
python3 -m pytest -o addopts='' -q tests/research/test_support_reaction*.py tests/research/test_thesis_*.py tests/research/test_lc_context_*.py tests/research/test_event_walkforward.py --tb=short
```

332 passed in48.40seconds, including21 new tests. Bare full-repository pytest
still exits3 during collection because `tests/test_integration_fixes.py` exits
after missing `configs/baseline_wyckoff_test.json`; this is not a repo-wide pass.

The single natural primary/stress launch completed in422.58seconds, inside its
600second/256MiB aggregate-output bounds. Output before receipt was13,671,912bytes;
peakRSS752,566,272bytes (the output cap is not a RAM cap). Root independently
verified112 bound files, all10 artifact hashes, all183 rows per arm/scenario,
component/runtime seals, cashflow accounting and nonoverlapping occupied fills.
No unknown outcomes, failure artifact or retry. Global drawdown remains unavailable.

Directory: `results/support_reaction_2026_10_03/economics_v1`.
Receipt SHA256:
`e6ab5ec785e0c1d1c9ca4e099a5d476f80b40b1a866a78a7f85d6a79a4598a8a`.
Receipt seal:
`5b7712dba0156db287c6edc9e867766450f7bc22487d1a0dba46323672a231e2`.
Arrow printed sandbox CPU-cache-query warnings; the process exited0 and verified
all output. These were not missing market-data errors.

Launch command already used; do not repeat into an existing directory:

```sh
python3 -m scripts.research.support_reaction_economics --output results/support_reaction_2026_10_03/economics_v1 --review results/support_reaction_2026_10_03/review_v1/semantic_receipt.json
```

## Economic results

The following separate books permit one pending/open position at a time. All
dollar amounts are hypothetical net accounting under the declared sizing and
cost assumptions, not actual live losses or account percentage returns.

| Policy | Closed fills | Primary net | Stress net |
| --- | ---: | ---: | ---: |
| A old entry | 3 | −$13.87 | −$29.26 |
| B revised structure | 11 | −$10.62 | −$119.63 |
| C structure plus volume gate | 7 | −$542.35 | −$566.37 |

Independent attribution permits overlapping positions only to compare rules:

| Policy | Closed fills | Wins / losses | Primary net | Stress net |
| --- | ---: | ---: | ---: | ---: |
| A | 3 | 1 / 2 | −$13.87 | −$29.26 |
| B | 13 | 6 / 7 | +$20.70 | −$107.22 |
| C | 8 | 1 / 7 | −$660.21 | −$683.99 |

The volume gate removes five B winners and none of its seven losers under both
scenarios. Primary independent net deteriorates by$680.91; stress by$576.78.
Four removed winners have neutral/neutral evidence; one has supportive/adverse
evidence. This does not justify reversing the filter on the same known outcomes.
C has negative gross price PnL before costs as well; its failure is not only a
fee or funding assumption. B's positive gross is largely consumed by costs.

| Independent accounting | Gross price PnL | Fees | Assumed adverse funding | Net |
| --- | ---: | ---: | ---: | ---: |
| B primary | +$448.12 | $129.79 | $297.63 | +$20.70 |
| B stress | +$405.26 | $233.20 | $279.27 | −$107.22 |
| C primary | −$459.36 | $82.71 | $118.13 | −$660.21 |
| C stress | −$427.13 | $147.89 | $108.98 | −$683.99 |

Rounding can shift the displayed sum by a cent. Funding is a fixed adverse
assumption, not downloaded realized funding. Higher fees also change position
size, so individual stressed funding/loss amounts need not monotonically grow.
Do not remove costs to manufacture an edge or call these exact venue economics.

## Complete B intent reconciliation

Fifteen source intents became thirteen independent fills and two admission
cancellations. Occupancy skips two of those fills, leaving eleven. The following
UTC times identify the original spring close, not the eventual entry time.

| Origin UTC | Primary independent net | Resolution | Occupied book | C gate |
| --- | ---: | --- | --- | --- |
| 2024-02-24 12:00 | +$150.90 | Target | Filled | Excludes winner |
| 2024-05-10 04:00 | −$100.00 | Stop | Filled | Retains loser |
| 2024-05-11 00:00 | +$155.27 | Target | Filled | Excludes winner |
| 2024-05-11 12:00 | +$149.18 | Target | Busy | Excludes winner |
| 2024-11-02 00:00 | −$115.06 | Stop | Filled | Retains loser |
| 2025-01-10 00:00 | −$131.01 | Stop | Filled | Retains loser |
| 2025-01-10 16:00 | −$117.86 | Stop | Busy | Retains loser |
| 2025-06-02 16:00 | $0.00 | Admission room failed | No fill | Same cancellation |
| 2025-08-22 00:00 | −$106.21 | Stop | Filled | Retains loser |
| 2025-12-24 16:00 | +$116.59 | Target | Filled | Excludes winner |
| 2026-01-17 12:00 | −$112.12 | Stop | Filled | Retains loser |
| 2026-01-25 12:00 | −$107.81 | Stop | Filled | Retains loser |
| 2026-02-18 04:00 | $0.00 | Admission room failed | No fill | Same cancellation |
| 2026-04-25 20:00 | +$129.87 | Target | Filled | Retains winner |
| 2026-08-01 20:00 | +$108.98 | Target | Filled | Excludes winner |

Case07's source-only hypothetical entry was the June2 setup. At22:17 its close
105215 had2.135R room; after the90second delay, the22:19 eligible opening105377.1
failed the room check. It never filled. Semantic eligibility is not an agent
winning-trade claim, and the runtime did not pretend the earlier price remained
available. Both source cancellations also occur in stress.

## Chronology and research decision

Occupied B results across the three fixed evaluation blocks were:

| Evaluation window UTC | Raw test cases after purge | Fills | Primary net | Stress net |
| --- | ---: | ---: | ---: | ---: |
| 2024-09-01 to 2025-05-01 | 43 | 2 | −$246.08 | −$243.05 |
| 2025-05-01 to 2026-01-01 | 43 | 2 | +$10.38 | −$4.69 |
| 2026-01-01 to 2026-08-24 | 39 | 4 | +$18.92 | −$22.04 |

These are fixed-policy chronological reports, not trained walk-forward or CPCV
certification. First-block development history is excluded from the above table;
exposed history does not become pristine out-of-sample data. B fills span nine
origin months, C six; both fail the existing50fill/12month floor, which itself is
not a power guarantee. There are239 overlapping maximum-label origin pairs.

The delegated quant recommendation is to park this frozen policy line as a
dependable-archetype candidate and finish failure attribution from saved ledgers.
That closure deliverable is now completed above. C is not supported as an
improvement; B may remain a research comparison baseline, not a deployment
candidate. Keep the tested suite and evidence; do not keep tuning this same small
set until its report turns green.

Next decision: select one materially different, source-backed setup hypothesis
from the existing archetype/teaching audit before commissioning another backtest.
It must address an observed interpretation gap and have enough eligible history
for testing. Freeze its rule and evaluation calendar first; do not invert C,
loosen room, move stops or shorten holding time to rescue these known trades.
No additional model calls or experiment were launched for this closure memo.

## Continuity and safety

Existing quant branch and frozen strategy/runtime/source files remain unchanged.
No native/live/config/fusion changes, external paid market API calls, installs,
commit, push or PR. Session agents still consume model usage. All new work is
local/uncommitted. Another CLI must first read PROJECT.md and retain the local
minute parquet, parent ledger and old/new source receipts. See the frozen
follow-on spec/plan and matching `.superpowers/sdd` progress ledger.
All workers and local test/replay processes are finished; nothing remains running.
