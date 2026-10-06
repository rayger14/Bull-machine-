# Full calendar thesis source census results

## What finished

The unchanged range/spring/test/strength/support/minute-entry hypothesis produced
**183 raw episodes but only 3 complete entry signals across 32 origin months**.
This source-only census cannot meet the previously specified minimum of 50 thesis
fills across 12 months: entry signals are an upper bound on fills, and only three
months contain a signal. No new profits, losses or trade outcomes were calculated.

This is a new mechanical research hypothesis, not the native liquidity-compression
archetype, all 17 archetypes, complete Wyckoff accumulation, or an intelligent
master trader. The Fib and elapsed-time rules are declared project hypotheses,
not authenticated hidden-Fibonacci or Gann formulas. Daily context is annotation,
not a required bullish daily filter. Macro and order flow are not supplied here.

## Coverage and counts

The run scanned BTC origin closes from January 1, 2024 through August 24, 2026
exclusive. Aggregation starts December 2, 2023; the source tail ends September 1,
2026 exclusive. All 1,445,760 expected minute rows were present. All 5,796 expected
four-hour origin decisions were recorded; no required candle was unknown.

| Reached stage | Episodes | Why the other episodes stopped before this stage |
| --- | ---: | --- |
| Spring below a previously known range floor, then close back inside | 183 | Common raw population |
| Subsequent hourly test | 80 | 58 test deadlines expired; 45 ranges invalidated |
| Subsequent hourly strength | 37 | 2 strength deadlines expired; 41 ranges invalidated |
| Successful first support retest | 17 | 19 first touches failed; 1 support deadline expired |
| Minute confirmation within the fixed window | 3 | 14 confirmation windows expired |

These are cumulative stages, not independent trade counts. Before raw admission,
2,978 four-hour closes were not springs, 1,794 had no active parent, and 841 belonged
to a lineage already consumed by an earlier spring. Lineages were never reset at
month or year boundaries. A later invalidation does not erase an earlier entry
signal or reclassify an earlier known expiry.

| Origin calendar | Raw episodes | Tests | Strength | Support | Entry signals |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2024 | 73 | 33 | 12 | 8 | 0 |
| 2025 | 71 | 28 | 18 | 7 | 2 |
| 2026 through August 23 | 39 | 19 | 7 | 2 | 1 |

The three confirmation times were March 31, 2025 at 23:07 UTC; July 30, 2025 at
04:08 UTC; and February 13, 2026 at 15:10 UTC. These are **signals, not verified
fills or profitable trades**. Daily parent context was known for 123 episodes
and absent for 60; absence is not fabricated bullish evidence.

## What this tells us

The implemented definition is too sparse for the planned strategy-advancement
study. This is not evidence that combining timeframes cannot work, nor proof that
every filter is useful. We now know exactly where this particular interpretation
removes candidates. At the final step, 14 of the 17 support setups did not close a
minute candle above the support candle's high within the frozen window. That
window is a research choice; failure to pass it does not prove a bad trade.
The [17-row source trace](thesis_support_trace_2026_10_03.md) shows every support
threshold, highest eligible minute close and confirmation/expiry. Two episode
rows share one support window; these are not 17 independent trading opportunities.

We must not tune that window merely to manufacture a profitable backtest. The next useful
step is a bounded source-fidelity review of those 17 support cases: compare the
known range, ordered sequence, support threshold and confirmation timing against
the saved original teachings, without ranking cases by later profit. Any justified
rule change must become a separately specified hypothesis with a fresh validation
plan; this revealed history is not an untouched holdout. No full economic campaign
or live deployment is authorized by these source counts.

## Fib and timing coverage

Fib anchors became defined in 37 episodes. There were 111 Fib-time and 1,281
elapsed-time Gann-style scheduled clocks, with 1,385 unique clocks after merging
seven overlaps. Complete hourly observations existed at every clock. Range
destinations were defined for all 183 episodes.

These are source definitions and observations, not post-entry reviews, executed
partials or successful management actions. The original summary's termination
split uses structural invalidation/unknown timestamps only; it does not include
the mandatory seven-day deadline. Do not read that split as action eligibility.
Independent review and root's separate arithmetic agree on the corrected split:
**449 before termination, 936 at or after**, including the deadline. The original
489/896 labels misclassified 40 deadline clocks; all 1,385 observations and entry
counts are unchanged. The correction is saved in `review_v1.json`; immutable
source artifacts are preserved. Consumers must use the correction, not the old
termination labels, when discussing action eligibility.

## Verification and resources

The single source run exited successfully in 55.62 seconds, including a real
January parity check and a cross-year origin-partition witness. It wrote
39,348,391 bytes, about 37.53 MiB, within the 256 MiB output cap. Peak process RSS
was 1,275,609,088 bytes, about 1.19 GiB; output size was never presented as a RAM cap.
No historical process remains running. PyArrow emitted sandbox CPU-cache probe
warnings but completed successfully; there was no retry.

The saved January packets match exactly, including 2,339 catalog events. A
January 1, 2025 partition rebuilt 73 earlier episodes from a truncated source
prefix and resumed 110 later episodes with continuous consumed-lineage state.
Packets, raw decisions, the closed prefix catalog and final admission state all
match the continuous census. This is an origin-partition restart with complete
episode tails, not a new online position-restart claim.

All 183 packet seals/events, 17 bound input/code/reference files and the five
receipt-listed artifact hashes were verified after the run. The two new modules
and their tests use the frozen compiler without modifying January's modules/spec.
Fresh regression: 186 tests passed in 5.92 seconds, including 19 new tests.

Fresh software review found two important issues before launch: reference
preflight failures were outside the bounded failure-record lifecycle, and missing
source data could yield a false definitive sample-size conclusion. Three new
regression cases reproduced them, then passed after correction. No minor issues
were deferred by that software review. Independent quant review subsequently
reconstructed 31,364 candles, 3,353 pivots, all 5,796 admission decisions, all 183
parent bindings/ATR/stops/ordered sequences, Fib/time rules and every monthly
funnel without importing the census builder or episode compiler. It confirms
source qualification and insufficient sample size. Its separate timing-label
finding is corrected above; the frozen code/artifact is not silently rewritten.
The 71 previous LC source bindings also still match.

The full repository suite is not green. Its attempted run aborted during collection
because `tests/test_integration_fixes.py` calls `sys.exit` when the legacy
`configs/baseline_wyckoff_test.json` file is absent. A collection-only diagnostic
excluding that aborting script reports ten unchanged legacy import failures:

- `tests/archetypes/test_bull_archetypes_mvp.py`: missing `wick_trap_moneytaur`.
- `tests/archive/versions/v170/test_macro_pulse.py` and `test_v17_integration.py`:
  missing legacy `FusionEngine` import.
- `tests/integration/test_macro_backtest.py` and `tests/test_macro_backtest.py`:
  missing legacy `FusionEngine` import.
- `tests/integration/test_wiring_gates.py`: missing `engine.context.regime_manager`.
- `tests/test_multi_position.py` and `tests/test_multi_position_full.py`: missing
  `bin.baseline_wyckoff_multi_position`.
- `tests/v170/test_macro_pulse.py` and `test_v17_integration.py`: missing legacy
  `FusionEngine` import.

These failures were reported, not silently repaired or counted as passing tests.

## Files and continuity

- [Implementation plan](../superpowers/plans/2026-10-03-thesis-source-census.md).
- `scripts/research/thesis_census.py`: indexed source census and monthly summaries.
- `scripts/research/run_thesis_census.py`: locked source-only launcher.
- `tests/research/test_thesis_census.py` and `test_thesis_census_run.py`: new tests.
- `results/thesis_census_2026_10_03/source_v1/`: launch, full bindings, source,
  monthly summary, parity witnesses and completion receipt.
- `results/thesis_census_2026_10_03/review_v1.json`: independent source audit and
  the deadline-aware timing-label correction. SHA256
  `38c36c8572d97f3a1f9595f2a47a1b7768589726c3acd0b33b7c18d850c27a72`.
- `.superpowers/sdd/2026-10-03-thesis-source-census/progress.md`: local work ledger.

Receipt SHA256:
`1c13cc6eabc94ec8272f92998738242fb94111536f341a833772815ff1328eb7`.
Source SHA256:
`b27440a9247ea93b77458f08fc13009ab56679a43bbfc66aba3a86d43039730b`.

Everything remains local and uncommitted on `quant/archetype-evidence-audit`.
The minute archive, saved parent ledger and January reference artifacts are local
dependencies; GitHub alone does not reproduce this run. No live/config/fusion
changes, paid market assessments, installs, downloads, commits, push or PR occurred.
Quant and software reviewers used ordinary session usage. The worktree and audit
artifacts are retained as requested, so a handoff still needs this local checkout.
