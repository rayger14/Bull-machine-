# Support reaction suite implementation and source results

The offline support-reaction suite is built and its first source run is complete.
It now distinguishes the larger price range, hourly recovery and support behavior,
contextual volume evidence, and a confirmed minute entry structure. These are
separate jobs, not a single fusion score. It does not call a pivot box confirmed
accumulation. The result is a testable mechanical hypothesis, not demonstrated
profitability or a replacement for the live engine.

## What the source run found

The same frozen population contains **183 raw spring episodes**, with origins
from January 1, 2024 through August 23, 2026. No old support-survivor filter was
used. Counts below are proposed signals before delayed admission, not fills.

| Policy | Signals | Origin months |
| --- | ---: | ---: |
| A unchanged old thesis sequence | 3 | 3 |
| B revised structure and minute confirmation | 15 | 11 |
| C same B signals plus contextual volume evidence | 10 | 8 |

C preserved six signals with neutral recovery evidence and supportive supply
evidence, and four with supportive recovery evidence and neutral supply evidence.
It excluded four neutral/neutral cases and one supportive/adverse case. **We do
not know whether the excluded cases would have won or lost.** More signals than
A does not establish a better strategy: A-to-B changes the whole structural
policy, whereas B-to-C isolates the evidence rule at the signal stage.

B's complete raw denominator is 15 signals, 144 known rejections and 24 expiries:
57 original-spring failures, 27 parent invalidations, 36 insufficient-room cases,
24 failed minute support structures, 14 child expiries, 4 recovery expiries and
6 support expiries. No required source-price or volume unknowns occurred in this
run; synthetic tests still cover those paths. An expiry is not a losing trade.

The 183 parent lineages are distinct, but their maximum seven-day labels overlap
in 239 pairs. They are not 183 independent observations. Even before fill losses,
15 or 10 signals cannot meet the prior research floor of 50 closed fills in
12 origin months; that floor itself is not a statistical-power guarantee.
This history has already been exposed during research, so it is development data.

## What was implemented

- `scripts/research/support_reaction.py`: signed origin/evidence/decision records,
  causal hourly and minute observations, strict confirmed child high/low pivots,
  explicit original stop, parent room, unknowns and phase uncertainty. Decisions
  cite their source events. Price-only B remains independent of missing volume.
- `scripts/research/support_reaction_replay.py`: separately tagged transport into
  the unchanged fixed runtime; independent A/B/C books, capacity-free attribution,
  delayed-opening room and support checks, unknown accounting, and per-period
  comparisons with conservative seven-day label exclusions. Occupied test books
  reset to empty at each evaluation window; they do not inherit training positions.
- `scripts/research/support_reaction_study.py` and `run_support_reaction_study.py`:
  bounded source launch, hash bindings, case explanations, deterministic blind
  packets and annotation validation. The CLI offers **source only**, not an
  automatic natural-history economic launch.
- Three new test modules and shared literal fixtures cover 70 cases. The new
  [spec](../superpowers/specs/2026-10-03-support-reaction-suite-design.md) and
  [plan](../superpowers/plans/2026-10-03-support-reaction-suite.md) are frozen run
  inputs; completion tracking is in the separate progress ledger and this report.

The numeric rules are project hypotheses, not authenticated trader formulas.
The reviewed [Wyckoff Analytics explanation](https://www.wyckoffanalytics.com/wyckoff-method/)
supports considering relative volume, price spread and location; it does not
authenticate our exact 20-hour baseline, thresholds or entry clocks.

Before the natural run, the original spring's lifecycle was clarified: a completed
hourly low at or below its original stop cancels that setup, without claiming the
whole range has failed. After hourly support, the tighter minute support guard
applies. This was decided before inspecting the new historical counts or any P&L.

## Verification and review

Final focused command, exit0: **311 passed in31.08s**, including all70 new cases.

```sh
python3 -m pytest -o addopts='' -q tests/research/test_support_reaction*.py tests/research/test_thesis_*.py tests/research/test_lc_context_*.py tests/research/test_event_walkforward.py --tb=short
```

Quant design review approved the bounded spec/plan. Independent software review
identified four important issues. Root reproduced them before fixing them:
preserving a known opening when a later candle close is missing; retaining prior
context in negative review packets; placing preflight inside the timed failure
boundary; and reporting actual per-fold comparisons rather than only split IDs.
Regression tests verify each correction. The unchanged runtime now receives
explicit clocked unknown transitions from the adapter; earlier fills/cashflows
are retained, not retroactively erased. Exact inclusive2R comparisons also have
separate decimal-boundary regressions for signal and delayed admission.

The bare full-repository command remains blocked by
`tests/test_integration_fixes.py`, which raises SystemExit while loading missing
`configs/baseline_wyckoff_test.json`. Collection excluding that file reports ten
existing errors: `tests/archetypes/test_bull_archetypes_mvp.py`, both
`test_macro_pulse.py`/`test_v17_integration.py` under `tests/archive/versions/v170/`
and `tests/v170/`, `tests/integration/test_macro_backtest.py`,
`tests/integration/test_wiring_gates.py`, `tests/test_macro_backtest.py`,
`tests/test_multi_position.py`, and `tests/test_multi_position_full.py`. Missing
legacy imports include `FusionEngine`, `regime_manager`, `wick_trap_moneytaur`
and `baseline_wyckoff_multi_position`. These were not repaired in this scope.
An optional broader research run was deliberately stopped after766passes in
624.19s inside the unrelated virtual-book replay code. It is not a full-suite pass.

## Frozen artifacts and reproducibility

The one authorized natural source launch completed in **23.20seconds**, producing
10,633,274bytes total with peak process RSS923,238,400bytes. Output and time were
below the fixed256MiB/600second caps; the byte cap is not a RAM cap. Arrow emitted
sandbox CPU-cache discovery warnings but the process exited0. No retry occurred.

Directory: `results/support_reaction_2026_10_03/source_v1/`.
The receipt binds34 files and8 artifacts. Root separately reverified every hash,
all183 raw IDs and assessment seals,12 blind packet seals, and12 exact source-prefix
witnesses. Old A intent count remains3. Receipt SHA256:
`5c261f850fc125d7523a92c9e319594ef259d4cd3d4cc607236cb10fac28d71b`.

Artifacts: `source.json`, `summary.json`, `case_explanations.json`,
`benchmark.json`, `chronology.json`, `witnesses.json`, `launch.json`,
`bindings.json`, `receipt.json`. The independent semantic roster uses the three
lowest raw-ID hashes in each fixed eight-month block. It contains nine B
rejections, two expiries and one qualifying signal. Do not replace it with a
prettier roster after seeing these labels. It is not broad coverage of accepted
signals, and packet export is not completed independent annotation.

The three chronological evaluation windows retain43,43 and39 test origins after
conservative boundary exclusions. No fitting or natural economic replay was run.
The economic comparison library has synthetic tests, including real included and
censored cases around a boundary. Library readiness is not a backtest result.

The executed source command was:

```sh
python3 -m scripts.research.run_support_reaction_study source --output results/support_reaction_2026_10_03/source_v1
```

That directory is exclusive and must not be overwritten or automatically rerun.
Source/receipt, saved parent ledger and minute parquet remain local dependencies.
Another CLI must read PROJECT.md first and verify the receipt before continuing.
All implementation/report changes are local and uncommitted on
`quant/archetype-evidence-audit`; nothing was pushed or published.

## Next action and completion boundary

First independently annotate the12 frozen packets, using only each packet and
its rubric, not PROJECT, MEMORY, code verdicts or future outcomes. Preserve
unknown/disputed labels. The validator checks provenance/schema, not whether a
judgment is correct. Resolve material source/semantic disagreements and bind the
review to the exact source, code and policy before any natural scoring launch.

Then use a separately bounded, review-gated launcher around `compare` for one
locked A/B/C economic comparison under the declared primary and stress costs.
Use original stops, fixed2R targets and seven-day limits; report unknown totals,
missed winners and avoided losers only from complete matched outcomes. With this
small sample, that would be exploratory accounting, not certification. Do not
loosen gates or search clocks merely to reach the sample floor. More independent
history or prospective observations would still be needed to assess robustness.

No natural P&L, independent market annotation, full Wyckoff phase constructor,
Fibonacci/Gann revision or adaptive-management test was completed here. All17
native archetypes, live orders/configuration and fusion are unchanged. No paid
external model assessments or installs occurred; reviewer agents used session
usage. The documentation skill kept this handoff's implementation, source counts
and untested profitability claims separate in the established repository location.
