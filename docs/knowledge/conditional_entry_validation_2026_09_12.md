# Conditional entries and validation foundation — September 12, 2026

## What is now implemented

The research replay can now resolve **enter, wait for a specified completed
minute close above a fixed level, or reject**. Waiting has explicit processing
latency, confirmation availability, routing delay, cancellation and expiry.
Entry uses an eligible archive open, never the earlier close that confirmed the
setup. Stops and the original absolute exit deadline stay fixed; the existing
2R target is computed from the eventual reference entry. This is a deterministic
execution helper, not a newly connected agent trader or live order service.

[`conditional_entry.py`](../../scripts/research/conditional_entry.py) separates
outcome-hidden resolution from future outcome scoring. Rejected, expired,
cancelled, pending, invalid and missing-data states remain distinct. Pending or
missing observations do not earn zero-return or avoided-loss credit.

[`event_walkforward.py`](../../scripts/research/event_walkforward.py) adds
expanding calendar splits for irregular candidate events. Training labels must
finish strictly before the next test window, including the declared gap. Test
labels extending beyond the test boundary are explicitly excluded. This is an
event-aware splitting utility, **not a completed walk-forward backtest**.

The quant reviewer approved the [frozen first-unit contract](../superpowers/specs/2026-09-12-conditional-entry-validation.md)
and implementation. User-delegated routine research review was used; all seventeen
archetypes, fusion, production and live execution remain unchanged. No new model
market calls, dependencies, push or PR were made.

## Why walk-forward comes first

The intended main experiment moves forward through calendar time. Teacher
examples, prompts, thresholds and transformations must be frozen from allowed
earlier data, then decisions on the next period recorded before outcomes. Choosing
prompts or rules using test results is fitting even without changing model weights.
Overlapping outcome windows must not cross training/test boundaries.

This follows the chronological purpose of
[TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html),
but uses calendar and label timestamps rather than assuming equally spaced trade
events. A splitter cannot repair wrong feature availability or contaminated
teaching examples; these remain separate requirements.

CPCV can later provide secondary development robustness checks. The inspected
[MLFinLab interface](https://github.com/hudson-and-thames/mlfinlab/blob/master/mlfinlab/cross_validation/combinatorial.py)
explicitly represents information intervals, purging and combinatorial paths;
its inspected public implementation contains `pass` bodies, so it was not
installed as a working dependency. Any alternative needs version/license/source
review and independent boundary tests before adoption.

Important local audit correction: `scripts/champion/cpcv_wick_trap.py` combines
six fixed-window backtests into fifteen overlapping pairs. That is period
combination robustness, not purged model fitting or reconstructed CPCV paths.
It does not support treating fifteen pairs as independent validation trials.
The old result files are preserved, but their strong certification language
must not be inherited by this experiment. The older walk-forward tuning script
also scores configurations on its validation windows; it is not automatically
an untouched outer evaluation for this new policy.

## Known-case integration check — not strategy validation

Before scoring, the contract froze the already revealed April E01/E02/E03 cases
as development fixtures: fixed wait level = last completed 1m high; 15-minute
entry expiry; 0 and 90 seconds assumed total processing latency; zero routing
delay; original stop/deadline. These choices are arbitrary plumbing fixtures,
not a newly learned Wyckoff rule, new agent decisions or an optimized policy.

All eighteen rows were retained: three cases × three actions × two delays.
Zero-delay immediate entries exactly reproduce the prior scorer outcomes.

| Case and decision UTC | Enter, 0s | Enter, 90s | Wait, 0s | Wait, 90s |
|---|---:|---:|---:|---:|
| E01 hourly, Apr 4 16:00 | −$471.85 | −$468.83 | Expired, no entry | Expired, no entry |
| E02 minute, Apr 2 03:53 | −$251.10 | −$215.55 | Expired, no entry | Expired, no entry |
| E03 minute, Apr 1 00:16 | −$202.92 | −$268.75 | −$268.75 | −$259.98 |

Every filled bracket stopped. Reject controls had no exposure. In E03 waiting
still lost, and changed the reference entry/risk; delay is not assumed beneficial.
The expired entries are legitimate deterministic no-exposure fixtures, not
credit earned by the earlier invalid agent assessments, which remain unchanged.

Starting equity unspecified; independent $50,000 notional and $60 round-trip
cost per filled case. Average initial price risk, excluding costs: immediate
0s $248.62 across three entries; immediate 90s $257.71 across three; wait 0s
$208.75 and wait 90s $199.98, each across one. These are alternative, dependent
case calculations, not eight independent trades or a combined portfolio.
The 90-second assumption rounds to minute sampling; it is not measured model
latency. Funding, impact and inference costs remain absent; execution uncertified.

## Verification

TDD: 34 initial tests failed because the helpers were absent, then passed.
Independent review reproduced a future-index defect: appending an irrelevant
duplicate future row changed an earlier decision. Two new regressions failed
before the fix and passed after consumed-only validation and outcome-window
slicing. Consumed duplicates still fail. Additional timezone/pre-arm/expiry
boundary tests bring the focused suite to **40 passed**; the independent reviewer
re-ran them and approved the bounded unit.

Fresh full research run: **596 passed**, one existing LibreSSL warning, 12.94s.
This is 556 pre-existing tests plus 40 new tests; prior pilot-private tests are
not included in that count. Repeated diagnostic execution produced identical
saved artifacts. Source, plans and implementation hashes were frozen before run.
The post-run quant audit independently verified all ten manifest hashes,
reproduced all eighteen resolution/outcome objects exactly, and matched all
three zero-delay immediate outcomes to the original pilot. No discrepancy found.
Implementation commit: `2e907b7`; prospective contract commit: `6c14adf`.

Private artifacts: `results/conditional_entry_2026_09_12/` contains the
`run_diagnostic.py` harness, manifest and results; it remains local/ignored.

| Artifact | SHA256 |
|---|---|
| Frozen spec | `e4b1c28e45f0c4e0448847251b624467f8782e2ba99006174dd59d917c29ed9c` |
| Diagnostic manifest | `c8b606979ed44a19388c326048e52942e70d1a2cce4afc42bd5581ec2033fe73` |
| Repeated diagnostic results | `527677a763971a81d72d1968245e3131de1bb37e4b8f5cdbcc5a9d3a0982da8b` |

## Next deliverable

Connect the specialist to a finite, evidence-ID-bound enter/wait/reject plan
menu and clarify the critic's rules using already locked development examples.
Then freeze the full candidate ledger and replay actual entry-readiness events
chronologically. The old candidate-order simulator cannot be reused unchanged:
different waits can reorder entries and alter which positions block others.

Only after that integration, preregister calendar folds and run matched
code-immediate, code-confirmation, agent and no-exposure arms, reporting hourly
and minute tracks separately with equal priority. Use full event labels, fixed
occupancy/tie handling, latency/cost stress and every invalid/missing case.
No fold dates, funded portfolio, new paid agent experiment, full walk-forward
performance or CPCV performance have been claimed or completed in this unit.
