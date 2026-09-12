# Conditional entry and chronological validation — bounded first unit

User approved enter/wait/reject and proper backtesting, with routine research
decisions delegated to the quant reviewer. The conditional-entry quant reviewer
approved this first unit with readiness, cancellation and missing-data safeguards.
This is research-only; keep all seventeen archetypes and production unchanged.

## First implementation

Add pure helpers beside the existing `entry_case_outcome.score_case`, not a new
live engine. A frozen long-only plan chooses `enter`, `wait_close_above`, or
`reject`; a wait names one fixed level available before decision. This means a
close strictly above the level, not necessarily a crossing from below.

Inputs include decision, absolute entry expiry, absolute exit deadline, fixed
stop, total assumed assessment/review latency seconds and separate routing lag.
Arm at ceil(decision + processing latency, minute). Monitor only full 1m candles
opening at or after arm time. First qualifying completed close may confirm;
entry samples ceil(confirmation availability + routing lag, minute). Immediate
entry uses arm time plus routing lag. Both confirmation and entry must occur
strictly before entry expiry and exit deadline. Waiting never extends the exit
deadline. Synthetic reference delays are assumptions, not measured inference.

During all pending time from decision, an observed completed-bar low <= stop
cancels; cancellation precedes confirmation when both occur. At a proposed fill,
open <= stop cancels; never inspect that candle's future low to approve entry.
Before arming, bars can cancel but cannot confirm. No new stop or second attempt.

Resolver takes an explicit `as_of`: candles close by it, opens exist at it. It
returns entry-ready, rejected, cancelled, expired, pending, data-unavailable or
invalid-plan, with observable resolution time and no hypothetical PnL for missing
data. It validates only observations consumed up to resolution. Unknown tails
must not be turned into early expiry. Same-source fills remain uncertified.

Scoring reuses the unchanged 2R bracket with the remaining time to the original
exit deadline, $50k independent notional and 12bps round-trip cost. This is a case
diagnostic, NOT a funded portfolio. A future portfolio must merge actual entry
readiness events chronologically, with fixed ties and occupancy policy; sorting
only original candidate timestamps is wrong when different waits reorder fills.

## Walk-forward first, CPCV second

Add a pure expanding chronological splitter for irregular candidate timestamps.
Each event records decision and conservative full policy label-end, including
waiting and any permitted exit extension (none in this first policy). Fold
boundaries are explicit UTC calendar times, not equal candidate-row counts.
Training decisions precede test start; training label-end must be strictly before
test start minus a predeclared gap. Test decisions are in [start,end); labels at
or beyond end are explicitly excluded/censored, not zero-return examples.
Retain IDs and exclusion reasons. Multiple archetypes/variants for one timestamp
stay on the same calendar side. No fitting, prompt selection or test-score
optimization is performed by this helper. Input histories may legitimately
overlap in backward-looking causal features; future outcome overlap cannot.

Later evaluation must freeze teacher dossier, examples, prompts, model request,
feature transforms and candidate gates using only allowed development data.
Prompt/model/threshold selection counts as fitting even if model weights do not
change. Fold test decisions lock before outcomes; examples from later folds
cannot teach earlier-fold agents. Historical pretraining contamination remains
unverifiable, so reserve prospective shadow evidence for final claims.

CPCV is a secondary development robustness tool once event intervals and a
fitted selection procedure exist; it can train on later periods and therefore
is not the primary deployment simulation. Purge overlapping label intervals,
apply explicit embargo, isolate each fold's examples/fit state, and reconstruct
paths before calling outputs CPCV paths. Do not relabel pairwise sums of fixed
period PnL as purged CPCV or count overlapping combinations as independent trials.

Existing `scripts/champion/cpcv_wick_trap.py` aggregates six fixed-period results
into fifteen pairs, with no purge/train/path machinery. Existing
`scripts/research/tools/tune_walkforward.py` scores candidate configurations on
its validation windows; its mere presence does not establish nested, untouched
evaluation for this agent experiment. Preserve old artifacts; do not inherit
their validation labels as certification.

Primary references checked September 12, 2026:

- [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html): chronological splits and gap, with equal-spacing assumptions for comparable sample-based folds. Our events are irregular and require label-aware calendar handling.
- [MLFinLab combinatorial interface](https://github.com/hudson-and-thames/mlfinlab/blob/master/mlfinlab/cross_validation/combinatorial.py): exposes information intervals, purging and combinatorial paths, but the inspected public implementation contains `pass` bodies; it is not a drop-in executable dependency.

No new dependency is necessary for this first unit. Any later CPCV package needs
source/version/license inspection and independent interval-boundary tests.

## Verification and stopping boundary

Test cancellation-before-confirmation, expiry equality, processing/routing lag,
pre-arm observations, truncated and malformed consumed data, ignored future
values, deadline preservation, invalid numeric inputs, timezone normalization,
and overlapping train/test outcome labels. Verify same-source immediate policy
parity with the existing scorer. Re-run the full research suite and obtain an
independent quant/code review.

Optional integration diagnostic is frozen NOW: the already revealed E01/E02/E03
April cases from the evidence-ID pilot, each enter versus wait above the last
completed 1m high, 15-minute entry expiry, 0 and 90 seconds assumed total
processing latency, zero additional routing lag, original stop/deadline and
economics. All six policy/delay combinations per case include reject as a zero
exposure control (three actions × two delays). These arbitrary known-case
fixtures are development plumbing checks, not agent choices, a new holdout,
walk-forward results, tuning or improved expectancy. Hash this spec and input
cases before scoring; preserve every row and do not select a winning variant.

Do not launch new paid agent market calls or claim proper full backtesting from
this unit. Next: correct critic contract, freeze the agent's finite evidence-ID
plan choices, and build the complete candidate/chronological occupancy ledger;
then register actual multi-period folds and run matched engine/agent arms.
