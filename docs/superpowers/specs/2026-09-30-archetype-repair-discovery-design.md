# Bounded archetype repair and discovery study

Started September 30, 2026; reviewed October 1. Written research design following approval of the broad
roadmap. Status: ready for written-spec review, not implemented or scored.
The [ranked scorecard](../../knowledge/archetype_repair_scorecard_2026_09_30.md)
contains current-source findings and the data/exposure inventory.

## Purpose and limits

Determine whether a precise hourly repair or a new minute-native sequence merits
forward paper validation. Share explanatory structure across the engine while
isolating each archetype's decisions, cooldown, positions and economics.
One profitable historical result is not the acceptance criterion.

Preserve all 17 production archetypes, their configuration and frozen research
artifacts. No live orders/config/fusion changes, model market roles, optimizer,
new dependencies, commit/push/PR or prospective collector deployment. Existing
research branch remains in use. This design does not authorize production risk.

At most three hypothesis slots are permitted: R1, R2 and R3 below. R1 and R3
are active; R2 is parked before implementation or scoring following the final
prior-study check. Keep its specification as a rejected proposal, not a launch
instruction. No replacement is added to use up the budget. Baselines and cost/
delay stresses are declared comparators, not extra candidates to select afterward.
The old upside-LC rule stays unchanged; neither room >=2R nor the old minute
confirmation is added. No additional archetype replaces a failed hypothesis
inside this campaign. A new idea requires a separately recorded next campaign.

## Source and opportunity records

Use the hash-bound Binance USD-M BTCUSDT minute archive and completed aggregates
from that same stream. Never join legacy Coinbase-labelled entry prices to its
outcome bars. Bar-close availability is a historical assumption; actual feed
receipt and execution latency are not established by a hash.

The native feature/signal producer remains the comparator, including its recorded
missing-feed defaults. A price-based repair does not turn those defaults into
observed macro/derivatives evidence. Funding/OI families are included in the
all-17 observability census but not ranked by simulated profits on invented inputs.

Persist the full feature/diagnostic rows before projecting individual candidates.
Each row retains source/instrument/version, source open and close, decision and
available-at clocks, current and previous features, observed/defaulted/missing
status, each archetype's identity result and error reason, gates, fusion stage,
cooldown before/after, pre-dedup signal and final selected status. Distinguish
"not evaluated" from fail and missing from observed zero. An exception in an
identity required by a scored arm blocks scoring that cohort until resolved in
a separately versioned, verified source run; do not silently exclude error rows.
Existing observer
aggregate gate output alone is not a complete per-gate evidence ledger.

Record parent lineage/version and available-at clocks, child/level IDs, location,
sequence stage, entry/stop/expiry and initial-risk version. Unknown parent context
remains unknown; it is not silently a favorable range or universal rejection.
No post-entry excursion, duration or known outcome is an input feature.

For hourly repair comparisons (R1; R2 only if separately unparked), define a raw
opportunity before either arm's cooldown/gates/selection:
the original native identity passes on the current causal feature row. Its stable
ID is `(instrument, source-hour-close, archetype, native-identity-version)`.
Both arms consume all these IDs, including rows with no eventual signal. Record
arm-specific repair permission, gate evaluation, emission, cooldown, native
all-17 selected status for observation only, pending/busy status, entry and outcome.
Raw structural errors are retained as errors, not assigned passing IDs.

Instantiate separate research state for each native baseline and repair.
An identity/repair rejection does **not** arm cooldown. A constructed signal
arms that arm's legacy signal-time cooldown even if its book is busy; preserve
the original point of arming before any later selection. Headline R1/R2 books
are single-archetype isolation, with no all-17 dedup or relative-score competition.
The unchanged full-engine observer records competition only as a diagnostic;
its selected winners are not the entry population. Do not filter an already-
completed trade list and call it an engine replay. The headline includes the
subsequent changes in emissions and busy skips caused by each arm's local state.
Never sum isolated books as a funded portfolio. Short/neutral books are unsupported
in the current wrapper and remain explicitly unscored, not converted into longs.

## R1 Directional permission for trap within trend

**Question:** does requiring actual upward EMA alignment improve the isolated
long TWT policy relative to its present nondirectional fusion fallback?

Baseline: current champion TWT identity, gates, fusion and signal-time cooldown,
under the declared common research execution model below.

Repair: add exactly one required predicate to the original identity result:
`price_above_ema_50` is a finite, observed value >=1.0 at the decision time.
Missing/defaulted directional evidence is unknown and ineligible in the repair.
Thus a high or low fusion value alone cannot authorize a below-EMA long trap.
Keep wick/ADX thresholds, all other gates, fusion weights, ATR stop and management
unchanged. Do not replace the scalar with a different unvalidated scalar.

The EMA is a limited existing trend proxy, not proof of a complete Wyckoff phase
or higher-timeframe structural thesis. Record actual 4H/daily structure as context
without adding a second veto. Synthetic witnesses must preserve above-EMA cases
and reject the demonstrated below-EMA/low-score case. The economic hypothesis
fails if the removed class is useful enough that the fixed repair does not meet
the advancement criteria. Do not reinstate chosen exceptions after outcomes.

## R2 Identified level for liquidity sweep — parked, not scheduled

**Disposition:** related hourly level-restoration studies already failed, as
documented in the scorecard's prior-work reconciliation. The exact old September
implementation was not recovered here; equivalence to this 24-hour rule is not
claimed. A changed window alone is not sufficient reason to reopen the family.
The definition below is retained to make the rejected proposal reproducible.
No R2 counterfactual state machine or economic replay is part of the current
pilot/campaign. Reopening requires separate rationale and approval, not favorable
source counts. Its native archetype remains in the all-17 diagnostic census.

**Question:** does an actual breach and close reclaim of a known prior low add
value to the native hourly lower-wick signal?

Baseline: current champion hourly liquidity-sweep identity/gates/fusion/cooldown.

At each setup-hour open, freeze L as the minimum low of the immediately preceding
24 complete UTC hourly bars, excluding the setup bar. Missing constituents make
this evidence unknown. Record the window and the earliest minimum's timestamp
as its anchor; equal prices do not establish independent liquidity pools.

Repair: original identity passes, setup low <L and setup close >L. Equality is
not a sweep or reclaim. Decide only after the setup hour closes. Retain the
native ATR stop, other gates and execution model. No extra volume threshold,
minute confirmation, parent-half veto, touch-count filter or stop change.

This is specifically a prior-24-hour-low hypothesis. Rolling extrema are not
being called a fixed Wyckoff range, observed resting orders or the full trader
sweep system. Fixed higher-timeframe parents are annotations here. R2 is also
not the previously tested minute equal-low strategy.

## R3 Minute native nested breakout and first retest

**Question:** can explicit parent/child objects and an ordered continuation
sequence add value beyond a simple breakout of the same child box?

This research family is separate from the 17 native hourly definitions. It
borrows compression, break and retest concepts; its numerical choices are project
hypotheses, not recovered exact teacher rules. It generates candidates on 5m/1m
clocks and does not require an hourly LC signal.

1. Parent: use the existing guarded `4H_N3` fixed-range constructor. Bind the
   latest intact parent whose available-at precedes the first child bar's open.
   Preserve that exact version. Daily context is recorded separately, not a vote
   based on the copied daily-fusion scalar.
2. Child: at a completed 5m close, form a box from the six most recent complete
   5m candles. Its high/low must be inside the bound parent, and its positive
   width must be <= one quarter of the parent's width. These six-bar and quarter-
   width constants are frozen hypotheses. No volume/RSI/Fibonacci search.
3. Lifetime: an arm-neutral census arms one child per parent lineage at a time.
   Its first eligible 5m
   close strictly above the child high within six subsequent 5m bars is the long
   breakout. A prior 5m close below the child low cancels it. No breakout in that
   window expires it. Do not slide the box after arming.
4. Retest: after the breakout, take the first 5m bar whose low touches or crosses
   the frozen child high. It must close strictly above that high. A close at or
   below it cancels, rather than waiting for a later successful retest. This must
   occur within six 5m bars after the breakout; otherwise expire.
5. Trigger: after that retest candle closes, take the first fully closed 1m bar
   with close above the retest candle's high, within the next five minutes.
   Stop is the retest candle's low, fixed before entry. Any touch of that stop
   before the trigger or fill cancels. No trigger expires the setup.
6. Parent changes: invalidate an unfilled setup only when the bound lineage has
   an observed `source_break_direction=down` transition. Consume it at its hourly
   `available_at`, never at the earlier hour-open or from an unfinished 4H candle.
   A break up is recorded, not automatically rejected. A new parent version does
   not rebind the existing child or retrospectively move its levels.
7. Census re-arm clock: let A be box-arm time and B the first breakout close.
   Reserve the lineage through A+30 minutes if no breakout, otherwise through
   B+40 minutes. Even an early cancellation, trigger, fill or exit does not shorten
   this reservation. Require six entirely new completed 5m bars after reservation
   end before constructing the next box for that lineage. Tie-break simultaneous
   eligible parents by latest available-at, then stable ID. This source-only
   clock is shared; neither trading arm creates its own later child boxes.

Use half-open candle intervals `[open, close)` and reveal their OHLC at close.
The six breakout bars are closes A+5, A+10, ..., A+30 minutes; exclude the box's
last candle. Retest bars close B+5 through B+30, excluding the breakout candle.
If the retest closes at T, trigger bars close T+1 through T+5 minutes, excluding
the retest candle. Equality at the last listed close is included; expiry follows
evaluation of that close. A stop touch in the same minute as a qualifying trigger
cancels first. At a shared timestamp, consume known parent-down transitions and
completed-candle stop touches before a trigger or new fill. A down transition
with available-at equal to fill time cancels the pending fill. Entry-time gap
at/below stop also cancels. Parent invalidation does not liquidate an already
open trade; the fixed bracket manages it.

Raw R3 opportunity ID is `(instrument, parent-lineage/version, box-arm-time,
child-first-open, child-last-close)`, defined at A. All boxes, including no-break
and failed-retest boxes, remain in the arm-neutral census. Both arms use these
same IDs and census schedule. Each book permits one pending or open position;
signals arriving while occupied are recorded as busy without later retroactive
entry. Process existing exits/cancellations before new eligible orders at the
same timestamp, then order simultaneous signals by availability and stable ID.
First report fixed-event counterfactual brackets ignoring occupancy as explicitly
nonportfolio diagnostics; separately replay independent occupied books for the
headline policy comparison. These are not autonomously rearming strategies.

The R3 comparator enters at the first eligible open after the breakout close,
with stop at the frozen child low, and the same costs, sizing and four-hour
decision-relative horizon. R3's stop is at its retest low. This is deliberately
a **whole entry-package** comparison, not a claim to isolate retest selection
from changed stop geometry. The shared-box event study additionally measures
whether breakout direction itself predicts a useful move. No retest-accepted-
only denominator: retain expired, cancelled and missed profitable breakouts.

## Execution and accounting contract

Research only: common long bracket executor, one open position per arm, market
entry at the first exact minute open strictly after signal availability. Model
five seconds of mechanical processing, rounded up to the next available minute;
stress at 65 seconds. On aligned minute decisions these sample one versus two
minutes after availability. Add no agent latency.
Minute OHLC cannot validate sub-minute execution. Receipt/slippage calibration
is required separately before forward promotion.

Primary all-in execution allowance 12 bps round trip, stress 24 bps, charged on
entry notional. This is a scenario assumption, not a measured fee schedule.
Historical funding is not silently zero-valued observed data: until same-venue
funding is qualified, show zero-funding research figures separately and an adverse
8 bps charge at each UTC 00:00/08:00/16:00 settlement while a position is open.
That stress is a hypothetical bound for sensitivity, not a claim that real
funding cannot exceed it. Forward advancement requires actual funding coverage.

For a long fill E, fixed stop S and round-trip execution allowance c in decimal
units, quantity `q = min(100 / ((E-S) + c*E), 50000/E)`. Require finite E>S>0.
Initial risk `R0 = q*((E-S) + c*E)` is immutable and includes the round-trip
allowance, but not uncertain future funding/gaps. Charge half of `c*q*E` on entry
and half on exit. Net PnL is `q*(exit-E) - c*q*E - funding`; per-fill net R is
net PnL/R0. Funding enters the numerator, never retroactively changes R0.
Also report net dollars per $100 intended risk budget per raw opportunity.
No compounding, leverage/account-return claim or cross-arm capital. Stop/gap
losses may exceed R0; the cap may make actual R0 less than $100.

For funding stress, each charge is `.0008*q*E`, on fixed entry notional.
Charge at settlement time tau when `entry_time < tau <= exit_time`: positions
carried into a settlement pay before their same-timestamp exit, while newly
entered positions at that timestamp do not pay for the preceding interval.
This is the declared stress convention. Qualified actual-funding mode must bind
the same-instrument historical settlement/mark-price contract before scoring;
its exact exchange ordering is not inferred from this stress convention.

For R1/R2, freeze the native source-hour ATR stop; target is actual fill plus
twice fill-to-stop distance. Entry expires 15 minutes after signal availability;
exit deadline is 24 hours after the signal decision, not extended by delay.
For R3 and its comparator, target is the same 2R price formula, first-open entry
must occur within five minutes of that arm's signal, and both deadlines are
four hours after the shared breakout decision. Stops at/above entry cancel.
No scale-out, trailing, breakeven or parameter optimization. These are research
brackets, not native exit parity. The old fixed-notional LC outputs are retained
as historical references; they are not directly ranked against new equal-risk
dollar totals. Replaying LC at common economics must receive a separate label.

Check entry-bar extremes; use adverse opening price for a gapped stop and stop
first when a minute could hit both stop and target. Pending stop touches cancel
unfilled orders. At a deadline open, process any adverse stop gap first, then
the time exit; do not consume that minute's later extremes. No synthetic bar
fills across data gaps. Retain unknown/censored outcomes, realized and minute-
close mark-to-market paths, opportunity/position counts and busy skips.

## Calendar and finite execution stages

1. **Source-only pilot:** January 2024, with December 2, 2023 00:00 UTC through
   January 1, 2024 00:00 UTC exclusive as the 30-day prehistory, selected
   by earliest campaign month rather than results. Export all 17 diagnostics,
   R1 counterfactual decisions and R3 lifecycle records; no outcome scoring.
   Verify schema, causality, restart and isolated-state contracts. Save raw output
   before validation, so a receipt-format failure does not require engine replay.
2. **Development census:** decisions from January 1, 2024 00:00 UTC through
   August 31, 2026 00:00 UTC exclusive, all qualifying opportunities, not selected
   teaching cases. Reserve the archive's final day for outcome completion. Report
   August as a partial setup month, not a full-month census. Use frozen artifacts.
   Independent monthly 30-day warmups reproduce legacy source construction but
   do not establish continuous detector state. The implementation must therefore
   carry verified state across months, or explicitly limit results to reset
   semantics and block deployment claims; never silently join cold-start sources.
   Warm detector/parent/census state on prehistory; start all economic books flat
   at January 1, 2024. R1/R2 opportunities belong to their source-hour-close time;
   R3 opportunities belong to box-arm time A, even if breakout is later. This
   common origin controls calendar inclusion, counts and PnL month attribution.
3. **Economic comparison:** score once after source/code/plan locks, using the
   four fixed execution-cost/delay combinations and separate funding stress.
   Calendar reporting blocks are 2024 H1, 2024 H2, 2025 H1, 2025 H2 and
   January–August 2026. They are exposed chronological robustness blocks, not
   pristine holdouts and not a claim of walk-forward optimization.
4. **Forward validation:** only a survivor receives a separately frozen genuine
   unexamined-period or prospective paper protocol. No collector is deployed by
   this design. Before any learned threshold/model is introduced, specify
   training-only fitting and purged walk-forward folds; overlapping label windows
   must not cross training/test boundaries. CPCV is optional later analysis, not
   a way to manufacture additional independent observations from these fixed arms.

Source count inspection cannot change rules/windows. If active R1 lacks replayable
input provenance, or R3 has too little parent/child coverage, report that as the
finding. Do not relax requirements until a pleasing number of trades appears.
The pilot is the next executable unit; it is not authorization for an unbounded
full-history run. Record its wall time and output size before scheduling the
32-month census. Preserve the large existing artifacts rather than rebuilding
them for a small metadata question.

## Decision rules

Before economic reveal, bind the trial ledger including the two active hypotheses,
the parked R2 proposal and their known predecessors. Older unrecorded research means selection bias
cannot be claimed fully corrected. Cluster uncertainty by calendar month with
5,000 paired resamples and seed 20260930; preserve zero-opportunity months and
pair each arm with its own comparator. Correlated parent lineages/overlapping
BTC opportunities are not independent positions merely because books differ.

Report net expectancy, opportunity-normalized value, risk, exposure, trade rate,
drawdown, calendar contributions, worst loss, maximum adverse/favorable excursion,
fees/funding and top-three-winner concentration. Excursions and duration are
outcome diagnostics, never filters selected after reveal. Include rejected winners
as well as avoided losers. Use simple price-break/reclaim controls defined before
scoring; R3 has an explicit same-box control, and R1/R2 retain the native baseline.
The primary incremental estimand is the difference in net dollars divided by
$100 and by the number of common raw opportunity IDs defined before treatment, with
zero contribution for known nonentries and null for unresolved outcomes. Bootstrap
paired monthly sums and opportunity counts together, then recompute this ratio;
do not average individual trade ratios or discard empty/losing months. Require
both arms' outcome windows to be complete before an economic comparison.
Assign a trade's complete net PnL, including subsequent-month exits and funding,
to its raw opportunity's origin month in both arms. Show marked-to-market paths
on their actual timestamps separately. Bootstrap all 32 calendar months as paired
blocks; retain empty months. A resample with no raw opportunities has an undefined
ratio: record it, do not replace with a redraw or zero. If more than 1% of the
5,000 draws are undefined, classify uncertainty as insufficient evidence. Otherwise
use the defined ratios and disclose the excluded count; do not claim exact coverage.

The primary inferential scenario is 12 bps/5 seconds with funding mode bound
before outcome reveal: qualified actual funding if the source contract passes,
otherwise the declared adverse funding stress. Never choose whichever mode looks
better afterward. Zero-funding figures without observations are diagnostic only.
Use the percentile 98.333333% two-sided paired-month interval for the incremental
estimand (lower quantile 0.0083333333, upper 0.9916666667). This is a conservative
three-slot Bonferroni screen (kept conservative despite parked R2), not a
correction for all historic searches.
Monthly resampling and unrecorded prior trials limit inferential validity.

Apply the following decision table in order, recording secondary failures too:

| Condition | Decision |
|---|---|
| Required data, causal/state, error handling or execution contract fails | Block economics for the affected arm; report the specific engineering/input gap. |
| Fewer than 50 completed repair trades or fewer than 12 origin months with repair fills; or bootstrap undefined-count limit fails | Insufficient evidence. Report totals but make no positive advancement claim. These floors are operational, not proof of adequate power. |
| Primary repair net <=0, primary incremental estimate <=0, or repair net <=0 in any of the four cost/delay scenarios under the bound funding mode | Park this frozen hypothesis. |
| Repair net is positive in fewer than three of the five declared reporting blocks, or repair net becomes <=0 after removing its three largest primary net-winning trades | Park for concentration/period fragility under this campaign's conservative screen; this does not prove tail-dependent strategies impossible. |
| Adjusted primary incremental lower confidence bound <=0 | Insufficient evidence; collect genuinely new observations, do not tune exceptions. |
| All preceding checks pass | Eligible for a separately approved forward-paper proposal, not deployment. Actual funding/receipt qualification is still required before that next launch if only stress evidence was available. |

The final campaign decision is advance, park or insufficient evidence, with
no obligation to produce a winner. No historical result alone enables live risk.

## Required implementation checks and deliverables

Before a source launch, require tests for exact 17-name coverage, unsupported
neutral/short handling, structural errors, missing/defaulted versus observed
values, current overrides, complete buckets, UTC boundaries, strict parent
availability, stable level IDs, event expiry, future-append invariance, restart
equivalence, local cooldown, cross-arm non-interference and raw-capture recovery.

Before economics, require entry-bar and gap ordering, ambiguous stop/target,
pending cancellation, deadline handling, immutable initial risk, funding clocks,
notional cap, occupancy displacement, unknown outcomes, exact consumed-file hash
binding, and matched baseline parity. Synthetic software fixtures do not establish
an economic edge. Existing full-suite collection blockers must not be called a
pass; run the relevant focused tests and report their exact scope.

Deliver source census/availability report first, then one comparison report with
per-arm ledgers and retained failures. Update PROJECT and MEMORY at checkpoints.
One bounded software/quant design review is sufficient at this stage; no market
assessor, fine-tuning or 17-agent runtime is part of the study.
