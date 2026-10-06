# Seeing eye market map and first playbook

October 1, 2026. Status: architecture approved in conversation; this written
specification awaits review. Revised after the user required proper backtesting
of ideas: the source-only unit below is a prerequisite, not the campaign finish
line. No implementation, pilot or economic result is claimed. This is developer
documentation, not an outcome-hidden assessor packet.

## What we are building first

The campaign must answer whether a frozen trading idea improves net results over
its declared comparator, with uncertainty and failure conditions reported. Build
one offline, replayable market case that connects larger structure, smaller
structure, location, sequence and an entry proposal. Then produce the same record
for every qualifying opportunity in the already selected source-only pilot.
The immediate question is whether the system can describe and enforce a complete
setup using information available at the time. Profitability is a later, separate
test; a convincing explanation does not establish an edge.

The design is one shared timestamped market map and one initial playbook consumer.
It is not a rewrite of 17 strategies or a committee of model traders. The map can
represent many teachings without treating every teaching as an entry requirement.
Existing archetypes keep separate decisions, cooldowns, books and economics.

The first consumer is the existing R3 minute-native nested breakout and first
retest hypothesis: fixed 4H parent, smaller 5m box, breakout, first retest and 1m
trigger. This is a new research family, not renamed liquidity compression and not
one of the 17 native hourly definitions. It was chosen because it makes structure
within structure testable, not because it has demonstrated profitability.

The [bounded study specification](2026-09-30-archetype-repair-discovery-design.md)
remains authoritative for R3 rules, execution and economics. This document adds
its evidence representation and defines the first implementation unit. R1 remains
an active, unimplemented hourly repair; R2 stays parked; LC and its failed room
filter experiment remain unchanged. Hourly and minute research retain equal
standing. Completing this R3 unit does not complete the broader pilot, which also
requires R1 decisions and a complete all-17 diagnostic ledger.

## Backtesting is the campaign deliverable

The case card is a debugging witness, not a strategy evaluation. The endpoint is
one reproducible economic comparison report under the existing bounded study,
or an explicit data/engineering blocker. Do not let an expanding map, agent
curriculum or collection of examples substitute for that result. Implement only
the evidence needed to run the frozen comparisons correctly.

| Frozen idea | Comparator | Question the backtest answers |
|---|---|---|
| R3 nested breakout, first retest and minute trigger | Immediate breakout of the same fixed child box | Does the complete later-entry package improve net value, after counting missed winners, cancellations and changed stop geometry? |
| R1 upward EMA permission for long trap within trend | Unchanged native TWT identity and gates under common research execution | Does the single directional requirement improve the isolated policy, including changed cooldown and busy-position effects? |

Both comparisons use their complete, pre-treatment raw-opportunity populations,
not only executed trades or handpicked examples. Keep economic state separate.
The development window remains `[2024-01-01, 2026-08-31)` UTC, with the final
archive day reserved for outcomes. January is the initial engineering check,
not the entire backtest or evidence of an edge. After it passes, the next research
milestones are the fixed full census and economic replay, subject to the existing
code/plan/resource locks; not another source-audit cycle without a specific gap.

Replay chronologically using only completed, available observations. Use the
declared same-venue stream, next-eligible-open entries, separate occupied books,
equal intended risk, adverse gap handling, stop-first ambiguous candles and the
frozen management policy. Report actual qualified funding or the declared funding
stress; do not treat missing funding as observed zero. The 12/24 bps cost and
5/65 second delay combinations are stress assumptions, not calibrated fills.

Report net expectancy and value per common raw opportunity, trade counts and
participation, drawdown, exposure, worst losses, uncertainty, calendar results,
and how much profit depends on the largest winners. Apply the existing paired
monthly uncertainty calculation and multiple-trial screen without changing their
thresholds after reveal. The 50-fill/12-origin-month floors are minimum evidence
requirements, not proof that a sample is statistically sufficient. Final status
is advance to separately approved forward testing, park, or insufficient evidence;
missing required data is a blocker, never silently an economic pass.

These first rules are fixed, so the chronological replay is not called walk-forward
optimization. If a later campaign tunes thresholds or learns weights, fit only on
earlier training data and evaluate on later periods, purging overlapping outcome
windows. CPCV may be a supplementary diagnostic; it does not make previously
inspected history unseen. A genuine new-period test is still required. Ordered
splits avoid future-to-past training leakage, but a generic time split alone does
not implement trade-label purging ([time-series split documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)).
Recording all tried ideas matters because repeated selection on the same history
can produce false discoveries ([Bailey and colleagues](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)).

Current implementation gap: R3 is specified but not implemented. Existing
`conditional_occupancy` uses fixed notional, a different initial-risk convention,
and pending-capacity rules; it is not the new equal-risk/funding executor. The
legacy minute-sweep simulator also has its own fixed lockout semantics. Reuse
qualified pieces, but implement and test a separately versioned study executor
and paired report before claiming a valid new backtest. No paid market assessor
is required. Fib/Gann additions would be separately frozen hypotheses, not extra
variants inserted into this campaign after viewing R1/R3 results.

## How the teachings become one view

Keep three separate records: what a source actually teaches, our numerical
implementation choice, and evidence that the choice works. Do not promote one
category into another. Source scope and gaps come from the completed
[teaching ledger](../../../research_notes/Trader%20origins%20and%20archetype%20fidelity/original_trader_sources.md)
and [active-code audit](../../../research_notes/Trader%20origins%20and%20archetype%20fidelity/active_code_fidelity.md).

| Layer | Meaning in the shared map | Role in the first playbook |
|---|---|---|
| Wyckoff and larger structure | Range identity, boundaries, originating timeframe, observed events and separately labelled detector interpretation | Require the existing fixed 4H parent. That constructor is not proof of a complete Wyckoff phase or institutional intent. Daily and other Wyckoff context remain annotations. |
| Multiple timeframes | Explicit relationships between the parent, child box and execution candles | Require the parent to exist before the child starts. Do not require every timeframe indicator to agree. |
| Location and sequence | Stable level IDs and ordered break, retest and confirmation events at those same levels | Hard setup identity. A similar indicator elsewhere cannot substitute for the required event. |
| Fib price and time | Anchors, direction, origin and confirmation clocks, ratios, units, tolerance and expiry | Unsupported where those definitions are absent. Legacy values may be labelled uncertified; no new gate or hidden-level claim. |
| Gann timing | Declared origin, calendar/session convention, conversion formula, tolerance and reset | Unsupported until defined and separately tested. Naming a cycle does not establish a timing signal. |
| Macro and derivatives | Provenance, observation time, freshness and missing/defaulted status for each feed | Context only. Defaults are not observed neutral conditions. R3 has no new fusion gate. |
| Destination and risk | Potential obstacles, proposed entry, pending cancellation, structural invalidation and protective stop as separate objects | Retain the frozen R3 stop and later research bracket. No new room >=2R veto; unknown room is not infinite room. |

The source-backed common idea is location and timeframe before execution, not a
universal formula. Wyckoff Insider distinguishes higher-timeframe analysis from
execution ([WI03](https://x.com/Wyckoff_Insider/status/2000223306620846408)).
Moneytaur describes waiting at identified levels and distinguishes wick movement
from confirming bodies, without defining the level constructor
([M01](https://x.com/Moneytaur_/status/1819660778359644193)). Crypto Chase ties
acceptance and invalidation to the timeframe that created the idea
([CC01](https://www.youtube.com/watch?v=MRMikMIlOwM&t=392s)).

Bojan's possible weekly-open destination is not permission for an immediate entry
([main](https://x.com/Bojan_618/status/2000564037780946977),
[qualification](https://x.com/Bojan_618/status/2000564041102901639)). ZeroIka treats
indicators as support for a price-action and fundamental thesis
([Z02](https://x.com/IamZeroIka/status/1779921797434982462)). These inform the
separation of context, trigger and risk; they do not endorse our R3 constants.

The inspected [Fib post](https://x.com/Wyckoff_Insider/status/1949423128217633091),
[Gann post](https://x.com/Wyckoff_Insider/status/1949399442873786607) and
[hidden-level post](https://x.com/Moneytaur_/status/1717106022265897019) do not
provide reproducible construction rules. Phoenix remains unresolved. Mancini's
saved ES teachings are later-transfer evidence, not BTC validation; the R3 retest
stop and fixed research target do not reproduce his whole-structure stop and
level-to-level management. Filling these gaps is separate source work, not a
condition for inventing a rule or silently expanding this implementation.

Correlated descriptions of the same price move are not independent confirmation
votes. A scalar score cannot compensate for a missing parent, wrong level or
failed sequence. Existing native fusion behavior is preserved, not redesigned.

## Components and responsibility

The flow is source receipts to immutable evidence objects, then a playbook state
machine, then a small as-of case packet and a readable case card. The map is a
versioned file contract, not a new database service or dashboard.

1. A source adapter qualifies same-stream completed candles and any optional
   context. It preserves raw captures before projection and records unavailable
   inputs rather than manufacturing them.
2. A causal map builder maintains parent, child, level and event relationships
   with explicit availability clocks. A source registry connects each rule to
   source teaching and project choices.
3. The R3 state machine consumes only its declared required evidence. It records
   attempted boxes, eligible raw opportunities and every subsequent state change.
4. A deterministic projector emits the minimum evidence needed to explain one
   decision. A case card presents those same facts, with no invented narration.
5. A coverage and validation report reports missing evidence and causal failures.
   It contains no performance scoring in this first unit.

Later, a separately approved shadow agent could interpret a frozen packet,
explain disagreements and propose wait or reject decisions. It cannot rewrite
observations, override required evidence or enable orders. Its value would be
measured against the mechanical baseline. No model call is needed to build or
validate this first unit. Persistent teaching files are not persistent model
weights or proof that an agent has learned profitable judgment.

## Evidence and case contracts

Planned schema names are `seeing_eye_market_map_v1` and `seeing_eye_case_v1`.
Implementation must freeze their required fields and validators before pilot use.

Every evidence object carries a stable ID, type, instrument, venue/data-stream ID,
source and constructor version, rule references, value and units. It also carries
origin time, source interval, `available_at`, and references to the evidence used
to construct it. Parent, child, level and event objects retain their own IDs;
numeric similarity does not establish identity.

Separate these status axes rather than overloading a score:

- Evidence status: observed, derived, defaulted, missing, unsupported or invalid.
- Quality: completeness, freshness, reason codes and semantic qualification.
- Scope binding: linked to this lineage/level, timeframe-only, or unbound.
- Predicate result: pass, fail, unknown, not evaluated or not applicable.

Derived does not mean unreliable, but it must name its inputs and constructor.
A Wyckoff scalar with only timeframe scope is not evidence for a specific parent.
Undefined Fib/Gann semantics are unsupported, not a favourable zero. Missing
optional annotations do not reject an otherwise valid R3 setup. Missing required
data blocks certification and, eventually, affected-cohort economics.

All market observations in a packet must have `available_at <= as_of`. Parent binding is stricter:
availability must precede the child's first open. Pivot origin and confirmation
are separate clocks. Historical bar-close availability is explicitly labelled
`historical_bar_close_assumption`; it is not measured live feed receipt.
No partial aggregate candles, NaN-as-pass or implicit forward fill. Unknown
numeric values use null and an explicit reason. Anchored bounds never move in an
old object; later evidence creates new versions or appended lifecycle events.

Missing/unsupported objects may have null clocks with a reason; they cannot
support a passing predicate. The fixed teaching curriculum is research policy,
not a market observation. Preserve its publication/read dates separately and
exclude outcome-bearing example charts/text from blind packets. Applying this
2026 design, partly informed by later publications, to 2024 data is retrospective
research on exposed history, not a historically available policy or fresh holdout.

Use the canonical raw R3 opportunity tuple: instrument, parent lineage/version,
box-arm time, child-first-open and child-last-close. Case identity additionally
binds its decision time and protocol version. Whole-archive checksums belong in
the run manifest, not event identity: appending future data must not change
pre-existing opportunity IDs. Hash each packet's selected predecision content;
exclude volatile write times and future run totals from its semantic identity.

The case record contains the following named sections:

| Section | Required content |
|---|---|
| Identity | Schema, playbook and source versions; case/opportunity IDs; instrument and as-of clock |
| Trusted inputs | Receipt references, availability assumptions, missing inputs and quality flags |
| Larger structure | Frozen parent ID/bounds, creation and availability, state known now, separately scoped daily/context evidence |
| Child and location | Six candle IDs, frozen box/level IDs, containment and width results |
| Sequence | Ordered events, linked levels, stage and each required predicate result |
| Context | Optional teaching-linked observations and explicit unsupported concepts; never implicit permissions |
| Proposal and risk | Trigger instruction, pending-entry policy, structural cancellation, stop and eventual bracket references; unavailable prices remain null |
| Decision | Deterministic state and reasons, next required evidence, next deadline and `execution_authorized=false` |
| Citations | Resolvable evidence/source references with no post-decision facts |

Native signals, when available, are namespaced diagnostics rather than R3 votes.
Preserve all-17 pre-dedup observations; selected winners are not the population.
The existing observer records aggregate gates, not every gate's evidence. Do not
label it complete instrumentation. The runner's outer dynamic threshold is not
evaluated by that observer; record this honestly. Its bypass does not remove the
inner native fusion floor. For R3 those native fusion gates are not applicable.

## Frozen first playbook

The following restates R3, not a new parameter search. The bounded study remains
the rule authority; any conflict must stop implementation for reconciliation.

- Bind the latest intact guarded `4H_N3` parent available strictly before the first
  child candle opens. Freeze its version and bounds. A newer parent never rebinds
  an existing child.
- At a completed 5m close A, form the child from the six most recent complete 5m
  candles. The box must lie inside the parent with positive width no greater than
  one quarter of parent width. Do not slide it after arming.
- Breakout B is the first 5m close strictly above child high at A+5 through A+30
  minutes. A prior close below child low cancels; no breakout expires the setup.
- The first subsequent 5m touch of child high must close strictly above it. That
  first touch closing at/below the level cancels; do not wait for a nicer retest.
  The retest must close at T between B+5 and B+30 minutes.
- Trigger on the first completed 1m close above that retest candle's high, from
  T+1 through T+5. Fix the stop at retest low. A stop touch before trigger or fill
  cancels, including a touch in the qualifying trigger minute.
- A bound-lineage `source_break_direction=down` transition cancels an unfilled
  setup only when known at its hourly `available_at`. A break up is recorded, not
  a cancellation. The causal interpretation proposed here for written review is
  that a down transition known by arm time during child construction makes that
  attempted box ineligible; do not ignore an already known break.
- Keep the common source-census reservation through A+30 without breakout or
  B+40 after breakout, even after early cancellation/trigger/fill/exit. Require
  six entirely new completed 5m bars after reservation end before the next box
  for that lineage. Preserve the specified availability/stable-ID parent tie-break.

Candles use half-open intervals and reveal OHLC at close. Include each last
listed window close, then expire. At a shared timestamp, known parent-down and
completed-candle stop touches take priority over a trigger or pending fill. A
fill-time down transition or opening gap at/below stop cancels the pending entry.
Parent invalidation does not liquidate an already open trade; later economics
use the frozen bracket instead. The source-only unit does not create open trades.

Track attempted-box failures separately from raw eligible opportunities. Once
armed, the lifecycle is `armed` to `awaiting_retest` to `awaiting_trigger` to
`triggered`, with `cancelled`, `expired`, `data_blocked` or `right_censored` where
appropriate. Here `armed` means waiting for breakout. Each transition records its
clock, evidence and reason. Triggered means a proposal exists, not filled or
authorized, and completes the core source sequence in this unit. No pending or
open trading book is instantiated. A later pre-fill validator must append any
cancellation rather than rewrite the as-of trigger packet; that validator and
economic replay are not implemented by this unit. Data-blocked and censored
records are not ordinary setup rejections or evidence of a losing trade.

Preserve all no-break, failed-retest and otherwise unfilled opportunities. Economic
arm occupancy cannot change which later boxes the source census constructs.
The eventual same-box comparator enters after breakout with child-low stop;
R3 enters after retest/trigger with retest-low stop. Their contrast is a whole
entry-package comparison, not pure filtering on an identical entry and stop.

No new execution policy is introduced here. Later economics retain the original
five-second/65-second delays, 12/24 bps scenarios, funding qualification/stress,
fixed risk sizing, 2R target and shared four-hour breakout-relative deadline.
The target is a research bracket, not a forecast of where price intends to go.

## Existing code to reuse and boundaries to preserve

The [parent ledger](../../../scripts/research/causal_parent_ledger.py) supplies
versioned ranges and `parent_asof(..., strict=True)`. Its existing `bind_parent`
uses sweep terminology; an R3 adapter should use an explicit child-start binding
policy without relabelling a breakout as a sweep or changing frozen behavior.

The [signal observer](../../../scripts/research/engine_signal_replay.py) captures
native diagnostics without rerunning detection. It excludes runner entry/book/
execution. Preserve its behavior; fill remaining all-17 instrumentation gaps in
the broader pilot work, not by claiming the current output already supplies them.

The [LC packet projector](../../../scripts/research/lc_structure_packet.py) offers
whitelist and citation-validation patterns, but assumes hourly LC timing and
omits optional Fib/derivatives. Use a separately versioned R3 projector, not a
silent generalization of frozen LC packets. The
[LC pre-entry check](../../../scripts/research/lc_structure_preentry.py) contains
LC-specific room, proposal and expiry rules; do not plug it into R3 unchanged.

The implementation plan should define small adapters, validators, the R3 source
state machine and case/report projection, with focused tests. No new library,
graph database, model runtime, fine-tuning, live gateway or indicator rewrite is
needed for this unit. Existing semantic defects remain labelled legacy evidence;
do not fix native archetypes under cover of building a map.

## Source pilot and deliverables

Use the frozen same-venue Binance USD-M BTCUSDT minute source identified in the
[data inventory](../../knowledge/archetype_repair_scorecard_2026_09_30.md).
Verify the actual consumed archive, helper and configuration hashes before use.
Local archive/helper absence is an explicit dependency failure, not permission
to download substitutes or manufacture missing derivatives.

The source pilot remains January 2024: raw opportunity arm times in
`[2024-01-01 00:00, 2024-02-01 00:00)` UTC, with prehistory
`[2023-12-02 00:00, 2024-01-01 00:00)`. Warm source, parent and census state on
prehistory, not positions. Carry state through the boundary. At the fixed pilot
end, retain incomplete pre-trigger lifecycles as right-censored; do not silently
extend the calendar or call an unfinished case expired. Later economic comparison requires
complete outcome windows under the existing protocol.

The first readable example is the earliest raw armed opportunity, irrespective
of whether it later triggers. If there are no eligible opportunities, deliver
the attempted-box census and reasons. Do not relax rules or select a later winner.
Each example has separate as-of snapshots; a later lifecycle event must not leak
into an earlier packet. Optional missing sources do not justify a new broad crawl.

Save to a new collision-safe run directory, retaining previous artifacts:

1. Manifest binding schema, protocol, teaching, source/helper, code and config
   versions and exact consumed-file hashes.
2. Raw source captures, attempted-box census, all raw R3 opportunities and state
   events. Save expensive captures before validation so projection fixes can
   reuse them. Developer-only full-run totals stay outside market-role packets.
3. As-of map snapshots, validated minimal case packets and readable case cards.
   Markdown/JSON is sufficient; no dashboard build is part of this deliverable.
4. Coverage and validation report, with explicit unknowns, lifecycle counts,
   restart results, runtime and output size. No PnL or winner/loser scoring.

Reuse qualified existing captures where equivalent, not merely similarly named.
Keep full-run manifests, PROJECT, MEMORY, known-outcome reports, future events,
trade duration, MFE/MAE and PnL out of assessor-safe packets. A future market role
may receive only the versioned teaching subset and evidence available at its
decision. Developer continuity and the teaching registry serve different roles.

## Acceptance and stopping conditions

Before the source pilot, focused automated fixtures must verify:

1. Strict parent-before-child availability, complete aggregate buckets and the
   separation of pivot origin from confirmation. No future or partial evidence.
2. Stable parent/child/level identity through breakout, first retest and trigger;
   replacement parents cannot retroactively improve the setup.
3. Exact six-bar geometry, quarter-width limit, equality/window boundaries,
   first-touch cancellation, stop-first and parent-down event priority.
4. Shared census reservation and re-arm rules independent of simulated arm state.
5. Observed false versus missing/defaulted/not-evaluated evidence, required-data
   error propagation and optional unknowns that do not change R3 permission.
6. Future-append invariance of existing semantic IDs/packets and restart
   equivalence across lifecycle stages and the prehistory/month boundary.
7. Unchanged native diagnostics and isolated state, with pre-dedup populations
   retained and unsupported native short/neutral economics not converted to longs.
8. No mutation of raw inputs, production configuration or frozen artifacts.
9. Strict packet whitelisting and citation resolution, rejecting future/outcome
   fields and unrelated metadata rather than passing an entire feature store.
10. Triggered versus filled separation, immutable as-of snapshots, explicit
    censored/data-blocked states and execution authorization always false.
11. Honest zero-opportunity and missing-local-dependency results, with no synthetic
    fills, network/model calls or automatic alternate strategy.

The unit is finished when those checks pass and the fixed pilot produces an
auditable record of its coverage and cases, including failure or no-case results.
Data or source failures block affected certification; do not relabel them a
strategy failure, repair silently after reveal or expand history to force success.
Software checks certify representation and chronology, not predictive accuracy.

After this unit, finish the remaining R1/all-17 source-pilot requirements under
the bounded study. Only then consider its already specified, separately locked
economic comparison. Preserve rejected winners and avoided losers in that later
comparison, use the common raw-opportunity denominator, and require genuine
forward evidence before deployment. Neither this spec nor a source-pilot pass
authorizes that launch, paid assessments, production changes, commit/push or PR.

## Current approval boundary

This written design is the next deliverable, not evidence that the map exists.
Following its review, write one bounded implementation plan with dependencies,
test commands, exact files and a source-run resource estimate. Keep implementation
and source launch within their approved stages. Record actual progress and local
dependencies in PROJECT and MEMORY so another CLI can continue without chat
history. No implementation work or experiment is running at this checkpoint.
