# Mechanical LC context and sequence design

October 2, 2026. Written specification for user review. Not implemented, tested,
or approved for trading. The user approved the architectural direction in chat:
make mechanical decisions depend on how evidence fits together before introducing
another market agent. This document makes that direction concrete.

## Objective and boundaries

Build one offline, deterministic LC controller that distinguishes a rebound at a
previously identified range floor, expansion within a range, and expansion above
an accepted range breakout. The same local signal must be interpreted differently
when its parent level, event order, or acceptance state differs. A scalar fusion
score and universal indicator agreement are not the decision mechanism.

The finished research unit must deliver an evidence map, executable decision
records, a correctness report, and a costed comparison against frozen mechanical
controls. A readable explanation or passing software tests alone is not the
economic deliverable. Profitability is a question, not an acceptance assumption.

This is a separately proposed LC study, not an amendment to the completed R3
study. R3 remains parked, R2 remains parked, and exact native R1 replay remains
blocked by missing model artifacts. Preserve all historical artifacts and all 17
production archetypes. No live orders/configuration/fusion changes, market-agent
calls, training, installations, downloads, collector deployment, commits, push,
or PR are authorized by this specification. Stay in the existing research branch.

October 2 continuation: the user explicitly delegated routine progression to a
quant reviewer and software reviewer and requested completion without recurring
approval pauses. Implementation proceeds under their review and the written
implementation plan. Economic launch still requires the source and accounting
checks below; it is not implicit in a source pilot. This authority does not
include live changes, publication, paid market assessments or a strategy search.

## Existing evidence that constrains this design

- The [upside LC baseline](../../knowledge/lc_upside_baseline_results_2026_09_30.md)
  was positive in exposed history, but uncertainty included zero and profits were
  concentrated. Its universal minute-confirmation comparator did not establish an
  improvement. Neither is a certified live strategy.
- The [known-room comparison](../../knowledge/lc_room_validation_results_2026_09_30.md)
  rejected many winners and did not justify a mandatory two-R room gate. This
  design does not introduce that gate, relabel absence as unlimited room, or tune
  a replacement threshold against those outcomes.
- The [12-case agent practice](../../knowledge/lc_practice_results_2026_09_30.md)
  did not demonstrate incremental value. It is not a reason to launch more paid
  assessments before the mechanical proposal is qualified.
- The [R3 results](../../knowledge/archetype_study_results_2026_10_02.md) concern
  a different minute-native box strategy. Its source/accounting infrastructure is
  reusable only through verified interfaces; its losing entry rule is not revived.

These observations informed the design, so the old history is development data.
Freezing this document cannot turn previously examined cases into a fresh holdout.

## Teaching support and project choices

Use the existing [original-source ledger](../../../research_notes/Trader%20origins%20and%20archetype%20fidelity/original_trader_sources.md).
No new exhaustive social-media audit is required to implement this bounded unit.

| Source principle in that ledger | Mechanical responsibility | Limit |
|---|---|---|
| WI03 separates analysis, context and execution timeframes | Bind daily/4H context, hourly LC candidates and minute execution by identity and clock | Does not establish a universal timeframe-agreement gate |
| M01 distinguishes level location and wick/body confirmation | Record the actual level touched and subsequent close events at that same level | Does not supply a universal level constructor |
| CC01 ties acceptance/invalidation to the originating timeframe | A completed 4H close can change 4H acceptance; a 1m close cannot impersonate it | Protective stops remain distinct from thesis invalidation |
| B01/B02 distinguish a possible destination from entry permission | Record destinations/obstacles without treating their existence as a trigger | A possible later destination does not rescue an earlier stop-out |
| Z02 treats indicators as supporting a price thesis | Keep optional evidence separate from necessary setup identity | No assumed numerical weighting or proven predictive contribution |

All exact predicates, N3 pivot construction, lookback, expiry, and economic
parameters below are project research choices. They are not author-endorsed rules.
The output is not a complete Wyckoff accumulation/distribution diagnosis.
Undefined hidden-Fibonacci anchors, Gann clocks and unresolved Phoenix teachings
remain explicitly unsupported. They cannot add permission points.

## Architecture and reuse

Use five small responsibilities, with no LLM in the runtime:

1. Source adapter verifies the saved LC candidate census, candle provenance and
   construction dependencies. It does not select candidates using outcomes.
2. Evidence builder creates immutable levels and timestamped observations,
   retaining both legacy detector state and separately computed acceptance state.
3. Context classifier identifies the applicable scenario or an explicit gap.
4. Decision state machine applies that scenario's sequence and execution rules.
5. Replay/report adapter compares isolated books and explains every nonentry.

Reuse candidates include [LC facts](../../../scripts/research/lc_context_facts.py),
[structure packets](../../../scripts/research/lc_structure_packet.py),
[frozen mechanical plans](../../../scripts/research/lc_mechanical_extension.py),
[parent construction](../../../scripts/research/causal_parent_ledger.py),
[continuous parent adapter](../../../scripts/research/study_parents.py), and
[execution primitives](../../../scripts/research/study_execution.py).
Their existence is not certification of this new composition. In particular,
the R3 executor is not assumed to accept LC plans without a new, tested adapter.

Proposed new research modules are `lc_context_contract.py`,
`lc_context_controller.py`, `lc_context_study.py`, and a bounded command-line
runner, with matching tests under `tests/research/`. Names are planned files,
not claims that code already exists. Do not modify frozen modules to make the
new consumer pass. Keep source-only preparation separate from economic scoring.

## Source population and clocks

The first development population is the complete saved January 2024 through
July 2026 LC census plus the separately captured complete August census, after
receipt verification. Existing reports describe 142 original candidates and
four August candidates. Reconcile those counts and unique IDs from receipts;
do not hard-code a filtered list of successful or convenient cases.

Preserve the existing predecision subtype rule: current hourly close above the
preceding hourly high is an upside-expansion candidate; otherwise a close below
the preceding hourly low or a sweep/reclaim of that low is a downside-rebound
candidate. Other cases remain unresolved. These labels are geometry, not proof
of the market's intent. Preserve unavailable candidates in the census.

For decision time T, the source setup hour opens at S = T minus one hour.
The preceding hour is a comparison observation, not a second invented LC setup.
All candles are completed, same-stream Binance USD-M BTCUSDT observations with
explicit UTC boundaries. Reconstruct daily, 4H, hourly, 5m and 1m views from the
same minute stream, with complete constituent coverage. Never mix a spot/different
venue level into this map without a separately declared transformation.

Freeze candidate and policy identifiers before loading outcome windows. Every
event carries observation start/end, availability, source IDs, constructor
version and level/lineage binding. Historical bar-close availability is an
assumption, not authenticated historical receipt time. No partial HTF candles,
future-confirmed pivots, NaN-as-pass, or implicit forward filling.

The existing native candidate census has monthly warm-up, derivative defaults
and regime fallback limitations. This study is conditional on that reconstructed
census; it is not an exact all-live-signal reproduction. Missing R1 assets do not
prevent a candle-derived context prototype, but neither may those fallbacks be
relabeled as verified macro/regime evidence.

## Parent levels and timeframe acceptance

Use the existing confirmed N3 range-version geometry as an experimental level
constructor, preserving its recorded parameter and helper-source hashes. At S,
select the most recently available valid 4H-derived range version with
`available_at < S` and formation within the preceding 30 days. Include retained
historical versions even if the legacy detector later broke that range; breakout
references must not disappear just because an active-range field was cleared.
Formation means the version's own `formation_hour`, with
`S - 30 days <= formation_hour < S`. Order ties by lexicographically greatest
version ID. Do not try older ranges until one permits a trade.
Freeze that version's L/U boundaries and lineage for this candidate.

Prelaunch source ruling: use the already saved continuous N3 ledger seeded on
December 2, 2023, with its continuous hourly ATR contract, after raw-hour/ATR,
anchor-bucket and helper-hash qualification. Native LC candidates and stops retain
their original monthly 30-day initialization. Do not compare the economics of
monthly and continuous parents to choose between them. Both reviewers accepted
this explicit persistence choice before scoring; seeded absence is not proof of
absence of real structure before initialization.

Apply the same selection to daily context, independently. A daily view is useful
context, not a universal long permission. No qualifying version is `absent`;
incomplete or inconsistent evidence is `unknown`. An absent detector-defined
range does not prove that the market itself lacks structure.

Important source distinction: the existing parent contract declares
`range_update_timeframe = 1h`, even for 4H/daily pivot anchors. Its legacy
`broken_up` label is therefore not a completed 4H breakout confirmation. Preserve
that field as a detector fact and create a separate acceptance event stream.

For a frozen range, acceptance uses only complete candles on the range's
originating timeframe whose open is at or after the range became available:

- close strictly above U: `accepted_above`;
- close strictly below L: `accepted_below`;
- close strictly between L and U: `inside`;
- close equal to a boundary: `boundary`, not a strict acceptance;
- no eligible completed candle yet: `not_established`;
- missing required constituents: `unknown`.

Retain transitions and earlier breaks; returning inside does not erase them.
State is the latest qualifying close relative to the frozen version, not a
probability or a Wyckoff phase. One complete originating-timeframe close and zero
price buffer are explicit first-version research choices, not fitted thresholds.
Record intrabar excursions separately. Do not move the frozen levels to explain
new prices, or promote an old close that preceded level availability.

## Evidence roles

Required facts are source identity/coverage, candidate geometry, the selected
4H level, its acceptance clock, the scenario-specific sequence and valid risk
geometry. Their failure cannot be outweighed by optional indicators.

Daily context and qualified volume/macro/funding/OI/order-flow observations are
separate annotations in version one; they do not change order permission or
sizing. Each records direction where meaningful, age, availability, evidence
status and shared underlying sources. A missing optional observation does not
become either a veto or a bullish vote. Correlated descriptions are not independent
confirmations. Old fusion values are recorded with their actual semantics, not
assumed to mean signed daily/4H agreement.

Destinations and obstacles are references, not guaranteed price magnets. Record
distance, origin and lifecycle without adding the rejected universal room gate.
No mapped obstacle means no known mapped obstacle, not infinite tradable room.

## One proposed context policy

The controller is one versioned policy with separate subtype records. It is not
a search over many combinations. Let H5 be the high of the final complete 5m
candle before T. All price inequalities below are strict unless stated otherwise.

| Candidate and evidence | Scenario | Research entry action |
|---|---|---|
| Expansion; 4H accepted above U; source close above U | Accepted expansion | Immediate plan, subject to availability and risk checks |
| Expansion; 4H inside; source close inside; preceding hourly candle fully inside L/U | Local range expansion | Wait for a post-availability 1m close above H5 |
| Expansion; source close above U; 4H inside, boundary or not established | Parent acceptance unconfirmed | Watch annotation only; no order for this candidate |
| Rebound; source hour low below L; source close below U; 4H inside | Range-floor rebound | Wait for a post-availability 1m close above both L and H5 |
| Rebound with an accepted-below 4H state, or accepted expansion that loses its required acceptance while pending | Invalidated scenario | No entry for this candidate |
| Other known geometry/state combinations | Outside this playbook | No entry; record the specific failed predicates |
| Required source/parent/clock evidence unknown | Insufficient evidence | No entry; distinguish this from economic rejection |

The preceding-hour containment in the range-expansion row means
`L <= prior_low <= prior_high <= U`. No ATR proximity band is fitted. Daily
weakness can label a rebound as counter-context without becoming an automatic
veto. Conversely, an optional bullish score cannot restore a broken required
4H thesis. Daily `inside` is not labeled a bullish daily trend.

Evaluate all applicable predicates and report conflicts rather than use an
undocumented first-match fallback. Subtypes and scenarios are fixed at T. A
stopped/invalidated/expired candidate cannot silently restart as a different
scenario. No fallback from failed expansion to rebound and no migration to
another parent version. Known absence of a parent is `outside_playbook` with
`no_parent_reference`; unknown parent evidence is `insufficient_evidence`.

States are `observed`, `watching`, `awaiting_confirmation`, `eligible`, `filled`,
`invalidated`, `expired`, `outside_playbook`, and `insufficient_evidence`.
At each clock, process source availability and invalidation before entry. A
range-expansion pending plan remains valid only while its 4H state is inside;
a rebound plan likewise requires inside state and no new accepted-below event.
Leaving those requirements invalidates that candidate, rather than changing
families. Boundary equality cannot satisfy a strict trigger.

Retain the 15-minute T-relative entry expiry. Because T is an hourly close, the
next new 4H close is at least 60 minutes away. A not-yet-confirmed 4H breakout
cannot become confirmed within this entry window. Record it as watching/no-entry,
without reserving book capacity; only a later independently supplied LC candidate
can reconsider it. Do not build an impossible pending-acceptance branch, use a
1m close as a substitute, or extend the expiry. Sparse opportunities are a
reportable result, not authorization to loosen the rule.

An admitted immediate or minute-wait plan reserves its own book until fill,
cancel or expiry. Watching, rejected, absent and unknown cases do not reserve it.
No candidate can occupy or release another subtype/arm's book. Record a busy skip
without replaying that skipped candidate later after the position closes.

For minute-wait plans, eligible confirmation candles open at or after the first
minute boundary at or after T plus the processing allowance. Use their completed
close, not their high or an earlier already-completed signal, as the trigger.

For a new required observation at time E, order readiness is no earlier than
max(T plus processing allowance, E plus processing allowance). Fill at the next
available minute open at or after readiness, strictly before expiry. Apply this
same observation-to-order convention to the unconditional-wait control below.
Recheck the bound scenario and the protective stop before filling. A stop breach
before fill cancels the candidate. Gap/ambiguity handling is conservative and
identical across arms; no fill at an already elapsed candle close.

At the executable opening price, accepted expansion still requires price above U;
range expansion and range-floor rebound require price strictly inside L/U.
Otherwise cancel with `entry_location_changed`, not an invented HTF invalidation.
This checks the chosen scenario's entry location, not a universal two-R room gate.

After fill, the first experiment holds exits constant. Record further structural
changes as diagnostics, but do not introduce adaptive exits, widen stops or
reclassify a stopped trade as a later successful thesis. This isolates the
complete entry-controller policy, not every individual component's contribution.

## Decision record

Each transition records candidate, subtype, scenario and frozen parent IDs;
as-of time; necessary predicate results; supporting, opposing, missing and
irrelevant evidence IDs; trigger/expiry; protective stop; indicative destinations;
and the exact reason for its action. Predicate values are pass, fail, unknown
or not applicable. Observed facts and policy interpretations are separate fields.

A readable case card is generated from these records, not invented narrative.
It must explain why a bearish daily annotation did not veto a local rebound,
or why a good-looking minute trigger could not override invalidated 4H context.
Every output has `execution_authorized = false` and a policy/source seal.

## Verification and economic finish line

First establish behavior with synthetic and source-reconstruction tests:

1. Same local candles and different valid parent context produce the specified
   different scenarios/actions; stale or unbound context cannot do so.
2. A 1h break of a 4H-derived range does not become a 4H acceptance. Equal-boundary
   closes and incomplete candles do not qualify either.
3. A wick without the required reclaim cannot enter the rebound branch. Sequence
   events reference the same frozen level, with correct availability order.
4. Changing optional indicators, duplicating correlated facts, or marking an
   optional feed unavailable does not change version-one permission.
5. Future append, chunk/restart boundaries and processing order cannot rewrite
   earlier IDs or decisions. A new parent cannot repair an old candidate.
6. Pending invalidation wins over simultaneous entry; stop breaches, expiry,
   observation latency and source gaps have explicit terminal outcomes.
7. All census cases remain represented, including absent parents and unresolved
   subtypes. Native-source limitations remain attached after projection.

Then run a source-only development replay over the complete saved census, with
no outcome-based revision of this policy. Produce coverage by subtype/scenario,
clock failures and full decision records. Verify raw-minute aggregation and
parent-source bindings; a valid JSON seal alone is not source qualification.
Source-only success is an intermediate checkpoint, not completion of the study.

For economic comparison, predeclare three arms: immediate mechanical LC,
unconditional H5 minute confirmation, and the context controller above. Keep
rebound and expansion in separate single-position books; unresolved cases stay
in the census and coverage report. Do not pool subtypes to conceal a losing arm.
All arms see the same candidate IDs. Pending reservations and busy skips are
modeled per book, not calculated by deleting rows from a finished trade table.
Neither control requires parent coverage, existence, acceptance, containment or
the context arm's fill-location checks. Immediate entry does not require H5;
unconditional minute confirmation does. Classify subtype before validating ATR
and stop so a missing risk input cannot erase known geometry from its denominator.
A verified insufficient-evidence abstention is a zero nonentry; missing execution
evidence after admission is an unknown outcome, not zero, and can poison later
capacity decisions. Unresolved geometry remains visible outside both subtype books.

Use the same source-close minus 2.7 ATR14 protective stop in all arms, actual-fill
2R target, and T-relative 24-hour exit deadline. These are comparison brackets,
not a claim of faithful trader-style management or exact live scale-outs.
Use $100 intended stop-plus-roundtrip-cost risk with a $50,000 notional cap;
for entry price P, stop D and fractional roundtrip rate c, quantity is
`min(100 / (P - D + c * P), 50000 / P)`, requiring P > D > 0. Initial R is
quantity times `(P - D + c * P)`. Funding is charged separately and can increase
the loss beyond that initial R, as can gaps. Do not interpret old fixed-notional
dollar totals as this equal-risk book's returns.
The target uses price-distance R, `P + 2 * (P - D)`, not cost-inclusive R.
Total fees are `c * quantity * P`, split equally between entry and exit; both
use entry notional. Stops take precedence when both barriers occur within one
minute; opening stop gaps exit at the opening price and target gaps at target.
Intraminute barriers are recorded at the completed-minute close. A settlement
coincident with that modeled exit is charged first. Deadline exits use the
deadline opening price after any completed prior-bar barrier is processed.

Freeze 12/24 bps roundtrip costs and 90/300 second processing scenarios. These are
historical research allowances, not measured live latency. Include a zero-funding
reference and adverse 8 bps notional funding at UTC 00:00/08:00/16:00 settlements
with entry < settlement <= exit. Label the latter stress, not observed funding.
Apply identical assumptions to all arms and report both sets, not whichever wins.
The primary case is 12 bps/90 seconds with adverse funding stress; zero funding
is a sensitivity reference. Stress settlement charges use entry notional.
Fractional quantities and archive prices are retained without venue lot/tick
rounding, matching the research nature of the comparison; impact and an actual
funded margin account are not modeled. These limitations preclude live execution
certification. Funding accounting, gap fills and executor parity must be tested
before economic launch. The added per-observation processing convention is part
of this new common contract, not exact parity with the older wait-control clock.

Report net PnL, cost-inclusive R, marked-to-market drawdown, trade frequency,
calendar contributions, exposure, missingness, winners preserved/missed and
losers avoided/introduced. Include zero nonentries in per-supplied-candidate
measures. Unknown economic outcomes are not zero; they block complete totals.
Report hypothetical independent books, not a funded combined account.

Use paired calendar-month resampling over the full declared calendar, including
empty months, for descriptive uncertainty. Freeze 5,000 draws, seed 20261002 and
95% intervals before scoring. This does not correct prior strategy selection or
establish independence across adjacent months. No optimizer or fitted weights
exist in version one, so do not label chronological replay as walk-forward
training or CPCV. Future fitted models require purged training/validation and
their own trial registry; those methods do not make exposed history untouched.
The calendar contains all 32 months January 2024 through August 2026; attribution
uses candidate decision month. Primary improvement is the sum of paired context
minus immediate net dollars divided by every supplied candidate of that subtype.
Resample the same month indices on both sides and record undefined zero-count
draws. A dollars/100 display is budget normalization, not realized-risk R.

The economic report ends `unsupported`, `inconclusive`, or `worth_forward_test`.
Negative primary net performance is unsupported; positive but sparse, uncertain
or non-improving evidence is inconclusive. `worth_forward_test` requires positive
primary net and positive paired per-candidate improvement over immediate entry
for that subtype, a positive lower descriptive interval for that improvement,
at least 50 fills across 12 distinct months, and no negative aggregate under the
declared cost/delay stress scenarios. These conservative project screening rules
are not a power calculation, multiple-testing correction or live certification.
Do not extend samples or change gates simply to cross them.
Zero primary net or missing/undefined evidence is inconclusive. Stress screening
uses all four cost/delay scenarios under adverse funding; all zero-funding
counterparts are reported. The unconditional-wait comparison is required reporting,
not an additional undeclared promotion gate.

Any final edge claim requires genuinely unexamined or prospective same-stream
data with a separate calendar/exposure record. None is certified available now.
Downloading later dates does not make previously discussed September trades
unseen. No forward collector or new download starts under this document.

## Deliverables and remaining dependencies

| Deliverable | Completion evidence |
|---|---|
| Mechanical evidence and policy contract | Reviewed definitions, clocks, source limits and one frozen policy |
| Implemented controller | Focused behavior tests plus deterministic decision records; no live hooks |
| Complete development source replay | Verified receipts and coverage for every supplied case, with source limitations |
| Economic study | Matched isolated books, independent arithmetic/outcome checks and one finite verdict |
| Future validation decision | Explicit new-data/exposure contract, or a clear statement that certification remains unavailable |

Local dependencies remain the minute archive, prior source receipts, recovered
parent helper files and saved LC census. They are not all supplied by GitHub.
Verify exact hashes and consumed paths before implementation's source launch;
do not retrain missing native models or substitute a different market stream.

The repository knowledge graph records incomplete coverage of recent research.
Use actual source and dated results for authority; no graph rebuild is required.
The next step is the bounded implementation plan and reviewed implementation,
then source qualification and economic replay. No economic outcome is claimed by
this design. The clarifications above were adopted from independent quant review
before new outcome scoring; the original design hash was
`c8e696460dc8a7dad56940bdd3cfe4ec162f65df730e820d8b656968cc07fa9d`.
