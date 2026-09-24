# LC structure-first decision contract

September 24, 2026. **Design for user review; not implemented or validated as a strategy.**

## Intent and boundary

The user wants an agent that understands *structure within structure*: why an
hourly LC event at a particular location could produce a rebound or continuation,
what minute-scale sequence confirms it, where that interpretation fails, and
where price could reasonably travel. The objective is a falsifiable trade
proposal, not a persuasive story or a guarantee about future price.

This first slice is BTC long LC only, with daily/4H context and hourly/15m/5m/1m
evidence. It preserves both downside-rebound and upside-expansion hypotheses.
Minute bars are execution context, not the separate equal-low minute archetype.
All 17 native archetypes, live orders, fusion configuration and frozen research
remain unchanged. No new assessments, training, replay or deployment are
authorized by this document. The immediate implementation candidate is an
**offline proposal contract and deterministic validator**, not an order service.

## Why this changes the earlier test

The old contract in `scripts/research/lc_context_assessment.py` fixes the stop at
source close minus 2.7 ATR, the target at actual entry plus 2R, the horizon at
24 hours, and notional at $50,000. The agent can enter, wait for one fixed
confirmation level, or reject. It cannot propose a nearer structural destination
or a different invalidation. `conditional_entry.py` resolves that limited menu.

The [142-case comparison](../../knowledge/lc_mechanical_results_2026_09_22.md)
found negative aggregate results for both fixed policies in all four scenarios.
The [missed-rebound review](../../knowledge/lc_missed_rebounds_2026_09_22.md)
also shows that agents already noticed local rebound evidence. This is not proof
that another prompt, more acceptance, or a new stop would improve performance.

Three possible approaches:

- **Constrained structural proposals (recommended):** the agent selects among
  verifiable levels and explains their roles; code validates the proposal. Tests
  more of the intended judgment without granting unrestricted trading authority.
- Revise the old filter: cheaper, but still asks whether to accept someone else's
  fixed trade geometry. Preserve it as a baseline, not the full vision.
- Unrestricted agent trader: too many changing choices to audit or attribute;
  outside this first slice.

## Teachings become questions, not invented rules

Use the versioned [specialist rulecards](../../knowledge/master_specialist_rulecards_2026_09_12.md)
and [primary-source ledger](../../knowledge/trader_primary_source_ledger_2026_09_12.md).
WI03 motivates distinct timeframe roles; M01 motivates location and destination;
B01/B02 distinguish a wick reaching a level from an entry signal. These are
project interpretations with source limitations, not exact teacher-approved
algorithms. A broken 4H range is opposing evidence, not a universal rebound veto.

Require the agent to answer:

1. Which observations are trustworthy, and which are missing or defaulted?
2. Which larger structure existed **before** the setup, and what is its state now?
3. Where is the child sequence inside, outside, or against that structure?
4. Is the thesis exhaustion/rebound or expansion/continuation? What competing
   explanation could account for the same observations?
5. What is already confirmed, what would have to happen next, and what would
   invalidate the thesis before entry?
6. What are the entry, structural invalidation, protective stop, nearest opposing
   level and destination? Why is that destination relevant to the intended horizon?

Fusion components may be cited as provenance-tagged measurements, not as an
unvalidated probability or a reason to override contradictory price structure.
Missing/defaulted funding or OI is not observed neutral positioning. Fibonacci
price/time evidence is optional: require anchors, anchor availability, formula,
units and matching horizon. Unsupported hidden-Fibonacci claims are unresolved,
not extra confirmation. No new fusion or Fibonacci tuning belongs in this slice.

## Division of responsibility

| Code verifies | Agent interprets |
|---|---|
| Instrument, venue, closed bars, timestamps, gaps, source lineage | Which trustworthy observations matter for this thesis |
| A level's price, origin and first availability | Whether that level is support, opposition, invalidation or destination |
| Parent state before setup versus changes during setup | Whether a countertrend rebound or continuation remains plausible |
| Observed sequence order and registered trigger conditions | Why this sequence at this location is meaningful; competing explanation |
| Distances, costs, fill eligibility and external risk limits | Whether available room and contrary evidence justify proposing a trade |

A valid citation proves that evidence exists, not that the interpretation is
correct. Semantic judgment must be evaluated separately from schema validity.

## Input: a bounded, causal evidence packet

Reuse authenticated source packets and publication/citation machinery where
compatible. Introduce a separately versioned projection; do not alter frozen
packets or feed project MEMORY, this dossier, or known outcomes to an assessor.

Code constructs a finite level catalog using a frozen extraction policy. Each
entry contains an ID, price, instrument/venue, timeframe, source field/candle,
observation interval, `available_at`, level kind and lifecycle. Distinguish:

- A confirmed parent bound available strictly before setup start.
- A raw completed-candle high/low: an observed price, **not** automatically a
  confirmed pivot or established support.
- Child levels created within the setup, available by the decision cutoff.
- Confirmed pivots, only after the causal confirmation delay.

The catalog preserves nearest intervening levels; it must not hide an obstacle
because an agent selected a more attractive distant target. No future-derived
anchors, provisional candles disguised as closed bars, or assumed live receipts.
Existing historical sources remain reconstructions, not receipt-authenticated data.

Absent parent structure, broken structure, and unavailable data are different
states. A post-setup parent cannot be relabeled as pre-existing. A lack of a
catalogued daily range does not mean that raw daily price history is unavailable.

## Output: one proposal, not an order

Return one record bound to case, packet, curriculum and contract hashes:

- `decision`: `enter_proposal`, `wait_proposal`, `reject`, or `insufficient_evidence`.
- `thesis`: `downside_rebound`, `upside_expansion`, or `unresolved`.
- Cited parent/child relationship, chronological sequence, supporting evidence,
  strongest opposing evidence, competing explanation and material unknowns.
- For enter/wait: entry condition, structural-invalidation level, protective-stop
  level, destination level and intervening obstacles, all referenced by catalog
  IDs; intended horizon, expiry and cancellation conditions selected from the
  external policy's permitted values. Explain why these form one coherent trade.
- For reject: why no supported proposal is offered; for insufficient evidence:
  the missing facts. Both have no executable plan. They are distinct from an
  invalid response or model/controller failure.
- `execution_authorized: false` in every record in this offline slice.

Level prices cannot be invented or moved to improve reward/risk. V1 proposals
select exact catalog levels; tick rounding is code-owned, and discretionary
buffers are excluded. Structural invalidation is the thesis-failure condition;
the protective stop is an explicit price order. They may coincide but must not
be silently substituted for one another. No martingale, stop widening, scaling,
free-form management or autonomous re-anchoring.

For the first contract, support only immediate-after-processing proposals and
one post-arm completed 1m close above a selected level. Immediate entry still
requires cited predecision confirmation. A future reclaim/retest/hold sequence
is a different trigger type: record it as unsupported, not as an executable wait
or a successful confirmation. This keeps the first implementation small.

The validator checks references, temporal eligibility, allowed operations,
long-price ordering and required contrary evidence. At a hypothetical eligible
fill it must recompute distances and costs, rather than reuse indicative R.
External policy owns maximum entry price, expiry, holding limit, minimum room,
risk budget, notional/leverage cap and cost assumptions; the agent cannot relax
them. A gap above the price cap or destination cancels entry; a pre-entry stop
breach cancels even while the model is responding.

**Scope boundary:** this document does not select or certify production numerical
risk limits. The offline validator accepts an explicit, versioned policy fixture;
missing policy values make a proposal non-executable. Synthetic fixtures are not
economic policy approval. An economic-study protocol must freeze real values
before testing, not optimize them after seeing outcomes.

## Reuse versus new work

Reuse source lineage, immutable packet hashes, one-catalog citations, raw response
capture, timing records and terminal locking. Reuse old resolvers only where
their semantics match. Existing projection and outcome scoring enforce the old
fixed stop/2R geometry; a prompt change alone cannot enable structural targets.

The small next implementation should supply only the new packet projection,
proposal schema/validator and synthetic fixtures. It must reject unsupported
trigger/management types and leave existing modules and experiments intact.
Structural-target execution, risk-normalized sizing and economic accounting need
their own tested adapter before any such proposal can be scored. Do not build
a new campaign controller or 17-agent orchestration system for this contract.

## Verification and eventual evaluation

First verify the contract without market-model calls. Acceptance tests cover
future-known pivots, incorrect parent chronology, absent-versus-missing context,
defaulted data, nonexistent citations, invented prices, hidden intervening levels,
unsupported retest triggers, contradictory decisions, invalid long geometry,
late responses, gaps past entry caps, pre-entry invalidation and missing policy.
A positive synthetic fixture must survive validation; otherwise reject-all could
appear to be working software. Validation cannot certify a profitable judgment.

The [worked examples](../../knowledge/lc_structure_first_examples_2026_09_24.md)
are development illustrations, explicitly exposed and not new model assessments.
All 142 previously scored cases are development data now. Do not rename a subset
as a holdout or require the new agent to accept known winners.

Before another economic study, freeze the curriculum, causal data eligibility,
candidate roster, subtype-specific independent books, policy, budget, latency
assumptions and every comparator. A structural code-only comparator must have the
same candidate-level menu, sizing, costs and execution semantics as the agent;
otherwise better stops or targets could be misattributed to agent reasoning.
Keep the old fixed strategy as a separate end-to-end baseline. Missing/invalid
assessments remain visible and cannot become profitable zero-PnL rejections.

Use genuinely unexposed eligible periods or prospective shadow data, and never
feed outcomes back during a locked evaluation. Chronological walk-forward tuning,
if later authorized, belongs inside development windows, with overlapping trade
labels purged at boundaries and an untouched final evaluation. CPCV is not a way
to undo exposure or establish causality for future-generated teaching material.
Report participation, winners preserved, losses avoided, costs, drawdowns,
concentration and uncertainty—not just win rate or a few attractive charts.

## Deliverable and next gate

This specification and the worked examples finish the approved design step.
No new strategy results, model training or runtime implementation are claimed.
After user review of this written design, prepare a short implementation plan for
the offline contract and its tests. Do not start another paid campaign from this
document. The design choice to approve is **bounded discretion over source-backed
entry, stop and destination**, with code retaining factual and risk constraints.
