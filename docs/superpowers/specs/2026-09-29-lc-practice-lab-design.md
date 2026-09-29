# LC historical practice lab

September 29, 2026. **Scope approved; written integration design for review.
Not implemented, not a completed backtest, and not permission to launch paid roles.**

## The question and the deliverable

Can an agent using our trader teachings interpret structure within structure and
make better LC trade proposals than a simple mechanical rule given the same
evidence and constraints?

Build one reusable, offline historical practice workflow. Its first run has12
real BTC LC setups. The user gets a self-contained HTML report: what information
existed, what each decision was, charts with entry/stop/target, what price did,
and the comparison against code. Machine-readable JSON and a short Markdown
summary accompany it. A test count is engineering evidence, not the study result.

LC remains the first proving ground for the larger master-trader vision. This
does not test all17 archetypes or certify a profitable strategy. Rebound and
expansion are reported separately. Minute data is nested LC execution evidence,
not the separate minute equal-low archetype. No live orders/configuration/fusion
changes, new trader-source research, new training, autonomous tuning or dashboard
service. Preserve every previous experiment and its original responses.

Chosen approach: historical replay plus a static inspectable report, reusing
the existing packet, proposal and outcome functions. A live shadow service would
provide new observations but requires waiting for signals and integrating live
receipts; it is later work. A full interactive dashboard adds a separate product
surface and is not needed for this first practice run.

## Exact first practice roster

Take the first12 cases sorted by `(decision_time, case_id)` from the existing
20-case `results/lc_consolidated_2026_09_15/judgment_v1/evidence/roster.json`.
This is not the first12 from the entire142-case census. Retain the original
selection limitations of that20-case source. Do not balance by outcome, replace
failures or add cases to obtain a positive result.

All timestamps below are decision times in UTC, in2026. Full case IDs are
`hourly-lc:<YYYY-MM-DDTHH:MM:SS+00:00>`.

| # | Decision | Source-derived candidate subtype |
|---|---|---|
| 1 | January20 06:00 | Downside rebound |
| 2 | January25 09:00 | Downside rebound |
| 3 | January29 16:00 | Downside rebound |
| 4 | January31 15:00 | Downside rebound |
| 5 | February23 02:00 | Downside rebound |
| 6 | February25 02:00 | Upside expansion |
| 7 | February28 07:00 | Downside rebound |
| 8 | March7 20:00 | Downside rebound |
| 9 | March8 23:00 | Downside rebound |
| 10 | March22 22:00 | Downside rebound |
| 11 | May2 22:00 | Upside expansion |
| 12 | May3 23:00 | Upside expansion |

The9/3 split is descriptive, not a claim that three expansion cases establish
anything statistically. Keep source-derived subtype as the grouping key; record
an agent's differing thesis separately rather than regrouping after its result.
These cases have been examined before: **exposed practice/development**, never
an untouched test set. Old menu responses cannot be translated into new proposals
or retried in their old experiment. This is a new contract/run namespace.

## Reuse and the missing integration

Reuse unchanged:

- `lc_structure_packet.py`: source-bound causal observations and level catalog.
- `lc_structure_proposal.py`: schema, citations, geometry and policy binding.
- `lc_structure_preentry.py`: hypothetical entry eligibility and risk geometry.
- `lc_structure_outcome.py`: fixed structural bracket and minute exit replay.
- Existing outcome-free subtype annotation and immutable-file/hash primitives.

New integration has four responsibilities: prepare/freeze source requests and
mechanical controls; capture one assessment per case with immutable terminals;
resolve entries and score after the terminal lock; render the report. These
must be independently testable. A small command interface exposes `prepare`,
`status`, `lock`, `score`, `report`; role dispatch uses an explicit host bridge
connected to the persistent capture owner, not a hidden API call in `prepare`.
These commands/interfaces are planned, not currently available.

Do not reuse `SingleCampaign`, its fixed18-case policy, its published response
grader, or `assessment_evidence_guard.build_envelope` as though they accept this
contract. The latter derives old fixed2R economics. Reuse neutral canonical
serialization/hash/immutable-publish helpers where appropriate; provide a new
structure-only request envelope and validate its exact delivered bytes. Never
put the old trade menu back in the packet merely to satisfy a transport helper.

## Source and evidence boundary

Use the existing original source requests and verify them against their saved
evidence lock. Freeze selected raw-file hashes, packet seals, curriculum/brief,
proposal schema, role instructions, comparator rules, policy, all executable
dependency hashes, and the archive before any new assessments.

Local archive:
`data/recovered_binance_minute_2026_09_14/btc_1m_2021_2026.parquet`.
SHA256: `5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035`.
Recovery metadata identifies Binance USD-M futures BTCUSDT; source packets use
instrument label `BTC` and stream `btc_1m_2021_2026_saved_5b8a4533f70b8ccd`.
Record that mapping explicitly, not a claim that arbitrary BTC venue data match.
Parquet contains UTC `ts`, OHLC and `vol`; rename `vol` to `volume` at the adapter
boundary. Verify unique minute-aligned clocks and exact uninterrupted intervals.

During preparation, compare each packet's supplied completed candles with the
matching archive aggregates: OHLC absolute tolerance1e-8, volume tolerance
`max(1e-6, abs(source_volume)*1e-9)`, UTC left-closed/right-open bins. Check all
provided rows, not only the setup close. Reject mismatches without resealing or
repairing frozen sources. Parent lineage remains bound to existing source
manifests; do not claim this reruns/authenticates the parent detector or exchange
receipt times. Any source failure keeps its roster row as unavailable, with no
model call and no replacement case.

Each first12 packet currently provides10 daily bars,18 four-hour bars,24 hourly
bars,16 fifteen-minute bars,12 five-minute bars and30 one-minute bars. Preserve
all supplied bars/levels and their timestamps; no provisional candle or future
pivot. These are bounded windows, not unlimited market memory. Missing evidence
must remain missing. In particular, this projection omits raw fusion, funding,
OI and Fibonacci features; the practice result cannot validate those inputs or
the full all-seeing-eye vision.

Only the current case's packet, unchanged outcome-free curriculum/brief, explicit
practice policy and governing structure-response instructions reach a fresh
assessor. No PROJECT/MEMORY, old answers, outcome reports, other cases, archive
access or web browsing. Historical teaching text is evidence, not permission
to override the new response contract. Preserve citations to source teachings
when the interpretation uses them. The controller may reconstruct a case from
its own predecision prefix; no outcome scoring/reports are available before the
all-case terminal lock. Never supply later-case data to an earlier assessor.

## Decisions, role boundary and spending

The agent must state which inputs it trusts, the pre-existing larger structure,
the child sequence/location, opposing evidence, an alternative explanation and
its thesis. It chooses `enter_proposal`, `wait_proposal`, `reject`, or
`insufficient_evidence` using the existing strict response schema. Entry/stop/
destination/confirmation refer to actual catalog IDs. No invented prices or
post-outcome explanation rewriting.

For this first economic adapter, invalidation must equal the effective
tick-rounded stop. Immediate entry requires predecision confirmation citations;
wait means the first post-arm completed1m close strictly above its frozen level.
Separate post-entry thesis exits, retest state machines, scaling and trailing
management remain unsupported/null, never silently ignored or repaired.

At most12 fresh assessor launches, one per source-eligible case, zero market
critics/retries/replacement answers, max1 in flight,600seconds per attempt.
Retain the previously requested Astra/high role unless the user explicitly
changes it; record requested and actually observed runtime identity separately.
Schema-valid means **unreviewed**, not semantically correct or teacher-approved.
Factual/citation membership checks do not evaluate the truth of the rationale.
An unavailable exact model snapshot is reported as unknown, not fabricated or
automatically treated as invalid JSON. Missing/contradictory role identity,
wrong-case delivery, missing bytes or truncation are transport failures. Each
case requires a distinct role instance and a recorded requested-model setting;
an observed runtime contradicting that setting is a failed assessment.

Use one persistent capture owner for durable reservation, packet delivery and
raw-response receipt. Record monotonic reserve-to-capture duration, including
delivery overhead, and exact input/response bytes. A controller restart makes an
in-flight attempt interrupted/null; do not silently redispatch or reconstruct
timing from a new process. Resume untouched cases without rerunning completed
ones. If abandoning a run, unstarted cases become explicit not-run terminals.

Deliver each full request once where the host supports it; if chunking is needed,
verify every chunk and reject truncation without silently shortening evidence.
Current packet-only payloads are146,741–151,131 canonical ASCII bytes; this is
not a billed token estimate. A12-call ceiling is **not** a credit/dollar ceiling.
Before launch, show actual request sizes, available usage telemetry and the
call-limited spending terms; obtain explicit launch authorization tied to this
run's manifest. If actual charges/model identity cannot be observed, say so.
No paid role is authorized by writing or implementing this design alone.

## Proposed fixed paper policy — requires written-design approval

These are research assumptions, not recommended live risk settings, optimized
thresholds or verified venue order filters. Freeze once; no search on these12.

| Field | First-run value |
|---|---|
| Instrument/stream | Exact source labels and archive mapping above |
| Modeled account / max notional / max leverage | 50,000 / 50,000 / 1 |
| Modeled risk budget per independent case | 100 including flat modeled costs |
| Round-trip cost | 12bps of entry notional, charged once |
| Entry expiry | Decision +15minutes, exclusive |
| Holding limit | Fill +1,440minutes |
| Minimum net reward/risk | 0.5, calculated at actual modeled fill |
| Maximum entry price | Floor(source close ×1.005 to the0.01 research tick) |
| Research tick | 0.01 quote units; not certified exchange constraints |
| Minimum processing / routing | 90seconds / 0seconds |

Amounts are paper USD-equivalent using a1:1 USDT/USD assumption, not a modeled
conversion or real account. No funding, market impact or lot-size execution is
claimed. Label results “after flat modeled costs; excludes funding/impact.”

Primary agent-versus-mechanical comparison uses the same per-case measured
reserve-to-capture latency, floored at90seconds, mapped onto historical decision
time. This isolates decision/geometry choice under matched timing, **not** a
claim that code needs an agent's delay. Also show the same mechanical rule at
its fixed90second assumption as an operational control. Unknown elapsed timing
makes the matched comparison unavailable; do not backdate a slow or failed agent
to90seconds. The policy hash stays unchanged: measured availability belongs in
the execution record, not a rewritten assessor response. No cost/latency grid
search in this first practice run.

## Mechanical structural control and legacy reference

Freeze this intentionally simple control before any new outcomes. It is not
described as Wyckoff's or another teacher's validated trading rule:

1. Use the last completed5m bar's high as the `close_above` trigger and its low
   as both stop and invalidation. Require its low below indicative close and
   exact stop-grid compatibility; otherwise no mechanical setup.
2. Among supplied1h/4h/1d candle highs and pre-existing parent high boundaries,
   choose the nearest price strictly above both indicative close and trigger.
   Tie-break by full catalog ID in lexicographic order. This is a raw observed
   reference, not asserted proven resistance. Do not choose a farther level to
   improve reward/risk. If absent, report no mechanical setup.
3. Include every intervening catalog level, apply exactly the agent's external
   policy, same pre-entry resolver and structural-target scorer. If frozen
   geometry fails the policy, retain the nonentry; never try another combination.

The agent can choose different catalog levels and immediate versus wait under
those same limits. Both have access to the same catalog; no baseline/agent
capital competition. Mechanical explanatory fields are labeled mechanical
facts, not fabricated human/agent convictions. Retain the original source subtype
and comparator version. Source uncertainty is unavailable, not “no setup.”

Include the original fixed-rule LC immediate reference (source stop/2R target,
fixed50,000 notional,90second delay,12bps and its old decision-relative24h deadline)
as a separately labeled reference, using the frozen scorer. It is **not** the
matched test of agent value: its stop, sizing and clock differ. Do not compare
its dollar total with the risk-capped arms as if the difference proves better
judgment. Also display the stay-flat reference to expose reject-all behavior.

## Entry replay, terminal locking and accounting

Save every valid/invalid/rejected/failed/not-run terminal before revealing any
new study outcomes. All12 rows must exist in the lock, even with zero usable
assessments. Reopening verifies original bytes and re-derives validations; no
regrading after outcomes or best-run overwrite. Raw archive hash and raw failure
evidence are retained: the scorer's accepted-prefix hash alone is insufficient
to distinguish corrupt sources.

After the lock, a new thin resolver examines possible minute-open fills in
chronological order, supplying the pre-entry checker only completed prior bars
plus that open. Stop at the first eligible fill or known cancellation/expiry.
Never search past an earlier eligible trade to find a later better one. A gap
in required history is unavailable, not an assumed no-touch or harmless expiry.
The outcome scorer then stops at the first bracket exit or deadline. Open gaps
precede intrabar extremes; target gaps get no improvement; both-touched bars are
stop-first and flagged. Intrabar exact exit time remains unknown.

The study accounting layer distinguishes states that the single-case scorer
deliberately leaves unscored:

- Valid deliberate reject: zero exposure, with its stated reason.
- Valid mechanical no-setup or verified no-fill expiry/cancellation: zero
  exposure, separately identified; not called an agent rejection.
- Valid filled proposal: use scored modeled PnL/costs and risk units.
- Missing data, insufficient evidence, invalid response, unsupported plan,
  uncertain timing, transport failure, interruption or not-run: null, never a
  successful rejection or a zero-PnL trading judgment.

Cases are independent hypothesis replays, not a capital-constrained portfolio:
do not let overlapping positions suppress another case or subtype. Report overlap
counts, sums/averages of per-case modeled results and matched deltas with coverage.
Do not label these sums an equity curve, account return, Sharpe or portfolio
drawdown. Full-roster totals/deltas are null if any required case is unknown;
known subtotals and matched-complete subsets carry explicit denominators.

## What the user sees

One offline HTML report, no external CDN or server requirement:

- Overview:12 rows,9/3 source subtype split, model delivery validity, elapsed
  timing, entries/rejects/expiries/cancellations/failures and cost assumptions.
- Case card: predecision daily/4H context, hourly setup and minute execution
  panels; declared trigger, stop, target, barriers, availability/decision/fill/
  exit markers; separately identified future price path. Show unavailable data
  as gaps, not interpolated prices. Any downsampling is disclosed.
- Verbatim agent rationale with resolvable cited observations, mechanical
  comparison and a plain-language explanation of the modeled outcome. Escape
  model/source text as data so it cannot execute HTML or script.
- Scorecard, separately by source subtype: after-cost outcomes, participation,
  comparator winners still captured profitably, deliberate rejects of winners,
  deliberate rejects of losers, losing trades still taken, and expiry/cancellation
  misses. A rejection's avoided/missed amount is a labeled comparator
  counterfactual, not money earned by the agent. Unknown cases get no such credit.
- Teaching review: observed factual contradictions flagged by code versus
  unreviewed interpretation. Post-outcome commentary never overwrites the
  original reasoning or enters this run's curriculum. The owner can identify
  possible lessons for a separately versioned future practice batch.

The HTML may contain outcomes; it is never an assessor input, even if future
panels are visually collapsed. Export `case_results.json` and `summary.md` from
the same locked records. Rerendering/reopening must not invoke a model or change
the results. Show the entire run even when it is all-reject, negative or failed.

## Acceptance and stop condition

Before paid launch: synthetic end-to-end prepare/capture/lock/replay/report tests,
including a winner, loser, ambiguous bar, deadline, rejection, expiry, bad data,
invalid agent output and restart. Test no scoring before all terminals, no
double dispatch, immutable reruns, matched timing, fractional sizing, report
escaping, null propagation and subtype isolation. Verify prepared real sources
and source-only mechanical proposals without scoring real future prices. Obtain
one independent software review of the integrated path. Preserve existing tests
and disclose the repository's pre-existing collection failure separately.

Completion of the first authorized run means all12 terminals and the actual
case-by-case report are delivered, not that a particular win rate was achieved.
If every proposal rejects, none fills or invalidity is high, that is the result;
do not expand/retune until something wins. This small exposed practice batch
cannot establish consistent profitability. A separate fresh/prospective
evaluation is needed before promotion, then live shadow validation. No automatic
transition to another campaign, other archetypes or live trading.

## Verified at this design checkpoint / next gate

Read-only source inspection rebuilt all20 structure packets successfully, and
all20 raw source-file hashes matched their original evidence lock. The
selected12 and their9/3 subtype split were derived without outcome inspection.
Each selected packet has the six window lengths above and the same curriculum
hash `5f8c38b29af1df2059d675c0ef9660465ed88e0a5950e8f7714094111024fda4`.
Archive SHA256 was rechecked, and its metadata reports2,979,360 rows. Existing
matplotlib3.9.4 is available; rendering should use a writable temporary cache.
No archive-aggregate comparison, practice runner, new assessment or economic
replay has been performed for this design. Existing scorer tests belong to the
[prior checkpoint](../../knowledge/lc_structure_outcome_checkpoint_2026_09_29.md).

Next: user reviews this written design, including the paper-policy assumptions;
then prepare the implementation plan and its execution handoff. The earlier
“proceed” approved the scope and this design-writing step, not an unbounded
paid evaluation. Keep the existing research branch. No new library download,
data download, push or PR is needed for this checkpoint. Source requests,
archive and prior manifests remain local-only dependencies; public tests must
use synthetic fixtures and run without them.
