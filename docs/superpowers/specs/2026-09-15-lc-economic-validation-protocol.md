# LC economic validation protocol

## Status, authority and question

September 15, 2026. User approved preparing an institutional-quality testing
protocol after the January19 validity pilot. This document preregisters **D1,
a four-case exposed-development economic comparison**, and defines the gates
for later validation. It does not certify an industry standard, authorize live
trading, or claim that four cases establish an edge. Implementation and artifact
freeze must pass independent review before any D1 market-role call.

Independent quant design review approved this protocol after corrections to
critic-packet freezing, missing-delivery terminals, drawdown/fee accounting and
fixed missed-opportunity comparisons. Source-only feasibility passes all four
cases. Neither approval nor feasibility is an executed economic experiment.

Question: on the same native LC opportunities, does reviewed multi-timeframe
agent selection improve net outcomes relative to immediate entry and generic
minute confirmation? This evaluates the frozen supplied-information policy,
not an all-data master trader, agent-created entries or discretionary exits.

Three approaches considered: (1) repeat more judgment-only cases, which cannot
measure economics; (2) jump to optimization/CPCV on the small exposed cohort,
which cannot establish independent evidence; (3) first run the fixed economic
comparison below, then collect suitable validation data. Choose (3).

## D1: fixed population and budget

Study ID `lc_economic_development_v1`. New private namespace:
`results/lc_economic_validation_2026_09_15/run_v1/`. Never reuse an older job or
change a historical request, source, lock, response, grade or outcome.

Use the next four chronological saved native **pre-winner LC long** candidates
after the completed Jan19 pilot. Selection uses only identity and decision time,
not market structure, labels or returns. Case IDs:

| New case | Exact candidate ID |
|---|---|
| LCE1 | `hourly-lc:2026-01-20T06:00:00+00:00` |
| LCE2 | `hourly-lc:2026-01-25T09:00:00+00:00` |
| LCE3 | `hourly-lc:2026-01-29T16:00:00+00:00` |
| LCE4 | `hourly-lc:2026-01-31T15:00:00+00:00` |

Exactly four specialists maximum and four conditional critics maximum. Each is
a fresh, history-free requested `gpt-6-astra`/high role. Same model family is a
correlated-review limitation, not independent institutional certification.
Actual snapshot and billed usage remain unknown unless observed. No extra master
brief, replacement candidates, retries, repaired responses or best-of-N selection.

Q1 outcomes have already been examined by the research process. **All four are
exposed development**, even if no prior agent assessment is found. Do not read
old outcome artifacts to select or prepare this run. Preserve an exposure ledger
with prior registry matches, exclusions and the fixed ordering. Exclude previous
LC1–LC4 and LCV1; do not pool their results with D1.

Read-only inventory finds 13 operationally remaining Q1 candidates after those
five, of which three also match earlier development assessment registries
(February2, March6 and March15). The four selected January cases have no such
registry matches; that does not change their exposed-development status. The
reviewed curriculum excludes all18Q1 candidate IDs. Its source-only monthly
reconstructions use independently reset histories, not one continuous live-state
replay, and depend on ignored/local and external parent-source files.

## Source, evidence and memory

Reuse `results/lc_persistent_master_2026_09_14/source/2026-01/source.json`, SHA256
`6ba371d2ded2fb152176b6e82ea2bd56f4ba5714e982f27ee04f44efaa0ad51e`, and archive
`data/recovered_binance_minute_2026_09_14/btc_1m_2021_2026.parquet`, SHA256
`5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035`.
Verify original code/config/source/replay manifests, including explicit
expected-absent leaves; reject collisions before adding current runtime hashes.
Do not rerun the engine, extend candidate dates or repair missing observations.

For each case build exactly its continuous 17,280-minute `[decision−12days,
decision)` prefix through existing `build_lc_packet` with BTC provenance, then
`build_context_request` and `PublishedContextResearchJob`. Parent ledgers are
bounded causally by the existing builder. Preserve all six completed candle
layers and distinguish known adverse structure from unknown evidence.

Reuse unchanged curriculum under
`results/lc_context_discrimination_2026_09_15/curriculum_v1/`: snapshot ID
`299193b77af10d1382bc02d537a8f919b1e2cd9a3fe5557543e4bb5184fb727d`, snapshot file SHA
`e42238cf2f647f672f833c8f91d04c8bcc390bc8a42b79645eea14b7e08d9fd7`, brief SHA
`c7d63d6675a0903604a5bd124835acf5cb880cef28530affe2ed7c40b53664fc`, review SHA
`ec87b84a112057f0cb614e0aa271578d28bc39a142708ada6bcd08adce2cc783`.
Verify source-ID exclusions before building all four requests. No outcome-derived
memory update between cases. Do not supply project handoffs or this outcome-aware
protocol as market evidence. Macro/derivatives defaults are not historical
observations; unverified Fibonacci anchors are not added as positive evidence.

## Fair comparison and unchanged research economics

| Arm | Fixed choice | What it tests |
|---|---|---|
| A | Native immediate entry | Original signal benchmark |
| B | Always `wait_5m_high` | Value of generic confirmation |
| C | Reviewed enter/wait/reject; uncertain remains null | Added contextual selection |

Same candidate list and frozen menu. Long-only, $50,000 notional per admitted
position, 12bps assumed round-trip cost, 90s processing rounded up to minute arm,
zero routing, exclusive decision+15min entry expiry, stop=source close−2.7ATR14,
target=actual entry+2(actual entry−stop), deadline=decision+1440min.

Waiting freezes the last completed aligned 5m high. A fully closed **post-arm
1m candle** must close strictly above it; fill is next-minute open before expiry.
Pending stop cancellation remains active from decision, including latency. No
alternative trigger, hard parent gate, calibrated confidence or fusion tuning.
The critic is an offline quality gate in D1, not assumed to operate in 90 seconds.

Use `resolve_entry` and `replay_isolated_sleeves` as the existing accounting
authority, feeding plans from the **new published job grades**, not the legacy
review-request helper that validates another schema. Each arm has an independent
single-position, unfunded book, initially empty at LCE1. No cross-archetype or
cross-arm competition. Pending plans reserve no capacity; busy entries do not
retry. Release capacity at actual known exit, not nominal deadline. This is the
four-case book only, not a full-January/Q1 or funded portfolio backtest.

Gap stops fill at adverse open; ambiguous both-barrier candles use the existing
stop-first rule and are counted. Target gaps retain the existing capped-target
assumption. These are OHLC conventions, not fill certification. No funding,
order-book queue or calibrated market impact is inferred from missing data.

## Prespecified stress scenarios — no extra model calls

Score all arms under each scenario; never select whichever scenario looks best:

| Scenario | Round-trip cost | Processing delay | Purpose |
|---|---|---|---|
| S0 primary | 12bps | 90s | Registered research baseline |
| S1 friction | 24bps | 90s | Additional cost burden |
| S2 delay | 12bps | 300s | Slower decision availability |
| S3 joint | 24bps | 300s | Combined sensitivity |

These values are **stress assumptions, not measured venue estimates**. Derive
scenario execution plans from the locked S0 choice; bind the transformation and
scenario ID without changing the original response, request, grade or menu. Keep
stop, confirmation level, expiry and deadline unchanged. A changed delay can
change eligibility/fill/target/occupancy, so rerun the resolver/book rather than
subtracting an arbitrary PnL penalty. S1 changes cost only and must preserve fills.
Do not claim the agent would make the same decision if shown different economics.

Record controller-observed elapsed time for every role, including orchestration
overhead, without calling it pure inference latency. Report the distribution and
whether 90s/300s cover it; do not revise the stress grid after observing timings.
If actual runtime exceeds the assumptions, state the execution mismatch. A later
live design must specify whether the critic is on-path and model that actual path.

## Frozen sequence, failures and outcome boundary

1. Independent fixture/code review, then source-only assembly of all four cases.
   Verify exact roster, source readiness, memory, source hashes and menu equality.
2. Freeze all source/specialist requests, detached specialist packets, envelopes,
   expanded specialist instructions, critic instruction template and builder/code
   hashes, source/protocol hashes, scenarios and metrics **before the first role**.
   Use the tested `{case_id,plan,request}`
   wrapper; inner request is the sole contract. Capture actual chunk returns.
3. Run one specialist per case and at most one eligible fresh critic. Lock each
   raw answer before constructing its critic request; then freeze that exact
   answer-bound critic packet, envelope and expanded instructions before critic
   invocation. Same catalog, no grading/repair feedback to roles.
4. A schema/transport/material-review failure ends that case without replacement;
   continue the remaining fixed cases unless shared integrity is compromised.
   A shared source/hash/chronology defect halts the whole run. Resume only an
   identical saved computation, never replace a delivered assessment.
5. Before any outcome access, lock one terminal record per fixed case: either its
   recomputable published grade or a controller failure record binding request/
   attempt IDs, invoked/captured state, available raw/capture hashes, failure
   reason and null research_plan. A controller record is not a published grade or
   valid reviewed skip; preserve the incomplete job unchanged. Failed/missing
   deliveries consume that role's attempt and cannot be retried. No critic follows
   an invalid/missing specialist. Preserve explicit not-invoked events without
   fictional captures. Uncertain/null is not a reject. Lock all four terminals.
6. Only then authorize the new study's outcome reader. For each case allow OHLC
   from decision through the minute before deadline and **deadline open only**.
   Do not use deadline high/low/close, later observations, or old outcome tables.
7. Independently reproduce accounting and reopen/recompute saved results before
   publishing. Never amend the frozen protocol after calls; deviations belong in
   a separate report and may invalidate conclusions.

## Metrics and honest decisions

Primary descriptive quantities: full-book S0 net PnL for A/B/C and paired
`C−A`, `C−B`, expressed both in dollars and per **four eligible opportunities**.
Report all four scenarios together. Explicit reject, valid expiry and stop
cancellation have zero exposure; invalid/missing/null agent output is **unknown**.
If any C plan/outcome is unavailable, primary C total and both primary contrasts
are null, even if other C trades made money. Valid-case subsets may be reported
separately with their denominators, never substituted for the full cohort.

Secondary report: decision frequencies; admitted/closed positions; rejection,
expiry, cancellation, busy and unavailable counts; fees; initial monetary risk
by arm; gross/net average win and loss; position holding times; ambiguous bars;
missed winners/avoided losers only for valid matched decisions. Scale-outs and
adjacent observations are not independent trials. $50k fixed notional is not
equal monetary risk after different entry prices; disclose this, not an equal-risk
claim. Rejecting a winner counts as an opportunity cost in paired comparisons.
For missed-winner/avoided-loser counts, comparator is always matched **A net PnL**:
a valid zero-exposure C result with A>0 is a missed winner, A<0 an avoided loser,
and A=0 neutral. Break these counts out by C reject/expiry/cancellation/busy reason
so mechanical nonentry is not credited as intentional agent rejection. Any B
comparison is separately labeled, not substituted after seeing results. Unknown
plans/outcomes contribute to neither count.

Include minute-close **dollar marked-to-market drawdown** for each unfunded book:
cumulative **gross** realized PnL plus open-position unrealized PnL marked at each
completed minute close minus cumulative costs, charging the fixed full round-trip
cost once on admission (never subtracting it again from net realized PnL). Respect
causal exit/release event ordering. Maximum drawdown is the peak-to-trough fall of this
PnL series starting at zero. Label it dollar research drawdown, not percentage
equity drawdown, intraminute worst loss, or funded risk. Also report worst closed
trade loss. Include an additional realized-PnL observation immediately after each
deadline-open exit, including the final deadline, so a terminal gap loss is not
omitted. Between exposures carry flat PnL forward without reading extra prices.
Missing marks invalidate this metric; do not replace it with a
closed-trade curve without relabeling it.
If any C plan/outcome is unknown, full-cohort C drawdown is also null; a valid
subset's drawdown must be separately labeled and cannot replace the cohort metric.

There is **no D1 statistical/economic pass**. Four cases cannot establish edge.
No Sharpe significance, bootstrap confidence claim, CPCV, optimized cutoff or
deployment conclusion is produced from this batch.

- **Technical complete:** all four have valid reviewed contracts or truthful
  reviewed uncertain decisions; captured delivery and accounting/restart checks
  pass. A coherent uncertain answer can pass the interface while economics stay
  inconclusive. Invalid contracts/material errors are reported as failures.
- **Descriptive better/worse/tied:** report signs and magnitudes of both paired
  contrasts only where complete. If C equals B, no added selection benefit over
  generic waiting is demonstrated. If C never enters, entry quality is untested.
- **Economic conclusion:** always inconclusive for durable edge; no change to
  production regardless of positive PnL. Further research must not be conditional
  on hiding a negative D1 result. Technical failures call for bounded diagnosis;
  changing the policy requires a separately registered version, never D1 repair.

## D2 and D3: requirements, not permission to run unspecified validation

D2 is a separately registered broader historical study. It cannot start until a
source/exposure inventory identifies enough complete **candidate episodes**, not
merely millions of candles, and specifies the following before test labels:

- Chronological outer walk-forward windows with dates, update schedule and final
  untouched evaluation interval; exposed history stays development. Every learned
  gate, prompt, retrieval rule and outcome-derived memory record obeys the fold's
  training cutoff. Trader teachings available later are retrospective priors,
  not evidence of what a live historical system could have known.
- Primary economic margin, risk budget, stopping/sample-size rule and dependence-
  aware uncertainty calculation, justified using development-only variance and
  episode clustering. No arbitrary universal trade count. Current18Q1 candidates
  do not establish sufficient sample size or regime coverage.
- Purge training labels overlapping test outcome intervals; document additional
  embargo for remaining dependencies. Overlapping feature history alone is not
  automatically forbidden: reproduce what is causally available. Any CPCV is
  supplementary robustness analysis, with fresh fold-specific learned memory and
  nested model selection, not simulated forward deployment or independent new
  paths. Do not run CPCV on four cases or a fixed policy merely for branding.
- Prespecified paired opportunity metrics plus a funded, exposure-constrained
  portfolio analysis and marked-to-market risk. Confidence intervals must account
  for time/episode dependence and the planned comparisons. Track every searched
  archetype, threshold, prompt and model variant; disclose unknown historical
  search extent. DSR/PBO may inform selection-bias analysis when their assumptions
  and input history support them; they do not rehabilitate contaminated tests.
- Venue/instrument reconciliation, spreads, fees, funding where applicable,
  impact/fill/latency calibration, missing-data behavior, and stress scenarios.
  Missing historical feeds restrict the claim; do not fill them with defaults
  and describe the result as an all-data agent.
- Agent factual/chronology tests with independently annotated cases, registered
  repeatability tests that retain all outputs, and predefined context ablations
  or a deterministic contextual challenger. No best-of-N answers. Model changes
  create a new version; two same-family models can share errors. Human/domain
  review and software checks complement, not disappear behind, model critics.

D3 is a separately approved prospective shadow/paper study: decisions and input
receipts recorded before outcomes, actual runtime and rejected/missed orders
measured, simulation/live-feed reconciliation, immutable policy version and
predeclared reporting/stopping dates. Model training contamination cannot be
excluded merely by hiding historical candles; forward collection supplies a
stronger temporal test. No real-capital promotion is automatic: funding, loss
limits, controls and deployment authorization require a later explicit decision.

## Reuse, implementation boundary and completion artifacts

Keep existing research branch. No shared source or old private-run edits, engine
reruns, new dependencies, downloads, push/PR, fusion changes or production orders.
Reuse the reviewed builders, published jobs, transport validator, immutable
artifact primitives, conditional resolver and isolated books. Create a separate
small D1 runner/scenario-report layer; do not extend the frozen Jan19 runner in
place. No new agent platform or general backtesting framework is required.

Execution readiness requires fixture tests for: exact roster and null manifest
semantics; unchanged sources and copied scenario plans; all-grades-before-outcome
access; conditional critic/terminal failures; stress timing and stop cancellation;
scenario book isolation; dollar MTM/fee/exit ordering; unknown aggregate handling;
independent recomputation and restart. Implementation plan follows approved design.

The eventual D1 deliverables are the frozen roster/exposure/source manifest,
four saved decisions/reviews, three independent books in four scenarios,
per-case evidence-to-choice-to-outcome table, full failure/latency report,
accounting audit, and a plain-language better/worse/inconclusive explanation.
Protocol approval alone is not a new test result.

## Method references and scope of adoption

- [Financial cross-validation documentation](https://random-docs.readthedocs.io/en/latest/implementations/cross_validation.html):
  purging, embargo and combinatorial paths inform D2 split design, not a claim
  that D1 executes CPCV.
- [Bailey and Lopez de Prado, Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf):
  motivates the complete search ledger and caution about multiple comparisons;
  no D1 DSR estimate is justified.
- [QuantConnect reconciliation guidance](https://www.quantconnect.com/docs/v2/writing-algorithms/live-trading/reconciliation):
  motivates explicit execution assumptions and later measured reconciliation;
  no library installation or realistic-fill certification is implied.
- [NIST AI RMF Measure](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/):
  informs documented tests, benchmarks, uncertainty, independent challenge and
  ongoing monitoring. This protocol is not certification of compliance.
