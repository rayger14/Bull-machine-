# Bounded LC single-assessor experiment — September 21, 2026

## Verdict

Finished the approved bounded screen. Do not promote this agent to live gating.
Its three selected trades produced a positive known subtotal, but it rejected
valuable rebound winners and did not demonstrate incremental value over the
simple mechanical-confirmation control. This is not evidence that all agent
trading is ineffective; this particular context/menu/prompt combination failed
to establish the required advantage. No production engine or order changes.

## What actually ran

18 frozen January–July 2026 LC cases, excluding the two previously exposed cases.
11 downside-rebound candidates and 7 upside-expansion candidates. One fresh
specialist per created assessment, requested Astra/high; actual model snapshot
and billed credits are not independently available. 17 specialists were created,
zero market critics, zero retries. Separate software review occurred before the
market run and is not included in that specialist count.

16 schema-valid, **unreviewed** decisions: 13 rejects, 3 conditional waits, zero
immediate entries. Feb23 failed dispatch before agent creation; May10 returned
an incorrect request hash. Both remain unavailable, not successful rejects.
Capture delays for the 17 returned responses were 144–203 seconds, median168s,
including delivery and controller overhead. All18 terminals were locked before
new future-price windows were loaded. No response was repaired after outcomes.

This tested minute confirmation **inside hourly LC**, not a separate minute
archetype, all17 archetypes, or the live engine's scale-out management.

## Economics

Frozen $50,000 notional per admitted trade, original stop, actual-entry-based2R
target,15minute entry expiry,24hour original deadline. Independent unfunded
single-position books.12/24bps are modeled round-trip costs ($60/$120 per trade).
No funding, market impact, account capital/margin or live execution certification.

Primary comparison:12bps and common90second processing.

| Policy | Filled trades | Net dollars | Dollar MTM drawdown |
|---|---:|---:|---:|
| Immediate control |18|+2,682.78|4,491.63|
| Mechanical wait for published five-minute-high confirmation |12|+5,470.81|2,815.86|
| Agent choices |3|+2,342.13 **known subtotal only**|Unavailable|
| Reject all |0|0.00|0.00|

Full18-case agent-policy return and drawdown are **null**, because two choices
are unavailable. Never silently impute no-trade to these cases. As a descriptive
same-16-valid-cases comparison, immediate subtotal is+$3,406.29, mechanical wait
+$6,428.98, and agent+$2,342.13. This matched subset is not a repaired full policy
or an unbiased replacement cohort. No occupancy conflicts occurred.

Against immediate entries, the13 deliberate rejects avoided8 losers but missed5
winners. Of the3 admitted trades,2 preserved winners and1 still lost. Against
mechanical confirmation, rejects avoided3 losses and missed4 winners; six other
rejected cases never triggered the mechanical entry. Six mechanical expiries
and two unavailable agent choices account for the8 noncomparable cases there.

| Selected trade (UTC) | Agent decision | Net at12bps/90s | Exit |
|---|---|---:|---|
|Feb25 02:00|Wait|+2,756.01|2R target|
|May05 14:00|Wait|+83.94|24h deadline|
|Jul26 23:00|Wait|-497.82|Stop|

The total relies on the Feb25 winner; the other two sum to-$413.88. Three filled
trades cannot establish a dependable edge. Measured-delay fills yielded the same
three results, but five-minute assumed processing reduced the known subtotal.

### All prespecified sensitivities

Agent column is always a **known subtotal**, never full-policy PnL. Equal-measured
rows also have incomplete controls because Feb23 has no measured timing. Their
control columns are known subtotals too. Reject-all has zero known PnL throughout;
its full-policy PnL is null in equal-measured rows, zero in the other six.

| Scenario | Immediate net/subtotal | Mechanical net/subtotal | Agent subtotal (fills) |
|---|---:|---:|---:|
|12bps /90s|2,682.78|5,470.81|2,342.13 (3)|
|24bps /90s|1,602.78|4,750.81|2,162.13 (3)|
|12bps /300s|2,742.34|4,507.79|971.50 (2)|
|24bps /300s|1,662.34|3,907.79|851.50 (2)|
|12bps /measured agent,90s controls|2,682.78|5,470.81|2,342.13 (3)|
|24bps /measured agent,90s controls|1,602.78|4,750.81|2,162.13 (3)|
|12bps /equal measured, incomplete|3,873.16|6,336.09|2,342.13 (3)|
|24bps /equal measured, incomplete|2,853.16|5,736.09|2,162.13 (3)|

## All decisions and interpretation

Reasons below summarize agent claims, not independent semantic approval. The
agents considered daily/4H direction, previously established parent bounds,
hourly setup sequence, minute response, opposing price areas and the fixed
trade horizon. Structural-invalidation prose did not alter the fixed stop.
Immediate outcomes below are12bps/90s counterfactuals, not actual live PnL.

| UTC decision | Subtype | Usable choice | Reason summary | Immediate net |
|---|---|---|---|---:|
|Jan29 16:00|Downside|Reject|Broken larger floors; failed rebound and overhead recovery burden|-1,025.82|
|Jan31 15:00|Downside|Reject|Persistent decline; local bounce below broken4H floor|-1,044.54|
|Feb23 02:00|Downside|Unavailable|Dispatch thread limit; no agent created|-276.58|
|Feb25 02:00|Upside|Wait|Recovery and expansion; pullback requires minute confirmation|+2,857.13|
|Feb28 07:00|Downside|Reject|Accelerating decline; failed bounce, broken larger floor|+2,814.51|
|Mar07 20:00|Downside|Reject|Expanding sell leg and weakening local recoveries|+95.39|
|Mar08 23:00|Downside|Reject|Daily decline and fresh4H floor break|+2,078.34|
|Mar22 22:00|Downside|Reject|Repeated failed4H recoveries within daily decline|+1,238.94|
|May02 22:00|Upside|Reject|Expansion rejected around pre-existing4H ceiling|-7.85|
|May03 23:00|Upside|Reject|Contested daily range top; no accepted escape|-553.81|
|May05 14:00|Upside|Wait|Continuation thesis but upper wick/weak acceptance needs confirmation|+191.93|
|May10 16:00|Upside|Unavailable|Raw wait response failed exact request-hash validation|-446.93|
|May14 15:00|Upside|Reject|Daily selling and prior4H rejection outweigh hourly rebound|-939.81|
|May28 04:00|Downside|Reject|Accelerating decline and failed recoveries|+7.24|
|Jun01 13:00|Downside|Reject|Weak rebound beneath broken larger structure|-758.81|
|Jun02 15:00|Downside|Reject|Accelerating selloff and repeated failed bounce|-1,032.99|
|Jun30 13:00|Downside|Reject|Broken4H floor; liquidation without demonstrated absorption|-36.57|
|Jul26 23:00|Upside|Wait|Recovery and supporting shelf; weak final minutes need confirmation|-477.00|

All10 assessed downside cases were rejected, including all5 missed immediate
winners. This is consistent with excessive trend-alignment conservatism for a
rebound strategy, but does not identify its cause conclusively. It may reflect
the taught thesis, available context, fixed bracket/horizon or prompt framing.
Do not convert this observation into an outcome-fitted new threshold.

## Verification and durable evidence

- Fresh focused regression command: `python3 -m pytest -o addopts= -q tests/research/test_lc_single_assessment.py tests/research/test_lc_single_campaign.py tests/research/test_lc_single_accounting.py tests/research/test_lc_setup_preflight.py tests/research/test_conditional_entry.py tests/research/test_conditional_occupancy.py --tb=short` —94 passed in6.75s.
-554 available case/arm/scenario resolutions matched `score_conditional` against
  the occupancy replay within1e-7 dollars. These paths share underlying primitives;
  this is a cross-check, not independent external backtest certification.
-18 exact decision-through-deadline windows each have1,441 unique minute rows;
  total25,938. No new data download or outcome-based sample change.
- Local private artifact directory: `results/lc_single_bounded_2026_09_20/run_v1/`.
  `economic_result.json` contains all eight ledgers and MTM outputs;
  `scoring_verification.json` records coverage and cross-checks; per-case captures
  preserve raw answers, delivery records, elapsed time and immutable grades.
- Manifest: `5503eb45a8adc669f895e2f9a0853690f42d0dfe65c931bcb5486e7644b751ec`.
- Terminal lock: `9956a185eaba8fd416a22e681cd46a5f0f0ea258ec5ebf4de0902da7e76a8cfb`.
- Minute archive SHA256: `5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035`.

## Stop and next decision

The bounded deliverable is complete. No further assessments, retries, tuning or
live gating are authorized by this run. This is a filtered retrospective screen,
not pristine holdout, walk-forward validation or CPCV. Do not pool the earlier
three-case exploratory results or claim significance from these18.

Recommended next work, requiring a separately scoped approval: prioritize a
larger chronological **code-only** comparison of immediate versus the existing
fixed mechanical confirmation, with costs and overlap handled. It is cheaper and
was the stronger control here. Keep the agent in research, not order authority.
Before paying for a revised agent campaign, write a falsifiable downside-rebound
thesis from the trader teachings that distinguishes a valid countertrend rebound
from continuation; separate developing that hypothesis from its untouched test.
Do not repeat these now-exposed18 to claim improved validation.

New modules, tests and this report are local/uncommitted at this checkpoint;
they have not been pushed. Existing PR83 does not yet contain this work. Private
prices/captures/results are not included in GitHub and need separate authorized
transfer if another machine must reproduce the experiment.
