# Master-rulecard hourly/minute pilot — September 12, 2026

## Result in plain language

The dossier was turned into two distinct [specialist rulecards](master_specialist_rulecards_2026_09_12.md),
then used in one new hourly liquidity-compression assessment and one new minute
sweep/reclaim assessment. The raw choices were accept-hourly and reject-minute.
The fixed hourly trade subsequently won; the fixed minute trade hit its stop.
**Neither assessment passed the complete evidence contract.** These are not two
validated agent decisions or proof that the agent adds trading value.

The observed weakness is evidence binding: relevant facts were largely read
correctly, but explanations did not consistently cite the exact supporting
records. The minute answer also omitted parts of the exit contract. Changing
strategy gates or fusion thresholds would not address these failures.

## What was actually tested

- [Selection protocol](../superpowers/specs/2026-09-12-master-rulecard-entry-pilot.md)
  was committed before reconstruction. First eligible case per track in
  **February 1–14, 2026 UTC**, with exact January 2 warmup seed; no outcome-based
  replacement. Decisions fell on February 1 and 2; final hourly exit was February 3.
- Source-only reconstruction processed 1,392 hourly updates and loaded all 17
  intended champion archetypes. Three example-YAML loader errors and the existing
  LibreSSL warning were retained; no intended archetype was silently removed.
  Second-half preparation candidates were unused, not replacement cases.
- Astra authored static, source-grounded shared and specialist instructions.
  Only the applicable specialist section entered each packet; known June worked
  examples were excluded. This was **not a persistent master-agent runtime** or
  model training. Four fresh requested `gpt-5.6-sol`/high contexts performed hourly
  assessment, hourly review, minute assessment, minute review, in that order.
- Both packets included same-stream completed 1m/5m/15m/1h/4h/daily evidence and
  causal detector/parent witnesses. Assessors had no future bars or actual fills.
  Each reviewer received only its case and locked answer. All answers and reviews
  were locked before economic reveal. No retries, repairs or substitutions.
- Exact captured inner returns passed for all **28 chunks across four calls**.
  This verifies captured delivery, not model attention, outer rendering, enforced
  filesystem isolation, source authenticity or freedom from pretraining exposure.

February was unused in these model case assessments, but the archive had already
been researched: this is development evidence, not a pristine holdout. Actual
model snapshots and billed tokens were not available.

## Decisions, evidence quality and fixed outcomes

| Track / decision UTC | Raw decision | Full assessment valid? | Hypothetical exit | Net PnL | Initial price-stop risk |
|---|---|---|---|---:|---:|
| Hourly LC / Feb 2 04:00 | Accept; uncalibrated p=0.56 | No: citation entailment | Feb 3 04:00 deadline | +$1,549.38 | $1,970.19 |
| Minute sweep / Feb 1 05:44 | Reject; uncalibrated p=0.36 | No: invalid paths and citation entailment | Feb 1 06:37 stop | −$154.75 | $94.75 |

Starting equity: **unspecified**. Each independent case uses **$50,000 notional**
and **$60 round-trip cost** (12 bps), not a $50,000 portfolio. Mean initial
price-stop risk across the two cases is $1,032.47; the track-specific risks above
are more informative because they differ substantially. Costs are excluded from
initial risk. No portfolio return, win-rate estimate or calibrated probability is
claimed.

### Hourly explanation

The specialist identified extreme volume, oversold RSI, preceding compression,
a sweep below the pre-existing 4H floor and recovery above it. It also recognized
the larger downtrend and overhead resistance. All five fixed facts were correct;
the reviewer passed identity, chronology, numerics, no-invention and scope, with
no material errors. However, two claim groups did not cite sufficient earlier
candles for their trend/resistance assertions. Relevant evidence elsewhere in the
packet does not repair the frozen claim-specific citations.

Entry was 76,256.20, stop 73,251.4196 and target 82,265.7609. Neither bracket was
hit before the 24-hour deadline; exit at 78,710.70 produced $1,609.38 gross less
$60 cost. Positive PnL does not retroactively validate the acceptance.

### Minute explanation

The specialist recognized a causal sweep/reclaim but only a **$0.60 reclaim above
the 78,630.30 child level**, with no qualifying 4H/N3 parent. It also recognized
that costs consumed approximately **0.6333R**: the indicative net stop loss was
$154.75 versus $129.49 net at the 2R target. The named parent prerequisite is a
separate research hypothesis, not a universal minute-detector/native-engine gate.

All five fixed facts were correct and the reviewer passed the five review
criteria. Nevertheless, deterministic validation failed five citation groups:
several omitted the required `evidence` root, and the missing-input citation
pointed to null. The reviewer additionally identified five citation-support
errors, including a wrong daily-candle field, and omissions of the four-hour
expiry and actual-entry-based target formula. The reviewer did not enumerate
every root/path error; deterministic and semantic review serve different roles.

Entry was 78,630.90, stop 78,481.9003 and target 78,928.8994. The stop was hit
53 minutes later. The four-hour deadline price was above entry, but the fixed
trade had already stopped out: endpoint direction alone would misdescribe it.
The rejection remains unvalidated; it cannot be credited with avoiding this loss.

## Comparators and interpretation

| Fixed comparator | Hourly selection | Minute selection |
|---|---|---|
| Native hourly signal / separate minute research detector | Emit | Detect |
| Hourly numerical eligibility | Pass | Not applicable |
| Fixed 4H/N3 parent hypothesis | Pass | Reject: parent absent |
| Always skip | Skip | Skip |
| Validated agent selection | Unavailable (`null`) | Unavailable (`null`) |

Native hourly **emission is not a reconstructed live book entry**. The private
legacy `native_emitted=true` field on the minute record means detector eligibility,
not a champion minute engine. Its `mechanical=false` means the minute H3 comparator
rejected; hourly `mechanical=true` means numerical eligibility.

The raw agent directions match the simple H3 comparator on both cases. Even
without the citation failures, these observations would not demonstrate value
beyond that fixed rule. `null` is neither rejection nor skip: no agent strategy
PnL, avoided-loss credit or comparison with always-skip is calculable from two
invalid assessments. Every registered counterfactual was revealed regardless.

## Verification and reproducibility

The existing outcome scorer was unchanged: decision-minute open, 2R target,
hourly setup-close minus 2.7 ATR stop/1,440-minute horizon, minute sweep-low times
.9985 stop/240-minute horizon. Opening gaps precede intrabar touches; otherwise
both-touch bars are stop-first. Deadline uses open only. Neither case had an
ambiguous exit bar. Full-horizon excursions include prices after hypothetical
exit and must not be treated as managed-trade excursions.

Zero added model delay is an optimistic assumption, especially for minute entries.
No funding, market impact, inference cost, portfolio interaction or certified
live fills are modeled. This tests the frozen immediate entry, not execution of
new conditional triggers. It does not test teaching ON/OFF or master-versus-
specialist incremental value, and cannot establish profitability or deployment
readiness.

The final combined verification passed **46 private contract tests and 489
research tests**, with the existing LibreSSL warning. A preparation-only hash-lock
bug initially treated deliberately absent model files as mandatory. Before any
role call, absence-aware manifest verification was test-first corrected and
independently reviewed. The failed manifest was preserved; regenerated packets
were byte-identical and no source replay or case replacement occurred. Mandatory
runtime captures still require present files.

Independent final quant review verified the nested locks and recomputed both
outcomes directly from the source windows, matching the scorer exactly.
Private/ignored records remain under `results/master_rulecard_pilot_2026_09_12/`.
Repeated scoring reproduced identical hashes:

| Artifact | SHA-256 |
|---|---|
| Source archive | `5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035` |
| Prepared sources | `01fcac546ef5cc1f72cd29fb7cecda5ca80daf505bd19d2af37d6140fc58c61f` |
| Final manifest | `0d3baeaf9c8427d71f40a20d74823ce7af72d6af6f4ec4818943ac3031ec679d` |
| Pre-outcome grades | `1d64189713afaf1c8133f4ff39a5402346970ac8734efac840ac13c6e8c41a86` |
| All pre-outcome records lock | `91ff0fe9c5bd72f6451c7688de1a9c6db930e3658e4f09270ec91861d0a83f36` |
| Outcomes | `e3fb8522e24fa930e9b734f3afd213d3764921b07933d36b79e7420fb532a3ce` |

Tracked documentation describes the result, but does not include the private
archive, harness or role captures; another CLI needs those local artifacts to
reproduce this exact pilot. No live code, archetype gates, fusion thresholds,
production configuration, orders or deployments changed. No push or PR.

## Next bounded step

Replace model-written raw paths with atomic `{claim, status, evidence_ids}`
records and a compiler-built ID-to-typed-path table excluding null targets.
Require structured entry, stop, target formula, horizon, cost and fill-known
fields. Keep unknown inputs explicit and retain independent semantic review:
valid IDs alone cannot establish that a claim follows from its evidence. Do not
repair or rerun these two cases as new successes.

The quant-approved next real test is a separately registered maximum four-case
development sample: hourly/minute × H3 pass/reject, first chronological eligible
case in each frozen stratum when populated, without looking at outcomes. Missing
strata stay missing. Measure disagreement with H3; never force it. This tests
the narrow binding/completeness correction against the same mechanical comparators
without another synthetic-only milestone. Only subsequent preregistered evidence
can assess improved reliability or incremental selection value. No optimizer,
new library, production promotion or expansion into all 17 specialists is justified
by this two-case result. This next interface and experiment are not implemented
or launched by the current pilot.
