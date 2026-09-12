# Evidence-ID hourly/minute entry pilot — September 12, 2026

## Result

The bounded pilot is complete. All three specialist answers passed the new
deterministic evidence-ID and exact-plan contract; all fifteen fixed factual
fields were correct. None passed the full semantic-review gate. Review oversight,
recorded before outcomes, found both genuine missing support and critic overreach.
All three registered hypothetical entries subsequently hit their stops.

This advances checkable research assessments, not profitable trading. There is
no validated agent selection or demonstrated advantage over the fixed comparator.
No assessment, review, gate or trade plan was repaired after seeing results.

## Scope and selection

The [prospective specification](../superpowers/specs/2026-09-12-evidence-id-entry-pilot.md)
froze April 2026, `[April 1 00:00, May 1 00:00)` UTC, with exactly thirty days
of seed history starting March 2. April had no prior model case assessments,
but its archive had already been researched: these are development cases, not
a pristine holdout or out-of-sample strategy test.

The unchanged hourly LC eligibility and separate causal minute sweep/reclaim
detector produced the following complete classification ledger:

| Track | Eligible | Fixed 4H/N3 H3 pass | Reject | Unknown | Selected |
|---|---:|---:|---:|---:|---:|
| Hourly LC | 3 | 0 | 3 | 0 | 1 |
| Minute sweep/reclaim | 242 | 120 | 122 | 0 | 2 |

Each available track × pass/reject stratum contributed its first chronological
candidate. There was no hourly-pass case to select; the planned maximum of four
therefore became three, with no replacement, different month or outcome-based
selection. Native replay retained all seventeen champion definitions and full
prehistory. Its 1,440 hourly updates completed in 756.93 seconds. Three existing
example-YAML missing-name errors were retained; they did not remove any intended
champion definition.

Exactly six fresh role invocations requested as `gpt-5.6-sol`/high completed, assessor then
independent one-case reviewer for E01, E02 and E03, without retries. Actual model
snapshot and billed token use were not observed. The existing
master-authored rulecards supplied teaching/code context; this was not a new
Astra master invocation or persistent master runtime. Completed 1m, 5m, 15m,
1h, 4h and daily evidence was available. Missing macro/derivatives observations
remained missing, not inferred confirmation.

All 66 captured chunks passed inner-runtime delivery checks. Fresh contexts and
outcome withholding are workflow separation, not enforced filesystem isolation,
proof of model attention, or proof against pretraining contamination.

## What changed

Commit `5c7b9b1` adds
[`evidence_id_assessment.py`](../../scripts/research/evidence_id_assessment.py)
and [67 tests](../../tests/research/test_evidence_id_assessment.py).
The compiler assigns stable IDs to nonnull evidence paths, retains observed
zero/false values and available parent-lineage fields, and exposes candle rows
and their columns. Specialists submit six categories of claim records plus an
exact entry/stop/target/horizon/notional/cost/deadline plan. Invalid or tampered
catalogs, unknown IDs and mismatched plans fail deterministically.

The compiler checks references and structure, not whether a cited fact supports
an interpretation. Semantic review remains necessary. Pre-call regression fixes
also covered the actual reviewer-packet builder, source instrument parity and
OHLCV hashing before feature enrichment. No production code, fusion threshold,
archetype gate or position-management rule changed.

## Assessment and review findings

| Case | Raw choice | Uncalibrated estimate | Mechanical contract | Frozen review | Usable selection |
|---|---|---:|---|---|---|
| E01 hourly / H3 reject | Accept | 0.54 | Pass | Fail | Null |
| E02 minute / H3 pass | Reject | 0.41 | Pass | Fail | Null |
| E03 minute / H3 reject | Reject | 0.46 | Pass | Fail | Null |

The independent quant guide audited the allegations before outcome reveal:

- **E01:** Interpretive claims omitted required teaching IDs, and native identity,
  provenance and structural cancellation were underexplained. But the critic
  incorrectly called an explicitly uncalibrated estimate calibrated, invented
  collector/H1 acceptance requirements for this offline task, disputed ambiguous
  “closed lower” wording as conclusively false, and failed numerics despite
  correct fixed facts and plan numbers. Some other omissions were overstated.
- **E02:** A claim asserted the permitted post-sweep timing sequence without
  citing its first-sweep timestamp. That timestamp existed elsewhere in the
  packet, but not in the claim's cited support. All five factual criteria passed;
  this claim-specific citation failure coherently failed the strict review gate.
- **E03:** Correct facts showed the child above the parent midpoint, failing the
  named lower-half condition. The claim was labeled “contradicted.” The interface
  had not explicitly distinguished support for the whole factual proposition
  from failure of the named condition. The polarity objection is defensible;
  labeling it invented factual content overstates the defect.

All original failures remain unchanged. Zero review passes is not zero factual
accuracy, evidence of poor trading judgment, or a reviewer precision/recall
estimate. Null selection is not rejection, a zero-return skip, or avoided-loss
credit. Economic outcomes cannot retroactively validate a critic's allegations.

## Locked hypothetical outcomes

Starting equity: unspecified. Each case uses independent $50,000 notional and
$60 round-trip cost; no portfolio or compounded equity curve. Average initial
price risk across these three hypothetical entries was **$248.62**, excluding
costs. Each fixed long enters at the decision-minute open, targets actual entry
+2R, and expires after 1,440 minutes hourly or 240 minutes minute-scale.

| Case | Decision time UTC | Entry | Initial price risk | Stop exit UTC | Net after $60 cost |
|---|---|---:|---:|---|---:|
| E01 | Apr 4 16:00 | 67,357.40 | $411.85 | Apr 5 06:07 | −$471.85 |
| E02 | Apr 2 03:53 | 66,610.80 | $191.10 | Apr 2 04:23 | −$251.10 |
| E03 | Apr 1 00:16 | 68,067.90 | $142.92 | Apr 1 00:28 | −$202.92 |

All paths were scorable with no stop/target-ambiguous bar. Opening gaps precede
intrabar touches; otherwise simultaneous touches use stop-first. Deadline exit
uses the deadline open only. E02 finished its full horizon above entry, but had
already stopped out after thirty minutes: later recovery is not a winning trade.
Reported full-horizon excursions include prices after hypothetical exit.

E01 was emitted by the native hourly replay and passed the numeric comparator,
but H3 rejected it. The raw agent accepted it. E02 passed the separate minute
detector and H3, but the raw agent rejected it. E03 passed the minute detector,
but both H3 and the raw agent rejected it. The minute records' legacy
`native_emitted=true` denotes detector emission, not a native minute champion
engine or an executed order.

Thus the raw hourly acceptance lost; both raw minute rejections aligned with
losing counterfactuals. This does not establish incremental value. All agent
selections are formally null, the strata are deliberately selected rather than
natural prevalence, and three cases cannot estimate policy expectancy. The
always-skip control would have no exposure, but invalid assessments cannot be
silently assigned to that control. No formal agent strategy PnL is reported.

Execution remains uncertified: zero added model delay, no funding, market impact,
inference expense or portfolio interaction. This is not the engine's live PnL.

## Verification and handoff

Fresh root verification: **570 passed**, one existing LibreSSL warning, 12.46s:

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -o addopts='' -q tests/research results/evidence_id_pilot_2026_09_12/test_selection.py results/evidence_id_pilot_2026_09_12/test_review_packet.py
```

This includes 489 existing research tests, 67 new evidence-ID tests, thirteen
private selector tests and one private reviewer-packet regression. Re-running
the unchanged reveal command reproduced identical outcomes and grades. Private
artifacts live under `results/evidence_id_pilot_2026_09_12/`; they and the source
archive remain ignored/local. Tracked code and this report alone are insufficient
to reproduce the historical/model calls on another machine.

The independent post-reveal quant audit also verified the source archive,
assessment/review/preparation locks, reveal authorization and embedded frozen
grades, and reproduced all three scorer output objects exactly. The $0.10
difference between indicative close and actual reference open in E01/E02
correctly accounts for their small indicative-versus-scored risk differences.

| Artifact | SHA256 |
|---|---|
| Source minute archive | `5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035` |
| Prospective specification | `79b1c07f07ec9a588384463998d01afc4ef9b3ec1e507426672c4b33ccf897d1` |
| Prepared source ledger | `077e4ee17ec25b7ad39ff6d07c2ba0980ea3e9e2b16d41f115d99ebd6eb91854` |
| Final manifest | `04359346b3eb8cfc0087d77ced5efb66896e728f2a785040525df758c1f50975` |
| Pre-outcome grades | `c8c73c90596cd5103842b49a3878b700f4b3413f0e5a0f3b69a2df6b0dbd89b7` |
| Pre-outcome oversight lock | `95a1decf6d0b8d5e4ea9104809349b42d977bb2ae8606eaae9a0770e665d8d12` |
| All pre-outcome records lock | `33b803b64eaa02e94ba6c45da891134123c4d3672e0d45cb3ac5b35423770ba6` |
| Repeated outcomes | `38961648834e7a02a731338df6efe2e7d2995c7de8788cd3c46bcbd2febdd39f` |

## Next bounded step — not started

Clarify the critic contract using these already locked allegations as explicitly
labeled development examples: claim status refers to the whole proposition;
condition pass/fail is separate; requested uncalibrated judgments are permitted;
each failed criterion must identify an actual defect; ambiguous wording is not
automatically contradiction or invention. Preserve these grades and outcomes.

Do not spend on another market sample until that small critic-contract correction
has been reviewed. Then separately preregister the next equal-priority
hourly/minute comparison. No new market cases, live changes, strategy promotion,
push or PR were started in this completion step.
