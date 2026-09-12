# Four-case blinded agent entry study

## Result

One fresh assessor received code, completed local/4H/daily candles, selected
features, child events and available parent records. It made one accept, one
reject and two insufficient-evidence decisions before subsequent prices were
scored. The accepted hourly entry reached its fixed 2R target. This is an actual
conditional entry/outcome experiment, but **does not demonstrate agent advantage**.

The hourly decisions exactly matched the saved emission mask and numeric gate
baseline. Minute abstentions cannot be credited as demonstrated loss prediction:
the agent lacked important detector evidence and mixed in an hourly gate.

## Frozen experiment

See [protocol](../superpowers/specs/2026-09-11-agent-entry-mini-study.md) and
[pre-outcome clarification](../superpowers/specs/2026-09-11-agent-entry-mini-study-addendum.md).
Within the already-researched June 10–20, 2026 population, take the earliest
structural-pass/non-emitted hourly LC and earliest emitted hourly LC after the
first 24 replay hours; then take the earliest minute detector event on each of
those UTC dates. Selection uses strata/time, not outcome. C2 was previously
examined in the hourly pilot; this is not a pristine holdout.

The assessor saw neither these strata labels nor emission/baseline masks,
outcome-bearing filenames, or subsequent prices. It received exactly twelve
local candles, six completed 4H candles and three completed daily candles per
case. Root checked every displayed candle closes by decision, setup closes
match the child record, bound parents precede first sweep, and displayed
transitions do not exceed decision. V23 hourly and Binance minute streams stay
separate. Actual source-receipt certification and external macro/derivatives
were not supplied; missing certification alone did not mandate abstention.

Plans were fixed before assessment: long at next same-source open, hourly stop
at preceding close minus 2.7 native ATR, minute stop 0.15% below sweep low, and
target at actual entry plus twice its stop distance. Horizons: 24 hours hourly,
240 minutes minute. Entry-through-predeadline bars include bracket touches;
deadline uses only its open. Each case independently uses $50,000 notional and
$60 round-trip costs (12bps). Starting equity is unspecified: no compounding,
book, funding, market impact or model-latency simulation. Initial risk below is
stop-distance dollar risk before costs, not an account-risk percentage.

The reviewer flagged ambiguous prose about opening gaps versus both-hit bars.
A dated addendum, preserved alongside the original freeze, clarifies the
already-frozen scorer: known opening touches precede later intrabar extremes;
otherwise both-hit bars are conservative stop-first. No case used the disputed
favorable-opening-gap/both-hit exception, and no scored exit was ambiguous.

## Decisions and subsequent outcomes

All times UTC. Outcome column scores **every candidate**, including rejected
and abstained candidates; these are not all agent trades or actual live fills.

| Case / decision time | Agent | Emission baseline | Mechanical baseline | Initial risk | Fixed-bracket outcome, net |
|---|---|---|---|---:|---:|
| C1 hourly June 11 13:00 | reject | no | fail | $975.82 | deadline, +$296.81 |
| C2 hourly June 14 22:00 | accept | yes | pass | $691.75 | 2R target, +$1,323.50 |
| C3 minute June 11 02:25 | insufficient | yes | pass | $106.61 | stop, −$166.61 |
| C4 minute June 14 02:35 | insufficient | yes | fail | $109.35 | stop, −$169.35 |

Hourly emission means saved native signal emission, not proof of an executed
trade. Minute emission means this separate research detector, not the live
hourly engine. Hourly mechanical baseline uses finite volume/RSI/BB/chop gates
plus H2; minute baseline uses the fixed 4H/N3 H3 permission evaluator.
Separately, the frozen optional hourly H3 subtype returns **reject for both C1
and C2** (neither is unknown). It is a restrictive parent-floor reclaim test,
not a universal hourly gate: C2 instead breaks above the parent high. Its
rejection does not override the native/numeric hourly comparison.

Hourly native/mechanical/agent arms each accept C2 only: +$1,323.50, average
initial risk $691.75. The two minute detector cases total −$335.96, average
initial risk $107.98; minute H3 accepts C3 only, −$166.61 with $106.61 risk.
The agent accepts zero minute cases, abstains twice and rejects none: no
minute realized hypothetical PnL or average taken-trade risk to estimate.
There are no invalid plans. No cross-track portfolio sum, win-rate estimate,
Sharpe, significance or claim of independent observations is warranted.

| Case | Entry | Stop | Target | Exit UTC | Full-horizon best / worst move | Deadline return |
|---|---:|---:|---:|---|---:|---:|
| C1 | 62,939.15 | 61,710.80 | 65,395.85 | June 12 13:00 | +1.499% / −1.086% | +0.714% |
| C2 | 65,280.67 | 64,377.51 | 67,086.99 | June 15 15:00 bar | +3.038% / −0.570% | +1.616% |
| C3 | 62,070.50 | 61,938.15 | 62,335.19 | June 11 02:32 bar | +1.487% / −0.256% | +1.414% |
| C4 | 64,524.20 | 64,383.08 | 64,806.44 | June 14 04:39 bar | +0.013% / −0.409% | −0.376% |

Excursions deliberately continue after bracket exit. Intrabar exit times are
bar-open labels, not exact fill timestamps. C3 hit its stop at 02:32 before
first reaching the target price in the 03:33 bar: its positive four-hour move
does not rescue the specified entry. C2's target was first touched in the
June 15 15:00 bar; its stop was not touched anywhere in the 24-hour window.

## What the reasoning did and did not establish

- **C1:** The agent identified failed RSI/chop hard gates and overhead levels.
  Rejecting was rule-consistent even though the candidate later had a modest
  positive deadline return. A profitable counterfactual does not make a gate
  violation a correct qualifying entry; equally, this does not establish that
  those gates improve expected returns.
- **C2:** It interpreted rising daily structure and the high-volume break above
  64,727.27 as continuation, explicitly acknowledging exhaustion/retracement
  risk. It did not mistakenly require a parent-floor reclaim subtype for a
  breakout. The fixed target was reached, but the engine and numeric baseline
  already selected the same candidate. No incremental selection was shown.
- **C3:** It correctly recognized preexisting parent/lower-half geometry, but
  abstained because pivot confirmation, earlier matching touches and cooldown
  could not be reconstructed from twelve minute candles. These are real packet
  gaps. Its discussion of missing compression/H2 also leaks an hourly-specific
  rule into minute qualification: H2 is not a required gate of this detector.
- **C4:** It recognized shallow reclaim and missing parent evidence, but again
  referenced hourly H2/compression. Parent `null` alone does not tell it whether
  no eligible parent exists or extraction evidence is missing; the hidden H3
  result rejects. This ambiguity belongs in the evidence contract, not in a
  narrative confidence score.

Schema coverage, nonnull citations, text limits and indicative target arithmetic
were checked. Prose was manually inspected, not given an automated accuracy
score. Confidence labels are uncalibrated. A single context-free invocation
reduces conversational leakage but cannot prove absence of model-memory
contamination; exact model snapshot, token accounting and latency are unavailable.

## Implementation and reproducibility

New reusable code: `scripts/research/entry_case_outcome.py`, with 15 hand-fixture
tests failing before implementation and passing afterward. Independent review
requested two more regression cases (malformed deadline HLC ignored; new
post-exit extrema included); both pass without changing the frozen scorer.
Full research suite: **399 passed**, existing LibreSSL warning. This is not a
full legacy suite run. No production engine/config, live state or trading
authority changed.

Private artifacts under `results/agent_entry_mini_2026_09_11/`: frozen `probe.py`,
`input.json`, `hidden_reference.json`, `manifest.json`, saved `response.json`,
`response_lock.json`, and `outcomes.json`. The response was locked before
`probe.py reveal`; reveal checks frozen source/code/input/response hashes and
refuses unequal output overwrites. Repeated reveal produced identical outcomes.
Root independently checked first barrier touches and PnL arithmetic from raw
bars. Full private extraction artifacts/data are not on GitHub; tracked scorer,
tests and documentation alone do not reproduce this entire study elsewhere.
Final independent quant review approved the evidence/claim separation and next
experiment, subject to explicit hourly H3 disclosure and the completed 399-test
run; both conditions are met. That final review covered the report/addendum,
not an independent reconstruction of private outcome calculations.

- Input SHA256: `4558f2a70a82696b73d4eb04f3a0d22b3af4c8d8771f5170065b2ce6277bab46`.
- Response SHA256: `dc543687bf2f67f9edc08d43b05b349774a03eb37b96f9c63c5b85b28defaabe`.
- Outcome SHA256: `7dae40d1f96e44aa9b57c506987ad6d969be26626adf979c9f94155950a3ce05`.
- Pre-outcome addendum SHA256: `ed757708af30ddb5afb37e2a7a3157099c64039deae74ac9bbf076c9e3e5718c`.

## Next useful experiment, not implemented here

Fix evidence sufficiency before buying more assessor calls: separate hourly LC
and minute sweep rulecards; include timestamped pivot-confirmation, prior-touch,
first-sweep and cooldown witnesses; distinguish no eligible parent from missing
parent data. Deterministic code should validate those witnesses, leaving the
agent to assess location, larger structure and plausible room. Preserve these
four answers unchanged; do not retry them after revealing outcomes.

Then freeze a small new chronological, outcome-hidden batch with complete
packets and compare engine versus mechanical checks versus agent decisions.
Measure coverage/abstention as well as net outcomes and disagreements. Existing
data has already been researched, so call it new assessor cases, not pristine
holdout; genuine forward shadow collection is a later step. No threshold tuning,
new archetype promotion, live integration, push or PR update in this experiment.
