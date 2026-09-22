# Why the agent missed LC rebounds — exposed-case diagnosis

## Finding

The agent's handling of bearish larger structure was insufficiently discriminating
on the ten assessed downside cases. It rejected all ten, including five winners.
That does not prove those decisions were irrational ex ante or identify a safe
new entry gate. It shows that this version did not separate successful rebounds
from continuing liquidation on this sample.

The existing curriculum already prohibits universal multi-timeframe agreement,
same-hour reclaim or intact-parent gates. Adding that warning again is not a
substantive repair. We need to specify the evidence that can overcome adverse
context for a particular rebound thesis, and test that judgment independently.

## Source-to-code-to-agent chain

1. [Saved trader-teaching comprehension](trader_teaching_comprehension_2026_09_11.md)
   distinguishes locally attributed Wyckoff/confirmation notes from authenticated
   primary doctrine and from project-invented thresholds. Existing synthetic
   comprehension results are not economic validation.
2. [LC rulecard](master_specialist_rulecards_2026_09_12.md), section B, describes
   volume exhaustion/absorption as a candidate, not proof of institutional intent.
   Supporting evidence includes location, ordered rejection/reclaim and an
   identifiable destination. Opposing HTF direction alone does not prohibit a
   shorter reversal. [Moneytaur source notes](moneytaur_primary_sources_2026_09_12.md)
   support investigating level, horizon and confirmation, not a universal HTF veto.
   No new primary-source retrieval or authentication was performed in this pass.
3. `engine/archetypes/logic.py::_check_E` accepts positive climax/absorption OR
   volume-z>2 plus RSI<35 or>65. Its implementation does not establish a preceding
   compression sequence, reversal confirmation or target reachability. Both
   directional geometries can produce long candidates. No native code was changed.
4. The actual frozen `curriculum_v1/master_brief.json` supplied to these agents
   explicitly says no mandatory multi-timeframe agreement, minimum room,
   same-hour reclaim or intact-parent gate. It also says the wait plan is a
   prospective one-minute close above the frozen five-minute high, not a completed
   five-minute retest. This is controller synthesis, not verbatim trader doctrine.
5. Saved raw answers acknowledged local rebound evidence but generally judged
   it insufficient against the larger decline, broken floors and recovery burden
   of the fixed24h/2R plan. They often explicitly denied using a universal veto.
   Behavioral rejection of every downside candidate is nevertheless the result.
   Therefore the observed issue is judgment weighting, not proven ignorance of
   the written teaching or absence of all higher/lower-timeframe evidence.

The raw supporting arguments make this distinction concrete: Feb28's assessor
identified the earlier62401.7 daily low beneath the new62979.5 low; Mar08's
identified an intact daily range and a sequence of rising local lows after the
65569.2 selloff low; Mar22's identified a location about27% up the daily range
and an intrahour rejection/recovery. All three still chose reject. These are
assessor descriptions of supplied evidence, not a new independently authenticated
market-data audit or sufficient conditions for buying.

## All ten assessed downside cases

Fixed $50k notional,12bps,90seconds; historical modeled results, not live trades.
The other downside candidate, Feb23, failed dispatch and is excluded from this
diagnostic table **only**, not removed from the experiment denominator.

| UTC date | Immediate net | Confirmation net | Agent | Relevant predecision observation |
|---|---:|---:|---|---|
|Jan29|−1,025.82|Expired|Reject|Both parent floors broken; last1m and5m improved|
|Jan31|−1,044.54|−1,288.97|Reject|4H floor broken; last1m and5m improved|
|Feb28|+2,814.51|+2,890.00|Reject|Daily floor broken; last1m improved, last5m declined|
|Mar07|+95.39|+75.29|Reject|Both parents intact; final minute weakened|
|Mar08|+2,078.34|+2,217.13|Reject|4H floor broken; last1m and5m improved|
|Mar22|+1,238.94|+1,667.41|Reject|4H parent absent; final minute improved|
|May28|+7.24|Expired|Reject|Daily floor broken; mixed final minute|
|Jun01|−758.81|Expired|Reject|4H floor broken; hourly prior-low reclaim present|
|Jun02|−1,032.99|Expired|Reject|Parents absent; final1m and5m weakened|
|Jun30|−36.57|Expired|Reject|4H floor broken; final1m and5m weakened|

Immediate subtotal+$2,335.70; mechanical confirmation+$5,560.86; agent$0 on
these ten deliberate rejects. Confirmation retained4/5 winners, avoided4/5
losers, but worsened the Jan31 loss. This table is post-outcome diagnosis,
not a new validation sample or a sufficient trading rule.

## What the evidence rules out—and what it does not

- **Not simply insufficient target room:** Feb28, Mar08 and Mar22 reached the
  original2R target within24h. The agent's concerns were plausible obstacles,
  not reliable forecasts that the targets could not be reached.
- **Not a clean parent-state filter:** broken4H floors occurred in both Jan31's
  loss and Mar08's winner. An absent4H parent occurred in Feb28/Mar22 winners
  and Jun02's loss. A parent label alone does not separate them.
- **Not a clean last-minute filter:** improved last1m/5m candles appeared in
  Jan29/Jan31 losers and Mar08's winner. Confirmation *after* the cutoff is
  distinct evidence from two improving candles *before* it.
- **Not a defensible mandatory same-hour reclaim:** none of the five winners
  reclaimed the previous hourly low by decision; Jun01's losing candidate did.
  Introducing that condition to these cases would move in the wrong direction.
- **Not proven optimal24h/2R management:** three target hits refute a blanket
  mismatch explanation, but do not establish optimal stop, target or horizon.
  We did not search alternate exits or inspect maximum-favorable-excursion targets.
- **Not yet a causal diagnosis of model failure:**17 unreviewed raw answers,
  fixed menu and one prompt version cannot disentangle curriculum, information,
  response incentives and model calibration. Full derivative/receipt evidence
  remains unavailable; do not attribute missing orderflow to observed absorption.

## Actionable research implication

The [broader chronological comparison](lc_mechanical_results_2026_09_22.md) is now
complete: confirmation reduced the primary loss but remained unprofitable and
period/delay-sensitive. Keep it as a benchmark, not an approved live fix. A
future agent should compare two explicit competing
theses—temporary exhaustion/rebound versus continued liquidation—and state what
future observation would favor each, at the intended execution horizon. Larger
structure supplies context and obstacles; it is neither irrelevant nor an
automatic trend-alignment requirement.

That is a **candidate instruction design**, not a implemented or validated repair.
Do not tell the next agent which exposed cases won, instruct it to accept more,
or optimize cutoffs against this table. Any revised judgment needs separate
approval, a frozen instruction set and different, outcome-hidden evaluation cases.

Audit trail: private `results/lc_single_bounded_2026_09_20/run_v1/` manifest,
per-case raw responses and `economic_result.json`; original sealed requests under
`results/lc_consolidated_2026_09_15/judgment_v1/evidence/`; actual frozen curriculum
under `results/lc_context_discrimination_2026_09_15/curriculum_v1/`.
