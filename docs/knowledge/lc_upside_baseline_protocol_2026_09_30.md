# Upside LC mechanical baseline protocol

September 30, 2026. The user approved prioritizing one mechanical edge and
pausing agent trading development. This first bounded deliverable establishes
an isolated upside LC baseline using the existing replay, not a new engine.
No market assessors, live changes, new dependencies or parameter search.

## Scope and fixed calculations

- Reuse the frozen 142-case January 2024 through July 2026 census and its existing
  predecision subtype labels. Select every upside-expansion candidate before
  replay; retain unavailable upside cases rather than dropping them. Show all
  source subtype counts. This is conditional on the reconstructed source census,
  not a fresh continuous native detector or the full live runner.
- Replay the existing immediate policy as the baseline in its own single-position
  book. Replay the existing five-minute-high confirmation policy as a diagnostic
  comparator, separately. These are not the two new variants in the broader plan.
- Preserve the source-close minus 2.7 ATR14 stop, actual-fill-based 2R target,
  $50,000 fixed notional, 15-minute entry expiry and original 24-hour deadline.
  Reuse all four existing cost/delay scenarios: 12/24bps and 90/300 seconds.
  Primary is 12bps/90 seconds. No funding, impact, compounding or live scale-outs.
- Verify frozen source/code/archive hashes and rebuild causal plans from the
  two completed predecision hours. Compare all isolated resolutions with the
  prior full-book rows and with the existing direct conditional scorer.
- Report fills, wins/losses, profit factor, net PnL, calendar-year and quarter
  contributions, dollar marked-to-market drawdown for the primary scenario,
  top-three-winner concentration, and costs/delay sensitivity.
- Normalize each closed result by its initial stop-distance dollar risk plus
  modeled round-trip costs. Report net R per fill; this is a linear normalization,
  not a funded equal-risk portfolio or guaranteed maximum loss through gaps.
- Descriptive uncertainty: 5,000 calendar-month cluster bootstrap resamples,
  fixed seed 20260930, 2.5th/97.5th percentiles of mean net R per fill. Include
  all 31 calendar months, including zero-trade months. Preserve whole months,
  not independently shuffled trades. Unknown outcomes make full metrics and
  intervals unavailable. This diagnostic assumes exchangeable month blocks,
  does not model all cross-month dependence, and is not selection-adjusted.
- No favorable group, threshold or variant will be selected from this scorecard
  and called validated. The entire cohort is previously exposed development data.

## Deliverable and stopping point

Produce one reproducible scorecard, preserve the old run unchanged, and record
the approved research direction in project continuity. Check the specific source
and live-parity limitations rather than claiming full-engine correctness.
The next stage is a small source-backed location/room diagnostic followed by
predeclared variants, not an automatic paid assessment or live promotion.

The wider approved plan remains: one baseline plus at most two new variants,
chronological validation, then a promising/inconclusive/unsupported decision.
Fresh validation requires demonstrably unexamined or prospective observations;
repartitioning these 142 cases cannot manufacture an untouched holdout.
