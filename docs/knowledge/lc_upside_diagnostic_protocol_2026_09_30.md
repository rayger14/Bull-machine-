# Upside LC structure and matched breakout study

September 30, 2026. This is the next bounded stage of the approved mechanical-first
plan, not a new strategy or a certification exercise. All 142 source candidates,
including the 68 upside cases, remain exposed development history. No market
assessors, live changes, threshold search, commits, pushes or PRs are authorized.

## Questions and fixed comparisons

Does the upside LC label add anything beyond a similar ordinary hourly upside
break? Which predecision structural descriptions distinguish its winners and
losers? Neither association establishes causation or an investable edge.

Freeze these six categorical descriptions before joining outcomes. No cross-product
search or optimized thresholds:

1. Strict-before-setup 4H N3 parent lifecycle: intact, broken up, broken down,
   broken multiple, absent or unknown.
2. The same daily parent lifecycle.
3. Close below, inside including boundaries, or above that 4H range; absent and
   unknown retained separately.
4. The same daily location.
5. Nearest mapped parent boundary strictly above the setup close, expressed in
   source-close-to-stop risk units: below 2R, at least 2R, no mapped overhead
   reference, or unknown. Consider both high and low of each bound. Broken bounds
   remain historical reference levels, not certified active resistance. Missing
   either parent means unknown; known absence is not missing evidence. This is
   a level-distance proxy, never a claim that the path is unobstructed.
6. Whether the last two fully completed 5m bars have both a higher low and a
   higher close: both, other, or unknown. This is predecision sequence, not the
   existing postdecision confirmation policy.

Reuse the existing causal parent/context validators and record their facts.
Do not turn an upward break into an automatic veto. Report every category,
including small/empty/unavailable groups, and annual contributions. Do not rank
combinations or claim selection-adjusted significance.

## Control selection without outcome inputs

Construct complete hourly candles from the verified same-stream minute archive,
with the source month's independent 30-day seed and TA-Lib ATR14 convention.
Require exact source-candidate OHLC and ATR parity before scoring. Eligibility
is a completed hourly close above the previous hourly high, with no native LC
emission at that decision in the frozen 142-case census. The correct name is
**non-emitted-LC breakout control**: suppressed LC patterns may still be present.

For each upside LC case in chronological order, select at most one control,
without replacement, using these fixed rules:

- Same UTC decision hour, 24 hours to 90 days earlier, within the covered
  January 2024 through July 2026 census months.
- Same sign of the preceding 24-hour return ending at setup open. This excludes
  the setup candle itself.
- Previous-hour ATR14 / previous close must be within a factor of 1.5 of the
  LC case's same pre-setup measure.
- Minimize absolute log volatility ratio; ties prefer the latest eligible time.
- Retain unmatched cases. Do not relax rules after observing match counts or
  economics. Never exclude a control because of a later LC emission.

The 90-day window and volatility caliper are fixed research choices, not universal
standards. Matching does not control every market condition or prove causation.
Report balance, unmatched cases, reused controls (must be zero), and overlap of
the paired outcome windows. Historical controls are not a fresh holdout.

## Economic comparison and uncertainty

Keep the previous immediate reference unchanged: long, source close minus 2.7ATR
stop, actual fill plus twice actual stop distance target, 90-second processing
rounded up to the next minute, zero routing, 15-minute exclusive entry expiry,
decision plus 24-hour exit deadline, $50,000 notional, 12bps flat roundtrip cost.
Stop-first ambiguous minute, adverse open gap, pending-stop cancellation. No
funding, market impact, account sizing, native scale-outs or live parity claim.

Use independent event replays, not a shared LC/control portfolio. Net R is net
PnL divided by initial stop risk plus modeled fees. Resolved nonentries contribute
zero per-candidate R; missing/invalid outcomes remain null and invalidate the full
matched contrast. Primary contrast is mean paired LC minus control net R.
Resample pairs by their LC calendar month, including all 31 months, with 5,000
draws and seed 20260930. Report a descriptive 95% interval, not selection-adjusted
or a full dependence model. Past-only pairing avoids within-pair overlap, but
different pairs can overlap and span months. Also report dollars, wins/losses,
and raw full-horizon excursions as diagnostics, not achievable profits.

## Reproducibility and stopping rule

Save source-only features, all control candidates, selected IDs, matches, code,
protocol and original input bindings in an immutable preflight before the score
command can read outcome windows. Verify bindings again before and after scoring.
Keep prior helpers/protocols/artifacts byte-identical. Save full per-case results
and all six tables, then summarize the actual findings in a separate report.

This stage ends with the comparison and an explicit next decision. At most two
new variants may be frozen in a subsequent step, one structural/location rule
and that same rule with a minute trigger. A weak, inconsistent or tiny-sample
association is not sufficient reason to enable a filter. Fresh validation is
still required; do not repeatedly optimize this exposed sample.
