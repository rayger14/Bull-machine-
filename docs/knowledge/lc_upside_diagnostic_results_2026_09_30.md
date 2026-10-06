# Upside LC structure and matched breakout results

September 30, 2026. The bounded comparison is complete. Upside LC outperformed
similar non-emitted-LC hourly breakouts in this exposed sample, but the uncertainty
still includes no advantage. Mapped overhead distance is a useful next research
hypothesis, not an enabled filter or a proven edge. No live rules changed.

## What we tested

We recovered all 142 original source candidates exactly and retained the same
68 upside LC events. The fixed matching rules selected 65 earlier breakout
controls; three LC events had no match and were retained explicitly. Control
selection and six structural descriptions were saved before new control outcome
scoring. See the [fixed protocol](lc_upside_diagnostic_protocol_2026_09_30.md).

Every event had a complete minute window. All 133 hypothetical entries filled
under the unchanged immediate research policy: $50,000 notional, source close
minus 2.7ATR stop, actual-fill 2R target, 90-second processing rounded to the next
minute, 12bps roundtrip costs, and a decision-relative 24-hour deadline. These
are independent event sums, not funded portfolio returns or native live results.

Here, R is net profit divided by initial stop-distance dollar risk plus modeled
fees. It helps compare different price/volatility periods; it is not a promise
that a real loss cannot exceed one R.

## LC versus matched ordinary breakouts

| Primary comparison | LC | Control breakouts |
| --- | ---: | ---: |
| Matched events | 65 | 65 |
| Winners / losers | 27 / 38 | 30 / 35 |
| Net event sum | $10,210.97 | $1,856.12 |
| Profit factor | 1.425 | 1.086 |
| Mean net R per event | 0.1799 | 0.0294 |

Mean paired advantage was **+0.1505R**. The descriptive calendar-month bootstrap
95% interval was **−0.1905R to +0.5297R**. This does not establish a dependable LC
advantage. The comparison is exploratory, not selection-adjusted or randomized.
LC won less often than the controls; its better aggregate payoff is why win rate
alone is not the right objective.

All 68 LC events still total +$9,384.99, exactly matching the prior isolated
baseline. The matched table omits three cases from the *contrast*, not from the
full LC scorecard. Their combined result was −$825.98.

| LC decision year | Pairs | LC event sum | Control event sum | Mean paired difference |
| --- | ---: | ---: | ---: | ---: |
| 2024 | 25 | $4,106.67 | $6,031.91 | −0.0737R |
| 2025 | 25 | $1,668.61 | −$2,819.72 | +0.3292R |
| 2026 January through July | 15 | $4,435.70 | −$1,356.06 | +0.2264R |

The advantage was not consistent across years. Full-horizon price movement from
the delayed entry averaged +0.4034% for matched LC versus +0.0044% for controls;
mean maximum favorable excursion was 2.1446% versus 1.6083%, and mean adverse
excursion −1.5364% versus −1.5470%. These excursions continue after hypothetical
bracket exits and are not achievable strategy profits.

## Structural patterns in all 68 LC cases

These categories were fixed before this outcome association. They are descriptive
splits of the same sample, not separately trained strategies or multiple successful
replications. We did not search combinations.

| Nearest mapped boundary above setup close | Cases | Winners / losers | Net sum | Mean net R |
| --- | ---: | ---: | ---: | ---: |
| At least two source stop-distances away | 14 | 9 / 5 | $4,367.77 | +0.4412 |
| Less than two stop-distances away | 34 | 10 / 24 | −$1,199.30 | −0.0042 |
| No mapped overhead reference | 20 | 9 / 11 | $6,216.52 | +0.2025 |

Interpretation: room above entry is worth testing next. But the 14-case group is
small: removing its three biggest winners leaves only +$212.41. Its 2025 dollar
sum was negative even though its risk-normalized mean was positive. The closer
boundary group was positive in 2025 and negative in 2024 and 2026. None of this
justifies a production filter yet.

“Mapped boundary” is the high or low of a validated 4H/daily range established
strictly before the setup; even a broken range can remain a historical reference.
It is not proof of resistance or an unobstructed route. No mapped reference does
not mean unlimited room. A rule requiring a known distant level would also exclude
the profitable 20-case no-reference group, and that lost opportunity must be counted.

| Last two completed 5m bars | Cases | Net sum | Mean net R |
| --- | ---: | ---: | ---: |
| Both higher low and higher close | 29 | $8,881.25 | +0.3186 |
| Other sequence | 39 | $503.75 | +0.0216 |

The first sequence looks useful overall, but its annual means were +0.3412R,
+0.4693R and **−0.0027R**. The other sequence's 2026 mean was +0.3200R. Therefore
do not promote the five-minute pattern as a universal confirmation gate. It is
predecision sequence, not the previous postdecision wait-for-close entry rule.

| Pre-existing parent lifecycle | Cases | Net sum | Mean net R |
| --- | ---: | ---: | ---: |
| 4H intact | 25 | $2,164.17 | +0.1372 |
| 4H broken upward during setup | 25 | $277.54 | +0.0554 |
| 4H absent under the N3 definition | 18 | $6,943.28 | +0.2926 |
| Daily intact | 41 | $6,656.34 | +0.1960 |
| Daily broken upward during setup | 4 | −$3,001.30 | −0.7249 |
| Daily absent under the N3 definition | 23 | $5,729.95 | +0.2151 |

Requiring an intact 4H parent would discard the profitable absent-parent group;
the four daily-break losses are too few to establish a veto. Absence under this
specific pivot algorithm does not mean the market has no larger structure. In
this sample, parent location exactly duplicates the lifecycle partitions: intact
means inside, broken-up means above, and absent remains absent. Those are not two
independent confirmations. All six full tables, zero-size categories and annual
breakdowns are retained in result.json. There were no unknown categories here.

## Matching quality and limits

Controls matched the same UTC hour and pre-setup 24-hour trend sign, with prior
ATR/price within a factor of 1.5. The control-to-LC volatility ratio had median
0.99984 and range 0.77842–1.35578. Mean prior 24-hour return was +1.3781% for LC
and +1.4293% for controls; standardized mean difference was −0.0390. Matching
trend sign is not matching all structural or macro conditions. Median control
lag was 44 days, range 1–90 days; residual time/regime confounding remains.

No control was reused or selected because of its subsequent outcome. “Non-emitted
LC” is not proof that no LC-shaped pattern occurred: gates/cooldowns can suppress
native emissions. There were 16 overlapping pairs of 24-hour event windows across
the combined sample, though none within an LC/control pair. Monthly resampling
does not fully model cross-pair or cross-month dependence.

Unmatched LC decisions were January 28, 2024 05:00 UTC; April 21, 2025 01:00 UTC;
and July 26, 2026 23:00 UTC. The rules were not relaxed to include them.

All history remains exposed development data. Source monthly cold starts,
defaulted derivative inputs, regime fallbacks, missing live receipt evidence,
flat cost assumptions, and differences from native targets/scale-outs remain.
No claim of complete Wyckoff interpretation, live parity or industry certification.

## Decision and next action

Keep the baseline and live engine unchanged. The first candidate to freeze for
the next bounded test is a **known mapped room of at least 2R** rule, using exactly
the current causal feature definition, with unknown/no-reference cases kept
separate rather than treated as infinite room. This is a proposed research rule,
not yet an implemented or validated strategy.

Compare it with the unchanged baseline and, at most, the same rule plus the
existing minute confirmation entry. Do not add a blanket intact-parent veto,
the four-case daily veto, fusion tuning or a new collection of fitted thresholds.
Record all rejected winners and trade scarcity. Audit data exposure and freeze
the validation population before any fresh outcome scoring; if no genuinely
unexamined chronological data exists, collect prospective shadow cases. A failure
or inconclusive result is a valid stopping result, not permission to keep searching
this same sample until it wins. No paid market-agent work is needed for this step.

## Verification and reproducibility

The corrected run is `results/lc_upside_diagnostic_2026_09_30/run_v2`.
It contains source-only features, contexts, matches and plans; preflight.json;
all 133 event outcomes and six tables in result.json; and verification.json.
The archive and source/result artifacts remain local-only dependencies.

- Preflight digest: `70658bacca291a6f1b161a56344d0f282584f4a142fc7363abfa345cf0beba73`.
- Result file SHA256: `10b0af571fcc4ab6d93b73e749af1f9d8c8fa7bd4edc518068c6d6b192b91afb`.
- 140 file bindings, including all 120 original bindings, checked before/after.
- 142 reconstructed cases and source/current/previous ATR parity; 31 months,
  22,632 decision-indexed hourly feature rows.
- Independent exhaustive matching agrees for all 68 cases: 65 controls, 3 unmatched.
- All 133 complete windows and bracket entry/exit/fee/risk records independently
  recomputed from raw minutes, with zero ambiguous bars; all 68 LC PnLs match the
  previous isolated baseline. Six partitions and paired bootstrap reconcile.
- 260 focused tests pass, including 24 new tests. Bare repo pytest still exits 3
  during collection at `tests/test_integration_fixes.py` because
  `configs/baseline_wyckoff_test.json` is absent. No full-suite green claim.
- A collection-only diagnostic excluding that aborting file collected 2,351 tests,
  not executed, and reported the same ten existing import failures:
  `tests/archetypes/test_bull_archetypes_mvp.py`,
  `tests/archive/versions/v170/test_macro_pulse.py`,
  `tests/archive/versions/v170/test_v17_integration.py`,
  `tests/integration/test_macro_backtest.py`, `tests/integration/test_wiring_gates.py`,
  `tests/test_macro_backtest.py`, `tests/test_multi_position.py`,
  `tests/test_multi_position_full.py`, `tests/v170/test_macro_pulse.py`,
  `tests/v170/test_v17_integration.py`.

One read-only software/accounting review found an Important copied-directory
binding flaw. Two tests reproduced it before the fix; the scorer now requires
the actual consumed paths to be bound. Source-only run_v1 is preserved, never
economically scored, and superseded. The corrected run_v2's selections, plans,
features and contexts are exactly identical. No rule was changed after seeing
control outcomes. No additional reviewer or market assessor was launched.

Reproduce from repository root with the same pinned local inputs and code:

```sh
python3 -m scripts.research.run_lc_upside_diagnostic prepare
python3 -m scripts.research.run_lc_upside_diagnostic score
```

Exact-equal reruns are permitted; unequal overwrites fail. Frozen files must not
be edited and resealed to make a rerun pass. Code/tests/docs remain local and
uncommitted on quant/archetype-evidence-audit at HEAD85923a4. No push, PR, live
changes, paid trading assessments or work left running at this checkpoint.
