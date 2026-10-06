# R3 backtest results and next steps

The frozen minute-scale R3 study is complete. The decision is **park this version**:
waiting for a retest reduced total losses compared with buying the breakout, but
it did not produce a profitable strategy. The separate hourly R1 experiment is
blocked by missing native model files. Neither result changes live trading.

## What was tested

Both methods used the same 13,497 BTC setup boxes from January 1, 2024 through
August 30, 2026, with later prices reserved to finish outcomes. The baseline buys
the breakout of a six-bar 5-minute box inside a previously available 4-hour range.
R3 waits for the first retest and a 1-minute confirmation. Each method runs its
own book with one pending or open position at a time. A separate overlapping-event
diagnostic does not claim to be a funded portfolio.

This is a new minute-native research family, **not LC or a backtest of all 17
archetypes**. Its explicit parent/child/event relationships implement one part of
the seeing-eye design. They do not establish a complete Wyckoff phase or test the
unresolved Fibonacci, Gann and other teaching hypotheses.

## Primary result

The primary simulation uses $100 intended initial risk per entry, a $50,000
notional cap, a 12-basis-point round-trip execution allowance, five seconds of
processing rounded to the next minute opening, and the declared adverse funding
stress. These are simulated research dollars, not actual account losses.

| Measure | Immediate breakout | Retest and minute confirmation |
|---|---:|---:|
| Completed trades | 3,324 | 612 |
| Net simulated PnL | -$94,844.84 | -$30,104.34 |
| Average net per trade | -$28.53 | -$49.19 |
| Net winning trade rate | 34.7% | 27.9% |
| Profit factor | 0.51 | 0.27 |
| Exposure hours | 7,240.1 | 371.7 |
| Positive reporting blocks out of five | 0 | 0 |

R3 satisfied the minimum sample floors, with 612 completed trades across all 32
origin months. Its failure is therefore not merely a four-case sample problem.
Every cost/delay scenario lost money:

| Round-trip allowance and processing delay | R3 net PnL |
|---|---:|
| 12 bps and 5 seconds | -$30,104.34 |
| 12 bps and 65 seconds | -$29,319.81 |
| 24 bps and 5 seconds | -$39,825.29 |
| 24 bps and 65 seconds | -$39,053.85 |

Relative to the losing baseline, R3 improved net value by about $4.80 per common
raw opportunity. The declared paired-month interval is approximately +$3.77 to
+$5.92. This measures losing less, not an absolute edge. It compares the whole
entry package, including different stops, trading frequency and occupancy; it
does not isolate the predictive skill of the retest rule.

Removing funding alone still leaves R3 at -$29,044.62. Adding back both execution
charges and funding leaves -$2,438.29 of price PnL at the same primary quantities.
That is an accounting decomposition, not a newly tested zero-cost strategy.
Among boxes R3 did not trade, the baseline had 963 winners and 1,929 losers.
Those are whole-policy counterfactual outcomes, not agent judgment accuracy.

## Verification and limits

The source census completed in 68.28 seconds; economic scoring took 311.57 seconds.
All 18 saved books have the same raw opportunity IDs and zero unknown outcomes.
The controller independently verified saved hashes, risk/fee/funding/net-PnL
arithmetic, occupied capacity, calendar attribution and the 5,000 paired monthly
resamples. The 41,132 audited closed-position rows include repeated scenarios;
they are not 41,132 independent trades. The focused software suite passed 225
tests with one existing LibreSSL warning. The 29 protected engine/config files
remain unchanged. A fresh quant reviewer confirmed the park decision.

History was already exposed to prior research. These are fixed-rule chronological
development tests, not pristine holdouts or walk-forward optimization. Actual
funding, feed receipt, slippage and subminute fills remain unqualified. The
conservative interval does not correct all historical strategy searches.

## Next action

Freeze R3 and retain the reusable source, execution and audit machinery. Do not
add filters to rescue the revealed results. Resume R1 only after recovering both
`models/logistic_regime_v4_no_funding_stratified.pkl` and
`models/confidence_calibrator_v1.pkl` from an authoritative backup or deployment,
with provenance establishing their intended configuration/version.

Related project-folder searches, exact-path git history and an indexed home-file
search did not recover them; indexed search is not an exhaustive disk guarantee.
Git history contains older GMM models, which are not valid substitutes. Recovery
would be followed by a separately versioned source qualification and reviewed
R1 launch, not automatic scoring. No retraining, new hypothesis, live change,
paid market-role call, commit or push was performed. No job remains running.

## Evidence and reproduction

- [Frozen study contract](../superpowers/specs/2026-09-30-archetype-repair-discovery-design.md)
- [Final launch review](../../results/archetype_study_2026_10_01/review_v2.json)
- [Source audit](../../results/archetype_study_2026_10_01/census_audit_v1.json)
- [Economic comparison](../../results/archetype_study_2026_10_01/economics_v1/comparison.json)
- [Economic audit](../../results/archetype_study_2026_10_01/economics_audit_v1.json)
- [Read-only audit script](../../results/archetype_study_2026_10_01/audit_saved_economics.py)

Source receipt SHA256: `ce46283a7180b9f70d69e8c6ed871cdb1325e960f674fccc7137d92180e5d0e5`.
Economic receipt SHA256: `84e0b8b8fe2073a6a0f99ae875b628fb8690327f434b85d70728f1372d5bda76`.
Comparison SHA256: `888e34274d0c238e3187ecbf4d93bf90ba6cb25d4d324ee28c1086a86118a3e8`.
The minute archive, recovered helpers and approximately 1.53 GB of pilot/source/
economic artifacts remain local dependencies. New implementation and this report
are local/uncommitted; GitHub alone cannot reproduce the study.
