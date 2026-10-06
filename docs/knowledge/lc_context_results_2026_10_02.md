# Mechanical LC context: completed comparison, not a trading approval

October 2, 2026. One source run and one economic run completed, followed by
independent quant/software reviews and a separate price/accounting audit.

## Decision in plain language

The software now represents how a local LC signal relates to a previously known
larger range and a subsequent minute confirmation. That mechanical capability
works under the tested contract. The particular entry policy did **not** establish
an economic improvement. Expansion is **inconclusive**; rebound is **unsupported**.
Park this contextual policy. Do not enable it, tune its thresholds on these
results, or describe it as a successful all-seeing-eye strategy.

This was hypothetical trading, not money lost or earned in the live account.
All 17 production archetypes, live orders/configuration and fusion remain unchanged.

## What was actually tested

The fixed population contains all 146 saved reconstructed LC candidates from
January 2024 through August 2026: 70 upside-expansion candidates, 73 downside-
rebound candidates and three unresolved geometries. Every resolved candidate
remains in its comparison denominator, including nonentries. This is the complete
**saved reconstructed population**, not an exact reproduction of every live signal.
Monthly cold starts, missing native models, derivative defaults and macro/source-
availability limitations remain recorded in the source coverage.

Three entry methods were compared in separate expansion/rebound books:

1. Immediate entry after processing delay.
2. Wait for a completed minute to close above the setup's last-five-minute high,
   then apply processing delay again.
3. Context: enter accepted expansion above the frozen 4H range; require minute
   confirmation for contained range expansion or a sweep/reclaim of its floor;
   otherwise explicitly abstain or invalidate the setup.

The latest qualifying range version must exist before the setup hour. Acceptance
uses completed candles from its originating timeframe, not a lower-timeframe
wick or the old detector's hourly break flag. Daily context is recorded but does
not universally veto entry. Unknown optional macro/derivative/fusion snapshots do
not supply permission points. Exact predicates are project research choices,
not trader-certified formulas or a complete Wyckoff phase diagnosis.

This is an **hourly LC setup with minute execution**, not a new minute-native
archetype. The other minute-native R3 experiment remains separately parked.

All methods share a stop at setup close minus 2.7 hourly ATR, a target two price-
risk distances above the actual fill, and a 24-hour deadline from candidate time.
Entry expires strictly before T+15 minutes. Intended risk is $100 including modeled
roundtrip costs, capped at $50,000 entry notional. Fees use entry notional; fractional
quantity is allowed. Funding stress charges 8 bps per UTC eight-hour settlement
while held, including settlement at the modeled exit clock. This is a stress
assumption, not actual historical funding. A stop and target touched in the same
minute resolve stop-first; opening gaps and observation latency are explicit.

Two costs (12/24 bps), two processing delays (90/300 seconds), two funding modes
(adverse/zero) and the isolated books produced 48 comparisons. These are stress
variants, not 48 independent samples or a search for the best setting.

## Primary results

Primary assumptions: 12 bps roundtrip costs, 90-second processing, adverse funding.
Values are aggregate hypothetical net dollars, not portfolio percentage returns.

| Subtype | Immediate | Unconditional minute wait | Context |
|---|---:|---:|---:|
| Expansion | 70 fills; +$585.97 | 47 fills; +$425.10 | 34 fills; +$139.32 |
| Rebound | 72 fills; −$2,249.05 | 37 fills; −$1,030.70 | 2 fills; −$204.60 |

Expansion context underperformed immediate by $446.65. Per supplied candidate,
the difference was −$6.38; its descriptive paired 95% interval was −$26.69 to
+$12.34. Context preserved 15 baseline winners, missed 15, avoided 21 losers and
retained 19. It introduced no losses from baseline nonlosses. Mark-to-market
drawdown fell from $781.24 to $695.45, but profit and participation also fell.
Removing context's three largest winners leaves −$409.54.

Rebound context improved the losing baseline by $2,044.45, or $28.01 per supplied
candidate, with a descriptive interval of +$4.67 to +$48.40. That is **smaller
losses, not positive expectancy**: it missed all 20 baseline winners, avoided 50
losers and retained two losers. Both contextual entries lost. The result cannot
support a claim that the controller successfully selected profitable rebounds.

Intervals use 5,000 paired resamples of the same 32 calendar months, seed20261002,
with every raw subtype candidate counted. Zero undefined draws occurred. The
history was already exposed and informed this design; intervals are descriptive,
not selection-adjusted significance or an untouched holdout. No model was fitted,
so this was not walk-forward training or CPCV validation.

## Execution stress

All values below include adverse funding; columns show cost / processing delay.

| Subtype / method | 12 bps / 90s | 12 bps / 300s | 24 bps / 90s | 24 bps / 300s |
|---|---:|---:|---:|---:|
| Expansion immediate | +$585.97 | +$502.62 | +$48.33 | −$36.53 |
| Expansion wait | +$425.10 | −$337.56 | +$89.64 | −$528.70 |
| Expansion context | +$139.32 | +$80.30 | −$86.54 | −$113.64 |
| Rebound immediate | −$2,249.05 | −$2,305.63 | −$2,539.28 | −$2,594.97 |
| Rebound wait | −$1,030.70 | −$521.40 | −$1,190.17 | −$645.62 |
| Rebound context | −$204.60 | −$101.69 | −$204.43 | −$101.65 |

Expansion context was positive in all four zero-funding diagnostics but failed the
declared higher-cost/adverse-funding checks. Rebound context lost even with zero
funding. The simpler expansion baseline is therefore a research reference, not
a certified alternative: it also turns slightly negative in the toughest stress.

The frozen forward-screen required positive primary net and incremental result,
a positive lower interval, at least50fills/12filled months, and no negative adverse-
funding stress. Source inspection established before economics that at most42
expansion and22rebound candidates could be admitted, so neither could meet the
50-fill floor. No sample extension or threshold relaxation followed that finding.

## Where the proposed reasoning lost opportunities

These are descriptions of frozen scenario groups, not newly selected strategies.

- Of23expansion cases with unconfirmed 4H acceptance, context entered none. The
  immediate comparison on those same cases contained11winners/12losers, net+$237.52.
  A 15-minute window cannot wait for the next new 4H close after an hourly decision.
  That is a genuine limitation of this policy's timing, not a software failure.
- The21contained-expansion candidates produced13context fills and+$54.70 versus
  +$474.19 from immediate entry on the same21. Seven expired; one changed location.
- All21already-accepted expansions used the same immediate entry in both methods,
  so both earned+$84.62. Context adds no incremental judgment in that branch.
- Of22floor-rebound candidates,20expired and two entered/lost. Separately, the
  32rebounds rejected for accepted-below-floor context contained26baseline losers
  and six baseline winners, net−$1,597.25. This is useful defensive description,
  but not evidence for a new profitable entry strategy.

No saved primary book had a busy skip, so occupied capacity did not explain these
differences. More context and more gates did not automatically mean better selection.

## Verification and limits

Source qualification reconstructed monthly hourly OHLCV/ATR, continuous parent
input/ATR and all6024four-hour/1004daily anchors. All146legacy projections and
independent parent cuts passed. Root and quant review independently reconciled raw
IDs/subtypes and current bindings:71files, four source artifacts, both parent
selections and originating-timeframe acceptance events.

The separate [audit script](../../results/lc_context_study_2026_10_02/audit_saved_books.py)
imports no strategy, replay or report functions. Its competing-event calculation
checked all48books,3,432candidate rows,1,984position rows and3,682,300MTM marks
against the archive; entry/expiry/pending-stop clocks, first exits, quantity,
fees/funding, risk, occupancy, calendar sums, drawdown and all32paired comparisons
reconciled. Position counts repeat trades across stress books; they are not1,984
independent setups. Software review independently verified54source/economic
artifact hashes, exact book combinations and row sets. No unknown outcomes remain.

Fresh focused regression: **159passed in4.18s**. A repository-wide attempt stopped
at three unchanged collection errors: missing
`engine.strategies.archetypes.bull.wick_trap_moneytaur` in
`tests/archetypes/test_bull_archetypes_mvp.py`, plus missing `FusionEngine` exports
for archived `test_macro_pulse.py` and `test_v17_integration.py`. The full repository
suite is not green; unrelated archived imports were not repaired. Protected old
research hashes match the pre-edit values; no tracked engine/config/live diff.

Known limitations/deferred review items:

- No saved busy skips, opening-gap exits or missing execution paths occurred;
  those behaviors are verified by synthetic tests, not this historical sample.
- Post-fill structural changes are not logged or used for adaptive management.
  The integrated review classified this as a minor diagnostic omission and it was
  disclosed before economic launch. Fixed exits/PnL are unaffected.
- The audit-v1 scope phrase suggesting trader fidelity was separately qualified is
  overbroad: **data/source construction was qualified, not exact trader-rule fidelity**.
  This report corrects that scope; preserve the original hash-bound audit record.
- No complete Wyckoff thesis, hidden-Fibonacci/Gann timing, qualified funding/OI/
  order-flow permission, learned fusion, agent judgment or cross-archetype selection
  was tested. Historical bar-close availability, no market impact and no exchange
  lot-size rounding remain assumptions, not authentic execution receipts.

## Completed deliverables and next action

New evidence, controller, execution, source and reporting modules plus their CLI
and tests are implemented. One frozen source run took154.35s (about2.2MB before
receipt); one frozen economic run took117.54s (about42.2MB); independent audit
took6.23s. Peak process RSS was about1.43GB/1.19GB respectively. Both600-second/
512MiB-artifact ceilings were respected. No research process remains running.

The quant recommendation is to **park this context policy**, preserve immediate
upside LC as an unproven reference, and not launch more agents, retune this exposed
sample or deploy anything. The next justified validation requires a separately
specified fresh, source-qualified population and realistic execution evidence;
it must not be created by relabeling these146cases as holdout. No additional
experiment is needed to finish this handoff. R3/R2 remain parked; unrelated native
R1 model recovery remains blocked as already documented.

All work is local/uncommitted on `quant/archetype-evidence-audit`. No paid market
assessment calls, installs, downloads, commits, push or PR occurred. Review agents
used this session's model usage; this is not a claim of zero overall usage.
Approximately44.5MB of new ignored artifacts, the minute archive, saved source
receipts and external recovered parent helpers are local-only dependencies.

Artifacts: `results/lc_context_study_2026_10_02/` contains `source_v1`,
`economics_v1`, `review_v1.json`, `audit_saved_books.py` and `independent_audit_v1.json`.
Do not rerun a launch into these directories or overwrite the historical records.

| Artifact | SHA256 |
|---|---|
| Source receipt | `8eb2d58a680f444713472af760219a23ee7cf2c7be0b1b134d1acc2b883f9b2b` |
| Economic receipt | `7f316f7bd64a3851389ecfd237782a62787460116d13878f232f46db9b784cba` |
| Comparison | `35e24bdad56edd111ea4b0de8da0e3dd9e36c4f22cbd7e3c356e82cb7b1d18ad` |
| Independent audit receipt | `6869d56021de14985526100491a12c0bc7829e85439584da844b597371e9b673` |

See the frozen [design](../superpowers/specs/2026-10-02-mechanical-lc-context-design.md),
[plan](../superpowers/plans/2026-10-02-mechanical-lc-context.md) and current
[project handoff](../../PROJECT.md). Their preimplementation wording is historical;
this report and the newest handoff give actual completion status.
