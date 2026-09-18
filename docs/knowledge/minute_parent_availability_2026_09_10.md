# Minute parent availability versus geometry

## Scope and review status

Research probe complete and independently approved for **LOCAL RESEARCH**, with no blocking findings. It uses the unchanged reviewed minute comparator and simulator. Independent quant design approval preceded the new availability-only economic outcomes; the reviewer subsequently reran the probe and reproduced the exact artifact hash. No reusable feature, production code, configuration, live order, new dependency, optimizer or fitted threshold was added. The probe script is a throwaway diagnostic retained privately for audit, not a supported engine interface.

This follows [the first frozen parent comparison](minute_parent_economics_2026_09_10.md). The purpose is to distinguish parent availability from geometry, not to choose a winning pivot N or claim a profitable archetype.

## Frozen protocol

- Candidate window: **June 10 00:00–June 20 00:00, 2026 UTC**, end exclusive; unchanged 76 previously selected minute equal-low sweep candidates and original 60-minute selector spacing.
- Parent prehistory stays at the declared **June 1 cold start**. The same 27,601 contiguous OHLC minutes cover June 1 through June 20 04:00 inclusive. Frozen child history/identities are unchanged. Every sleeve starts flat with no inherited lockout on June 10.
- Long next-minute-open entry, zero extra decision delay, $50,000 fixed notional, stop 0.15% below sweep low, 240-minute deadline-open exit and 240-minute entry lockout even after an early stop. Entry-bar stops and adverse stop gaps retain the previous simulator's rules.
- **Starting equity unspecified**: unfunded reference sleeves, not $100,000 accounts. $30 assumed cost each side, $60/12 bps round trip. Commission/spread/slippage are not independently calibrated; no funding, impact, margin, liquidation, compounding or measured host latency. No account-return percentage is defined.

All four hypotheses (4H/N3, 4H/N5, 1D/N3, 1D/N5) remain fixed. For each, run independent candidate-filtered sleeves, not filters of completed trades:

1. **A — availability:** validated `binding.status == "bound"`, with parent availability strictly before first sweep open.
2. **L — availability plus lifecycle:** A and strict `bound_lineage_broken is False`. Supersession alone does not veto or replace the original frozen geometry.
3. **G — full policy:** unchanged L plus `parent_low <= child_level <= parent_midpoint` and `parent_low < reclaim_close < parent_high`.

Baseline→A measures the sample's availability restriction; A→L the lifecycle restriction; L→G the geometry restriction. These are conditional replay contrasts, not identified causal predictive effects. Unknown/malformed/contradictory evidence is nonpermitted; this frozen probe aborts on any unexpected unknown instead of inferring availability from a rejection-reason string. Original annotations and IDs are unchanged; control maps are stored separately.

## Coverage before economics

| Variant | A candidates | L candidates | G candidates | Absent parent | Available but geometry-rejected |
|---|---:|---:|---:|---:|---:|
| 4H/N3 | 52 | 52 | 16 | 24 | 36 |
| 4H/N5 | 54 | 54 | 19 | 22 | 35 |
| 1D/N3 | 8 | 8 | 8 | 68 | 0 |
| 1D/N5 | 0 | 0 | 0 | 76 | 0 |

All 304 annotations reproduce exactly from frozen events/ledgers. No unknowns occur; all lifecycle-break and supersession flags are strict false. Therefore A=L exactly: this sample does not measure lifecycle-veto effectiveness. Both daily variants have A=L=G exactly, so daily geometry has no incremental selection here.

## Economic results

A and L have identical full reports. Costs below are only the fixed modeled costs; average initial stop risk is per completed position.

| Variant/control | Positions | Gross PnL | Costs | Net diagnostic PnL | Net PF | Avg initial stop risk |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 36 | −$1,458.95 | $2,160 | −$3,618.95 | 0.3434 | $139.62 |
| 4H/N3 A=L | 25 | −$1,417.45 | $1,500 | −$2,917.45 | 0.2830 | $143.57 |
| 4H/N3 G | 9 | −$135.58 | $540 | −$675.58 | 0.5176 | $135.36 |
| 4H/N5 A=L | 26 | −$1,432.72 | $1,560 | −$2,992.72 | 0.2789 | $144.69 |
| 4H/N5 G | 11 | −$565.41 | $660 | −$1,225.41 | 0.3736 | $164.14 |
| 1D/N3 A=L=G | 4 | +$628.37 | $240 | +$388.37 | 2.1543 | $113.56 |
| 1D/N5 A=L=G | 0 | $0 | $0 | $0 | unavailable | unavailable |

Four-hour geometry reduces losses in this sample but leaves both variants negative **before and after costs**. N3's $2,241.87 net improvement comprises $1,281.87 gross improvement and $960 fewer modeled costs. N5's $1,767.31 comprises $867.31 gross improvement and $900 fewer costs. This is not merely a fee effect, but reduced activity and changed selected populations still matter.

Mean net PnL per trade for N3 changes from −$116.70 to −$75.06; mean per-trade net-PnL/initial-risk ratio changes from −0.6719 to −0.3876, but its median worsens from −1.3870 to −1.4368. For N5, mean net changes only from −$115.10 to −$111.40, while mean risk ratio is slightly worse: −0.8487 to −0.8517. A smaller dollar loss must not be called improved risk-adjusted edge or general distributional improvement.

All four daily-N3 positions remain on June 19 under one parent version available that day at 01:00 UTC. Their +$388.37 result is exactly an availability/calendar subset in this experiment, not evidence that daily geometry helped. Daily N5 remains inactive, not demonstrably safe.

## Opportunity displacement accounting

| Contrast | Shared entered IDs | Removed entries | Newly exposed entries | Net change |
|---|---:|---:|---:|---:|
| 4H/N3 baseline→A | 24 | 12 | 1 | +$701.50 |
| 4H/N5 baseline→A | 24 | 12 | 2 | +$626.23 |
| 4H/N3 L→G | 9 | 16 | 0 | +$2,241.87 |
| 4H/N5 L→G | 9 | 17 | 2 | +$1,767.31 |

Shared entries have exactly identical fill and management records. N3 geometry removes 16 previously entered positions totaling −$2,241.87. N5 removes 17 totaling −$2,167.35 and exposes two formerly busy positions totaling −$400.03. Of N5's 17 removed entries, **16 are permission-denied and one remains permitted but becomes busy** after the schedule changes. Removing an entered ID is not synonymous with a direct gate rejection.

Full per-event status transitions and day/parent-version candidate, permission, completed-position and net-PnL groups are retained in the artifact. N3 G entries span June 10, 11, 17 and 19; N5 G entries span June 10, 17, 18 and 19. These clustered observations are not independent validation periods. No p-values, confidence intervals or post-hoc winner selection are claimed.

## Verification and private artifacts

- `results/research_validation_2026_09_10/minute_parent_availability/probe.py`: disposable reproduction/audit script, run from root with `PYTHONPATH=. python3 <path>`; final script hash embedded in output. It refuses to overwrite a different existing result.
- `results/research_validation_2026_09_10/minute_parent_availability/june_availability_comparison.json`, SHA256 **`7adadb86deca1cecb51ab59ee3926654bfa81e8fc0c5cf2cebab45a1bfc8bcaa`**.
- Existing prepared input SHA256 `1a1ff35ef857f2a75bf16f62afef6e6efcb1018e31b99f5e22d07716b69aa365`; prior comparison SHA256 `1c21470eacbf2b00a5098acedd7d58b469eba144b6200f879eda9d6baff8388b`. All original source hashes and comparator/policy/simulator hashes verified unchanged.
- Exact bounded source OHLC/index/dtypes verified, 76 unique events/variant, source values and clocks checked, complete 304 fresh annotation equality, strict parent availability and G⊆L⊆A nesting, complete ledgers, full tail, independent direct simulator equality for every sleeve, copied-input repeat equality/nonmutation, gross/cost/net and mean-risk arithmetic, shared-fill equality and displacement decomposition pass. G's entire comparison exactly equals the prior frozen result.
- Fresh full research suite: **359 passed**, one existing urllib3/LibreSSL warning, 12.25 seconds. These tests verify software properties, not empirical fills or profitability.
- Independent quant result review reran the full probe with identical artifact hash, and separately recomputed annotation-derived masks, entry/exit/sweep-price PnL and initial risk, fees, mean/median risk ratios, PF, day/parent group sums, pairwise sets/shared fills/deltas and all 76-ID transition totals. No blocking findings. Root rechecked the cited median values and final source/probe hashes.
- Pre-outcome diagnostic correction: an initial audit compared the four-column bounded OHLC frame to the five-column source including volume. Direct inspection proved exact OHLC equality; the audit now explicitly projects OHLC. Exact simulator status/summary field names were also corrected in the throwaway auditor. No source data or simulator behavior was altered.

## Decisions and next evidence

The delegated quant required the lifecycle control to prevent mislabeling a combined geometry/lifecycle effect. Retaining this extra identity costs some redundant computation here, but preserves the scientific question. We retained the fixed cold start/window instead of changing prehistory after seeing results; the cost is boundary-sensitive, sparse daily coverage and no generalization claim.

No variant is promoted or tuned. The next bounded quant recommendation is a **coverage and parent-diversity audit before new economics**: record complete source/tail coverage, cold-start convention, frozen candidate counts, bound-parent days/lineages and A/L/G coverage across additional already available periods for all four configurations. Then freeze a common retrospective schedule and unchanged execution contract, without selecting periods by PnL. Reused data remains retrospective; extending history must be a separately frozen experiment, not silently modifying this result. Hourly remains equally important under its separate [cutoff-stage and execution-readiness contract](hourly_cutoff_experiment_readiness_2026_09_10.md); this minute experiment does not validate or replace hourly fusion.
