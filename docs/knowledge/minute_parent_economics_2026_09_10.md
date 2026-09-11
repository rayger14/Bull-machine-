# Minute parent-context economics: frozen June comparison

## Status and question

The registered comparison is complete and approved for LOCAL RESEARCH after independent implementation, integration and economic-interpretation review. Final scoped verification closed the OHLC input-boundary finding; all frozen economic results remained exactly unchanged. This asks whether the four existing parent permissions change economic outcomes within a frozen, preselected minute-event population. It is not a new archetype graduation, historically authenticated live replay, or funded account backtest.

The [design](../superpowers/specs/2026-09-10-minute-parent-economics-design.md) was frozen before this outcome run and approved by an independent quant. All four parent variants are retained; none is selected as a winner. Hourly source-stage work is separately documented in [hourly cutoff readiness](hourly_cutoff_experiment_readiness_2026_09_10.md).

## Exact experiment

- Candidate window: June 10 00:00 through June 20 00:00, 2026 UTC, end exclusive; 76 previously frozen equal-low sweep events, retaining the original 60-minute selector spacing.
- Same-source input slice: June 1 00:00 through June 20 04:00 inclusive, 27,601 contiguous minutes. Original child history is preserved via the frozen event list; parent construction keeps its declared June 1 cold start and same-stream TA-Lib ATR14 reference. The full tail is present.
- Each sleeve starts flat with no lockout at June 10. Baseline and four permission arms replay independently. Permission rejection/unknown does not arm lockout; removing an earlier entry can expose a later candidate.
- Long entry at the next exact minute open after reclaim close, zero additional delay. Stop 0.15% below the frozen sweep low, four-hour deadline-open exit and four-hour lockout even after an early stop. Entry-bar stops included; adverse stop gaps fill at the lesser of stop and bar open. No target, scale-out, parent exit or trailing rule.
- $50,000 fixed notional; $30 assumed cost each side ($60 round trip). **Starting equity is unspecified.** No compounding, capital/margin constraint, funding, impact, independently calibrated spread/slippage or measured host latency. These are unfunded reference sleeves; no percentage account return is available.

## Results: all arms

N3/N5 are the existing pivot-confirmation hypotheses, not optimized settings. Dollar results below are after the specified $60 round-trip cost, not fully net exchange returns. Average risk is the modeled initial stop-distance dollar risk of completed positions.

| Arm | Permitted events | Completed positions | Gross PnL | Assumed costs | Net diagnostic PnL | Net PF | Avg initial stop risk |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 76 | 36 | −$1,458.95 | $2,160 | −$3,618.95 | 0.3434 | $139.62 |
| 4H / N3 | 16 | 9 | −$135.58 | $540 | −$675.58 | 0.5176 | $135.36 |
| 4H / N5 | 19 | 11 | −$565.41 | $660 | −$1,225.41 | 0.3736 | $164.14 |
| 1D / N3 | 8 | 4 | +$628.37 | $240 | +$388.37 | 2.1543 | $113.56 |
| 1D / N5 | 0 | 0 | $0 | $0 | $0 | unavailable | unavailable |

There are no open/censored, unfilled or invalid-stop events in these runs. Busy skips are 40/7/8/4/0 respectively. Permission rejections are 0/60/57/68/76. All 76 IDs occur exactly once in each joined status ledger; permitted count is not trade count.

Mean net-PnL / modeled initial stop-risk ratios are −0.6920, −0.3876, −0.8517, +0.9525 and unavailable, respectively. These are means of per-position ratios, not aggregate PnL divided by aggregate risk. Net win rates are 22.22%, 22.22%, 18.18%, 50% and unavailable.

Minute-close marked-dollar drawdowns are −$4,726.81, −$1,783.45, −$1,763.91, −$576.93 and $0. These are unfunded diagnostic marked-PnL series; smaller magnitude can simply reflect fewer trades and less exposure. They omit intraminute excursions and cannot be quoted as capital-constrained drawdown percentages. Full entry/exit records are retained rather than inventing exact exposure minutes from ambiguous stop-bar labels.

## What the apparent positive result actually contains

All four 1D/N3 entries occur on **June 19**, bound to one parent version available at June 19 01:00 UTC. Its 68 rejected events all have `absent_parent`, not an observed unfavorable geometry. Its eight permitted events have no available-parent geometry rejects in this sample. Thus this arm largely selects a calendar/parent-availability subset; the run does not isolate the predictive effect of geometry from availability.

The four net outcomes are −$177.88, −$158.57, +$590.23 and +$134.59. One win exceeds the entire +$388.37 subtotal. Four trades on one day are neither independent multi-period confirmation nor evidence of a dependable strategy. Daily N5's zero result is complete inactivity/absent-parent coverage, not demonstrated protection.

Both 4H arms remain negative even before costs. Their smaller total losses do not alone prove useful filtering: they take fewer trades, and N5 has a worse mean modeled risk ratio than baseline. Fees worsen the observed results, but they are not the only source of baseline/4H losses in this window.

## Opportunity displacement is real

4H/N3 shares all nine entered IDs with baseline. 4H/N5 shares nine, removes 27 baseline entries and adds **two new entries** that baseline had skipped as busy. Those two new positions lose $207.81 and $192.22 under the fixed assumptions. A completed-trade filter would miss these extra entries and their combined $400.03 loss.

Daily N3 shares its four entries with baseline; daily N5 enters none. These are conditional changes within the already spaced 76-event population, not recovery of opportunities the upstream selector excluded. No native all-archetype allocator or live book is replayed.

## Verification and artifacts

Implementation `9637ec0`, reviewed boundary fixes `6197d88`, `8415876` and final OHLC representation fix `cd45a6d`. TDD observed missing-module and stub-assertion failures, then price/window and OHLC boundary regressions before fixes. Final focused suite: 38 passed. Root fresh full research suite: **359 passed**, one existing urllib3/LibreSSL warning, 12.00 seconds. Final scoped review approved the fix with no new breakage and independently checked native integer/unsigned/float preservation and unsupported representations. Production source/config and the existing minute simulator are unchanged from phase base `73b5a72`.

Private prepared input: `results/research_validation_2026_09_10/minute_parent_economics/prepared_input.json`, SHA256 `1a1ff35ef857f2a75bf16f62afef6e6efcb1018e31b99f5e22d07716b69aa365`. It identifies/hash-binds the full original minute file, baseline artifact, frozen child reference and annotations, plus the rebased slice. All 76 IDs, source prices/sweep lows/reclaim closes, confirmation/sweep/decision clocks and 304 annotation joins were checked. Permission counts remain exactly 16/19/8/0. Parent availability strictly precedes sweep open on every passing annotation.

Private refreshed output at implementation `cd45a6d`: `results/research_validation_2026_09_10/minute_parent_economics/june_comparison.json`, SHA256 `1c21470eacbf2b00a5098acedd7d58b469eba144b6200f879eda9d6baff8388b`. Comparison source SHA256 `93e5f6f2912482fb20613739f15cc80b4b3029bcd1d014b78b928d886571b73b`; unchanged simulator SHA256 `95d586516720504e4f8a34f702dc3dc710e29961d3214724040b89283fac4c86`; parent policy SHA256 `b0ebfe75e0222aa65460a1f21b747f4ab238a5f3c2708ff454a12f56755d97ee`. The refreshed whole comparison equals the prior frozen comparison exactly; only implementation/provenance/check metadata changed.

All five full simulation payloads equal independent direct calls to the existing simulator on the same permitted candidate lists and same bounded flat state. Whole-report fresh-copy equality, input nonmutation, source hash stability, complete ledger/transition accounting and strict JSON pass. Gross/fee/net totals and mean/median risk ratios were independently recomputed from returned fill records. These checks establish comparator integration and arithmetic, not empirical fill accuracy or edge.

## Interpretation and next evidence

No parent variant is promoted. The next comparison should separate **parent availability** from **geometry conditional on an available parent**, with an explicitly frozen prehistory policy, before attributing the daily result to structural intelligence. More reused periods can test stability retrospectively but cannot become a pristine holdout; genuinely new forward evidence remains necessary after a candidate freeze. Do not optimize parent N, stop, costs or exit based on this ten-day result.

Hourly needs its own immutable research variant identity, exact pre-mutation outer-cutoff trace and separately executable fill contract. A raw bypass toggle mixes several policy changes and cannot answer whether fusion alone helps. This phase implements no such active runner/config change.
