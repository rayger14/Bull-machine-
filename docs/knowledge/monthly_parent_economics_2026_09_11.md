# Four-month minute parent economics

September 11, 2026. Completed local research; independently reviewed with no blocking findings. **No validated edge, variant promotion or production change.**

## Frozen scope

Protocol committed as `e465e0c` before outcomes: [specification](../superpowers/specs/2026-09-11-monthly-parent-economics-design.md). Four separate UTC candidate months: October 1–November 1, 2025; January 1–February 1, April 1–May 1 and July 1–August 1, 2026 (end exclusive). Parent history uses independent 30-day cold starts. These previously researched months are not pristine holdouts.

All 941 frozen minute equal-low sweep/reclaim candidates retained. Four parent variants: 4-hour or daily anchors, N3 or N5 pivot settings. A = parent available strictly before first sweep; L = A plus no bound-lineage break through decision; G = L plus child level in the parent's inclusive lower half and reclaim strictly inside its frozen range. Four months × (one baseline + four variants × three controls) = 52 reports, not 52 independent strategies.

Starting equity **unspecified**. Each month/arm starts flat, with $50,000 fixed notional per position, next exact minute-open entry, no extra decision delay, stop 0.15% below frozen sweep low, entry-bar stops included, adverse gap fills, 240-minute deadline-open exit and fixed entry-based lockout. Modeled round-trip costs $60/12 bps. No target, scale-out, trailing, parent exit, compounding, funding, margin, liquidation or empirical spread/impact/latency calibration. These are diagnostic dollar results, not funded-account returns. Average initial stop risk appears below; it excludes modeled costs and is not a guaranteed maximum loss.

## Findings

- 51 of 52 reports are net negative; 15 are gross positive.
- The only net-positive report is April G/daily N5: +$101.49 across 30 positions under one parent lineage. Additional fixed round-trip cost of $3.382981 (0.676596 bps on $50,000) brings it to zero; more makes it negative. This is arithmetic sensitivity, not an empirical cost estimate or stress rerun.
- Every G variant has negative net PnL when its four independently flat monthly results are descriptively summed. These sums are not continuous portfolio returns. G/4H N3 is +$500.22 gross and −$12,879.78 net; the other G sums are gross negative. The baseline is −$3,681.17 gross and −$31,161.17 net.
- Parent filtering sometimes reduces losses but does not establish a dependable edge. Results do not justify selecting N, treating sparse parent coverage as safety, or adding an agent as a presumed profitability fix.
- Permissions were applied before replay. Removing a candidate can expose another entry or leave a permitted candidate busy; completed-trade filtering is not an equivalent experiment. All 48 adjacent-arm contrasts reconcile shared, removed and exposed entries.

## All 52 outcomes

Dollar columns rounded only for display. A, L and G labels are permission controls, not management changes. Repeated positions across arms are not independent observations.

| Month | Arm | Positions | Gross $ | Costs $ | Net $ | Avg initial risk $ |
|---|---|---:|---:|---:|---:|---:|
| 2025-10 | baseline | 113 | -2449.85 | 6780.00 | -9229.85 | 148.92 |
| 2025-10 | A / 4H:3 | 83 | -2913.17 | 4980.00 | -7893.17 | 147.88 |
| 2025-10 | A / 4H:5 | 87 | -1098.16 | 5220.00 | -6318.16 | 154.01 |
| 2025-10 | A / 1D:3 | 84 | -1925.91 | 5040.00 | -6965.91 | 161.61 |
| 2025-10 | A / 1D:5 | 78 | -2144.91 | 4680.00 | -6824.91 | 163.16 |
| 2025-10 | L / 4H:3 | 83 | -2913.17 | 4980.00 | -7893.17 | 147.88 |
| 2025-10 | L / 4H:5 | 87 | -1098.16 | 5220.00 | -6318.16 | 154.01 |
| 2025-10 | L / 1D:3 | 84 | -1925.91 | 5040.00 | -6965.91 | 161.61 |
| 2025-10 | L / 1D:5 | 78 | -2144.91 | 4680.00 | -6824.91 | 163.16 |
| 2025-10 | G / 4H:3 | 56 | 1516.57 | 3360.00 | -1843.43 | 136.23 |
| 2025-10 | G / 4H:5 | 53 | -885.47 | 3180.00 | -4065.47 | 168.96 |
| 2025-10 | G / 1D:3 | 78 | -2071.41 | 4680.00 | -6751.41 | 162.22 |
| 2025-10 | G / 1D:5 | 74 | -1590.57 | 4440.00 | -6030.57 | 164.49 |
| 2026-01 | baseline | 114 | -2194.08 | 6840.00 | -9034.08 | 127.03 |
| 2026-01 | A / 4H:3 | 89 | -2428.91 | 5340.00 | -7768.91 | 130.25 |
| 2026-01 | A / 4H:5 | 74 | -3343.52 | 4440.00 | -7783.52 | 124.42 |
| 2026-01 | A / 1D:3 | 88 | -2478.50 | 5280.00 | -7758.50 | 127.64 |
| 2026-01 | A / 1D:5 | 73 | -692.20 | 4380.00 | -5072.20 | 124.27 |
| 2026-01 | L / 4H:3 | 89 | -2428.91 | 5340.00 | -7768.91 | 130.25 |
| 2026-01 | L / 4H:5 | 74 | -3343.52 | 4440.00 | -7783.52 | 124.42 |
| 2026-01 | L / 1D:3 | 88 | -2497.47 | 5280.00 | -7777.47 | 127.86 |
| 2026-01 | L / 1D:5 | 73 | -711.17 | 4380.00 | -5091.17 | 124.52 |
| 2026-01 | G / 4H:3 | 70 | -2132.97 | 4200.00 | -6332.97 | 129.48 |
| 2026-01 | G / 4H:5 | 54 | -1740.72 | 3240.00 | -4980.72 | 124.71 |
| 2026-01 | G / 1D:3 | 57 | -2420.96 | 3420.00 | -5840.96 | 134.07 |
| 2026-01 | G / 1D:5 | 22 | -662.23 | 1320.00 | -1982.23 | 129.16 |
| 2026-04 | baseline | 115 | 2730.66 | 6900.00 | -4169.34 | 130.16 |
| 2026-04 | A / 4H:3 | 85 | 3071.07 | 5100.00 | -2028.93 | 129.64 |
| 2026-04 | A / 4H:5 | 85 | 3868.79 | 5100.00 | -1231.21 | 129.36 |
| 2026-04 | A / 1D:3 | 67 | 3342.98 | 4020.00 | -677.02 | 127.37 |
| 2026-04 | A / 1D:5 | 63 | 2091.72 | 3780.00 | -1688.28 | 131.66 |
| 2026-04 | L / 4H:3 | 85 | 3071.07 | 5100.00 | -2028.93 | 129.64 |
| 2026-04 | L / 4H:5 | 85 | 3868.79 | 5100.00 | -1231.21 | 129.36 |
| 2026-04 | L / 1D:3 | 67 | 3342.98 | 4020.00 | -677.02 | 127.37 |
| 2026-04 | L / 1D:5 | 63 | 2091.72 | 3780.00 | -1688.28 | 131.66 |
| 2026-04 | G / 4H:3 | 59 | 3076.96 | 3540.00 | -463.04 | 120.30 |
| 2026-04 | G / 4H:5 | 68 | 2051.56 | 4080.00 | -2028.44 | 124.45 |
| 2026-04 | G / 1D:3 | 35 | 1974.73 | 2100.00 | -125.27 | 128.79 |
| 2026-04 | G / 1D:5 | 30 | 1901.49 | 1800.00 | 101.49 | 126.26 |
| 2026-07 | baseline | 116 | -1767.90 | 6960.00 | -8727.90 | 123.52 |
| 2026-07 | A / 4H:3 | 84 | -1118.89 | 5040.00 | -6158.89 | 123.79 |
| 2026-07 | A / 4H:5 | 87 | -2929.31 | 5220.00 | -8149.31 | 125.87 |
| 2026-07 | A / 1D:3 | 110 | -782.01 | 6600.00 | -7382.01 | 122.35 |
| 2026-07 | A / 1D:5 | 115 | -1628.13 | 6900.00 | -8528.13 | 123.38 |
| 2026-07 | L / 4H:3 | 84 | -1118.89 | 5040.00 | -6158.89 | 123.79 |
| 2026-07 | L / 4H:5 | 87 | -2929.31 | 5220.00 | -8149.31 | 125.87 |
| 2026-07 | L / 1D:3 | 110 | -782.01 | 6600.00 | -7382.01 | 122.35 |
| 2026-07 | L / 1D:5 | 115 | -1628.13 | 6900.00 | -8528.13 | 123.38 |
| 2026-07 | G / 4H:3 | 38 | -1960.34 | 2280.00 | -4240.34 | 130.06 |
| 2026-07 | G / 4H:5 | 41 | -1273.42 | 2460.00 | -3733.42 | 127.43 |
| 2026-07 | G / 1D:3 | 17 | -962.69 | 1020.00 | -1982.69 | 130.75 |
| 2026-07 | G / 1D:5 | 22 | 238.27 | 1320.00 | -1081.73 | 113.89 |

## Reproduction and verification

Private artifacts under `results/research_validation_2026_09_11/monthly_parent_economics/`: frozen `probe.py`, `frozen_maps.json`, `manifest.json`, and full `monthly_comparison.json`. Source/code/specification/probe hashes are bound by the manifest. Full JSON preserves trade records, candidate statuses, PF, win fraction, average risk, mean/median net PnL over initial risk, dollar marked drawdown, concentration and pairwise displacement. Drawdown is minute-close/realized-mark based, not account-percent or worst intraminute drawdown.

- Frozen maps SHA256: `25a4d2142387fc9033989c845b1738db550410bcd61ee571a0d38f20f9ee5d8d`.
- Manifest SHA256: `42da22335c35787487ef29723a21b835ffdaba049c5b1aabe31e4c9789e83639`.
- Output SHA256: `a2efec93e7e22219fe50e9bf51abb8e3446e4ac3ce8b7c0cc41de9e4565da80a`.

Root checks passed full direct simulator equality, copied-repeat equality, source/index/mask identity, complete candidate ledgers, shared fill identity, displacement arithmetic, gross/cost/net/risk arithmetic, nonmutation, full tails and strict JSON. No open, unfilled or invalid-stop outcomes in these runs.

Independent reviewer reran the probe and reconstructed all 52 reports, 3,936 position records (not unique trades), 1,438 day/parent groups and 48 pairwise contrasts from raw source. Approved local research with no blockers. Fresh full research suite: **359 passed in 12.18 seconds**, one existing urllib3/LibreSSL warning. Test success validates tested machinery, not economic edge.

## Next steps

Diagnose these existing entered-trade records by stop/time exit, entry-bar stops, adverse gaps, gross expectancy versus modeled costs and parent concentration. This is distinct from a raw-candidate predictive event study, which needs its own frozen specification. Keep hourly stage-isolation/execution work separate and equally important. See [agentification advisory](agentification_design_2026_09_11.md) for the proposed evidence-first pilot; no agents participated in this experiment and none are enabled live.
