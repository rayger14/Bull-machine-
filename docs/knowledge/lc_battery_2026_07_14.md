# liquidity_compression Battery — A Real But Thin Second Edge

**Date**: 2026-07-14. **Trigger**: LC is the live account's biggest all-time earner (+$2,919, 26
positions, 50% WR; +$4,876 last 30d) yet had NEVER been run through the standalone battery.
**Setup**: v14rq configs, thresholds enforced, V14L store. **Artifact**: `results/lc_battery.log`

## Results
| window | PF | PnL | WR |
|---|---|---|---|
| wfo_train 2018-22 | 1.14 | +$11,714 | 75% |
| **holdout 2025-26** | **1.14** | **+$2,982** | 77% |
| full 2018-24 | 1.21 | +$28,617 | 75% |
| bears (2018 / 2022) | 0.51 / 0.27 | −$4.6K / −$7.9K | — |
| 2019/2020/2023/2024 | 1.30-1.56 | all positive | — |

## Verdict: REAL, THIN, CONSISTENT — a second-tier edge
- **Holdout-positive with perfect train/holdout co-move (1.14/1.14)** — no overfit signature at all;
  the edge is small but stable. Positive 5 of 7 years.
- Does NOT meet the champion bar (holdout PF ≥1.3): it's roughly a third of wick_trap's margin.
- Same long-only bear weakness as everything else (2018/2022 negative).
- Live record (+$2.9K, only live-green archetype, driving the current recovery) is CONSISTENT with
  this profile rather than luck — thin edge + favorable recent tape.

## Standing
The book now has TWO honestly-positive strategies: **wick_trap (champion, PF 1.43 holdout)** and
**liquidity_compression (thin second, PF 1.14 holdout, live-confirmed)**. funding_divergence
remains a positive garnish (~4 trades/yr). Everything else is live- and validation-negative.
No config action required (LC already trades in the full book); its per-trade expectations are now
documented for live scoring alongside wick_trap's.

## Cross-references
[[strategy_book_review_2026_07_10]] (flagged LC for battery — resolved) · [[trailing_sweep_verdict_2026_07_13]]
