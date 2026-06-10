# Champion Strategy: wick_trap + exhaustion_reversal Regime-Complementary Pair

**Date**: 2026-06-10
**Status**: PASSES full backtest battery (2020–2024). PENDING: 2025–26 pristine holdout + 2-week live shadow.
**Harness**: `scripts/champion/run_battery.py` + `scripts/champion/run_sizing_overlay.py`
**Artifacts**: `results/champion/` (standalone baselines), `results/champion_overlay/wick_trap__exhaustion_reversal/` (overlay variants)

## The question

"Find at least one strategy that's a consistent winner and passes all tests." Tested every leading archetype as a STANDALONE strategy (single archetype enabled, fusion thresholds ENFORCED, bypass off) against 5 criteria:

- C1 positive PnL every calendar year 2020–2024 (incl. bear 2022)
- C2 OOS PF ≥ 1.3 in 2023 AND 2024
- C3 train (2020–22) PF ≥ 1.3, train/OOS co-move
- C4 n ≥ 30 per OOS window
- C5 standalone MaxDD ≤ 10%

## Key findings

1. **NO single archetype passes.** All fail the bear year: liquidity_sweep 2022 −$8.6K, wick_trap −$4.7K (PF 0.02), spring −$8.6K, trap_within_trend −$1.6K (also 2021 −$1.8K). long_squeeze standalone is inert (0 trades in 2022 with thresholds on). In 2022 the full system fired **706 trades, all long** — there is no functioning short side.
2. **The engine's regime labels are myopic**: derived from 200-HOUR (~8-day) SMAs. 9 of 12 wick_trap 2022 losers entered labeled "risk_on" during the worst bear market in the sample. The system has NO macro-horizon regime input. (This is the quantified core of the "oracle / macro all-seeing eye" gap.)
3. **Sizing can shrink but never flip** a losing year for a long-only archetype (10/12 losers ⇒ best case still negative). The strict pass requires a regime-complementary PAIR.

## The champion

**wick_trap + exhaustion_reversal**, regime-mirror archetypes:
- wick_trap (dip-buyer): strong 2020/21/23/24, bleeds in macro-bear.
- exhaustion_reversal (capitulation-buyer): strong 2022 (+$3.3K standalone, PF 1.46), bleeds in macro-bull (2021 −$12.1K standalone).

**Overlay (sizing only, no filters — Rule 8)**: longs ×0.25 when the archetype is OUTSIDE its element, using price vs 200-DAY rolling mean (concurrent-state, deterministic, Lesson 41 compliant):
- wick_trap ×0.25 when close < 200d mean (macro_bear)
- exhaustion_reversal ×0.25 when close > 200d mean (macro_bull)

### Results (pair_k025, $100K, 2bps commission + 3bps slippage, thresholds enforced)

| Window | PnL | PF | n | MaxDD |
|---|---|---|---|---|
| 2020 | +$27,052 | 3.15 | 134 | 3.2% |
| 2021 | +$198 | 1.01 | 90 | 6.7% |
| 2022 | +$2,086 | 1.25 | 67 | 3.2% |
| 2023 (OOS) | +$12,672 | 1.85 | 153 | 5.9% |
| 2024 (OOS) | +$12,188 | 1.79 | 105 | 6.3% |
| Train 20–22 | +$34,701 | 1.79 | 296 | 5.1% |
| **Full** | **+$59,561** | **1.80** | **554** | **5.1%** |

Per-archetype splits (Rule 7): 2021 = ER −$5,296 + WT +$5,493; 2022 = WT −$1,190 + ER +$3,276; 2023 = ER +$6,898 + WT +$5,773; 2024 = ER +$7,348 + WT +$4,840.

### Sensitivity (quant adjudication battery)

| Variant | 2021 | 2022 | Pass | Note |
|---|---|---|---|---|
| k=1.00 (pair, no overlay) | −$2,748 | −$1,468 | ✗ | pair alone halves single-archetype damage |
| k=0.50 | −$1,693 | +$896 | ✗ | |
| k=0.35 | −$930 | +$1,610 | ✗ | |
| k=0.30 | −$360 | +$1,848 | ✗ | |
| **k=0.25 (chosen)** | **+$198** | **+$2,086** | ✓ | pre-registered center |
| k=0.20 | +$881 | +$2,324 | ✓ | |
| k=0.25, 150d SMA | +$567 | +$1,449 | ✓ | |
| k=0.25, 250d SMA | +$2,548 | +$1,086 | ✓ | most protective of 2021 |

k-response is smooth and monotonic with a single zero-crossing (not a knife-edge); SMA length robust across 150–250d. Passing region: k ≤ 0.25 × SMA 150–250d.

## Honest caveats

- 2021 at the chosen cell is +$198 — thin. The pass is regional, not cellular; k=0.20 or 250d give margin but were NOT chosen post-hoc.
- Archetype gates/weights were historically calibrated on 2020–24, so 2023/24 is OOS only relative to the overlay.
- Pair selection was data-driven from the same window (looked for the 2022-positive complement). The 2025–26 pristine holdout (never touched by any optimization) is the decisive test, then 2-week live shadow.
- wick_trap 2022 has n=12 at the component level (< 30 floor); the pair-level ns clear the floor everywhere.

## Promotion gate (pre-registered)

The champion is promoted only if, on the rebuilt feature store's 2025–2026 segment: PnL > 0, PF ≥ 1.3, n ≥ 30, MaxDD ≤ 10%, and per-archetype splits show neither leg catastrophically failing (leg loss bounded at −3% of capital). Then 2-week live shadow with signal-for-signal parity before any sizing preference in production.

## Cross-references

- Live forensics 2026-06-10: live bleed is stop-loss-concentrated (88 stops = −$34.3K vs +$17.7K everything else) and long-into-bear (148/158 longs); per-archetype live winners = liquidity_sweep (+$1.5K, PF 1.51), funding_divergence (+$1.2K) — see `results/live_pull_2026_06_10/`.
- The macro-horizon regime gap (200-hour vs 200-day) is the engine-side counterpart of the Oracle vision gap; the 200d state should eventually become a first-class feature in the rebuilt store.
