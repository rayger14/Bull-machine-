# MFE Analysis — "Were targets too far / did we not take profit enough?"

**Date**: 2026-06-28
**Question (user)**: for entered trades, what was the best each did — maybe targets
were too far and we didn't take profit enough.
**Method**: reconstruct each trade's in-trade price path (V14 OHLC entry→exit),
measure in R-multiples (R = entry-to-stop): MFE (high-water mark), realized
(price-based, same units), capture = realized/MFE per trade, giveback = MFE−realized.
**Artifact**: `results/champion_v14/mfe_analysis.json`, `scripts/champion/mfe_analysis.py`

## Answer: the hypothesis is the OPPOSITE of what's true

| archetype | WR | winner MFE | kept | capture% | giveback | losers +1R then lost | reach 2R | reach 3R |
|---|---|---|---|---|---|---|---|---|
| exhaustion_reversal | 78% | 1.47R | 1.07R | 81% | 0.24R | **24%** | 19% | 10% |
| liquidity_sweep | 76% | 1.30R | 1.03R | 83% | 0.19R | 15% | 25% | 8% |
| spring | 65% | 2.02R | 1.36R | 83% | 0.39R | 16% | 33% | 18% |
| trap_within_trend | 75% | 1.18R | 1.01R | 87% | 0.14R | 10% | 17% | 4% |
| wick_trap_v14rq | 78% | 1.18R | 1.01R | 84% | 0.15R | 5% | 17% | 4% |
| **ALL** | **73%** | **1.42R** | **1.07R** | **84%** | **0.20R** | **14%** | **23%** | **10%** |

**We capture 84% of the available favorable move on winners, giving back a median
of only 0.20R from the peak.** We are NOT leaving profit on the table — if
anything we harvest slightly early. So "targets too far / didn't take profit
enough" is not the leak.

## What the MFE actually reveals

1. **Trades don't run far.** Median winner peaks at ~1.4R; only **23% of all
   trades ever reach 2R, <10% reach 3R.** The far scale-out tiers (2R/2.5R) and
   the "runner" rarely fire — they ARE vestigial (the kernel of truth in the
   hypothesis) — but since the 0.5R/1R tiers already capture 84%, fixing them
   adds little. BTC-1H reversals from these patterns simply top out near 1–1.5R.

2. **The real leak: winners turning into losers.** 14% of LOSING trades
   (exhaustion_reversal **24%**) were up **+1R** at some point before reversing to
   a loss. These are mismanaged winners — not a take-profit problem, a
   give-it-back problem.

## The precise, testable fix this points to
**Move the stop to breakeven after +1R** (optionally scale a portion at 1R and
trail the remainder from breakeven). This directly converts the ~14% "green-then-red"
losers into scratches/small wins. It is NOT "wider targets" (we already capture
84%) — it is protecting the profit a trade has already shown. Trade-off: some
trades that dip to breakeven then recover would be stopped early; only a WFO test
of the breakeven-after-1R rule settles the net. This is the highest-precision
exit hypothesis the data has produced and the natural next study.

## Reconciliation with prior findings
- Confirms the duration finding ([[winner_loser_forensic_2026_06_28]]): winners run,
  losers stop fast — and 14% of "losers" were winners that reversed.
- Refines my audit: the exit lever is NOT "let winners run further / wider targets"
  (no room — 1.4R median) but "stop winners from becoming losers" (breakeven stop).
- Refines the risk-overlay verdict ([[risk_overlay_verdict_2026_06_27]]): wider
  stops overfit; this is a different, profit-protection lever, untested.

## Caveats
- realized uses final/blended exit_price vs scale-out ladder — capture% is a close
  approximation, not exact per-leg accounting.
- MAE/MFE from hourly OHLC (intrabar path unknown); fine at this resolution.
- funding_divergence n=27 (shown, not weighted).
