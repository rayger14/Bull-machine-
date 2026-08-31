# Within-Gate Quality Study — 2026-08-30

**Question (user):** within the exact conditions our hard gates look for, are there
distinguishable sub-conditions that separate better trades from worse? Do Mancini's
chop-vs-trend lessons apply?

**Method:** trio silo trade logs on V23 parity store (wick_trap n=150,
liquidity_sweep n=226, liquidity_compression n=132), joined to features at the
entry bar. 8 axes tested (velocity3 = Mancini's speed law, bar_range_atr,
range24_atr, atr_percentile, chop, adx, volume_zscore, wick_lower_ratio).
**Pre-registered bar (declared before running):** top-vs-bottom tercile WR gap
>= 8pp AND sign holds >= 4/5 years. Survivors then face the **funeral-protection
PnL check**: tercile PnL totals + per-year sign (WR gaps hide big-winner classes —
the exact illusion behind funerals 1-13).

## Results

| Axis | WR gap | Years | PnL check | Verdict |
|---|---|---|---|---|
| wick_trap x volume_z | -20pp | 5/5 | high-vol tercile **net +$17.1K** (40% WR, giant crash-bottom winners) | **KILLED** — WR illusion |
| wick_trap x bar_range | -18pp | 5/5 | top tercile +$3.9K, sign flips by year | KILLED |
| LC x wick_lower | -25pp | 4/4 | top tercile PnL sign flips by year | KILLED |
| **LC x volume_z** | **+20pp** | **4/4** | **bottom tercile -$10,767, negative ALL 5 years** | **SURVIVED** |
| sweep x chop / adx | ±9pp | 4/5 | regime restatement ("in the impulse leg"), not setup-specific | noted, no action |
| Mancini velocity3 (all 3) | <8pp / sign flips | — | — | **does not transfer** to BTC 1H |

## Mancini transfer verdict

His literal velocity law fails here. His deeper lesson — *different setups want a
different tape* — is confirmed and quantified: wick_trap wins on **calm** entries
(modest volume, modest bars), liquidity_compression wins on **loud** ones (high
volume, wickless break), sweep only cares about being in a trend leg. Two flush
archetypes wanting opposite tapes is why every global entry filter died: any
one-size volume rule helps one and guts the other. Per-archetype only.

## The action: LC volume floor 1.5 -> 2.0

The 1.5-2.0 sigma band = "breaks nobody showed up for": lost money in 2020, 2021,
2022, 2023, 2024 independently. Pre-registered silo A/B (V23, floor 2.0, nothing
else touched, PATCH-BOUND-OK verified, logs diverge):

| | Floor 1.5 | Floor 2.0 |
|---|---|---|
| PnL (5yr) | $29,836 | **$37,553** |
| PF | 1.43 | **1.59** |
| Sharpe | 0.65 | **0.84** |
| WR | 78.7% | 79.5% |
| Positions | 132 | 127 |
| By year | — | 2020 flat (-$159), 2021/22/23/24 all improve |

Honest note: direct cut was only -$1,350 (17 entries); the rest of the +$7.7K is
re-sequencing — freed same-archetype capacity refilled by stronger entries. Both
effects point the same way every year.

**Why this survived where 13 gate proposals died:** it targets a class that is
net-NEGATIVE 5/5 years, not a low-WR class hiding big winners. The wick_trap
vol-floor funeral (-$23K) cut the opposite (net-positive) side — both verdicts are
consistent: keep net-positive classes, cut net-negative ones, and only PnL-by-year
(never WR) can tell them apart.

**Status:** PR opened for user decision (LC is live at full size). Tercile
boundary was 2.57; 2.0 was the pre-registered conservative round number — a 2.5
floor was NOT tested (would be tuning past the pre-registration).

## Addendum (2026-08-30): Mancini-native micro study + CPCV

1H velocity failed because it's the wrong ruler. Re-measured at Mancini's resolution
using owned Databento CME 1m bars + 8.35M aggressor-tagged trades, 60min pre-entry,
direction-adjusted (trio silo entries 2021-2024; 30-40% excluded for CME gaps):

- **wick_trap x aggr_imb15 (+19pp, 3/3 yrs):** buyer-aggressed reclaims in the final
  15min win 56%/+$16.2K; seller-hammered "reclaims" win 37%/lose. = Mancini's
  "recovers with conviction" + July flush-qualification consensus, invisible at 1H.
- wick_trap violent flushes: WR flat but PnL +$28.5K vs -$1.8K — third independent
  confirmation the violent tercile is the jackpot class; never filter it.
- sweep: slow/already-turning approaches better (flush_speed -16pp, approach_v30
  +17pp, both 3/4 yrs) — the sweep edge is NOT knife-catching.
- LC: biggest raw gaps of the study (reclaim +44pp, anti-chase -30pp) but only 2
  evaluable years — parked below the pre-registered floor.

**CPCV (pre-registered: 6 blocks, 15 combos, 7d purge, threshold learned on train
only, pass >= 12/15 positive AND median > 0):**
- wick_trap aggr_imb15 1.25x boost: **PASS 12/15, median +$1,546** (worst -$2.0K,
  best +$4.9K). Exactly at the bar; 2022 thin (5 trades). "Survived, not crowned."
- sweep axes: FAIL 11/15 both — parked, no design re-rolls.

**Path:** aggr_imb15 live sensor from Coinbase public trades feed (free), shadow-only;
final promotion verdict reserved for the 2025-26 store extension (data with zero role
in discovery). Discovery-window caveat applies to everything above: CPCV kills
regime-luck, not selection bias.

## Addendum 2 (2026-08-31): floor moved 2.0 -> 3.0 on three-era band evidence

User challenge ("silo purity; at what volume does the engine start making money")
produced the band table: 2.0-2.5 and 2.5-3.0 negative/dead in ALL THREE eras
(virgin 2018-19, discovery, fresh); 3.0-4.0 and 4.0+ positive in all three.
Pre-registered virgin assertion PASSED (>=3.0: $343/trade 55% WR vs $156 42%).
Engine A/B at floor 3.0 (full range): PF 1.43->1.86, discovery +$8.6K,
fresh -$6.4K -> +$3.1K PURE SILO, virgin flat with fewer trades. 7/9 years
green. LC's true thesis: explosive coil breaks ONLY. PR #76 amended to 3.0.
Robustness: bootstrap P(win) 98% fresh; floor curve is a plateau 1.9-3.0+.
