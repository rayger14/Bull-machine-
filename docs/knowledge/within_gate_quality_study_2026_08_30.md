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
