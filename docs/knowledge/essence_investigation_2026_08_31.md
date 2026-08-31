# The Essence Investigation — 2026-08-31

**Question:** why did the graduated book bleed in 2025? Can "real absorption vs
fake bounce" (the essence behind the wick form) be detected pre-trade?

## Established facts
1. **2025 regime:** slow −35% markdown, lowest big-move count since 2023 (365
   vs 1,764 in 2021). Book is 100% long flush-buyers.
2. **Mechanics are fine:** winners resolve +0.86R@6h with MAE −0.43R; losers
   −0.65R@6h blowing to −2.97R, 95% die by stop; avg win $983 > avg loss $799.
   Failure is at selection, decided within hours of entry.
3. **ANTI-SELECTION (the core fact, replicated both eras):** our entered
   2-sigma flushes bounce LESS than skipped ones — discovery 44% vs 59%,
   fresh 47% vs 72%. Always true; discovery profits came from exit asymmetry
   (right-tail R), not from picking well.

## Mechanisms tested and killed (all pre-registered or confounder-checked)
- fear_greed separation → CONFOUNDED (greed trades made most PnL in discovery;
  within fresh, fear bucket loses worst)
- 200d trend filter → INVERTED between eras
- flush-in-strong-trend law (ADX) → fails virgin 2018-19, flips in fresh
- volume at flush → sign flips between eras
- taker/oi/funding hourly aggregates → no separation anywhere
- buyer-aggression pre-entry (CME 15-min) → failed OOS (25% WR boosted class)
- **absorption effort-vs-result at the minute-scale low** (textbook: huge sell
  effort + no price result = accumulation → buy):
  - CME tape: 1/4 years, absorption quadrant WORST bounce rate (43% vs ~54%),
    2025 inverted (33% vs 76%)
  - Binance tape (dominant venue, 1m taker splits): 1/4 years, 50% vs 53% — nothing

## Strategic conclusion (evidence-backed, not hunch)
Pre-trade essence detection has now failed at EVERY resolution (hourly, 15-min,
1-min) and EVERY venue we can own (CME institutional, Binance dominant). The
regime shift that killed 2025 is invisible in the feature space; the textbook
orderflow signatures do not predict bounces in BTC at our scale. **Stop buying
sensors. The essence registers reliably in exactly one place: realized PnL.**
Licensed path: outcome-feedback (loss-triggered stand-down K=2/48h — positive
on two independent datasets: live +$5.0K, fresh +$6.2K), exit asymmetry (keep),
trade scorecard, golden-master replay. Prediction of regime = 0-for-everything;
adaptation to regime = 2-for-2.

Assets: binance_1m_taker parquet (scratchpad, re-fetchable in 5 min, free);
absorption/anti-selection study CSVs in scratchpad.
