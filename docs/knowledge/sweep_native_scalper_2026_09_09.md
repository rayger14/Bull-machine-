# ZeroIka Equal-Lows Sweep — Native-Resolution Validation (2026-09-09)

**Origin:** the 1H swept-level restoration of liquidity_sweep FAILED (starved to
n=14 — at 1H the textbook event can't fit one bar; the deep-wick proxy IS the
1H edge). Question: is the teaching real at its native resolution? Tested on
Binance 1m bars 2021-2026 (2.98M bars), all parameters DECLARED before running
(0.1% cluster tolerance, >=2 touches, 30m standing, <=30m reclaim, stop under
sweep low, 7bps base costs). One shot, no tuning.

## Results (all pre-registered clauses PASS)
- 7,592 fast-reclaim events (~22/day available): after-cost +9.2bps@4h,
  +25.5bps@24h, WR 32% (tight-stop positive skew). **Positive 6/6 years,
  band +6..+11bps — the most era-stable edge ever measured in this project.**
- Slow-reclaim control (n=440): similar mean, unstable (4/6, sign flips) —
  speed separates CONSISTENCY, partial pass of the mechanism check.
- Cost stress: positive to ~15bps at 4h; 24h horizon robust (+17.5 at 15bps).
- Non-overlap sim (1 position, 4h, 12bps): **6/6 years (+22..+53%/yr on
  constant notional), ~$16.6K/yr per $50K, ~2.3 trades/day, maxDD -29%.**
- Latency stress (next-minute-open fills, 12bps): +4.7bps/trade, 6/6 years.

## Character
ALL-WEATHER EDGE: its best years are OUR drought years (2022 +39%, 2025 +23%)
— stop-runs thrive in volatile chop. Anti-complementary to the wave book;
unlike the parked carry sleeve it is an edge (not a fading toll) with no
liquidation-risk leg. Thin per-trade margin -> EXECUTION IS THE STRATEGY:
adverse selection during stop-cascades is unmodeled offline and could eat
several bps; venue is Binance-perp-grade fills.

## Status: VALIDATED DISCOVERY, PARKED PENDING BUILD DECISION
This is a second machine (minute-scale scalper runner), not a Bull Machine
archetype. Path if commissioned: paper-scalper on live minute feed (the true
adverse-selection measurement) -> then sizing. Do not bolt into the 1H engine.
