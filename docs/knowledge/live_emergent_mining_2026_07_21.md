# Live emergent-knowledge mining — full data sweep (2026-07-21)

Two-agent sweep of ALL live data (196 positions Feb-Jul, 791 hourly feature
rows since 06-18, phantoms, maker shadow, funding, equity).

## Headline emergent findings
1. **Concurrency pileups are the book's real killer** (n=43, clears bar):
   entries made with 2+ positions already open ran PF 0.42 (-$14,725) vs
   0.87 solo. Peak: 7 concurrent same-direction longs = $367.5K notional
   (~4.7x on equity) 48h before the all-time trough. FIVE close-days
   (Jul 17/13, Mar 8/6, Apr 2) = 97% of all losses — episodic correlated
   blowouts, not steady bleed. "Losses follow losses" (z=-4.67 trade-level)
   is THIS artifact — day-level clustering z=0.06 = none. No cooldown logic.
2. **Junk book at full size WAS the drawdown**: validated-only counterfactual
   (LC+WT+OBR actual trades) = +$2,355, MaxDD -4.9% vs actual -$24K, -31.1%.
   $26.4K equity difference. Every rejected-archetype position sized $52.5K
   same as validated. Junk at 10% size ≈ breakeven book, identical data.
3. **Live exits capture ~59% of winners' peaks** vs 84% backtest benchmark
   (n=21 winners; median winner MFE 1.85R vs loser 0.26R — separation
   healthy). 3/24 losers touched +1R first. Monitor, don't tune (exit mods
   0-for-5).
4. **oi_divergence OI-gate anti-selects** (suggestive): taken 2/16 wins vs
   gate-BLOCKED phantoms 12/26 (Fisher p~0.025, multiplicity-borderline).
   Trigger: blocked n>=30 with WR gap >20pts -> WFO study of the gate.
5. **funding_z live inversion**: losers entered at funding_z 0.44 vs winners
   0.10 — OPPOSITE of the backtest discriminator. Re-check at n>=200
   stratified.
6. **Maker shadow first read: 16/16 filled within 1 bar**, est. ~$2.1K/mo
   recoverable. n tiny; audit the fill rule before acting.
7. **New frozen features** (auto-caught by health sweep): macro_regime
   pinned 'neutral' 33d, tpi_signal pinned 1.0, tf1d PTI block flat,
   wyckoff_bullish_score froze 07-07 (near-dead before: 2 distinct values),
   tf1d_wyckoff_bearish_score flat since 06-24. Wiring-check list.
8. LC live PF 1.20 (n=27) vs holdout 1.23 — ON TRACK. LC time-cut ledger:
   +1 event (0.20R@24h, -$848, loser) -> 2 total, 0 winners cut.

## Nothing-list (checked, no signal)
Funding costs (-$38.50 total — ignore forever), size-outcome (no variation,
flat $52.5K; naive per-fill quartile cut is an ARTIFACT — documented),
leverage (all 1.0), day-of-week/hour (pileups in costume), day-level loss
streaks, shorts (n=9), crisis_prob dose-response, volume_z/ADX/RSI/
fear_greed entry deltas (|d|<=0.11).

## Actionable (user decision, house-rule compliant)
1. **Data-collection sizing tier**: rejected 12 archetypes at 5-10% size —
   keeps ALL archetypes trading + full telemetry, cuts ~$21K of era loss.
   THE lever. (Not a filter; nothing disabled; bypass untouched.)
2. **Concurrency governor**: scale entry size down when open notional
   >$105K. Pre-registered: shadow-ledger 60 days, accept if 2+ bucket
   deficit persists at n>=60.
3. Wiring checks on the frozen-feature list.

Artifacts documented: position_size_usd is per-fill (sum per position_id);
trade_outcomes.csv is fill-level (65% "WR" = scale-out inflation).
