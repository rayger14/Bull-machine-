# CME Open Interest — first regime study (2026-08-21)

**Data**: CME BTC futures daily OI 2021-01→2026-08 (`data/databento/btc_fut_oi_2021_2026.parquet`, $0.13). Closes the "derivatives_heat pending 3+ years OI" gap. Quadrants = sign of 5-day price change x 5-day OI change.

## Market-level (CAUSAL: yesterday's quadrant → fwd returns from today's close)
| regime | n | fwd21d | up% |
|---|---|---|---|
| P↑OI↑ new-longs (backed rally) | 348 | **+4.50%** | 59 |
| P↓OI↑ new-shorts (squeeze fuel) | 230 | +4.57% | 58 |
| **P↑OI↓ "hollow rally"** | 405 | **+1.36%** | 51 |
| P↓OI↓ unwind | 429 | +2.52% | 51 |
| baseline | | +2.99% | 54 |
Direction replicates in fresh 2025 (outside every spent fold). Non-overlap subsample MWU p=0.083 — marginal, monthly-horizon signal. **Candidate for the regime stand-down/exposure DIAL** (ledger root-cause), NOT a per-trade filter.

## Trade-level: KILLED by causality (record this so it isn't re-run)
Same-day quadrant join on 1,190 backtest positions showed "new-shorts = worst regime" (WR 28%, −$45K). **Lagged one day (the only causal version — CME OI publishes next morning), the effect vanishes**: WR 41%, +$4K, year signs flip (+8.6/−5.2/+9.3/−8.7K). The same-day result was look-ahead (the day's OI close encodes the day's outcome). OI does not time entries; at best it scales exposure on a weeks horizon.

## Next (needs its own study before touching anything)
- Hollow-rally exposure dial: block-bootstrap significance + 2025-26 OOS + interventional replay (sizing-shaped: de-size longs while yesterday's quadrant = squeeze, or boost in backed-rally). Rules 7-10 apply.
- derivatives_heat re-enable study can now cite real OI; still a CMI change = filter-adjacent, full gauntlet required.

## Addendum (same day): two interventional tests

**OI sizing dial: REJECT.** Pre-registered (squeeze 0.75x / backed-rally 1.25x, causal t-1, sizing-only; divergence verified 23% of positions resized): PnL −$4.5K, DD −0.6pp worse, 1/4 dial years improved. The 21-day market-level signal does not transfer to sizing ~30h mean-reversion entries (consistent: trades ENTERED on squeeze days performed fine). OI at trade/daily granularity: three doors now closed with evidence (trade-level timing, daily sizing dial; monthly exposure observation remains background context only).

**oi_divergence archetype re-fed with CME OI: PREMISE PARTIALLY VINDICATED.** Swapped oi_change_24h to causal CME daily unwind, oi_change_4h to NaN (nan_policy:skip); zero parameter changes. Full period: −$8,670/WR 38% → −$825/WR 52%. **CME-covered era 2021-2024: −$5,335 → +$2,982 (2024: +$5,094).** The archetype was reading garbage (perp snapshot), not wrong in premise. n=31 trades/4yrs — thin; NOT deployable without a live daily CME OI feed (trivial: $0.13 schema) and a proper follow-up study. Status upgrade: DEAD(leak) → instrument with proven data-quality path.

## Confirmation-stack study (same day): swap the OI witness everywhere the architecture already consults it
Variant store: oi_change_24h/4h, oi_price_divergence, funding_oi_divergence re-derived from causal CME daily OI. One run, read per confirmation point (CME era 2021-2024):
- **oi_divergence: −$5,334/WR 37% → +$6,804/WR 59% — CONFIRMED, 2nd independent test, stronger than the single-column swap.** The archetype's only problem was its data.
- funding_divergence: $4.3K→$2.5K on n=4→6 — no benefit, un-callable n.
- long_squeeze: n≈5, noise.
- **Whale-conflict penalty: rest-of-book $132.7K → $107.9K (−$24.7K). DO NOT SWAP.** Root cause = SEMANTICS MISMATCH, not data quality: penalty threshold oi_change_4h<−3% was calibrated to the 4h perp distribution; −3% daily fires constantly. A witness swap must preserve the distribution the threshold was tuned on; recalibrating = fold-burning. Whale penalty keeps the perp witness.
- derivatives_heat: NOT A/B-able under bypass (threshold non-binding) — observational only, deferred.
**Net: CME OI has exactly one proven home, oi_divergence. Follow-up for promotion: live daily CME OI fetch ($0.13/pull family) + shadow the CME-fed variant live before any config change.**

## Live-audit corollary (same day, rally to 77K): identity-gate vacuity
Forensic on the rally winners: LC (x2) and liquidity_sweep won TO SPEC (volume 3.4-3.9σ, coil, 0.71 lower wick at range lows); confluence_breakout's best entry was a flush-buy wearing the wrong name (gate-immune); OBR's big "winner" matched nothing (BOS gate SOFT and false, threshold bypassed, boms floor 0.0 inert, RSI 90, no actual block — rally beta in costume). Pattern: archetypes with binding structural gates win to-spec; archetypes with soft/inert/bypassed identity gates produce beta in costume. → CANDIDATE STUDY (not run): restore identity gates to binding strength (OBR BOS→hard, boms_strength floor>0, CB gate-immunity revisit). Alpha remains negative in trends (−$40.9K vs hold since Aug 7) — entries are skilled (median MAE −0.23R), exits donate the move (ladder caps 11-23R runs at 2R); runner-slot demotion is the reconciling build.
