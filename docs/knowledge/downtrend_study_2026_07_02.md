# Downtrend Study — Defensive Skip WORKS, Naive Short has NO Edge

**Date**: 2026-07-02
**Purpose**: scope the direction-fix — (1) build a downtrend detector, (2) test skipping longs
in downtrends (defensive), (3) test whether shorting downtrends has edge (offensive).
**Artifact**: `results/champion_v14_downtrend/`, `scripts/champion/downtrend_study.py`. Full-16 book, thresholds enforced, V14.

## STEP 1 — Detector: 200-day is the cleanest
% of bars flagged "downtrend": dt_200d fires 100% of 2022, 72% of 2018 (real bears) but only
23% of 2020 (bull) — best regime separation. dt_50d/dt_death are noisier (flag ~33% of the 2020
bull). **Use price < 200-day mean.**

## STEP 3 — SHORT FEASIBILITY: no naive edge (important)
Forward returns DURING detected downtrends:
| detector | fwd 24h | fwd 72h | % negative (72h) |
|---|---|---|---|
| dt_200d | +0.03% | +0.10% | 49% |
| dt_50d | −0.03% | −0.05% | 49% |
| dt_death | +0.03% | +0.06% | 48% |

**Being below the 200-day mean does NOT predict the next few days are down — it's a coin flip
(~49% negative, ~0% mean).** BTC spends long stretches below its 200d mean chopping sideways and
ripping bear-market rallies. So a naive "short the downtrend regime" has NO edge and would lose to
fees/funding. **The right posture in a downtrend is CASH (skip), not SHORT.** A profitable short
needs a precise short SETUP predicting imminent downside — a separate, harder study with no evidence
yet that such an edge exists. DO NOT build a blind regime-short.

## STEP 2 — DEFENSIVE SKIP: strong, deployable (the win)
Full-16 book, skip long entries when detector fires:
| variant | 2018 | 2022 | holdout | full PnL | full PF | MaxDD |
|---|---|---|---|---|---|---|
| baseline | −40,389 | −43,156 | −14,366 | 163,279 | 1.16 | **51.2%** |
| **skip_dt_200d** | **−7,485** | **0** | −8,427 | **197,891** | **1.29** | **16.4%** |
| skip_dt_50d | −35,512 | −21,602 | −6,902 | 189,601 | 1.26 | 42.0% |
| skip_dt_death | −20,977 | −26,382 | −1,735 | 174,178 | 1.23 | 28.5% |

**skip_dt_200d is the clear winner and a genuinely strong fix:**
- 2022 (worst bear): −$43,156 → **$0** (stayed in cash all year)
- 2018: −$40,389 → −$7,485
- **MaxDD: 51% → 16.4%** — cut portfolio-destroying drawdown by two-thirds
- Full PnL: $163K → **$198K** — MORE money (avoiding the big losses beats the skipped bull-pullback longs)
- Full PF: 1.16 → 1.29
- Holdout 2025-26: −$14.4K → −$8.4K (much better, still negative — 2025-26 was often above 200d so fewer skips)

## Bottom line
1. **DEPLOY the 200-day downtrend skip.** It doesn't make money in bears but it STOPS THE BLEED —
   cuts drawdown 51%→16%, turns 2022 from −$43K to flat, and raises total profit. Highest-value
   deployable change found. (Note: on the LIVE full book in bypass mode this would prevent the
   long-into-bear losses the [[live_trade_forensic_2026_07_02]] documented.)
2. **Do NOT build a naive short.** Downtrend regime ≠ imminent downside (coin flip). Shorting needs
   a precise setup, not the regime tag.
3. Holdout still slightly negative — the skip is a bleed-STOP, not a profit engine. Profit in bears
   still requires either a real short SETUP (unproven) or waiting for the regime to turn.

## Caveats
- Detector uses price < 200-day mean at bar close (concurrent, no lookahead). Simple/standard,
  low overfit risk, but only ONE definition swept for the winner.
- Needs quant-analyst adjudication + CPCV before production (per discipline). n of bear years is small (2018, 2022, 2025-26 partial).
- Skip is a real entry-block (position never opens), not size=0.

## Cross-references
[[live_trade_forensic_2026_07_02]] (18 longs-straight-to-stop this addresses) ·
[[sizing_studies_verdict_2026_06_16]] (regime sizing couldn't flip bears; this SKIPS instead) ·
[[risk_overlay_verdict_2026_06_27]]
