# Confluence_Breakout Forensic — What Separates Its Winners From Losers

**Date**: 2026-07-04. **Trigger**: user audit request on last 10 live CB positions (3W/7L, net −$3,417).
**Artifacts**: `results/live_audit_2026_07_02/last10_cb_enriched.csv`

## Live n=10 pattern (hypothesis)
Fusion/confidence: ZERO separation (winners 0.15 median = losers 0.15) — user's suspicion confirmed again.
The visible live pattern: all 3 winners entered within **~0.8–1.7% of the 7-day low after a −3–5% flush**
(washout reversals); losers entered extended (3–8% above the low) or after rallies — i.e. CB wins when it
accidentally catches a capitulation reversal AT a level, loses when it actually chases a "breakout" mid-bear.

## Scale validation (771 CB positions, V14 2018–2026) — honest result
- WR separation: nearly none (near-low 44% vs extended 42%) — proximity does NOT make CB a winner.
- **BUT dollar damage concentrates in extended entries**: >4% above the 7-day low = 410 positions,
  avg **−$78/trade, −$31,883 total**; near-low (<2%) = 141 positions, avg +$8 (breakeven).
- prior-72h return: no separation at scale.

## Verdict
There is no entry feature that makes CB a *winner* (42% WR, ~breakeven at best — consistent with every
prior finding that entry features don't predict outcomes). What IS real: **CB's losses concentrate in
extended/chasing entries**. A "don't chase" condition (entry ≤ ~2–4% above the 7-day low) would have
avoided ~−$32K of backtest damage while keeping the breakeven-ish near-low population. That's loss
avoidance, not edge creation — CB remains a structurally weak archetype; the discriminator only tells
you where its bleeding lives. n=10 live + tercile scale check; no WFO run; hypothesis-grade for any config change.

## Cross-references
[[live_trade_forensic_2026_07_02]] (fusion inverted) · [[winner_loser_forensic_2026_06_28]] ·
[[trader_knowledge_failed_breakdown_es_transfer_2026_06_11]] (no-chase guard = same principle)
