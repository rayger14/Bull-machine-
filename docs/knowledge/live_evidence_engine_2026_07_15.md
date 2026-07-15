# Live Evidence Engine — first report (2026-07-15)

**Tool**: `bin/live_evidence.py` — reads the three metadata streams the paper
runner already collects (trades.json, signal_log.json, phantom_trades.json)
and produces four decision-grade sections. Read-only; no strategy changes.

```bash
# on the server (or locally against synced files)
python3 bin/live_evidence.py --dir results/coinbase_paper
```

## First-report findings (128 positions, 2026-02-15 → 2026-07-14)

### 1. Scorecard
- **liquidity_compression: live PF 1.30 (n=26, +$2,919) vs holdout 1.14 — ON TRACK.**
  First strategy with live sample large enough to judge, and it's beating its
  offline expectation.
- wick_trap: n=1 live (the July 6→? loser). No verdict possible; the months-long
  liquidity_score lockout means its live sample only started accumulating after
  the 2026-07 repairs.
- Biggest live bleeders (all data-collection-only, never validated):
  oi_divergence −$6,489 (PF 0.23, WR 12%), confluence_breakout −$3,623,
  liquidity_vacuum −$3,029.

### 2. Counterfactual — LIVE confirmation of Lesson #54
Splitting positions by `threshold_margin` (did the signal clear the fusion
threshold the validated config enforces?):

| subset | n | WR | PF | PnL |
|---|---|---|---|---|
| threshold-cleared | 20 | 25% | **0.27** | −$9,270 |
| bypass-only (sub-threshold) | 108 | 42% | **0.76** | −$13,205 |

High-fusion signals did WORSE live — the fusion-inversion finding (Lesson #54)
now confirmed on live data, not just backtest. Reinforces: never filter or
gate on fusion score.

### 3. LC time-cut watch-item ledger (automated)
Pre-registered cell x0.25_h24 (exit if MFE < 0.25R by 24h). Live count:
**1 / 15 cut events needed, 0 baseline-winners.** Status: accumulating.
The engine reconstructs MFE paths from Binance klines (`api.binance.us`
fallback — `.com` geo-blocks 451). Deploy trigger stays as pre-registered:
n≥15 cut events with ≤1 baseline-winner.

### 4. Execution costs
$13.2M round-trip notional over 5 months → $2,647 sim taker fees + $3,971
modeled slippage = **29% of |net PnL|**. Maker-first entries would recover up
to ~$3,309 over the period. This is the industry-playbook "execution alpha"
item, now measured on our own fills.

## Cadence
Run after each week of live trading (or after any notable loss). The three
questions it answers: is each validated strategy on-track vs holdout (with
Wilson CIs, so small-n can't lie)? has any pre-registered watch-item trigger
fired? are costs drifting?
