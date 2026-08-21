# Sizing-Inversion Study — flat-notional vs ATR-inverse — 2026-08-21

**Origin**: Mancini transfer map #1 — their meta-label audit retired "size down on wide stops" as INVERTED (deep flushes = best class). BM sizes notional = risk$/stop_distance_pct (backtest_v11_standalone.py:1449) → corr(size, stop_width) = −0.53.

## Observational (baseline trade log, 1,603 sized positions)
Stop-width quartiles: Q1-Q3 WR 42%, avg ~$100. **Q4 (wide, ~4.9%): WR 51%, avg $369 — 55% of ALL profits from one quartile, at the smallest size.** Inversion signature confirmed in BM's own book.

## Interventional A/B (pre-registered; worktree; divergence verified corr −0.53→+0.06, mean size unchanged $36.6K, entries identical 3,512)
Variant: `notional = risk$/0.025` (constant ≈ median stop) — flat notional, ATR stop placement unchanged.
| | ATR-inverse | FLAT | gate |
|---|---|---|---|
| PnL | $268.9K | **$300.3K (+11.7%)** | PASS |
| MaxDD | −16.5% | −16.0% | PASS |
| PF | 1.41 | 1.41 | PASS |
| Sharpe | 1.53 | 1.44 | caveat: lumpier per-trade variance |
**Consistency: 5/5 years improve (incl. 2022 bear −18.0K→−13.2K); 11/15 archetypes improve; wick_trap +$13.5K (+22%), trap_within_trend +$10.3K, liquidity_sweep +$9.2K. Degraded: exhaustion_reversal −$6.6K, spring −$3.0K, confluence_breakout, long_squeeze — all non-validated instruments.**

## Method note (3rd golden-master catch this week)
First "run" was byte-identical — worktree creation had silently failed ('main' already checked out) and the unpatched baseline re-ran. Fixed with --detach + a pre-launch PATCH-BOUND check + post-run divergence check. Never read a result before verifying the variant diverged.

## Status
Candidate LIVE change (sizing-family; boosts/sizing historically 2/2 accepted vs filters 0/9). NOT applied anywhere. Honest limits: single-instrument, 2020-2024 (no offline holdout exists — store ends 2024-12); real OOS = live forward. Proposed rollout if approved: config-level change to live sizing with a 4-week live sample review; optionally scope to validated archetypes only (wick_trap/LC/OBR all improve or hold).
