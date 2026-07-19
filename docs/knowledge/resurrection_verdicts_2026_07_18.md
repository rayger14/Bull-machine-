# Founding-spec resurrection verdicts — round 1 (2026-07-18)

Queue from founding_knowledge_archaeology_2026_07_17.md. Both top items
tested exactly as pre-registered (original spec verbatim, zero tuning).

## #2 HOB demand-reaction (S9 hob_reaction) — REJECTED
Full original HOBDetector, untouched defaults, V15 battery:
| window | n | PF | PnL |
|---|---|---|---|
| wfo_train 2018-22 | 66 | **2.10** | +$12,920 |
| y2023 | 62 | 0.80 | −$2,296 |
| y2024 | 55 | 1.47 | +$4,562 |
| holdout 2025-26 | 64 | **0.66** | −$4,390 |
Train brilliant, holdout dead — the edge (if ever real) existed in
2018-2022 market structure and is gone. Notable defensive trait: ZERO
trades in 2022 (consolidation prerequisite avoids bear). CPCV not run:
battery co-move failure is terminal per pre-registration. hob_reaction
stays in configs/archetypes (batteries) but must NEVER be added to the
live dir (configs/champion/archetypes_v14rq).

## #1 Bojan trap-reset spec — REJECTED at 1H BTC
Signature (direction flip + body ≥1.25×ATR + opposite wick >30%) fires
**42 times in 8.5 years** (0.06% of bars) — big-body and big-wick almost
never co-occur on 1H BTC (0.58% jointly). Zero overlap with wick_trap
entries (n=0 in train AND holdout). Forward 24h return after signature:
**−1.17%** vs +0.11% baseline (n=42, small). Too rare AND wrong-signed.
The spec likely describes higher-timeframe/other-asset behavior.

## Calibration (important)
The "richer original beats the distilled proxy" hypothesis is now 0-for-2.
The surviving proxies (wick_trap, order_block_retest) are not degraded
shells — they are the distillations that actually carry edge on 1H BTC.
Remaining queue items (#3 reclaim-speed, #4 equal-cluster magnets,
#5 PO3 sequencing, #7 dynamic risk, #8 wick-magnet targets) keep their
pre-registrations but with tempered priors; test opportunistically, one
at a time, cheap overlay studies first.

## #10 Moneytaur structure trailing — REJECTED (5th exit-mod with same signature)
Config-gated `structure_trail` in exit_logic (OFF by default, kept in code
like breakeven): after +1R, stop = max(entry+0.5R, tf1h_prev_low − 1×ATR).
| | train | holdout |
|---|---|---|
| wick_trap | 1.10 → 1.17 ✓ | 1.71 → 1.37 ✗ |
| order_block_retest | 1.29 → 1.60 ✓ | 2.08 → 0.86 ✗ |
Monotone train-up/holdout-down — identical gradient to trailing-start sweep,
wider stops, breakeven, early protection. The 84% capture ratio explains it:
exits are already near-optimal; every "protective" tweak sells the holdout's
winners early. Exit-modification studies are now 0-for-5 with this exact
shape — future exit proposals need an extraordinary prior to justify a run.

## #8/#12 Fib time + hidden-fib price zones — descriptive study, DOCTRINE INVERTED
Split V15 champion positions by store fib features (entry-time). wick_trap
(n=85 train / 35 holdout positions):
- **fib_in_discount: PF 0.12 train / 0.45 holdout** (n=9/8) vs base 1.10/1.72
  — buying the flush in fib DISCOUNT is consistently terrible in BOTH windows.
  Premium-zone flushes: 1.13/4.67. Fib doctrine ("buy discount") is INVERTED
  for the washout-reversal champion: flushes near highs reverse, flushes in
  discount keep falling.
- temporal_confluence>=0.6: 0.67/1.01 vs base — high time-confluence also
  consistently WORSE for wick_trap. Explains retest_cluster's (fib-time-gated
  identity) rejection.
- OBR: train positives (time_cluster PF 2.28, temporal 5.71) but holdout
  n=1-5 — no conclusion possible.
**Status: WATCH-ITEM, not action** (filters 0-for-8 house rule; combined
discount n=17 < 30 floor). Pre-registered trigger: wick_trap discount-skip
becomes a deploy candidate at n>=30 combined discount entries with PF still
<0.7 in both windows. Note: the production ×0.92 premium PENALTY points the
wrong way for wick_trap per this data — do not extend it.
