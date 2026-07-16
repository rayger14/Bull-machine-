# Trader-Knowledge Audit — full wiring × evidence cross-reference (2026-07-16)

Two-agent audit: (A) code inventory of every trader methodology implemented and
whether it's wired live; (B) evidence grading of each concept against recorded
validation (batteries, pristine holdout, CPCV, live book). Cross-referenced.

## The one-line verdict
~7 trader-knowledge domains are implemented; exactly ONE concept — buying the
liquidity flush (wick_trap champion PF 1.43, LC thin cousin 1.14) — has cleared
every honest test. Fusion scoring is a confirmed anti-signal (4 independent
replications incl. live). A third of the book has never actually been tested
because the wires were cut.

## Live bug found and fixed by this audit
**CHoCH was dead on every live bar despite the 2026-07-13 fix**: the repaired
derivation in `_smc_features` was unconditionally overwritten to 0 by a
leftover "simplified" stub in `_extra_archetype_features`, which runs later in
`update()`. Fixed 2026-07-16 (pass-through instead of zeroing); regression
test `tests/test_choch_not_clobbered.py` pins the ordering contract. The
overlap sweep confirmed CHoCH's two keys were the ONLY clobbered features.

## Domain verdicts (evidence agent, anchored numbers)
| Domain | Verdict | Anchor number |
|---|---|---|
| Liquidity (wick_trap, LC) | VALIDATED — only real edges | wt holdout 1.43, CPCV 15/15; LC 1.14/1.14 co-move, live 1.30 |
| Fusion-as-quality | REJECTED ×4 incl. live | live threshold-cleared PF 0.27 vs bypass 0.76 |
| Wyckoff trading claims | REJECTED (detection layer real: 12/14 events) | spring holdout −$0.02K; ER holdout 0.85; spring accepts utad (1,860 vs 826 events) |
| SMC (BOS family) | TAINTED — never honestly tested | BOS flags 0 on ALL 73,829 V14 rows + all live bars pre-07-13 |
| liquidity_sweep | REJECTED post-repair (honest test) | holdout 1.10 → 0.61 after impossible-gate removal |
| Whale/derivs | oi_div rejected live (PF 0.23, −$6.5K); funding_div weak-real starved (PF 1.53 n=6, live gate was crippled pre-07-13) | derivatives_heat Rule-9 overfit (+$1.2K OOS / −$26K train) |
| Mancini/levels | REJECTED at 1H BTC (U1 starves n=3, U2 co-move fail); U3 watch-item n=23 | level_quality AUC 0.37 (inverted) |
| Exits | boring defaults won; BE retracted 2×, trailing sweep monotone overfit, invalidation exits −$54K | capture ratio already 84%; LC time-cut ledger 1/15 |

## Wiring gaps (inventory agent, top findings)
1. CHoCH clobber (fixed, above).
2. **effort_result_ratio is backtest-only** — never computed live; its gates
   (failed_continuation, volume_fade_chop) use nan_policy:skip → silently pass
   live. Backtest and live trade different rule sets for those archetypes.
3. **whipsaw + volume_fade_chop can never fire** — direction:neutral hard-returns
   None in ArchetypeInstance.detect(). Their YAMLs/Optuna params are dead weight.
4. **17 Mancini level features computed hourly live, consumed by nothing**
   (deliberate post-validation, but the doc's structural-stops and day-type
   sizing pillars were never built at all).
5. **Dead config strata**: YAML exit_logic ladders (only max_hold read),
   regime_preferences (mode not enabled), sm_* Wyckoff tuning, regime ML model
   loaded-never-called, long_squeeze vol_shock gate (feature never emitted),
   OI/funding exit invalidations (wrong feature names + disabled).
6. Schema drift: live emits tf1d_wyckoff_bullish/bearish_score; V14 store
   lacks them. Store bos_bullish/tf1h_bos_bullish still all-zero (V15 target).

## False alarm resolved
Daily Wyckoff and BOMS are NOT broken: tf1d_wyckoff_score nonzero on 70% of
all 675 live bars; historical median zero-streak is 11 days (max 68d), so
short flatlines are normal. BOMS fires 0.7% of bars (once went 277 days
silent); its 0.0 gate thresholds are deliberate (0.30 default would starve
OBR, <1% pass rate). Do NOT "fix" these.

## V15 retest candidates (most likely wrong verdicts, per evidence agent)
1. fvg_continuation / SMC-BOS family — verdict produced by the bug, not the
   market; first honest test comes with V15 backfilled emitters.
2. liquidity_vacuum — 0 trades ever from two stacked wiring faults (both fixed).
3. funding_divergence LIVE record — accrued while ls_ratio_extreme could never
   go negative; score fresh post-repair.

V15 plan: structure features have ≤1000-bar lookback and need only OHLCV →
recompute ~30-50 structure columns in parallel chunks (1000-bar warmup),
splice into V14 (add the missing 1D wyckoff graded columns), validate overlap,
re-baseline BEFORE batteries (revived gates change behavior at zero config diff).
