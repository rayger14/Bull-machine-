# Feature-Store Metrology Audit — FULL FINDINGS (2026-08-28)

(Agent audit, read-only. 53 consumed features: ~13 clean, 24 drift, 8 frozen, 8 one-sided.)

## THE HEADLINE: two mutually-broken store lineages; neither is live
- **V12_ENHANCED** (backtester default): Wilder indicators CORRECT, but chop = deprecated 1−ADX/100 (median 0.748 vs live 0.519) + 27 columns 100% NaN pre-2022 (40% of the 2020-24 window) + placeholder constants (fusion_smc 0.5, oi 1e9, BTC.D 50, tf4h_fvg 0...). NOT REPRODUCIBLE — build pipeline deleted; can only be replaced.
- **V14→V22** (every study since 2026-06-24): chop formula correct but **ADX/RSI/ATR are SMA approximations** — build ran LiveFeatureComputer on a machine WITHOUT TA-Lib; silent fallback branches (live_feature_computer.py:2944-3030). ADX +38% vs Wilder; P(rsi<30) 2.7x live. Server HAS TA-Lib 0.6.8 → live is Wilder-correct.
- Pre- and post-Jun-24 study cohorts sit on stores with OPPOSITE bias — not comparable to each other.

## Ranked repairs (consumed features only)
1. tf4h_wyckoff_bearish/bullish_score — PINNED 0.89/0.93 in V12 vs live 0.02/0.00 → **Boosts 1-2's wyckoff leg is a 97.6% free pass in backtest vs 3.6% live; 3-of-3 boost fires 42.9x too often.** REBUILD
2. drawdown_persistence — 7.3x (risk_temp weight 0.50 + distress exits). REBUILD
3. chop_score — 7.5x at ≤0.5, 83x at ≤0.25 (failed_continuation's PF 13.47 = starved-gate artifact). REBUILD
4. taker_imbalance — whale-conflict leg <−0.5 has fired 0.00% in EVERY backtest vs 12.8% live. REBUILD
5. wick_lower_ratio — store uses wick/BODY 0-10 (and duplicates wick_ratio); live wick/RANGE 0-1 → 4.8x. liquidity_sweep (#1 PnL archetype) gates 4.3-4.8x over-permissive. REBUILD
6. liquidity_score — **LIVE side broken**: reads atr_percentile/oi_change_4h before they're written (order-of-computation bug, live_feature_computer:809-817) → live pinned low. RECONCILE (fix live).
7. ema_slope_21/50 — live computes pct_change(1) not (21/50): ~16x scale vs store; exits mis-fed. RECONCILE
8. adx_14 — absent from every store → defaults 20.0 → adx<25 always true in backtest. REBUILD (alias adx)
9. V12 pre-2022 blackout (27 cols NaN incl. all BOS/CHoCH/smc/fusion). REBUILD
10. crash_frequency_7d — store fraction vs live count; >=2 unreachable in backtest. REBUILD
11. funding_oi_divergence 22x starved; whale oi_4h leg 14x over-fires. RECONCILE
12. Known-dead confirmations + placeholder constants. Also: regime_service dead (missing .pkl), logic_v2_adapter dead, YAML exit blocks dead, **Boosts 3-7 ON live but OFF in backtest config** (config asymmetry independent of features).

## Systemic number
Reconstructed CMI on V12 vs live inputs: median dynamic threshold **0.514 backtest vs 0.430 live (+19.5%)** — stress-scaled limits/regime derivation shifted book-wide.

## Exposure of conclusions
- Same-store A/B deltas (sizing package, OBR gate, exit tests): mostly IMMUNE (bias on both sides).
- NOT immune: all absolute baselines (PF/PnL/trade counts); wyckoff_phase & distribution_exhaustion boost validations (free-pass leg); liquidity_sweep's #1-earner status; whale-conflict sizing (taker leg never fired offline); failed_continuation PF 13.47; every live-vs-backtest parity verdict; Optuna tunings on V14+ ATR.

## Not covered: bar-level timing (no live/store overlap window); server TA-Lib drift unverified; 116 store-only cols not formula-diffed unless consumed.
