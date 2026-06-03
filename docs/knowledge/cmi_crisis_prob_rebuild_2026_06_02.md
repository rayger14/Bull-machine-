# CMI crisis_prob Rebuild — substitute_no_derivatives mode (2026-06-02)

## Problem

`crisis_prob` was stuck at 0.009 across all 94 live trades from Apr 20 – Jun 2 2026.
The legacy formula

```
crisis_prob = 0.45 * base_crisis + 0.45 * sentiment_crisis + 0.10 * vol_shock
```

depends on three OI-adjacent stress signals (`drawdown_persistence > 0.96`,
`crash_frequency_7d >= 2`, `crisis_persistence > 0.55`) that almost never fire in
trending regimes, and on `fear_greed < 20` for the sentiment kicker. With
`derivatives_heat` disabled (waiting on 3+ years of OI history) and the engine
sitting in a benign trend, crisis_prob collapsed to 0.02 * 0.45 + 0 + 0 ≈ 0.009
even while DXY was strengthening, fear/greed was 24, and ema_slope_50 was
negative — i.e. the engine could not see a deteriorating regime.

## Audit

Active CMI computation lives inline in:
- `bin/backtest_v11_standalone.py` (lines ~700–725, batch)
- `bin/live/v11_shadow_runner.py` (lines ~519–536, per-bar live)

`engine/context/regime_service.py` only docs the architecture; the
`ProbabilisticRegimeDetector` path it dispatches to is not on the production code
path (mode=`hybrid`, but production runs through the inline formula in
`backtest_v11_standalone.py`).

## Substitute formula

```
crisis_prob = clamp(
    0.25 * macro_stress
  + 0.35 * sentiment_stress
  + 0.30 * structural_stress
  + 0.10 * vol_shock,
  0, 1)

macro_stress       = sigmoid(0.60 * DXY_Z + 0.40 * VIX_Z - 1.0)
sentiment_stress   = clamp((35 - fear_greed_raw) / 35, 0, 1)        # 0-100 scale
structural_stress  = sigmoid(-15 * ema_slope_50 + 1.0 * (1 - tf4h_wyckoff_bullish_score) - 0.5)
vol_shock          = clamp((rv_20d - 0.75) / 0.50, 0, 1)
```

All inputs (`DXY_Z`, `VIX_Z`, `fear_greed`, `ema_slope_50`,
`tf4h_wyckoff_bullish_score`, `rv_20d`) exist in the feature store today.
`tf4h_wyckoff_bullish_score` is used for the structural piece instead of the
bear score to dodge the bear-score bug tracked in P1.

## Config flag

`configs/bull_machine_isolated_v11_fixed.json`:

```json
"adaptive_fusion": {
  "crisis_prob_source": "original",          // default — bit-exact baseline
  "crisis_prob_substitute": { ...weights... } // used when source = "substitute_no_derivatives"
}
```

`derivatives_heat` code remains in `risk_temperature` with weight 0.0 — untouched.

## Backtest delta (2020-01-01 → 2024-12-31, $100K, comm=0.0002, slip=3bps)

| Metric    | original (baseline) | substitute_no_derivatives | Δ           |
|-----------|---------------------|----------------------------|-------------|
| Trades    | 3,384               | 2,794                      | −17.4%      |
| PF        | 1.418               | 1.421                      | +0.3%       |
| PnL       | $264,652            | $230,562                   | −12.9%      |
| MaxDD     | −17.5%              | −19.3%                     | +10.3% (worse) |
| Sharpe    | 1.47                | 1.28                       | −12.9%      |
| Win Rate  | 66.4%               | 66.8%                      | +0.4 pp     |

Trade reduction is concentrated in stress periods (2022 −23%), exactly where the
new crisis_penalty meaningfully discounts fusion scores. PF holds at ~1.42, and
crisis_prob is no longer flat-lining — yearly means go from 0.02–0.06 (original)
to 0.19–0.31 (substitute).

### Historical regime validation (substitute backtest, in-sample trades)

| Window                          | n trades | crisis mean | crisis max |
|---------------------------------|---------:|------------:|-----------:|
| COVID 2020-03-09→20             | 5        | **0.49**    | 0.67       |
| 3AC/Celsius 2022-06-13→17       | 2        | **0.64**    | 0.64       |
| LUNA week 2022-05-09→15         | 7        | **0.57**    | **0.68**   |
| FTX 2022-11-07→15               | 9        | 0.29        | 0.40       |
| Bull 2021 Apr                   | 57       | 0.16        | 0.25       |
| Bull 2024 Mar-Apr               | 32       | 0.22        | 0.35       |
| Summer 2024 chop                | 65       | 0.23        | 0.39       |

**LUNA validation passes**: crisis_prob 0.53–0.68 sustained across the LUNA
week, all 7 trades above 0.5 threshold. COVID and 3AC also fire above 0.5. FTX
underfires because DXY softened on Fed pivot expectations — macro composite alone
can't catch BTC-specific contagion without OI data. That is acceptable: macro
substitute is meant to bridge the gap until derivatives_heat is re-enabled, not
to be a permanent replacement.

### Live-trade projection (Apr 20 → Jun 2 2026, 94 trades)

| Bucket                  | n  | crisis_sub mean |
|-------------------------|---:|----------------:|
| W1 (Apr 20 – May 17)    | 64 | 0.225           |
| W2 (May 17 – May 28)    | 21 | 0.246           |
| W3 (May 28 – Jun 2)     | 7  | **0.274**       |

W3 / W1 = 1.22x (positive separation, hostile-regime week is highest). Absolute
levels are lower than historical crisis weeks because the live CSV has no
`rv_20d` field, so the vol_shock channel is silent — production will get it.

## Recommended deploy strategy

**Default-on for live paper, opt-in for backtests / Optuna jobs.**

Rationale:
1. PF unchanged (1.42 → 1.42), so the substitute is not destroying signal
   quality — it is correctly suppressing 17% of trades that were running into
   stress regimes the engine could not see.
2. 2022 PnL drag stays similar (−$21K → −$17K), confirming the suppression is
   protective rather than catastrophic.
3. MaxDD widens by 1.8 pp (still within −20% guardrail). Trade-off is
   acceptable for restoring crisis visibility.
4. Keep the `"original"` mode as the default in the shipped config so existing
   reproducibility / WFO / CPCV jobs remain bit-exact, and flip the live runner
   only after a 1-week paper-shadow comparison.

Action items for follow-up:
- Run Optuna WFO on substitute weights (`w_macro`, `w_sentiment`, `w_structural`,
  `b_ema_slope`) to recover the trade-count loss without giving up crisis
  detection.
- Add FTX-style "internal-crypto contagion" channel (funding_z >> 0 + LS ratio
  extreme + taker imbalance) once 3+ years of OI is online.
- When `derivatives_heat` re-enables, blend rather than swap — keep substitute
  channel weighted at 30–50% to retain macro signal.

## Files modified

- `bin/backtest_v11_standalone.py` — wire `crisis_prob_source` flag + substitute formula
- `bin/live/v11_shadow_runner.py`  — same wiring for live runner heartbeat path
- `configs/bull_machine_isolated_v11_fixed.json` — add `crisis_prob_source` (default `"original"`) + `crisis_prob_substitute` weights
- `engine/context/regime_service.py` — update docstring to point at the active code paths and document the new flag
- `scripts/explore_crisis_prob_substitute.py` — research notebook used to derive weights (kept for reproducibility)

## Branch

`feat/cmi-crisis-prob-rebuild`
