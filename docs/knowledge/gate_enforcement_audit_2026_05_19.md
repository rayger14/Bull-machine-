# Gate-Enforcement Audit — Live Trades

**Date**: 2026-05-19
**Source**: `/tmp/trade_outcomes_live.csv` (67 live trades, 2026-04-20 → 2026-05-17)
**Method**: Read-only evaluation. For every live trade, loaded the archetype's configured `hard_gates` from YAML and tested each gate against the entry-time features captured in the trade row. Reported pass/fail per gate per trade, then aggregated per archetype.

## TL;DR

**The gate-immune architecture (Rule 10) is real and actively bleeding live PnL.**

- **22 of 67 live trades (33%) fired despite explicit hard_gate violations.**
- Violators net PnL: **−$5,543**. Clean trades: **+$3,449**.
- **`oi_divergence` and `long_squeeze` are 100% gate-immune** in live — every single live trade violated their own configured entry gates.
- **`liquidity_compression` is the lone counter-example**: `gate_mode: hard` set in its YAML, 0% violation rate, +$2,485 PnL, 17W/3L. The hard-mode flip works when it's applied.

## Per-archetype breakdown

| Archetype | mode | n | violators | viol % | win\|v | lose\|v | win\|c | lose\|c | PnL viol | PnL clean |
|-----------|------|--:|----------:|-------:|-------:|--------:|-------:|--------:|---------:|----------:|
| `liquidity_compression` | **hard** | 20 | 0 | **0%** | 0 | 0 | 17 | 3 | $0 | **+$2,485** |
| `liquidity_sweep` | soft | 4 | 0 | 0% | 0 | 0 | 3 | 1 | $0 | −$382 |
| `retest_cluster` | soft | 1 | 0 | 0% | 0 | 0 | 0 | 1 | $0 | −$1,090 |
| `funding_divergence` | soft | 7 | 1 | 14% | 0 | 1 | 6 | 0 | −$885 | +$1,159 |
| `confluence_breakout` | soft | 22 | 8 | 36% | 4 | 4 | 11 | 3 | −$239 | +$1,277 |
| `long_squeeze` | soft | 7 | **7** | **100%** | 4 | 3 | 0 | 0 | −$618 | $0 |
| `oi_divergence` | soft | 6 | **6** | **100%** | 0 | 6 | 0 | 0 | **−$3,800** | $0 |
| **TOTAL** | | 67 | 22 | 33% | | | | | **−$5,543** | **+$3,449** |

## The most-violated gates

| Count | Archetype | Gate |
|------:|-----------|------|
| 8× | `confluence_breakout` | `volume_zscore min 0.5` |
| 7× | `long_squeeze` | `ls_ratio_extreme min 1.5` |
| 5× | `oi_divergence` | `rsi_14 max 35.0` |
| 5× | `oi_divergence` | `oi_change_4h max -0.02` |
| 3× | `oi_divergence` | `oi_change_24h max -0.03` |
| 2× | `long_squeeze` | `rsi_14 min 60.0` |
| 1× | `funding_divergence` | `ls_ratio_extreme max -0.5` |

## Worst examples

### `oi_divergence` — 6/6 trades violated gates, all 6 lost, −$3,800

The configured gates require **declining OI + extreme oversold RSI** (the contrarian setup):

```yaml
hard_gates:
- feature: rsi_14;          op: max; value: 35.0   # RSI below 35 (oversold)
- feature: oi_change_4h;    op: max; value: -0.02  # OI dropping 2%/4h
- feature: oi_change_24h;   op: max; value: -0.03  # OI dropping 3%/24h
```

Live reality:

| Date | RSI | OI Δ 4h | OI Δ 24h | PnL |
|------|----:|--------:|---------:|----:|
| Apr 27 | **40.2** | **+0.5%** | +1.7% | −$735 |
| Apr 29 | **46.6** | −0.9% | — | −$599 |
| May 6 | **53.2** | — | **+0.6%** | −$644 |
| May 7 | **42.1** | −1.0% | — | −$577 |
| May 13 | — | **+0.9%** | −1.4% | −$556 |
| May 14 | **72.4** | −0.9% | — | −$686 |

May 14 fired with RSI **72.36** when the gate requires RSI ≤ 35. The intended setup ("extreme oversold + OI capitulating") wasn't even close. Engine fired anyway.

### `long_squeeze` — 7/7 trades violated `ls_ratio_extreme min 1.5`

The archetype needs longs "extremely overcrowded" (`ls_ratio_extreme ≥ 1.5`). Live values were all NEGATIVE (-0.5 to -1.4) — the OPPOSITE setup. Yet all 7 trades fired.

| Date | ls_ratio_extreme | rsi_14 | PnL |
|------|-----------------:|-------:|----:|
| Apr 23 | −1.07 | **48.5** | −$459 |
| May 1 | −0.50 | 64.0 | −$346 |
| May 4 (×4) | −1.40 | 73.9 | +$616 net |
| May 7 | −1.00 | **29.5** | −$428 |

The May 4 wins are a single fired-and-scaled-out trade. The other entries fired against the archetype's stated identity.

### `confluence_breakout` — 36% violation rate, violators net-losing

Configured: `volume_zscore min 0.5`. Live shows 8 trades fired at vol_z ≤ 0 (some as low as −0.71). Among violators: 4 wins / 4 losses, net **−$239**. Among clean trades: 11W/3L, net **+$1,277**. The gate has measurable predictive value.

## Why this is happening (architecture)

Per Rule 10 (codified May 18):

- 6 of 7 archetypes with hard_gates declared have `gate_mode: soft`.
- Global `bypass_threshold: true` is set in `bin/live/v11_shadow_runner.py:1009`-area.
- With `gate_mode: soft`, a failed gate only PENALIZES fusion (does not block).
- With `bypass_threshold: true`, the fusion penalty has no consequence either.
- Net: hard_gate violations are silently ignored in live.

Only `liquidity_compression.yaml` has `gate_mode: hard`. Result: 0% violations, +$2,485 PnL. The hard-mode mechanism itself works correctly — it just isn't applied anywhere else.

## Counterfactual

If we'd hard-enforced gates across the board:
- Saved: −$5,543 from violator losses
- Lost: ~$617 from `long_squeeze` May 4 wins + $2,200 from `confluence_breakout` Apr 20 win (both would have been blocked)
- **Net counterfactual PnL: ~+$2,094 vs actual −$2,094 (a ~$4,200 swing across 27 days)**

This isn't a small effect.

## Recommendation

**Path A (system-wide, simplest)**: couple the flags. When `bypass_threshold: true`, force hard_gate evaluation regardless of `gate_mode`. One config-level change, no per-archetype tuning, prevents anyone from re-creating the gate-immune state by accident.

**Path B (per-archetype, safer)**: flip `gate_mode: soft → hard` on the 4 worst offenders: `oi_divergence`, `long_squeeze`, `confluence_breakout`, `funding_divergence`. CB was tested with hard mode May 17 and over-blocked (lost $17K in backtest). However, the CB rejection was about a *different* hard_gate (the fusion-related one). The simple `volume_zscore min 0.5` hard-enforce we'd add here is a single feature, narrower in scope. Worth re-testing.

**Path C (research)**: keep the current architecture but add a `pre_entry_validation: true` per-gate flag that bypasses the bypass — gates with this flag always block regardless of gate_mode or fusion threshold.

I'd start with **Path A** — it's the cleanest and would have prevented all 22 violator trades in the audit. If LC + the 3 clean-trade archetypes show no degradation in a full backtest, ship it. If LC degrades, fall back to Path B with per-archetype tuning.

## Files

- Audit script: `/tmp/audit_gate_enforcement.py`
- Per-trade details: `/tmp/gate_audit_details.csv`
- Raw output: `/tmp/gate_audit_output.txt`
- Live trade source: `/tmp/trade_outcomes_live.csv` (May 17 snapshot)

## Caveats

- 67 trades is small (~30 days). Pattern is directionally clear but the dollar magnitudes are noisy.
- 5 gate features couldn't be evaluated (not captured in trade_outcomes.csv schema): `vol_shock`, `instability`, `rsi_divergence`, `adx_14` (csv has `adx` but format unclear), plus all `derived:*` gates. Real violation rate is likely HIGHER than 33%, not lower.
- Production trade_outcomes.csv schema is older than the `feat/trade-outcomes-schema-fix` branch which adds more columns. If that branch lands first, future audits will catch more violations.
