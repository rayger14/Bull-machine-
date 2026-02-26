# Bull Machine v17 — Development Guidelines

## Current Architecture (2026-02-25)

**v17 Whale Footprint**: 16+1 archetypes + YAML hard_gates + 301-col feature store + Optuna ATR + signal dedup + heuristic fixes + whale conflict penalty + Smart Exits V2 + $100K capital

### Key Files
| Purpose | File |
|---------|------|
| **Backtester** | `bin/backtest_v11_standalone.py` (--commission-rate 0.0002, --slippage-bps 3, --start-date, --initial-cash 100000) |
| **Config** | `configs/bull_machine_isolated_v11_fixed.json` (dynamic threshold adaptive fusion) |
| **Feature Store** | `data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet` (61,306 bars x 301 cols, 2018-2024) |
| **Archetype YAMLs** | `configs/archetypes/*.yaml` (hard_gates, fusion_weights, regime_preferences, exit params) |
| **Exit Logic** | `engine/archetypes/exit_logic.py` (Smart Exits V2: composite invalidation, distress half-exit, chop trailing) |
| **Archetype Engine** | `engine/archetypes/archetype_instance.py` (whale conflict penalty, hard gates, fusion scoring) |
| **Live Runner** | `bin/live/coinbase_runner.py` (Coinbase BTC-PERP-INTX paper trading) |
| **Dashboard** | `dashboard/` (React 19 + TypeScript + Vite + Tailwind CSS 4, served at port 8081) |

### Production Results (2020-2024, $100K)
| Metric | Value |
|--------|-------|
| PF | 1.58 |
| PnL | $223K |
| MaxDD | -13.3% |
| Sharpe | 1.41 |
| Trades | 915 |
| Win Rate | 78.8% |

---

## Quick Validation

```bash
# Full backtest (2020-2024)
python3 bin/backtest_v11_standalone.py --start-date 2020-01-01 --commission-rate 0.0002 --slippage-bps 3

# Expect: ~915 trades, PF ~1.58, PnL ~$223K
```

---

## Critical Paths That Affect Signal Generation

- `engine/archetypes/archetype_instance.py`: Hard gates + fusion scoring + whale penalty
- `engine/archetypes/logic_v2_adapter.py`: Feature mapping adapter layer
- `engine/archetypes/threshold_policy.py`: Dynamic threshold resolution
- `configs/archetypes/*.yaml`: Per-archetype hard gates and weights
- `configs/bull_machine_isolated_v11_fixed.json`: CMI parameters and thresholds
- `engine/context/regime_service.py`: CMI v0 regime scoring

---

## Architecture Overview

### CMI v0 Regime System (Orthogonal to Archetype Fusion)
- **Dynamic threshold** = base + (1 - risk_temp) * temp_range + instability * instab_range
- **Config**: base=0.18, temp_range=0.38, instab_range=0.15, crisis_coeff=0.50
- **risk_temperature**: trend_align(45%) + ADX(25%) + fear_greed(15%) + drawdown(15%)
- **instability**: chop(40%) + wick_score(25%) + vol_instab(25%) + adx_weakness(10%)
- **crisis_prob**: base_crisis(45%) + sentiment_crisis(45%) + vol_shock(10%)
- CMI must be orthogonal to archetype fusion — no double-counting

### Smart Exits V2 (exit_logic.py)
Priority chain: hard_stop → invalidation → distress → profit_targets → time_exit → reason_gone → trailing_stop → runner
- **Composite invalidation**: 5-feature score (BOS, RSI, EMA slope, volume), threshold 4/5, wick_trap + retest_cluster only
- **Distress half-exit**: 50% exit when underwater + 4/5 distress signals, all archetypes
- **Chop-aware trailing**: 0.75x at chop>0.45, 0.88x at chop>0.35
- **Regime exit scaling**: DISABLED (all factors = 1.0, was net negative)

### Whale Footprint System
- **Whale conflict penalty** (archetype_instance.py): Direction-aware 4-signal check, 5%/10%/15%/20% tiers
- **Hard gates**: Per-archetype YAML configs, evaluated BEFORE fusion scoring
- **derivatives_heat**: CMI component, DISABLED (weight=0%) pending 3+ years OI data

---

## Common Gotchas

### NaN Guards
```python
# Always check for NaN in feature extraction
val = bar.get('some_feature', 0.0)
if val is not None and val == val:  # val != val is True for NaN
    use_value(val)
```

### Float Comparison
```python
# WRONG
if position.quantity == 0: ...

# CORRECT
if abs(position.quantity) < 1e-8: ...
```

### Exit Percentage Calculations
```python
# WRONG - compounds
exit_qty = current_qty * pct

# CORRECT - from original
exit_qty = original_qty * pct
```

---

## Deployment

- **Server**: `165.1.79.19` | **SSH**: `ssh -i ~/.ssh/oracle_bullmachine ubuntu@165.1.79.19`
- **Deploy**: `./deploy/deploy.sh` (builds dashboard, syncs code, restarts services)
- **Services**: coinbase-paper (800MB limit) + dashboard (8081, 200MB limit)
- **Monitor**: `sudo journalctl -u coinbase-paper -f`

---

## Testing Checklist

Before committing code changes:

- [ ] Run backtest: `python3 bin/backtest_v11_standalone.py --start-date 2020-01-01 --commission-rate 0.0002 --slippage-bps 3`
- [ ] Check PF >= 1.50 (floor), PnL >= $200K (floor)
- [ ] Check MaxDD <= -15.0% (ceiling)
- [ ] Validate config JSON: `python3 -m json.tool < configs/bull_machine_isolated_v11_fixed.json`
- [ ] Check git diff is reasonable (not 2,000+ lines)

---

**Last Updated**: 2026-02-25
**Architecture Version**: v17 Whale Footprint + Smart Exits V2
