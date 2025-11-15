# Phase 1 Quick Start Guide
## Classical Optimization with Optuna

**Ready to Run**: All Phase 0 tasks complete ✅

---

## Priority 1: Fix Trap Within Trend (CRITICAL) 🔥🔥🔥

**Problem**: 104 trades (83% of all trades), NET LOSS -$352.95, 46% WR

**Command**:
```bash
python3 bin/optuna_trap_v10.py \
  --asset BTC \
  --train-start 2022-01-01 \
  --train-end 2023-12-31 \
  --val-start 2024-01-01 \
  --val-end 2024-12-31 \
  --n-trials 200 \
  --objective maximize_pf_wr \
  --output results/optuna_trap_v10
```

**Hyperparameters to Optimize**:
| Parameter | Current | Target Range | Rationale |
|-----------|---------|--------------|-----------|
| `trap_quality_threshold` | 0.35 | [0.45, 0.65] | Too lenient, letting in junk |
| `trap_confirmation_bars` | 2 | [2, 5] | Need more confirmation |
| `trap_volume_ratio` | 1.2 | [1.5, 2.5] | Require stronger volume |
| `trap_stop_multiplier` | 2.5 | [0.8, 1.5] | **CRITICAL** - stops too wide |

**Objective Function**:
```python
def objective(trial):
    # Maximize (PF × WR) while constraining max_dd < 0.10
    pf = results['profit_factor']
    wr = results['win_rate']
    dd = results['max_drawdown']

    if dd > 0.10:
        return 0.0  # Reject

    return pf * wr
```

**Success Criteria**:
- Win Rate: 46% → **55%+**
- Avg Loss: -$78 → **-$45**
- Profit Factor: 0.88 → **1.5+**
- Expected Gain: **+$400-600/year**

---

## Priority 2: Optimize Bear Config 🔥

**Problem**: 2022 lost -$965 (25% WR), bear config threshold 0.75 = almost never trades

**Command**:
```bash
python3 bin/optuna_bear_v10.py \
  --asset BTC \
  --train-start 2022-01-01 \
  --train-end 2022-12-31 \
  --val-start 2023-01-01 \
  --val-end 2023-12-31 \
  --n-trials 150 \
  --objective minimize_loss_maintain_2024 \
  --output results/optuna_bear_v10
```

**Hyperparameters**:
| Parameter | Current | Target Range | Rationale |
|-----------|---------|--------------|-----------|
| `fusion_entry_threshold` | 0.75 | [0.30, 0.60] | Way too high, never triggers |
| `bear_position_size` | 1.0 | [0.5, 1.0] | Reduce risk in crisis |
| `bear_stop_multiplier` | 2.0 | [0.6, 1.2] | Tighter for DD control |
| `bear_min_confidence` | 0.60 | [0.55, 0.75] | Regime filter |

**Objective**: Minimize 2022 loss while maintaining 2024 performance

**Success Criteria**:
- 2022 PF: 0.88 → **1.0+** (break even or small profit)
- 2022 Loss: -$965 → **-$400 max**
- 2024 unchanged: Maintain +$1,361 profit
- Expected Gain: **+$565 total** (2022 improvement)

---

## Priority 3: Scale Order Block Retest 🔥🔥

**Problem**: Only 3.3 trades/year, but 90% WR and $151 avg PNL (GOLDMINE)

**Command**:
```bash
python3 bin/optuna_ob_retest_v10.py \
  --asset BTC \
  --train-start 2022-01-01 \
  --train-end 2023-12-31 \
  --val-start 2024-01-01 \
  --val-end 2024-12-31 \
  --n-trials 150 \
  --objective maximize_trades_maintain_wr \
  --output results/optuna_ob_retest_v10
```

**Hyperparameters**:
| Parameter | Current | Target Range | Rationale |
|-----------|---------|--------------|-----------|
| `ob_quality_threshold` | 0.40 | [0.25, 0.50] | Lower to catch more |
| `ob_retest_tolerance` | 0.010 | [0.005, 0.025] | Widen retest zone |
| `ob_volume_confirm` | 1.2 | [0.8, 1.5] | Volume strictness |
| `ob_timeframe_weight` | 1.0 | [0.5, 1.5] | MTF alignment weight |

**Objective**: Maximize trade count while WR ≥ 70%, avg PNL ≥ $100

**Success Criteria**:
- Trades/year: 3.3 → **10+**
- Win Rate: Maintain **70%+** (currently 90%)
- Avg PNL: Maintain **$100+** (currently $151)
- Expected Gain: **+$800/year**

---

## Optuna Script Template

If the scripts don't exist yet, here's the template:

```python
#!/usr/bin/env python3
"""
Optuna optimization for trap_within_trend archetype.
"""

import optuna
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bin.backtest_router_v10_full import run_backtest

def objective(trial):
    """Optuna objective function for trap optimization."""

    # Sample hyperparameters
    config = {
        'trap_quality_threshold': trial.suggest_float('trap_quality', 0.45, 0.65),
        'trap_confirmation_bars': trial.suggest_int('confirmation_bars', 2, 5),
        'trap_volume_ratio': trial.suggest_float('volume_ratio', 1.5, 2.5),
        'trap_stop_multiplier': trial.suggest_float('stop_mult', 0.8, 1.5),
    }

    # Run backtest with these params
    results = run_backtest(
        asset='BTC',
        start='2022-01-01',
        end='2023-12-31',
        config_overrides=config
    )

    # Check constraints
    if results['max_drawdown'] > 0.10:
        return 0.0  # Reject high DD

    # Objective: Maximize PF × WR
    pf = results['profit_factor']
    wr = results['win_rate']

    return pf * wr

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=200)

    print("\nBest trial:")
    print(f"  Value: {study.best_value}")
    print(f"  Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # Save results
    with open('results/optuna_trap_v10/best_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)
```

---

## Timeline

### Week 1-2: Trap Optimization
- [ ] Run Optuna on trap params (200 trials)
- [ ] Analyze Pareto frontier (PF vs WR tradeoff)
- [ ] Select best config, validate on 2024 OOS
- [ ] Update `configs/v10_bases/trap_optimized.json`

**Target**: WR 55%+, PF 1.5+, gain +$400-600/year

---

### Week 3: Bear Config Optimization
- [ ] Run Optuna on bear config (150 trials)
- [ ] Isolate test on 2022 data only
- [ ] Validate: 2022 improved, 2024 unchanged
- [ ] Update `configs/v10_bases/bear_optimized.json`

**Target**: 2022 loss reduced to -$400, gain +$565 total

---

### Week 4: Order Block Scaling
- [ ] Run Optuna on OB quality threshold (150 trials)
- [ ] Maximize detection while maintaining 70%+ WR
- [ ] Validate on 2024 OOS
- [ ] Update `configs/v10_bases/ob_retest_optimized.json`

**Target**: 10 trades/year, 70%+ WR, gain +$800/year

---

## Validation Protocol

After each optimization:

1. **In-Sample Check** (2022-2023):
   - Verify params don't overfit
   - Check improvement vs baseline

2. **Out-of-Sample Test** (2024):
   - MANDATORY - never deploy without OOS validation
   - Acceptable degradation: < 10% vs in-sample

3. **Walk-Forward Validation**:
   - Split 2022-2024 into rolling windows
   - Train on 6 months, test on 3 months
   - Ensure consistent performance across windows

4. **Combined Test**:
   - Run full 2022-2024 backtest with ALL optimized configs
   - Compare to baseline: +$1,140 → +$2,500+ target

---

## Success Gate

**Phase 1 complete when**:
- ✅ Trap WR ≥ 55%, PF ≥ 1.5
- ✅ 2022 loss reduced to -$400 or better
- ✅ OB retest scaled to 10+ trades/year, WR ≥ 70%
- ✅ Combined gain ≥ +$1,000/year vs baseline
- ✅ OOS validation passed (2024 test)

**Then proceed to**: Archetype re-enablement (H, K, L, S) with feature flags

---

## Commands Cheat Sheet

```bash
# 1. Trap optimization
python3 bin/optuna_trap_v10.py --n-trials 200

# 2. Bear config optimization
python3 bin/optuna_bear_v10.py --n-trials 150

# 3. OB retest scaling
python3 bin/optuna_ob_retest_v10.py --n-trials 150

# 4. Combined validation
python3 bin/backtest_router_v10_full.py \
  --asset BTC \
  --start 2022-01-01 \
  --end 2024-12-31 \
  --bull-config configs/v10_bases/bull_optimized.json \
  --bear-config configs/v10_bases/bear_optimized.json \
  --output results/router_v10_phase1_validation

# 5. Compare to baseline
python3 bin/compare_backtest_results.py \
  --baseline results/router_v10_full_2022_2024_combined \
  --optimized results/router_v10_phase1_validation
```

---

## Reference Files

- 📊 **Baseline Results**: `results/router_v10_full_2022_2024_combined/`
- 📝 **Analysis Doc**: `docs/analysis/ROUTER_V10_ANALYSIS_AND_RECOMMENDATIONS_CORRECTED.md`
- ⚙️ **Baseline Configs**: `configs/v10_bases/btc_*_v10_baseline.json`
- 🗂️ **Schema Lock**: `schema/v10_feature_store_locked.json`
- 🎯 **Feature Flags**: `configs/archetype_feature_flags_v10.json`
- 🧠 **PyTorch Spec**: `docs/META_FUSION_MLP_SPEC.md`

---

**Phase 0 Status**: ✅ COMPLETE
**Phase 1 Status**: 🚀 READY TO START
**Next Action**: Run Optuna on trap_within_trend
