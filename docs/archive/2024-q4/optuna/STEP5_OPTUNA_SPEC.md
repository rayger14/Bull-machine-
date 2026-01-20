# Step 5: Optuna Constrained Sweep - Specification

## Current State (After Steps 1-3)

### Infrastructure Built ✅
1. **Stabilization**: Cooldowns (6-10 bars), daily limits (8/day), tighter gates
2. **Regime Routing**: Per-regime archetype weights + gate deltas
3. **Monthly Share Caps**: Prevent archetype over-concentration
4. **Score Propagation**: Bug fixed - archetype scores correctly propagated

### Key Findings from 2022-2024 Baseline

| Archetype | Trades | PF | Win Rate | Avg Win | Avg Loss | Status |
|-----------|--------|----|---------|---------| ---------|--------|
| **Order Block** | 9 | **43.45** | 88.9% | $133 | -$25 | 🎯 GOLDMINE |
| **Volume Exh** | 322 | 0.87 | 40.7% | $46 | -$36 | ❌ BLEEDING |
| **Trap** | 934 | 0.86 | 48.6% | $19 | -$20 | ❌ BLEEDING |

**Overall**: 1,268 trades, PF 0.92, Win Rate 46.8%, -$1,312 PNL

---

## Optuna Optimization Spec

### Objective Function
```python
def objective(trial):
    # Run on 4 rolling windows
    windows = [
        ('2022-01-01', '2022-12-31'),  # Bear market
        ('2023-01-01', '2023-12-31'),  # Recovery
        ('2024-01-01', '2024-12-31'),  # Bull market
        ('2022-01-01', '2024-12-31'),  # Full period
    ]

    # Calculate per-window PF
    pfs = []
    for start, end in windows:
        metrics = run_backtest(config, start, end)
        pfs.append(metrics['profit_factor'])

    # Objective: Median PF across windows (robust to outliers)
    return np.median(pfs)
```

### Search Space (Per Archetype)

**Trap (suppress/improve)**:
- `trap_final_fusion_gate`: [0.38, 0.50] (current: 0.42)
- `trap_archetype_weight`: [0.5, 1.0] (current: 0.8)
- `trap_cooldown_bars`: [8, 15]

**Volume Exhaustion (fix bleeding)**:
- `ve_final_fusion_gate`: [0.30, 0.40] (current: 0.34)
- `ve_archetype_weight`: [1.0, 1.5] (current: 1.2)
- `ve_cooldown_bars`: [6, 12]
- `ve_trail_atr_mult`: [0.8, 1.5] (exit tuning)
- `ve_max_bars`: [40, 80] (exit tuning)

**Order Block (expand)**:
- `ob_final_fusion_gate`: [0.28, 0.38] (current: 0.32)
- `ob_archetype_weight`: [1.2, 1.6] (current: 1.3)
- `ob_cooldown_bars`: [4, 10]

**Global**:
- `max_trades_per_day`: [6, 12] (current: 8)

### Constraints (Hard Requirements)

```python
def check_constraints(metrics):
    # Per-archetype constraints (extract from backtest logs)
    archetype_metrics = parse_archetype_metrics(output)

    for archetype, data in archetype_metrics.items():
        if data['trades'] >= 20:  # Only constrain if meaningful sample
            if data['profit_factor'] < 1.0:
                return False  # Reject if any archetype bleeds

    # Overall constraints
    if metrics['profit_factor'] < 1.2:
        return False  # Must be profitable

    if metrics['max_drawdown'] > 25.0:
        return False  # DD cap

    if metrics['trades'] < 100 or metrics['trades'] > 2000:
        return False  # Trade count sanity

    return True
```

### Optuna Settings
- **Sampler**: TPE (Tree-structured Parzen Estimator)
- **Trials**: 100 (about 5-6 hours on MacBook)
- **Pruning**: MedianPruner (stop bad trials early)
- **Storage**: SQLite for persistence

---

## Expected Outcomes

### Success Criteria
1. **Overall PF ≥ 1.2** (currently 0.92)
2. **Each archetype PF ≥ 1.0** (no bleeders)
3. **OB trades 20-50** (currently 9)
4. **Trap suppressed** or improved (PF ≥ 1.0)
5. **VE fixed** (PF ≥ 1.0)

### Likely Discoveries
- OB should dominate (already proven high-edge)
- Trap needs severe suppression (higher gate or lower weight)
- VE needs exit tuning (longer holds or tighter trails)

---

## Implementation Command

```bash
python3 bin/optuna_archetype_portfolio_v1.py \
  --trials 100 \
  --asset BTC \
  --output results/optuna_step5_constrained \
  --base-config configs/baseline_btc_bull_regime_routed_v1.json
```

---

## Next Steps After Optuna

1. **Analyze best trial**: Which archetypes dominate?
2. **Validate on 2025 YTD**: Forward test
3. **If OB dominates**: Consider OB-only portfolio
4. **If multi-archetype works**: Proceed to ML filter retraining (Step 4)
5. **If still bleeding**: Hard pivot to OB-only or bear strategies

---

## Key Insights to Remember

> "Order Block Retest has PF 43.45 with 9 trades. This is not noise—it's a real edge. The goal is to scale it safely while suppressing bleeders (trap/VE)."

> "Regime routing works (verified in Sep 2022 crisis). But 93% of 2022 is neutral, so impact is limited. Optuna should optimize for neutral-regime performance primarily."

> "VE was profitable in 2022 (PF 1.09) but bleeding in 2023-2024 (PF 0.81-0.87). This suggests regime-specific issues or exit problems, not detector issues."
