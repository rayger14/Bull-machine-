# Optuna Optimization Status - 2025-11-10

## Current Status
- **Optimization**: 2/100 trials complete, ~4h remaining
- **Best Value**: -1.11 (both trials failed constraints)
- **Process**: Running (PID 12954)

## Key Findings from Manual Tuning

### Baseline (Regime Routed Config)
```
Overall: 1,268 trades, PF 0.92, -$1,312
├─ Order Block:        9 trades (0.7%), PF 43.45, +$1,041 ✅
├─ Trap:             934 trades (73.7%), PF 0.86, -$1,318 ❌
└─ Volume Exhaustion: 322 trades (25.4%), PF 0.87, -$894 ❌
```

### OB Expansion Attempt (Failed)
```
Overall: 1,268 trades, PF 0.92, -$1,432 (WORSE)
├─ Order Block:       14 trades (1.1%), PF 10.83 ⚠️, +$1,338
├─ Trap:             932 trades (73.5%), PF 0.86, -$1,350
└─ Volume Exhaustion: 319 trades (25.2%), PF 0.81 ⚠️, -$1,277
```

**Result**: Adding 5 OB trades (55% increase) degraded OB quality 4x (PF 43→11) and mysteriously worsened VE (PF 0.87→0.81) despite no VE config changes.

## The Core Challenge

**Order Block Dilemma**:
- 9 trades @ PF 43.45 = exceptional edge
- Lowering gates/cooldown = more trades but quality degradation
- Need to find sweet spot: 20-50 OB trades while maintaining PF > 5.0

**Trap/VE Problem**:
- Both bleed at scale (PF 0.86-0.87)
- Need: Either suppress aggressively OR improve exit timing
- Current exit: Both use macro_echo which neutralizes on fusion drop
- Hypothesis: Maybe VE needs tighter trailing stops, Trap needs suppression

## Optimization Constraints

```python
HARD CONSTRAINTS:
- Each archetype PF ≥ 1.0 (min 20 trades)  # No bleeders
- Overall PF ≥ 1.2                         # Profitable system
- Max DD ≤ 25%                             # Risk control
- Trades: 50-2000 per window               # Sample size
```

## Search Space

```python
# TRAP (suppress or improve)
trap_final_fusion_gate: [0.38, 0.50]   # Higher = fewer trades
trap_archetype_weight:  [0.5, 1.0]     # Lower = less priority
trap_cooldown_bars:     [8, 15]        # Higher = fewer trades

# VOLUME EXHAUSTION (fix bleeding)
ve_final_fusion_gate:   [0.30, 0.40]
ve_archetype_weight:    [1.0, 1.5]
ve_cooldown_bars:       [6, 12]
ve_trail_atr_mult:      [0.8, 1.5]     # Exit tuning
ve_max_bars:            [40, 80]       # Exit tuning

# ORDER BLOCK (expand safely)
ob_final_fusion_gate:   [0.28, 0.38]   # Lower = more trades
ob_archetype_weight:    [1.2, 1.6]     # Higher = more priority
ob_cooldown_bars:       [4, 10]        # Lower = more trades

# GLOBAL
max_trades_per_day:     [6, 12]
```

## Expected Outcomes (Ranked by Likelihood)

### Scenario 1: OB Dominance (60% probability)
- Optuna finds: OB weight 1.5+, gate 0.30, cooldown 5-6
- Result: 25-40 OB trades, PF 5-15, carries portfolio
- Trap/VE suppressed via higher gates/cooldowns
- **Action**: Deploy OB-dominant config

### Scenario 2: Multi-Archetype Balance (25% probability)
- VE exit tuning works (trail_atr 1.2+, max_bars 60+)
- Trap suppressed (gate 0.45+)
- Result: OB 20 trades, VE 80 trades (both PF > 1.0)
- **Action**: Deploy balanced config

### Scenario 3: Constraint Failure (15% probability)
- No parameter combination satisfies all constraints
- Best trial: PF 1.0-1.15 (below 1.2 threshold)
- **Action**:
  - Option A: Relax constraint to PF ≥ 1.1
  - Option B: Hard pivot to OB-only strategy
  - Option C: Add bear archetypes (breakdown below support)

## Next Steps After Optuna Completes

1. **Analyze best trial**:
   ```bash
   python3 bin/analyze_archetype_perf.py results/optuna_step5_full/validation_backtest.txt
   ```

2. **If successful (PF ≥ 1.5)**:
   - Deploy best config
   - Run 2025 YTD validation
   - Proceed to ML filter retraining (Step 4)

3. **If marginal (PF 1.2-1.4)**:
   - Proceed to PyTorch meta-learning (Stage 4A)
   - Implement context-aware dynamic parameters

4. **If constraint failure**:
   - Analyze top 10 trials to identify bottleneck
   - Consider relaxing constraints or pivoting strategy

## Technical Notes

- **TPE Sampler**: Needs ~10-20 trials to learn parameter relationships
- **MedianPruner**: Early stops bad trials after 2 windows
- **Objective**: `median_pf + 0.2*min_pf` (robust + consistent)
- **Validation Windows**: 2022 (bear), 2023 (recovery), 2024 (bull), full

## Files
- Script: `bin/optuna_archetype_portfolio_v1.py`
- Config: `configs/baseline_btc_bull_regime_routed_v1.json`
- Output: `results/optuna_step5_full/`
- Spec: `STEP5_OPTUNA_SPEC.md`
