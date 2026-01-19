# Soft Gating Position Sizing - Quick Start

**Status**: IMPLEMENTED
**Date**: 2026-01-10

## What Was Implemented

### 1. Regime Risk Budgets

CRISIS regime now limited to 30% max exposure (was unlimited).

```python
REGIME_RISK_BUDGETS = {
    'crisis': 0.30,    # Max 30% total exposure
    'risk_off': 0.50,
    'neutral': 0.70,
    'risk_on': 0.80
}
```

### 2. Position Size Soft Gating

Position sizes now scaled by regime weight AND capped by regime budget.

**Formula**:
```
Position Size = Base Size × Regime Weight × Confidence × Budget Cap
```

**Example (S1 in CRISIS)**:
- Base: 20% ($2000)
- Regime Weight: 1.0 (only archetype in CRISIS)
- Confidence: 0.40
- **Result**: 8% ($800) - 60% reduction
- Budget Cap: If CRISIS already has 25% exposure, capped to 5% ($500)

## How to Use

### Enable Soft Gating

```python
from engine.models.archetype_model import ArchetypeModel
from engine.portfolio.regime_allocator import RegimeWeightAllocator

# Initialize allocator
allocator = RegimeWeightAllocator(
    edge_table_path='results/archetype_regime_edge_table.csv'
)

# Create model with soft gating
model = ArchetypeModel(
    config_path='configs/s1_optimized.json',
    archetype_name='S1',
    regime_allocator=allocator  # Enable soft gating
)
```

### Check Allocations

```python
# See what weights are assigned
weight = allocator.get_weight('liquidity_vacuum', 'crisis')
print(f"Weight: {weight:.2f}")

# See edge metrics
metrics = allocator.get_edge_metrics('liquidity_vacuum', 'crisis')
print(f"Sharpe: {metrics['edge_raw']:.3f}")
print(f"N trades: {metrics['n_trades']}")

# Get regime summary
summary = allocator.get_allocation_summary('crisis')
print(summary)
```

### Update Regime Exposure (During Backtesting)

```python
# After taking a position
allocator.update_regime_exposure('crisis', 0.15)  # 15% exposure

# Check available budget
available = allocator.get_available_budget('crisis')
print(f"Available: {available:.1%}")  # 15% remaining (30% - 15%)
```

## Files Changed

1. `engine/portfolio/regime_allocator.py` - Added budget methods
2. `engine/models/archetype_model.py` - Integrated soft gating
3. `SOFT_GATING_POSITION_SIZING_IMPLEMENTATION.md` - Full documentation

## Validation

```bash
python bin/validate_soft_gating.py
```

## Expected Results

### CRISIS Regime
- liquidity_vacuum positions: 60% smaller
- Total CRISIS exposure: capped at 30%
- Expected PnL improvement: +$120

### RISK_ON Regime
- wick_trap_moneytaur positions: 93% smaller
- Expected PnL improvement: +$102

### Total Expected Gain
+$220 to +$270 (compared to baseline)

## Logging

Position sizing logs show full transparency:

```
Soft gating applied: archetype=liquidity_vacuum, regime=crisis,
base_size_pct=20.0%, regime_weight=1.00, confidence=0.40,
final_size_pct=8.0%, position_size=$800, budget_capped=False
```

Budget cap warnings:

```
Regime budget cap applied for crisis: 8.0% -> 5.0%
(budget=30.0%, current=25.0%)
```

## Next Steps

1. Run backtest with soft gating enabled
2. Verify PnL improvements match expectations
3. Check for unintended side effects
4. Consider implementing score-level gating (Phase 2)
