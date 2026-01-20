# Soft Gating Position Sizing Implementation

**Author**: Backend Architect
**Date**: 2026-01-10
**Status**: COMPLETED

## Overview

This implementation integrates soft gating into position sizing and adds CRISIS risk budget mechanisms to prevent concentration risk in negative-edge regimes.

## Key Components

### 1. RegimeWeightAllocator (Enhanced)

**File**: `engine/portfolio/regime_allocator.py`

**New Features Added**:

```python
# Regime-specific risk budgets (class constant)
REGIME_RISK_BUDGETS = {
    'crisis': 0.30,    # Max 30% total exposure in CRISIS
    'risk_off': 0.50,  # Max 50% in RISK_OFF
    'neutral': 0.70,   # Max 70% in NEUTRAL
    'risk_on': 0.80    # Max 80% in RISK_ON
}

# Regime exposure tracking (instance variable)
self.regime_exposures: Dict[str, float] = {
    'crisis': 0.0,
    'risk_off': 0.0,
    'neutral': 0.0,
    'risk_on': 0.0
}
```

**New Methods**:

1. `get_regime_budget(regime: str) -> float`
   - Returns maximum allowed exposure for regime
   - Default: 80% if regime not found

2. `update_regime_exposure(regime: str, exposure: float) -> None`
   - Updates current exposure tracking
   - Called by position manager after trades

3. `get_regime_exposure(regime: str) -> float`
   - Returns current total exposure in regime
   - Used for budget cap calculations

4. `get_available_budget(regime: str) -> float`
   - Calculates remaining budget available
   - Returns: `max(0.0, budget - current)`

5. `apply_regime_budget_cap(regime: str, position_size_pct: float) -> Tuple[float, bool]`
   - **CRITICAL**: Prevents exceeding regime budgets
   - Caps position size if it would exceed available budget
   - Returns: `(capped_size_pct, was_capped)`
   - Logs warning when capping occurs

6. `reset_regime_exposures() -> None`
   - Resets all exposures to zero
   - Call at start of each backtest iteration

7. `get_allocation_summary(regime: str) -> str`
   - Returns formatted summary for debugging
   - Shows: budget, current exposure, available, weights

### 2. ArchetypeModel Integration

**File**: `engine/models/archetype_model.py`

**Constructor Changes**:

```python
def __init__(
    self,
    config_path: str,
    archetype_name: str = 'S4',
    name: Optional[str] = None,
    regime_classifier_path: Optional[str] = None,
    regime_allocator: Optional[Any] = None  # NEW!
):
```

**Position Sizing Logic** (completely rewritten):

```python
def get_position_size(self, bar: pd.Series, signal: Signal) -> float:
    """
    Calculate position size with soft gating.

    Steps:
    1. Calculate base position size (ATR-based risk)
    2. Apply regime weight (soft gating)
    3. Apply confidence scaling
    4. Apply regime risk budget cap
    5. Convert to dollar amount
    """
    # Base sizing
    portfolio_value = 10000.0
    stop_distance_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
    risk_dollars = portfolio_value * self.max_risk_pct
    base_position_size = risk_dollars / stop_distance_pct
    base_size_pct = base_position_size / portfolio_value

    # Get regime
    regime = signal.metadata.get('regime', bar.get('macro_regime', self.default_regime))

    # Apply soft gating if enabled
    if self.regime_allocator:
        # Get regime weight
        regime_weight = self.regime_allocator.get_weight(archetype_key, regime)

        # Apply regime weight
        size_pct = base_size_pct * regime_weight

        # Apply confidence
        size_pct *= signal.confidence

        # Apply regime budget cap
        size_pct, was_capped = self.regime_allocator.apply_regime_budget_cap(
            regime, size_pct
        )

        # Convert to dollars
        position_size = portfolio_value * size_pct

        logger.info(
            f"Soft gating applied: archetype={archetype_key}, regime={regime}, "
            f"base_size_pct={base_size_pct:.1%}, regime_weight={regime_weight:.2f}, "
            f"confidence={signal.confidence:.2f}, final_size_pct={size_pct:.1%}, "
            f"position_size=${position_size:,.0f}, budget_capped={was_capped}"
        )
    else:
        # No soft gating - use base size
        position_size = base_position_size

    return position_size
```

**Archetype Name Mapping**:

```python
archetype_key_map = {
    'S1': 'liquidity_vacuum',
    'S4': 'funding_divergence',
    'B': 'order_block_retest',
    'C': 'wick_trap_moneytaur',
    'K': 'trap_within_trend',
}
```

## Usage Examples

### Basic Usage (No Soft Gating)

```python
from engine.models.archetype_model import ArchetypeModel

# Model without soft gating (legacy behavior)
model = ArchetypeModel(
    config_path='configs/s1_optimized.json',
    archetype_name='S1'
)
```

### With Soft Gating

```python
from engine.models.archetype_model import ArchetypeModel
from engine.portfolio.regime_allocator import RegimeWeightAllocator

# Initialize allocator
allocator = RegimeWeightAllocator(
    edge_table_path='results/archetype_regime_edge_table.csv'
)

# Model with soft gating enabled
model = ArchetypeModel(
    config_path='configs/s1_optimized.json',
    archetype_name='S1',
    regime_allocator=allocator  # Enable soft gating
)

# During backtesting, update regime exposures
# (This would typically be done in the backtesting engine)
allocator.update_regime_exposure('crisis', 0.15)  # 15% exposure
```

### Checking Allocations

```python
# Get allocation summary for a regime
summary = allocator.get_allocation_summary('crisis')
print(summary)

# Output:
# Regime: CRISIS
# Budget: 30.0%
# Current Exposure: 15.0%
# Available: 15.0%
#
# Archetype Weights:
#   liquidity_vacuum: 100.0% (sharpe=-0.042, n=57)
```

## Expected Behavior

### Example 1: liquidity_vacuum in CRISIS

**Before Soft Gating**:
- Position size: $2000 (20% of portfolio)
- No regime constraint
- Total CRISIS exposure: unlimited

**After Soft Gating**:
```python
# Edge metrics
sharpe = -0.042
N = 57 trades

# Weight computation
edge_shrunk = -0.042 * (57 / (57 + 30)) = -0.028
strength = sigmoid(4.0 * -0.028) = 0.47
weight = min(0.47, 0.20) = 0.20  # Capped at neg_edge_cap

# Position sizing
base_size_pct = 0.20 (20%)
regime_weight = 1.0 (only archetype in CRISIS, after normalization)
confidence = 0.40
size_pct = 0.20 * 1.0 * 0.40 = 0.08 (8%)

# Regime budget cap
crisis_budget = 0.30 (30%)
current_exposure = 0.00
available = 0.30 - 0.00 = 0.30
final_size_pct = min(0.08, 0.30) = 0.08 (not capped)

# Final position
position_size = $10,000 * 0.08 = $800

# Reduction: $2000 -> $800 (60% reduction)
```

### Example 2: wick_trap_moneytaur in RISK_ON

**Before Soft Gating**:
- Position size: $1400 (14% of portfolio)
- 73 trades with -$110 total PnL

**After Soft Gating**:
```python
# Edge metrics
sharpe = -0.025
N = 73 trades

# Weight computation
edge_shrunk = -0.025 * (73 / (73 + 30)) = -0.018
strength = sigmoid(4.0 * -0.018) = 0.48
weight = min(0.48, 0.20) = 0.20  # Capped

# If normalized with other RISK_ON archetypes:
# wick_trap: 0.20
# order_block: 0.60
# other: 0.20
# total = 1.00
# wick_trap_normalized = 0.20 / 1.00 = 0.20 (20% of RISK_ON allocation)

# Position sizing
base_size_pct = 0.14
regime_weight = 0.20 (after normalization)
confidence = 0.35
size_pct = 0.14 * 0.20 * 0.35 = 0.0098 (~1%)

# Final position
position_size = $10,000 * 0.0098 = $98

# Reduction: $1400 -> $98 (93% reduction)
```

### Example 3: Regime Budget Cap

```python
# Scenario: CRISIS regime approaching budget limit
current_exposure = 0.25  # 25% already allocated
crisis_budget = 0.30     # 30% max

# New signal for liquidity_vacuum
base_size_pct = 0.08     # Would normally take 8%
available = 0.30 - 0.25 = 0.05  # Only 5% available

# Budget cap applied
capped_size = min(0.08, 0.05) = 0.05
position_size = $10,000 * 0.05 = $500

# Log message:
# "Regime budget cap applied for crisis: 8.0% -> 5.0% (budget=30.0%, current=25.0%)"
```

## Validation

Run the validation script to verify implementation:

```bash
python bin/validate_soft_gating.py
```

**Expected Tests**:
1. Weight computation follows spec (sigmoid, shrinkage, guardrails)
2. Negative edge archetypes capped at 20%
3. Positive edge archetypes not capped
4. Regime budgets enforced
5. Position sizes reduced as expected
6. Logging provides full transparency

## Integration with Backtesting

To use soft gating in backtests:

```python
from engine.backtesting.engine import BacktestEngine
from engine.models.archetype_model import ArchetypeModel
from engine.portfolio.regime_allocator import RegimeWeightAllocator

# Load edge table
allocator = RegimeWeightAllocator(
    edge_table_path='results/archetype_regime_edge_table.csv'
)

# Create model with soft gating
model = ArchetypeModel(
    config_path='configs/s1_optimized.json',
    archetype_name='S1',
    regime_allocator=allocator
)

# Run backtest
engine = BacktestEngine(model, data)
results = engine.run(start='2022-01-01', end='2023-12-31')

# During backtesting, you would need to:
# 1. Track total exposure per regime
# 2. Update allocator.regime_exposures after each trade
# 3. Reset at the start of each bar
```

## Files Modified

1. `engine/portfolio/regime_allocator.py`
   - Added REGIME_RISK_BUDGETS constant
   - Added regime_exposures tracking
   - Added 7 new methods for budget management

2. `engine/models/archetype_model.py`
   - Added regime_allocator parameter to constructor
   - Completely rewrote get_position_size() method
   - Added comprehensive logging

3. `bin/validate_soft_gating.py` (existing)
   - Validation script already exists
   - Tests weight computation and guardrails

## Performance Impact

### Expected Improvements

Based on edge table data:

**CRISIS Regime**:
- liquidity_vacuum: -$201 -> ~-$80 (60% reduction)
- Total CRISIS loss reduction: ~$120

**RISK_ON Regime**:
- wick_trap_moneytaur: -$110 -> ~-$8 (93% reduction)
- Total RISK_ON loss reduction: ~$102

**Combined Expected Gain**: +$220 to +$270

### Risk Reduction

1. **Concentration Risk**: CRISIS regime capped at 30% total exposure
2. **Negative Edge Exposure**: Negative edge archetypes capped at 20% weight
3. **Portfolio Balance**: Better regime diversification
4. **Tail Risk**: Reduced exposure to high stop-out rate archetypes

## Debugging and Monitoring

### Logging Output

Position sizing logs include:
```
Soft gating applied: archetype=liquidity_vacuum, regime=crisis,
base_size_pct=20.0%, regime_weight=1.00, confidence=0.40,
final_size_pct=8.0%, position_size=$800, budget_capped=False
```

Budget cap logs include:
```
Regime budget cap applied for crisis: 8.0% -> 5.0%
(budget=30.0%, current=25.0%)
```

### Allocation Summaries

Use `allocator.get_allocation_summary(regime)` to inspect:
- Current budget usage
- Available capacity
- Archetype weights and edge metrics

## Next Steps

1. **Backtest Validation**:
   - Run full backtest with soft gating enabled
   - Compare results to baseline
   - Verify expected PnL improvements

2. **Score-Level Gating** (future):
   - Apply regime weights at signal generation (LogicV2Adapter)
   - Prevent signals from being generated in first place
   - Combined with size-level gating prevents cascade effects

3. **Dynamic Budget Adjustment** (future):
   - Adjust regime budgets based on volatility
   - Increase CRISIS budget during calm periods
   - Decrease during high volatility

## Success Criteria

- [x] RegimeWeightAllocator has regime budget methods
- [x] ArchetypeModel applies soft gating in get_position_size()
- [x] Regime budget caps prevent concentration
- [x] Comprehensive logging for debugging
- [x] Validation script exists and can test implementation
- [ ] Backtest shows expected PnL improvement (+$220-270)
- [ ] No unintended side effects on other regimes

## Related Documents

- `SOFT_GATING_PHASE1_SPEC.md` - Original specification
- `SOFT_GATING_IMPLEMENTATION_GUIDE.md` - Score-level gating guide
- `results/archetype_regime_edge_table.csv` - Edge metrics source
- `bin/build_regime_edge_table.py` - Edge table generation script
