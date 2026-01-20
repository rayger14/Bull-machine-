# Soft Gating Implementation Guide

**Architectural Pattern**: Regime-Conditioned Portfolio Allocation (Mixture of Experts)

**Goal**: Replace binary on/off switches with continuous capital allocation weights

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     REGIME DETECTION                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ regime_label = "risk_on" | "neutral" | "crisis" | ...    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  ARCHETYPE SIGNAL GENERATION                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ For each archetype:                                       │  │
│  │   raw_score = archetype.detect(row, regime_label)        │  │
│  │   # Returns confidence ∈ [0, 1]                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              REGIME-CONDITIONED SOFT GATING (NEW!)               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ weight = get_regime_weight(archetype, regime_label)      │  │
│  │ # weight ∈ [0, 1] based on empirical edge                │  │
│  │                                                            │  │
│  │ gated_score = raw_score × weight                          │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  ENSEMBLE / FUSION LOGIC                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ All gated scores compete for capital                     │  │
│  │ Winner(s) get positions sized by:                        │  │
│  │   position_size = base_size × weight × confidence        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Components

### 1. RegimeWeightAllocator Class

**File**: `engine/portfolio/regime_allocator.py` (NEW)

```python
import json
import numpy as np
from typing import Dict, Tuple

class RegimeWeightAllocator:
    """
    Manages regime-conditioned weights for soft gating.

    This replaces binary on/off switches with continuous allocation weights
    based on empirical edge metrics from historical performance.
    """

    def __init__(
        self,
        weights_path: str = 'configs/regime_weights.json',
        threshold_sharpe: float = -0.05,
        min_trades: int = 10,
        k_shrinkage: int = 30,
        penalty_negative: float = 0.1,
        beta: float = 2.0,
        min_weight: float = 0.02,
        max_weight_negative: float = 0.20
    ):
        """
        Args:
            weights_path: Path to precomputed regime weights JSON
            threshold_sharpe: Don't allocate below this Sharpe ratio
            min_trades: Minimum sample size for reliable estimates
            k_shrinkage: Sample size shrinkage constant
            penalty_negative: Multiplier for negative-edge archetypes
            beta: Softmax temperature (higher = more winner-takes-all)
            min_weight: Floor for positive-edge archetypes
            max_weight_negative: Cap for negative-edge archetypes
        """
        self.threshold_sharpe = threshold_sharpe
        self.min_trades = min_trades
        self.k_shrinkage = k_shrinkage
        self.penalty_negative = penalty_negative
        self.beta = beta
        self.min_weight = min_weight
        self.max_weight_negative = max_weight_negative

        # Load precomputed weights
        with open(weights_path, 'r') as f:
            self.weights = json.load(f)

    def get_weight(self, archetype: str, regime: str) -> float:
        """
        Get allocation weight for (archetype, regime) pair.

        Returns:
            weight ∈ [0, 1] indicating capital allocation
        """
        key = f"{archetype}|{regime}"

        # Default to min_weight if not in table
        weight = self.weights.get(key, self.min_weight)

        # Apply quality controls
        weight = self._apply_quality_gates(weight, archetype, regime)

        return weight

    def _apply_quality_gates(self, weight: float, archetype: str, regime: str) -> float:
        """
        Apply quality thresholds and constraints.
        """
        # TODO: Load edge metrics (Sharpe, N trades) from edge table
        # For now, trust precomputed weights

        # Example quality gates (would use real edge data):
        # if sharpe < self.threshold_sharpe:
        #     return 0.0
        # if n_trades < self.min_trades:
        #     weight *= (n_trades / (n_trades + self.k_shrinkage))

        return weight

    def get_regime_distribution(self, regime: str) -> Dict[str, float]:
        """
        Get full weight distribution for a regime.

        Returns:
            Dict of {archetype: weight} for all archetypes in regime
        """
        regime_weights = {
            arch: w
            for key, w in self.weights.items()
            if key.endswith(f"|{regime}")
            for arch in [key.split("|")[0]]
        }

        # Normalize to sum to 1.0
        total = sum(regime_weights.values())
        if total > 0:
            regime_weights = {k: v/total for k, v in regime_weights.items()}

        return regime_weights
```

### 2. Integration with Archetype Scoring

**File**: `engine/archetypes/logic_v2_adapter.py` (MODIFY)

```python
from engine.portfolio.regime_allocator import RegimeWeightAllocator

class LogicV2Adapter:
    def __init__(self, ...):
        # ... existing init ...

        # NEW: Add regime weight allocator
        self.regime_allocator = RegimeWeightAllocator(
            weights_path='configs/regime_weights.json'
        )

    def _check_order_block_retest(self, context: ArchetypeContext) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Example archetype with soft gating applied.
        """
        regime_label = self.g(context.row, "regime_label", "neutral")

        # Step 1: Compute raw score (existing logic)
        raw_score = self._compute_order_block_score(context)

        if raw_score < self.min_fusion_score:
            return False, 0.0, {'veto_reason': 'low_fusion_score'}

        # Step 2: Apply regime-conditioned weight (NEW!)
        regime_weight = self.regime_allocator.get_weight('order_block_retest', regime_label)

        # Step 3: Gate the score
        gated_score = raw_score * regime_weight

        # Step 4: Check if gated score still meets minimum
        if gated_score < self.min_fusion_score:
            return False, 0.0, {
                'veto_reason': 'soft_gating_regime_penalty',
                'regime': regime_label,
                'raw_score': raw_score,
                'regime_weight': regime_weight,
                'gated_score': gated_score,
                'explanation': f'{regime_label} regime has low allocation weight ({regime_weight:.2f}) for this archetype'
            }

        return True, gated_score, {
            'regime_weight': regime_weight,
            'raw_score': raw_score,
            'gated_score': gated_score
        }
```

### 3. Position Sizing with Weights

**File**: `engine/position_manager.py` (MODIFY)

```python
def calculate_position_size(
    self,
    archetype: str,
    regime: str,
    confidence: float,
    base_size_pct: float = 0.20
) -> float:
    """
    Calculate position size with regime-conditioned weight applied.

    position_size = base_size × regime_weight × confidence × other_scalers
    """
    # Get regime weight
    regime_weight = self.regime_allocator.get_weight(archetype, regime)

    # Base sizing
    position_size_pct = base_size_pct

    # Apply regime weight
    position_size_pct *= regime_weight

    # Apply confidence scaling
    position_size_pct *= confidence

    # Apply direction balance scaling (existing)
    direction_scaler = self.direction_balance.get_scaler(...)
    position_size_pct *= direction_scaler

    # Apply risk budget constraints (existing)
    risk_scaler = self.risk_budget.get_scaler(...)
    position_size_pct *= risk_scaler

    # Convert to dollar amount
    position_size = self.portfolio_value * position_size_pct

    return position_size
```

---

## Example: How Soft Gating Solves Your Issues

### Issue #1: wick_trap_moneytaur RISK_ON (-$110.61)

**Current Behavior (Binary Disable)**:
```python
if regime_label == 'risk_on':
    return None, 0.0  # Hard veto
```

**New Behavior (Soft Gating)**:
```python
raw_score = 0.35  # Archetype detects signal
regime_weight = 0.0  # Zero weight due to negative Sharpe < -0.05
gated_score = 0.35 × 0.0 = 0.0  # Effectively disabled
position_size = base × 0.0 × confidence = $0  # No position
```

**Result**: Same outcome (no trades in RISK_ON), but via allocation logic, not hard veto

**If we improve parameters later** and Sharpe becomes positive:
```python
raw_score = 0.40
regime_weight = 0.15  # Small allocation (improved but still not great)
gated_score = 0.40 × 0.15 = 0.06  # Low score, unlikely to win capital
position_size = base × 0.15 × confidence = $300  # Small position
```

**Result**: Archetype can gradually earn back allocation as it proves itself

---

### Issue #2: liquidity_vacuum CRISIS (-$201.03)

**Current Behavior**:
```python
# Only archetype in CRISIS → 100% of capital
position_size = $10,000 × 0.20 = $2,000
```

**New Behavior (Soft Gating with Negative Edge Cap)**:
```python
raw_score = 0.38
regime_weight = 0.20  # Capped due to negative Sharpe (was 1.0)
gated_score = 0.38 × 0.20 = 0.076
position_size = $10,000 × 0.20 × 0.20 × confidence = $400  # 80% smaller
```

**Result**: Same trades, 80% smaller positions, -$201 → -$40 (80% reduction)

**If we add other archetypes to CRISIS**:
```python
# 3 archetypes now compete:
liquidity_vacuum: weight 0.20 (negative edge, capped)
order_block_crisis: weight 0.50 (new variant, positive edge)
funding_divergence_crisis: weight 0.30 (extreme conditions)

# Capital distributed across all three
# liquidity_vacuum gets 20% of 30% total CRISIS allocation = 6% of portfolio
```

**Result**: Diversification reduces concentration risk

---

## Testing Strategy

### Phase 1: Shadow Mode (No Position Changes)

```python
# In scoring logic, compute both:
old_score = raw_score  # Current behavior
new_score = raw_score × regime_weight  # New behavior

# Log both for comparison
logger.info(f"Soft gating: raw={old_score:.3f}, weight={regime_weight:.3f}, gated={new_score:.3f}")

# Still use old_score for actual trading
return old_score
```

**Benefit**: See what would change without risking capital

### Phase 2: Backtesting Validation

```bash
python3 bin/backtest_full_engine_replay.py --use-soft-gating
```

**Expected Results**:
- wick_trap RISK_ON: 73 trades → 0 trades
- liquidity_vacuum CRISIS: position sizes reduced 80%
- Total PnL: +$139 → +$400-450

### Phase 3: Paper Trading (1-2 Weeks)

Deploy with soft gating enabled, monitor:
- Do weights make sense in live conditions?
- Are scores being gated as expected?
- Does performance match backtest expectations?

### Phase 4: Production Deployment

Once validated in paper trading.

---

## Configuration Files

### Regime Weights JSON

**File**: `configs/regime_weights.json`

```json
{
  "order_block_retest|neutral": 0.261,
  "order_block_retest|risk_on": 0.0,
  "order_block_retest|crisis": 0.50,

  "wick_trap_moneytaur|neutral": 0.258,
  "wick_trap_moneytaur|risk_on": 0.0,

  "liquidity_vacuum|crisis": 0.20,
  "liquidity_vacuum|risk_off": 0.487,

  "funding_divergence|neutral": 0.258,
  "funding_divergence|risk_off": 0.513,
  "funding_divergence|crisis": 0.30
}
```

### Allocator Config

**File**: `configs/regime_allocator_config.json`

```json
{
  "threshold_sharpe": -0.05,
  "min_trades": 10,
  "k_shrinkage": 30,
  "penalty_negative": 0.1,
  "beta": 2.0,
  "min_weight": 0.02,
  "max_weight_negative": 0.20,

  "regime_risk_budgets": {
    "crisis": 0.30,
    "risk_off": 0.50,
    "neutral": 0.70,
    "risk_on": 0.80
  }
}
```

---

## Implementation Checklist

### Step 1: Create RegimeWeightAllocator (30 min)
- [ ] Create `engine/portfolio/regime_allocator.py`
- [ ] Implement get_weight() method
- [ ] Add quality thresholds
- [ ] Unit test with mock weights

### Step 2: Integrate with Archetype Scoring (1-2 hours)
- [ ] Add regime_allocator to LogicV2Adapter.__init__()
- [ ] Modify all _check_*() methods to apply gating
- [ ] Add logging for raw vs gated scores
- [ ] Preserve existing veto logic

### Step 3: Update Position Sizing (1 hour)
- [ ] Modify position_manager.py
- [ ] Apply regime_weight to position calculations
- [ ] Test with sample scenarios

### Step 4: Generate Regime Weights (30 min)
- [ ] Run build_regime_edge_table.py
- [ ] Review weights for sanity
- [ ] Copy to configs/regime_weights.json
- [ ] Version control

### Step 5: Backtesting Validation (2-3 hours)
- [ ] Add --use-soft-gating flag to backtest script
- [ ] Run comparison backtest
- [ ] Analyze trade-by-trade differences
- [ ] Validate expected PnL improvement

### Step 6: Shadow Mode Deployment (1 week)
- [ ] Deploy to paper trading with shadow logging
- [ ] Monitor soft gating decisions
- [ ] Compare against hard vetoes
- [ ] Collect performance data

### Step 7: Production Rollout (After validation)
- [ ] Enable soft gating in production
- [ ] Monitor first 48 hours closely
- [ ] Track regime weight effectiveness
- [ ] Iterate on weights if needed

---

## Next: Regime-Specific Parameters

Once soft gating is working, create parameter variants:

```
configs/archetypes/
  wick_trap_moneytaur/
    base.json           # Default parameters
    neutral.json        # Optimized for ranging markets
    risk_on.json        # Exhaustion-only (strict filters)

  liquidity_vacuum/
    base.json
    crisis.json         # Improved logic for volatile regime

  order_block_retest/
    base.json
    neutral.json
    crisis.json         # Stricter entry, tighter SL
```

**Loading Logic**:
```python
def load_archetype_params(archetype: str, regime: str) -> Dict:
    # Try regime-specific first
    path = f"configs/archetypes/{archetype}/{regime}.json"
    if Path(path).exists():
        return load_json(path)

    # Fall back to base
    return load_json(f"configs/archetypes/{archetype}/base.json")
```

---

## Summary

**Soft gating replaces:**
```python
if regime == 'risk_on': return 0  # Binary
```

**With:**
```python
weight = empirical_edge[archetype][regime]  # Continuous
gated_score = raw_score × weight
position_size = base × weight × confidence
```

**Benefits:**
1. Preserves architectural flexibility
2. Graceful degradation (not hard cutoff)
3. Can adapt as archetypes improve
4. Quant-correct portfolio allocation
5. No manual "disable" decisions

**Solves your issues:**
- wick_trap RISK_ON: weight = 0.0 (negative edge)
- liquidity_vacuum CRISIS: weight = 0.20 (capped)
- Same outcomes as "disable" but via allocation logic

**Ready to implement?**
