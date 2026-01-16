# Cash Bucket Calculation: Step-by-Step Example

**Scenario:** CRISIS regime with liquidity_vacuum archetype
**Date:** 2026-01-10

---

## Input Data (from edge table)

```
Archetype:        liquidity_vacuum
Regime:           crisis
Sharpe-like:      -0.042 (NEGATIVE EDGE)
N trades:         57
Total PnL:        -201.03
Expectancy:       -3.53
Profit Factor:    0.91 (losing strategy)
```

---

## Step 1: Compute Raw Archetype Weight

### Configuration
```python
k_shrinkage = 30      # Sample size shrinkage parameter
alpha = 4.0           # Sigmoid steepness
neg_edge_cap = 0.20   # Max weight for negative edge
min_weight = 0.01     # Floor for exploration
```

### Calculation

#### 1.1 Shrink Edge by Sample Size (Empirical Bayes)
```python
edge_shrunk = edge * (N / (N + k_shrinkage))
            = -0.042 * (57 / (57 + 30))
            = -0.042 * (57 / 87)
            = -0.042 * 0.6552
            = -0.0275
```

**Interpretation:** Shrinkage reduces magnitude from -0.042 to -0.0275 due to moderate sample size (57 trades)

#### 1.2 Map to Strength using Sigmoid
```python
strength = 1 / (1 + exp(-alpha * edge_shrunk))
        = 1 / (1 + exp(-4.0 * -0.0275))
        = 1 / (1 + exp(0.110))
        = 1 / (1 + 1.116)
        = 1 / 2.116
        = 0.4726
```

**Interpretation:** Sigmoid maps negative edge to below-neutral strength (0.47 < 0.50)

#### 1.3 Apply Guardrails
```python
# Guardrail A: Cap negative edge
if edge_shrunk < 0:  # True (-0.0275 < 0)
    weight = min(strength, neg_edge_cap)
          = min(0.4726, 0.20)
          = 0.20  ← CAPPED

# Guardrail B: Floor for exploration
if N >= min_trades:  # True (57 >= 5)
    weight = max(weight, min_weight)
          = max(0.20, 0.01)
          = 0.20  ← No change (already above floor)
```

**Final Raw Weight:** `0.20` (20%)

**Interpretation:** Negative edge caps weight at 20% maximum, preventing over-allocation to losing strategy

---

## Step 2: Calculate Cash Bucket

### Sum All Archetype Weights
```python
weights = {'liquidity_vacuum': 0.20}
total_weight = sum(weights.values()) = 0.20
```

### Compute Cash Bucket
```python
cash_bucket = max(0.0, 1.0 - total_weight)
            = max(0.0, 1.0 - 0.20)
            = 0.80
```

**Cash Bucket:** `0.80` (80%)

**Interpretation:** Only 20% of regime budget deserves allocation. 80% stays in cash due to weak edge.

---

## Step 3: Effective Allocation

### Build Allocation Dictionary
```python
effective_allocation = {
    'liquidity_vacuum': 0.20,
    'CASH': 0.80
}

# Verify sum
total = sum(effective_allocation.values())
     = 0.20 + 0.80
     = 1.00  ✓
```

**Guaranteed to sum to 1.0**

---

## Step 4: Portfolio Allocation with Regime Budget

### Regime Budget
```python
regime = 'crisis'
regime_budget = REGIME_RISK_BUDGETS['crisis'] = 0.30  (30%)
```

**Interpretation:** CRISIS regime gets maximum 30% of total portfolio

### Calculate Portfolio Exposures

#### Archetype Exposure
```python
liquidity_vacuum_portfolio = archetype_weight * regime_budget
                           = 0.20 * 0.30
                           = 0.06  (6% of portfolio)
```

#### Cash Within Regime
```python
cash_regime = cash_bucket * regime_budget
            = 0.80 * 0.30
            = 0.24  (24% of portfolio)
```

#### Cash Outside Regime (Reserve)
```python
cash_reserve = 1.0 - regime_budget
             = 1.0 - 0.30
             = 0.70  (70% of portfolio)
```

#### Total Cash
```python
total_cash = cash_regime + cash_reserve
          = 0.24 + 0.70
          = 0.94  (94% of portfolio)
```

---

## Final Portfolio Breakdown

```
┌─────────────────────────────────────────────────────────────┐
│ PORTFOLIO ALLOCATION (100%)                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ CRISIS REGIME (30% budget)                                  │
│ ├── liquidity_vacuum:  6%  (negative edge, capped)         │
│ └── Cash (regime):    24%  (80% of regime unused)          │
│                                                             │
│ OTHER REGIMES (70% reserve)                                 │
│ └── Cash (reserve):   70%  (unallocated to CRISIS)         │
│                                                             │
│ TOTAL CASH: 94%                                             │
│ TOTAL AT RISK: 6%                                           │
└─────────────────────────────────────────────────────────────┘
```

### Summary Table

| Component | Weight | Budget | Portfolio % |
|-----------|--------|--------|-------------|
| liquidity_vacuum | 20% | 30% | **6%** |
| Cash (regime) | 80% | 30% | **24%** |
| Cash (reserve) | - | 70% | **70%** |
| **TOTAL** | **100%** | **100%** | **100%** |

---

## Comparison: Legacy vs. Cash Bucket

### Legacy Mode (BROKEN)
```python
# Force renormalization
weights = {'liquidity_vacuum': 0.20}
total = 0.20
normalized = {k: v / total for k, v in weights.items()}
           = {'liquidity_vacuum': 0.20 / 0.20}
           = {'liquidity_vacuum': 1.0}  ← FORCED TO 100%

# Portfolio allocation
liquidity_vacuum_portfolio = 1.0 * 0.30 = 0.30  (30%)
cash_reserve = 0.70  (70%)
total_cash = 0.70  (70%)

Result:
  - 30% in NEGATIVE-EDGE strategy ❌
  - 70% cash
```

### Cash Bucket Mode (CORRECT)
```python
# Preserve weights, remainder becomes cash
weights = {'liquidity_vacuum': 0.20}
total = 0.20

if total < 1.0:
    normalized = weights.copy()  # {'liquidity_vacuum': 0.20}
    cash_bucket = 1.0 - 0.20 = 0.80

# Portfolio allocation
liquidity_vacuum_portfolio = 0.20 * 0.30 = 0.06  (6%)
cash_regime = 0.80 * 0.30 = 0.24  (24%)
cash_reserve = 0.70  (70%)
total_cash = 0.24 + 0.70 = 0.94  (94%)

Result:
  - 6% in negative-edge strategy ✓
  - 94% cash ✓
```

### Impact Comparison

| Metric | Legacy | Cash Bucket | Improvement |
|--------|--------|-------------|-------------|
| Archetype exposure | 30% | 6% | **80% reduction** |
| Total cash | 70% | 94% | **+24%** |
| Risk in negative edge | 30% | 6% | **5x safer** |

---

## Mathematical Proof

### Property 1: Effective Allocation Sums to 1.0
```python
effective_allocation = archetype_weights + cash_bucket

# By definition:
cash_bucket = 1.0 - sum(archetype_weights)

# Therefore:
sum(effective_allocation) = sum(archetype_weights) + cash_bucket
                          = sum(archetype_weights) + (1.0 - sum(archetype_weights))
                          = 1.0  ✓
```

### Property 2: Portfolio Allocation Sums to 1.0
```python
portfolio_allocation = (archetype_weights * regime_budget) +
                       (cash_bucket * regime_budget) +
                       (1.0 - regime_budget)

# Simplify:
= (archetype_weights + cash_bucket) * regime_budget + (1.0 - regime_budget)

# Since archetype_weights + cash_bucket = 1.0:
= 1.0 * regime_budget + (1.0 - regime_budget)
= regime_budget + 1.0 - regime_budget
= 1.0  ✓
```

### Property 3: Cash Bucket ∈ [0.0, 1.0]
```python
# Case 1: Under-allocated
if sum(archetype_weights) < 1.0:
    cash_bucket = 1.0 - sum(archetype_weights) > 0  ✓

# Case 2: Exactly allocated
if sum(archetype_weights) == 1.0:
    cash_bucket = 1.0 - 1.0 = 0  ✓

# Case 3: Over-allocated (after normalization)
if sum(archetype_weights) > 1.0:
    # Normalize first
    normalized_weights = {k: v / sum(weights) for k, v in weights.items()}
    sum(normalized_weights) = 1.0
    cash_bucket = 1.0 - 1.0 = 0  ✓
```

---

## Edge Cases

### Case 1: No Archetypes
```python
weights = {}
total = 0
cash_bucket = 1.0 - 0 = 1.0  (100% cash)
effective_allocation = {'CASH': 1.0}
```

### Case 2: Single Archetype, Strong Edge
```python
weights = {'archetype_a': 0.95}  # Strong edge
total = 0.95
cash_bucket = 1.0 - 0.95 = 0.05  (5% cash)
effective_allocation = {'archetype_a': 0.95, 'CASH': 0.05}
```

### Case 3: Over-Allocated
```python
weights = {'archetype_a': 0.60, 'archetype_b': 0.55}
total = 1.15

# Normalize
normalized = {k: v / 1.15 for k, v in weights.items()}
         = {'archetype_a': 0.522, 'archetype_b': 0.478}
cash_bucket = 0.0  (no cash when over-allocated)
```

---

## Verification with Actual Data

### From Test Output
```
CRISIS Regime Test Results:

Raw Archetype Weights:
  liquidity_vacuum: 0.2000 (sharpe=-0.042, N=57)
  Total weight: 0.2000

Cash Bucket Calculation:
  Cash bucket: 0.8000 (80.0%)
  → Archetype allocation: 20.0%
  → Cash (unused): 80.0%

Effective Allocation:
  CASH: 0.8000 (80.0%)
  liquidity_vacuum: 0.2000 (20.0%)
  Total: 1.000000 ✓

Portfolio Allocation (30% regime budget):
  liquidity_vacuum: 0.2000 * 0.30 = 6.0% of portfolio
  CASH (regime): 0.8000 * 0.30 = 24.0% of portfolio
  CASH (reserve): 70.0% of portfolio
  TOTAL CASH: 94.0% of portfolio

✓ All calculations verified
```

---

## Code Implementation

### Get Cash Bucket Weight
```python
def get_cash_bucket_weight(self, regime: str) -> float:
    weights = self.get_regime_distribution(regime)
    total = sum(weights.values())
    cash_bucket = max(0.0, 1.0 - total)
    return cash_bucket

# Usage
allocator = RegimeWeightAllocator('edge_table.csv')
cash = allocator.get_cash_bucket_weight('crisis')
# Returns: 0.80
```

### Get Effective Allocation
```python
def get_effective_allocation(self, regime: str, include_cash: bool = True) -> Dict[str, float]:
    weights = self.get_regime_distribution(regime)
    cash_bucket = self.get_cash_bucket_weight(regime)

    allocation = weights.copy()
    if include_cash and cash_bucket > 0:
        allocation['CASH'] = cash_bucket

    return allocation

# Usage
allocation = allocator.get_effective_allocation('crisis')
# Returns: {'liquidity_vacuum': 0.20, 'CASH': 0.80}
```

---

## Summary

**Input:**
- liquidity_vacuum: sharpe=-0.042, N=57

**Computation:**
1. Raw weight = 0.20 (capped at neg_edge_cap)
2. Cash bucket = 1.0 - 0.20 = 0.80
3. Regime budget = 30%

**Output:**
- Portfolio exposure: 6% (archetype) + 94% (cash)
- Risk reduction: 80% vs. forced allocation
- Correct defensive posture for negative-edge regime

**Status:** ✅ Verified with test suite

---

**Test Command:**
```bash
python3 bin/test_cash_bucket.py
```

**Expected:** `🎉 ALL TESTS PASSED`
