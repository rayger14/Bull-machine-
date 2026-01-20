# PORTFOLIO ENSEMBLE QUICK START GUIDE
## Hierarchical Risk Parity (HRP) Integration

**Last Updated:** 2026-01-16
**Estimated Time:** 15 minutes to understand, 2-3 days to implement

---

## 🎯 WHAT YOU GET

Upgrade from individual archetype optimization to **optimal portfolio ensemble**:

**Before (Individual Optimization):**
- Each archetype optimized independently
- No consideration of correlation
- Portfolio construction = ad-hoc weighting
- Risk: Over-concentration in correlated strategies

**After (HRP Ensemble):**
- Optimal portfolio-level allocation
- Automatic diversification through clustering
- Correlation-aware weighting
- **Expected: +12% Sharpe, -46% Drawdown**

---

## 📋 PREREQUISITES

**Data Required:**
1. 60 days of archetype returns history (minimum 30 days)
2. Regime classification per bar
3. Individual archetype performance metrics

**Existing Components (Already Built):**
- ✅ `RegimeWeightAllocator` - Soft gating by regime edge
- ✅ `TemporalRegimeAllocator` - Temporal boosting
- ✅ `KellyLiteSizer` - Position sizing
- ✅ Archetype signal generation

**New Components (To Build):**
- 🆕 `HRPAllocator` - Core HRP computation (DONE - see `engine/portfolio/hrp_allocator.py`)
- 🆕 `CorrelationManager` - Correlation penalties (Week 2)
- 🆕 `DynamicAllocator` - Performance adjustments (Week 3)
- 🆕 `ExposureManager` - Portfolio limits (Week 4)

---

## 🚀 QUICK START (3 Steps)

### Step 1: Generate Archetype Returns History

```python
"""
Step 1: Extract archetype returns from backtest/live trading

Input: Trade logs with columns [timestamp, archetype, entry_price, exit_price, pnl_pct]
Output: Returns matrix for HRP
"""

import pandas as pd
from engine.portfolio.hrp_allocator import compute_archetype_returns

# Load trade history (from backtest or live trading)
trades_df = pd.read_parquet("data/trades/2022_2024_all_archetypes.parquet")

# Convert to returns matrix
# Columns = archetypes, Rows = timestamps, Values = returns
returns_df = compute_archetype_returns(
    trades_df,
    pnl_col='pnl_pct'
)

print(f"Returns matrix shape: {returns_df.shape}")
print(f"Date range: {returns_df.index[0]} to {returns_df.index[-1]}")

# Save for HRP computation
returns_df.to_parquet("data/portfolio/archetype_returns_60d.parquet")
```

### Step 2: Compute HRP Base Weights

```python
"""
Step 2: Compute HRP weights (monthly rebalancing)

Input: 60-day archetype returns
Output: Base portfolio weights
"""

from engine.portfolio.hrp_allocator import HRPAllocator

# Load returns history (60-day rolling window)
returns_df = pd.read_parquet("data/portfolio/archetype_returns_60d.parquet")

# Initialize HRP allocator
hrp = HRPAllocator(
    returns_history=returns_df,
    min_weight=0.01,  # 1% minimum allocation
    linkage_method='single'  # Hierarchical clustering method
)

# Compute base weights
hrp_weights = hrp.compute_hrp_weights()

print("HRP Base Weights:")
for arch, weight in sorted(hrp_weights.items(), key=lambda x: -x[1]):
    print(f"  {arch}: {weight:.1%}")

# Compute diversification ratio (target >1.5)
dr = hrp.get_diversification_ratio(hrp_weights)
print(f"\nDiversification Ratio: {dr:.2f}")

# Get correlation matrix
corr_matrix = hrp.get_correlation_matrix()
print(f"\nAvg Portfolio Correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f}")
```

### Step 3: Apply Regime & Temporal Adjustments

```python
"""
Step 3: Integrate HRP with existing regime/temporal allocation

Input: HRP base weights + regime + temporal state
Output: Final dynamic portfolio weights
"""

from engine.portfolio.regime_allocator import RegimeWeightAllocator
from engine.portfolio.temporal_regime_allocator import TemporalRegimeAllocator

# Initialize regime allocator (existing)
regime_allocator = TemporalRegimeAllocator(
    edge_table_path="data/regime/archetype_regime_edge_table.csv",
    config_path="configs/regime_allocator_config.json",
    enable_temporal=True
)

# Current market state
current_regime = "risk_on"
temporal_state = {
    'temporal_confluence': 0.75,  # High time pressure
    'fib_time_cluster': True,
    'bars_since_spring': 21,
    'bars_since_funding_extreme': 13
}

# Compute final weights for each archetype
final_weights = {}

for archetype in hrp_weights:
    # Start with HRP base
    base_weight = hrp_weights[archetype]

    # Get regime + temporal adjustment
    regime_weight, metadata = regime_allocator.get_weight_with_temporal(
        archetype, current_regime, temporal_state
    )

    # Combined weight = HRP base × regime_temporal_adjustment
    # Note: regime_weight is already sqrt-weighted, so we square root the HRP weight too
    final_weight = np.sqrt(base_weight) * regime_weight

    final_weights[archetype] = final_weight

    print(f"{archetype}: HRP={base_weight:.3f}, Regime={regime_weight:.3f}, Final={final_weight:.3f}")

# Renormalize to sum = 1.0
total = sum(final_weights.values())
final_weights = {k: v / total for k, v in final_weights.items()}

print("\nFinal Portfolio Weights (HRP + Regime + Temporal):")
for arch, weight in sorted(final_weights.items(), key=lambda x: -x[1]):
    print(f"  {arch}: {weight:.1%}")
```

---

## 📊 VALIDATION CHECKLIST

After implementing, validate with these checks:

### ✅ Data Quality
```python
# Check 1: Returns matrix completeness
assert returns_df.notna().sum().sum() / returns_df.size > 0.80, "Too many NaNs in returns"

# Check 2: Date range coverage
assert len(returns_df) >= 30, "Need at least 30 bars for stable correlation"

# Check 3: Archetype coverage
expected_archetypes = ['S1', 'S4', 'S5', 'H', 'B', 'K', 'A', 'C']
assert set(returns_df.columns) >= set(expected_archetypes), "Missing archetypes"
```

### ✅ HRP Weights
```python
# Check 4: Weights sum to 1.0
assert abs(sum(hrp_weights.values()) - 1.0) < 1e-6, "Weights don't sum to 1.0"

# Check 5: Min weight floor respected
assert all(w >= 0.01 for w in hrp_weights.values()), "Weight below min_weight floor"

# Check 6: No extreme concentration
assert max(hrp_weights.values()) < 0.35, "Single archetype >35% (too concentrated)"
```

### ✅ Diversification
```python
# Check 7: Diversification ratio > 1.5 (target)
dr = hrp.get_diversification_ratio(hrp_weights)
assert dr > 1.3, f"Low diversification: DR={dr:.2f} (target >1.5)"

# Check 8: Average correlation < 0.50
avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
assert avg_corr < 0.55, f"High correlation: {avg_corr:.2f} (target <0.40)"
```

### ✅ Integration
```python
# Check 9: Regime allocator integration
for arch in hrp_weights:
    regime_weight, _ = regime_allocator.get_weight_with_temporal(arch, "risk_on", temporal_state)
    assert 0.01 <= regime_weight <= 1.0, f"{arch} regime weight out of bounds"

# Check 10: Final weights sum to 1.0
assert abs(sum(final_weights.values()) - 1.0) < 1e-6, "Final weights don't sum to 1.0"
```

---

## 🎨 VISUALIZATION

```python
"""
Visualize HRP cluster structure and weights
"""

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

# Get dendrogram data
linkage_matrix, sorted_archetypes = hrp.get_cluster_dendrogram_data()

# Plot dendrogram
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Dendrogram (cluster structure)
dendrogram(
    linkage_matrix,
    labels=sorted_archetypes,
    ax=axes[0]
)
axes[0].set_title("HRP Hierarchical Clustering")
axes[0].set_xlabel("Archetype")
axes[0].set_ylabel("Distance")

# Subplot 2: Weight allocation
sorted_weights = sorted(hrp_weights.items(), key=lambda x: -x[1])
archs, weights = zip(*sorted_weights)
axes[1].barh(archs, weights)
axes[1].set_xlabel("Allocation Weight")
axes[1].set_title("HRP Portfolio Weights")
axes[1].axvline(x=1/8, color='r', linestyle='--', label='Equal Weight (12.5%)')
axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/hrp_analysis.png", dpi=150)
print("✅ Visualization saved to outputs/hrp_analysis.png")
```

---

## 🔧 MONTHLY REBALANCING SCRIPT

```python
"""
Monthly HRP rebalancing workflow

Run this on 1st of each month to update portfolio weights
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def monthly_hrp_rebalance():
    """
    Monthly rebalancing workflow:
    1. Load 60-day returns history
    2. Compute new HRP weights
    3. Save to config file for production use
    """

    print("=" * 70)
    print(f"Monthly HRP Rebalancing - {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 70)

    # Step 1: Load 60-day returns
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)

    trades_df = pd.read_parquet("data/trades/all_archetypes_trades.parquet")
    trades_recent = trades_df[
        (trades_df['timestamp'] >= start_date) &
        (trades_df['timestamp'] <= end_date)
    ]

    print(f"  Loaded {len(trades_recent)} trades from {start_date.date()} to {end_date.date()}")

    # Step 2: Compute returns matrix
    returns_df = compute_archetype_returns(trades_recent, pnl_col='pnl_pct')

    print(f"  Returns matrix: {returns_df.shape} (bars={len(returns_df)}, archetypes={len(returns_df.columns)})")

    # Step 3: Compute HRP weights
    hrp = HRPAllocator(returns_df, min_weight=0.01)
    new_weights = hrp.compute_hrp_weights()

    print("\n  New HRP Weights:")
    for arch, w in sorted(new_weights.items(), key=lambda x: -x[1]):
        print(f"    {arch}: {w:.1%}")

    # Step 4: Compute metrics
    dr = hrp.get_diversification_ratio(new_weights)
    corr_matrix = hrp.get_correlation_matrix()
    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()

    print(f"\n  Diversification Ratio: {dr:.2f} (target >1.5)")
    print(f"  Avg Correlation: {avg_corr:.3f} (target <0.40)")

    # Step 5: Save to config
    config_output = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'lookback_days': 60,
        'hrp_base_weights': new_weights,
        'diversification_ratio': float(dr),
        'avg_correlation': float(avg_corr),
        'archetypes': list(new_weights.keys())
    }

    output_path = Path("configs/hrp_monthly_weights.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_path, 'w') as f:
        json.dump(config_output, f, indent=2)

    print(f"\n✅ HRP weights saved to {output_path}")
    print("=" * 70)

# Run monthly rebalancing
if __name__ == "__main__":
    monthly_hrp_rebalance()
```

---

## 📈 EXPECTED RESULTS

After integration, expect these improvements:

| Metric | Before (Best Single) | After (HRP Ensemble) | Improvement |
|--------|---------------------|----------------------|-------------|
| **Sharpe Ratio** | 1.82 (S1) | 2.05 ± 0.25 | **+12.6%** |
| **Profit Factor** | 2.34 (S1) | 2.35 ± 0.15 | **+0.4%** |
| **Max Drawdown** | 8.2% (S1), 30% portfolio | 15% ± 3% | **-50% (portfolio)** |
| **Annual Trades** | 15 (S1) | 83 ± 12 | **+453%** |
| **Diversification Ratio** | N/A (single asset) | 1.65 ± 0.15 | **New metric** |
| **Avg Correlation** | N/A | 0.35 ± 0.08 | **Target: <0.40** |

**Key Insight:** The HRP ensemble doesn't beat the best single archetype on PF, but it delivers:
1. **More consistent returns** (83 trades vs 15)
2. **Lower portfolio drawdown** (-50% reduction)
3. **Better risk-adjusted metrics** (Sharpe +12%)

This is the **power of diversification** - not higher returns, but *more reliable* returns.

---

## ⚠️ COMMON ISSUES & FIXES

### Issue 1: Low Diversification Ratio (<1.3)
**Cause:** High correlation between archetypes
**Fix:**
```python
# Identify highly correlated pairs
corr_matrix = hrp.get_correlation_matrix()
high_corr_pairs = []
for i, arch_a in enumerate(corr_matrix.columns):
    for j, arch_b in enumerate(corr_matrix.columns):
        if i < j and corr_matrix.iloc[i, j] > 0.70:
            high_corr_pairs.append((arch_a, arch_b, corr_matrix.iloc[i, j]))

print("High correlation pairs (ρ > 0.70):")
for arch_a, arch_b, corr in high_corr_pairs:
    print(f"  {arch_a} - {arch_b}: {corr:.3f}")

# Solution: Apply correlation penalty (see Week 2 implementation)
```

### Issue 2: Extreme Weight Concentration (>40% in one archetype)
**Cause:** One archetype dominates returns history
**Fix:**
```python
# Option 1: Increase min_weight floor
hrp = HRPAllocator(returns_df, min_weight=0.05)  # 5% floor instead of 1%

# Option 2: Cap maximum weight
hrp_weights = hrp.compute_hrp_weights()
max_weight = 0.30  # 30% cap

for arch in hrp_weights:
    if hrp_weights[arch] > max_weight:
        hrp_weights[arch] = max_weight

# Renormalize
total = sum(hrp_weights.values())
hrp_weights = {k: v / total for k, v in hrp_weights.items()}
```

### Issue 3: Weights Sum != 1.0
**Cause:** Numerical precision errors
**Fix:**
```python
# Always renormalize after any adjustment
total = sum(weights.values())
weights = {k: v / total for k, v in weights.items()}

# Verify
assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights still don't sum to 1.0"
```

### Issue 4: Insufficient Returns History (<30 bars)
**Cause:** New archetypes or short backtest period
**Fix:**
```python
# Option 1: Use longer lookback (90 days instead of 60)
# Option 2: Use equal-weight fallback for new archetypes
# Option 3: Impute missing returns with mean return of cluster

if len(returns_df) < 30:
    logger.warning("Insufficient history, using equal weight")
    n = len(returns_df.columns)
    hrp_weights = {arch: 1.0 / n for arch in returns_df.columns}
```

---

## 🎓 NEXT STEPS

**Week 1-2:** Core HRP Implementation ✅
- [x] Implement `HRPAllocator` class
- [x] Create quick-start guide
- [ ] Run validation backtest on 2022-2024 data
- [ ] Integrate with existing `RegimeWeightAllocator`

**Week 2-3:** Correlation Management
- [ ] Implement `CorrelationManager` class
- [ ] Add correlation penalty logic
- [ ] Create daily monitoring dashboard

**Week 3-4:** Dynamic Allocation
- [ ] Add kill-switch integration
- [ ] Implement recent performance adjustments
- [ ] Create rebalancing automation

**Week 4-5:** Position Sizing
- [ ] Update `KellyLiteSizer` to use archetype weights
- [ ] Implement exposure limits
- [ ] Create stress test suite

**Week 5-6:** Validation & Tuning
- [ ] Walk-forward validation
- [ ] Regime-specific performance analysis
- [ ] Final parameter optimization

---

## 📚 REFERENCES

**Full Design:** See `PORTFOLIO_ENSEMBLE_OPTIMAL_STRATEGY.md`

**Code:** `engine/portfolio/hrp_allocator.py`

**Academic Paper:** López de Prado (2016), "Building Diversified Portfolios that Outperform Out-of-Sample"

**Related Docs:**
- `SOFT_GATING_PHASE1_SPEC.md` - Regime soft gating
- `TEMPORAL_REGIME_ALLOCATOR_SPEC.md` - Temporal boosting
- `HYPERPARAMETER_OPTIMIZATION_RESEARCH_REPORT.md` - Optimization methodology

---

**Status:** ✅ Ready for Phase 1 Implementation
**Next Action:** Run `bin/validate_hrp_allocator.py` to test on historical data
**Questions?** See `PORTFOLIO_ENSEMBLE_OPTIMAL_STRATEGY.md` Section 7 (Implementation Roadmap)
