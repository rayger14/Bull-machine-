# Soft Gating Backtest Validation Guide

**Purpose**: Comprehensive validation of soft gating implementation across score-level and sizing-level mechanisms.

**Date**: 2026-01-10
**Status**: Ready for production testing

---

## Overview

This validation suite tests the soft gating implementation across multiple dimensions:

1. **Score-level gating**: Apply regime weights to signal scores (7 archetypes: B, C, S1, H, K, S4, S5)
2. **Position sizing integration**: Scale position sizes by regime weights and confidence
3. **Cash bucket mechanism**: Allow under-allocation when archetype edge is weak
4. **Regime budget caps**: Hard limits on total exposure per regime (e.g., 30% max in CRISIS)

## Files

### Primary Scripts

1. **`bin/validate_soft_gating_backtest.py`**
   - Main validation backtest script
   - Runs comprehensive tests across multiple periods and modes
   - Generates detailed reports and comparisons

2. **`bin/soft_gating_backtest_quick_test.py`**
   - Quick test with synthetic data
   - Validates framework logic without production data
   - Useful for development and debugging

### Supporting Files

- **`engine/portfolio/regime_allocator.py`**: RegimeWeightAllocator implementation
- **`engine/models/archetype_model.py`**: ArchetypeModel with soft gating integration
- **`results/archetype_regime_edge_table.csv`**: Edge metrics by archetype+regime

---

## Test Periods

### 1. 2022 Crisis (June-Dec)
- **Purpose**: Validate CRISIS regime handling
- **Expected**: ~$120 improvement from reduced liquidity_vacuum positions
- **Key metric**: Max CRISIS exposure ≤ 30%

### 2. 2023 Q1 Recovery (Jan-Apr)
- **Purpose**: Validate RISK_ON regime transition
- **Expected**: ~$102 improvement from reduced wick_trap positions
- **Key metric**: Smooth regime transitions

### 3. 2023 H2 Mixed (Aug-Dec)
- **Purpose**: Validate NEUTRAL regime behavior
- **Expected**: Stable performance, no unexpected regressions
- **Key metric**: Position sizing distribution

### 4. Full 2022-2024
- **Purpose**: Comprehensive system validation
- **Expected**: +$220 to +$270 total improvement
- **Key metric**: No unintended side effects

---

## Comparison Modes

### Mode 1: Baseline
- **Config**: No soft gating
- **Purpose**: Establish baseline performance
- **Features**:
  - Standard position sizing (20% per trade)
  - No regime weight adjustments
  - No budget caps

### Mode 2: Score-Only
- **Config**: Score-level gating enabled
- **Purpose**: Validate score weight application
- **Features**:
  - Signal scores multiplied by regime weights
  - Signals rejected if gated score < threshold
  - Position sizing unchanged

### Mode 3: Sizing-Only
- **Config**: Position sizing gating enabled
- **Purpose**: Validate position size scaling
- **Features**:
  - Signals unchanged
  - Position sizes scaled by: `base_size × regime_weight × confidence`
  - Regime budget caps enforced

### Mode 4: Full
- **Config**: All soft gating features enabled
- **Purpose**: Production-ready configuration
- **Features**:
  - Score-level gating
  - Position sizing gating
  - Cash bucket mechanism
  - Regime budget caps

---

## Usage

### Quick Test (Synthetic Data)

```bash
# Run quick test to validate framework
python bin/soft_gating_backtest_quick_test.py
```

**Output**:
- Console logs with results for each mode
- Synthetic edge table at `results/soft_gating_validation_test/edge_table_synthetic.csv`
- Validation that backtest logic works correctly

### Full Validation (Production Data)

```bash
# Run all periods, all modes (comprehensive)
python bin/validate_soft_gating_backtest.py --mode all --periods all

# Run specific period
python bin/validate_soft_gating_backtest.py --periods crisis

# Run specific mode
python bin/validate_soft_gating_backtest.py --mode full --periods full

# Custom output directory
python bin/validate_soft_gating_backtest.py --output results/my_validation

# Custom data paths
python bin/validate_soft_gating_backtest.py \
  --data data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet \
  --edge-table results/archetype_regime_edge_table.csv
```

### Command-Line Options

```
--mode {baseline,score_only,sizing_only,full,all}
    Comparison mode to run (default: all)

--periods {crisis,recovery,mixed,full,all}
    Test periods to run (default: all)

--output OUTPUT
    Output directory for results (default: results/soft_gating_validation)

--data DATA
    Path to feature data parquet file
    (default: data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet)

--edge-table EDGE_TABLE
    Path to edge table CSV
    (default: results/archetype_regime_edge_table.csv)
```

---

## Output Files

### 1. Trade Blotter CSVs
**Location**: `results/soft_gating_validation/trades_<period>_<mode>_<timestamp>.csv`

**Columns**:
- `timestamp`: Trade execution time
- `archetype`: Archetype name
- `direction`: long/short
- `regime`: Market regime at entry
- `entry_price`, `exit_price`: Trade prices
- `size_usd`: Position size in dollars
- `pnl_net`: Net PnL after fees/slippage
- `holding_hours`: Duration of trade
- `exit_reason`: stop_loss, take_profit, end_of_data
- `confidence`: Signal confidence score
- `regime_weight`: Regime weight applied (soft gating)
- `position_size_pct`: Actual position size as % of portfolio
- `budget_capped`: Whether regime budget cap was applied

### 2. Markdown Report
**Location**: `results/soft_gating_validation/SOFT_GATING_VALIDATION_REPORT_<timestamp>.md`

**Sections**:
- Executive summary with comparison table
- Detailed results by period
- PnL breakdown by regime and archetype
- Key insights and validation checks
- Expected vs actual results comparison

### 3. Console Output
Real-time progress logs with:
- Backtest execution details
- Entry/exit trade logs
- Rejection reason tracking
- Performance summaries
- Mode comparisons

---

## Metrics Tracked

### Overall Performance
- Total trades
- Win rate (%)
- Total PnL ($)
- Total return (%)
- Profit factor
- Sharpe ratio
- Max drawdown (%)

### Regime Breakdown
- PnL by regime (crisis, risk_off, neutral, risk_on)
- Trade count by regime
- Stop-out rate by regime

### Archetype Breakdown
- PnL by archetype
- Trade count by archetype
- Average regime weight by archetype
- Average position size by archetype

### Soft Gating Metrics
- Average regime weight across all trades
- Average position size percentage
- Budget cap trigger count
- Cash bucket utilization by regime
- Rejection reasons distribution

### Rejection Tracking
- `low_score`: Signal score too low
- `regime_mismatch`: Hard regime filter
- `regime_weight_too_low`: Gated score below threshold
- `budget_cap`: Regime budget exhausted
- `position_limit`: Max positions reached
- `cooldown`: Re-entry cooldown active

---

## Expected Results

### Based on Spec (SOFT_GATING_POSITION_SIZING_IMPLEMENTATION.md)

#### CRISIS Regime
- **Archetype**: liquidity_vacuum (S1)
- **Baseline**: Heavy losses due to negative edge (-0.042 Sharpe)
- **With soft gating**:
  - Position size: 20% → 8% (60% reduction)
  - Budget cap: 30% max total exposure
  - Expected improvement: ~$120

#### RISK_ON Regime
- **Archetype**: wick_trap_moneytaur (C)
- **Baseline**: Losses due to negative edge (-0.025 Sharpe)
- **With soft gating**:
  - Position size: 20% → 1.4% (93% reduction via low regime weight 0.07)
  - Expected improvement: ~$102

#### Total Expected
- **Combined improvement**: +$220 to +$270
- **Mechanism**: Smaller positions in negative-edge regimes
- **No side effects**: Other regimes unchanged

---

## Hard Gates (Acceptance Criteria)

### Test 1: No Forced 100% Allocation in Negative Edge
```python
# CRISIS regime should not force full allocation when edge is negative
assert crisis_regime_weight < 1.0 or crisis_edge > 0
```

### Test 2: CRISIS Budget Respected
```python
# Average CRISIS exposure should stay within 30% budget
assert crisis_avg_exposure <= 0.30
```

### Test 3: No Single-Archetype Dominance
```python
# RISK_ON shouldn't have massive losses from one archetype
assert wick_trap_risk_on_pnl > baseline_pnl - 150
```

### Test 4: Trade Count Stability
```python
# Total trades shouldn't explode (max 50% increase)
assert total_trades_ratio < 1.5
```

---

## Interpretation Guide

### Success Indicators

✓ **CRISIS regime improves by ~$120**
- Validates position size reduction working
- Confirms budget cap effectiveness

✓ **RISK_ON regime improves by ~$102**
- Validates regime weight application
- Confirms negative-edge archetypes reduced

✓ **Total improvement +$220 to +$270**
- Validates overall soft gating benefit
- Confirms no major regressions

✓ **No unintended side effects**
- Neutral regime stable
- Trade count reasonable
- No unexpected stops

### Red Flags

❌ **Worse performance than baseline**
- Check regime weights in edge table
- Verify budget caps not too restrictive
- Review rejection reasons

❌ **Massive trade count changes**
- Check score-level gating thresholds
- Review archetype signal generation
- Verify cooldown settings

❌ **High stop-out rates**
- Review stop loss calculations
- Check position sizing formulas
- Validate regime transitions

❌ **Budget caps trigger excessively**
- Review regime exposure tracking
- Check budget values in config
- Verify exposure calculation logic

---

## Troubleshooting

### Issue: No trades in Full mode

**Possible causes**:
1. Regime weights too low → all signals rejected
2. Budget caps too restrictive → no available allocation
3. Edge table missing archetype-regime pairs

**Solutions**:
- Check edge table has all 7 archetypes
- Review RegimeWeightAllocator logs
- Lower `min_threshold` in score gating
- Increase regime budgets temporarily

### Issue: Results identical across modes

**Possible causes**:
1. Edge table not loaded correctly
2. RegimeWeightAllocator not initialized
3. Config flags not applied

**Solutions**:
- Verify edge table path exists
- Check console logs for initialization errors
- Add debug logging to `_process_archetype()`

### Issue: Huge PnL swings

**Possible causes**:
1. Position sizing too aggressive
2. Slippage/fees incorrect
3. Stop losses not triggering

**Solutions**:
- Review position size calculations
- Verify fee_pct and slippage_pct settings
- Check `_manage_positions()` logic

---

## Next Steps After Validation

### If Results Match Expectations

1. **Enable in production config**:
   ```python
   from engine.portfolio.regime_allocator import RegimeWeightAllocator

   allocator = RegimeWeightAllocator(
       edge_table_path='results/archetype_regime_edge_table.csv'
   )

   model = ArchetypeModel(
       config_path='configs/s1_optimized.json',
       archetype_name='S1',
       regime_allocator=allocator  # Enable soft gating
   )
   ```

2. **Monitor in paper trading**:
   - Track regime weight application
   - Monitor budget cap triggers
   - Validate PnL improvements

3. **Gradual rollout**:
   - Start with 1-2 archetypes
   - Expand to all 7 after validation
   - Monitor for unintended side effects

### If Results Don't Match

1. **Review edge table**:
   - Ensure data quality
   - Check sample sizes sufficient
   - Verify Sharpe calculations

2. **Tune parameters**:
   - Adjust `k_shrinkage` (30 default)
   - Review `alpha` (4.0 default)
   - Modify `neg_edge_cap` (0.20 default)

3. **Re-run validation**:
   - Test with adjusted parameters
   - Compare against baseline again
   - Document changes and rationale

---

## Advanced Usage

### Custom Edge Table

```python
# Generate new edge table with custom parameters
import pandas as pd
from engine.portfolio.regime_allocator import RegimeWeightAllocator

# Create edge table
edge_data = [
    ('liquidity_vacuum', 'crisis', 57, -0.042, ...),
    # ... more rows
]
edge_df = pd.DataFrame(edge_data, columns=[...])
edge_df.to_csv('results/custom_edge_table.csv', index=False)

# Run validation with custom table
python bin/validate_soft_gating_backtest.py \
  --edge-table results/custom_edge_table.csv
```

### Custom Regime Budgets

```python
# Modify BacktestConfig in validate_soft_gating_backtest.py
config = BacktestConfig(
    mode='full',
    regime_risk_budgets={
        'crisis': 0.20,    # More conservative (was 0.30)
        'risk_off': 0.40,
        'neutral': 0.70,
        'risk_on': 0.90    # More aggressive (was 0.80)
    }
)
```

### Integration Testing

```python
# Test soft gating with full production stack
from bin.validate_soft_gating_backtest import SoftGatingBacktest
from engine.context.regime_service import RegimeService

# Use real regime classification
regime_service = RegimeService(
    model_path='models/logistic_regime_v2.pkl',
    enable_event_override=True
)

# Classify data
data = regime_service.classify_batch(raw_data)

# Run backtest
bt = SoftGatingBacktest(config, edge_table_path)
results = bt.run(data, start_date, end_date, archetypes)
```

---

## Key Insights to Extract

### 1. Which archetypes benefit most?
- Compare PnL by archetype across modes
- Identify biggest improvements
- Validate against edge table expectations

### 2. Which regimes see biggest improvements?
- Review regime breakdown tables
- Check if CRISIS/RISK_ON improved as expected
- Validate no regressions in NEUTRAL

### 3. Cash bucket impact on CRISIS?
- Check cash bucket utilization metrics
- Verify CRISIS has highest cash allocation
- Confirm weak archetypes stay in cash

### 4. Budget cap trigger frequency?
- Review budget cap counts
- Check if caps too restrictive
- Validate caps prevent concentration risk

### 5. Unexpected behavior?
- Review rejection reasons
- Check for anomalies in trade counts
- Identify any edge cases

---

## References

- **Spec**: `SOFT_GATING_POSITION_SIZING_IMPLEMENTATION.md`
- **Quick Start**: `SOFT_GATING_QUICK_START.md`
- **Integration Report**: `SOFT_GATING_INTEGRATION_REPORT.md`
- **Edge Table**: `results/archetype_regime_edge_table.csv`
- **RegimeWeightAllocator**: `engine/portfolio/regime_allocator.py`
- **ArchetypeModel**: `engine/models/archetype_model.py`

---

## Support

For questions or issues:
1. Check console logs for errors
2. Review rejection reasons distribution
3. Verify edge table loaded correctly
4. Compare against quick test results
5. Consult implementation documentation

**Author**: System Architect
**Last Updated**: 2026-01-10
