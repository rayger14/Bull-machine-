# Baseline Suite Guide

## Overview

The Baseline Suite provides 6 simple, transparent trading strategies that serve as benchmarks for all "fancy" Bull Machine archetypes. Any complex strategy must beat these baselines or it doesn't justify its existence.

**Philosophy:**
- If you can't beat buy-and-hold, why trade at all?
- If you can't beat a simple SMA crossover, your complexity isn't justified
- Baselines provide an honesty check against over-optimization

## The Baselines

### Baseline 0: Buy and Hold
**Strategy:** Buy on first bar, hold until end
**Purpose:** Ultimate sanity check - does the market have positive expectancy?
**Parameters:** None
**Expected Performance:**
- High profit factor IF market trends up
- Zero profit factor IF market trends down
- Exactly 1 trade for entire period
- No drawdown control

**When to use:**
- Always run this first
- If your strategy can't beat this in a bull market, you're just churning

### Baseline 1: SMA 200 Trend Following
**Strategy:** Long when close > SMA(200), cash when close < SMA(200)
**Purpose:** Classic trend-following (institutional standard)
**Parameters:**
- `sma_period`: 200 (default)
- `stop_loss_pct`: 0.05 (5% stop)

**Expected Performance:**
- Outperforms buy-and-hold in ranging/bear markets
- Underperforms in strong bull markets (whipsaws)
- Lower drawdowns than buy-and-hold
- Low trade frequency

**When to use:**
- Tests if trend-following has edge
- Benchmark for drawdown management

### Baseline 2: SMA Crossover (Golden Cross)
**Strategy:** Long when SMA(50) > SMA(200), cash otherwise
**Purpose:** Tests momentum strategy (very popular retail approach)
**Parameters:**
- `fast_period`: 50 (default)
- `slow_period`: 200 (default)
- `stop_loss_pct`: 0.05 (5% stop)

**Expected Performance:**
- Fewer trades than single SMA (waits for confirmation)
- Lags entries/exits more
- Classic "late to the party" problem
- May have better win rate but worse profit factor

**When to use:**
- Tests if dual confirmation adds value
- Benchmark for lagging indicator approaches

### Baseline 3: RSI Mean Reversion
**Strategy:** Long when RSI(14) < 30, exit when RSI(14) > 70
**Purpose:** Tests "buy the dip" approach
**Parameters:**
- `rsi_period`: 14 (Welles Wilder's original)
- `entry_threshold`: 30 (oversold)
- `exit_threshold`: 70 (overbought)
- `stop_loss_pct`: 0.05 (5% stop)

**Expected Performance:**
- Works well in ranging markets
- Catastrophic in strong trends (catches falling knives)
- Higher trade frequency than trend-following
- Win rate may be deceptive (small wins, big losses)

**When to use:**
- Tests counter-trend approaches
- Identifies ranging vs trending markets

### Baseline 4: Volatility-Targeted Trend
**Strategy:** SMA(200) trend with volatility-adjusted position sizing
**Purpose:** Tests if risk-adjusted sizing improves performance
**Parameters:**
- `sma_period`: 200 (trend)
- `atr_period`: 14 (volatility)
- `target_vol`: 0.02 (2% daily vol target)
- `stop_atr_mult`: 2.5 (ATR-based stops)

**Expected Performance:**
- Smoother equity curve than fixed sizing
- Better risk-adjusted returns (Sharpe ratio)
- Automatically de-risks in volatile markets
- May underperform in low-vol trending markets

**When to use:**
- Benchmark for risk management approaches
- Tests value of volatility scaling

### Baseline 5: Cash (Do Nothing)
**Strategy:** Never trade, always hold cash
**Purpose:** Engine sanity check
**Parameters:** None
**Expected Performance:**
- Total PnL: $0.00 (exactly)
- Trades: 0
- If this shows ANY PnL, your engine is broken

**When to use:**
- Always include in suite
- Validates backtesting engine integrity

## Quant Suite Runner

### Basic Usage

```bash
# Run baselines only
python bin/run_quant_suite.py --config configs/experiment_btc_1h_2020_2025.json --baselines-only

# Run baselines + archetypes
python bin/run_quant_suite.py --config configs/experiment_btc_1h_2020_2025.json

# Verbose mode
python bin/run_quant_suite.py --config configs/experiment_btc_1h_2020_2025.json --verbose

# Custom output directory
python bin/run_quant_suite.py --config configs/experiment_btc_1h_2020_2025.json --output results/my_test
```

### What It Does

1. **Loads Configuration:** Reads experiment parameters (asset, periods, costs)
2. **Loads Data:** Loads feature store data, adds basic indicators
3. **Builds Model List:** Discovers all baselines (and optionally archetypes)
4. **Runs Backtests:** Executes train/test/OOS backtests for each model
5. **Calculates Metrics:** Computes PF, WR, Sharpe, MDD, etc.
6. **Generates Report:** Creates ranked table, CSV, and markdown report

### Output Files

After running, you'll get:

```
results/quant_suite/
├── quant_suite_results_20231205_143022.csv    # All metrics in CSV
└── quant_suite_report_20231205_143022.md      # Analysis report
```

### Results Table Format

```
Rank  Model                            Train PF  Test PF  OOS PF   Overfit  Trades  Status
----  ------                           --------  -------  ------   -------  ------  ------
1     Baseline1_SMA200Trend            2.45      2.12     1.98     0.33     67      ✅
2     Baseline4_VolTarget2pct          2.30      2.05     1.85     0.25     71      ✅
3     Baseline2_SMA50x200              2.20      1.75     1.60     0.45     34      🔧
4     Baseline0_BuyAndHold             1.80      1.65     1.50     0.15     1       🔧
5     Baseline3_RSI14MR                1.50      1.10     0.85     0.40     142     ❌
6     Baseline5_Cash                   0.00      0.00     0.00     0.00     0       ❌
```

**Legend:**
- ✅ = Test PF >= 2.0 (excellent)
- 🔧 = Test PF 1.5-2.0 (acceptable)
- ❌ = Test PF < 1.5 (needs work)

### Interpreting Results

**1. Check Buy-and-Hold First**
- If Buy-and-Hold has high PF, market is in strong trend
- If Buy-and-Hold is negative, you're in bear market
- Any active strategy should beat it (or you're wasting effort)

**2. Check Overfit Score**
- Overfit = Train PF - Test PF
- < 0.3: Good (consistent performance)
- 0.3-0.5: Acceptable (some optimization)
- \> 0.5: Red flag (likely overfit)

**3. Check Trade Count**
- < 50 trades: Not enough data (low confidence)
- 50-200 trades: Good sample size
- \> 200 trades: Very active strategy (watch costs)

**4. Check OOS Performance**
- If OOS << Test, model doesn't generalize
- If OOS > Test, you got lucky (or market changed)
- OOS should be within 20% of Test PF

**5. Red Flags**
- High overfit (Train >> Test)
- Low trade count (< 50)
- Poor OOS (PF < 1.0)
- High Sharpe on train but not test (curve-fitted)

## Configuration File

### Example Config

```json
{
  "experiment_name": "BTC 1H Standard Test (2020-2025)",
  "asset": "BTC",
  "timeframe": "1H",
  "data_path": "data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet",

  "periods": {
    "train": {
      "start": "2022-01-01",
      "end": "2023-06-30",
      "description": "Bear market + recovery"
    },
    "test": {
      "start": "2023-07-01",
      "end": "2023-12-31",
      "description": "Bull market run (primary validation)"
    },
    "oos": {
      "start": "2024-01-01",
      "end": "2024-12-31",
      "description": "True out-of-sample"
    }
  },

  "costs": {
    "slippage_bps": 5,
    "fee_bps": 3,
    "total_bps": 8,
    "description": "Conservative cost assumptions"
  },

  "initial_capital": 10000.0,

  "acceptance_criteria": {
    "min_test_pf": 1.5,
    "min_test_sharpe": 0.5,
    "max_overfit": 0.5,
    "min_trades": 50
  }
}
```

### Key Parameters

**periods:** Define train/test/OOS splits
- Train: Fit model parameters
- Test: Primary validation (rank models by this)
- OOS: True holdout (should never be used in optimization)

**costs:** Transaction costs
- `slippage_bps`: Price impact (5bp = 0.05%)
- `fee_bps`: Exchange fees (3bp = 0.03%)
- `total_bps`: Combined costs (8bp = 0.08%)

**acceptance_criteria:** Minimum requirements
- `min_test_pf`: Minimum profit factor on test
- `min_test_sharpe`: Minimum Sharpe ratio
- `max_overfit`: Maximum Train-Test PF difference
- `min_trades`: Minimum trades for statistical significance

## Adding New Baselines

### Step 1: Create Model File

```python
# engine/models/baselines/my_baseline.py

from engine.models.base import BaseModel, Signal, Position

class Baseline6_MyStrategy(BaseModel):
    def __init__(self, param1=10):
        super().__init__(name="Baseline6_MyStrategy")
        self.param1 = param1

    def fit(self, train_data, **kwargs):
        # Calibration logic
        self._is_fitted = True

    def predict(self, bar, position=None):
        # Signal generation logic
        return Signal(...)

    def get_position_size(self, bar, signal):
        # Position sizing logic
        return 1000.0
```

### Step 2: Add to __init__.py

```python
# engine/models/baselines/__init__.py

from .my_baseline import Baseline6_MyStrategy

__all__ = [
    # ... existing baselines
    'Baseline6_MyStrategy',
]
```

### Step 3: Update get_all_baselines()

```python
def get_all_baselines():
    return [
        # ... existing baselines
        Baseline6_MyStrategy,
    ]
```

### Step 4: Run Tests

```bash
python3 -m pytest tests/test_baselines.py -v
```

## Best Practices

### 1. Always Run Baselines First
Before optimizing any complex strategy, run baselines to understand:
- Does the market have edge? (Buy-and-hold PF)
- Does trend-following work? (SMA trend PF)
- Is it ranging or trending? (Compare SMA vs RSI)

### 2. Use Baselines as Kill Criteria
If your optimized archetype can't beat the best baseline:
- It's overfit
- It's not adding value
- Kill it and move on

### 3. Check Multiple Regimes
Run baselines on:
- Bull markets (should trend-following win)
- Bear markets (should buy-and-hold lose)
- Ranging markets (should mean-reversion win)

### 4. Watch Transaction Costs
Baselines help you understand cost impact:
- High-frequency strategies need higher PF
- Low-frequency can survive on lower PF
- If costs kill your edge, rethink approach

### 5. Use for Sanity Checks
- Baseline5_Cash validates engine (should be $0.00)
- Baseline0_BuyAndHold validates market direction
- If baselines show weird results, debug before proceeding

## Troubleshooting

### "Insufficient history" errors
Baselines need enough bars for indicators:
- SMA(200): Needs 200+ bars
- RSI(14): Needs 14+ bars
- ATR(14): Needs 14+ bars

**Solution:** Ensure data starts earlier or use shorter periods

### All baselines show low PF
Possible causes:
- High transaction costs (check `total_bps`)
- Ranging market (no edge for trend-following)
- Poor data quality (missing bars, wrong prices)

**Solution:** Review costs and data quality

### Baseline5_Cash shows non-zero PnL
**This is critical - engine has a bug!**
- Check commission calculation
- Check position sizing
- Check equity curve computation

### High overfit on all models
- Train period too short (overfitting noise)
- Test period very different from train (regime change)
- Parameters too flexible (need constraints)

**Solution:** Use longer train period or lock parameters

## Summary

The Baseline Suite is your reality check. It tells you:
1. **Does the market have edge?** (Buy-and-hold)
2. **What type of market is it?** (Trending vs ranging)
3. **What's the minimum bar?** (Best baseline PF)
4. **Is my engine working?** (Cash = $0.00)

Run baselines first, understand the landscape, then build strategies that beat them. If you can't beat simple, you don't deserve complex.
