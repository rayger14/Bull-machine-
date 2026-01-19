# Baseline Suite Quick Start

## TL;DR

```bash
# Run baselines on your data
python bin/run_quant_suite.py --config configs/experiment_btc_1h_2020_2025.json --baselines-only

# Check results
cat results/quant_suite/quant_suite_report_*.md
```

## What Are Baselines?

6 simple strategies that any "fancy" system must beat:

1. **Buy-and-Hold** - Just buy and hold (sanity check)
2. **SMA200 Trend** - Follow 200-period moving average
3. **SMA Crossover** - Golden cross / death cross (50/200 SMA)
4. **RSI Mean Reversion** - Buy dips (RSI < 30), sell rips (RSI > 70)
5. **Vol-Targeted Trend** - SMA trend with volatility-adjusted sizing
6. **Cash** - Do nothing (validates engine works)

## Why Use Baselines?

**Honesty Check:**
- Can't beat buy-and-hold? → Don't trade
- Can't beat SMA200? → Your complexity isn't justified
- Cash shows non-zero PnL? → Your engine is broken

**Strategy Selection:**
- Rank all strategies (baselines + your archetypes)
- Kill anything below best baseline
- Only deploy strategies in top tier

## 5-Minute Setup

### 1. Check Your Data

Your data needs these columns:
```
timestamp, open, high, low, close, volume
```

Optional (will be calculated if missing):
```
sma_50, sma_200, rsi_14, atr_14
```

### 2. Create Config File

Copy `configs/experiment_btc_1h_2020_2025.json` and edit:

```json
{
  "experiment_name": "My Test",
  "asset": "BTC",
  "timeframe": "1H",
  "data_path": "data/features_mtf/YOUR_DATA.parquet",

  "periods": {
    "train": {"start": "2022-01-01", "end": "2023-06-30"},
    "test": {"start": "2023-07-01", "end": "2023-12-31"},
    "oos": {"start": "2024-01-01", "end": "2024-12-31"}
  },

  "costs": {
    "slippage_bps": 5,
    "fee_bps": 3,
    "total_bps": 8
  },

  "initial_capital": 10000.0
}
```

### 3. Run Suite

```bash
python bin/run_quant_suite.py --config configs/YOUR_CONFIG.json --baselines-only
```

### 4. Check Results

Results appear in `results/quant_suite/`:
- `quant_suite_results_TIMESTAMP.csv` - Full metrics
- `quant_suite_report_TIMESTAMP.md` - Analysis

## Reading Results

### Console Output

```
Rank  Model                   Train PF  Test PF  OOS PF   Overfit  Trades  Status
----  -----                   --------  -------  ------   -------  ------  ------
1     Baseline1_SMA200Trend   2.45      2.12     1.98     0.33     67      ✅
2     Baseline0_BuyAndHold    1.80      1.65     1.50     0.15     1       🔧
3     Baseline3_RSI14MR       1.50      1.10     0.85     0.40     142     ❌
4     Baseline5_Cash          0.00      0.00     0.00     0.00     0       ❌
```

**What to look for:**
- ✅ = Test PF >= 2.0 (excellent)
- 🔧 = Test PF 1.5-2.0 (acceptable)
- ❌ = Test PF < 1.5 (needs work)

### Key Metrics

**Profit Factor (PF):** Gross wins / gross losses
- < 1.0 = Losing strategy
- 1.0-1.5 = Marginal (costs will kill it)
- 1.5-2.0 = Decent
- \> 2.0 = Good

**Overfit Score:** Train PF - Test PF
- < 0.3 = Consistent (good)
- 0.3-0.5 = Some optimization (acceptable)
- \> 0.5 = Overfit (red flag)

**Trade Count:**
- < 50 = Low confidence
- 50-200 = Good sample
- \> 200 = Very active (watch costs)

**Sharpe Ratio:** Risk-adjusted returns
- < 0.5 = Poor
- 0.5-1.0 = Decent
- \> 1.0 = Good

## Common Patterns

### Strong Bull Market
```
Baseline0_BuyAndHold:  PF = 3.5  ← Market has strong edge
Baseline1_SMA200Trend: PF = 2.2  ← Trend-following works but lags
Baseline3_RSI14MR:     PF = 0.8  ← Mean reversion fails (trend too strong)
```
**Interpretation:** Trend is king. Any counter-trend strategy will lose.

### Bear Market
```
Baseline0_BuyAndHold:  PF = 0.3  ← Market has negative edge
Baseline1_SMA200Trend: PF = 1.8  ← Trend-following protects (goes to cash)
Baseline3_RSI14MR:     PF = 0.6  ← Catches falling knives
```
**Interpretation:** Buy-and-hold loses. Trend-following provides protection.

### Ranging Market
```
Baseline0_BuyAndHold:  PF = 1.0  ← No net movement
Baseline1_SMA200Trend: PF = 0.9  ← Whipsaws kill it
Baseline3_RSI14MR:     PF = 1.8  ← Mean reversion shines
```
**Interpretation:** Oscillator strategies work. Trend-following struggles.

## Decision Framework

### Step 1: Check Buy-and-Hold
- **PF > 2.0:** Strong bull market → Focus on trend-following
- **PF 1.0-2.0:** Mild trend → Any strategy can work
- **PF < 1.0:** Bear/ranging → Focus on protection/mean-reversion

### Step 2: Find Best Baseline
- This is your **minimum bar**
- Any strategy must beat this PF or it's not worth deploying

### Step 3: Check Your Strategies
- Rank all strategies (baselines + your archetypes)
- Kill anything below best baseline
- Only deploy top tier

### Step 4: Validate
- Check overfit (< 0.5)
- Check trade count (> 50)
- Check OOS (within 20% of test)
- Check Sharpe (> 0.5)

## Red Flags

**🚨 Baseline5_Cash shows PnL != $0.00**
- Your backtesting engine is broken
- Fix before proceeding

**🚨 All baselines have low PF (< 1.5)**
- Market has no edge, or
- Costs are too high, or
- Data quality issues

**🚨 All strategies show high overfit (> 0.5)**
- Train period too short
- Parameters too flexible
- Regime change between train/test

**🚨 Your archetype beats all baselines on train but loses on test**
- Classic overfit
- You curve-fitted to noise
- Start over with simpler strategy

## Next Steps

### 1. Understand the Landscape
Run baselines first to understand:
- Market regime (bull/bear/ranging)
- Best simple approach (trend/mean-reversion)
- Minimum acceptable PF

### 2. Build Strategies
Only build strategies that:
- Have clear edge hypothesis
- Can beat best baseline
- Work in expected regime

### 3. Test Rigorously
- Always compare to baselines
- Check overfit
- Validate on OOS
- Check multiple regimes

### 4. Deploy Conservatively
Only deploy strategies that:
- Beat best baseline by 20%+ on test
- Low overfit (< 0.3)
- Good OOS (within 20% of test)
- Enough trades (> 50)

## Advanced Usage

### Compare Different Periods

```bash
# Bull market period
python bin/run_quant_suite.py --config configs/bull_2023.json --baselines-only

# Bear market period
python bin/run_quant_suite.py --config configs/bear_2022.json --baselines-only

# Compare which baselines work in which regime
```

### Test Different Costs

Edit config `costs.total_bps`:
- Conservative: 8bp (5 slip + 3 fees)
- Aggressive: 4bp (2 slip + 2 fees)
- Institutional: 2bp (0.5 slip + 1.5 fees)

See how costs impact PF.

### Add Your Archetypes

```bash
# Run baselines + archetypes
python bin/run_quant_suite.py --config configs/experiment.json

# See how your archetypes rank vs baselines
```

## Summary

**The Baseline Suite is your reality check.**

1. Run baselines first (always)
2. Understand the market regime
3. Identify minimum bar (best baseline)
4. Build strategies that beat it
5. Kill anything that doesn't

If you can't beat simple, you don't deserve complex.

---

**Full Documentation:** See `docs/BASELINE_SUITE_GUIDE.md`
