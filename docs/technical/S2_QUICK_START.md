# S2 Runtime Enrichment - Quick Start Guide

**5-minute guide to using S2 runtime features**

---

## What is This?

Runtime feature enrichment for S2 (Failed Rally) archetype. Adds 5 advanced features to improve pattern detection without changing the feature store.

**Features Added:**
1. Wick ratios (rejection strength)
2. Volume fade (declining volume)
3. RSI divergence (momentum weakness)
4. Order block approximation (resistance levels)
5. MTF confirmation (4H trend)

---

## Quick Start (3 Steps)

### Step 1: Enrich Your Dataframe

```python
from engine.strategies.archetypes.bear.failed_rally_runtime import apply_runtime_enrichment

# Load your data
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet')

# Apply enrichment (adds 5 columns)
df_enriched = apply_runtime_enrichment(df, lookback=14)

# New columns:
# - wick_upper_ratio
# - wick_lower_ratio
# - volume_fade_flag
# - rsi_bearish_div
# - ob_retest_flag
```

### Step 2: Use Enriched Features in S2 Detector

```python
# In _check_S2() method
def _check_S2(self, context: RuntimeContext) -> Tuple[bool, float, dict]:
    row = context.row

    # Use enriched features
    wick_ratio = row.get('wick_upper_ratio', 0.0)
    volume_fade = row.get('volume_fade_flag', False)
    rsi_div = row.get('rsi_bearish_div', False)
    ob_retest = row.get('ob_retest_flag', False)

    # Updated scoring
    components = {
        'wick_rejection': min(wick_ratio / 0.6, 1.0),
        'volume_fade': 1.0 if volume_fade else 0.3,
        'rsi_divergence': 1.0 if rsi_div else 0.5,
        'ob_retest': 1.0 if ob_retest else 0.3,
        'fusion': row.get('fusion_score', 0.0)
    }

    weights = {'wick_rejection': 0.25, 'volume_fade': 0.20, 'rsi_divergence': 0.25, 'ob_retest': 0.20, 'fusion': 0.10}
    score = sum(components[k] * weights[k] for k in components)

    # ... rest of logic
```

### Step 3: Run Backtest

```python
# Run your normal backtest - features are already in dataframe
python3 bin/backtest_knowledge_v2.py --config configs/bear/s2_enriched_test.json
```

---

## Performance Impact

**Negligible** - adds <1% to backtest time

```
10,000 bars: +250 milliseconds
Performance: 15-25 μs per bar
```

---

## Expected Results

**Before Enrichment:**
- Signal count: ~100-150
- Win rate: 50-55%
- Profit factor: 0.8-1.0

**After Enrichment:**
- Signal count: 150-250 (better detection)
- Win rate: 55-60% (better quality)
- Profit factor: 1.3-1.6 (target)

---

## Troubleshooting

### Issue: "Column not found" error

**Solution:** Make sure you enriched the dataframe before backtest
```python
df = apply_runtime_enrichment(df)  # Don't forget this!
```

### Issue: Low signal count after enrichment

**Solution:** Features are too strict. Relax thresholds:
```python
# In config
"failed_rally": {
  "weights": {
    "wick_rejection": 0.20,  # Lower from 0.25
    "volume_fade": 0.15,     # Lower from 0.20
    "rsi_divergence": 0.20,  # Lower from 0.25
    "ob_retest": 0.15,       # Lower from 0.20
    "fusion": 0.30           # Increase from 0.10
  }
}
```

### Issue: Too many signals (low quality)

**Solution:** Features are too loose. Tighten thresholds (reverse of above)

---

## Testing

```bash
# Test enrichment module standalone
python3 engine/strategies/archetypes/bear/failed_rally_runtime.py

# Expected output:
# ================================================================================
# ENRICHMENT STATISTICS
# ================================================================================
# Strong upper wicks (>0.4): 2388 (27.3%)
# Volume fades: 2324 (26.6%)
# RSI bearish divs: 494 (5.7%)
# OB retests: 3242 (37.1%)
# PERFECT S2 SIGNALS (all 4 features): 6 (0.07%)
```

---

## Feature Details

### Wick Upper Ratio
```
Formula: (high - max(open, close)) / (high - low)
Meaning: Rejection wick as % of total candle
Threshold: >0.4 = strong rejection
```

### Volume Fade Flag
```
Formula: current_volume < (0.8 * rolling_mean_3)
Meaning: Volume declining over 3 bars
Interpretation: Weak rally, no conviction
```

### RSI Bearish Divergence
```
Formula: price_higher_high AND rsi_lower_high
Meaning: Momentum weakening despite price rise
Lookback: 14 bars
```

### OB Retest Flag
```
Formula: price within 2% of recent swing high
Meaning: Price testing resistance
Fallback: Uses tf1h_ob_high if available
```

---

## Configuration Template

```json
{
  "archetypes": {
    "enable_S2": true,
    "failed_rally": {
      "fusion_threshold": 0.32,
      "use_enriched_features": true,
      "weights": {
        "wick_rejection": 0.25,
        "volume_fade": 0.20,
        "rsi_divergence": 0.25,
        "ob_retest": 0.20,
        "fusion": 0.10
      }
    }
  }
}
```

---

## Complete Example

```python
#!/usr/bin/env python3
"""
S2 Enriched Backtest Example
"""
import pandas as pd
from engine.strategies.archetypes.bear.failed_rally_runtime import apply_runtime_enrichment
from bin.backtest_knowledge_v2 import run_backtest

# 1. Load data
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet')

# 2. Enrich with runtime features
df_enriched = apply_runtime_enrichment(df, lookback=14)

print(f"Enriched {len(df_enriched)} bars")
print(f"New columns: {[c for c in df_enriched.columns if 'wick_' in c or '_flag' in c or '_div' in c]}")

# 3. Run backtest with enriched data
config = {
    'archetypes': {
        'enable_S2': True,
        'failed_rally': {
            'fusion_threshold': 0.32,
            'weights': {
                'wick_rejection': 0.25,
                'volume_fade': 0.20,
                'rsi_divergence': 0.25,
                'ob_retest': 0.20,
                'fusion': 0.10
            }
        }
    }
}

results = run_backtest(df_enriched, config)

print(f"\nResults:")
print(f"Trades: {results['trade_count']}")
print(f"Win Rate: {results['win_rate']:.1%}")
print(f"Profit Factor: {results['profit_factor']:.2f}")
```

---

## Documentation

**Full Design Doc:**
- `docs/technical/S2_RUNTIME_FEATURES_DESIGN.md` (600+ lines)

**Implementation Summary:**
- `docs/technical/S2_RUNTIME_ENRICHMENT_SUMMARY.md` (400+ lines)

**Source Code:**
- `engine/strategies/archetypes/bear/failed_rally_runtime.py` (377 lines)

---

**END OF QUICK START**

Questions? See full documentation or ask Claude Code.
