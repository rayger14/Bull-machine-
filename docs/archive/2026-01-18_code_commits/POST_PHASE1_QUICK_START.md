# Post-Phase 1 Quick Start Guide

**TL;DR:** Phase 1 infrastructure is complete (liquidity backfilled, features verified), but bear archetypes aren't firing because they're disabled in config. Here's how to fix it in 2 hours.

---

## The Problem

✅ **What works:**
- liquidity_score: 100% coverage, mean=0.450
- S2 (Failed Rally) detection logic: Implemented
- S5 (Long Squeeze) detection logic: Implemented + funding fix
- Feature store: All macro features verified

❌ **What's broken:**
- 2022 validation: 109/118 trades = `tier1_market` (bull patterns)
- S2 produced: 0 trades (should be 15-30)
- S5 produced: 8 trades (correct archetype but low count)
- Problem: No config file has `enable_S2=true, enable_S5=true`

---

## The Fix (2 hours)

### Step 1: Create Bear Config (30 min)

Create `configs/bear_market_2022_test.json`:

```json
{
  "archetypes": {
    "use_archetypes": true,
    
    "enable_A": false,
    "enable_B": false,
    "enable_C": false,
    "enable_D": false,
    "enable_E": false,
    "enable_F": false,
    "enable_G": false,
    "enable_H": false,
    "enable_K": false,
    "enable_L": false,
    "enable_M": false,
    
    "enable_S1": false,
    "enable_S2": true,
    "enable_S3": false,
    "enable_S4": false,
    "enable_S5": true,
    "enable_S6": false,
    "enable_S7": false,
    "enable_S8": false,
    
    "thresholds": {
      "failed_rally": {
        "fusion_threshold": 0.36,
        "wick_ratio_min": 2.0,
        "rsi_min": 70.0,
        "vol_z_max": 0.5,
        "use_runtime_features": false
      },
      "long_squeeze": {
        "fusion_threshold": 0.35,
        "funding_z_min": 1.2,
        "rsi_min": 70,
        "liquidity_max": 0.25
      }
    },
    
    "routing": {
      "neutral": {
        "weights": {
          "failed_rally": 1.5,
          "long_squeeze": 1.5
        }
      },
      "risk_off": {
        "weights": {
          "failed_rally": 2.0,
          "long_squeeze": 2.2
        }
      },
      "crisis": {
        "weights": {
          "failed_rally": 2.5,
          "long_squeeze": 2.8
        }
      }
    }
  }
}
```

### Step 2: Run Isolation Test (60 min)

```bash
python bin/backtest_knowledge_v2.py \
  --asset BTC \
  --start 2022-01-01 \
  --end 2022-12-31 \
  --config configs/bear_market_2022_test.json \
  --export-trades results/validation/activation_test.csv
```

### Step 3: Verify Results (30 min)

```python
import pandas as pd

df = pd.read_csv('results/validation/activation_test.csv')

print(f"Total trades: {len(df)}")
print("\nArchetype distribution:")

# Count each archetype
arch_counts = {}
for col in df.columns:
    if 'archetype' in col or 'tier1' in col:
        count = df[col].sum()
        if count > 0:
            arch_counts[col] = count
            subset = df[df[col] == 1]
            wr = subset['trade_won'].sum() / len(subset) if len(subset) > 0 else 0
            print(f"  {col}: {count} trades ({count/len(df)*100:.1f}%), WR={wr:.1%}")

# Overall performance
wins = df['trade_won'].sum()
wr = wins / len(df) if len(df) > 0 else 0
total_r = df['r_multiple'].sum()

win_r = df[df['trade_won']==1]['r_multiple'].sum()
loss_r = abs(df[df['trade_won']==0]['r_multiple'].sum())
pf = win_r / loss_r if loss_r > 0 else 0

print(f"\nOverall Performance:")
print(f"  Win Rate: {wr:.1%}")
print(f"  Profit Factor: {pf:.2f}")
print(f"  Total R: {total_r:.2f}")
```

### Expected Results

**Success:**
```
Total trades: 18-35
Archetype distribution:
  archetype_failed_rally: 12-22 trades (60-80%), WR=45-55%
  archetype_long_squeeze: 4-10 trades (20-30%), WR=40-50%

Overall Performance:
  Win Rate: 42-52%
  Profit Factor: 1.2-1.6
  Total R: 3.5-8.2
```

**Failure (if still broken):**
```
Total trades: 100+
Archetype distribution:
  tier1_market: 100+ trades (still using bull logic)
```

If you see failure, check:
1. Config file loaded correctly (`grep "Loading config" in backtest logs`)
2. Enable flags respected (`grep "enable_S2" in logs`)
3. Archetypes actually enabled (`grep "S2 enabled:" in logs`)

---

## Next Steps After Success

Once isolation test passes (S2 + S5 firing):

1. **Add gate logging** (Priority 2) to understand rejections
2. **Test routing** (Priority 3) with mixed bull+bear archetypes
3. **Tune thresholds** (Priority 4) to optimize trade count vs quality
4. **Full validation** (Priority 5) on 2022-2024 period

See `POST_PHASE1_ROADMAP.md` for full details.

---

## Troubleshooting

### Problem: tier1_market still dominates

**Diagnosis:**
```bash
grep -E "enable_S2|enable_S5" results/validation/activation_test.log
```

**Fix:** Verify config is actually loaded:
```python
import json
with open('configs/bear_market_2022_test.json') as f:
    cfg = json.load(f)
print(cfg['archetypes']['enable_S2'])  # Should be true
print(cfg['archetypes']['enable_S5'])  # Should be true
```

### Problem: Zero trades

**Diagnosis:**
```bash
grep -E "\[S2|S5\]" results/validation/activation_test.log | head -20
```

**Fix:** Thresholds too strict, create relaxed config:
```json
{
  "archetypes": {
    "thresholds": {
      "failed_rally": {
        "fusion_threshold": 0.30,  // Relaxed
        "wick_ratio_min": 1.5,     // Relaxed
        "rsi_min": 65.0            // Relaxed
      },
      "long_squeeze": {
        "fusion_threshold": 0.30,  // Relaxed
        "funding_z_min": 0.8,      // Relaxed
        "rsi_min": 60              // Relaxed
      }
    }
  }
}
```

### Problem: Crashes or errors

**Check feature availability:**
```python
import pandas as pd
from pathlib import Path

# Load feature store
files = list(Path('data/features_mtf').glob('BTC_1H_*.parquet'))
df = pd.read_parquet(files[0])

# Check required features
required = ['funding_Z', 'rsi_14', 'tf1h_ob_high', 'volume_zscore', 
            'tf4h_external_trend', 'liquidity_score']

for feat in required:
    if feat in df.columns:
        coverage = df[feat].notna().sum() / len(df) * 100
        print(f"✅ {feat}: {coverage:.1f}% coverage")
    else:
        print(f"❌ {feat}: MISSING")
```

---

## Success Criteria

After this quick start, you should have:

- ✅ Config file that enables S2 + S5
- ✅ Backtest runs without errors
- ✅ S2 + S5 produce trades (not tier1_market)
- ✅ Reasonable performance (WR >40%, PF >1.2)

Then proceed to full roadmap for optimization and production deployment.
