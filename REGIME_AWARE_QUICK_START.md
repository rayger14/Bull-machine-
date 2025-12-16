# Regime-Aware Optimization Quick Start Guide

**Goal:** Optimize archetype thresholds per regime in 30 minutes.

---

## Prerequisites

1. Feature data with `regime_label` column
2. Regime classifier trained (`models/regime_classifier_gmm.pkl`)
3. Python environment with Optuna

**Check regime labels exist:**
```bash
python -c "import pandas as pd; df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet'); print('regime_label' in df.columns)"
```

If `False`, run:
```bash
python bin/quick_add_regime_labels.py
```

---

## Workflow

### Step 1: Optimize S1 (Liquidity Vacuum) - Crisis Archetype

**Optimizes S1 separately on risk_off and crisis bars.**

```bash
cd /Users/raymondghandchi/Bull-machine-/Bull-machine-

python bin/optimize_s1_regime_aware.py
```

**Output:**
- `configs/s1_regime_aware_v1.json` - Config with per-regime thresholds
- `results/s1_regime_aware_optimization_results.json` - Detailed metrics
- `optuna_s1_*_regime_aware.db` - Optuna study databases

**Expected Runtime:** 15-30 minutes (200 trials × 2 regimes)

**Expected Results:**
- Crisis PF: >2.5
- Risk_off PF: >2.0
- Event recall: ≥80% (LUNA, FTX)

---

### Step 2: Optimize S4 (Funding Divergence) - Short Squeeze

**Optimizes S4 on risk_off and neutral bars.**

```bash
python bin/optimize_archetype_regime_aware.py \
  --archetype funding_divergence \
  --regimes risk_off neutral \
  --n-trials 200
```

**Output:**
- `configs/funding_divergence_regime_aware_v1.json`
- `results/funding_divergence_regime_aware_optimization_results.json`

**Expected Results:**
- Risk_off PF: >2.0
- Neutral PF: >1.8
- Trade frequency: 6-10/year

---

### Step 3: Optimize S5 (Long Squeeze) - Positive Funding Cascade

**Optimizes S5 on risk_on and neutral bars.**

```bash
python bin/optimize_archetype_regime_aware.py \
  --archetype long_squeeze \
  --regimes risk_on neutral \
  --n-trials 200
```

**Output:**
- `configs/long_squeeze_regime_aware_v1.json`
- `results/long_squeeze_regime_aware_optimization_results.json`

**Expected Results:**
- Risk_on PF: >1.8
- Neutral PF: >1.6
- Trade frequency: 10-20/year

---

### Step 4: Walk-Forward Validation (Optional but Recommended)

**Validates that optimized parameters generalize to OOS data.**

```bash
python bin/walk_forward_regime_aware.py \
  --archetype liquidity_vacuum \
  --regimes risk_off crisis \
  --train-days 180 \
  --test-days 60 \
  --n-trials 100
```

**Output:**
- `results/walk_forward_liquidity_vacuum_regime_aware.json`

**Key Metric:** OOS Consistency (train/test PF correlation)
- \>0.6: Excellent generalization
- 0.4-0.6: Acceptable
- <0.4: Overfitting detected

**Expected Runtime:** 30-60 minutes (6 windows × 100 trials)

---

### Step 5: Portfolio Optimization

**Optimizes weights across archetypes accounting for regime distribution.**

```bash
python bin/optimize_portfolio_regime_weighted.py \
  --archetypes liquidity_vacuum funding_divergence long_squeeze \
  --start-date 2022-01-01 \
  --end-date 2023-12-31
```

**Output:**
- `results/portfolio_weights_regime_aware.json`

**Example Output:**
```json
{
  "weights": {
    "liquidity_vacuum": 0.25,
    "funding_divergence": 0.20,
    "long_squeeze": 0.30
  },
  "expected_pf": 2.45,
  "regime_coverage": {
    "risk_on": 0.30,
    "neutral": 0.50,
    "risk_off": 0.45,
    "crisis": 0.25
  }
}
```

---

## Understanding the Output

### Config Structure

**Before (Static):**
```json
{
  "liquidity_vacuum": {
    "fusion_threshold": 0.45,
    "liquidity_max": 0.15
  }
}
```

**After (Regime-Aware):**
```json
{
  "liquidity_vacuum": {
    "allowed_regimes": ["risk_off", "crisis"],
    "fusion_threshold": 0.45,
    "liquidity_max": 0.15,
    "regime_thresholds": {
      "risk_off": {
        "fusion_threshold": 0.48,
        "liquidity_max": 0.18
      },
      "crisis": {
        "fusion_threshold": 0.42,
        "liquidity_max": 0.12
      }
    }
  }
}
```

**Hierarchy:**
1. Try `regime_thresholds[current_regime][param]` (most specific)
2. Fallback to `thresholds[param]` (base)
3. Use hardcoded default (last resort)

---

## Validation Checklist

After optimization, verify:

✅ **Regime Distribution:**
```bash
python -c "
import pandas as pd
from bin.backtest_regime_stratified import get_regime_distribution

df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')
df_2022 = df[(df.index >= '2022-01-01') & (df.index < '2023-01-01')]
print(get_regime_distribution(df_2022))
"
```

Expected:
- risk_off: 30-40%
- crisis: 5-10%
- neutral: 20-30%
- risk_on: 30-40%

✅ **Event Recall (S1):**
- LUNA (2022-05-12): Trade within ±48h
- Capitulation (2022-06-18): Trade within ±48h
- FTX (2022-11-09): Trade within ±48h

Target: ≥66.7% (2/3 events)

✅ **OOS Consistency:**
- Run walk-forward validation
- Check `oos_consistency` in results JSON
- Target: ≥0.6

✅ **Regime Routing:**
```python
from engine.archetypes.logic_v2_adapter import ARCHETYPE_REGIMES

print(ARCHETYPE_REGIMES['liquidity_vacuum'])
# Should print: ['risk_off', 'crisis']
```

---

## Troubleshooting

### Error: "Data missing regime_label"

**Solution:**
```bash
python bin/quick_add_regime_labels.py
```

This adds `regime_label` column to feature data.

---

### Error: "Insufficient regime bars"

**Cause:** Some regimes have <500 bars in date range.

**Solution:** Expand date range or reduce `min_bars_per_regime`:
```bash
python bin/optimize_archetype_regime_aware.py \
  --archetype liquidity_vacuum \
  --regimes risk_off crisis \
  --train-start 2021-01-01 \
  --train-end 2023-12-31
```

---

### Warning: "OOS consistency <0.4"

**Cause:** Overfitting detected. Parameters don't generalize to test data.

**Solutions:**
1. Increase regularization (wider search space)
2. Reduce n_trials (less optimization pressure)
3. Use simpler model (fewer parameters)
4. Collect more training data

---

### Low Event Recall (<66%)

**Cause:** S1 thresholds too conservative for crisis events.

**Solutions:**
1. Lower `fusion_threshold` in crisis regime
2. Lower `liquidity_max` (allow lower liquidity)
3. Increase `volume_z_min` (require more panic)
4. Check ground truth dates are correct

---

## Performance Expectations

### S1 (Liquidity Vacuum)

**2022 Backtest (crisis + risk_off bars only):**
- Profit Factor: 2.2 - 2.8
- Win Rate: 50-60%
- Event Recall: 66-100%
- Trades/Year: 10-15
- Sharpe Ratio: 1.5-2.2

**Ground Truth Events:**
- 2022-05-12: LUNA death spiral → violent 25% bounce
- 2022-06-18: Market capitulation → explosive reversal
- 2022-11-09: FTX collapse → 15% bounce in 48h

---

### S4 (Funding Divergence)

**2022-2023 Backtest (risk_off + neutral):**
- Profit Factor: 1.8 - 2.5
- Win Rate: 50-55%
- Trades/Year: 6-10
- Sharpe Ratio: 1.3-1.8

**Pattern:** Negative funding → shorts overcrowded → squeeze UP

---

### S5 (Long Squeeze)

**2024 Backtest (risk_on + neutral):**
- Profit Factor: 1.6 - 2.2
- Win Rate: 55-62%
- Trades/Year: 10-20
- Sharpe Ratio: 1.4-1.9

**Pattern:** Positive funding extreme → longs overcrowded → cascade DOWN

---

## Next Steps

After optimization:

1. **Review Results:**
   - Check `results/*.json` files
   - Verify PF, WR, event recall meet targets

2. **Deploy to Backtest:**
   - Copy configs to production directory
   - Run full backtest on 2022-2024 data
   - Compare vs static thresholds

3. **Monitor Live:**
   - Track regime classification accuracy
   - Log threshold selection per trade
   - Compare live PF vs historical

4. **Iterate:**
   - Re-optimize quarterly with new data
   - Adjust regimes if macro shifts
   - Expand to more archetypes

---

## Example: Full Pipeline (Copy-Paste)

```bash
# 1. Optimize S1
python bin/optimize_s1_regime_aware.py

# 2. Optimize S4
python bin/optimize_archetype_regime_aware.py \
  --archetype funding_divergence \
  --regimes risk_off neutral \
  --n-trials 200

# 3. Optimize S5
python bin/optimize_archetype_regime_aware.py \
  --archetype long_squeeze \
  --regimes risk_on neutral \
  --n-trials 200

# 4. Walk-forward validation (S1)
python bin/walk_forward_regime_aware.py \
  --archetype liquidity_vacuum \
  --regimes risk_off crisis \
  --n-trials 100

# 5. Portfolio optimization
python bin/optimize_portfolio_regime_weighted.py \
  --archetypes liquidity_vacuum funding_divergence long_squeeze

# 6. Review results
cat results/s1_regime_aware_optimization_results.json
cat results/portfolio_weights_regime_aware.json
```

**Total Runtime:** 1-2 hours

---

## Philosophy

**"Only optimize what you can trade. Only test where you can profit."**

- S1 learns ONLY from crisis/risk_off (not from bull markets)
- S5 learns ONLY from risk_on/neutral (not from bear markets)
- Portfolio weighted by ACTUAL regime distribution

This is regime-aware intelligence. This is the Bull Machine's learning cortex.

---

**Questions? Issues?**

Check `/Users/raymondghandchi/Bull-machine-/Bull-machine-/REGIME_AWARE_OPTIMIZATION_IMPLEMENTATION.md` for full documentation.
