# S1 & S4 Quick Reference Card

**One-page cheat sheet for operators**

---

## S1: Liquidity Vacuum Reversal

**Pattern:** Capitulation bounce when sellers exhaust + orderbook vacuum
**Bias:** Long (buy the panic)
**Frequency:** 40-60 trades/year (production), 10-15 ideal
**Win Rate:** 50-60%

### Minimum Feature Requirements
```
MUST HAVE:
✓ OHLCV data (open, high, low, close)
✓ liquidity_score (feature store)
✓ volume_zscore (feature store)
✓ Runtime enrichment enabled

SHOULD HAVE (V2 mode):
✓ VIX_Z, DXY_Z (macro)
✓ funding_Z (funding rates)
✓ regime_label (classifier)

COMPUTED ON-DEMAND:
✓ 12 runtime features (auto-calculated)
```

### Detection Logic (V2 Confluence)
```
STEP 1: Regime Filter (optional)
  regime in ['risk_off', 'crisis'] OR drawdown > 10%

STEP 2: Hard Gates
  capitulation_depth < -20%
  crisis_composite >= 0.35

STEP 3: Confluence (3-of-4)
  ✓ Capitulation depth check
  ✓ Crisis environment check
  ✓ Volume climax (3-bar max > 0.50)
  ✓ Wick exhaustion (3-bar max > 0.60)

STEP 4: Weighted Score
  confluence_score >= 0.65
```

### Key Thresholds (Production)
```json
{
  "capitulation_depth_max": -0.20,
  "crisis_composite_min": 0.35,
  "confluence_min_conditions": 3,
  "confluence_threshold": 0.65,
  "volume_climax_3b_min": 0.50,
  "wick_exhaustion_3b_min": 0.60
}
```

### Integration Code
```python
from engine.strategies.archetypes.bear.liquidity_vacuum_runtime import (
    apply_liquidity_vacuum_enrichment
)

# BEFORE backtest
df = apply_liquidity_vacuum_enrichment(df, lookback=24)
```

### Common Issues
1. **"0 trades in 2023"** → CORRECT (bull market, no capitulations)
2. **"Missed FTX"** → Lower `crisis_composite_min` to 0.35 (from 0.40)
3. **"Too many trades"** → Raise `confluence_threshold` to 0.70
4. **"Missing V2 features"** → Check runtime enrichment ran
5. **"Liquidity paradox"** → Use `liquidity_drain_pct` not absolute

---

## S4: Funding Divergence (Short Squeeze)

**Pattern:** Overcrowded shorts + price strength = violent squeeze UP
**Bias:** Long (fade the shorts)
**Frequency:** 6-10 trades/year
**Profit Factor:** > 2.0

### Minimum Feature Requirements
```
MUST HAVE:
✓ close (OHLCV)
✓ funding_rate or funding_Z (feature store)
✓ liquidity_score (feature store)
✓ Runtime enrichment enabled

SHOULD HAVE:
✓ volume_zscore (for volume_quiet)

COMPUTED ON-DEMAND:
✓ 4 runtime features (auto-calculated)
```

### Detection Logic
```
STEP 1: Negative Funding Extreme
  funding_z < -1.2 (shorts overcrowded)

STEP 2: Thin Liquidity
  liquidity_score < 0.30 (amplifies squeeze)

STEP 3: Price Resilience
  price_resilience >= 0.5 (price NOT falling)

STEP 4: Fusion Score
  s4_fusion_score >= 0.40 (default)
  s4_fusion_score >= 0.78 (optimized)
```

### Key Thresholds (Optimized)
```json
{
  "funding_z_max": -1.976,
  "resilience_min": 0.555,
  "liquidity_max": 0.348,
  "fusion_threshold": 0.7824
}
```

### Integration Code
```python
from engine.strategies.archetypes.bear.funding_divergence_runtime import (
    apply_s4_enrichment
)

# BEFORE backtest
df = apply_s4_enrichment(
    df,
    funding_lookback=24,
    price_lookback=12
)
```

### Common Issues
1. **"Funding sign wrong"** → Threshold is NEGATIVE (-1.2, not +1.2)
2. **"No trades"** → Check funding_z available (derivatives data)
3. **"Too sensitive"** → Raise `fusion_threshold` to 0.60-0.80
4. **"Resilience always fails"** → Check `price_lookback=12` (not too long)

---

## Feature Priority Matrix

### S1 Features (Importance Ranking)
| Rank | Feature | Impact | Source | Missing = |
|------|---------|--------|--------|-----------|
| 1 | `liquidity_score` | CRITICAL | Store | Pattern fails |
| 2 | `capitulation_depth` | CRITICAL | Runtime | V2→V1 fallback |
| 3 | `crisis_composite` | CRITICAL | Runtime | V2→V1 fallback |
| 4 | `volume_climax_last_3b` | HIGH | Runtime | Confluence signal |
| 5 | `wick_exhaustion_last_3b` | HIGH | Runtime | Confluence signal |
| 6 | `volume_zscore` | HIGH | Store | V1 fallback only |
| 7 | `liquidity_drain_pct` | MEDIUM | Runtime | Bonus signal |
| 8 | `funding_Z` | LOW | Store | Optional boost |
| 9 | `VIX_Z`, `DXY_Z` | LOW | Macro | Optional context |
| 10 | `regime_label` | LOW | Classifier | Optional filter |

### S4 Features (Importance Ranking)
| Rank | Feature | Impact | Source | Missing = |
|------|---------|--------|--------|-----------|
| 1 | `funding_Z` or `funding_rate` | CRITICAL | Store | Pattern fails |
| 2 | `liquidity_score` | CRITICAL | Store | Pattern fails |
| 3 | `funding_z_negative` | CRITICAL | Runtime | Gate 1 fails |
| 4 | `price_resilience` | CRITICAL | Runtime | Gate 3 fails |
| 5 | `close` | HIGH | OHLCV | Resilience calc fails |
| 6 | `volume_quiet` | MEDIUM | Runtime | Bonus signal |
| 7 | `volume_zscore` | MEDIUM | Store | Volume quiet calc |
| 8 | `s4_fusion_score` | LOW | Runtime | Auto-calculated |

---

## Shared Features (Both Patterns)

| Feature | S1 Usage | S4 Usage | Priority |
|---------|----------|----------|----------|
| `close` | Drawdown calc | Price resilience | CRITICAL |
| `liquidity_score` | Vacuum detection | Thin orderbook | CRITICAL |
| `volume_zscore` | Panic selling | Volume quiet | HIGH |
| `funding_Z` | Optional boost | **Primary signal** | S4: CRITICAL, S1: LOW |
| `atr_20` | Position sizing | Position sizing | MEDIUM |
| `regime_label` | Regime filter | Regime routing | LOW |

---

## Configuration Files

### S1 Configs
```
Production (Recommended):
  configs/s1_v2_production.json
    - Confluence mode enabled
    - Regime filter enabled
    - Validated: 60.7 trades/year

Variations:
  configs/s1_v2_production_confluence.json  (same as above)
  configs/s1_v2_quick_fix.json              (dev testing)
  configs/s1_regime_aware_example.json      (regime demo)
  configs/s1_test_relaxed.json              (exploration)
```

### S4 Configs
```
Production (Recommended):
  configs/s4_optimized_oos_test.json
    - Optimized thresholds
    - 2023 H1 validation
    - Stricter fusion_threshold (0.78)

Variations:
  configs/s4_optimized_oos_2023h2.json      (2023 H2)
  configs/s4_optimized_oos_2024.json        (2024)
  configs/s4_test_relaxed.json              (exploration)
```

---

## Runtime Enrichment Details

### S1 Runtime Features (12 total)
```python
# V1 Features (4) - Legacy
wick_lower_ratio          # Lower wick as % of candle range
liquidity_vacuum_score    # Inverted liquidity (low = high vacuum)
volume_panic              # Volume z-score normalized [0,1]
crisis_context            # VIX + DXY composite
liquidity_vacuum_fusion   # V1 weighted score (DEPRECATED)

# V2 Features (7) - Multi-bar capitulation
liquidity_drain_pct       # KEY FIX - Relative vs 7d avg
liquidity_velocity        # Speed of drain
liquidity_persistence     # Consecutive bars in drain
capitulation_depth        # Drawdown from 30d high (CRITICAL)
crisis_composite          # Enhanced macro score (CRITICAL)
volume_climax_last_3b     # Max volume in 3-bar window
wick_exhaustion_last_3b   # Max wick in 3-bar window
```

### S4 Runtime Features (4 total)
```python
funding_z_negative   # Rolling z-score of funding (24h)
price_resilience     # Price strength vs funding expectation (12h)
volume_quiet         # Boolean: volume_zscore < -0.5
s4_fusion_score      # Weighted combination of above
```

---

## Validation Quick Checks

### S1 Health Checks
```python
# After enrichment
assert 'capitulation_depth' in df.columns      # V2 mode
assert 'crisis_composite' in df.columns        # V2 mode
assert 'volume_climax_last_3b' in df.columns  # V2 mode
assert 'wick_exhaustion_last_3b' in df.columns # V2 mode

# Stats
print((df['capitulation_depth'] < -0.20).sum())  # Should have some
print((df['crisis_composite'] > 0.35).sum())     # Should have some
```

### S4 Health Checks
```python
# After enrichment
assert 'funding_z_negative' in df.columns
assert 'price_resilience' in df.columns
assert 'volume_quiet' in df.columns
assert 's4_fusion_score' in df.columns

# Stats
print((df['funding_z_negative'] < -1.2).sum())  # Should have some
print((df['price_resilience'] > 0.5).sum())     # Should have some
```

---

## Performance Expectations

### S1 Expected Results (2022-2024 backtest)
```
2022 (Bear):    ~40-50 trades (capitulation year)
2023 (Bull):    0-5 trades    (CORRECT - no capitulations)
2024 (Mixed):   ~10-15 trades (flash crashes)

Win Rate:       50-60%
Sharpe:         Varies (concentrated in events)
Max DD:         Moderate (reversal pattern)

Major Events Caught (2022):
  ✓ LUNA May-12     (-80% → +25%)
  ✓ LUNA Jun-18     (Final capitulation)
  ✓ FTX Nov-9       (Exchange collapse)
  ✓ Multiple smaller events
```

### S4 Expected Results
```
Frequency:      6-10 trades/year
PF:             > 2.0
Edge:           Violent short squeezes

Regime Performance:
  risk_off:     Best (overcrowded shorts)
  neutral:      Good (occasional setups)
  risk_on:      Reduced weight (less shorts)
  crisis:       Boosted (panic shorts)
```

---

## Troubleshooting

### S1 Problems

**Problem:** "No trades in 2023"
**Solution:** Expected behavior (bull market, no capitulations)

**Problem:** "Too many trades (>100/year)"
**Solution:**
1. Check `confluence_threshold` (raise to 0.70)
2. Check `confluence_min_conditions` (set to 3)
3. Enable regime filter (`use_regime_filter: true`)

**Problem:** "Missed FTX crash"
**Solution:** Lower `crisis_composite_min` to 0.35 (FTX = 0.34)

**Problem:** "Runtime features not found"
**Solution:** Call `apply_liquidity_vacuum_enrichment()` BEFORE backtest

**Problem:** "All trades failing regime filter"
**Solution:** Check `allowed_regimes` includes correct labels

---

### S4 Problems

**Problem:** "Funding check always fails"
**Solution:**
1. Verify `funding_z_max` is NEGATIVE (e.g., -1.2)
2. Check funding data available in period

**Problem:** "Price resilience always 0.5"
**Solution:** Runtime features not calculated, check enrichment

**Problem:** "Too sensitive (too many trades)"
**Solution:** Raise `fusion_threshold` to 0.60-0.80

**Problem:** "No trades at all"
**Solution:**
1. Check funding data exists
2. Verify `funding_z_max` not too strict (try -1.0)
3. Check `liquidity_max` not too tight (try 0.40)

---

## Optimization Tips

### S1 Parameter Tuning
```
Trade Frequency Control:
  ↓ confluence_threshold (0.60) → More trades
  ↑ confluence_threshold (0.70) → Fewer trades

Precision Control:
  ↑ volume_climax_3b_min (0.60) → Higher precision, fewer trades
  ↑ wick_exhaustion_3b_min (0.70) → Higher precision, fewer trades

Regime Sensitivity:
  allowed_regimes: ['risk_off', 'crisis', 'neutral'] → More trades
  allowed_regimes: ['risk_off'] → Only bear markets
```

### S4 Parameter Tuning
```
Trade Frequency Control:
  ↓ fusion_threshold (0.40) → More trades (default)
  ↑ fusion_threshold (0.78) → Fewer trades (optimized)

Funding Sensitivity:
  ↑ funding_z_max (-1.0) → More trades (relaxed)
  ↓ funding_z_max (-2.5) → Fewer trades (strict)

Divergence Strength:
  ↓ resilience_min (0.4) → More trades
  ↑ resilience_min (0.7) → Fewer trades
```

---

## Code Integration Template

### Complete S1 Integration
```python
import pandas as pd
from engine.strategies.archetypes.bear.liquidity_vacuum_runtime import (
    apply_liquidity_vacuum_enrichment
)

# 1. Load feature data
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')

# 2. Apply runtime enrichment
df_enriched = apply_liquidity_vacuum_enrichment(
    df,
    lookback=24,
    volume_lookback=24
)

# 3. Verify enrichment
print("Runtime features added:")
for col in df_enriched.columns:
    if col in ['capitulation_depth', 'crisis_composite',
               'volume_climax_last_3b', 'wick_exhaustion_last_3b']:
        print(f"  ✓ {col}")

# 4. Run backtest with S1 enabled
config = load_config('configs/s1_v2_production.json')
results = run_backtest(df_enriched, config)
```

### Complete S4 Integration
```python
import pandas as pd
from engine.strategies.archetypes.bear.funding_divergence_runtime import (
    apply_s4_enrichment
)

# 1. Load feature data
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')

# 2. Apply runtime enrichment
df_enriched = apply_s4_enrichment(
    df,
    funding_lookback=24,
    price_lookback=12,
    volume_lookback=24
)

# 3. Verify enrichment
print("Runtime features added:")
for col in df_enriched.columns:
    if col in ['funding_z_negative', 'price_resilience',
               'volume_quiet', 's4_fusion_score']:
        print(f"  ✓ {col}")

# 4. Run backtest with S4 enabled
config = load_config('configs/s4_optimized_oos_test.json')
results = run_backtest(df_enriched, config)
```

---

**END OF QUICK REFERENCE**
