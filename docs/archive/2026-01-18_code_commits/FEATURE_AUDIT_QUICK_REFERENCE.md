# Feature Audit Quick Reference Card

**Last Updated:** 2025-12-03
**Dataset:** BTC_1H_2022-01-01_to_2024-12-31.parquet

---

## Quick Status

| Metric | Value | Status |
|--------|-------|--------|
| **Total Features** | 167 columns | ✅ |
| **Total Rows** | 26,236 (3 years) | ✅ |
| **Complete Features** | 163/167 (97.6%) | ✅ |
| **Archetype Readiness** | 5/6 full, 1/6 partial | ✅ |
| **Can Start Development?** | **YES** | ✅ |

---

## Archetype Feature Checklist

```
✅ S1_FAILED_RALLY         - 100% ready (9/9 required, 3/3 optional)
✅ S2_TRAP_WITHIN_TREND    - 100% ready (7/7 required, 2/2 optional)
⚠️ S3_ORDER_BLOCK_RETEST   - 71% ready (5/7 required, 2/2 optional)*
✅ S4_BOS_CHOCH            - 100% ready (5/5 required, 2/2 optional)
⚠️ S5_LONG_SQUEEZE         - 80% ready (4/5 required, 2/2 optional)**
✅ CAPITULATION            - 100% ready (7/7 required, 2/2 optional)

* OB levels have expected nulls (33-36%) - not a blocker
** OI data only in 2024 - use fallback logic for 2022-2023
```

---

## Missing Features (can derive in 30 min)

```python
# 1. Volatility Spike (5 min)
volatility_spike = atr_14 > atr_14.rolling(20).mean() * 1.5

# 2. Oversold (10 min) - regime adaptive
oversold = rsi_14 < regime_thresholds[macro_regime]

# 3. Funding Reversal (10 min)
funding_reversal = (funding_Z.shift(1) < -1.5) & (funding_Z > -1.0)

# 4. Resilience (5 min)
resilience = (close - low) / (high - low)

# Add all 4 at once:
python3 bin/add_derived_features.py
```

---

## Critical Data Issues

### 🔴 OI Data Gap (known limitation)
- **2022-2023:** 0% coverage
- **2024:** 100% coverage
- **Impact:** S5 archetype limited for historical period
- **Solution:** Use fallback logic (documented in DATA_COVERAGE_NOTICE.md)

### 🟢 Funding Data (use correct column)
- ❌ `funding`: 33% coverage (don't use)
- ✅ `funding_rate`: 100% coverage (use this)
- ✅ `funding_Z`: 100% coverage (use this)

---

## Feature Categories

```
OHLCV:              5 features   (100% complete)
Technical:         11 features   (100% complete)
Derivatives:        8 features   (50% complete - OI issue)
Macro:             30 features   (99%+ complete)
Liquidity:          6 features   (100% complete)
Crisis:             5 features   (100% complete)
SMC:               19 features   (100% complete)
Wyckoff:           30 features   (100% complete)
Multi-Timeframe:   40 features   (100% complete)
Fusion:             6 features   (100% complete)
Volume Analysis:    4 features   (100% complete)
Other:              3 features   (100% complete)
```

---

## Key Features by Archetype

### S1 (Failed Rally)
```python
wyckoff_ut, wyckoff_utad, wyckoff_lpsy       # Wyckoff distribution
volume_climax_last_3b                        # Volume confirmation
funding_Z                                     # Funding pressure
liquidity_drain_pct                          # Liquidity context
tf4h_boms_direction, tf1h_bos_bearish        # Structure
```

### S2 (Trap Within Trend)
```python
wyckoff_lpsy, wyckoff_ut                     # Wyckoff events
tf1h_fakeout_detected, tf1h_pti_trap_type    # Trap detection
liquidity_drain_pct, funding_Z               # Context
```

### S3 (Order Block Retest)
```python
is_bearish_ob, ob_strength_bearish           # OB identification
tf1h_ob_high, tf1h_ob_low                    # OB levels
liquidity_velocity, funding_Z                # Context
```

### S4 (BOS/CHOCH)
```python
tf1h_bos_bearish, tf4h_choch_flag           # Structure breaks
liquidity_drain_pct, volume_climax_last_3b  # Confirmation
funding_Z                                    # Context
```

### S5 (Long Squeeze)
```python
funding_Z, oi_change_pct_24h                # Funding pressure
volume_climax_last_3b, wick_exhaustion_3b   # Panic signals
liquidity_drain_pct                         # Liquidity exodus
```

### Capitulation
```python
crisis_composite, capitulation_depth        # Crisis detection
wick_exhaustion_3b, volume_climax_3b        # Exhaustion
funding_Z, liquidity_drain_pct              # Market stress
macro_regime                                # Regime context
```

---

## Common Patterns

### Crisis Detection
```python
crisis_composite > 0.5          # Composite crisis score
capitulation_depth < -0.2       # Deep capitulation
wick_exhaustion_last_3b > 0.7   # Exhaustion wicks
volume_climax_last_3b > 0.5     # Volume climax
macro_regime == 'crisis'        # Regime confirmation
```

### Liquidity Stress
```python
liquidity_drain_pct > 0.5       # Significant drain
liquidity_velocity < -0.3       # Fast outflow
liquidity_persistence > 3       # Sustained drain
```

### Funding Extremes
```python
funding_Z < -2.0                # Extreme negative
funding_reversal == 1.0         # Reversal detected
```

### Wyckoff Distribution
```python
wyckoff_ut or wyckoff_utad      # Upthrust events
wyckoff_lpsy                    # Last point of supply
wyckoff_sow                     # Sign of weakness
```

---

## Data Quality Checks

### Before Running Backtests
```python
# Check date range
df.index.min(), df.index.max()  # Should be 2022-01-01 to 2024-12-31

# Check for unexpected nulls
critical_features = [
    'close', 'volume', 'rsi_14', 'atr_14',
    'crisis_composite', 'funding_Z', 'macro_regime'
]
df[critical_features].isnull().sum()  # Should all be 0

# Check OI availability for your period
if your_period >= '2024-01-01':
    oi_available = True  # Can use S5 fully
else:
    oi_available = False  # Use S5 fallback logic
```

### Expected Null Patterns
```python
# These are EXPECTED to have nulls:
tf1h_fvg_high/low        # 50-54% null (FVGs don't form every bar)
tf1h_ob_high/low         # 33-36% null (OBs don't form every bar)
oi, oi_change_*          # 67% null (2024 only)
funding                  # 67% null (use funding_rate instead)
```

---

## Sample Data Ranges

```python
# Price
close: ~15,000 to ~70,000 USD

# Technical
rsi_14: 0 to 100
atr_14: ~100 to ~3,000
adx_14: 0 to 100

# Crisis
crisis_composite: 0.0 to 0.86 (0.86 = Aug 5, 2024 crash)
capitulation_depth: -0.45 to 0.0 (more negative = deeper)
wick_exhaustion_3b: 0.0 to 1.0
volume_climax_3b: 0.0 to 1.0

# Liquidity
liquidity_drain_pct: -0.82 to 1.0
liquidity_velocity: -0.80 to 1.0
liquidity_persistence: 0 to 107 bars

# Funding
funding_Z: -12.5 to +14.6 (±2.0 is extreme)
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `FEATURE_STORE_AUDIT_REPORT.md` | Full 500+ line detailed audit |
| `FEATURE_AUDIT_EXECUTIVE_SUMMARY.md` | Executive summary |
| `DATA_COVERAGE_NOTICE.md` | OI data gap analysis |
| `FEATURE_AUDIT_QUICK_REFERENCE.md` | This file (quick reference) |
| `bin/add_derived_features.py` | Script to add 4 missing features |

---

## Next Steps

1. ✅ **Proceed with archetype development** - Feature store is ready
2. 🔧 **Add derived features** when convenient (30 min)
   ```bash
   python3 bin/add_derived_features.py
   ```
3. 📋 **Document OI limitation** in S5 archetype
4. 🧪 **Validate on 2024 data first** (full features)
5. 🔄 **Test fallback logic** on 2022-2023 (reduced features)

---

**Status: ✅ CLEARED FOR DEVELOPMENT**

All archetypes have sufficient features to begin implementation.
S5 requires period-aware logic but is not blocked.

---

*For detailed analysis, see `FEATURE_STORE_AUDIT_REPORT.md`*
*For OI data details, see `DATA_COVERAGE_NOTICE.md`*
