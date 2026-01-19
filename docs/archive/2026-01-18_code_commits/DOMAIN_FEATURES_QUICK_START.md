# Domain Features Quick Start Guide

**Phase 3: Domain Features Backfill**

Quick reference for using the domain features implementation.

---

## Quick Test

```bash
# Test implementation (5 seconds)
python3 bin/backfill_domain_features_full.py --test

# Quality validation (10 seconds)
python3 bin/test_domain_features_quality.py
```

**Expected Output:**
```
✅ Test completed in 0.15s
   Total features: 72
   Rows processed: 1,000
   Speed: 6617 rows/sec
```

---

## Integration (Copy-Paste Ready)

```python
from bin.backfill_domain_features_full import (
    compute_wyckoff_features,
    compute_smc_features,
    compute_liquidity_features,
    compute_funding_oi_features,
    compute_temporal_features
)

# Your backfill pipeline
df = pd.read_parquet("your_data.parquet")

# Add domain features (69 total)
df = compute_wyckoff_features(df)      # 33 features
df = compute_smc_features(df)          # 12 features
df = compute_liquidity_features(df)    # 5 features
df = compute_funding_oi_features(df)   # 10 features
df = compute_temporal_features(df)     # 9 features (depends on Wyckoff)

# Save
df.to_parquet("data_with_domain_features.parquet")
```

---

## Feature Inventory (69 features)

### Wyckoff (33)

**Events (12):** wyckoff_spring, wyckoff_utad, wyckoff_ar, wyckoff_bc, wyckoff_st, wyckoff_sos, wyckoff_sof, wyckoff_lps, wyckoff_lpsy, wyckoff_ps, wyckoff_as, wyckoff_ut

**Confidence (8):** wyckoff_spring_confidence, wyckoff_utad_confidence, wyckoff_ar_confidence, wyckoff_bc_confidence, wyckoff_st_confidence, wyckoff_ps_confidence, wyckoff_as_confidence, wyckoff_ut_confidence

**PTI (6):** wyckoff_pti_accumulation, wyckoff_pti_distribution, wyckoff_pti_markup, wyckoff_pti_markdown, wyckoff_pti_confluence, wyckoff_pti_reversal

**Phase (1):** wyckoff_phase_abc

**Temporal (8):** bars_since_spring, bars_since_utad, bars_since_ar, bars_since_bc, bars_since_st, bars_since_ps, bars_since_sc, bars_since_sos_long

**Volume (1):** volume_climax_last_3b

### SMC (12)

smc_bos, smc_choch, smc_demand_zone, smc_supply_zone, smc_liquidity_sweep, ob_confidence, ob_strength_bullish, ob_strength_bearish, hob_demand_zone, hob_supply_zone, hob_imbalance, smc_score

### Liquidity (5)

liquidity_score, liquidity_drain_pct, liquidity_velocity, liquidity_vacuum_score, liquidity_vacuum_fusion

### Funding/OI (10)

funding_rate, funding_extreme, funding_flip, funding_reversal, funding_stress_ewma, oi_z, oi_delta_1h_z, oi_change_24h, oi_change_pct_24h, oi_cascade

### Temporal (9)

bars_since_sos_short, temporal_support_cluster, temporal_resistance_cluster, temporal_confluence

---

## Performance Expectations

**1,000 rows:** 0.15 seconds (6,600 rows/sec)
**35,000 rows:** ~5-6 seconds (target: <20 minutes)
**Memory:** <1GB for 35K rows

---

## Troubleshooting

### Issue: "ufunc not supported" error
**Fix:** Update validation to skip string columns (already fixed in latest version)

### Issue: NaN values in features
**Check:** Do you have OHLCV columns? (open, high, low, close, volume)
**Fix:** Ensure data has no gaps and proper OHLC logic

### Issue: No events detected
**Check:** Is your data realistic? (prices, volumes)
**Fix:** Thresholds tuned for BTC ~$40K, adjust if needed

### Issue: bars_since not monotonic
**Check:** Are event columns binary (0 or 1)?
**Fix:** Ensure event detection returns 0.0 or 1.0

---

## Feature Ranges (Sanity Check)

| Feature Type | Expected Range | Notes |
|--------------|----------------|-------|
| Binary events | 0 or 1 | wyckoff_spring, smc_choch, etc. |
| Confidence scores | 0 to 1 | All *_confidence, *_score features |
| bars_since_* | 0 to 999 | 0 = event just occurred, 999 = no event yet |
| PTI scores | 0 to 1 | Composite psychological indicators |
| funding_rate | -0.1 to 0.1 | Proxy (real data: -0.01 to 0.01) |
| liquidity_drain_pct | -1 to 1 | Negative = draining |
| liquidity_velocity | ~0 | Should average near 0 |
| smc_bos | -1, 0, or 1 | -1=bearish, 0=none, 1=bullish |

---

## Quick Validation

```python
# After computing features, validate:
assert df['wyckoff_spring'].isin([0, 1]).all()
assert df['wyckoff_pti_accumulation'].between(0, 1).all()
assert df['bars_since_spring'].between(0, 999).all()
assert df['smc_bos'].isin([-1, 0, 1]).all()
assert df['funding_rate'].between(-0.1, 0.1).all()

print("✅ All features validated!")
```

---

## Common Patterns

### Check for Wyckoff Springs

```python
springs = df[df['wyckoff_spring'] > 0]
print(f"Found {len(springs)} spring events")
print(springs[['wyckoff_spring', 'wyckoff_spring_confidence', 'bars_since_spring']].head())
```

### Find High Liquidity Vacuums

```python
vacuums = df[df['liquidity_vacuum_fusion'] > 0.6]
print(f"Found {len(vacuums)} high-vacuum events")
```

### Detect BOS Clusters

```python
bos_events = df[df['smc_bos'] != 0]
print(f"BOS events: {len(bos_events)} ({len(bos_events)/len(df)*100:.1f}%)")
```

### Temporal Confluence

```python
high_confluence = df[df['temporal_confluence'] > 0.5]
print(f"High timing confluence: {len(high_confluence)} bars")
```

---

## Files

- **bin/backfill_domain_features_full.py** - Main implementation
- **bin/test_domain_features_quality.py** - Quality tests
- **DOMAIN_FEATURES_IMPLEMENTATION_REPORT.md** - Full documentation

---

## Support

**Issues?** Check DOMAIN_FEATURES_IMPLEMENTATION_REPORT.md for:
- Detailed implementation logic
- Validation results
- Known limitations
- Edge cases

**Questions?** All implementations are vectorized pandas operations - check the source code for logic.
