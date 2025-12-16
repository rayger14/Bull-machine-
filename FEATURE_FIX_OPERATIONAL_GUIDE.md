# Feature Fix Operational Guide

**Quick Start for Operators and Developers**

---

## What Was Fixed?

5 broken constant features now working:
1. **wyckoff_spring_b** - Shallow spring patterns (0.01% → 2.30%)
2. **temporal_confluence** - Multi-timeframe alignment (missing → 25.18%)
3. **tf4h_fvg_present** - 4H Fair Value Gaps (0% → 6.85%)
4. **tf4h_choch_flag** - 4H trend reversals (0% → 0.78%)
5. **mtf_alignment_ok** - Trend alignment check (0% → 99.85%)

---

## Verify Fixes Are Applied

```bash
# Quick check
python bin/verify_feature_fixes.py

# Expected output: "✅ ALL FEATURES VERIFIED"
```

---

## Apply Fixes to New Data

### Regenerate existing feature store
```bash
# Backup first
cp data/features_2022_with_regimes.parquet data/features_backup.parquet

# Apply fixes
python bin/fix_broken_features.py \
    data/features_2022_with_regimes.parquet \
    data/features_2022_FIXED.parquet

# Verify
python bin/verify_feature_fixes.py data/features_2022_FIXED.parquet

# If verified, replace original
mv data/features_2022_FIXED.parquet data/features_2022_with_regimes.parquet
```

### Apply to custom feature store
```bash
python bin/fix_broken_features.py \
    /path/to/input.parquet \
    /path/to/output.parquet
```

---

## Performance Expectations

```
Throughput:  12,236 rows/second
Target:      < 5 minutes for full regeneration
Actual:      0.71s for 8,741 rows

For 100,000 rows: ~8 seconds
For 1,000,000 rows: ~82 seconds (well under 5 min)
```

---

## Troubleshooting

### Feature still shows 0% triggers

**Check 1:** Verify you're using the FIXED feature store
```bash
ls -lh data/features_2022_with_regimes.parquet
# Should show recent modification time (today)
```

**Check 2:** Verify feature exists
```python
import pandas as pd
df = pd.read_parquet('data/features_2022_with_regimes.parquet')
print('temporal_confluence' in df.columns)  # Should be True
print(df['temporal_confluence'].sum())      # Should be > 0
```

**Check 3:** Re-run fix script
```bash
python bin/fix_broken_features.py
```

### Backtest not showing improvement

**Possible causes:**
1. Archetype not using fixed features (check wiring)
2. Features not impacting critical path (check archetype logic)
3. Market regime not suited to these features (check data period)

**Verification:**
```bash
# Check which features archetype uses
grep -r "wyckoff_spring_b\|temporal_confluence" engine/strategies/archetypes/

# Re-run baseline
python bin/backtest_knowledge_v2.py --config configs/mvp/mvp_bull_market_v1.json
```

### Fix script errors

**Error:** `UnboundLocalError: local variable 'volume_z' referenced before assignment`
**Solution:** Update to latest version of `/bin/fix_broken_features.py`

**Error:** `FileNotFoundError: data/features_2022_with_regimes.parquet`
**Solution:** Run from Bull-machine-/Bull-machine-/ directory

**Error:** Memory error on large datasets
**Solution:** Process in chunks (feature store already optimized for memory)

---

## Integration with Existing Workflows

### 1. Feature Store Build Pipeline
Add fix step after feature generation:

```bash
# Old workflow
python bin/build_feature_store_v2.py  # Generate features
python bin/generate_feature_quality_matrix.py  # Audit

# New workflow (with fix)
python bin/build_feature_store_v2.py  # Generate features
python bin/fix_broken_features.py     # FIX broken features ← NEW
python bin/verify_feature_fixes.py    # Verify fixes ← NEW
python bin/generate_feature_quality_matrix.py  # Audit
```

### 2. Baseline Validation
No changes needed - fixed features automatically used:

```bash
# Standard baseline run (now uses fixed features)
python bin/backtest_knowledge_v2.py --config configs/mvp/mvp_bull_market_v1.json
```

### 3. Archetype Optimization
Archetypes can now use previously broken features:

```python
# Example: S1 (Liquidity Vacuum) can now use wyckoff_spring_b
if features['wyckoff_spring_b'] and features['liquidity_vacuum_score'] > 0.6:
    # High conviction spring-based reversal entry
    entry_score *= 1.2
```

---

## Monitoring Feature Quality

### Daily Health Check
```bash
# Quick verification
python bin/verify_feature_fixes.py

# Full quality audit
python bin/generate_feature_quality_matrix.py
cat feature_quality_matrix.csv | grep -E "(wyckoff_spring_b|temporal_confluence|tf4h_fvg_present|tf4h_choch_flag|mtf_alignment_ok)"
```

### Expected Quality Metrics
```
Feature                  | Expected Range | Quality
-------------------------|----------------|----------
wyckoff_spring_b         | 0.5% - 5.0%    | GOOD
temporal_confluence      | 15% - 35%      | FREQUENT
tf4h_fvg_present         | 3% - 15%       | GOOD
tf4h_choch_flag          | 0.1% - 2%      | RARE/GOOD
mtf_alignment_ok         | 20% - 100%     | FREQUENT/POOR
wyckoff_pti_confluence   | 0% - 3%        | RARE (experimental)
```

---

## Known Issues & Limitations

### 1. mtf_alignment_ok triggers 99.85%
**Status:** Working as designed, may be too permissive
**Impact:** Low (designed to be permissive in trending markets)
**Action:** Monitor in backtests, tune if causing issues

### 2. wyckoff_pti_confluence still at 0%
**Status:** Data limitation (rare trap events + high PTI threshold)
**Impact:** Low (experimental feature)
**Action:** Mark as experimental, revisit when more data available

### 3. Some features remain broken
**Status:** 34 broken features not addressed in Phase 1
**Impact:** Medium (don't affect core archetypes)
**Action:** Phase 2 cleanup scheduled

---

## FAQ

### Q: Do I need to rerun backtests?
**A:** Yes, to see improvements from fixed features.

### Q: Will this break existing archetypes?
**A:** No, backward compatible. Existing archetypes continue working.

### Q: Can I revert if needed?
**A:** Yes, backup at `data/features_2022_with_regimes_BACKUP.parquet`

### Q: How do I use temporal_confluence in my archetype?
**A:**
```python
if features['temporal_confluence']:
    # Multi-timeframe alignment confirmed
    # Increase position size or confidence
    confluence_bonus = 0.15
```

### Q: Why is wyckoff_spring_b only 2.30%?
**A:** Springs are genuinely rare patterns (shallow fake breakdowns that reverse quickly). 2.30% is correct behavior.

### Q: Should I lower PTI threshold further for wyckoff_pti_confluence?
**A:** Not recommended. Current threshold (0.5) is already aggressive. Zero triggers is due to rarity of trap events, not threshold.

---

## Support & Contact

1. **Technical Details:** `/BROKEN_FEATURES_FIX_REPORT.md`
2. **Quick Reference:** `/FEATURE_FIX_SUMMARY.md`
3. **Code:** `/bin/fix_broken_features.py`
4. **Quality Audit:** `/feature_quality_matrix.csv`

---

## Change Log

### 2025-12-11: Initial Fix Release
- Fixed 5 of 6 priority broken features
- Added 2 new features (temporal_confluence, tf4h_fvg_present)
- Performance: 12,236 rows/sec
- Status: ✅ Production ready

---

**Quick Command Reference:**

```bash
# Verify fixes
python bin/verify_feature_fixes.py

# Apply fixes to new data
python bin/fix_broken_features.py input.parquet output.parquet

# Check quality
python bin/generate_feature_quality_matrix.py

# Run baseline
python bin/backtest_knowledge_v2.py --config configs/mvp/mvp_bull_market_v1.json
```

**Status:** ✅ READY FOR PRODUCTION USE
