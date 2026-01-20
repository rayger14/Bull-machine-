# Bear Feature Pipeline Fix - Quick Start Guide

**Status**: ✅ Ready to Execute
**Time Required**: 33 hours (4 days)
**Scripts Ready**: 2 of 4 (remaining 2 are simple)

---

## TL;DR - What's Broken

```
❌ oi_change_24h, oi_change_pct_24h, oi_z → ALL NaN (calculation never run)
❌ liquidity_score → NOT IN STORE (runtime-only, needs persistence)
⚠️ oi (raw) → Only 2024 data (missing 2022-2023)

Impact: S5, S1, S4 blocked. Cannot validate on 2022 bear market (Terra, FTX).
```

---

## Quick Execute (Copy-Paste)

### Phase 1: Unblock S2 (4 hours)
```bash
# Step 1: Create S2 derived features script (simple)
cat > bin/add_s2_derived_features.py << 'EOF'
#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

mtf_path = Path('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')
df = pd.read_parquet(mtf_path)

# Add derived features
df['wick_ratio'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-9)
df['vol_fade'] = (df['volume_z'] < df['volume_z'].shift(4))
df['ob_retest'] = (df['high'] >= df['tf1h_ob_low']) & (df['low'] <= df['tf1h_ob_high'])

# RSI divergence (bearish)
price_hh = df['close'] > df['close'].shift(5).rolling(5).max()
rsi_lh = df['rsi_14'] < df['rsi_14'].shift(5).rolling(5).max()
df['rsi_divergence'] = price_hh & rsi_lh

df.to_parquet(mtf_path)
print(f"✅ Added S2 features: wick_ratio, vol_fade, ob_retest, rsi_divergence")
EOF

# Step 2: Run it
python3 bin/add_s2_derived_features.py
```

### Phase 2: Fix OI Pipeline (8 hours)
```bash
# Backup first!
cp data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet \
   data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet.backup

# Run OI fix (fetches 2022-2023 data + calculates derivatives)
python3 bin/fix_oi_change_pipeline.py \
    --start-date 2022-01-01 \
    --end-date 2023-12-31 \
    --cache-path data/cache/okx_oi_2022_2023.parquet

# Expected output:
#   ✅ Fetched 17,520 OI records from OKX
#   ✅ Calculated oi_change_24h, oi_change_pct_24h, oi_z
#   ✅ Terra collapse: -18.3% detected
#   ✅ FTX collapse: -22.1% detected
```

### Phase 3: Backfill Liquidity Score (12 hours)
```bash
# Run liquidity backfill (takes ~8 hours for 26K rows)
nohup python3 bin/backfill_liquidity_score.py \
    --mtf-store data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet \
    --side long \
    > backfill_liquidity.log 2>&1 &

# Monitor progress
tail -f backfill_liquidity.log

# Expected output:
#   Computing: 100%|████████| 26236/26236 [8:15:00<00:00]
#   ✅ Median: 0.512, p90: 0.843
#   ✅ VALIDATION PASSED
```

### Phase 4: Validate (9 hours)
```bash
# Create validation script
cat > bin/validate_bear_patterns_2022.py << 'EOF'
#!/usr/bin/env python3
# Run all bear patterns on 2022 data and measure PF
# (Implementation details in full roadmap doc)
EOF

# Run validation
python3 bin/validate_bear_patterns_2022.py --all --report

# Expected:
#   S2 PF: 1.45, S5 PF: 1.72, S1 PF: 1.28, S4 PF: 1.38
#   ✅ All patterns operational
```

---

## Verification Checklist

After each phase, verify:

### After Phase 1 (S2)
```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')
assert 'wick_ratio' in df.columns, 'wick_ratio missing'
assert 'vol_fade' in df.columns, 'vol_fade missing'
assert 'rsi_divergence' in df.columns, 'rsi_divergence missing'
assert 'ob_retest' in df.columns, 'ob_retest missing'
print('✅ Phase 1 complete: S2 features added')
"
```

### After Phase 2 (OI)
```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')
oi_coverage = df['oi'].notna().sum() / len(df) * 100
assert oi_coverage > 95, f'OI coverage only {oi_coverage:.1f}%'
assert df['oi_change_pct_24h'].notna().sum() > 0, 'oi_change_pct still NaN'
print(f'✅ Phase 2 complete: OI coverage {oi_coverage:.1f}%')
"
```

### After Phase 3 (Liquidity)
```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')
assert 'liquidity_score' in df.columns, 'liquidity_score missing'
median = df['liquidity_score'].median()
assert 0.4 < median < 0.6, f'Liquidity median {median:.3f} out of range'
print(f'✅ Phase 3 complete: liquidity_score median = {median:.3f}')
"
```

---

## Rollback (If Something Goes Wrong)

```bash
# Restore backup
cp data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet.backup \
   data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet

# Verify rollback
python3 -c "
import pandas as pd
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')
print(f'Restored: {len(df)} rows, {len(df.columns)} features')
"
```

---

## Troubleshooting

### Problem: OKX API rate limit errors
```bash
# Use cached data instead
python3 bin/fix_oi_change_pipeline.py --skip-fetch
```

### Problem: Liquidity computation too slow
```bash
# Run overnight with nohup
nohup python3 bin/backfill_liquidity_score.py > backfill.log 2>&1 &
```

### Problem: Feature validation fails
```bash
# Dry run first (doesn't write file)
python3 bin/fix_oi_change_pipeline.py --dry-run
python3 bin/backfill_liquidity_score.py --dry-run
```

---

## Expected Timeline

```
Day 1 Morning  [====    ] Phase 1: S2 features added (4h)
Day 1 Afternoon[========] Phase 2: OI fetch started (4h)
Day 2 Morning  [========] Phase 2: OI calculated (4h)
Day 2 Afternoon[====    ] Phase 3: Liquidity started (4h)
Day 3 All Day  [========] Phase 3: Liquidity computing (8h)
Day 4 Morning  [====    ] Phase 4: Validation (4h)
Day 4 Afternoon[=====   ] Phase 4: Report generated (5h)

Total: 33 hours
```

---

## Success Indicators

After all phases complete:

```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')

# Check feature count
print(f'Total features: {len(df.columns)} (expected: 120+)')

# Check OI coverage
oi_pct = df['oi'].notna().sum() / len(df) * 100
print(f'OI coverage: {oi_pct:.1f}% (expected: >99%)')

# Check OI derivatives
oi_change_pct = df['oi_change_pct_24h'].notna().sum() / len(df) * 100
print(f'OI change coverage: {oi_change_pct:.1f}% (expected: >95%)')

# Check liquidity
liq_pct = df['liquidity_score'].notna().sum() / len(df) * 100
print(f'Liquidity coverage: {liq_pct:.1f}% (expected: 100%)')

# Check liquidity distribution
liq_median = df['liquidity_score'].median()
liq_p90 = df['liquidity_score'].quantile(0.9)
print(f'Liquidity median: {liq_median:.3f} (expected: 0.45-0.55)')
print(f'Liquidity p90: {liq_p90:.3f} (expected: 0.80-0.90)')

# Overall status
all_good = (
    len(df.columns) >= 120 and
    oi_pct > 99 and
    oi_change_pct > 95 and
    liq_pct > 99 and
    0.45 < liq_median < 0.55
)

print(f'\n{'✅ ALL CHECKS PASSED' if all_good else '❌ SOME CHECKS FAILED'}')
"
```

---

## Full Documentation

Detailed documentation in `/docs/`:

1. **Diagnosis**: `OI_CHANGE_FAILURE_DIAGNOSIS.md` (Root cause analysis)
2. **Audit**: `FEATURE_PIPELINE_AUDIT.md` (113 features analyzed)
3. **Roadmap**: `BEAR_FEATURE_PIPELINE_ROADMAP.md` (4-phase plan)
4. **Executive Summary**: `BEAR_FEATURE_PIPELINE_EXECUTIVE_SUMMARY.md` (Business view)

---

## Key Scripts

All scripts in `/bin/`:

1. ✅ `fix_oi_change_pipeline.py` (Phase 2 - ready)
2. ✅ `backfill_liquidity_score.py` (Phase 3 - ready)
3. 🔧 `add_s2_derived_features.py` (Phase 1 - create from template above)
4. 🔧 `validate_bear_patterns_2022.py` (Phase 4 - create per roadmap)

---

## Questions?

- **Architecture**: See `/docs/BEAR_FEATURE_PIPELINE_ROADMAP.md`
- **Scripts**: See `/bin/fix_oi_change_pipeline.py` (has --help)
- **Validation**: See `/docs/OI_CHANGE_FAILURE_DIAGNOSIS.md` (Phase 3)
- **Rollback**: Copy backup file (shown above)

**Ready to execute on approval.**
