# FIX GHOST FEATURES - ACTION CHECKLIST

**Problem:** 29 features claimed generated but missing from feature store
**Impact:** Domain wiring non-functional (falls back to defaults)
**Fix Time:** 45 minutes
**Priority:** CRITICAL - Must fix before any testing

---

## PRE-FLIGHT CHECK

Before running any scripts, verify the problem:

```bash
# 1. Check feature store timestamp
stat -f "%Sm %N" -t "%Y-%m-%d %H:%M:%S" data/features_2022_with_regimes.parquet
# Expected: 2025-11-25 21:19:54 (OLD - needs update)

# 2. Check if claimed features exist
python3 << 'EOF'
import pandas as pd
df = pd.read_parquet('data/features_2022_with_regimes.parquet')
missing = ['smc_score', 'hob_demand_zone', 'wyckoff_ps',
           'wyckoff_pti_confluence', 'fib_time_cluster']
print("Missing features check:")
for f in missing:
    print(f"  {f}: {'❌ MISSING' if f not in df.columns else '✅ EXISTS'}")
EOF
# Expected: All ❌ MISSING
```

If all features show ❌ MISSING, proceed with fix.

---

## STEP 1: BACKUP CURRENT FEATURE STORE (2 min)

Create safety backup before modifying:

```bash
# Create timestamped backup
cp data/features_2022_with_regimes.parquet \
   data/features_2022_with_regimes_backup_$(date +%Y%m%d_%H%M%S).parquet

# Verify backup created
ls -lh data/features_2022_with_regimes_backup*.parquet
```

**Success Criteria:** Backup file exists with today's timestamp

---

## STEP 2: RUN DOMAIN FEATURE BACKFILL (20 min)

Generate 15 missing domain features (SMC, HOB, Wyckoff PTI):

```bash
# Run fast vectorized backfill
python3 bin/backfill_domain_features_fast.py \
  --input data/features_2022_with_regimes.parquet \
  --output data/features_2022_with_regimes.parquet

# Expected output:
# - Processing 8,741 bars
# - Adding 15 features
# - Execution time: ~30 seconds
```

**Features Added:**
- `smc_score` - SMC composite score
- `smc_bos` - Break of Structure flag
- `smc_choch` - Change of Character flag
- `smc_liquidity_sweep` - Liquidity sweep detection
- `smc_demand_zone` - Demand zone (bullish OB)
- `smc_supply_zone` - Supply zone (bearish OB)
- `hob_demand_zone` - HOB demand zone (same as SMC)
- `hob_supply_zone` - HOB supply zone (same as SMC)
- `hob_imbalance` - HOB net imbalance
- `wyckoff_pti_confluence` - PTI + Wyckoff trap confluence
- `wyckoff_pti_score` - Composite PTI score
- `wyckoff_ps` - Preliminary Support flag
- `temporal_confluence` - Temporal alignment flag
- `temporal_support_cluster` - Support time confluence
- `temporal_resistance_cluster` - Resistance time confluence

**Success Criteria:**
```bash
# Verify domain features added
python3 << 'EOF'
import pandas as pd
df = pd.read_parquet('data/features_2022_with_regimes.parquet')
domain = ['smc_score', 'hob_demand_zone', 'wyckoff_ps', 'wyckoff_pti_confluence']
print("Domain features check:")
for f in domain:
    exists = f in df.columns
    print(f"  {f}: {'✅ EXISTS' if exists else '❌ MISSING'}")
EOF
```

Expected: All ✅ EXISTS

---

## STEP 3: RUN TEMPORAL TIMING FEATURE GENERATION (15 min)

Generate 14 missing temporal timing features:

```bash
# Run temporal timing feature generator
python3 bin/generate_temporal_timing_features.py \
  --input data/features_2022_with_regimes.parquet \
  --output data/features_2022_with_regimes.parquet

# Expected output:
# - Processing 8,741 bars
# - Adding 14 features
# - Execution time: ~15 seconds
```

**Features Added:**
- `bars_since_sc` - Bars since Selling Climax
- `bars_since_ar` - Bars since Automatic Rally
- `bars_since_st` - Bars since Secondary Test
- `bars_since_sos_long` - Bars since SOS Long
- `bars_since_sos_short` - Bars since SOS Short
- `bars_since_spring` - Bars since Spring
- `bars_since_utad` - Bars since UTAD
- `bars_since_ps` - Bars since Preliminary Support
- `bars_since_bc` - Bars since Buying Climax
- `fib_time_cluster` - Fibonacci time cluster flag
- `fib_time_score` - Fib confluence strength
- `fib_time_target` - Fib levels aligned
- `gann_cycle` - Gann cycle point flag
- `volatility_cycle` - Volatility regime cycle

**Success Criteria:**
```bash
# Verify temporal features added
python3 << 'EOF'
import pandas as pd
df = pd.read_parquet('data/features_2022_with_regimes.parquet')
temporal = ['bars_since_sc', 'fib_time_cluster', 'gann_cycle']
print("Temporal features check:")
for f in temporal:
    exists = f in df.columns
    print(f"  {f}: {'✅ EXISTS' if exists else '❌ MISSING'}")
EOF
```

Expected: All ✅ EXISTS

---

## STEP 4: VERIFY ALL 29 FEATURES ADDED (5 min)

Comprehensive verification:

```bash
python3 << 'EOF'
import pandas as pd

df = pd.read_parquet('data/features_2022_with_regimes.parquet')

print("="*80)
print("FEATURE BACKFILL VERIFICATION")
print("="*80)
print(f"\nFeature Store: {df.shape[0]:,} bars × {df.shape[1]} columns")

# Check all 29 claimed features
domain_features = [
    'smc_score', 'smc_bos', 'smc_choch', 'smc_liquidity_sweep',
    'smc_demand_zone', 'smc_supply_zone',
    'hob_demand_zone', 'hob_supply_zone', 'hob_imbalance',
    'wyckoff_pti_confluence', 'wyckoff_pti_score', 'wyckoff_ps',
    'temporal_confluence', 'temporal_support_cluster', 'temporal_resistance_cluster'
]

temporal_features = [
    'bars_since_sc', 'bars_since_ar', 'bars_since_st',
    'bars_since_sos_long', 'bars_since_sos_short',
    'bars_since_spring', 'bars_since_utad', 'bars_since_ps', 'bars_since_bc',
    'fib_time_cluster', 'fib_time_score', 'fib_time_target',
    'gann_cycle', 'volatility_cycle'
]

print("\nDomain Features (15):")
domain_present = 0
for f in domain_features:
    exists = f in df.columns
    status = "✅ EXISTS" if exists else "❌ MISSING"
    print(f"  {f:35s} {status}")
    if exists:
        domain_present += 1

print(f"\n  VERDICT: {domain_present}/15 domain features present")

print("\nTemporal Features (14):")
temporal_present = 0
for f in temporal_features:
    exists = f in df.columns
    status = "✅ EXISTS" if exists else "❌ MISSING"
    print(f"  {f:35s} {status}")
    if exists:
        temporal_present += 1

print(f"\n  VERDICT: {temporal_present}/14 temporal features present")

total_present = domain_present + temporal_present
print("\n" + "="*80)
print(f"OVERALL: {total_present}/29 features present ({total_present/29*100:.1f}%)")
print("="*80)

if total_present == 29:
    print("\n✅ SUCCESS: All 29 features successfully added!")
else:
    print(f"\n❌ INCOMPLETE: {29-total_present} features still missing")

EOF
```

**Success Criteria:** "✅ SUCCESS: All 29 features successfully added!"

---

## STEP 5: VERIFY DATA QUALITY (3 min)

Check that features have valid values (not all NULL/zero):

```bash
python3 << 'EOF'
import pandas as pd

df = pd.read_parquet('data/features_2022_with_regimes.parquet')

print("="*80)
print("DATA QUALITY CHECK")
print("="*80)

# Check key features have non-null/non-zero values
key_features = {
    'smc_score': 'float (should have mean ~0.3-0.5)',
    'hob_demand_zone': 'bool (should have 300-500 events)',
    'wyckoff_ps': 'bool (should have 200-300 events)',
    'fib_time_cluster': 'bool (should have 8000+ events)',
    'bars_since_sc': 'int (should have mean 5000-8000)',
}

print("\nKey Feature Statistics:\n")
for feat, desc in key_features.items():
    if feat in df.columns:
        if df[feat].dtype == bool:
            events = df[feat].sum()
            pct = events / len(df) * 100
            print(f"  {feat:25s}: {events:5d} events ({pct:5.1f}%) - {desc}")
        else:
            mean_val = df[feat].mean()
            std_val = df[feat].std()
            non_zero = (df[feat] != 0).sum()
            print(f"  {feat:25s}: mean={mean_val:.3f} std={std_val:.3f} non-zero={non_zero} - {desc}")
    else:
        print(f"  {feat:25s}: ❌ MISSING")

EOF
```

**Success Criteria:**
- smc_score: mean ~0.3-0.5 (not all zeros)
- hob_demand_zone: 300-500 events (not all False)
- wyckoff_ps: 200-300 events (not all False)
- fib_time_cluster: 8000+ events (not all False)
- bars_since_sc: mean 5000-8000 (not all zeros)

---

## STEP 6: RE-RUN DOMAIN WIRING VERIFICATION (5 min)

Test that domain boosts now activate:

```bash
# Test S1 with domain features enabled
python3 bin/test_domain_wiring.py --archetype S1 --enable-all-domains

# Expected output (BEFORE fix):
#   Wyckoff boosts: 0
#   SMC boosts: 0
#   Temporal boosts: 0
#   HOB boosts: 0
#   Macro penalties: 0
#   (All 0 because features missing)

# Expected output (AFTER fix):
#   Wyckoff boosts: 50-150
#   SMC boosts: 200-400
#   Temporal boosts: 10-50
#   HOB boosts: 30-100
#   Macro penalties: 20-80
#   (Non-zero counts = domain wiring now functional)
```

**Success Criteria:** Non-zero boost/veto counts for all domain engines

---

## STEP 7: UPDATE TIMESTAMP CHECK (1 min)

Verify feature store was actually updated:

```bash
# Check new timestamp
stat -f "%Sm %N" -t "%Y-%m-%d %H:%M:%S" data/features_2022_with_regimes.parquet
# Expected: Today's date (2025-12-10)

# Check file size increased
ls -lh data/features_2022_with_regimes.parquet
# Expected: Larger than backup (more columns = more bytes)
```

**Success Criteria:** Feature store timestamp is today

---

## ROLLBACK PROCEDURE (if something goes wrong)

If any step fails or produces incorrect data:

```bash
# 1. Restore from backup
cp data/features_2022_with_regimes_backup_YYYYMMDD_HHMMSS.parquet \
   data/features_2022_with_regimes.parquet

# 2. Verify restoration
python3 -c "
import pandas as pd
df = pd.read_parquet('data/features_2022_with_regimes.parquet')
print(f'Restored: {df.shape[0]:,} bars × {df.shape[1]} columns')
"

# 3. Debug the issue
# - Check script logs for errors
# - Verify input file integrity
# - Test on small subset first
```

---

## POST-FIX ACTIONS

After all 29 features successfully added:

### 1. Re-run Logic Tree Audit (10 min)

```bash
python3 bin/audit_logic_tree.py
```

Expected changes:
- GREEN features: 38 → 50 (12 ghost references now real)
- YELLOW features: 18 → 10 (8 features now wired)
- Ghost wiring rate: 24% → 0%

### 2. Update Documentation (5 min)

Mark gap analysis as resolved:

```bash
echo "✅ RESOLVED: 2025-12-10 22:30 UTC" >> AUDIT_GAP_SUMMARY.txt
echo "  - All 29 features successfully backfilled" >> AUDIT_GAP_SUMMARY.txt
echo "  - Domain wiring now functional" >> AUDIT_GAP_SUMMARY.txt
echo "  - Verified non-zero boost/veto counts" >> AUDIT_GAP_SUMMARY.txt
```

### 3. Test Full Engine (Ready for optimization)

Now safe to run full backtests with domain features:

```bash
# Test S1 with all domains enabled
python3 bin/backtest_archetype.py \
  --config configs/test/s1_all_domains.json \
  --start 2022-01-01 \
  --end 2024-12-31

# Compare with/without domain features
python3 bin/compare_domain_impact.py --archetype S1
```

---

## EXPECTED TIMELINE

| Step | Task | Time | Cumulative |
|------|------|------|------------|
| 0 | Pre-flight check | 2 min | 2 min |
| 1 | Backup feature store | 2 min | 4 min |
| 2 | Run domain backfill | 20 min | 24 min |
| 3 | Run temporal generation | 15 min | 39 min |
| 4 | Verify all features | 5 min | 44 min |
| 5 | Check data quality | 3 min | 47 min |
| 6 | Test domain wiring | 5 min | 52 min |
| 7 | Verify timestamp | 1 min | 53 min |

**Total Time:** ~53 minutes (allowing for margin)

---

## SUCCESS CRITERIA SUMMARY

Fix is complete when ALL of these are true:

- [ ] Feature store timestamp is today (2025-12-10)
- [ ] All 29 features present in feature store (100% completion)
- [ ] Domain features have valid statistics (mean smc_score ~0.3-0.5)
- [ ] Temporal features have valid statistics (8000+ fib_time_cluster events)
- [ ] Domain wiring tests show non-zero boost/veto counts
- [ ] No features return all NULL/zero values
- [ ] Backup created successfully (can rollback if needed)

---

## QUICK COMMAND SEQUENCE (Copy-Paste)

For experienced users, run all steps in sequence:

```bash
# 0. Pre-check
stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" data/features_2022_with_regimes.parquet

# 1. Backup
cp data/features_2022_with_regimes.parquet \
   data/features_2022_with_regimes_backup_$(date +%Y%m%d_%H%M%S).parquet

# 2. Domain backfill
python3 bin/backfill_domain_features_fast.py \
  --input data/features_2022_with_regimes.parquet \
  --output data/features_2022_with_regimes.parquet

# 3. Temporal generation
python3 bin/generate_temporal_timing_features.py \
  --input data/features_2022_with_regimes.parquet \
  --output data/features_2022_with_regimes.parquet

# 4. Verify
python3 -c "
import pandas as pd
df = pd.read_parquet('data/features_2022_with_regimes.parquet')
required = ['smc_score', 'hob_demand_zone', 'wyckoff_ps',
            'wyckoff_pti_confluence', 'fib_time_cluster', 'bars_since_sc']
print('Feature check:')
all_present = all(f in df.columns for f in required)
for f in required:
    print(f'  {f}: {\"✅\" if f in df.columns else \"❌\"}')
print(f'\n{\"✅ SUCCESS\" if all_present else \"❌ FAILED\"}')
"

# 5. Test wiring
python3 bin/test_domain_wiring.py --archetype S1 --enable-all-domains
```

Expected final output: "✅ SUCCESS"

---

**Generated:** 2025-12-10 22:00 UTC
**Priority:** CRITICAL - Must complete before any engine testing
**Estimated Time:** 45-55 minutes
**Risk Level:** LOW (scripts create backups, easy rollback)
