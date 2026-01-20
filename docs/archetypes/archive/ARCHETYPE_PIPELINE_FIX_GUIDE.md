# ARCHETYPE PIPELINE FIX GUIDE

**Quick reference for fixing identified plumbing issues**

---

## ISSUE 1: OI DATA GAP (CRITICAL)

### Problem
67% of OI data missing (2022-2023). Only 2024 data present.

### Impact
- S4/S5 cannot be tested on Terra Collapse (May 2022)
- S4/S5 cannot be tested on FTX Collapse (Nov 2022)
- Historical benchmarks unverifiable

### Fix
```bash
# Step 1: Backfill OI from OKX API
python3 bin/fix_oi_change_pipeline.py

# Step 2: Verify coverage (should be <5% null)
python3 -c "
import pandas as pd
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')
print('OI null %:', df['oi'].isna().mean() * 100)
print('OI date range:', df[df['oi'].notna()].index.min(), 'to', df[df['oi'].notna()].index.max())
"

# Step 3: Re-run audit
python3 bin/audit_archetype_pipeline.py
```

### Expected Outcome
- OI null % drops from 67.0% to <5%
- Date range: 2022-01-01 to 2024-12-31 (full coverage)
- Terra/FTX validation passes

### Time Estimate
30 minutes (10 min fetch + 20 min validation)

---

## ISSUE 2: IDENTICAL RESULTS (SUSPICIOUS)

### Problem
All three archetypes (S1, S4, S5) produce identical results:
- S1: 27 trades, PF 0.30
- S4: 27 trades, PF 0.30
- S5: 27 trades, PF 0.30

### Hypothesis
1. Configs not routing to archetype-specific logic
2. bin/backtest_knowledge_v2.py using generic fusion scoring
3. Archetypes falling back to default when features missing

### Investigation Steps

#### Step 1: Check if configs differentiate archetypes
```bash
# Run each archetype separately with trade output
python3 bin/backtest_knowledge_v2.py \
    --config configs/s1_v2_production.json \
    --start 2022-05-01 --end 2022-08-01 \
    > s1_output.txt

python3 bin/backtest_knowledge_v2.py \
    --config configs/s4_optimized_oos_test.json \
    --start 2022-05-01 --end 2022-08-01 \
    > s4_output.txt

python3 bin/backtest_knowledge_v2.py \
    --config configs/system_s5_production.json \
    --start 2022-05-01 --end 2022-08-01 \
    > s5_output.txt
```

#### Step 2: Compare outputs
```bash
# Check if outputs are identical
diff s1_output.txt s4_output.txt
diff s1_output.txt s5_output.txt

# If identical → configs not being used
# If different → results coincidentally similar (need deeper analysis)
```

#### Step 3: Check config structure
```bash
# Examine configs for archetype type/ID
python3 -c "
import json
for name, path in [('S1', 'configs/s1_v2_production.json'),
                   ('S4', 'configs/s4_optimized_oos_test.json'),
                   ('S5', 'configs/system_s5_production.json')]:
    with open(path) as f:
        cfg = json.load(f)
    print(f'{name}:')
    print(f'  Archetype ID: {cfg.get(\"archetype_id\", \"NOT FOUND\")}')
    print(f'  Strategy Type: {cfg.get(\"strategy_type\", \"NOT FOUND\")}')
    print()
"
```

#### Step 4: Check backtest_knowledge_v2.py logic
```bash
# Search for archetype-specific entry conditions
grep -n "S1\|S4\|S5\|archetype_id" bin/backtest_knowledge_v2.py

# Look for generic vs. archetype-specific logic
grep -n "def compute_entry_signal\|def should_enter" bin/backtest_knowledge_v2.py
```

### Expected Findings

**If configs are correct:**
- Each config has unique `archetype_id` field
- backtest_knowledge_v2.py reads this field
- Different logic paths for S1/S4/S5

**If plumbing is broken:**
- Configs missing `archetype_id` field
- backtest_knowledge_v2.py ignores configs
- Generic fusion scoring used for all archetypes

### Time Estimate
1 hour (30 min investigation + 30 min analysis)

---

## ISSUE 3: VERIFY HISTORICAL BENCHMARKS

### Problem
Cannot verify claimed performance:
- S4 PF 2.22 (claimed)
- S5 PF 1.86 (claimed)
- S1 60.7 trades/year (claimed)

### Prerequisites
- MUST fix OI data gap first (Issue 1)
- MUST investigate identical results (Issue 2)

### Verification Steps

#### Run on Full Historical Period
```bash
# S1 (2022-2024)
python3 bin/backtest_knowledge_v2.py \
    --config configs/s1_v2_production.json \
    --start 2022-01-01 --end 2024-12-31 \
    > s1_full_backtest.txt

# S4 (2022-2024) - needs OI fix
python3 bin/backtest_knowledge_v2.py \
    --config configs/s4_optimized_oos_test.json \
    --start 2022-01-01 --end 2024-12-31 \
    > s4_full_backtest.txt

# S5 (2022-2024) - needs OI fix
python3 bin/backtest_knowledge_v2.py \
    --config configs/system_s5_production.json \
    --start 2022-01-01 --end 2024-12-31 \
    > s5_full_backtest.txt
```

#### Extract Performance Metrics
```bash
# Parse results
python3 -c "
for name, file in [('S1', 's1_full_backtest.txt'),
                   ('S4', 's4_full_backtest.txt'),
                   ('S5', 's5_full_backtest.txt')]:
    with open(file) as f:
        lines = f.read()
    # Extract trades, PF, Sharpe from output
    # (Adjust parsing based on actual output format)
    print(f'{name} Results:')
    print('  (Parse metrics from output)')
    print()
"
```

#### Compare to Claims
```
CLAIMED vs ACTUAL:

S1:
  Claimed: 60.7 trades/year
  Actual: ??? (from backtest)
  Verdict: MATCH / MISMATCH

S4:
  Claimed: PF 2.22
  Actual: ??? (from backtest)
  Verdict: MATCH / MISMATCH

S5:
  Claimed: PF 1.86
  Actual: ??? (from backtest)
  Verdict: MATCH / MISMATCH
```

### Time Estimate
2 hours (1 hour running + 1 hour analysis)

---

## COMPLETE FIX WORKFLOW

### Total Time: 3-4 hours

```bash
# 1. Fix OI data gap (30 min)
python3 bin/fix_oi_change_pipeline.py
python3 bin/audit_archetype_pipeline.py  # verify fix

# 2. Investigate identical results (1 hour)
# Run separate backtests and compare
python3 bin/backtest_knowledge_v2.py --config configs/s1_v2_production.json --start 2022-05-01 --end 2022-08-01 > s1.txt
python3 bin/backtest_knowledge_v2.py --config configs/s4_optimized_oos_test.json --start 2022-05-01 --end 2022-08-01 > s4.txt
python3 bin/backtest_knowledge_v2.py --config configs/system_s5_production.json --start 2022-05-01 --end 2022-08-01 > s5.txt
diff s1.txt s4.txt

# 3. Verify historical benchmarks (2 hours)
# Run full backtests on 2022-2024
python3 bin/backtest_knowledge_v2.py --config configs/s1_v2_production.json --start 2022-01-01 --end 2024-12-31
python3 bin/backtest_knowledge_v2.py --config configs/s4_optimized_oos_test.json --start 2022-01-01 --end 2024-12-31
python3 bin/backtest_knowledge_v2.py --config configs/system_s5_production.json --start 2022-01-01 --end 2024-12-31

# 4. Final audit (5 min)
python3 bin/audit_archetype_pipeline.py
```

### Success Criteria

After all fixes:
- [ ] OI null % < 5% (was 67%)
- [ ] S1/S4/S5 produce DIFFERENT results (not identical)
- [ ] Historical benchmarks verified or explained
- [ ] All audit checks pass

**ONLY THEN proceed to test ArchetypeModel wrapper.**

---

## QUICK STATUS CHECK

```bash
# Check OI coverage
python3 -c "import pandas as pd; df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet'); print('OI null:', df['oi'].isna().mean()*100, '%')"

# Check feature store
ls -lh data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet

# Re-run audit
python3 bin/audit_archetype_pipeline.py
```

---

## TROUBLESHOOTING

### OI Backfill Fails
```bash
# Test API connectivity
curl "https://www.okx.com/api/v5/public/open-interest-history?instId=BTC-USDT-SWAP&period=1H&limit=10"

# Run dry-run first
python3 bin/fix_oi_change_pipeline.py --dry-run

# Check error logs
tail -100 bin/fix_oi_change_pipeline.py.log
```

### Configs Don't Differentiate
```bash
# Check if configs have archetype_id field
grep -r "archetype_id\|strategy_type" configs/s*.json

# Check if backtest engine uses this field
grep -n "archetype_id\|config\[" bin/backtest_knowledge_v2.py
```

### Benchmarks Don't Match
Possible reasons:
1. Different feature set (features added/removed over time)
2. Different date range (original benchmarks may use different periods)
3. Different configs (production vs. optimized versions)
4. Bugs in original engine (claims may be incorrect)

**Document discrepancies and investigate root cause.**

---

## AFTER FIXES: WRAPPER TESTING

Once plumbing verified:

1. Test ArchetypeModel wrapper on same periods
2. Compare wrapper results to original engine
3. Identify wrapper-specific bugs
4. Fix wrapper issues
5. Final validation

**Do NOT skip plumbing fixes. Wrapper testing on broken plumbing = wasted time.**
