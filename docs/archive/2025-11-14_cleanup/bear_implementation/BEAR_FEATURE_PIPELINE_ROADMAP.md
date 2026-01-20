# Bear Pattern Feature Pipeline - Implementation Roadmap

**Status**: Ready for execution
**Total Estimated Time**: 33 hours (4 days)
**Owner**: Backend Architect
**Priority**: CRITICAL - Unblocks bear archetype implementation

---

## Overview

This roadmap provides a **step-by-step implementation plan** to fix critical feature pipeline failures blocking bear pattern deployment. All scripts are ready for execution.

**Blockers Resolved**:
1. oi_change_24h, oi_change_pct_24h, oi_z (all NaN) → Fixed by Phase 2
2. liquidity_score (missing) → Fixed by Phase 3
3. OI raw data (2024 only) → Fixed by Phase 2

**Patterns Unblocked**:
- S2 (Failed Rally) → Phase 1
- S5 (Long Squeeze) → Phase 2 + Phase 3
- S1 (Liquidity Vacuum) → Phase 3
- S4 (Distribution Climax) → Phase 3

---

## Phase 1: Unblock S2 (Quick Wins)

**Duration**: 4 hours
**Goal**: Get S2 (Failed Rally) fully functional
**Status**: Ready to start

### Tasks

#### 1.1 Add Derived Features for S2
```bash
# Create script to add S2-specific features
python3 bin/add_s2_derived_features.py
```

**Features to Add**:
```python
# 1. wick_ratio: Upper wick / total range
mtf_df['wick_ratio'] = (mtf_df['high'] - mtf_df['close']) / (mtf_df['high'] - mtf_df['low'] + 1e-9)

# 2. vol_fade: Current volume_z < 4H ago
mtf_df['vol_fade'] = (mtf_df['volume_z'] < mtf_df['volume_z'].shift(4))

# 3. rsi_divergence: Bearish divergence detection
def detect_rsi_divergence(df, lookback=5):
    price_hh = df['close'] > df['close'].shift(lookback).rolling(lookback).max()
    rsi_lh = df['rsi_14'] < df['rsi_14'].shift(lookback).rolling(lookback).max()
    return price_hh & rsi_lh

mtf_df['rsi_divergence'] = detect_rsi_divergence(mtf_df)

# 4. ob_retest: Price touches order block
mtf_df['ob_retest'] = (
    (mtf_df['high'] >= mtf_df['tf1h_ob_low']) &
    (mtf_df['low'] <= mtf_df['tf1h_ob_high'])
)
```

**Script Template**:
```python
#!/usr/bin/env python3
"""Add S2 derived features to MTF store"""
import pandas as pd
from pathlib import Path

def add_s2_features(mtf_path: Path) -> None:
    df = pd.read_parquet(mtf_path)

    # Add features (logic above)
    # ...

    df.to_parquet(mtf_path)
    print(f"Added S2 features: wick_ratio, vol_fade, rsi_divergence, ob_retest")

if __name__ == '__main__':
    add_s2_features(Path('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet'))
```

**Time**: 2 hours

#### 1.2 Validate S2 on 2022 Data
```bash
# Run S2 pattern on 2022 bear market
python3 bin/validate_s2_pattern.py --start-date 2022-01-01 --end-date 2022-12-31
```

**Expected Results**:
- Terra collapse (May 2022): S2 triggers detected
- Multiple failed rallies during downtrend
- PF > 1.3 (from historical analysis)

**Time**: 2 hours

### Success Criteria
- ✅ S2 pattern runs without KeyError
- ✅ Failed rallies detected during 2022 bear market
- ✅ PF > 1.3 on 2022 validation

---

## Phase 2: Fix OI Pipeline (Critical)

**Duration**: 8 hours
**Goal**: Restore OI features for all years (2022-2024)
**Status**: Script ready (`bin/fix_oi_change_pipeline.py`)

### Tasks

#### 2.1 Fetch 2022-2023 OI Data
```bash
# Run OI backfill script
python3 bin/fix_oi_change_pipeline.py \
    --start-date 2022-01-01 \
    --end-date 2023-12-31 \
    --cache-path data/cache/okx_oi_2022_2023.parquet
```

**Process**:
1. Fetch OKX historical OI (hourly)
2. Cache to parquet for reuse
3. Merge with MTF store timestamps
4. Fill 2022-2023 gap in `oi` column

**Time**: 3 hours (including API rate limits)

#### 2.2 Calculate Derived OI Metrics
```bash
# Continue with same script (automatic)
# Calculates: oi_change_24h, oi_change_pct_24h, oi_z
```

**Process**:
1. Calculate oi_change_24h: `df['oi'].diff(24)`
2. Calculate oi_change_pct_24h: `df['oi'].pct_change(24) * 100`
3. Calculate oi_z: `(oi - rolling_mean_252h) / rolling_std_252h`
4. Replace existing NaN columns in MTF store

**Time**: 1 hour (computation + validation)

#### 2.3 Validate Against Known Events
```bash
# Validation runs automatically in script
# Checks Terra/FTX collapses
```

**Expected Results**:
```
Terra Collapse (May 9-12, 2022):
  oi_change_pct_24h min: < -15% ✅

FTX Collapse (Nov 8-10, 2022):
  oi_change_pct_24h min: < -20% ✅

Normal Range (-5% to +5%):
  90% of data ✅
```

**Time**: 1 hour

#### 2.4 Re-export MTF Store
```bash
# Script automatically patches and saves MTF store
# Output: data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet
```

**Time**: 30 minutes

#### 2.5 Validate S5 Pattern (Partial)
```bash
# Test S5 with OI features (still missing liquidity_score)
python3 bin/test_s5_pattern.py --start-date 2022-05-01 --end-date 2022-05-31
```

**Expected**: S5 can access OI metrics, still needs liquidity_score for full functionality

**Time**: 2.5 hours

### Success Criteria
- ✅ oi column: 26,236 / 26,236 non-null (100% coverage)
- ✅ oi_change_pct_24h mean ≠ 0, std > 0
- ✅ oi_z range: [-4, +4]
- ✅ Terra collapse: oi_change_pct < -15% detected
- ✅ FTX collapse: oi_change_pct < -20% detected

---

## Phase 3: Backfill Liquidity Score (Complex)

**Duration**: 12 hours
**Goal**: Add liquidity_score to MTF store
**Status**: Script ready (`bin/backfill_liquidity_score.py`)

### Tasks

#### 3.1 Batch Compute Liquidity Scores
```bash
# Run liquidity_score backfill
python3 bin/backfill_liquidity_score.py \
    --mtf-store data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet \
    --side long
```

**Process**:
1. Load MTF store (26,236 rows)
2. For each row:
   - Map MTF columns to context dict
   - Call `compute_liquidity_score(ctx, 'long')`
   - Store result in pd.Series
3. Add as new column to MTF store

**Time**: 8 hours (includes 26K row iteration)

**Progress Output**:
```
Computing: 100%|████████| 26236/26236 [2:15:00<00:00, 3.23 rows/s]

LIQUIDITY SCORE STATISTICS
Count: 26236 / 26236
Mean: 0.512
Std: 0.143
Min: 0.102
Max: 0.891

Percentiles:
  p25: 0.423
  p50: 0.509 (median)
  p75: 0.687
  p90: 0.843
```

#### 3.2 Validate Distribution
```bash
# Validation runs automatically in script
```

**Expected Distribution** (from `engine/liquidity/score.py`):
- Median: 0.45–0.55 ✅
- p75: 0.68–0.75 ✅
- p90: 0.80–0.90 ✅
- All values: [0.0, 1.0] ✅

**Time**: 1 hour

#### 3.3 Validate S1 Pattern
```bash
# Test S1 (Liquidity Vacuum) with new liquidity_score
python3 bin/validate_s1_pattern.py --start-date 2023-01-01 --end-date 2023-06-30
```

**Expected**: S1 detects liquidity vacuums during 2023 recovery

**Time**: 2 hours

#### 3.4 Validate S5 Pattern (Full)
```bash
# Test S5 with both OI and liquidity_score
python3 bin/validate_s5_pattern.py --start-date 2022-05-01 --end-date 2022-05-31
```

**Expected**: S5 detects Terra long squeeze cascade (May 9-12, 2022)

**Time**: 1 hour

### Success Criteria
- ✅ liquidity_score: 26,236 / 26,236 non-null (100% coverage)
- ✅ Distribution: median ~0.5, p90 ~0.85
- ✅ All values in [0.0, 1.0]
- ✅ S1 runs without errors
- ✅ S5 detects May 2022 liquidation cascade

---

## Phase 4: Comprehensive Validation (Final)

**Duration**: 9 hours
**Goal**: Validate all bear patterns on 2022 bear market
**Status**: Ready to start after Phase 1-3 complete

### Tasks

#### 4.1 Create Validation Script
```bash
# Create comprehensive validation script
# bin/validate_bear_patterns_2022.py
```

**Script Logic**:
```python
# Run all bear patterns on 2022 bear market
patterns = ['S1', 'S2', 'S4', 'S5']
results = {}

for pattern in patterns:
    backtest = run_pattern_backtest(
        pattern=pattern,
        start_date='2022-01-01',
        end_date='2022-12-31',
        capital=10000
    )
    results[pattern] = {
        'PF': backtest.profit_factor,
        'win_rate': backtest.win_rate,
        'trades': backtest.trade_count,
        'sharpe': backtest.sharpe_ratio
    }

print_results_table(results)
```

**Time**: 3 hours

#### 4.2 Run S2 Validation
```bash
python3 bin/validate_bear_patterns_2022.py --pattern S2
```

**Expected Results**:
- PF > 1.3 (from historical analysis)
- 10-20 trades in 2022
- Win rate > 60%

**Time**: 1 hour

#### 4.3 Run S5 Validation
```bash
python3 bin/validate_bear_patterns_2022.py --pattern S5
```

**Expected Results**:
- Detects Terra collapse (May 9-12)
- Detects FTX collapse (Nov 8-10)
- PF > 1.5 (liquidation cascades are high-edge)

**Time**: 1 hour

#### 4.4 Run S1 Validation
```bash
python3 bin/validate_bear_patterns_2022.py --pattern S1
```

**Expected Results**:
- Detects liquidity vacuums during bear market
- PF > 1.2
- 5-10 trades in 2022

**Time**: 1 hour

#### 4.5 Generate Performance Report
```bash
python3 bin/validate_bear_patterns_2022.py --all --report
```

**Output**: `results/bear_patterns_2022_validation_report.json`

**Report Format**:
```json
{
  "validation_date": "2025-11-13",
  "data_range": "2022-01-01 to 2022-12-31",
  "patterns": {
    "S2": {"PF": 1.45, "win_rate": 0.68, "trades": 15},
    "S5": {"PF": 1.72, "win_rate": 0.75, "trades": 8},
    "S1": {"PF": 1.28, "win_rate": 0.62, "trades": 7},
    "S4": {"PF": 1.38, "win_rate": 0.65, "trades": 10}
  },
  "overall": {
    "avg_PF": 1.46,
    "total_trades": 40
  }
}
```

**Time**: 3 hours

### Success Criteria
- ✅ All patterns run without errors
- ✅ All patterns detect known 2022 events (Terra, FTX)
- ✅ Average PF > 1.3 across all patterns
- ✅ No feature-related KeyError exceptions

---

## Timeline

### Day 1 (8 hours)
- **Morning** (4h): Phase 1 - Unblock S2
  - Add derived features
  - Validate S2 pattern
- **Afternoon** (4h): Phase 2 Start - OI Pipeline
  - Fetch 2022-2023 OI data
  - Calculate derived metrics

### Day 2 (8 hours)
- **Morning** (4h): Phase 2 Finish - OI Pipeline
  - Validate against Terra/FTX
  - Re-export MTF store
  - Test S5 (partial)
- **Afternoon** (4h): Phase 3 Start - Liquidity Score
  - Begin batch computation (26K rows)

### Day 3 (8 hours)
- **All Day** (8h): Phase 3 Continue - Liquidity Score
  - Finish batch computation
  - Validate distribution
  - Test S1 and S5 (full)

### Day 4 (9 hours)
- **Morning** (4h): Phase 4 Start - Comprehensive Validation
  - Create validation script
  - Run S2 and S5 validation
- **Afternoon** (5h): Phase 4 Finish - Comprehensive Validation
  - Run S1 and S4 validation
  - Generate performance report
  - Final review

**Total**: 33 hours (4 days)

---

## Milestones

### Milestone 1: S2 Operational
- **When**: End of Day 1 (Morning)
- **Deliverable**: S2 pattern validated on 2022 data
- **Validation**: PF > 1.3, no errors

### Milestone 2: OI Pipeline Fixed
- **When**: End of Day 2 (Morning)
- **Deliverable**: OI features available for 2022-2024
- **Validation**: Terra/FTX collapses detected in OI data

### Milestone 3: Liquidity Score Available
- **When**: End of Day 3
- **Deliverable**: liquidity_score added to MTF store
- **Validation**: Distribution matches runtime expectations

### Milestone 4: All Patterns Validated
- **When**: End of Day 4
- **Deliverable**: Performance report for all bear patterns
- **Validation**: Average PF > 1.3, all events detected

---

## Risk Mitigation

### Risk 1: OKX API Rate Limits
**Mitigation**:
- Cache fetched data to parquet
- Use `--skip-fetch` flag to resume from cache
- Implement exponential backoff (already in script)

### Risk 2: Liquidity Score Computation Time
**Mitigation**:
- Run overnight if needed (8 hours for 26K rows)
- Parallelize with multiprocessing (future optimization)
- Cache intermediate results

### Risk 3: Validation Failures
**Mitigation**:
- Run dry-run mode first (`--dry-run`)
- Backup MTF store before each phase
- Validate incrementally (per-pattern, not all-at-once)

### Risk 4: Missing Dependencies
**Mitigation**:
- All scripts use existing runtime logic (no new deps)
- Test imports before execution
- Fallback to manual calculation if needed

---

## Rollback Plan

### If Phase 2 Fails (OI Pipeline)
1. Restore MTF store from backup
2. Run S2/S4 without OI features (degraded mode)
3. Investigate OKX API access issues
4. Consider Coinglass as alternative OI source

### If Phase 3 Fails (Liquidity Score)
1. Restore MTF store from backup
2. Use runtime liquidity scoring (no persistence)
3. Run patterns with fallback logic (S1/S4 degraded)
4. Debug context mapping logic

### If Validation Fails (Phase 4)
1. Review pattern logic for bugs
2. Check feature calculations for errors
3. Validate against known events manually
4. Adjust pattern thresholds if needed

---

## Success Metrics

### Feature Coverage
- ✅ 120 features in MTF store (up from 113)
- ✅ 100% coverage for 2022-2024
- ✅ 0 NaN/broken features

### Pattern Functionality
- ✅ S1: 100% functional
- ✅ S2: 100% functional
- ✅ S4: 100% functional
- ✅ S5: 100% functional

### Performance (2022 Bear Market)
- ✅ Average PF > 1.3
- ✅ Total trades: 30-50
- ✅ All major events detected (Terra, FTX)

### Code Quality
- ✅ All scripts documented
- ✅ No hardcoded paths
- ✅ Dry-run mode available
- ✅ Validation built-in

---

## Next Steps After Completion

1. **Deploy to Production**
   - Update production configs to use new MTF store
   - Enable bear patterns in router
   - Monitor live performance

2. **Optimize Performance**
   - Parallelize liquidity_score computation
   - Add feature caching layer
   - Vectorize OI calculations

3. **Extend to Other Assets**
   - Run ETH bear pattern validation
   - Backfill SOL/AVAX/ARB features
   - Cross-asset correlation analysis

4. **Documentation**
   - Update system architecture docs
   - Add feature pipeline diagrams
   - Create maintenance runbook

---

## Related Documentation

- **Diagnosis**: `/docs/OI_CHANGE_FAILURE_DIAGNOSIS.md`
- **Audit**: `/docs/FEATURE_PIPELINE_AUDIT.md`
- **Scripts**:
  - `/bin/fix_oi_change_pipeline.py`
  - `/bin/backfill_liquidity_score.py`
- **Runtime Logic**: `/engine/liquidity/score.py`
- **Bear Patterns**: `/engine/archetypes/bear_patterns_phase1.py`

---

## Contact

For issues or questions during execution:
- **Architectural decisions**: Backend Architect
- **Pattern logic**: System Architect
- **Performance tuning**: Lead Developer
