# Liquidity Score Pipeline Specification

**Status**: Phase 1 - Critical Data Blocker
**Priority**: HIGH (Blocks S1, S4, S5 bear archetypes)
**Existing Work**: Backfill script exists (`bin/backfill_liquidity_score.py`), runtime logic exists (`engine/liquidity/score.py`)
**Current Issue**: Feature computed at runtime but never persisted to feature store

---

## 1. Feature Definition

### Canonical Name
`liquidity_score`

### Formula
The liquidity score is a composite metric combining 4 pillars, each contributing to a final score in `[0, 1]`:

```python
liquidity_score = 0.35*S + 0.30*C + 0.20*L + 0.15*P + 0.08*HTF_boost
```

**Pillar Breakdown:**

1. **Strength/Intent (S)** - 35% weight
   - Primary: `tf1d_boms_strength` (0-1, already normalized)
   - Secondary: `tf4h_boms_displacement` (normalized by cap=1.5)
   - Combine: `S = 0.75 * boms_strength + 0.25 * disp_norm`

2. **Structure Context (C)** - 30% weight
   - FVG quality: `fvg_quality` (0-1) or fallback to binary `fvg_present`
   - BOS freshness: `fresh_bos_flag` (boolean, adds 0.10 bonus)
   - Combine: `C = clip(fvg_quality + 0.10 * bos_fresh)`

3. **Liquidity Conditions (L)** - 20% weight
   - Volume z-score: `volume_zscore` (mapped to [0,1] via sigmoid)
   - Spread proxy: `(high - low) / close` (inverted, tighter = better)
   - Combine: `L = 0.70 * vol_score + 0.30 * spread_score`

4. **Positioning & Timing (P)** - 15% weight
   - Discount/premium: `close <= range_eq` (long) or `close >= range_eq` (short)
   - ATR regime: Prefer mid-regime (peak at 0.5, penalty at extremes)
   - Time-of-day: `tod_boost` (0.5 default, crypto: US/EU overlap)
   - Combine: `P = 0.50 * in_discount + 0.30 * atr_adj + 0.20 * tod_boost`

5. **HTF Boost** - 8% additive boost
   - 4H fusion score: `tf4h_fusion_score` (0-1)
   - Boost: `+0.08 * fusion4h` (max +0.08)

### Scaling
- **Range**: `[0.0, 1.0]` (hard-clipped)
- **Target Distribution** (from runtime calibration):
  - Median: `0.45–0.55` (neutral baseline)
  - p75: `0.68–0.75` (good setups)
  - p90: `0.80–0.90` (excellent setups)

### Required Features
| Feature | Source | Tier | Fallback |
|---------|--------|------|----------|
| `tf1d_boms_strength` | MTF store | 2 | `0.0` |
| `tf4h_boms_displacement` | MTF store | 2 | `0.0` |
| `fvg_quality` | MTF store | 2 | Use `fvg_present` (binary) |
| `fresh_bos_flag` | MTF store | 2 | `False` |
| `volume_zscore` | MTF store | 2 | `0.0` |
| `atr` | MTF store | 1 | `600.0` (BTC default) |
| `high`, `low`, `close` | MTF store | 1 | **Required** |
| `range_eq` | MTF store | 2 | Compute from `rolling_high/low` |
| `tf4h_fusion_score` | MTF store | 2 | `0.5` (neutral) |
| `tod_boost` | MTF store | 2 | `0.5` (neutral) |

---

## 2. NaN Handling Strategy

### Pre-Computation Validation
1. **Check OHLCV**: `high`, `low`, `close` must exist and be non-null
2. **Missing Features**: All missing features default to neutral values:
   - Numeric features → `0.0` or neutral baseline (e.g., `0.5` for scores)
   - Boolean features → `False`
   - ATR → `600.0` (typical BTC 1H ATR)

### During Computation
- Use defensive `_clip01()` function (handles `None`, `NaN`, `inf`)
- No exceptions raised (graceful degradation)
- Row-by-row fallbacks via `map_mtf_row_to_context()` function

### Post-Computation Validation
- **Bounds Check**: All scores must be in `[0.0, 1.0]`
- **Non-Null Coverage**: Expect 100% coverage (defaults applied)
- **Distribution Check**: Compare to target percentiles (relaxed validation)

---

## 3. Feature Registry Entry

Add to `engine/features/registry.py`:

```python
FeatureSpec(
    canonical="liquidity_score",
    dtype="float64",
    tier=2,  # MTF tier (derived feature)
    required=False,  # Optional for most strategies, critical for S1/S4/S5
    aliases=["liq_score"],
    range_min=0.0,
    range_max=1.0,
    description="Composite liquidity availability score (S+C+L+P+HTF pillars)"
)
```

### Validation Rules
```python
def validate_liquidity_score(df: pd.DataFrame) -> Dict[str, bool]:
    """Validate liquidity_score column in feature store"""
    checks = {}

    # 1. Bounds check
    checks['in_bounds'] = (df['liquidity_score'] >= 0.0).all() and \
                          (df['liquidity_score'] <= 1.0).all()

    # 2. Coverage check
    checks['no_nulls'] = df['liquidity_score'].notna().all()

    # 3. Distribution check (relaxed)
    median = df['liquidity_score'].median()
    p75 = df['liquidity_score'].quantile(0.75)
    p90 = df['liquidity_score'].quantile(0.90)

    checks['median_sane'] = 0.30 <= median <= 0.70  # Relaxed from 0.45-0.55
    checks['p75_sane'] = 0.60 <= p75 <= 0.85       # Relaxed from 0.68-0.75
    checks['p90_sane'] = 0.75 <= p90 <= 0.95       # Relaxed from 0.80-0.90

    return checks
```

---

## 4. Backfill Script Design

### Script: `bin/backfill_liquidity_score.py`

**Status**: Already exists and is production-ready

#### Architecture
1. **Load MTF Store**: Read parquet file (indexed by timestamp)
2. **Map Context**: Convert MTF row to runtime context dict (`map_mtf_row_to_context()`)
3. **Compute Scores**: Call `compute_liquidity_score(ctx, side='long')` per row
4. **Validate**: Check distribution against target percentiles
5. **Patch Store**: Add `liquidity_score` column and write back

#### Performance Optimizations
- **Progress Bar**: TQDM for long-running computation
- **Vectorization**: Partially vectorized (pillars), row-by-row for composition
- **Error Handling**: Graceful degradation (set to `0.5` on error)
- **Memory**: Process in-place (no intermediate DataFrames)

#### CLI Usage
```bash
# Full backfill (all rows)
python3 bin/backfill_liquidity_score.py

# Dry run (compute but don't write)
python3 bin/backfill_liquidity_score.py --dry-run

# Custom MTF store
python3 bin/backfill_liquidity_score.py \
    --mtf-store data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet

# Short side (for bear patterns)
python3 bin/backfill_liquidity_score.py --side short
```

#### Pseudocode
```python
def main():
    # 1. Load MTF store
    mtf_df = pd.read_parquet(args.mtf_store)

    # 2. Compute liquidity scores (batch)
    liquidity_scores = pd.Series(index=mtf_df.index, dtype=float)

    for idx, row in tqdm(mtf_df.iterrows(), total=len(mtf_df)):
        ctx = map_mtf_row_to_context(row)  # Map features
        score = compute_liquidity_score(ctx, side=args.side)
        liquidity_scores[idx] = score

    # 3. Validate distribution
    validation = validate_liquidity_distribution(liquidity_scores)
    if not validation['relaxed_passed']:
        warn("Distribution anomalies detected")

    # 4. Patch MTF store
    mtf_df['liquidity_score'] = liquidity_scores

    if not args.dry_run:
        mtf_df.to_parquet(args.mtf_store)
        print(f"✅ Patched {len(mtf_df)} rows")

    return 0 if validation['relaxed_passed'] else 1
```

---

## 5. Execution Plan

### Pre-Flight Checklist
- [ ] Verify MTF store exists: `data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet`
- [ ] Check for required input features (run `validate_feature_store.py`)
- [ ] Backup MTF store before patching

### Backfill Steps
```bash
# 1. Dry run (validation only)
python3 bin/backfill_liquidity_score.py --dry-run

# 2. Review validation output
#    - Check distribution percentiles
#    - Verify no errors during computation

# 3. Full backfill (write to store)
python3 bin/backfill_liquidity_score.py

# 4. Validate patched store
python3 bin/validate_feature_store.py \
    --mtf-store data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet \
    --check liquidity_score
```

### Post-Backfill Validation
1. **Check distribution**: Median ~0.45-0.55, p75 ~0.68-0.75, p90 ~0.80-0.90
2. **Spot check**: Manually inspect 10 random rows for sensible values
3. **Integration test**: Run S1 archetype with new feature (expect non-zero matches)

### Performance Expectations
- **Rows**: ~26,000 (3 years of 1H data)
- **Time**: ~10-20 minutes (depends on feature complexity)
- **Memory**: ~500 MB (single MTF store in RAM)

---

## 6. Known Issues & Mitigations

### Issue 1: Slow Row-by-Row Computation
**Mitigation**: Future optimization could vectorize pillar calculations:
- Compute all `S`, `C`, `L`, `P` pillars as DataFrame columns
- Combine vectorized (avoids Python loop)
- **Current priority**: LOW (backfill is one-time operation)

### Issue 2: Missing `fvg_quality` Feature
**Mitigation**: Fallback to binary `fvg_present`:
```python
if 'fvg_quality' not in row:
    fvg_quality = 1.0 if row.get('tf1h_fvg_present', False) else 0.0
```

### Issue 3: Missing `range_eq` Feature
**Mitigation**: Compute from rolling high/low:
```python
if 'range_eq' not in row:
    rolling_high = row.get('rolling_high', row['high'])
    rolling_low = row.get('rolling_low', row['low'])
    range_eq = (rolling_high + rolling_low) / 2.0
```

---

## 7. Success Criteria

### Functional Requirements
- ✅ `liquidity_score` column added to MTF store
- ✅ 100% non-null coverage (defaults applied)
- ✅ All values in `[0.0, 1.0]` range

### Quality Requirements
- ✅ Distribution matches target percentiles (relaxed validation)
- ✅ S1 archetype produces non-zero matches (integration test)
- ✅ No runtime errors during backfill

### Performance Requirements
- ✅ Backfill completes in <30 minutes
- ✅ No memory errors (stays under 1 GB)

---

## 8. Dependencies

### Upstream (Must Exist)
- MTF feature store with required input features
- Runtime liquidity scorer (`engine/liquidity/score.py`)

### Downstream (Unblocked by This)
- S1 (Liquidity Vacuum) archetype
- S4 (Distribution Climax) archetype
- S5 (Long Squeeze) archetype

---

## 9. References

- **Runtime Logic**: `engine/liquidity/score.py`
- **Backfill Script**: `bin/backfill_liquidity_score.py`
- **Optimized Version**: `bin/backfill_liquidity_score_optimized.py` (vectorized)
- **Feature Registry**: `engine/features/registry.py`
- **Validation Tests**: `tests/unit/test_liquidity_score.py`
