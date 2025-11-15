# PR#1: Infrastructure & Safety

**Branch**: `pr1-infra-patch-tool` → `integrate/v4-prep`
**Type**: Infrastructure
**Status**: Ready for Review
**Part of**: 5-PR Integration Sequence

---

## Overview

PR#1 establishes infrastructure for safely patching feature store columns without full rebuilds. This is foundational work for the upcoming 5-PR sequence that will integrate:
- Feature calculation fixes (BOMS displacement/strength, fusion scoring)
- Runtime liquidity scorer
- Phase 4 re-entry gates
- 3-Archetype entry system

**All new behavior is OFF by default** - this PR only adds tools, not new runtime behavior.

---

## What's Included

### 1. Production-Ready Patch Tool (`bin/patch_feature_columns.py`)

**Purpose**: Patch specific columns in existing feature stores without full rebuild

**Architecture** (5 logical sections):
1. **Core loader/writer**: Parquet I/O with atomic saves (tmp → os.replace)
2. **Column registry**: Metadata for patchable columns (P0: displacement, strength, fusion)
3. **Calculator functions**: Modular recomputation logic for each column
4. **Health checks**: JSON output with non-zero rates and validation
5. **CLI wrapper**: argparse with health-only and patch modes

**Key Features**:
- `--health-only` mode for CI validation (no modifications)
- Atomic writes with automatic backups
- JSON health output for programmatic validation
- Modular calculators for easy extension
- Expected non-zero thresholds for quality gates

**Usage**:
```bash
# Health check only (CI)
python bin/patch_feature_columns.py \
    --asset BTC --tf 1H --start 2024-01-01 --end 2024-12-31 \
    --health-only

# Patch specific columns
python bin/patch_feature_columns.py \
    --asset BTC --tf 1H --start 2024-01-01 --end 2024-12-31 \
    --cols tf4h_boms_displacement,tf1d_boms_strength,tf4h_fusion_score
```

### 2. CI Smoke Test (`tests/test_patch_tool_smoke.sh`)

**Purpose**: Validate patch tool in CI without modifying data

**Validations**:
- Tool can load existing feature stores
- Health-only mode produces valid JSON
- JSON contains required keys: `timestamp`, `total_rows`, `columns_patched`, `metrics`, `health_checks`
- Metrics exist for all P0 columns with `non_zero_pct` field
- Gracefully skips if feature store doesn't exist (not a failure)

**Exit Codes**:
- `0` = success
- `1` = failure

**Usage**:
```bash
bash tests/test_patch_tool_smoke.sh
```

### 3. Cleanup

**Removed**:
- `bin/fix_feature_columns_inplace.py` - replaced by production tool

---

## P0 Columns (Supported)

Currently patchable columns with calculators:

| Column | Description | Expected Non-Zero | Calculator |
|--------|-------------|-------------------|------------|
| `tf4h_boms_displacement` | BOMS displacement on 4H (absolute price) | > 5% | `patch_boms_displacement()` |
| `tf1d_boms_strength` | BOMS strength on 1D (normalized 0-1) | > 5% | `patch_boms_strength()` |
| `tf4h_fusion_score` | Fusion score from 4H indicators (0-1) | > 15% | `patch_tf4h_fusion()` |

---

## Health Check Example

```json
{
  "timestamp": "2025-10-23T12:07:10.147736",
  "total_rows": 8761,
  "columns_patched": [
    "tf4h_boms_displacement",
    "tf1d_boms_strength",
    "tf4h_fusion_score"
  ],
  "metrics": {
    "tf4h_boms_displacement": {
      "non_zero_count": 0,
      "non_zero_pct": 0.0,
      "min": 0.0,
      "max": 0.0,
      "mean": 0.0,
      "p50": 0.0,
      "p75": 0.0,
      "p95": 0.0
    },
    "tf1d_boms_strength": {
      "non_zero_count": 288,
      "non_zero_pct": 3.29,
      "min": 0.0,
      "max": 1.0,
      "mean": 0.020022643925226097,
      "p50": 0.0,
      "p75": 0.0,
      "p95": 0.0,
      "non_zero_mean": 0.6090916091281452,
      "non_zero_p50": 0.5862755761036653,
      "non_zero_p95": 1.0
    },
    "tf4h_fusion_score": {
      "non_zero_count": 1636,
      "non_zero_pct": 18.67,
      "min": -0.11255802745532657,
      "max": 0.3009464866082064,
      "mean": 0.0290093363503027,
      "p50": 0.0,
      "p75": 0.0,
      "p95": 0.22378110763859255,
      "non_zero_mean": 0.15534889716687159,
      "non_zero_p50": 0.20117810448245674,
      "non_zero_p95": 0.27176611715485105
    }
  },
  "health_checks": {
    "tf4h_boms_displacement": "FAIL",
    "tf1d_boms_strength": "FAIL",
    "tf4h_fusion_score": "PASS"
  }
}
```

---

## Testing

### Manual Testing Performed

1. **Health-only mode**:
   - Ran on BTC 2024 feature store (8,761 rows × 80 columns)
   - JSON output validated for all P0 columns
   - Health checks correctly identify FAIL/PASS status

2. **CI smoke test**:
   - Validates JSON structure and required keys
   - Confirms metrics for all P0 columns
   - Passes with existing feature stores

### CI Integration

Add to `.github/workflows/test.yml`:
```yaml
- name: Run patch tool smoke test
  run: bash tests/test_patch_tool_smoke.sh
```

---

## Files Changed

```
bin/patch_feature_columns.py          +383 (new production tool)
bin/fix_feature_columns_inplace.py    -154 (removed legacy script)
tests/test_patch_tool_smoke.sh        +89  (new CI smoke test)
```

**Total**: +472 lines added, -154 lines removed

---

## Dependencies

**Python Packages** (already in project):
- `pandas` - Parquet I/O
- `numpy` - Numerical operations

**Project Modules**:
- `engine.structure.boms_detector.detect_boms` - BOMS displacement/strength calculation

---

## Future Extensions (PR#2+)

This tool is designed for easy extension. To add a new patchable column:

1. Add entry to `COLUMN_REGISTRY`:
```python
COLUMN_REGISTRY = {
    'new_column_name': {
        'description': 'What this column does',
        'expected_nonzero_min': 0.10,  # Expected > 10% non-zero
        'calculator': 'patch_new_column'
    },
}
```

2. Implement calculator function:
```python
def patch_new_column(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute new_column_name."""
    logging.info("Patching new_column_name...")
    # Your calculation logic here
    df['new_column_name'] = calculate_values()
    return df
```

3. Register in `COLUMN_CALCULATORS`:
```python
COLUMN_CALCULATORS = {
    'new_column_name': patch_new_column,
}
```

---

## Checklist for Reviewers

- [ ] Tool can load existing feature stores without errors
- [ ] Health-only mode produces valid JSON with all required keys
- [ ] CI smoke test passes
- [ ] Atomic saves work correctly (backup created, tmp → final)
- [ ] Calculator functions are modular and well-documented
- [ ] Health checks correctly identify FAIL/PASS based on thresholds
- [ ] No new runtime behavior added (tool-only PR)
- [ ] Legacy patch scripts removed

---

## Next Steps

After PR#1 merges to `integrate/v4-prep`:

1. **PR#2**: Wire calculator functions to feature store builder
   - Add `engine/utils_align.py` for HTF→1H alignment
   - Add health assertions (no blanket `fillna(0)`)
   - Add `tests/test_feature_health.py`

2. **PR#3**: Runtime liquidity scorer
   - Compute `liquidity_score` at runtime (not stored)
   - Behind `runtime_liquidity_enabled` flag (OFF by default)

3. **PR#4**: Re-entry Gate-5 & assist exits
   - Phase 4 re-entry with sensible defaults
   - Convert structure/pattern exits to "assist mode"
   - All behind flags (OFF by default)

4. **PR#5**: 3-Archetype engine
   - Finalize A/B/C gates with adjusted thresholds
   - Behind `use_archetypes` + individual flags
   - Keep legacy path available

---

## Questions or Concerns?

Reach out with any questions about:
- Tool architecture or design decisions
- Health check thresholds
- CI integration
- Future extension plans
