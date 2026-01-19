# Data Validation Framework

**Purpose**: Systematic validation for all Phase 1 data backfills
**Scope**: Pre-backfill checks, post-backfill validation, audit reporting
**Integration**: Used by all backfill scripts (`bin/backfill_*.py`)

---

## 1. Pre-Backfill Checks

### File Existence & Readability

```python
def validate_file_exists(file_path: Path) -> Dict[str, Any]:
    """
    Validate file exists and is readable.

    Returns:
        Dict with:
        - exists: bool
        - readable: bool
        - size_mb: float
        - error: str (if any)
    """
    result = {
        'exists': False,
        'readable': False,
        'size_mb': 0.0,
        'error': None
    }

    try:
        # Check existence
        if not file_path.exists():
            result['error'] = f"File not found: {file_path}"
            return result

        result['exists'] = True

        # Check readability
        if not file_path.is_file():
            result['error'] = f"Path is not a file: {file_path}"
            return result

        # Check size
        size_bytes = file_path.stat().st_size
        result['size_mb'] = size_bytes / (1024 * 1024)
        result['readable'] = True

    except Exception as e:
        result['error'] = str(e)

    return result
```

### Required Columns Check

```python
def validate_required_columns(
    df: pd.DataFrame,
    required_cols: list,
    optional_cols: list = None
) -> Dict[str, Any]:
    """
    Validate DataFrame has required columns.

    Args:
        df: Input DataFrame
        required_cols: List of required column names
        optional_cols: List of optional column names (for reporting)

    Returns:
        Dict with:
        - has_required: bool (all required cols present)
        - missing_required: list (missing required cols)
        - has_optional: list (present optional cols)
        - missing_optional: list (missing optional cols)
    """
    result = {
        'has_required': False,
        'missing_required': [],
        'has_optional': [],
        'missing_optional': []
    }

    # Check required columns
    missing_req = [col for col in required_cols if col not in df.columns]
    result['missing_required'] = missing_req
    result['has_required'] = len(missing_req) == 0

    # Check optional columns (for reporting)
    if optional_cols:
        has_opt = [col for col in optional_cols if col in df.columns]
        missing_opt = [col for col in optional_cols if col not in df.columns]
        result['has_optional'] = has_opt
        result['missing_optional'] = missing_opt

    return result
```

### Timestamp Index Check

```python
def validate_timestamp_index(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate DataFrame has proper timestamp index.

    Returns:
        Dict with:
        - is_datetime_index: bool
        - is_sorted: bool
        - has_duplicates: bool
        - is_tz_aware: bool
        - timezone: str
        - first_timestamp: datetime
        - last_timestamp: datetime
        - total_rows: int
    """
    result = {
        'is_datetime_index': False,
        'is_sorted': False,
        'has_duplicates': False,
        'is_tz_aware': False,
        'timezone': None,
        'first_timestamp': None,
        'last_timestamp': None,
        'total_rows': len(df)
    }

    # Check if index is DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        result['is_datetime_index'] = True
        result['is_sorted'] = df.index.is_monotonic_increasing
        result['has_duplicates'] = df.index.has_duplicates
        result['is_tz_aware'] = df.index.tz is not None
        result['timezone'] = str(df.index.tz) if df.index.tz else None
        result['first_timestamp'] = df.index[0]
        result['last_timestamp'] = df.index[-1]

    return result
```

---

## 2. Post-Backfill Validation

### NaN Coverage Check

```python
def validate_nan_coverage(
    df: pd.DataFrame,
    columns: list,
    max_nan_pct: float = 5.0
) -> Dict[str, Any]:
    """
    Check for NaN values in specified columns.

    Args:
        df: Input DataFrame
        columns: Columns to check
        max_nan_pct: Maximum acceptable NaN percentage (default 5%)

    Returns:
        Dict with per-column NaN statistics
    """
    results = {}

    for col in columns:
        if col not in df.columns:
            results[col] = {'error': 'Column not found'}
            continue

        total = len(df)
        null_count = df[col].isna().sum()
        null_pct = (null_count / total) * 100.0

        results[col] = {
            'total_rows': total,
            'null_count': null_count,
            'null_pct': null_pct,
            'acceptable': null_pct <= max_nan_pct,
            'non_null_count': total - null_count
        }

    return results
```

### Value Range Check

```python
def validate_value_ranges(
    df: pd.DataFrame,
    column_ranges: Dict[str, tuple]
) -> Dict[str, Any]:
    """
    Validate column values are within expected ranges.

    Args:
        df: Input DataFrame
        column_ranges: Dict mapping column names to (min, max) tuples
                      Use None for unbounded (e.g., (None, 1.0) for max only)

    Returns:
        Dict with per-column range validation results

    Example:
        ranges = {
            'liquidity_score': (0.0, 1.0),
            'oi_change_pct_24h': (-50.0, 50.0),
            'dxy_z': (None, None)  # Unbounded
        }
    """
    results = {}

    for col, (min_val, max_val) in column_ranges.items():
        if col not in df.columns:
            results[col] = {'error': 'Column not found'}
            continue

        series = df[col].dropna()  # Ignore NaNs for range check

        actual_min = series.min()
        actual_max = series.max()

        violations = 0
        if min_val is not None:
            violations += (series < min_val).sum()
        if max_val is not None:
            violations += (series > max_val).sum()

        results[col] = {
            'actual_min': actual_min,
            'actual_max': actual_max,
            'expected_min': min_val,
            'expected_max': max_val,
            'violations': violations,
            'in_range': violations == 0
        }

    return results
```

### Distribution Sanity Check

```python
def validate_distribution(
    df: pd.DataFrame,
    column: str,
    expected_stats: Dict[str, tuple] = None
) -> Dict[str, Any]:
    """
    Validate column distribution matches expectations.

    Args:
        df: Input DataFrame
        column: Column to check
        expected_stats: Dict with expected (min, max) ranges for:
                       - 'mean': (min, max)
                       - 'median': (min, max)
                       - 'std': (min, max)
                       - 'p25': (min, max)
                       - 'p75': (min, max)
                       - 'p90': (min, max)

    Returns:
        Dict with distribution statistics and validation flags

    Example:
        expected = {
            'median': (0.45, 0.55),
            'p75': (0.68, 0.75),
            'p90': (0.80, 0.90)
        }
    """
    if column not in df.columns:
        return {'error': 'Column not found'}

    series = df[column].dropna()

    # Calculate statistics
    stats = {
        'count': len(series),
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'p25': series.quantile(0.25),
        'p50': series.quantile(0.50),
        'p75': series.quantile(0.75),
        'p90': series.quantile(0.90),
        'p95': series.quantile(0.95),
        'p99': series.quantile(0.99)
    }

    # Validate against expected ranges
    validation = {'all_passed': True}

    if expected_stats:
        for stat_name, (min_val, max_val) in expected_stats.items():
            actual_val = stats.get(stat_name)
            if actual_val is None:
                continue

            passed = (min_val <= actual_val <= max_val)
            validation[f'{stat_name}_ok'] = passed
            validation['all_passed'] = validation['all_passed'] and passed

    stats['validation'] = validation
    return stats
```

---

## 3. Audit Report Template

### Report Structure

```python
def generate_backfill_audit_report(
    feature_name: str,
    df: pd.DataFrame,
    validation_results: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Generate comprehensive audit report for backfill.

    Args:
        feature_name: Name of feature (e.g., 'liquidity_score')
        df: Backfilled DataFrame
        validation_results: Dict with all validation results
        output_path: Path to save report (markdown file)
    """
    report = []

    # Header
    report.append(f"# Backfill Audit Report: {feature_name}")
    report.append(f"\n**Date**: {datetime.now(timezone.utc).isoformat()}")
    report.append(f"**Total Rows**: {len(df):,}")
    report.append(f"\n---\n")

    # 1. Coverage Summary
    report.append("## 1. Coverage Summary")
    coverage = validation_results.get('coverage', {})
    for col, stats in coverage.items():
        null_pct = stats.get('null_pct', 0.0)
        status = "✅" if stats.get('acceptable', False) else "❌"
        report.append(f"\n{status} **{col}**: {null_pct:.2f}% NaN ({stats.get('non_null_count', 0):,} / {stats.get('total_rows', 0):,})")

    # 2. Value Range Validation
    report.append("\n\n## 2. Value Range Validation")
    ranges = validation_results.get('ranges', {})
    for col, stats in ranges.items():
        if 'error' in stats:
            report.append(f"\n❌ **{col}**: {stats['error']}")
            continue

        status = "✅" if stats.get('in_range', False) else "❌"
        report.append(f"\n{status} **{col}**:")
        report.append(f"  - Actual: [{stats['actual_min']:.3f}, {stats['actual_max']:.3f}]")
        if stats['expected_min'] is not None or stats['expected_max'] is not None:
            report.append(f"  - Expected: [{stats['expected_min']}, {stats['expected_max']}]")
        if stats['violations'] > 0:
            report.append(f"  - Violations: {stats['violations']:,} rows")

    # 3. Distribution Statistics
    report.append("\n\n## 3. Distribution Statistics")
    dist = validation_results.get('distribution', {})
    for col, stats in dist.items():
        if 'error' in stats:
            continue

        report.append(f"\n**{col}**:")
        report.append(f"  - Count: {stats['count']:,}")
        report.append(f"  - Mean: {stats['mean']:.4f}")
        report.append(f"  - Median: {stats['median']:.4f}")
        report.append(f"  - Std: {stats['std']:.4f}")
        report.append(f"  - Min: {stats['min']:.4f}")
        report.append(f"  - Max: {stats['max']:.4f}")
        report.append(f"  - Percentiles:")
        report.append(f"    - p25: {stats['p25']:.4f}")
        report.append(f"    - p50: {stats['p50']:.4f}")
        report.append(f"    - p75: {stats['p75']:.4f}")
        report.append(f"    - p90: {stats['p90']:.4f}")

        # Validation flags
        val = stats.get('validation', {})
        if val.get('all_passed', False):
            report.append(f"  - ✅ All validation checks passed")
        else:
            report.append(f"  - ❌ Some validation checks failed")

    # 4. Histogram (Text-based)
    report.append("\n\n## 4. Distribution Histogram")
    for col in validation_results.get('histogram_cols', []):
        if col not in df.columns:
            continue
        report.append(f"\n**{col}**:")
        hist = generate_text_histogram(df[col].dropna(), bins=20)
        report.append(hist)

    # 5. Sample Rows
    report.append("\n\n## 5. Sample Rows")
    report.append("\n**First 5 rows:**")
    report.append(df.head(5).to_markdown())
    report.append("\n**Last 5 rows:**")
    report.append(df.tail(5).to_markdown())

    # 6. Edge Cases
    report.append("\n\n## 6. Edge Cases")
    edge_cases = validation_results.get('edge_cases', {})
    for case_name, case_data in edge_cases.items():
        report.append(f"\n**{case_name}**:")
        report.append(f"  - Detected: {case_data.get('detected', False)}")
        report.append(f"  - Details: {case_data.get('details', 'N/A')}")

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(report))
    print(f"✅ Audit report written to: {output_path}")
```

### Text-Based Histogram

```python
def generate_text_histogram(series: pd.Series, bins: int = 20) -> str:
    """Generate ASCII histogram for terminal display"""
    counts, bin_edges = np.histogram(series, bins=bins)
    max_count = counts.max()

    lines = []
    for i, count in enumerate(counts):
        bar_length = int((count / max_count) * 50) if max_count > 0 else 0
        bar = '█' * bar_length
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        lines.append(f"  [{bin_start:7.3f}, {bin_end:7.3f}): {bar} {count}")

    return '\n'.join(lines)
```

---

## 4. Integration Tests

### Sample Data Validation

```python
def validate_sample_data(
    df: pd.DataFrame,
    feature_name: str,
    sample_size: int = 100
) -> Dict[str, Any]:
    """
    Validate random sample of data for manual inspection.

    Args:
        df: Input DataFrame
        feature_name: Column to validate
        sample_size: Number of rows to sample

    Returns:
        Dict with sample statistics and flagged anomalies
    """
    if feature_name not in df.columns:
        return {'error': 'Feature not found'}

    # Random sample
    sample = df[feature_name].dropna().sample(min(sample_size, len(df)))

    # Detect anomalies (values >3 std from mean)
    mean = sample.mean()
    std = sample.std()
    anomalies = sample[abs(sample - mean) > 3 * std]

    return {
        'sample_size': len(sample),
        'sample_mean': mean,
        'sample_std': std,
        'anomaly_count': len(anomalies),
        'anomaly_pct': (len(anomalies) / len(sample)) * 100.0,
        'anomaly_values': anomalies.tolist()[:10]  # First 10 anomalies
    }
```

### Pre-Backfill Integration Test

```python
def integration_test_pre_backfill(mtf_path: Path) -> bool:
    """
    Run pre-backfill integration test.

    Checks:
    1. MTF store exists and is readable
    2. Required input features exist
    3. Timestamp index is valid
    4. No major data corruption

    Returns:
        True if all checks pass, False otherwise
    """
    print("\n" + "=" * 80)
    print("PRE-BACKFILL INTEGRATION TEST")
    print("=" * 80)

    all_passed = True

    # 1. File check
    file_check = validate_file_exists(mtf_path)
    if not file_check['readable']:
        print(f"❌ File check failed: {file_check['error']}")
        return False
    print(f"✅ File exists and is readable ({file_check['size_mb']:.2f} MB)")

    # 2. Load DataFrame
    try:
        df = pd.read_parquet(mtf_path)
        print(f"✅ Loaded DataFrame: {df.shape}")
    except Exception as e:
        print(f"❌ Failed to load parquet: {e}")
        return False

    # 3. Timestamp index check
    ts_check = validate_timestamp_index(df)
    if not ts_check['is_datetime_index']:
        print(f"❌ Index is not DatetimeIndex")
        all_passed = False
    else:
        print(f"✅ Valid DatetimeIndex: {ts_check['first_timestamp']} to {ts_check['last_timestamp']}")

    if ts_check['has_duplicates']:
        print(f"⚠️ Warning: Index has duplicates")
        all_passed = False

    if not ts_check['is_sorted']:
        print(f"⚠️ Warning: Index is not sorted")
        all_passed = False

    # 4. Required columns check (OHLCV)
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    col_check = validate_required_columns(df, required_cols)
    if not col_check['has_required']:
        print(f"❌ Missing required columns: {col_check['missing_required']}")
        all_passed = False
    else:
        print(f"✅ All required OHLCV columns present")

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ PRE-BACKFILL TEST PASSED")
    else:
        print("❌ PRE-BACKFILL TEST FAILED")
    print("=" * 80)

    return all_passed
```

---

## 5. Usage Examples

### Example 1: Liquidity Score Backfill Validation

```python
# Pre-backfill check
if not integration_test_pre_backfill(mtf_path):
    print("❌ Pre-backfill checks failed, aborting")
    sys.exit(1)

# Perform backfill
mtf_df['liquidity_score'] = compute_liquidity_scores_batch(mtf_df)

# Post-backfill validation
validation_results = {
    'coverage': validate_nan_coverage(mtf_df, ['liquidity_score'], max_nan_pct=1.0),
    'ranges': validate_value_ranges(mtf_df, {'liquidity_score': (0.0, 1.0)}),
    'distribution': {
        'liquidity_score': validate_distribution(
            mtf_df, 'liquidity_score',
            expected_stats={
                'median': (0.45, 0.55),
                'p75': (0.68, 0.75),
                'p90': (0.80, 0.90)
            }
        )
    },
    'histogram_cols': ['liquidity_score']
}

# Generate audit report
generate_backfill_audit_report(
    'liquidity_score',
    mtf_df,
    validation_results,
    Path('results/liquidity_score_audit.md')
)
```

### Example 2: OI Pipeline Validation

```python
# Validate OI metrics against known events
validation_results = {
    'coverage': validate_nan_coverage(
        oi_df, ['oi', 'oi_change_24h', 'oi_change_pct_24h', 'oi_z'],
        max_nan_pct=5.0
    ),
    'ranges': validate_value_ranges(
        oi_df, {
            'oi': (0.0, None),  # OI must be non-negative
            'oi_change_pct_24h': (-50.0, 50.0),  # Reasonable change range
            'oi_z': (-5.0, 5.0)  # Z-score rarely exceeds ±5
        }
    ),
    'distribution': {
        'oi_z': validate_distribution(
            oi_df, 'oi_z',
            expected_stats={
                'mean': (-0.5, 0.5),  # Should be ~0
                'std': (0.7, 1.3)     # Should be ~1
            }
        )
    },
    'edge_cases': {
        'terra_collapse': {
            'detected': (oi_df['oi_change_pct_24h'] < -15.0).any(),
            'details': f"Min OI change: {oi_df['oi_change_pct_24h'].min():.2f}%"
        },
        'ftx_collapse': {
            'detected': (oi_df['oi_change_pct_24h'] < -20.0).any(),
            'details': f"FTX period min: {oi_df.loc['2022-11-08':'2022-11-10', 'oi_change_pct_24h'].min():.2f}%"
        }
    }
}

generate_backfill_audit_report(
    'oi_pipeline',
    oi_df,
    validation_results,
    Path('results/oi_pipeline_audit.md')
)
```

---

## 6. Validation Checklist

### Universal Pre-Backfill Checklist
- [ ] MTF store file exists and is readable
- [ ] MTF store has valid DatetimeIndex (sorted, no duplicates)
- [ ] Required OHLCV columns present
- [ ] Backup of MTF store created
- [ ] Sufficient disk space (2x MTF store size)

### Universal Post-Backfill Checklist
- [ ] New feature column(s) added to MTF store
- [ ] NaN coverage <5% (or within expected range)
- [ ] Value ranges validated (no out-of-bounds values)
- [ ] Distribution statistics look sane (no extreme skew)
- [ ] Audit report generated
- [ ] Sample rows manually inspected (spot check)

### Feature-Specific Checklist
- **Liquidity Score**:
  - [ ] Distribution: median 0.45-0.55, p75 0.68-0.75, p90 0.80-0.90
  - [ ] All values in [0.0, 1.0]
  - [ ] S1 archetype produces non-zero matches

- **OI Pipeline**:
  - [ ] Terra collapse detected (May 2022)
  - [ ] FTX collapse detected (Nov 2022)
  - [ ] Normal range 85%+ data in ±5%
  - [ ] S5 archetype produces non-zero matches

- **Macro Features**:
  - [ ] Z-scores have mean ~0, std ~1
  - [ ] Coverage >95% (forward fill applied)
  - [ ] Crisis events detected (Terra, FTX, SVB)
  - [ ] Regime routing works without NaN errors

---

## 7. References

- **Validation Scripts**: `bin/validate_feature_store.py`
- **Test Suite**: `tests/unit/test_data_validation.py`
- **Audit Reports**: `results/*_audit.md`
- **Pre-Backfill Tests**: `tests/integration/test_pre_backfill.py`
