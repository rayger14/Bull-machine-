#!/usr/bin/env python3
"""
Export canonical v10 feature store schema for future validation.
"""

import json
import pandas as pd
from pathlib import Path

def export_schema(parquet_path, output_path):
    """Extract schema from parquet file and save as JSON."""
    df = pd.read_parquet(parquet_path)

    schema = {
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'shape': df.shape,
        'index_name': df.index.name if df.index.name else 'timestamp',
        'required_columns': [
            'close', 'open', 'high', 'low', 'volume',
            'regime_label', 'regime_confidence',
            'funding_Z',
            'k2_fusion_score'
        ],
        'version': 'v10_baseline_corrected',
        'source_file': str(parquet_path)
    }

    # Check for missing values
    missing_stats = {}
    for col in df.columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            missing_stats[col] = {
                'null_count': int(null_count),
                'null_pct': float(null_count / len(df) * 100)
            }

    if missing_stats:
        schema['missing_values'] = missing_stats

    with open(output_path, 'w') as f:
        json.dump(schema, f, indent=2)

    return schema

def compare_schemas(schema1, schema2):
    """Compare two schemas and report differences."""
    cols1 = set(schema1['columns'])
    cols2 = set(schema2['columns'])

    only_in_1 = cols1 - cols2
    only_in_2 = cols2 - cols1
    common = cols1 & cols2

    report = {
        'common_columns': len(common),
        'total_columns': {
            'schema1': len(cols1),
            'schema2': len(cols2)
        },
        'only_in_schema1': sorted(list(only_in_1)),
        'only_in_schema2': sorted(list(only_in_2)),
        'column_agreement': len(common) / max(len(cols1), len(cols2))
    }

    return report

def main():
    # Export schemas for both stores
    print("Exporting schema for 2022-2023 feature store...")
    schema_2022_2023 = export_schema(
        'data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet',
        'schema/v10_feature_store_2022_2023.json'
    )
    print(f"  ✓ {len(schema_2022_2023['columns'])} columns, shape: {schema_2022_2023['shape']}")

    print("\nExporting schema for 2024 feature store...")
    schema_2024 = export_schema(
        'data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet',
        'schema/v10_feature_store_2024.json'
    )
    print(f"  ✓ {len(schema_2024['columns'])} columns, shape: {schema_2024['shape']}")

    # Compare schemas
    print("\nComparing schemas...")
    comparison = compare_schemas(schema_2022_2023, schema_2024)

    print(f"  Common columns: {comparison['common_columns']}")
    print(f"  Agreement: {comparison['column_agreement']*100:.1f}%")

    if comparison['only_in_schema1']:
        print(f"\n  Only in 2022-2023 ({len(comparison['only_in_schema1'])}):")
        for col in comparison['only_in_schema1'][:5]:
            print(f"    - {col}")
        if len(comparison['only_in_schema1']) > 5:
            print(f"    ... and {len(comparison['only_in_schema1'])-5} more")

    if comparison['only_in_schema2']:
        print(f"\n  Only in 2024 ({len(comparison['only_in_schema2'])}):")
        for col in comparison['only_in_schema2'][:5]:
            print(f"    - {col}")
        if len(comparison['only_in_schema2']) > 5:
            print(f"    ... and {len(comparison['only_in_schema2'])-5} more")

    # Create locked canonical schema (intersection)
    print("\nCreating canonical locked schema (common columns)...")
    canonical_cols = sorted(list(set(schema_2022_2023['columns']) & set(schema_2024['columns'])))

    canonical_schema = {
        'version': 'v10_baseline_corrected',
        'locked_date': '2025-11-05',
        'description': 'Canonical feature store schema for Router v10 baseline (2022-2024)',
        'columns': canonical_cols,
        'column_count': len(canonical_cols),
        'required_columns': [
            'timestamp', 'close', 'open', 'high', 'low', 'volume',
            'regime_label', 'regime_confidence',
            'funding_Z',
            'k2_fusion_score'
        ],
        'validation_rules': {
            'no_null_columns': ['timestamp', 'close', 'open', 'high', 'low', 'volume'],
            'regime_label_values': ['RISK_ON', 'RISK_OFF', 'NEUTRAL', 'CRISIS', 'TRANSITIONAL'],
            'funding_Z_max_null_pct': 1.0
        },
        'source_stores': {
            '2022-2023': {
                'columns': len(schema_2022_2023['columns']),
                'shape': schema_2022_2023['shape']
            },
            '2024': {
                'columns': len(schema_2024['columns']),
                'shape': schema_2024['shape']
            }
        },
        'comparison': comparison
    }

    with open('schema/v10_feature_store_locked.json', 'w') as f:
        json.dump(canonical_schema, f, indent=2)

    print(f"  ✓ Locked schema: {len(canonical_cols)} columns")
    print(f"  Saved to: schema/v10_feature_store_locked.json")

    # Save comparison report
    with open('schema/schema_comparison_report.json', 'w') as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "="*80)
    print("✓ Schema export complete!")
    print("="*80)

if __name__ == '__main__':
    main()
