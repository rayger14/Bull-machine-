#!/usr/bin/env python3
"""
Validator for Knowledge v10 Multi-Timeframe Feature Stores

Catches common issues:
- Flat/hardcoded fusion scores (k2_fusion_score = 0.5 bug)
- Missing required columns
- NaN cascades
- Invalid dtypes
- Warmup section issues

Usage:
    python3 bin/validate_feature_store_v10.py --file data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np


class FeatureStoreValidator:
    """Validates Knowledge v10 feature stores"""

    # Required columns that MUST exist
    REQUIRED_CORE = [
        'open', 'high', 'low', 'close', 'volume',
        'k2_fusion_score', 'tf1h_fusion_score', 'tf1d_fusion_score'
    ]

    # Fusion scores that must have variance (catch hardcoded values)
    FUSION_SCORES = [
        'k2_fusion_score',
        'tf1h_fusion_score',
        'tf1d_fusion_score',
        'tf4h_fusion_score'
    ]

    # Minimum unique values for fusion scores (catch flat distributions)
    MIN_UNIQUE_VALUES = {
        'k2_fusion_score': 100,      # Should have lots of variance
        'tf1h_fusion_score': 50,     # Decent variance
        'tf1d_fusion_score': 20,     # Lower variance OK (daily changes slower)
        'tf4h_fusion_score': 30      # Moderate variance
    }

    # Maximum allowed NaN percentage
    MAX_NAN_PCT = {
        'core_ohlcv': 0.0,           # OHLCV cannot have NaNs
        'fusion_scores': 5.0,        # Fusion scores: max 5% NaN (warmup period)
        'indicators': 10.0           # Indicators: max 10% NaN (lookback warmup)
    }

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.df = None
        self.errors = []
        self.warnings = []

    def validate(self) -> bool:
        """Run all validation checks. Returns True if valid, False otherwise."""

        # Check 1: File exists
        if not self.file_path.exists():
            self.errors.append(f"File does not exist: {self.file_path}")
            return False

        # Check 2: Can read parquet
        try:
            self.df = pd.read_parquet(self.file_path)
        except Exception as e:
            self.errors.append(f"Cannot read parquet file: {e}")
            return False

        # Check 3: Not empty
        if len(self.df) == 0:
            self.errors.append("Feature store is empty (0 rows)")
            return False

        # Run validation checks
        self._check_required_columns()
        self._check_fusion_score_variance()
        self._check_nan_levels()
        self._check_dtypes()
        self._check_value_ranges()

        # Return True if no errors
        return len(self.errors) == 0

    def _check_required_columns(self):
        """Verify all required columns exist"""
        missing = [col for col in self.REQUIRED_CORE if col not in self.df.columns]
        if missing:
            self.errors.append(f"Missing required columns: {missing}")

    def _check_fusion_score_variance(self):
        """Catch hardcoded/flat fusion scores (THE BUG WE FIXED!)"""

        for col in self.FUSION_SCORES:
            if col not in self.df.columns:
                continue  # Optional column, skip

            series = self.df[col].dropna()

            if len(series) == 0:
                self.warnings.append(f"{col}: All NaN (no valid data)")
                continue

            unique_count = series.nunique()
            min_required = self.MIN_UNIQUE_VALUES.get(col, 10)

            # Critical check: catch hardcoded values
            if unique_count == 1:
                value = series.iloc[0]
                self.errors.append(
                    f"{col}: HARDCODED to {value:.3f} (all {len(series)} rows identical)"
                )
            elif unique_count < min_required:
                self.errors.append(
                    f"{col}: Low variance ({unique_count} unique values, expected >{min_required})"
                )
            else:
                # Good variance - informational only
                mean = series.mean()
                std = series.std()
                self.warnings.append(
                    f"{col}: ✅ mean={mean:.3f}, std={std:.3f}, unique={unique_count}"
                )

    def _check_nan_levels(self):
        """Check for excessive NaN values"""

        # Core OHLCV: ZERO NaNs allowed
        core_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in core_cols:
            if col not in self.df.columns:
                continue
            nan_pct = 100.0 * self.df[col].isna().sum() / len(self.df)
            if nan_pct > self.MAX_NAN_PCT['core_ohlcv']:
                self.errors.append(f"{col}: {nan_pct:.1f}% NaN (OHLCV must be complete)")

        # Fusion scores: Low NaN tolerance
        fusion_cols = [c for c in self.FUSION_SCORES if c in self.df.columns]
        for col in fusion_cols:
            nan_pct = 100.0 * self.df[col].isna().sum() / len(self.df)
            if nan_pct > self.MAX_NAN_PCT['fusion_scores']:
                self.errors.append(f"{col}: {nan_pct:.1f}% NaN (max {self.MAX_NAN_PCT['fusion_scores']}%)")

    def _check_dtypes(self):
        """Verify column dtypes are numeric"""

        for col in self.REQUIRED_CORE:
            if col not in self.df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                self.errors.append(f"{col}: Non-numeric dtype ({self.df[col].dtype})")

    def _check_value_ranges(self):
        """Sanity check value ranges"""

        # Fusion scores must be in [0, 1]
        for col in self.FUSION_SCORES:
            if col not in self.df.columns:
                continue

            series = self.df[col].dropna()
            if len(series) == 0:
                continue

            min_val = series.min()
            max_val = series.max()

            if min_val < 0.0 or max_val > 1.0:
                self.errors.append(
                    f"{col}: Out of range [0,1] (min={min_val:.3f}, max={max_val:.3f})"
                )

        # OHLCV must be positive
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in self.df.columns:
                continue
            if (self.df[col] <= 0).any():
                self.errors.append(f"{col}: Contains non-positive values")

    def print_summary(self):
        """Print validation summary"""

        if self.df is not None:
            print(f"\n{'='*70}")
            print(f"Feature Store Validation: {self.file_path.name}")
            print(f"{'='*70}")
            print(f"Shape: {self.df.shape[0]:,} bars × {self.df.shape[1]} features")
            print(f"Date range: {self.df.index[0]} to {self.df.index[-1]}")
            print()

        # Print warnings (informational)
        if self.warnings:
            print("ℹ️  Info:")
            for warning in self.warnings:
                print(f"   {warning}")
            print()

        # Print errors (critical)
        if self.errors:
            print("❌ VALIDATION FAILED:")
            for error in self.errors:
                print(f"   • {error}")
            print(f"\n{'='*70}")
            print("FAIL: Feature store has critical issues")
            print(f"{'='*70}\n")
        else:
            print(f"{'='*70}")
            print("✅ PASS: Feature store validation successful")
            print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Validate Knowledge v10 Multi-Timeframe Feature Store'
    )
    parser.add_argument(
        '--file',
        required=True,
        help='Path to feature store parquet file'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only print one-line pass/fail result'
    )

    args = parser.parse_args()

    # Run validation
    validator = FeatureStoreValidator(args.file)
    is_valid = validator.validate()

    if args.quiet:
        # One-line output for CI/CD
        if is_valid:
            print(f"✅ PASS: {Path(args.file).name}")
            sys.exit(0)
        else:
            error_summary = validator.errors[0] if validator.errors else "Unknown error"
            print(f"❌ FAIL: {Path(args.file).name} - {error_summary}")
            sys.exit(1)
    else:
        # Full validation report
        validator.print_summary()
        sys.exit(0 if is_valid else 1)


if __name__ == '__main__':
    main()
