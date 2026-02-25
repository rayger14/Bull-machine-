#!/usr/bin/env python3
"""
Feature Store Validator - Schema validation and contract enforcement.

Validates dataframes against feature registry before backtest/Optuna runs.
"""

import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass

from engine.features.registry import get_registry


@dataclass
class ValidationResult:
    """Result of validation checks."""
    passed: bool
    errors: List[str]
    warnings: List[str]

    def __str__(self) -> str:
        """Human-readable report."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"VALIDATION {'✅ PASSED' if self.passed else '❌ FAILED'}")
        lines.append("=" * 70)

        if self.errors:
            lines.append("\n❌ ERRORS:")
            for err in self.errors:
                lines.append(f"  - {err}")

        if self.warnings:
            lines.append("\n⚠️  WARNINGS:")
            for warn in self.warnings:
                lines.append(f"  - {warn}")

        lines.append("=" * 70)
        return "\n".join(lines)


class FeatureValidator:
    """
    Validates feature store data against registry schema.

    Contract validation includes:
    1. Required columns present
    2. Data types match specification
    3. Values within valid ranges
    4. Index is DatetimeIndex (not RangeIndex)
    5. No duplicates in index
    6. No nulls in required columns
    7. OHLC consistency (high >= low, etc.)
    """

    def __init__(self):
        self.registry = get_registry()

    def validate(
        self,
        df: pd.DataFrame,
        tier: int,
        strict: bool = True
    ) -> ValidationResult:
        """
        Validate dataframe against schema for given tier.

        Args:
            df: DataFrame to validate
            tier: Expected tier level (1, 2, or 3)
            strict: If True, treat warnings as errors

        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []

        # 1. Check index type
        if not isinstance(df.index, pd.DatetimeIndex):
            errors.append(
                f"Index must be DatetimeIndex, got {type(df.index).__name__}. "
                "This will cause slice failures in backtests."
            )

        # 2. Check for duplicate timestamps
        if df.index.duplicated().any():
            n_dups = df.index.duplicated().sum()
            errors.append(f"Found {n_dups} duplicate timestamps in index")

        # 3. Check monotonic increasing
        if not df.index.is_monotonic_increasing:
            errors.append("Index is not monotonic increasing")

        # 4. Check required columns present
        required_features = self.registry.get_required_features(tier)
        missing_required = []

        for feature in required_features:
            if feature not in df.columns:
                # Try to find via alias
                spec = self.registry.get_feature_spec(feature)
                found = False
                if spec:
                    for alias in spec.aliases:
                        if alias in df.columns:
                            found = True
                            warnings.append(
                                f"Using alias '{alias}' for required feature '{feature}'. "
                                "Consider normalizing to canonical name."
                            )
                            break

                if not found:
                    missing_required.append(feature)

        if missing_required:
            errors.append(
                f"Missing required features for tier {tier}: {', '.join(missing_required)}"
            )

        # 5. Check data types and ranges for present columns
        for col in df.columns:
            spec = self.registry.get_feature_spec(col)

            if spec is None:
                # Column not in registry
                warnings.append(
                    f"Column '{col}' not in registry. "
                    "This may be fine for custom features."
                )
                continue

            # Check data type
            actual_dtype = str(df[col].dtype)
            expected_dtype = spec.dtype

            # Allow some flexibility (e.g., int64 vs int8, float64 vs float32)
            if not self._dtypes_compatible(actual_dtype, expected_dtype):
                warnings.append(
                    f"Column '{col}': expected dtype '{expected_dtype}', "
                    f"got '{actual_dtype}'"
                )

            # Check value ranges (skip if all null)
            if not df[col].isna().all():
                if spec.range_min is not None:
                    min_val = df[col].min()
                    if pd.notna(min_val) and min_val < spec.range_min:
                        errors.append(
                            f"Column '{col}': min value {min_val:.4f} below "
                            f"allowed minimum {spec.range_min}"
                        )

                if spec.range_max is not None:
                    max_val = df[col].max()
                    if pd.notna(max_val) and max_val > spec.range_max:
                        errors.append(
                            f"Column '{col}': max value {max_val:.4f} above "
                            f"allowed maximum {spec.range_max}"
                        )

            # Check nulls in required columns
            if spec.required and df[col].isna().any():
                n_nulls = df[col].isna().sum()
                pct = n_nulls / len(df) * 100
                errors.append(
                    f"Column '{col}' (required): {n_nulls} nulls ({pct:.1f}%)"
                )

        # 6. Check OHLC consistency
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            ohlc_issues = self._check_ohlc_consistency(df)
            if ohlc_issues:
                errors.extend(ohlc_issues)

        # 7. Check regime probabilities sum to ~1.0 (if present)
        regime_prob_cols = [
            col for col in df.columns
            if col.startswith('regime_prob_')
        ]
        if len(regime_prob_cols) >= 2:  # At least 2 regime probs
            prob_sums = df[regime_prob_cols].sum(axis=1)
            if not prob_sums.isna().all():
                # Check if sums are close to 1.0 (within 0.01)
                valid_sums = prob_sums.between(0.99, 1.01)
                if not valid_sums.all():
                    n_invalid = (~valid_sums).sum()
                    warnings.append(
                        f"Regime probabilities don't sum to ~1.0 for {n_invalid} rows"
                    )

        # Determine pass/fail
        passed = len(errors) == 0
        if strict and len(warnings) > 0:
            passed = False

        return ValidationResult(passed, errors, warnings)

    def _dtypes_compatible(self, actual: str, expected: str) -> bool:
        """Check if actual dtype is compatible with expected."""
        # Exact match
        if actual == expected:
            return True

        # Float compatibility
        if expected in ['float64', 'float32', 'float16']:
            if actual in ['float64', 'float32', 'float16']:
                return True

        # Int compatibility
        if expected in ['int64', 'int32', 'int16', 'int8']:
            if actual in ['int64', 'int32', 'int16', 'int8']:
                return True

        # Bool compatibility
        if expected == 'bool' and actual in ['bool', 'boolean']:
            return True

        # Category compatibility
        if expected == 'category' and actual in ['category', 'object']:
            return True

        return False

    def _check_ohlc_consistency(self, df: pd.DataFrame) -> List[str]:
        """Check OHLC price consistency rules."""
        errors = []

        # high >= low
        violations = df['high'] < df['low']
        if violations.any():
            n = violations.sum()
            errors.append(f"OHLC violation: high < low in {n} rows")

        # high >= close
        violations = df['high'] < df['close']
        if violations.any():
            n = violations.sum()
            errors.append(f"OHLC violation: high < close in {n} rows")

        # high >= open
        violations = df['high'] < df['open']
        if violations.any():
            n = violations.sum()
            errors.append(f"OHLC violation: high < open in {n} rows")

        # low <= close
        violations = df['low'] > df['close']
        if violations.any():
            n = violations.sum()
            errors.append(f"OHLC violation: low > close in {n} rows")

        # low <= open
        violations = df['low'] > df['open']
        if violations.any():
            n = violations.sum()
            errors.append(f"OHLC violation: low > open in {n} rows")

        return errors

    def normalize_and_validate(
        self,
        df: pd.DataFrame,
        tier: int,
        strict: bool = True
    ) -> Tuple[pd.DataFrame, ValidationResult]:
        """
        Normalize column names via registry, then validate.

        Args:
            df: Input dataframe
            tier: Expected tier level
            strict: If True, treat warnings as errors

        Returns:
            Tuple of (normalized_df, validation_result)
        """
        # Normalize column names
        df_normalized = df.copy()
        rename_map = {}

        for col in df.columns:
            canonical = self.registry.normalize_column_name(col)
            if canonical != col:
                rename_map[col] = canonical

        if rename_map:
            df_normalized = df_normalized.rename(columns=rename_map)

        # Validate
        result = self.validate(df_normalized, tier, strict)

        return df_normalized, result

    def compute_parameter_bounds(
        self,
        df: pd.DataFrame,
        features: List[str],
        quantiles: Tuple[float, float] = (0.05, 0.95)
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute data-derived parameter bounds from feature quantiles.

        This prevents optimization ranges that never trigger in actual data.

        Args:
            df: Feature store dataframe
            features: List of feature names to compute bounds for
            quantiles: (low, high) quantile values (default 5th-95th percentile)

        Returns:
            Dict mapping feature name to (min, max) bounds

        Example:
            >>> bounds = validator.compute_parameter_bounds(
            ...     df, ['tf4h_fusion_score', 'adx_14']
            ... )
            >>> # Use in Optuna: suggest_float(..., bounds['tf4h_fusion_score'][0], bounds['tf4h_fusion_score'][1])
        """
        bounds = {}

        for feature in features:
            if feature not in df.columns:
                continue

            series = df[feature].dropna()
            if len(series) == 0:
                continue

            # Skip boolean columns
            if series.dtype == bool:
                continue

            q_low, q_high = quantiles
            bounds[feature] = (
                float(series.quantile(q_low)),
                float(series.quantile(q_high))
            )

        return bounds


def validate_feature_store(
    path: str,
    tier: int = 3,
    strict: bool = True,
    normalize: bool = True
) -> ValidationResult:
    """
    Convenience function: Load and validate a feature store file.

    Args:
        path: Path to parquet file
        tier: Expected tier level
        strict: Treat warnings as errors
        normalize: Normalize column names before validation

    Returns:
        ValidationResult
    """
    df = pd.read_parquet(path)

    validator = FeatureValidator()

    if normalize:
        df, result = validator.normalize_and_validate(df, tier, strict)
    else:
        result = validator.validate(df, tier, strict)

    return result
