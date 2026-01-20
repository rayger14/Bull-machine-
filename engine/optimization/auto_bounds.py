#!/usr/bin/env python3
"""
Auto-Bounds - Automatic parameter bound computation for Optuna optimizers.

This module prevents the critical issue where optimization ranges don't match
actual data ranges, causing zero variance and wasted compute time.

Usage:
    from engine.optimization.auto_bounds import compute_parameter_bounds

    # In your Optuna objective function:
    bounds = compute_parameter_bounds(df, {
        'quality_threshold': 'tf4h_fusion_score',
        'adx_threshold': 'adx_14',
    })

    # Use bounds in trial.suggest_float:
    trial.suggest_float('quality_threshold', *bounds['quality_threshold'])
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path


def compute_parameter_bounds(
    df: pd.DataFrame,
    param_to_feature_map: Dict[str, str],
    quantiles: Tuple[float, float] = (0.05, 0.95),
    expand_factor: float = 0.1,
    min_range: float = 0.01
) -> Dict[str, Tuple[float, float]]:
    """
    Compute data-derived parameter bounds from feature quantiles.

    This prevents optimization ranges that never occur in actual data,
    which causes zero variance and identical trial scores.

    Args:
        df: DataFrame with features
        param_to_feature_map: Mapping of parameter names to feature columns
            Example: {'quality_threshold': 'tf4h_fusion_score'}
        quantiles: (low, high) percentiles to use (default: 5th-95th)
        expand_factor: Expand bounds by this fraction to allow exploration (default: 10%)
        min_range: Minimum range width to prevent too-narrow bounds

    Returns:
        Dict mapping parameter names to (min, max) bounds

    Example:
        >>> bounds = compute_parameter_bounds(df, {
        ...     'quality_threshold': 'tf4h_fusion_score',
        ...     'adx_threshold': 'adx_14'
        ... })
        >>> bounds
        {'quality_threshold': (0.00, 0.22), 'adx_threshold': (14.0, 67.0)}

        >>> # Use in Optuna:
        >>> trial.suggest_float('quality_threshold', *bounds['quality_threshold'])
    """
    bounds = {}

    for param_name, feature_name in param_to_feature_map.items():
        if feature_name not in df.columns:
            print(f"⚠️  Feature '{feature_name}' not found in dataframe, skipping '{param_name}'")
            continue

        series = df[feature_name].dropna()

        if len(series) == 0:
            print(f"⚠️  Feature '{feature_name}' has no valid values, skipping '{param_name}'")
            continue

        # Skip boolean columns
        if series.dtype == bool:
            print(f"⚠️  Feature '{feature_name}' is boolean, skipping '{param_name}'")
            continue

        # Compute quantile bounds
        q_low, q_high = quantiles
        low = float(series.quantile(q_low))
        high = float(series.quantile(q_high))

        # Expand bounds slightly to allow exploration
        range_width = high - low
        if range_width < min_range:
            # Range too narrow - expand to min_range
            midpoint = (low + high) / 2
            low = midpoint - min_range / 2
            high = midpoint + min_range / 2
        else:
            # Expand by factor
            expansion = range_width * expand_factor
            low -= expansion
            high += expansion

        # Ensure non-negative if original data was non-negative
        if series.min() >= 0:
            low = max(0.0, low)

        bounds[param_name] = (low, high)

    return bounds


def print_bounds_report(
    bounds: Dict[str, Tuple[float, float]],
    param_to_feature_map: Dict[str, str],
    title: str = "Data-Derived Parameter Bounds"
):
    """
    Print human-readable bounds report.

    Args:
        bounds: Computed bounds from compute_parameter_bounds()
        param_to_feature_map: Original parameter-to-feature mapping
        title: Report title
    """
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print()
    print("These bounds prevent optimization ranges that never occur in data.")
    print("Use these in trial.suggest_float() calls:")
    print()

    for param_name, (low, high) in sorted(bounds.items()):
        feature_name = param_to_feature_map.get(param_name, "unknown")
        print(f"  {param_name:30s} [{low:8.4f}, {high:8.4f}]")
        print(f"    → From feature: {feature_name}")
        print(f"    → Optuna call:  trial.suggest_float('{param_name}', {low:.4f}, {high:.4f})")
        print()

    print("=" * 70)
    print()


def compute_bounds_from_file(
    file_path: str,
    param_to_feature_map: Dict[str, str],
    quantiles: Tuple[float, float] = (0.05, 0.95),
    print_report: bool = True
) -> Dict[str, Tuple[float, float]]:
    """
    Convenience function: Load parquet file and compute bounds.

    Args:
        file_path: Path to parquet file
        param_to_feature_map: Parameter to feature mapping
        quantiles: Percentiles to use
        print_report: Print human-readable report

    Returns:
        Computed bounds

    Example:
        >>> bounds = compute_bounds_from_file(
        ...     'data/cached/btc_features_2022-01-01_2024-12-31_cached.parquet',
        ...     {'quality_threshold': 'tf4h_fusion_score'}
        ... )
    """
    print(f"📂 Loading: {file_path}")
    df = pd.read_parquet(file_path)
    print(f"  ✓ Loaded {len(df)} rows, {len(df.columns)} columns")

    bounds = compute_parameter_bounds(df, param_to_feature_map, quantiles)

    if print_report:
        print_bounds_report(bounds, param_to_feature_map)

    return bounds


def suggest_with_bounds(
    trial,
    param_name: str,
    bounds: Dict[str, Tuple[float, float]],
    default_range: Tuple[float, float],
    step: Optional[float] = None,
    **kwargs
) -> float:
    """
    Suggest parameter value with automatic bounds fallback.

    If bounds exist for this parameter, use them. Otherwise, use default_range.

    Args:
        trial: Optuna trial object
        param_name: Parameter name
        bounds: Computed bounds dict (can be empty)
        default_range: Fallback (min, max) if not in bounds
        step: Step size (optional)
        **kwargs: Additional args for trial.suggest_float

    Returns:
        Suggested value

    Example:
        >>> # Compute bounds once before optimization
        >>> bounds = compute_bounds_from_file(cache_path, param_map)
        >>>
        >>> # In objective function:
        >>> def objective(trial):
        ...     quality = suggest_with_bounds(
        ...         trial, 'quality_threshold', bounds,
        ...         default_range=(0.0, 1.0), step=0.05
        ...     )
    """
    if param_name in bounds:
        low, high = bounds[param_name]
    else:
        low, high = default_range

    return trial.suggest_float(param_name, low, high, step=step, **kwargs)


# Predefined parameter mappings for common archetypes
TRAP_PARAM_MAP = {
    'quality_threshold': 'tf4h_fusion_score',
    'fusion_threshold': 'tf4h_fusion_score',
    'adx_threshold': 'adx_14',
    'rsi_threshold': 'rsi_14',
    'atr_threshold': 'atr_20',
}

OB_RETEST_PARAM_MAP = {
    'ob_proximity_threshold': 'tf1h_ob_bull_top',  # Distance to OB
    'adx_threshold': 'adx_14',
    'volume_threshold': 'volume_zscore',
}

EXHAUSTION_PARAM_MAP = {
    'rsi_extreme': 'rsi_14',
    'atr_threshold': 'atr_20',
    'volume_threshold': 'volume_zscore',
}
