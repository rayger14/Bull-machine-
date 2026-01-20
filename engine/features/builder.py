#!/usr/bin/env python3
"""
FeatureStoreBuilder - Single façade for all feature store operations.

This is NOT "another builder" - it's the single unifying interface that:
- Orchestrates existing stage functions (MTF builder, regime detector)
- Enforces schema validation and contracts
- Uses 1H base with computed MTF columns (tf4h_*, tf1d_*)
- Fails fast with human reports before backtest/Optuna

"A FeatureStoreBuilder isn't 'another builder'; it's the single façade that
orchestrates all stages behind one clean API and enforces the schema/contract."
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import json

from engine.features.registry import get_registry
from engine.features.validate import FeatureValidator, ValidationResult


@dataclass
class BuildSpec:
    """Specification for a feature store build."""
    asset: str
    start: str  # YYYY-MM-DD
    end: str    # YYYY-MM-DD
    tiers: List[int]  # [1, 2, 3]
    resolution: str = "1H"
    output_dir: Optional[str] = None
    validate: bool = True
    normalize: bool = True


class FeatureStoreBuilder:
    """
    Single entry point for feature store operations.

    Responsibilities:
    1. Orchestrate existing builders (don't rewrite them)
    2. Normalize column names via registry
    3. Validate contracts before saving
    4. Compute data-derived parameter bounds
    5. Generate human-readable reports

    Usage:
        builder = FeatureStoreBuilder()
        spec = BuildSpec(asset="BTC", start="2022-01-01", end="2024-12-31", tiers=[1,2,3])
        df, report = builder.build(spec)
    """

    def __init__(self):
        self.registry = get_registry()
        self.validator = FeatureValidator()

    def build(self, spec: BuildSpec) -> tuple[pd.DataFrame, dict]:
        """
        Build feature store according to specification.

        This orchestrates calls to existing stage functions and validates the result.

        Args:
            spec: BuildSpec defining what to build

        Returns:
            Tuple of (dataframe, build_report)

        Raises:
            ValueError: If validation fails
        """
        print("=" * 70)
        print(f"FEATURE STORE BUILD - {spec.asset}")
        print("=" * 70)
        print(f"Period: {spec.start} → {spec.end}")
        print(f"Tiers: {spec.tiers}")
        print(f"Resolution: {spec.resolution}")
        print()

        build_report = {
            "spec": spec.__dict__,
            "start_time": datetime.now().isoformat(),
            "stages": {}
        }

        df = None

        # Stage 1: Raw OHLCV + Technical Indicators
        if 1 in spec.tiers:
            print("▶ Stage 1: Loading raw OHLCV + technical indicators...")
            df = self._build_tier1(spec)
            build_report["stages"]["tier1"] = {
                "rows": len(df),
                "columns": len(df.columns),
                "status": "completed"
            }
            print(f"  ✓ Tier 1: {len(df)} rows, {len(df.columns)} columns")

        # Stage 2: Multi-Timeframe Features
        if 2 in spec.tiers:
            if df is None:
                raise ValueError("Tier 2 requires Tier 1 data")

            print("\n▶ Stage 2: Computing multi-timeframe features...")
            df = self._build_tier2(df, spec)
            build_report["stages"]["tier2"] = {
                "rows": len(df),
                "columns": len(df.columns),
                "status": "completed"
            }
            print(f"  ✓ Tier 2: {len(df)} rows, {len(df.columns)} columns")

        # Stage 3: Regime Labels + Macro
        if 3 in spec.tiers:
            if df is None:
                raise ValueError("Tier 3 requires Tier 2 data")

            print("\n▶ Stage 3: Adding regime labels + macro context...")
            df = self._build_tier3(df, spec)
            build_report["stages"]["tier3"] = {
                "rows": len(df),
                "columns": len(df.columns),
                "status": "completed"
            }
            print(f"  ✓ Tier 3: {len(df)} rows, {len(df.columns)} columns")

        # Normalize column names
        if spec.normalize:
            print("\n▶ Normalizing column names via registry...")
            df = self._normalize_columns(df)
            print(f"  ✓ Normalized to canonical names")

        # Validate contracts
        if spec.validate:
            print("\n▶ Validating contracts...")
            max_tier = max(spec.tiers)
            validation = self.validator.validate(df, max_tier, strict=False)

            print()
            print(validation)

            if not validation.passed:
                build_report["validation"] = {
                    "passed": False,
                    "errors": validation.errors,
                    "warnings": validation.warnings
                }
                raise ValueError(
                    "Feature store validation failed. "
                    "Fix errors before using in backtest/Optuna."
                )

            build_report["validation"] = {
                "passed": True,
                "errors": [],
                "warnings": validation.warnings
            }

        # Compute parameter bounds
        print("\n▶ Computing data-derived parameter bounds...")
        bounds = self._compute_bounds(df, spec)
        build_report["parameter_bounds"] = bounds
        print(f"  ✓ Computed bounds for {len(bounds)} features")
        for feature, (min_val, max_val) in sorted(bounds.items())[:5]:
            print(f"    - {feature}: [{min_val:.4f}, {max_val:.4f}]")
        if len(bounds) > 5:
            print(f"    ... and {len(bounds) - 5} more")

        # Save if output_dir specified
        if spec.output_dir:
            output_path = self._get_output_path(spec)
            print(f"\n▶ Saving to: {output_path}")
            self._save(df, output_path, build_report)
            print(f"  ✓ Saved {output_path.stat().st_size / 1024 / 1024:.1f} MB")

        build_report["end_time"] = datetime.now().isoformat()
        build_report["final_shape"] = {"rows": len(df), "columns": len(df.columns)}

        print("\n" + "=" * 70)
        print("✅ BUILD COMPLETE")
        print("=" * 70)

        return df, build_report

    def _build_tier1(self, spec: BuildSpec) -> pd.DataFrame:
        """
        Build Tier 1: Raw OHLCV + technical indicators.

        This wraps the existing TradingView loader and adds basic technical indicators.
        """
        from engine.io.tradingview_loader import load_tv

        # Load raw OHLCV
        df = load_tv(spec.asset, spec.start, spec.end, resolution=spec.resolution)

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            df.index = pd.to_datetime(df.index)

        # Add basic technical indicators if not present
        df = self._add_technical_indicators(df)

        return df

    def _build_tier2(self, df: pd.DataFrame, spec: BuildSpec) -> pd.DataFrame:
        """
        Build Tier 2: Multi-timeframe features.

        This wraps the existing build_mtf_feature_store.py logic.
        For MVP, we'll check if MTF file exists and load it; otherwise call the builder.
        """
        # Check if pre-built MTF file exists
        mtf_path = Path(f"data/features_mtf/{spec.asset}_1H_{spec.start}_to_{spec.end}.parquet")

        if mtf_path.exists():
            print(f"  → Loading existing MTF file: {mtf_path}")
            df_mtf = pd.read_parquet(mtf_path)

            # Ensure DatetimeIndex
            if not isinstance(df_mtf.index, pd.DatetimeIndex):
                if 'timestamp' in df_mtf.columns:
                    df_mtf = df_mtf.set_index('timestamp')
                df_mtf.index = pd.to_datetime(df_mtf.index)

            return df_mtf

        # Otherwise, call the existing MTF builder
        print(f"  → Building MTF features from scratch...")
        print(f"  ⚠️  This may take 10-20 minutes. Consider running:")
        print(f"      python3 bin/build_mtf_feature_store.py --asset {spec.asset} --start {spec.start} --end {spec.end}")

        # For now, we'll use a simplified version
        # In production, this would call the full build_mtf_feature_store.py logic
        df_mtf = self._build_mtf_simple(df, spec)

        return df_mtf

    def _build_tier3(self, df: pd.DataFrame, spec: BuildSpec) -> pd.DataFrame:
        """
        Build Tier 3: Regime labels + macro context.

        This wraps the RegimeDetector to add regime classification.
        """
        from engine.regime_detector import RegimeDetector

        regime_detector = RegimeDetector()

        # Add regime labels and confidence
        df = regime_detector.classify_batch(df)

        # TODO: Add macro context (DXY, VIX, yields) if available
        # For now, regime labels are sufficient

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators if not present."""
        # Check if indicators already present
        has_indicators = all(
            col in df.columns
            for col in ['atr_20', 'adx_14', 'rsi_14']
        )

        if has_indicators:
            return df

        # Add ATR
        if 'atr_20' not in df.columns:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr_20'] = true_range.rolling(20).mean()

        # Add RSI
        if 'rsi_14' not in df.columns:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))

        # Add ADX (simplified)
        if 'adx_14' not in df.columns:
            # Placeholder - full ADX calculation is complex
            df['adx_14'] = 25.0  # Neutral value

        return df

    def _build_mtf_simple(self, df: pd.DataFrame, spec: BuildSpec) -> pd.DataFrame:
        """
        Simplified MTF builder for MVP.

        In production, this would call the full build_mtf_feature_store.py pipeline.
        For now, we'll add placeholder MTF features.
        """
        # Add 4H trend features
        df['tf4h_fusion_score'] = 0.0
        df['tf4h_trend_strength'] = 0.5
        df['tf4h_bos_bullish'] = False
        df['tf4h_bos_bearish'] = False

        # Add 1D trend features
        df['tf1d_trend_direction'] = 0

        # Add 1H structure features
        df['tf1h_bos_bullish'] = False
        df['tf1h_bos_bearish'] = False
        df['tf1h_fvg_bull'] = False
        df['tf1h_fvg_bear'] = False

        # Add composite scores
        df['liquidity_score'] = 0.5
        df['smc_score'] = 0.5
        df['wyckoff_score'] = 0.5
        df['momentum_score'] = 0.5

        print("  ⚠️  Using simplified MTF features. For production, use full MTF builder.")

        return df

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to canonical form via registry."""
        rename_map = {}

        for col in df.columns:
            canonical = self.registry.normalize_column_name(col)
            if canonical != col:
                rename_map[col] = canonical

        if rename_map:
            df = df.rename(columns=rename_map)
            print(f"  → Renamed {len(rename_map)} columns to canonical names")

        return df

    def _compute_bounds(self, df: pd.DataFrame, spec: BuildSpec) -> dict:
        """Compute data-derived parameter bounds."""
        # Features commonly used in optimization
        optimization_features = [
            'tf4h_fusion_score',
            'adx_14',
            'rsi_14',
            'atr_20',
            'liquidity_score',
            'smc_score',
        ]

        # Filter to features present in dataframe
        features = [f for f in optimization_features if f in df.columns]

        return self.validator.compute_parameter_bounds(df, features)

    def _get_output_path(self, spec: BuildSpec) -> Path:
        """Generate output file path."""
        output_dir = Path(spec.output_dir) if spec.output_dir else Path("data/feature_store")
        output_dir = output_dir / spec.asset.lower() / "full"
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{spec.asset.lower()}_1h_full_v1.0_{spec.start}_{spec.end}.parquet"
        return output_dir / filename

    def _save(self, df: pd.DataFrame, path: Path, report: dict):
        """Save dataframe and metadata."""
        # Save parquet
        df.to_parquet(path, compression='zstd')

        # Save metadata
        metadata_path = path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"  ✓ Metadata: {metadata_path}")

    def load(
        self,
        asset: str,
        start: str,
        end: str,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Load feature store and optionally validate.

        Args:
            asset: Asset symbol (BTC, ETH, etc.)
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            validate: Run contract validation

        Returns:
            Validated dataframe

        Raises:
            ValueError: If validation fails
        """
        # Try to find feature store file
        search_paths = [
            Path(f"data/feature_store/{asset.lower()}/full/{asset.lower()}_1h_full_v1.0_{start}_{end}.parquet"),
            Path(f"data/cached/{asset.lower()}_features_{start}_{end}_cached.parquet"),
            Path(f"data/features_mtf/{asset}_1H_{start}_to_{end}.parquet"),
        ]

        for path in search_paths:
            if path.exists():
                print(f"📂 Loading: {path}")
                df = pd.read_parquet(path)

                # Ensure DatetimeIndex
                if not isinstance(df.index, pd.DatetimeIndex):
                    if 'timestamp' in df.columns:
                        df = df.set_index('timestamp')
                    df.index = pd.to_datetime(df.index)

                if validate:
                    # Infer tier from columns
                    tier = 3 if 'regime_label' in df.columns else 2 if 'tf4h_fusion_score' in df.columns else 1

                    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
                    print(f"✓ Inferred tier: {tier}")

                    # Validate
                    validation = self.validator.validate(df, tier, strict=False)
                    if not validation.passed:
                        print(validation)
                        raise ValueError("Validation failed. See errors above.")

                    print("✓ Validation passed")

                return df

        raise FileNotFoundError(
            f"Feature store not found for {asset} {start}→{end}. "
            f"Build it first with: python3 bin/feature_store.py --asset {asset} --start {start} --end {end}"
        )


def quick_load(
    asset: str,
    start: str,
    end: str,
    validate: bool = False
) -> pd.DataFrame:
    """
    Convenience function for quick loading.

    Args:
        asset: Asset symbol (BTC, ETH, etc.)
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        validate: Run validation (default False for speed)

    Returns:
        DataFrame

    Example:
        >>> df = quick_load('BTC', '2022-01-01', '2024-12-31')
    """
    builder = FeatureStoreBuilder()
    return builder.load(asset, start, end, validate)
