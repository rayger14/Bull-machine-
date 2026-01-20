#!/usr/bin/env python3
"""
Prepare ML dataset for engine weight optimization.

This script:
1. Loads historical trade data from validation results
2. Extracts and normalizes domain scores (structure, liquidity, momentum, wyckoff, macro)
3. Creates clean ML dataset with domain scores as features
4. Splits data by year/regime for proper train/validate/test sets
5. Generates summary statistics

Output structure:
- results/ml_dataset/train.csv (2022 bear market)
- results/ml_dataset/validate.csv (2023 transition)
- results/ml_dataset/test.csv (2024 bull market)
- results/ml_dataset/dataset_summary.txt
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class DomainScoreExtractor:
    """Extract and normalize domain scores from feature columns."""

    # Define features for each domain
    DOMAIN_FEATURES = {
        'structure_score': [
            'nested_structure_quality',
            'boms_strength',
            'fvg_quality',
            'poc_distance'
        ],
        'liquidity_score': [
            'entry_liquidity_score',
            'liquidity_sweep_strength',
            'lvn_trap_risk'
        ],
        'momentum_score': [
            'rsi_14',
            'macd_histogram',
            'adx_14',
            'volume_zscore'
        ],
        'wyckoff_score': [
            'wyckoff_phase_score'
        ],
        'macro_score': [
            'vix_z_score',
            'btc_volatility_percentile',
            'atr_percentile'
        ]
    }

    def __init__(self, df: pd.DataFrame):
        """Initialize with trade dataframe."""
        self.df = df.copy()
        self.missing_features = {}
        self._identify_missing_features()

    def _identify_missing_features(self):
        """Identify missing features and log them."""
        available_cols = set(self.df.columns)

        for domain, features in self.DOMAIN_FEATURES.items():
            missing = [f for f in features if f not in available_cols]
            if missing:
                self.missing_features[domain] = missing

    def _normalize_column(self, col: pd.Series) -> pd.Series:
        """Normalize a column to [0, 1] range, handling edge cases."""
        # Remove infinite values
        col = col.replace([np.inf, -np.inf], np.nan)

        # Get min and max
        col_min = col.min()
        col_max = col.max()

        # If all values are the same, return 0.5
        if col_min == col_max or pd.isna(col_min) or pd.isna(col_max):
            return pd.Series(0.5, index=col.index)

        # Normalize to [0, 1]
        normalized = (col - col_min) / (col_max - col_min)

        # Clip to [0, 1] and fill NaN with 0.5 (neutral)
        return normalized.clip(0, 1).fillna(0.5)

    def extract_domain_scores(self) -> pd.DataFrame:
        """Extract domain scores for each trade."""
        result_df = self.df.copy()

        for domain, features in self.DOMAIN_FEATURES.items():
            # Get available features for this domain
            available_features = [f for f in features if f in self.df.columns]

            if not available_features:
                # Domain not available, use neutral score
                result_df[domain] = 0.5
                continue

            # Extract component features and normalize
            normalized_components = []
            for feature in available_features:
                norm_feature = self._normalize_column(self.df[feature])
                normalized_components.append(norm_feature)

            # Average normalized components to get domain score
            domain_score = pd.concat(normalized_components, axis=1).mean(axis=1)
            result_df[domain] = domain_score.clip(0, 1).fillna(0.5)

        # Handle PTI (not yet implemented)
        if 'pti_score' not in result_df.columns:
            result_df['pti_score'] = 0.5  # Neutral placeholder

        return result_df

    def get_feature_summary(self) -> Dict:
        """Get summary of extracted features."""
        return {
            'total_features': len(self.DOMAIN_FEATURES),
            'available_features': {
                domain: [f for f in features if f in self.df.columns]
                for domain, features in self.DOMAIN_FEATURES.items()
            },
            'missing_features': self.missing_features
        }


class ArchetypeMapper:
    """Map archetype binary columns to single archetype name."""

    ARCHETYPE_COLUMNS = [
        'archetype_trap',
        'archetype_retest',
        'archetype_continuation',
        'archetype_failed_continuation',
        'archetype_compression',
        'archetype_exhaustion',
        'archetype_reaccumulation',
        'archetype_trap_within_trend',
        'archetype_wick_trap',
        'archetype_volume_exhaustion',
        'archetype_ratio_coil_break',
        'archetype_false_break_reversal'
    ]

    @staticmethod
    def get_archetype_name(col_name: str) -> str:
        """Convert archetype column name to readable name."""
        return col_name.replace('archetype_', '').replace('_', ' ').title()

    def map_to_archetype(self, row: pd.Series) -> str:
        """Map binary archetype columns to single archetype."""
        available_archetypes = [col for col in self.ARCHETYPE_COLUMNS if col in row.index]

        for archetype_col in available_archetypes:
            if row[archetype_col] == 1:
                return self.get_archetype_name(archetype_col)

        return 'Unknown'


class RegimeClassifier:
    """Classify macro regime from binary regime columns."""

    REGIME_COLUMNS = [
        'macro_regime_risk_on',
        'macro_regime_neutral',
        'macro_regime_risk_off',
        'macro_regime_crisis'
    ]

    @staticmethod
    def classify_regime(row: pd.Series) -> str:
        """Classify regime from binary columns."""
        for regime_col in RegimeClassifier.REGIME_COLUMNS:
            if regime_col in row.index and row[regime_col] == 1:
                return regime_col.replace('macro_regime_', '').title()
        return 'Unknown'

    @staticmethod
    def classify_by_date(date: pd.Timestamp) -> str:
        """Classify regime by year (proxy for macro regime)."""
        year = date.year
        if year == 2022:
            return 'Bear'
        elif year == 2023:
            return 'Transition'
        elif year == 2024:
            return 'Bull'
        else:
            return 'Unknown'


class MLDatasetPreparator:
    """Orchestrate ML dataset preparation."""

    def __init__(self, data_dir: Path):
        """Initialize dataset preparator."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(project_root) / 'results' / 'ml_dataset'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_data = None
        self.validate_data = None
        self.test_data = None
        self.summary = {}

    def load_data_files(self) -> pd.DataFrame:
        """Load all trade CSV files from validation directory."""
        print("Loading trade data files...")

        # Try to load the most recent unified dataset first
        unified_path = self.data_dir / 'validation' / 'unified_full_period_final.csv'
        if unified_path.exists():
            print(f"  Loading unified dataset: {unified_path}")
            df = pd.read_csv(unified_path)
            print(f"    Loaded {len(df)} trades")
            return df

        # Fall back to loading individual year files
        data_files = [
            self.data_dir / 'validation' / 'bear_2022_TRULY_FIXED.csv',
            self.data_dir / 'validation' / 'bull_2024_TRULY_FIXED.csv'
        ]

        dataframes = []
        for file_path in data_files:
            if file_path.exists():
                print(f"  Loading: {file_path.name}")
                df = pd.read_csv(file_path)
                print(f"    Loaded {len(df)} trades")
                dataframes.append(df)

        if not dataframes:
            raise FileNotFoundError(
                f"No trade data files found in {self.data_dir / 'validation'}"
            )

        # Combine all data
        combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
        print(f"  Total trades combined: {len(combined_df)}")

        return combined_df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data."""
        print("\nCleaning data...")
        initial_count = len(df)

        # Convert entry_time to datetime if not already
        if 'entry_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)

        # Remove rows with missing entry_time
        df = df.dropna(subset=['entry_time'])
        print(f"  Rows after removing missing entry_time: {len(df)}")

        # Ensure r_multiple and trade_won are present
        if 'r_multiple' not in df.columns:
            print("  WARNING: r_multiple column not found")
        if 'trade_won' not in df.columns:
            print("  WARNING: trade_won column not found")

        print(f"  Total rows removed: {initial_count - len(df)}")
        print(f"  Final cleaned rows: {len(df)}")

        return df

    def extract_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract domain scores from feature columns."""
        print("\nExtracting domain scores...")

        extractor = DomainScoreExtractor(df)

        # Log feature availability
        summary = extractor.get_feature_summary()
        print(f"  Domains with features: {summary['total_features']}")

        for domain, features in summary['available_features'].items():
            if features:
                print(f"    {domain}: {len(features)} features available")
            else:
                print(f"    {domain}: NO FEATURES (using neutral 0.5)")

        # Extract scores
        df = extractor.extract_domain_scores()

        # Verify domain score columns exist
        domain_columns = [
            'structure_score',
            'liquidity_score',
            'momentum_score',
            'wyckoff_score',
            'macro_score',
            'pti_score'
        ]

        for col in domain_columns:
            if col not in df.columns:
                df[col] = 0.5

        self.summary['feature_extraction'] = summary

        return df

    def extract_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract archetype and regime metadata."""
        print("\nExtracting metadata...")

        # Extract archetype
        mapper = ArchetypeMapper()
        df['archetype'] = df.apply(mapper.map_to_archetype, axis=1)

        # Extract regime (use date-based classification)
        classifier = RegimeClassifier()
        df['macro_regime'] = df['entry_time'].apply(classifier.classify_by_date)

        # Count archetype distribution
        archetype_counts = df['archetype'].value_counts()
        print(f"  Archetypes found: {len(archetype_counts)}")
        for archetype, count in archetype_counts.items():
            print(f"    {archetype}: {count} trades")

        self.summary['archetype_distribution'] = archetype_counts.to_dict()

        # Count regime distribution
        regime_counts = df['macro_regime'].value_counts()
        print(f"  Regimes found: {len(regime_counts)}")
        for regime, count in regime_counts.items():
            print(f"    {regime}: {count} trades")

        self.summary['regime_distribution'] = regime_counts.to_dict()

        return df

    def select_ml_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select final columns for ML dataset."""
        print("\nSelecting ML columns...")

        required_columns = [
            'entry_time',
            'exit_time',
            'r_multiple',
            'trade_won',
            'structure_score',
            'liquidity_score',
            'momentum_score',
            'wyckoff_score',
            'macro_score',
            'pti_score',
            'macro_regime',
            'archetype'
        ]

        # Only select columns that exist
        available_columns = [col for col in required_columns if col in df.columns]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"  WARNING: Missing columns: {missing_columns}")

        ml_df = df[available_columns].copy()

        print(f"  Selected {len(available_columns)} columns for ML dataset")

        return ml_df

    def split_by_regime(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data by year/regime into train/validate/test."""
        print("\nSplitting data by regime...")

        train_df = df[df['macro_regime'] == 'Bear'].copy()
        validate_df = df[df['macro_regime'] == 'Transition'].copy()
        test_df = df[df['macro_regime'] == 'Bull'].copy()

        print(f"  Train (Bear 2022): {len(train_df)} trades ({100*len(train_df)/len(df):.1f}%)")
        print(f"  Validate (Transition 2023): {len(validate_df)} trades ({100*len(validate_df)/len(df):.1f}%)")
        print(f"  Test (Bull 2024): {len(test_df)} trades ({100*len(test_df)/len(df):.1f}%)")

        self.summary['split_counts'] = {
            'train': len(train_df),
            'validate': len(validate_df),
            'test': len(test_df),
            'total': len(df)
        }

        return train_df, validate_df, test_df

    def save_datasets(self, train_df: pd.DataFrame, validate_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save train/validate/test splits to CSV."""
        print("\nSaving datasets...")

        # Save train
        train_path = self.output_dir / 'train.csv'
        train_df.to_csv(train_path, index=False)
        print(f"  Saved train set: {train_path}")

        # Save validate
        validate_path = self.output_dir / 'validate.csv'
        if len(validate_df) > 0:
            validate_df.to_csv(validate_path, index=False)
            print(f"  Saved validate set: {validate_path}")
        else:
            print(f"  Skipped validate set (no data)")

        # Save test
        test_path = self.output_dir / 'test.csv'
        test_df.to_csv(test_path, index=False)
        print(f"  Saved test set: {test_path}")

        self.summary['output_files'] = {
            'train': str(train_path),
            'validate': str(validate_path) if len(validate_df) > 0 else None,
            'test': str(test_path)
        }

    def generate_summary(self):
        """Generate and save summary statistics."""
        print("\nGenerating summary statistics...")

        summary_path = self.output_dir / 'dataset_summary.txt'

        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ML DATASET PREPARATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {datetime.now().isoformat()}\n\n")

            # Split counts
            f.write("DATASET SPLIT\n")
            f.write("-" * 80 + "\n")
            for key, value in self.summary.get('split_counts', {}).items():
                f.write(f"  {key:20s}: {value:6d}\n")
            f.write("\n")

            # Regime distribution
            f.write("REGIME DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            for regime, count in self.summary.get('regime_distribution', {}).items():
                total = self.summary.get('split_counts', {}).get('total', 1)
                pct = 100 * count / total if total > 0 else 0
                f.write(f"  {regime:20s}: {count:6d} ({pct:5.1f}%)\n")
            f.write("\n")

            # Archetype distribution
            f.write("ARCHETYPE DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            for archetype, count in self.summary.get('archetype_distribution', {}).items():
                total = self.summary.get('split_counts', {}).get('total', 1)
                pct = 100 * count / total if total > 0 else 0
                f.write(f"  {archetype:30s}: {count:6d} ({pct:5.1f}%)\n")
            f.write("\n")

            # Feature availability
            f.write("FEATURE EXTRACTION SUMMARY\n")
            f.write("-" * 80 + "\n")
            feature_summary = self.summary.get('feature_extraction', {})
            for domain, features in feature_summary.get('available_features', {}).items():
                f.write(f"  {domain:20s}: {len(features):2d} features\n")
            f.write("\n")

            # Output files
            f.write("OUTPUT FILES\n")
            f.write("-" * 80 + "\n")
            for file_type, file_path in self.summary.get('output_files', {}).items():
                if file_path:
                    f.write(f"  {file_type:20s}: {file_path}\n")
            f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("Domain Scores: structure, liquidity, momentum, wyckoff, macro, pti\n")
            f.write("All domain scores are normalized to [0, 1] range\n")
            f.write("=" * 80 + "\n")

        print(f"  Saved summary: {summary_path}")

    def prepare(self) -> Dict:
        """Run complete dataset preparation pipeline."""
        print("\n" + "=" * 80)
        print("ML DATASET PREPARATION PIPELINE")
        print("=" * 80)

        try:
            # Load data
            df = self.load_data_files()

            # Clean
            df = self.clean_data(df)

            # Extract scores
            df = self.extract_scores(df)

            # Extract metadata
            df = self.extract_metadata(df)

            # Select final columns
            df = self.select_ml_columns(df)

            # Split by regime
            train_df, validate_df, test_df = self.split_by_regime(df)

            # Save
            self.save_datasets(train_df, validate_df, test_df)

            # Generate summary
            self.generate_summary()

            print("\n" + "=" * 80)
            print("ML DATASET PREPARATION COMPLETE")
            print("=" * 80 + "\n")

            return {
                'status': 'success',
                'summary': self.summary,
                'output_dir': str(self.output_dir)
            }

        except Exception as e:
            print(f"\nERROR during dataset preparation: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'failed',
                'error': str(e)
            }


def main():
    """Main entry point."""
    # Get data directory
    data_dir = Path(project_root) / 'results'

    # Run preparation
    preparator = MLDatasetPreparator(data_dir)
    result = preparator.prepare()

    # Return exit code
    if result['status'] == 'success':
        print(f"Dataset prepared in: {result['output_dir']}")
        return 0
    else:
        print(f"Dataset preparation failed: {result.get('error', 'Unknown error')}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
