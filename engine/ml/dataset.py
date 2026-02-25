#!/usr/bin/env python3
"""
ML Dataset Management for Bull Machine v1.8.6
=============================================

Manages optimization results dataset for ML training:
- Append new optimization runs to CSV/Parquet
- Load training data with filtering
- Walk-forward train/test splits
- Feature/target extraction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime


class OptimizationDataset:
    """
    Manages optimization results dataset for ML training.
    """

    def __init__(self, dataset_path: str = "data/ml/optimization_results.parquet"):
        """
        Initialize dataset manager.

        Args:
            dataset_path: Path to Parquet file storing optimization results
        """
        self.dataset_path = Path(dataset_path)
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing data if available
        if self.dataset_path.exists():
            self.df = pd.read_parquet(self.dataset_path)
            print(f"📊 Loaded {len(self.df)} existing optimization results")
        else:
            self.df = pd.DataFrame()
            print("📊 Initialized empty optimization dataset")

    def append_results(self, results: List[Dict]) -> None:
        """
        Append new optimization results to dataset.

        Args:
            results: List of dicts from build_training_row()
        """
        if not results:
            print("⚠️  No results to append")
            return

        new_df = pd.DataFrame(results)

        # Add timestamp
        new_df['timestamp'] = datetime.now().isoformat()

        # Append to existing data
        if len(self.df) > 0:
            self.df = pd.concat([self.df, new_df], ignore_index=True)
        else:
            self.df = new_df

        # Save to disk
        self.save()

        print(f"✅ Appended {len(new_df)} results → Total: {len(self.df)}")

    def save(self) -> None:
        """Save dataset to Parquet."""
        self.df.to_parquet(self.dataset_path, index=False)
        print(f"💾 Saved dataset to {self.dataset_path}")

    def filter(
        self,
        asset: Optional[str] = None,
        min_trades: int = 50,
        max_dd_threshold: float = 0.20,
        min_sharpe: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Filter dataset by criteria.

        Args:
            asset: Filter by asset (e.g., 'BTC')
            min_trades: Minimum number of trades
            max_dd_threshold: Maximum drawdown threshold (e.g., 0.15 = 15%)
            min_sharpe: Minimum Sharpe ratio

        Returns:
            Filtered DataFrame
        """
        filtered = self.df.copy()

        if asset:
            filtered = filtered[filtered['asset'] == asset]

        # Filter by trade count
        filtered = filtered[filtered['total_trades'] >= min_trades]

        # Filter by max drawdown
        filtered = filtered[filtered['max_dd'].abs() <= max_dd_threshold]

        # Filter by Sharpe
        if min_sharpe is not None:
            filtered = filtered[filtered['sharpe'] >= min_sharpe]

        print(f"🔍 Filtered: {len(self.df)} → {len(filtered)} rows")
        print(f"   Criteria: asset={asset}, min_trades={min_trades}, max_dd<={max_dd_threshold:.1%}, min_sharpe={min_sharpe}")

        return filtered

    def get_feature_target_split(
        self,
        df: pd.DataFrame,
        target_col: str = 'sharpe',
        feature_prefix_exclude: List[str] = ['pf', 'sharpe', 'max_dd', 'total_trades', 'win_rate', 'total_return_pct', 'avg_r', 'timestamp', 'asset', 'start_date', 'end_date']
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Split dataset into features (X) and target (y).

        Args:
            df: Filtered DataFrame
            target_col: Target column name (e.g., 'sharpe', 'pf')
            feature_prefix_exclude: Columns to exclude from features

        Returns:
            (X, y, feature_names)
        """
        # Identify feature columns (exclude targets and metadata)
        all_cols = set(df.columns)
        exclude_cols = set(feature_prefix_exclude)

        # Also exclude any column starting with target prefix
        for col in all_cols:
            for exc in feature_prefix_exclude:
                if col.startswith(exc):
                    exclude_cols.add(col)

        feature_cols = sorted(all_cols - exclude_cols - {target_col})

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        print(f"📈 Features: {len(feature_cols)} columns")
        print(f"   Target: {target_col}")
        print(f"   Samples: {len(df)}")

        return X, y, feature_cols

    def walk_forward_split(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        test_size: float = 0.2
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create walk-forward train/test splits.

        Args:
            df: DataFrame to split
            n_splits: Number of splits
            test_size: Fraction of data for testing (e.g., 0.2 = 20%)

        Returns:
            List of (train_df, test_df) tuples
        """
        if len(df) < n_splits:
            print(f"⚠️  Not enough data for {n_splits} splits (only {len(df)} rows)")
            return []

        splits = []
        total_size = len(df)
        fold_size = total_size // n_splits

        for i in range(n_splits):
            # Test set: current fold
            test_start = i * fold_size
            test_end = test_start + int(fold_size * test_size)

            # Train set: all data before test set
            if test_start == 0:
                # First fold: use small initial window
                train_start = 0
                train_end = max(1, test_start + 1)
            else:
                train_start = 0
                train_end = test_start

            if train_end <= train_start or test_end <= test_start:
                continue

            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[test_start:test_end]

            if len(train_df) > 0 and len(test_df) > 0:
                splits.append((train_df, test_df))

        print(f"🔄 Walk-forward splits: {len(splits)} folds")
        for i, (train, test) in enumerate(splits):
            print(f"   Fold {i+1}: Train={len(train)}, Test={len(test)}")

        return splits

    def get_best_configs(
        self,
        df: pd.DataFrame,
        target_col: str = 'sharpe',
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get top N configs by target metric.

        Args:
            df: Filtered DataFrame
            target_col: Target column to rank by
            top_n: Number of top configs to return

        Returns:
            DataFrame with top N configs
        """
        sorted_df = df.sort_values(by=target_col, ascending=False)
        top_configs = sorted_df.head(top_n)

        print(f"🏆 Top {top_n} configs by {target_col}:")
        for idx, row in top_configs.iterrows():
            print(f"   {target_col}={row[target_col]:.3f}, PF={row.get('pf', 0):.2f}, Trades={row.get('total_trades', 0):.0f}")

        return top_configs

    def summary_stats(self) -> Dict:
        """
        Get summary statistics of dataset.

        Returns:
            Dict with summary stats
        """
        if len(self.df) == 0:
            return {'total_rows': 0}

        stats = {
            'total_rows': len(self.df),
            'unique_assets': self.df['asset'].nunique() if 'asset' in self.df.columns else 0,
            'date_range': (
                self.df['start_date'].min() if 'start_date' in self.df.columns else None,
                self.df['end_date'].max() if 'end_date' in self.df.columns else None
            ),
            'sharpe_stats': {
                'mean': self.df['sharpe'].mean() if 'sharpe' in self.df.columns else None,
                'std': self.df['sharpe'].std() if 'sharpe' in self.df.columns else None,
                'min': self.df['sharpe'].min() if 'sharpe' in self.df.columns else None,
                'max': self.df['sharpe'].max() if 'sharpe' in self.df.columns else None
            },
            'pf_stats': {
                'mean': self.df['pf'].mean() if 'pf' in self.df.columns else None,
                'std': self.df['pf'].std() if 'pf' in self.df.columns else None,
                'min': self.df['pf'].min() if 'pf' in self.df.columns else None,
                'max': self.df['pf'].max() if 'pf' in self.df.columns else None
            },
            'trades_stats': {
                'mean': self.df['total_trades'].mean() if 'total_trades' in self.df.columns else None,
                'std': self.df['total_trades'].std() if 'total_trades' in self.df.columns else None,
                'min': self.df['total_trades'].min() if 'total_trades' in self.df.columns else None,
                'max': self.df['total_trades'].max() if 'total_trades' in self.df.columns else None
            }
        }

        return stats

    def export_to_csv(self, output_path: str) -> None:
        """
        Export dataset to CSV for inspection.

        Args:
            output_path: Path to CSV file
        """
        self.df.to_csv(output_path, index=False)
        print(f"💾 Exported {len(self.df)} rows to {output_path}")


def load_optimization_results_from_json(json_path: str) -> List[Dict]:
    """
    Load optimization results from JSON file (output from optimize_v19.py).

    Args:
        json_path: Path to optimization_results.json

    Returns:
        List of result dicts
    """
    with open(json_path, 'r') as f:
        results = json.load(f)

    print(f"📂 Loaded {len(results)} results from {json_path}")
    return results


def convert_optimization_results_to_training_rows(
    results: List[Dict],
    macro_snapshot: Dict,
    metadata: Dict
) -> List[Dict]:
    """
    Convert raw optimization results to training rows.

    Args:
        results: List of dicts from optimize_v19.py
        macro_snapshot: Macro snapshot at time of optimization
        metadata: Metadata (asset, date range, etc.)

    Returns:
        List of training rows ready for dataset.append_results()
    """
    from engine.ml.featurize import build_regime_vector, build_training_row

    # Build regime vector once (same for all configs)
    regime_vector = build_regime_vector(macro_snapshot, lookback_window=None)

    training_rows = []
    for result in results:
        config = result.get('config', {})
        metrics = {
            'profit_factor': result.get('profit_factor', 0.0),
            'sharpe': result.get('sharpe', 0.0),
            'max_drawdown': result.get('max_drawdown', 0.0),
            'total_trades': result.get('total_trades', 0),
            'win_rate': result.get('win_rate', 0.0),
            'total_return_pct': result.get('total_return_pct', 0.0),
            'avg_r_multiple': result.get('avg_r_multiple', 0.0)
        }

        row = build_training_row(regime_vector, config, metrics, metadata)
        training_rows.append(row)

    print(f"🔄 Converted {len(results)} results to training rows")
    return training_rows
