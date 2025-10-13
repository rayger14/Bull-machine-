#!/usr/bin/env python3
"""
ML Model Training CLI for Bull Machine v1.8.6
=============================================

Train LightGBM/XGBoost models on optimization results to predict config performance.

Usage:
    python bin/research/train_ml.py --dataset data/ml/optimization_results.parquet --target sharpe --output models/sharpe_model

Example:
    # Train Sharpe predictor
    python bin/research/train_ml.py --target sharpe

    # Train Profit Factor predictor with XGBoost
    python bin/research/train_ml.py --target pf --model xgboost

    # Train with custom filtering
    python bin/research/train_ml.py --target sharpe --min-trades 100 --max-dd 0.15
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engine.ml.dataset import OptimizationDataset
from engine.ml.models import ConfigSuggestionModel
from engine.ml.featurize import normalize_features


def main():
    parser = argparse.ArgumentParser(description="Train ML model for config suggestion")

    parser.add_argument('--dataset', type=str, default='data/ml/optimization_results.parquet',
                        help='Path to optimization results dataset')
    parser.add_argument('--target', type=str, default='sharpe', choices=['sharpe', 'pf', 'total_return_pct'],
                        help='Target metric to predict')
    parser.add_argument('--model', type=str, default='lightgbm', choices=['lightgbm', 'xgboost', 'linear'],
                        help='Model type')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for trained model (default: models/{target}_model)')

    # Filtering options
    parser.add_argument('--asset', type=str, default=None,
                        help='Filter by asset (e.g., BTC)')
    parser.add_argument('--min-trades', type=int, default=50,
                        help='Minimum number of trades (default: 50)')
    parser.add_argument('--max-dd', type=float, default=0.20,
                        help='Maximum drawdown threshold (default: 0.20 = 20%%)')
    parser.add_argument('--min-sharpe', type=float, default=None,
                        help='Minimum Sharpe ratio (optional)')

    # Training options
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size (default: 0.2 = 20%%)')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize features (standardize)')

    args = parser.parse_args()

    print("=" * 80)
    print("🤖 Bull Machine v1.8.6 - ML Model Training")
    print("=" * 80)
    print()

    # Load dataset
    print("📂 Loading dataset...")
    dataset = OptimizationDataset(args.dataset)

    if len(dataset.df) == 0:
        print("❌ Dataset is empty! Run optimization first to collect training data.")
        print("   Example: python bin/optimize_v19.py --mode grid --asset BTC --years 3")
        return 1

    # Print summary stats
    print("\n📊 Dataset Summary:")
    stats = dataset.summary_stats()
    print(f"   Total rows: {stats['total_rows']}")
    print(f"   Unique assets: {stats['unique_assets']}")
    if stats['sharpe_stats']['mean'] is not None:
        print(f"   Sharpe: {stats['sharpe_stats']['mean']:.3f} ± {stats['sharpe_stats']['std']:.3f} (range: {stats['sharpe_stats']['min']:.3f} to {stats['sharpe_stats']['max']:.3f})")
    if stats['pf_stats']['mean'] is not None:
        print(f"   PF: {stats['pf_stats']['mean']:.3f} ± {stats['pf_stats']['std']:.3f} (range: {stats['pf_stats']['min']:.3f} to {stats['pf_stats']['max']:.3f})")
    print()

    # Filter dataset
    print("🔍 Filtering dataset...")
    filtered_df = dataset.filter(
        asset=args.asset,
        min_trades=args.min_trades,
        max_dd_threshold=args.max_dd,
        min_sharpe=args.min_sharpe
    )
    print()

    if len(filtered_df) < 20:
        print(f"❌ Not enough data after filtering ({len(filtered_df)} rows). Need at least 20 samples.")
        print("   Try relaxing filter criteria (--min-trades, --max-dd, etc.)")
        return 1

    # Split features and target
    print("📈 Preparing features and target...")
    X, y, feature_names = dataset.get_feature_target_split(filtered_df, target_col=args.target)
    print()

    # Train/test split
    split_idx = int(len(X) * (1 - args.test_size))
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    print(f"🔄 Train/test split: {len(X_train)} / {len(X_test)}")
    print()

    # Normalize features if requested
    normalization_params = None
    if args.normalize:
        print("📊 Normalizing features (standardization)...")
        X_train_normalized, normalization_params = normalize_features(X_train.values, method='standardize')
        X_test_normalized = (X_test.values - normalization_params['mean']) / normalization_params['std']

        X_train = X_train_normalized
        X_test = X_test_normalized
        print()

    # Create and train model
    print(f"🤖 Training {args.model} model for target: {args.target}...")
    print()

    model = ConfigSuggestionModel(model_type=args.model, target=args.target)
    model.normalization_params = normalization_params

    metrics = model.train(
        X_train, y_train,
        X_test, y_test,
        feature_names=feature_names
    )

    # Save model
    if args.output is None:
        output_path = f"models/{args.target}_model"
    else:
        output_path = args.output

    model.save(output_path)
    print()

    # Summary
    print("=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"Model saved to: {output_path}")
    print(f"Target: {args.target}")
    print(f"Training R²: {metrics['train_r2']:.4f}")
    if 'val_r2' in metrics:
        print(f"Test R²: {metrics['val_r2']:.4f}")
    print()
    print("Next steps:")
    print(f"  1. Use model for config suggestion:")
    print(f"     python bin/research/suggest_config.py --model {output_path} --asset BTC")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
