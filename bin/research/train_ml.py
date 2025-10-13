#!/usr/bin/env python3
"""
ML Model Training CLI for Bull Machine v1.8.6 - WITH WALK-FORWARD CV
====================================================================

Train LightGBM/XGBoost models on optimization results with:
- Walk-forward cross-validation (time-based splits, no leakage)
- Guardrails (refuse to save if PF < 1.0 or MaxDD > 20%)
- Feature importance analysis
- Fold-by-fold validation metrics

Usage:
    python bin/research/train_ml_v2.py --target sharpe --min-trades 50

Example:
    # Train Sharpe predictor with walk-forward CV
    python bin/research/train_ml_v2.py --target sharpe --normalize

    # Train PF predictor with XGBoost
    python bin/research/train_ml_v2.py --target pf --model xgboost --min-trades 100
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engine.ml.dataset import OptimizationDataset
from engine.ml.models import ConfigSuggestionModel
from engine.ml.featurize import normalize_features


def main():
    parser = argparse.ArgumentParser(description="Train ML model with walk-forward CV")

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
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize features (standardize)')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of walk-forward folds (default: 5)')

    args = parser.parse_args()

    print("=" * 80)
    print("🤖 Bull Machine v1.8.6 - ML Model Training (Walk-Forward CV)")
    print("=" * 80)
    print()

    # Load dataset
    print("📂 Loading dataset...")
    dataset = OptimizationDataset(args.dataset)

    if len(dataset.df) == 0:
        print("❌ Dataset is empty! Run optimization first to collect training data.")
        print("   Example: python bin/optimize_v19.py --mode grid --asset BTC")
        return 1

    # Print summary stats
    print("\n📊 Dataset Summary:")
    stats = dataset.summary_stats()
    print(f"   Total rows: {stats['total_rows']}")
    print(f"   Unique assets: {stats['unique_assets']}")
    if stats['sharpe_stats']['mean'] is not None:
        print(f"   Sharpe: {stats['sharpe_stats']['mean']:.3f} ± {stats['sharpe_stats']['std']:.3f}")
    if stats['pf_stats']['mean'] is not None:
        print(f"   PF: {stats['pf_stats']['mean']:.3f} ± {stats['pf_stats']['std']:.3f}")
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

    if len(filtered_df) < 50:
        print(f"❌ Not enough data after filtering ({len(filtered_df)} rows). Need at least 50 samples for CV.")
        print("   Try relaxing filter criteria or collecting more optimization data.")
        return 1

    # Split features and target
    print("📈 Preparing features and target...")
    X, y, feature_names = dataset.get_feature_target_split(filtered_df, target_col=args.target)
    print()

    # ========================================================================
    # WALK-FORWARD CROSS-VALIDATION
    # ========================================================================
    print(f"🔄 Walk-forward cross-validation ({args.n_folds} folds)...")
    print()

    fold_size = len(filtered_df) // args.n_folds
    fold_metrics = []
    fold_pfs = []
    fold_dds = []

    for fold in range(args.n_folds - 1):  # Last fold reserved for final test
        # Train on all data UP TO this fold
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = test_start + fold_size

        train_df = filtered_df.iloc[:train_end]
        test_df = filtered_df.iloc[test_start:test_end]

        if len(train_df) < 20 or len(test_df) < 5:
            print(f"   Fold {fold+1}: Skipped (insufficient data)")
            continue

        # Split features and target
        X_train_fold, y_train_fold, _ = dataset.get_feature_target_split(train_df, target_col=args.target)
        X_test_fold, y_test_fold, _ = dataset.get_feature_target_split(test_df, target_col=args.target)

        # Normalize
        if args.normalize:
            X_train_fold_np, norm_params = normalize_features(X_train_fold.values, method='standardize')
            X_test_fold_np = (X_test_fold.values - norm_params['mean']) / norm_params['std']
            X_train_fold = X_train_fold_np
            X_test_fold = X_test_fold_np

        # Train model
        fold_model = ConfigSuggestionModel(model_type=args.model, target=args.target)

        fold_result = fold_model.train(
            X_train_fold, y_train_fold,
            X_test_fold, y_test_fold,
            feature_names=feature_names
        )

        fold_metrics.append(fold_result)

        # Track PF and MaxDD from TEST SET
        if 'pf' in test_df.columns:
            fold_pfs.extend(test_df['pf'].tolist())
        if 'max_dd' in test_df.columns:
            fold_dds.extend(test_df['max_dd'].abs().tolist())

        val_r2 = fold_result.get('val_r2', 0.0)
        val_rmse = fold_result.get('val_rmse', 0.0)
        print(f"   Fold {fold+1}: Train={len(train_df):4d}, Test={len(test_df):3d} | R²={val_r2:6.4f}, RMSE={val_rmse:.4f}")

    print()

    # ========================================================================
    # GUARDRAILS (Safety Checks)
    # ========================================================================
    print("🛡️  Checking guardrails...")

    # Check 1: Median validation R²
    if fold_metrics:
        val_r2s = [m.get('val_r2', 0.0) for m in fold_metrics if 'val_r2' in m]
        median_r2 = np.median(val_r2s) if val_r2s else 0.0
        print(f"   Median validation R²: {median_r2:.4f}")

        if median_r2 < 0.2:
            print("   ⚠️  WARNING: Low R² (<0.2) - model predictions may be weak")
    else:
        print("   ⚠️  No fold metrics - skipping R² check")
        median_r2 = 0.0

    # Check 2: Profit Factor guard
    passed_pf = True
    if fold_pfs:
        median_pf = np.median(fold_pfs)
        print(f"   Median PF across folds: {median_pf:.3f}")

        if median_pf < 1.0:
            print("   ❌ GUARDRAIL FAILED: Median PF < 1.0")
            print("   Dataset may be broken or configs are all unprofitable")
            print("   Refusing to save model.")
            passed_pf = False
    else:
        print("   ⚠️  No PF data in dataset - skipping PF guard")

    # Check 3: Max Drawdown guard
    passed_dd = True
    if fold_dds:
        max_dd = np.max(fold_dds)
        print(f"   Max drawdown across folds: {max_dd:.1%}")

        if max_dd > 0.20:
            print("   ❌ GUARDRAIL FAILED: Max DD > 20%")
            print("   Configs have excessive risk")
            print("   Refusing to save model.")
            passed_dd = False
    else:
        print("   ⚠️  No MaxDD data in dataset - skipping DD guard")

    if not (passed_pf and passed_dd):
        print("\n❌ GUARDRAILS FAILED - Model NOT saved")
        return 1

    print("   ✅ All guardrails passed")
    print()

    # ========================================================================
    # FINAL MODEL TRAINING (on full dataset)
    # ========================================================================
    print(f"🤖 Training final {args.model} model on full dataset...")
    print()

    # Use 80/20 split for final model
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    # Normalize
    normalization_params = None
    if args.normalize:
        print("📊 Normalizing features...")
        X_train_np, normalization_params = normalize_features(X_train.values, method='standardize')
        X_test_np = (X_test.values - normalization_params['mean']) / normalization_params['std']
        X_train = X_train_np
        X_test = X_test_np

    # Train final model
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
    print(f"Walk-forward median R²: {median_r2:.4f}")
    print(f"Final training R²: {metrics['train_r2']:.4f}")
    if 'val_r2' in metrics:
        print(f"Final test R²: {metrics['val_r2']:.4f}")
    print()
    print("Next steps:")
    print(f"  1. Suggest configs:")
    print(f"     python bin/research/suggest_config.py --model {output_path}")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
