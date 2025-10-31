#!/usr/bin/env python3
"""
Config Optimizer Training - Phase 2 Meta-Optimizer

Trains XGBoost Regressor to predict Profit Factor from configuration parameters.
This model learns the config → performance landscape from historical Optuna trials,
enabling intelligent config search without running full backtests.

Usage:
    python3 bin/train/train_config_optimizer.py \
        --data reports/ml/config_training_data.csv \
        --target year_pf \
        --output models/btc_config_optimizer_v1.pkl \
        --test-size 0.2
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def load_and_prepare_data(data_path: str, target_col: str, test_size: float = 0.2):
    """Load trial data and prepare train/test split"""
    print(f"Loading trial data from {data_path}")
    df = pd.read_csv(data_path)

    print(f"  Total trials: {len(df)}")
    print(f"  Target: {target_col}")

    # Define config features (exclude metrics and metadata)
    exclude_cols = [
        'trial', 'q1_score', 'year_pnl', 'year_trades', 'year_pf', 'year_dd', 'year_wr',
        'Q1_pnl', 'Q1_trades', 'source_file', 'source_dir'
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"  Config features: {len(feature_cols)}")

    # Validate target exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    # Remove rows with missing target
    df = df.dropna(subset=[target_col])
    print(f"  Trials after dropping missing target: {len(df)}")

    # Fill missing features with 0 (some configs may not have all params)
    X = df[feature_cols].fillna(0)
    y = df[target_col]

    # Train/test split (stratified by performance bins if possible)
    # Create bins for stratification: low, medium, high PF
    try:
        y_bins = pd.qcut(y, q=3, labels=False, duplicates='drop')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y_bins
        )
    except ValueError:
        # If stratification fails due to too few samples or duplicate values,
        # fall back to simple random split
        print("  WARNING: Stratified split failed, using simple random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    print(f"\nTrain/test split:")
    print(f"  Train: {len(X_train)} trials")
    print(f"  Test: {len(X_test)} trials")
    print(f"  Train target range: {y_train.min():.2f} - {y_train.max():.2f}")
    print(f"  Test target range: {y_test.min():.2f} - {y_test.max():.2f}")

    return X_train, X_test, y_train, y_test, feature_cols


def train_config_optimizer(X_train, y_train, X_test, y_test):
    """Train XGBoost regressor with regularization for low sample count"""
    print(f"\n{'='*60}")
    print("TRAINING CONFIG OPTIMIZER")
    print(f"{'='*60}\n")

    # XGBoost params optimized for small dataset
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 3,  # Shallow trees for small sample size
        'learning_rate': 0.05,  # Low learning rate for stability
        'n_estimators': 100,
        'subsample': 0.7,  # Bagging for regularization
        'colsample_bytree': 0.7,  # Feature sampling
        'min_child_weight': 3,  # Require minimum samples per leaf
        'reg_alpha': 1.0,  # L1 regularization
        'reg_lambda': 2.0,  # L2 regularization
        'random_state': 42,
        'n_jobs': -1
    }

    print("Model hyperparameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    # Train model
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )

    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)

    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)

    print(f"\nTraining Performance:")
    print(f"  MAE: {train_mae:.3f}")
    print(f"  RMSE: {train_rmse:.3f}")
    print(f"  R²: {train_r2:.3f}")

    print(f"\nTest Performance:")
    print(f"  MAE: {test_mae:.3f}")
    print(f"  RMSE: {test_rmse:.3f}")
    print(f"  R²: {test_r2:.3f}")

    # Check for overfitting
    overfit_gap = train_r2 - test_r2
    if overfit_gap > 0.2:
        print(f"\n⚠️  WARNING: Significant overfitting detected (gap={overfit_gap:.3f})")
        print("  Consider: more regularization, fewer features, or more training data")
    elif test_r2 > 0.5:
        print(f"\n✓ Good generalization (R² gap={overfit_gap:.3f})")
    else:
        print(f"\n⚠️  Low test R² ({test_r2:.3f}) - model may have limited predictive power")

    return model, {
        'train_mae': float(train_mae),
        'train_rmse': float(train_rmse),
        'train_r2': float(train_r2),
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'test_r2': float(test_r2),
        'overfit_gap': float(overfit_gap)
    }


def analyze_feature_importance(model, feature_names, output_dir: Path):
    """Generate feature importance plots and SHAP analysis"""
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*60}\n")

    # XGBoost native importance
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print("Top 10 Config Parameters by Importance:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:.4f}")

    # Save full importance table
    importance_path = output_dir / 'feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    print(f"\nFull importance table saved to: {importance_path}")

    # Plot importance
    fig, ax = plt.subplots(figsize=(10, 8))
    importance_df.head(15).plot(
        x='feature', y='importance', kind='barh', ax=ax, legend=False
    )
    ax.set_xlabel('Importance')
    ax.set_title('Top 15 Config Parameters by Importance')
    plt.tight_layout()

    plot_path = output_dir / 'feature_importance.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Feature importance plot saved to: {plot_path}")
    plt.close()

    return importance_df


def generate_shap_analysis(model, X_train, X_test, feature_names, output_dir: Path):
    """Generate SHAP analysis for model interpretability"""
    print(f"\n{'='*60}")
    print("SHAP ANALYSIS")
    print(f"{'='*60}\n")

    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)

        # Compute SHAP values on test set (smaller for speed)
        shap_values = explainer.shap_values(X_test)

        # Summary plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()

        shap_path = output_dir / 'shap_summary.png'
        plt.savefig(shap_path, dpi=150, bbox_inches='tight')
        print(f"SHAP summary plot saved to: {shap_path}")
        plt.close()

        # Feature importance from SHAP
        shap_importance = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': shap_importance
        }).sort_values('shap_importance', ascending=False)

        print("\nTop 10 by SHAP Importance:")
        for i, row in shap_df.head(10).iterrows():
            print(f"  {row['feature']:30s} {row['shap_importance']:.4f}")

        shap_csv_path = output_dir / 'shap_importance.csv'
        shap_df.to_csv(shap_csv_path, index=False)

    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        print("Continuing without SHAP analysis...")


def save_model(model, feature_names, metrics, output_path: str):
    """Save trained model with metadata"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        'model': model,
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'metrics': metrics,
        'version': '1.0.0-config-optimizer',
        'description': 'XGBoost Regressor trained on Optuna trials to predict PF from config params'
    }

    joblib.dump(model_data, output_path)
    print(f"\nModel saved to: {output_path}")

    # Save metadata JSON
    metadata_path = output_path.with_suffix('.json')
    metadata = {
        'version': model_data['version'],
        'description': model_data['description'],
        'n_features': model_data['n_features'],
        'feature_names': feature_names,
        'metrics': metrics
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description='Train config optimizer for meta-learning')
    parser.add_argument('--data', required=True, help='Path to consolidated trial CSV')
    parser.add_argument('--target', type=str, default='year_pf',
                        help='Target metric to optimize (year_pf, year_wr, etc)')
    parser.add_argument('--output', required=True, help='Path to save model (.pkl)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set fraction (default: 0.2)')
    parser.add_argument('--skip-shap', action='store_true',
                        help='Skip SHAP analysis (faster)')

    args = parser.parse_args()

    # Create output directory for analysis artifacts
    output_path = Path(args.output)
    analysis_dir = output_path.parent / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data(
        args.data, args.target, args.test_size
    )

    # Train model
    model, metrics = train_config_optimizer(X_train, y_train, X_test, y_test)

    # Feature importance analysis
    importance_df = analyze_feature_importance(model, feature_names, analysis_dir)

    # SHAP analysis
    if not args.skip_shap:
        generate_shap_analysis(model, X_train, X_test, feature_names, analysis_dir)

    # Save model
    save_model(model, feature_names, metrics, args.output)

    print(f"\n{'='*60}")
    print("CONFIG OPTIMIZER TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Model: {args.output}")
    print(f"Test R²: {metrics['test_r2']:.3f}")
    print(f"Test MAE: {metrics['test_mae']:.3f}")
    print(f"\nNext steps:")
    print(f"  1. Review feature importance in {analysis_dir}")
    print(f"  2. Use model with Bayesian optimization to suggest new configs")
    print(f"  3. Validate suggested configs on target regime (2024)")


if __name__ == '__main__':
    main()
