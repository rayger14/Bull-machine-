#!/usr/bin/env python3
"""
XGBoost Trade Quality Filter Training

Trains a meta-classifier to predict trade success probability.
Uses time-series cross-validation to prevent lookahead bias.

Usage:
    python3 bin/train/train_trade_quality_filter.py \
        --data reports/ml/btc_trades_2022_2023.csv \
        --output models/btc_trade_quality_filter_v1.pkl
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score, classification_report
import joblib
import numpy as np
import argparse
from pathlib import Path


def load_and_prepare_data(data_path: str):
    """Load trade data and prepare features/target"""
    df = pd.read_csv(data_path)

    # Target: Win = R-multiple > 0
    df['target'] = (df['r_multiple'] > 0).astype(int)

    # Features (50+)
    feature_cols = [
        # Archetype one-hot
        'archetype_trap', 'archetype_retest', 'archetype_continuation', 'archetype_failed_continuation',
        'archetype_compression', 'archetype_exhaustion', 'archetype_reaccumulation',
        'archetype_trap_within_trend', 'archetype_wick_trap', 'archetype_volume_exhaustion',
        'archetype_ratio_coil_break', 'archetype_false_break_reversal',
        # Fusion & Liquidity
        'entry_fusion_score', 'entry_liquidity_score',
        # Market State
        'macro_regime_risk_on', 'macro_regime_neutral', 'macro_regime_risk_off', 'macro_regime_crisis',
        'vix_z_score', 'btc_volatility_percentile', 'volume_zscore', 'atr_percentile',
        'adx_14', 'rsi_14', 'macd_histogram',
        # MTF Alignment
        'tf1h_fusion', 'tf4h_fusion', 'tf1d_fusion',
        'tf4h_trend_aligned', 'tf1d_trend_aligned', 'nested_structure_quality',
        # Microstructure
        'boms_strength', 'fvg_quality', 'wyckoff_phase_score', 'poc_distance',
        'lvn_trap_risk', 'liquidity_sweep_strength',
        # Recent Performance
        'last_3_trades_wr', 'bars_since_last_trade', 'recent_dd_pct', 'streak_length',
        # Timing
        'hour_of_day', 'day_of_week', 'days_into_quarter'
    ]

    # Filter to only features that exist in data
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"Using {len(available_features)}/{len(feature_cols)} features")

    X = df[available_features].fillna(0)
    y = df['target']

    return X, y, available_features


def load_and_prepare_data_with_split(data_path: str, split_date: str):
    """Load trade data and split by time for train/validation"""
    df = pd.read_csv(data_path)

    # Parse entry time
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    split_dt = pd.to_datetime(split_date)

    # Make split_dt timezone-aware if entry_time has timezone
    if df['entry_time'].dt.tz is not None:
        split_dt = split_dt.tz_localize(df['entry_time'].dt.tz)

    # Time-based split
    train_df = df[df['entry_time'] < split_dt].copy()
    val_df = df[df['entry_time'] >= split_dt].copy()

    print(f"\n{'='*60}")
    print("TIME-BASED TRAIN/VAL SPLIT")
    print(f"{'='*60}")
    print(f"Split date: {split_date}")
    print(f"Train: {len(train_df)} trades ({train_df['entry_time'].min().date()} to {train_df['entry_time'].max().date()})")
    print(f"Val:   {len(val_df)} trades ({val_df['entry_time'].min().date()} to {val_df['entry_time'].max().date()})")

    # Target
    train_df['target'] = (train_df['r_multiple'] > 0).astype(int)
    val_df['target'] = (val_df['r_multiple'] > 0).astype(int)

    print(f"Train win rate: {train_df['target'].mean()*100:.1f}%")
    print(f"Val win rate:   {val_df['target'].mean()*100:.1f}%")

    # Features
    feature_cols = [
        # Archetype one-hot
        'archetype_trap', 'archetype_retest', 'archetype_continuation', 'archetype_failed_continuation',
        'archetype_compression', 'archetype_exhaustion', 'archetype_reaccumulation',
        'archetype_trap_within_trend', 'archetype_wick_trap', 'archetype_volume_exhaustion',
        'archetype_ratio_coil_break', 'archetype_false_break_reversal',
        # Fusion & Liquidity
        'entry_fusion_score', 'entry_liquidity_score',
        # Market State
        'macro_regime_risk_on', 'macro_regime_neutral', 'macro_regime_risk_off', 'macro_regime_crisis',
        'vix_z_score', 'btc_volatility_percentile', 'volume_zscore', 'atr_percentile',
        'adx_14', 'rsi_14', 'macd_histogram',
        # MTF Alignment
        'tf1h_fusion', 'tf4h_fusion', 'tf1d_fusion',
        'tf4h_trend_aligned', 'tf1d_trend_aligned', 'nested_structure_quality',
        # Microstructure
        'boms_strength', 'fvg_quality', 'wyckoff_phase_score', 'poc_distance',
        'lvn_trap_risk', 'liquidity_sweep_strength',
        # Recent Performance
        'last_3_trades_wr', 'bars_since_last_trade', 'recent_dd_pct', 'streak_length',
        # Timing
        'hour_of_day', 'day_of_week', 'days_into_quarter'
    ]

    available_features = [col for col in feature_cols if col in df.columns]
    print(f"Using {len(available_features)}/{len(feature_cols)} features")

    X_train = train_df[available_features].fillna(0)
    y_train = train_df['target']
    X_val = val_df[available_features].fillna(0)
    y_val = val_df['target']

    return X_train, y_train, X_val, y_val, available_features


def train_xgboost_time_split(X_train, y_train, X_val, y_val):
    """Train XGBoost with single time-based train/val split"""

    # Calculate class balance
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    print(f"\n{'='*60}")
    print(f"Training XGBoost with time-based split")
    print(f"{'='*60}")
    print(f"Train: {len(X_train)} samples, {y_train.mean()*100:.1f}% wins")
    print(f"Val:   {len(X_val)} samples, {y_val.mean()*100:.1f}% wins")
    print(f"Scale pos weight: {pos_weight:.2f}")

    # Train model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        max_depth=5,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=42,
        tree_method='hist',
        early_stopping_rounds=30
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Evaluate on validation set
    y_proba_val = model.predict_proba(X_val)[:, 1]
    auc_val = roc_auc_score(y_val, y_proba_val)

    # Find optimal threshold via precision-recall curve on validation set
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba_val)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-8)
    ix = np.argmax(f1_scores)
    optimal_threshold = thresholds[ix] if ix < len(thresholds) else 0.65
    optimal_f1 = f1_scores[ix]

    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"AUC: {auc_val:.3f}")
    print(f"Optimal F1: {optimal_f1:.3f} @ threshold {optimal_threshold:.3f}")
    print(f"Precision: {precision[ix]:.3f}, Recall: {recall[ix]:.3f}")

    # Also evaluate on training set for comparison
    y_proba_train = model.predict_proba(X_train)[:, 1]
    auc_train = roc_auc_score(y_train, y_proba_train)
    print(f"\nTrain AUC: {auc_train:.3f} (for reference)")

    # Create results dataframe
    results_df = pd.DataFrame([{
        'split': 'train',
        'auc': auc_train,
        'n_samples': len(X_train),
        'win_rate': y_train.mean()
    }, {
        'split': 'val',
        'auc': auc_val,
        'f1': optimal_f1,
        'threshold': optimal_threshold,
        'precision': precision[ix],
        'recall': recall[ix],
        'n_samples': len(X_val),
        'win_rate': y_val.mean()
    }])

    return model, optimal_threshold, results_df


def train_xgboost_tscv(X, y, n_splits=5):
    """Train XGBoost with time-series cross-validation"""

    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_threshold = 0.65
    best_f1 = 0
    best_model = None
    fold_results = []

    print(f"\n{'='*60}")
    print(f"Training XGBoost with {n_splits}-fold time-series CV")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Calculate class balance
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        print(f"\nFold {fold}/{n_splits}")
        print(f"  Train: {len(X_train)} samples, {y_train.mean()*100:.1f}% wins")
        print(f"  Val:   {len(X_val)} samples, {y_val.mean()*100:.1f}% wins")
        print(f"  Scale pos weight: {pos_weight:.2f}")

        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            max_depth=5,
            learning_rate=0.05,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            random_state=42,
            tree_method='hist',
            early_stopping_rounds=30
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Evaluate
        y_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_proba)

        # Find optimal threshold via precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
        f1_scores = 2 * recall * precision / (recall + precision + 1e-8)
        ix = np.argmax(f1_scores)
        fold_threshold = thresholds[ix] if ix < len(thresholds) else 0.65
        fold_f1 = f1_scores[ix]

        print(f"  AUC: {auc:.3f}")
        print(f"  Best F1: {fold_f1:.3f} @ threshold {fold_threshold:.3f}")
        print(f"  Precision: {precision[ix]:.3f}, Recall: {recall[ix]:.3f}")

        fold_results.append({
            'fold': fold,
            'auc': auc,
            'f1': fold_f1,
            'threshold': fold_threshold,
            'precision': precision[ix],
            'recall': recall[ix]
        })

        # Save best model
        if fold_f1 > best_f1:
            best_f1 = fold_f1
            best_threshold = fold_threshold
            best_model = model

    # Print summary
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    results_df = pd.DataFrame(fold_results)
    print(results_df.to_string(index=False))
    print(f"\nMean AUC: {results_df['auc'].mean():.3f} ± {results_df['auc'].std():.3f}")
    print(f"Mean F1:  {results_df['f1'].mean():.3f} ± {results_df['f1'].std():.3f}")
    print(f"\nBest Model: Fold {fold_results[np.argmax([r['f1'] for r in fold_results])]['fold']}")
    print(f"Best F1: {best_f1:.3f} @ threshold {best_threshold:.3f}")

    return best_model, best_threshold, results_df


def analyze_feature_importance(model, feature_names, top_n=20):
    """Analyze and display feature importance"""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]

    print(f"\n{'='*60}")
    print(f"TOP {top_n} MOST IMPORTANT FEATURES")
    print(f"{'='*60}")
    print(f"{'Rank':<6} {'Feature':<40} {'Importance':<10}")
    print("-" * 60)

    for rank, idx in enumerate(indices, 1):
        print(f"{rank:<6} {feature_names[idx]:<40} {importance[idx]:<10.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost trade quality filter')
    parser.add_argument('--data', required=True, help='Path to training CSV')
    parser.add_argument('--output', required=True, help='Path to save model (.pkl)')
    parser.add_argument('--n-splits', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--split-date', type=str, default=None,
                        help='Time-based split: train on data < split-date, validate on data >= split-date (e.g., 2024-10-01)')

    args = parser.parse_args()

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.data}")

    # Choose training strategy based on --split-date
    if args.split_date:
        # Time-based train/val split
        X_train, y_train, X_val, y_val, feature_names = load_and_prepare_data_with_split(
            args.data, args.split_date
        )

        print(f"\nTotal features: {len(feature_names)}")

        # Train with time-split
        model, threshold, results = train_xgboost_time_split(X_train, y_train, X_val, y_val)

        # Feature importance
        analyze_feature_importance(model, feature_names)

        # Save model
        model_data = {
            'model': model,
            'threshold': threshold,
            'feature_names': feature_names,
            'split_results': results,
            'training_stats': {
                'training_method': 'time_split',
                'split_date': args.split_date,
                'n_train': len(X_train),
                'n_val': len(X_val),
                'n_features': len(feature_names),
                'train_win_rate': float(y_train.mean()),
                'val_win_rate': float(y_val.mean()),
                'val_auc': float(results[results['split'] == 'val']['auc'].values[0]),
                'val_f1': float(results[results['split'] == 'val']['f1'].values[0])
            }
        }

        joblib.dump(model_data, args.output)
        print(f"\n✅ Model saved to {args.output}")
        print(f"   Training method: time-based split ({args.split_date})")
        print(f"   Optimal threshold: {threshold:.3f}")
        print(f"   Val F1: {model_data['training_stats']['val_f1']:.3f}")
        print(f"   Val AUC: {model_data['training_stats']['val_auc']:.3f}")

    else:
        # Traditional time-series cross-validation
        X, y, feature_names = load_and_prepare_data(args.data)

        print(f"\nDataset: {len(X)} samples, {len(feature_names)} features")
        print(f"Class balance: {y.mean()*100:.1f}% wins, {(1-y.mean())*100:.1f}% losses")

        # Train with CV
        model, threshold, cv_results = train_xgboost_tscv(X, y, n_splits=args.n_splits)

        # Feature importance
        analyze_feature_importance(model, feature_names)

        # Save model
        model_data = {
            'model': model,
            'threshold': threshold,
            'feature_names': feature_names,
            'cv_results': cv_results,
            'training_stats': {
                'training_method': 'time_series_cv',
                'n_splits': args.n_splits,
                'n_samples': len(X),
                'n_features': len(feature_names),
                'win_rate': float(y.mean()),
                'mean_cv_auc': float(cv_results['auc'].mean()),
                'mean_cv_f1': float(cv_results['f1'].mean())
            }
        }

        joblib.dump(model_data, args.output)
        print(f"\n✅ Model saved to {args.output}")
        print(f"   Training method: {args.n_splits}-fold time-series CV")
        print(f"   Optimal threshold: {threshold:.3f}")
        print(f"   Expected F1: {cv_results['f1'].mean():.3f}")


if __name__ == '__main__':
    main()
