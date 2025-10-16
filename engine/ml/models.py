#!/usr/bin/env python3
"""
ML Models for Bull Machine v1.8.6
=================================

LightGBM/XGBoost models for config suggestion based on regime features.

Models:
- Regression: Predict Sharpe/PF for given regime + config
- Ranking: Rank configs by expected performance
- Multi-target: Predict multiple metrics (Sharpe, PF, MaxDD) jointly
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import pickle

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available - install with: pip install lightgbm")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available - install with: pip install xgboost")


class ConfigSuggestionModel:
    """
    ML model for suggesting optimal configs based on regime features.
    """

    def __init__(
        self,
        model_type: str = 'lightgbm',
        target: str = 'sharpe',
        model_params: Optional[Dict] = None
    ):
        """
        Initialize model.

        Args:
            model_type: 'lightgbm', 'xgboost', or 'linear'
            target: Target metric ('sharpe', 'pf', 'total_return_pct')
            model_params: Model hyperparameters
        """
        self.model_type = model_type
        self.target = target
        self.model = None
        self.feature_names = None
        self.normalization_params = None

        # Default hyperparameters
        if model_params is None:
            if model_type == 'lightgbm':
                self.model_params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'min_data_in_leaf': 10,
                    'max_depth': 6
                }
            elif model_type == 'xgboost':
                self.model_params = {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 5,
                    'verbosity': 0
                }
            else:  # linear
                self.model_params = {}
        else:
            self.model_params = model_params

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Train model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            feature_names: Feature names (will be inferred if not provided)

        Returns:
            Dict with training metrics
        """
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = list(X_train.columns)

        print(f"🤖 Training {self.model_type} model for target: {self.target}")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Train samples: {len(X_train)}")
        if X_val is not None:
            print(f"   Val samples: {len(X_val)}")

        # Convert to numpy
        X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train

        # Fill NaN with 0 (neutral value for missing features like TOTAL/USDT.D)
        X_train_np = np.nan_to_num(X_train_np, nan=0.0, posinf=0.0, neginf=0.0)
        y_train_np = np.nan_to_num(y_train_np, nan=0.0, posinf=0.0, neginf=0.0)

        if self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            train_data = lgb.Dataset(X_train_np, label=y_train_np, feature_name=self.feature_names)

            valid_sets = [train_data]
            valid_names = ['train']

            if X_val is not None and y_val is not None:
                X_val_np = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
                y_val_np = y_val.values if isinstance(y_val, pd.Series) else y_val
                # Fill NaN in validation data too
                X_val_np = np.nan_to_num(X_val_np, nan=0.0, posinf=0.0, neginf=0.0)
                y_val_np = np.nan_to_num(y_val_np, nan=0.0, posinf=0.0, neginf=0.0)
                val_data = lgb.Dataset(X_val_np, label=y_val_np, reference=train_data, feature_name=self.feature_names)
                valid_sets.append(val_data)
                valid_names.append('val')

            self.model = lgb.train(
                self.model_params,
                train_data,
                num_boost_round=500,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
            )

            # Get feature importance
            importance = self.model.feature_importance(importance_type='gain')
            feature_importance = dict(zip(self.feature_names, importance))
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

            print("\n🔍 Top 10 important features:")
            for feat, imp in list(feature_importance.items())[:10]:
                print(f"   {feat}: {imp:.1f}")

        elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            dtrain = xgb.DMatrix(X_train_np, label=y_train_np, feature_names=self.feature_names)

            evals = [(dtrain, 'train')]
            if X_val is not None and y_val is not None:
                X_val_np = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
                y_val_np = y_val.values if isinstance(y_val, pd.Series) else y_val
                # Fill NaN in validation data too
                X_val_np = np.nan_to_num(X_val_np, nan=0.0, posinf=0.0, neginf=0.0)
                y_val_np = np.nan_to_num(y_val_np, nan=0.0, posinf=0.0, neginf=0.0)
                dval = xgb.DMatrix(X_val_np, label=y_val_np, feature_names=self.feature_names)
                evals.append((dval, 'val'))

            self.model = xgb.train(
                self.model_params,
                dtrain,
                num_boost_round=500,
                evals=evals,
                early_stopping_rounds=50,
                verbose_eval=50
            )

            # Get feature importance
            importance = self.model.get_score(importance_type='gain')
            feature_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

            print("\n🔍 Top 10 important features:")
            for feat, imp in list(feature_importance.items())[:10]:
                print(f"   {feat}: {imp:.1f}")

        else:
            # Fallback to simple linear regression
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=1.0)
            self.model.fit(X_train_np, y_train_np)

            # Get feature importance (absolute coefficients)
            feature_importance = dict(zip(self.feature_names, np.abs(self.model.coef_)))
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

            print("\n🔍 Top 10 important features:")
            for feat, imp in list(feature_importance.items())[:10]:
                print(f"   {feat}: {imp:.3f}")

        # Evaluate on train set
        y_train_pred = self.predict(X_train)
        train_rmse = np.sqrt(np.mean((y_train_np - y_train_pred) ** 2))
        train_mae = np.mean(np.abs(y_train_np - y_train_pred))
        train_r2 = 1 - (np.sum((y_train_np - y_train_pred) ** 2) / np.sum((y_train_np - np.mean(y_train_np)) ** 2))

        metrics = {
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2
        }

        # Evaluate on val set
        if X_val is not None and y_val is not None:
            y_val_pred = self.predict(X_val)
            # Ensure y_val_np is defined (fill NaN if needed)
            if 'y_val_np' not in locals():
                y_val_np = y_val.values if isinstance(y_val, pd.Series) else y_val
                y_val_np = np.nan_to_num(y_val_np, nan=0.0, posinf=0.0, neginf=0.0)
            val_rmse = np.sqrt(np.mean((y_val_np - y_val_pred) ** 2))
            val_mae = np.mean(np.abs(y_val_np - y_val_pred))
            val_r2 = 1 - (np.sum((y_val_np - y_val_pred) ** 2) / np.sum((y_val_np - np.mean(y_val_np)) ** 2))

            metrics.update({
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'val_r2': val_r2
            })

        print(f"\n✅ Training complete:")
        print(f"   Train RMSE: {metrics['train_rmse']:.4f}, MAE: {metrics['train_mae']:.4f}, R²: {metrics['train_r2']:.4f}")
        if 'val_rmse' in metrics:
            print(f"   Val RMSE: {metrics['val_rmse']:.4f}, MAE: {metrics['val_mae']:.4f}, R²: {metrics['val_r2']:.4f}")

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict target for given features.

        Args:
            X: Feature DataFrame

        Returns:
            Predictions (numpy array)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        X_np = X.values if isinstance(X, pd.DataFrame) else X

        # Fill NaN with 0 (same as training)
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)

        if self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            return self.model.predict(X_np)
        elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            dmatrix = xgb.DMatrix(X_np, feature_names=self.feature_names)
            return self.model.predict(dmatrix)
        else:
            return self.model.predict(X_np)

    def save(self, model_path: str) -> None:
        """
        Save model to disk.

        Args:
            model_path: Path to save model (will add appropriate extension)
        """
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        if self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            self.model.save_model(str(model_path.with_suffix('.txt')))
        elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            self.model.save_model(str(model_path.with_suffix('.json')))
        else:
            with open(model_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(self.model, f)

        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'target': self.target,
            'feature_names': self.feature_names,
            'model_params': self.model_params,
            'normalization_params': self.normalization_params
        }

        with open(model_path.with_suffix('.meta.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"💾 Model saved to {model_path}")

    @classmethod
    def load(cls, model_path: str) -> 'ConfigSuggestionModel':
        """
        Load model from disk.

        Args:
            model_path: Path to model (without extension)

        Returns:
            Loaded model instance
        """
        model_path = Path(model_path)

        # Load metadata
        with open(model_path.with_suffix('.meta.json'), 'r') as f:
            metadata = json.load(f)

        # Create instance
        instance = cls(
            model_type=metadata['model_type'],
            target=metadata['target'],
            model_params=metadata['model_params']
        )
        instance.feature_names = metadata['feature_names']
        instance.normalization_params = metadata.get('normalization_params')

        # Load model
        if metadata['model_type'] == 'lightgbm' and LIGHTGBM_AVAILABLE:
            instance.model = lgb.Booster(model_file=str(model_path.with_suffix('.txt')))
        elif metadata['model_type'] == 'xgboost' and XGBOOST_AVAILABLE:
            instance.model = xgb.Booster()
            instance.model.load_model(str(model_path.with_suffix('.json')))
        else:
            with open(model_path.with_suffix('.pkl'), 'rb') as f:
                instance.model = pickle.load(f)

        print(f"📂 Model loaded from {model_path}")
        return instance


def rank_configs_by_prediction(
    model: ConfigSuggestionModel,
    candidate_configs: List[Dict],
    current_regime: Dict,
    top_n: int = 10
) -> List[Tuple[Dict, float]]:
    """
    Rank candidate configs by predicted performance in current regime.

    Args:
        model: Trained ConfigSuggestionModel
        candidate_configs: List of config dicts to rank
        current_regime: Current regime feature dict from build_regime_vector()
        top_n: Number of top configs to return

    Returns:
        List of (config, predicted_score) tuples, sorted by score descending
    """
    from engine.ml.featurize import build_training_row

    # Build feature vectors for all configs
    feature_rows = []
    for config in candidate_configs:
        # Build dummy metrics (not used for prediction)
        dummy_metrics = {
            'profit_factor': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'total_return_pct': 0.0,
            'avg_r_multiple': 0.0
        }

        row = build_training_row(current_regime, config, dummy_metrics)
        feature_rows.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(feature_rows)

    # Extract features (exclude target columns)
    exclude_cols = ['pf', 'sharpe', 'max_dd', 'total_trades', 'win_rate', 'total_return_pct', 'avg_r', 'timestamp', 'asset', 'start_date', 'end_date']
    feature_cols = [col for col in df.columns if col not in exclude_cols and not any(col.startswith(exc) for exc in exclude_cols)]

    X = df[feature_cols]

    # Predict
    predictions = model.predict(X)

    # Rank
    ranked = list(zip(candidate_configs, predictions))
    ranked.sort(key=lambda x: x[1], reverse=True)

    print(f"🏆 Top {top_n} configs by predicted {model.target}:")
    for i, (config, score) in enumerate(ranked[:top_n], 1):
        fusion_threshold = config.get('fusion', {}).get('entry_threshold_confidence', 0.70)
        stop_atr = config.get('exits', {}).get('atr_k', 1.0)
        print(f"   {i}. Predicted {model.target}={score:.3f} | Threshold={fusion_threshold:.2f}, StopATR={stop_atr:.2f}")

    return ranked[:top_n]
