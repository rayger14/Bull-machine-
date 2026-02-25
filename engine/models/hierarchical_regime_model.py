"""
Hierarchical Regime Classifier (Reusable Model Class)
=====================================================

Two-stage hierarchical architecture for regime detection:
- Stage 1: Binary crisis detector (XGBoost with asymmetric loss)
- Stage 2: Ensemble 3-way classifier for normal regimes (LightGBM + LogReg + RF)
- Stage 3: Temporal smoothing (EMA)

Author: Claude Code
Date: 2026-01-27
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')


class HierarchicalRegimeClassifier:
    """
    Hierarchical two-stage regime classifier.

    Stage 1: Binary crisis detector (XGBoost)
    Stage 2: 3-way normal regime classifier (ensemble)
    Stage 3: Temporal smoothing (EMA)
    """

    def __init__(
        self,
        crisis_threshold: float = 0.5,
        temporal_smoothing: bool = True,
        ema_alpha: float = 0.15,
        random_state: int = 42
    ):
        """
        Initialize hierarchical classifier.

        Args:
            crisis_threshold: Probability threshold for crisis detection
            temporal_smoothing: Whether to apply EMA smoothing
            ema_alpha: EMA smoothing factor (higher = more reactive)
            random_state: Random seed
        """
        self.crisis_threshold = crisis_threshold
        self.temporal_smoothing = temporal_smoothing
        self.ema_alpha = ema_alpha
        self.random_state = random_state

        # Preprocessing
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()

        # Models
        self.crisis_detector = None
        self.normal_regime_classifier = None

        # Feature names
        self.feature_names_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train hierarchical classifier.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (crisis, risk_off, neutral, risk_on)

        Returns:
            self
        """
        # Impute missing values
        X_imputed = self.imputer.fit_transform(X)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_imputed)

        # Stage 1: Train crisis detector
        y_binary = (y == 'crisis').astype(int)
        self.crisis_detector = self._train_crisis_detector(X_scaled, y_binary)

        # Stage 2: Train normal regime classifier
        mask_normal = y != 'crisis'
        X_normal = X_scaled[mask_normal]
        y_normal = y[mask_normal]
        self.normal_regime_classifier = self._train_normal_classifier(X_normal, y_normal)

        return self

    def _train_crisis_detector(self, X: np.ndarray, y_binary: np.ndarray):
        """
        Train Stage 1: Binary crisis detector with XGBoost.

        Uses heavy class imbalance handling:
        - scale_pos_weight (automatic)
        - sample_weight (100x for crisis samples)
        """
        crisis_count = (y_binary == 1).sum()
        non_crisis_count = (y_binary == 0).sum()

        if crisis_count == 0:
            # No crisis samples, use dummy threshold
            return None

        scale_pos_weight = non_crisis_count / crisis_count

        crisis_model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',
            eval_metric='aucpr',
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbosity=0
        )

        # Extra sample weights (100x for crisis)
        sample_weights = np.where(y_binary == 1, 100, 1)
        crisis_model.fit(X, y_binary, sample_weight=sample_weights)

        return crisis_model

    def _train_normal_classifier(self, X: np.ndarray, y: np.ndarray):
        """
        Train Stage 2: Ensemble 3-way classifier for normal regimes.

        Ensemble components:
        - LightGBM (weight=2)
        - Calibrated Logistic Regression (weight=1)
        - Random Forest (weight=1)
        """
        # LightGBM
        lgbm = LGBMClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            class_weight='balanced',
            random_state=self.random_state,
            verbosity=-1
        )

        # Calibrated Logistic Regression
        logreg = LogisticRegression(
            C=0.1,
            class_weight='balanced',
            max_iter=1000,
            random_state=self.random_state
        )
        logreg_calibrated = CalibratedClassifierCV(
            logreg,
            method='isotonic',
            cv=5
        )

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )

        # Voting ensemble (soft voting)
        ensemble = VotingClassifier(
            estimators=[
                ('lgbm', lgbm),
                ('logreg', logreg_calibrated),
                ('rf', rf)
            ],
            voting='soft',
            weights=[2, 1, 1]  # LightGBM gets 2x weight
        )

        ensemble.fit(X, y)

        return ensemble

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regimes with hierarchical logic.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Array of predicted labels
        """
        # Impute missing values
        X_imputed = self.imputer.transform(X)

        # Scale features
        X_scaled = self.scaler.transform(X_imputed)

        # Stage 1: Crisis detection
        if self.crisis_detector is None:
            # No crisis detector, all samples go to stage 2
            is_crisis = np.zeros(len(X), dtype=bool)
        else:
            crisis_probs = self.crisis_detector.predict_proba(X_scaled)[:, 1]
            is_crisis = crisis_probs > self.crisis_threshold

        # Stage 2: Normal regime classification
        predictions = np.array([''] * len(X), dtype=object)
        predictions[is_crisis] = 'crisis'

        if (~is_crisis).sum() > 0:
            X_normal = X_scaled[~is_crisis]
            normal_preds = self.normal_regime_classifier.predict(X_normal)
            predictions[~is_crisis] = normal_preds

        # Stage 3: Temporal smoothing (if enabled)
        if self.temporal_smoothing:
            predictions = self._apply_temporal_smoothing(predictions)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regime probabilities.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Probability matrix (n_samples, 4) for [crisis, risk_off, neutral, risk_on]
        """
        # Impute missing values
        X_imputed = self.imputer.transform(X)

        # Scale features
        X_scaled = self.scaler.transform(X_imputed)

        # Initialize probabilities
        n_samples = len(X)
        regime_order = ['crisis', 'neutral', 'risk_off', 'risk_on']
        probs = np.zeros((n_samples, len(regime_order)))

        # Stage 1: Crisis probabilities
        if self.crisis_detector is None:
            crisis_probs = np.zeros(n_samples)
        else:
            crisis_probs = self.crisis_detector.predict_proba(X_scaled)[:, 1]

        probs[:, 0] = crisis_probs  # Crisis probability

        # Stage 2: Normal regime probabilities
        if self.normal_regime_classifier is not None:
            normal_probs = self.normal_regime_classifier.predict_proba(X_scaled)
            normal_classes = self.normal_regime_classifier.classes_

            # Map normal regime probabilities
            for i, regime in enumerate(normal_classes):
                if regime in regime_order:
                    regime_idx = regime_order.index(regime)
                    probs[:, regime_idx] = normal_probs[:, i] * (1 - crisis_probs)

        return probs

    def _apply_temporal_smoothing(self, predictions: np.ndarray) -> np.ndarray:
        """
        Apply EMA smoothing to reduce regime oscillations.

        Uses exponential moving average of regime probabilities.
        """
        # Convert predictions to indices
        regime_map = {'crisis': 0, 'risk_off': 1, 'neutral': 2, 'risk_on': 3}
        pred_indices = np.array([regime_map.get(p, 2) for p in predictions])

        # Apply EMA smoothing
        smoothed = np.zeros(len(predictions))
        smoothed[0] = pred_indices[0]

        for i in range(1, len(predictions)):
            smoothed[i] = self.ema_alpha * pred_indices[i] + (1 - self.ema_alpha) * smoothed[i-1]

        # Round to nearest regime
        smoothed_indices = np.round(smoothed).astype(int)
        smoothed_indices = np.clip(smoothed_indices, 0, 3)

        # Convert back to labels
        reverse_map = {0: 'crisis', 1: 'risk_off', 2: 'neutral', 3: 'risk_on'}
        smoothed_preds = np.array([reverse_map[idx] for idx in smoothed_indices])

        return smoothed_preds


def load_hierarchical_model(model_path: str) -> HierarchicalRegimeClassifier:
    """
    Load saved hierarchical model from disk.

    Args:
        model_path: Path to .pkl file

    Returns:
        Loaded model instance
    """
    import pickle
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    model = HierarchicalRegimeClassifier(
        crisis_threshold=model_data.get('crisis_threshold', 0.5),
        temporal_smoothing=model_data.get('temporal_smoothing', True),
        ema_alpha=model_data.get('ema_alpha', 0.15),
        random_state=model_data.get('random_state', 42)
    )

    model.crisis_detector = model_data['crisis_detector']
    model.normal_regime_classifier = model_data['normal_regime_classifier']
    model.scaler = model_data['scaler']
    model.feature_names_ = model_data.get('feature_names')

    return model
