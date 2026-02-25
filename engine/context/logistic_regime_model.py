"""
Logistic Regime Model - Production Implementation
==================================================

Machine learning-based regime classification using logistic regression with
calibrated probabilities. Designed to classify market conditions into one of
four regimes: crisis, risk_off, neutral, risk_on.

Architecture:
- Base Model: LogisticRegression with balanced class weights
- Calibration: CalibratedClassifierCV (Platt scaling)
- Preprocessing: StandardScaler for feature normalization
- Training: SMOTE oversampling for class balance

Features (12 total):
1. crash_frequency_7d - Recent crash count
2. crisis_persistence - EWMA of crisis composite score
3. aftershock_score - Decay-weighted recent events
4. RV_7 - 7-day realized volatility
5. RV_30 - 30-day realized volatility
6. drawdown_persistence - Sustained drawdown indicator
7. funding_Z - Funding rate z-score
8. volume_z_7d - Volume z-score
9. USDT.D - USDT dominance
10. BTC.D - BTC dominance
11. DXY_Z - DXY z-score
12. YC_SPREAD - 10Y-2Y yield spread

Performance Targets:
- Test accuracy: ~65%
- Crisis rate: 0.5-2% (rare, true crises only)
- Crisis recall: 18%+ (acceptable with rule-based hybrid)
- Crisis precision: 50%+ (avoid false positives)

Interface Contract (HybridRegimeModel compatible):
- classify(features: Dict) -> Dict with:
    - regime_label: str
    - regime_confidence: float (0.0-1.0)
    - regime_probs: Dict[str, float] (all 4 regimes)

Author: Claude Code (Backend Architect)
Date: 2026-01-19
"""

import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class LogisticRegimeModel:
    """
    Production-ready logistic regression model for regime classification.

    This model classifies market conditions into 4 regimes using 12 engineered
    features. It uses calibrated probabilities (Platt scaling) to provide
    well-calibrated confidence scores.

    The model is designed to be used in a hybrid system where rule-based
    crisis detection handles extreme events, and ML handles normal regimes.

    Attributes:
        model: Trained LogisticRegression model
        calibrator: CalibratedClassifierCV for probability calibration
        scaler: StandardScaler for feature normalization
        feature_order: List of feature names in correct order
        regime_labels: List of regime class labels
        use_calibration: Whether to use calibrated probabilities
        training_metadata: Dict of training configuration and metrics
        fallback_mode: Whether model is in fallback mode (no trained model)
    """

    # Expected feature set (in order)
    DEFAULT_FEATURE_ORDER = [
        'crash_frequency_7d',
        'crisis_persistence',
        'aftershock_score',
        'RV_7',
        'RV_30',
        'drawdown_persistence',
        'funding_Z',
        'volume_z_7d',
        'USDT.D',
        'BTC.D',
        'DXY_Z',
        'YC_SPREAD'
    ]

    # Regime class labels
    REGIME_LABELS = ['crisis', 'risk_off', 'neutral', 'risk_on']

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize LogisticRegimeModel.

        Args:
            model_path: Path to trained model pickle file. If None, searches
                       for latest version in models/ directory. If not found
                       or loading fails, operates in fallback mode (neutral).

        Raises:
            No exceptions raised - failures result in fallback mode with logging.
        """
        self.model = None
        self.calibrator = None
        self.scaler = None
        self.feature_order = self.DEFAULT_FEATURE_ORDER.copy()
        self.regime_labels = self.REGIME_LABELS.copy()
        self.use_calibration = True
        self.training_metadata = {}
        self.fallback_mode = True

        # Try to load trained model
        if model_path is None:
            model_path = self._find_latest_model()

        if model_path and Path(model_path).exists():
            try:
                self._load_model(model_path)
                self.fallback_mode = False
                logger.info(f"✓ LogisticRegimeModel loaded from {model_path}")
                logger.info(f"  Features: {len(self.feature_order)}")
                logger.info(f"  Regimes: {self.regime_labels}")
                logger.info(f"  Calibration: {'enabled' if self.use_calibration else 'disabled'}")
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}")
                logger.warning("Operating in fallback mode (always neutral)")
                self.fallback_mode = True
        else:
            if model_path:
                logger.warning(f"Model path not found: {model_path}")
            logger.warning("No trained model available - using fallback mode (always neutral)")
            self.fallback_mode = True

    def _find_latest_model(self) -> Optional[str]:
        """
        Find the latest trained model in models/ directory.

        Returns:
            Path to latest model file, or None if not found.
        """
        try:
            models_dir = Path(__file__).parent.parent.parent / 'models'

            # Search for logistic_regime_v*.pkl files
            patterns = [
                'logistic_regime_v*.pkl',
                'logistic_regime.pkl'
            ]

            for pattern in patterns:
                matches = list(models_dir.glob(pattern))
                if matches:
                    # Sort by version number (extract from filename)
                    def extract_version(path):
                        name = path.stem
                        if '_v' in name:
                            try:
                                return int(name.split('_v')[-1])
                            except ValueError:
                                return 0
                        return 0

                    matches.sort(key=extract_version, reverse=True)
                    return str(matches[0])

            return None

        except Exception as e:
            logger.warning(f"Error searching for models: {e}")
            return None

    def _load_model(self, model_path: str):
        """
        Load trained model from pickle file.

        Expected pickle format:
        {
            'model': LogisticRegression,
            'calibrator': CalibratedClassifierCV,
            'scaler': StandardScaler,
            'feature_order': List[str],
            'regime_labels': List[str],
            'use_calibration': bool,
            'training_metadata': Dict
        }

        Args:
            model_path: Path to pickle file

        Raises:
            Exception: If loading fails or format is invalid
        """
        with open(model_path, 'rb') as f:
            artifact = pickle.load(f)

        # Validate artifact structure
        required_keys = ['model', 'scaler', 'feature_order']
        for key in required_keys:
            if key not in artifact:
                raise ValueError(f"Model artifact missing required key: {key}")

        # Load components
        self.model = artifact['model']
        self.scaler = artifact['scaler']
        self.feature_order = artifact['feature_order']

        # Optional components
        self.calibrator = artifact.get('calibrator', None)
        self.regime_labels = artifact.get('regime_labels', self.REGIME_LABELS)
        self.use_calibration = artifact.get('use_calibration', True)
        self.training_metadata = artifact.get('training_metadata', {})

        # Validate feature count
        if len(self.feature_order) != 12:
            logger.warning(f"Expected 12 features, got {len(self.feature_order)}")

        # Validate regime labels
        if len(self.regime_labels) != 4:
            raise ValueError(f"Expected 4 regime labels, got {len(self.regime_labels)}")

        # Log training metadata if available
        if self.training_metadata:
            logger.info(f"  Model version: {self.training_metadata.get('version', 'unknown')}")
            logger.info(f"  Training date: {self.training_metadata.get('training_date', 'unknown')}")
            logger.info(f"  Test accuracy: {self.training_metadata.get('test_accuracy', 'unknown')}")

    def classify(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify regime from feature dict.

        This is the primary interface method used by HybridRegimeModel and
        other consumers. It returns a standardized dict with regime prediction
        and calibrated probabilities.

        Args:
            features: Dict mapping feature names to values. Must contain all
                     12 required features (or will be zero-filled).

        Returns:
            Dict with:
                - regime_label: str (crisis/risk_off/neutral/risk_on)
                - regime_confidence: float (probability of predicted regime)
                - regime_probs: Dict[str, float] (all 4 regime probabilities)

        Example:
            >>> features = {
            ...     'crash_frequency_7d': 0,
            ...     'crisis_persistence': 0.1,
            ...     'RV_7': 45.0,
            ...     ...
            ... }
            >>> result = model.classify(features)
            >>> print(result['regime_label'])  # 'neutral'
            >>> print(result['regime_confidence'])  # 0.65
            >>> print(result['regime_probs'])  # {'crisis': 0.02, ...}
        """
        if self.fallback_mode:
            return self._fallback_classification()

        try:
            # Validate and extract features
            if not self._validate_features(features):
                logger.warning("Feature validation failed, using fallback")
                return self._fallback_classification()

            # Extract features in correct order (zero-fill missing)
            X = self._extract_features(features)

            # Scale features
            X_scaled = self.scaler.transform(X.reshape(1, -1))

            # Predict probabilities
            if self.use_calibration and self.calibrator is not None:
                proba = self.calibrator.predict_proba(X_scaled)[0]
                y_pred = self.calibrator.predict(X_scaled)[0]
            else:
                proba = self.model.predict_proba(X_scaled)[0]
                y_pred = self.model.predict(X_scaled)[0]

            # Map to regime labels
            regime_probs = {
                label: float(prob)
                for label, prob in zip(self.regime_labels, proba)
            }

            # Predicted regime
            regime_label = str(y_pred)

            # Confidence is probability of predicted regime
            regime_confidence = float(regime_probs.get(regime_label, 0.5))

            return {
                'regime_label': regime_label,
                'regime_confidence': regime_confidence,
                'regime_probs': regime_probs
            }

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            logger.error("Falling back to neutral regime")
            return self._fallback_classification()

    def classify_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify regime for batch of bars (for backtesting).

        This method is optimized for bulk classification of historical data.
        It processes the entire dataframe at once for efficiency.

        Args:
            df: DataFrame with feature columns. Must contain all 12 required
                features as columns.

        Returns:
            DataFrame with added columns:
                - regime_label: str (predicted regime)
                - regime_confidence: float (confidence in prediction)
                - regime_probs_crisis: float
                - regime_probs_risk_off: float
                - regime_probs_neutral: float
                - regime_probs_risk_on: float

        Note:
            Original DataFrame is not modified. Returns a copy with new columns.
        """
        if self.fallback_mode:
            return self._fallback_classification_batch(df)

        try:
            # Extract features for all rows
            X = self._extract_features_batch(df)

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Predict probabilities
            if self.use_calibration and self.calibrator is not None:
                proba = self.calibrator.predict_proba(X_scaled)
                y_pred = self.calibrator.predict(X_scaled)
            else:
                proba = self.model.predict_proba(X_scaled)
                y_pred = self.model.predict(X_scaled)

            # Create result dataframe
            result = df.copy()

            # Add predictions
            result['regime_label'] = y_pred

            # Add probabilities for each regime
            for i, label in enumerate(self.regime_labels):
                result[f'regime_probs_{label}'] = proba[:, i]

            # Calculate confidence (probability of predicted regime)
            result['regime_confidence'] = np.array([
                proba[i, self.regime_labels.index(label)]
                for i, label in enumerate(y_pred)
            ])

            # Add regime_probs as dict column (for compatibility)
            result['regime_probs'] = [
                {label: float(proba[i, j]) for j, label in enumerate(self.regime_labels)}
                for i in range(len(result))
            ]

            return result

        except Exception as e:
            logger.error(f"Batch classification failed: {e}")
            logger.error("Falling back to neutral regime for entire batch")
            return self._fallback_classification_batch(df)

    def _extract_features(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Extract features in correct order, zero-filling missing values.

        Args:
            features: Dict mapping feature names to values

        Returns:
            1D numpy array of shape (12,) with features in correct order
        """
        X = np.zeros(len(self.feature_order), dtype=np.float64)

        for i, feature_name in enumerate(self.feature_order):
            value = features.get(feature_name, 0.0)

            # Handle NaN/None
            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = 0.0

            X[i] = float(value)

        # Replace any remaining NaN with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X

    def _extract_features_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract features for batch processing.

        Args:
            df: DataFrame with feature columns

        Returns:
            2D numpy array of shape (n_samples, 12)
        """
        # Check which features are missing
        missing_features = [f for f in self.feature_order if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing features (will zero-fill): {missing_features}")

        # Extract features in order
        X = np.zeros((len(df), len(self.feature_order)), dtype=np.float64)

        for i, feature_name in enumerate(self.feature_order):
            if feature_name in df.columns:
                X[:, i] = df[feature_name].values
            else:
                X[:, i] = 0.0

        # Replace NaN with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X

    def _validate_features(self, features: Dict[str, Any]) -> bool:
        """
        Validate that features dict has reasonable values.

        Args:
            features: Dict of feature values

        Returns:
            True if features are valid, False otherwise
        """
        if not isinstance(features, dict):
            logger.error(f"Features must be dict, got {type(features)}")
            return False

        # Check if we have at least some features
        available_features = [f for f in self.feature_order if f in features]
        if len(available_features) < 6:  # Require at least half
            logger.warning(f"Only {len(available_features)}/12 features available")
            return False

        return True

    def _fallback_classification(self) -> Dict[str, Any]:
        """
        Return neutral regime as fallback when model unavailable or fails.

        Returns:
            Dict with neutral regime prediction and uniform probabilities
        """
        return {
            'regime_label': 'neutral',
            'regime_confidence': 0.50,
            'regime_probs': {
                'crisis': 0.0,
                'risk_off': 0.25,
                'neutral': 0.50,
                'risk_on': 0.25
            }
        }

    def _fallback_classification_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return neutral regime for all bars in batch (fallback mode).

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with neutral predictions for all rows
        """
        result = df.copy()
        n = len(result)

        result['regime_label'] = 'neutral'
        result['regime_confidence'] = 0.50
        result['regime_probs_crisis'] = 0.0
        result['regime_probs_risk_off'] = 0.25
        result['regime_probs_neutral'] = 0.50
        result['regime_probs_risk_on'] = 0.25
        result['regime_probs'] = [{
            'crisis': 0.0,
            'risk_off': 0.25,
            'neutral': 0.50,
            'risk_on': 0.25
        }] * n

        return result

    @staticmethod
    def create_model_artifact(
        model: LogisticRegression,
        scaler: StandardScaler,
        feature_order: List[str],
        calibrator: Optional[CalibratedClassifierCV] = None,
        regime_labels: Optional[List[str]] = None,
        use_calibration: bool = True,
        training_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create model artifact dict for pickling.

        This is a helper method for training scripts to create properly
        formatted model artifacts that can be loaded by this class.

        Args:
            model: Trained LogisticRegression model
            scaler: Fitted StandardScaler
            feature_order: List of feature names in training order
            calibrator: Optional CalibratedClassifierCV for probabilities
            regime_labels: Optional list of regime labels
            use_calibration: Whether to use calibrated probabilities
            training_metadata: Optional dict with training info

        Returns:
            Dict ready to be pickled as model artifact

        Example:
            >>> artifact = LogisticRegimeModel.create_model_artifact(
            ...     model=trained_model,
            ...     scaler=fitted_scaler,
            ...     feature_order=['crash_frequency_7d', 'RV_7', ...],
            ...     calibrator=calibrated_classifier,
            ...     training_metadata={'version': 'v2', 'accuracy': 0.649}
            ... )
            >>> with open('logistic_regime_v2.pkl', 'wb') as f:
            ...     pickle.dump(artifact, f)
        """
        if regime_labels is None:
            regime_labels = LogisticRegimeModel.REGIME_LABELS

        if training_metadata is None:
            training_metadata = {}

        return {
            'model': model,
            'calibrator': calibrator,
            'scaler': scaler,
            'feature_order': feature_order,
            'regime_labels': regime_labels,
            'use_calibration': use_calibration,
            'training_metadata': training_metadata
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from logistic regression coefficients.

        Returns:
            DataFrame with columns:
                - feature: feature name
                - importance: absolute coefficient magnitude
                - coefficient: raw coefficient value
            Sorted by importance (descending)

        Raises:
            ValueError: If model not loaded (fallback mode)
        """
        if self.fallback_mode or self.model is None:
            raise ValueError("Model not loaded - cannot compute feature importance")

        # Get coefficients (one per regime, for multinomial)
        # Take L2 norm across all regime coefficients for each feature
        coefs = self.model.coef_  # Shape: (n_classes, n_features)

        # Calculate importance as L2 norm across classes
        importance = np.linalg.norm(coefs, axis=0)

        # Create dataframe
        df = pd.DataFrame({
            'feature': self.feature_order,
            'importance': importance,
        })

        # Add individual regime coefficients
        for i, regime in enumerate(self.regime_labels):
            df[f'coef_{regime}'] = coefs[i, :]

        # Sort by importance
        df = df.sort_values('importance', ascending=False)

        return df

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata and configuration.

        Returns:
            Dict with model information:
                - fallback_mode: bool
                - feature_count: int
                - regime_labels: List[str]
                - use_calibration: bool
                - training_metadata: Dict
        """
        return {
            'fallback_mode': self.fallback_mode,
            'feature_count': len(self.feature_order),
            'feature_order': self.feature_order.copy(),
            'regime_labels': self.regime_labels.copy(),
            'use_calibration': self.use_calibration,
            'training_metadata': self.training_metadata.copy()
        }


def load_model(model_path: str) -> LogisticRegimeModel:
    """
    Convenience function to load a trained model.

    Args:
        model_path: Path to model pickle file

    Returns:
        Loaded LogisticRegimeModel instance

    Example:
        >>> model = load_model('models/logistic_regime_v2.pkl')
        >>> result = model.classify(features)
    """
    return LogisticRegimeModel(model_path=model_path)


if __name__ == '__main__':
    # Simple test/demo

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("LogisticRegimeModel - Production Implementation")
    print("=" * 60)

    # Initialize model (will use fallback if no trained model available)
    model = LogisticRegimeModel()

    print("\nModel Status:")
    print(f"  Fallback Mode: {model.fallback_mode}")
    print(f"  Features: {len(model.feature_order)}")
    print(f"  Regimes: {model.regime_labels}")

    # Test with dummy features
    test_features = {
        'crash_frequency_7d': 0,
        'crisis_persistence': 0.1,
        'aftershock_score': 0.05,
        'RV_7': 45.0,
        'RV_30': 50.0,
        'drawdown_persistence': 0.2,
        'funding_Z': -0.5,
        'volume_z_7d': 0.3,
        'USDT.D': 5.2,
        'BTC.D': 54.0,
        'DXY_Z': 0.8,
        'YC_SPREAD': 0.4
    }

    print("\nTest Classification:")
    result = model.classify(test_features)
    print(f"  Regime: {result['regime_label']}")
    print(f"  Confidence: {result['regime_confidence']:.2%}")
    print("  Probabilities:")
    for regime, prob in result['regime_probs'].items():
        print(f"    {regime}: {prob:.2%}")

    # Test batch classification
    test_df = pd.DataFrame([test_features] * 3)
    batch_result = model.classify_batch(test_df)
    print("\nBatch Classification:")
    print(f"  Rows: {len(batch_result)}")
    print(f"  Regimes: {batch_result['regime_label'].tolist()}")

    print("\nImplementation complete. Model ready for production use.")
