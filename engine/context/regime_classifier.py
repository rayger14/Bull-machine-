"""
Regime Classifier for Bull Machine v1.9

Lightweight model that reads macro feature vectors and labels current regime
(risk_on / neutral / risk_off / crisis) using Gaussian Mixture Model or HMM.

V2 Enhancement: Supports both GMM (legacy) and HMM (rolling 21-day classifier).

Returns small, bounded deltas for fusion threshold, weights, and risk sizing.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class RegimeClassifier:
    """
    Regime classifier supporting both GMM and HMM models.

    V2 Enhancement: Can use either:
    - GMM (Gaussian Mixture Model) - legacy, static clustering
    - HMM (Hidden Markov Model) - rolling 21-day regime detection

    Uses macro features (VIX, MOVE, DXY, etc.) to classify market regime.
    Returns regime label + probability distribution.
    """

    def __init__(
        self,
        model,
        label_map: Dict[int, str],
        feature_order: list,
        model_type: str = 'gmm',
        zero_fill_missing: bool = False,
        regime_override: Optional[Dict[str, str]] = None
    ):
        """
        Initialize regime classifier

        Args:
            model: Trained model (GaussianMixture for GMM, HMMRegimeModel for HMM)
            label_map: Mapping from cluster ID to regime label
            feature_order: Ordered list of feature names
            model_type: 'gmm' or 'hmm_v2'
            zero_fill_missing: If True, fill missing features with 0 instead of falling back to neutral
            regime_override: Optional dict mapping date ranges to forced regimes (e.g., {"2022": "risk_off"})
        """
        self.model = model
        self.label_map = label_map
        self.feature_order = feature_order
        self.model_type = model_type
        self.zero_fill_missing = zero_fill_missing
        self.regime_override = regime_override or {}
        logger.info(f"Regime classifier initialized: model_type={model_type}")
        logger.info(f"Features: {len(self.feature_order)}")
        logger.info(f"Label map: {self.label_map}")
        if zero_fill_missing:
            logger.info("Zero-fill mode enabled for missing features")
        if self.regime_override:
            logger.info(f"Regime overrides active: {self.regime_override}")

    @classmethod
    def load(
        cls,
        model_path: str,
        feature_order: list,
        model_type: str = 'gmm',
        zero_fill_missing: bool = False,
        regime_override: Optional[Dict[str, str]] = None
    ):
        """
        Load trained regime classifier from pickle

        Args:
            model_path: Path to pickled model
            feature_order: Expected feature order
            model_type: 'gmm' or 'hmm_v2'
            zero_fill_missing: If True, fill missing features with 0 instead of falling back to neutral
            regime_override: Optional dict mapping date ranges to forced regimes

        Returns:
            RegimeClassifier instance
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        obj = pickle.loads(Path(model_path).read_bytes())

        # Detect model type from file if not specified
        if 'hmm' in str(model_path).lower() or obj.get('model_type') == 'hmm':
            detected_type = 'hmm_v2'
        else:
            detected_type = 'gmm'

        # Use detected type if model_type is default
        if model_type == 'gmm' and detected_type == 'hmm_v2':
            model_type = detected_type
            logger.info(f"Auto-detected model type: {model_type}")

        # Support both 'model' and 'gmm' keys for backward compatibility
        model_obj = obj.get("model") or obj.get("gmm")
        if model_obj is None or "label_map" not in obj:
            raise ValueError("Invalid model file: missing 'model'/'gmm' or 'label_map'")

        return cls(
            model=model_obj,
            label_map=obj["label_map"],
            feature_order=feature_order,
            model_type=model_type,
            zero_fill_missing=zero_fill_missing,
            regime_override=regime_override
        )

    def classify(self, macro_row: Dict[str, float], timestamp=None) -> Dict[str, Any]:
        """
        Classify market regime from macro features

        Args:
            macro_row: Dictionary of macro feature values
            timestamp: Optional pandas Timestamp for date-based overrides

        Returns:
            {
                "regime": str (risk_on/neutral/risk_off/crisis),
                "proba": Dict[str, float] (probability per regime),
                "features_used": int (number of non-NaN features)
            }
        """
        # Check for date-based regime override
        if timestamp is not None and self.regime_override:
            year_str = str(timestamp.year)
            if year_str in self.regime_override:
                forced_regime = self.regime_override[year_str]
                logger.debug(f"Regime override: {timestamp} → {forced_regime}")
                return {
                    "regime": forced_regime,
                    "proba": {forced_regime: 1.0, "risk_on": 0.0, "neutral": 0.0, "risk_off": 0.0, "crisis": 0.0},
                    "features_used": len(self.feature_order),
                    "override": True
                }

        # Extract features in correct order
        x = np.array([macro_row.get(f, np.nan) for f in self.feature_order], dtype=float)

        # Check for missing values
        n_valid = np.sum(~np.isnan(x))
        n_missing = np.sum(np.isnan(x))

        if np.isnan(x).any():
            if self.zero_fill_missing:
                # Zero-fill missing features and continue with classification
                x[np.isnan(x)] = 0.0
                logger.info(f"Zero-filled {n_missing}/{len(x)} missing features for classification")
            else:
                # Conservative fallback when features missing
                logger.warning(f"Missing {n_missing}/{len(x)} features, using neutral fallback")
                return {
                    "regime": "neutral",
                    "proba": {"neutral": 1.0, "risk_on": 0.0, "risk_off": 0.0, "crisis": 0.0},
                    "features_used": n_valid,
                    "fallback": True
                }

        # Predict using GMM
        try:
            proba = self.model.predict_proba([x])[0]
            label_int = int(np.argmax(proba))
            regime = self.label_map[label_int]

            # Build probability dict (sum probabilities for regimes with multiple clusters)
            proba_dict = {}
            for cluster_id, regime_name in self.label_map.items():
                if cluster_id < len(proba):
                    # Add to existing probability (multiple clusters may map to same regime)
                    proba_dict[regime_name] = proba_dict.get(regime_name, 0.0) + float(proba[cluster_id])
                else:
                    # Ensure regime exists in dict even if cluster ID is out of bounds
                    proba_dict[regime_name] = proba_dict.get(regime_name, 0.0)

            return {
                "regime": regime,
                "proba": proba_dict,
                "features_used": n_valid,
                "fallback": False
            }

        except Exception as e:
            logger.error(f"Regime classification failed: {e}, using neutral fallback")
            return {
                "regime": "neutral",
                "proba": {"neutral": 1.0, "risk_on": 0.0, "risk_off": 0.0, "crisis": 0.0},
                "features_used": n_valid,
                "fallback": True,
                "error": str(e)
            }

    def classify_series(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify regime for a time series of macro data

        Args:
            macro_df: DataFrame with macro features (columns = feature_order)

        Returns:
            DataFrame with regime, proba, features_used columns added
        """
        results = []

        for idx, row in macro_df.iterrows():
            macro_row = row.to_dict()
            result = self.classify(macro_row)
            results.append(result)

        # Convert to DataFrame
        regime_df = pd.DataFrame(results, index=macro_df.index)

        return regime_df

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from GMM cluster centers

        Returns:
            Dict mapping feature name to importance score
        """
        if not hasattr(self.model, 'means_'):
            return {}

        # Compute variance across cluster centers for each feature
        means = self.model.means_  # Shape: (n_components, n_features)
        variances = np.var(means, axis=0)

        importance = {}
        for i, feat in enumerate(self.feature_order):
            importance[feat] = float(variances[i])

        return importance


# Example usage and testing
if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python regime_classifier.py <model_path>")
        print("\nTest with:")
        print("  python regime_classifier.py models/regime_classifier_gmm.pkl")
        sys.exit(1)

    model_path = sys.argv[1]

    # Load classifier
    feature_order = ["VIX", "DXY", "MOVE", "YIELD_2Y", "YIELD_10Y",
                     "USDT.D", "BTC.D", "TOTAL", "TOTAL2",
                     "funding", "oi", "rv_20d", "rv_60d"]

    try:
        rc = RegimeClassifier.load(model_path, feature_order)
        print(f"\n✅ Loaded regime classifier from {model_path}")

        # Test classification
        test_macro = {
            "VIX": 18.5, "DXY": 102.0, "MOVE": 85.0,
            "YIELD_2Y": 4.2, "YIELD_10Y": 4.0,
            "USDT.D": 6.8, "BTC.D": 54.5,
            "TOTAL": 1000, "TOTAL2": 400,
            "funding": 0.008, "oi": 0.012,
            "rv_20d": 0.02, "rv_60d": 0.025
        }

        result = rc.classify(test_macro)

        print("\n📊 Test Classification:")
        print(f"   Regime: {result['regime']}")
        print(f"   Confidence: {result['proba'][result['regime']]:.1%}")
        print("   All probabilities:")
        for regime, prob in result['proba'].items():
            print(f"     {regime:12s}: {prob:.1%}")

        # Feature importance
        importance = rc.get_feature_importance()
        print("\n📈 Feature Importance (variance across clusters):")
        for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:5]:
            print(f"     {feat:12s}: {imp:.4f}")

        print("\n✅ Regime classifier test passed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
