"""
Logistic Regime Model - Stub Implementation
============================================

This is a stub/adapter implementation that wraps RegimeClassifier
to provide the LogisticRegimeModel interface expected by HybridRegimeModel.

The actual LogisticRegimeModel implementation was documented but never
committed to the repository. This stub allows HybridRegimeModel to function
by delegating to the existing RegimeClassifier (GMM-based).

Architecture:
- Wraps RegimeClassifier (GMM model)
- Translates classify() interface to match expected format
- Falls back to neutral regime if model unavailable

Author: Claude Code (System Restoration)
Date: 2026-01-19
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from engine.context.regime_classifier import RegimeClassifier

logger = logging.getLogger(__name__)


class LogisticRegimeModel:
    """
    Stub implementation of LogisticRegimeModel using RegimeClassifier backend.

    This provides the interface expected by HybridRegimeModel while delegating
    to the existing GMM-based RegimeClassifier.

    Expected interface (from HybridRegimeModel):
    - classify(features: Dict) -> Dict with keys:
      - 'regime_label': str
      - 'regime_confidence': float
      - 'regime_probs': Dict[str, float]
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize LogisticRegimeModel stub.

        Args:
            model_path: Path to trained GMM model (e.g., 'models/regime_gmm_v3.pkl')
                       If None or not found, uses neutral fallback mode.
        """
        self.backend = None
        self.fallback_mode = True

        if model_path and Path(model_path).exists():
            try:
                # Try to load as RegimeClassifier
                # Default feature order for regime classification
                feature_order = [
                    'VIX', 'VIX_Z', 'MOVE', 'DXY_Z',
                    'BTC.D', 'USDT.D', 'funding_z',
                    'volume_z', 'RV_7', 'RV_30'
                ]

                self.backend = RegimeClassifier.load(
                    model_path=model_path,
                    feature_order=feature_order,
                    model_type='gmm',
                    zero_fill_missing=True  # More robust
                )
                self.fallback_mode = False
                logger.info(f"✓ LogisticRegimeModel stub loaded GMM backend from {model_path}")

            except Exception as e:
                logger.warning(f"Failed to load GMM model from {model_path}: {e}")
                logger.warning("Using fallback mode (always neutral)")
                self.fallback_mode = True
        else:
            if model_path:
                logger.warning(f"Model path not found: {model_path}")
            logger.warning("LogisticRegimeModel stub using fallback mode (always neutral)")
            self.fallback_mode = True

    def classify(self, features: Dict) -> Dict:
        """
        Classify regime from features.

        Args:
            features: Dict of feature values

        Returns:
            Dict with:
                - 'regime_label': str (crisis/risk_off/neutral/risk_on)
                - 'regime_confidence': float (0.0-1.0)
                - 'regime_probs': Dict[str, float] (probability per regime)
        """
        if self.fallback_mode or self.backend is None:
            # Fallback: always return neutral
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

        try:
            # Use RegimeClassifier backend
            result = self.backend.classify(features)

            # Translate RegimeClassifier format to LogisticRegimeModel format
            regime_label = result.get('regime', 'neutral')
            proba_dict = result.get('proba', {})

            # Calculate confidence as probability of predicted regime
            confidence = proba_dict.get(regime_label, 0.5)

            # Ensure all 4 regimes in proba dict
            regime_probs = {
                'crisis': proba_dict.get('crisis', 0.0),
                'risk_off': proba_dict.get('risk_off', 0.0),
                'neutral': proba_dict.get('neutral', 0.0),
                'risk_on': proba_dict.get('risk_on', 0.0)
            }

            return {
                'regime_label': regime_label,
                'regime_confidence': float(confidence),
                'regime_probs': regime_probs
            }

        except Exception as e:
            logger.error(f"LogisticRegimeModel classify failed: {e}")
            logger.error("Falling back to neutral")
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

    def classify_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify regime for batch of bars.

        Args:
            df: DataFrame with feature columns

        Returns:
            DataFrame with added columns:
                - regime_label
                - regime_confidence
                - regime_probs (as dict column)
        """
        if self.fallback_mode or self.backend is None:
            # Fallback: all neutral
            df = df.copy()
            df['regime_label'] = 'neutral'
            df['regime_confidence'] = 0.50
            df['regime_probs'] = [{
                'crisis': 0.0,
                'risk_off': 0.25,
                'neutral': 0.50,
                'risk_on': 0.25
            }] * len(df)
            return df

        try:
            # Use RegimeClassifier backend
            result_df = self.backend.classify_series(df)

            # Translate columns if needed
            if 'regime' in result_df.columns and 'regime_label' not in result_df.columns:
                result_df['regime_label'] = result_df['regime']

            if 'regime_label' not in result_df.columns:
                result_df['regime_label'] = 'neutral'

            if 'regime_confidence' not in result_df.columns:
                # Estimate confidence from proba if available
                result_df['regime_confidence'] = 0.50

            return result_df

        except Exception as e:
            logger.error(f"LogisticRegimeModel classify_batch failed: {e}")
            df = df.copy()
            df['regime_label'] = 'neutral'
            df['regime_confidence'] = 0.50
            return df
