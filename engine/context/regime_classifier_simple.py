"""
Simple Rule-Based Regime Classifier

Uses crypto-specific metrics (market cap trends, funding, realized vol)
instead of traditional macro indicators (which aren't available).

Regimes:
- risk_on: Bull market (strong growth, positive sentiment)
- neutral: Sideways/choppy (moderate conditions)
- risk_off: Bear market (declining, negative sentiment)
- crisis: Extreme volatility/panic
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class SimpleRegimeClassifier:
    """
    Rule-based regime classifier using available crypto metrics.

    Uses:
    - Crypto market cap trends (TOTAL, TOTAL2)
    - Realized volatility (rv_20d, rv_60d)
    - Funding rates

    Does NOT use: VIX, DXY, MOVE, yields (placeholder data)
    """

    def __init__(self):
        """Initialize classifier with thresholds"""

        # Market cap growth thresholds (vs historical baseline)
        self.risk_on_mcap_min = 2.2e12  # >2.2T = potential bull
        self.risk_off_mcap_max = 1.8e12  # <1.8T = potential bear

        # Realized volatility thresholds
        self.crisis_rv_threshold = 0.08  # >8% daily vol = crisis
        self.high_rv_threshold = 0.05    # >5% daily vol = elevated
        self.low_rv_threshold = 0.02     # <2% daily vol = calm

        # Funding rate thresholds (annualized %)
        self.bullish_funding_min = 0.01   # >1% = bullish sentiment
        self.bearish_funding_max = -0.005 # <-0.5% = bearish sentiment

        logger.info("SimpleRegimeClassifier initialized")

    def classify(self, row: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify current regime based on crypto metrics.

        Args:
            row: Dict with keys TOTAL, TOTAL2, funding, rv_20d, rv_60d

        Returns:
            Dict with:
                - regime: str (risk_on/neutral/risk_off/crisis)
                - proba: Dict[str, float] (regime probabilities)
                - signals: Dict (intermediate signals for debugging)
        """

        # Extract metrics (with nan handling)
        total_mcap = row.get('TOTAL', np.nan)
        total2_mcap = row.get('TOTAL2', np.nan)
        funding = row.get('funding', 0.0)
        rv_20d = row.get('rv_20d', np.nan)
        rv_60d = row.get('rv_60d', np.nan)

        # Use 20d RV if available, else 60d, else default
        rv = rv_20d if not np.isnan(rv_20d) else (rv_60d if not np.isnan(rv_60d) else 0.03)

        # Signals
        signals = {
            'total_mcap': total_mcap,
            'total2_mcap': total2_mcap,
            'funding': funding,
            'rv': rv
        }

        # --- CRISIS DETECTION (highest priority) ---
        # Extreme realized volatility = crisis
        if rv > self.crisis_rv_threshold:
            regime = 'crisis'
            proba = {'risk_on': 0.1, 'neutral': 0.1, 'risk_off': 0.2, 'crisis': 0.6}
            signals['reason'] = f'crisis_rv_{rv:.3f}'

        # --- RISK ON (Bull Market) ---
        elif (not np.isnan(total_mcap) and total_mcap > self.risk_on_mcap_min and
              rv < self.high_rv_threshold and
              funding > self.bullish_funding_min):
            regime = 'risk_on'
            # Strong bull signals
            proba = {'risk_on': 0.7, 'neutral': 0.2, 'risk_off': 0.05, 'crisis': 0.05}
            signals['reason'] = f'bull_mcap_{total_mcap/1e12:.1f}T_funding_{funding:.3f}'

        elif (not np.isnan(total_mcap) and total_mcap > self.risk_on_mcap_min and
              rv < self.high_rv_threshold):
            regime = 'risk_on'
            # Moderate bull (no funding confirmation)
            proba = {'risk_on': 0.55, 'neutral': 0.3, 'risk_off': 0.1, 'crisis': 0.05}
            signals['reason'] = f'bull_mcap_{total_mcap/1e12:.1f}T'

        # --- RISK OFF (Bear Market) ---
        elif (not np.isnan(total_mcap) and total_mcap < self.risk_off_mcap_max and
              rv > self.low_rv_threshold and
              funding < self.bearish_funding_max):
            regime = 'risk_off'
            # Strong bear signals
            proba = {'risk_on': 0.05, 'neutral': 0.2, 'risk_off': 0.7, 'crisis': 0.05}
            signals['reason'] = f'bear_mcap_{total_mcap/1e12:.1f}T_funding_{funding:.3f}'

        elif (not np.isnan(total_mcap) and total_mcap < self.risk_off_mcap_max):
            regime = 'risk_off'
            # Moderate bear (no funding confirmation)
            proba = {'risk_on': 0.1, 'neutral': 0.3, 'risk_off': 0.55, 'crisis': 0.05}
            signals['reason'] = f'bear_mcap_{total_mcap/1e12:.1f}T'

        # --- HIGH VOLATILITY (not crisis, but elevated risk) ---
        elif rv > self.high_rv_threshold:
            regime = 'risk_off'
            # High vol without other signals = defensive
            proba = {'risk_on': 0.15, 'neutral': 0.25, 'risk_off': 0.5, 'crisis': 0.1}
            signals['reason'] = f'high_vol_{rv:.3f}'

        # --- NEUTRAL (default) ---
        else:
            regime = 'neutral'
            proba = {'risk_on': 0.25, 'neutral': 0.5, 'risk_off': 0.2, 'crisis': 0.05}

            # Determine why neutral
            if np.isnan(total_mcap):
                signals['reason'] = 'neutral_no_data'
            elif self.risk_off_mcap_max <= total_mcap <= self.risk_on_mcap_min:
                signals['reason'] = f'neutral_mcap_{total_mcap/1e12:.1f}T'
            else:
                signals['reason'] = 'neutral_default'

        return {
            'regime': regime,
            'proba': proba,
            'signals': signals
        }

    @classmethod
    def load(cls, model_path: str = None, feature_order: list = None):
        """
        Create classifier instance (compatible with RegimeClassifier API).

        Args:
            model_path: Ignored (no model file needed)
            feature_order: Ignored (rule-based, not ML)

        Returns:
            SimpleRegimeClassifier instance
        """
        return cls()
