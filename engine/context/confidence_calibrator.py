"""
Confidence Calibrator - Maps ensemble agreement to calibrated confidence.

This module contains the calibrator classes used to map raw ensemble agreement
scores to calibrated confidence based on empirical outcomes.

Author: Claude Code
Date: 2026-01-15
"""

from typing import Dict
import numpy as np


class CompositeCalibrator:
    """
    Composite calibrator as weighted combination of individual calibrators.

    This class is pickleable and can be saved/loaded.

    Attributes:
        return_cal: Isotonic regression calibrator for returns
        volatility_cal: Isotonic regression calibrator for volatility
        stability_cal: Isotonic regression calibrator for stability
        weights: Dict with weights for each calibrator
    """

    def __init__(
        self,
        return_cal,
        volatility_cal,
        stability_cal,
        weights: Dict[str, float] = None
    ):
        """
        Initialize composite calibrator.

        Args:
            return_cal: Return calibrator (IsotonicRegression)
            volatility_cal: Volatility calibrator (IsotonicRegression)
            stability_cal: Stability calibrator (IsotonicRegression)
            weights: Optional weights dict {'return': float, 'volatility': float, 'stability': float}
        """
        self.return_cal = return_cal
        self.volatility_cal = volatility_cal
        self.stability_cal = stability_cal

        # Default weights based on empirical performance
        if weights is None:
            weights = {'return': 0.0, 'volatility': 0.3, 'stability': 0.7}

        self.weights = weights

    def predict(self, X):
        """
        Predict calibrated confidence from raw agreement.

        Args:
            X: Array-like of raw confidence scores (ensemble agreement)

        Returns:
            Array of calibrated confidence scores
        """
        # Ensure X is array
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # Get predictions from each calibrator
        return_conf = self.return_cal.predict(X)
        vol_conf = self.volatility_cal.predict(X)
        stability_conf = self.stability_cal.predict(X)

        # Weighted combination
        composite = (
            self.weights['return'] * return_conf +
            self.weights['volatility'] * vol_conf +
            self.weights['stability'] * stability_conf
        )

        return composite

    def __repr__(self):
        return (
            f"CompositeCalibrator("
            f"weights={self.weights})"
        )
