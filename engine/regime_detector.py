#!/usr/bin/env python3
"""
Regime Detector v10 - GMM v3.1 Wrapper

Loads trained GMM model and classifies market regime from macro features.
Computes all required derived features (z-scores, spreads, etc.) from raw macro data.

Usage:
    from engine.regime_detector import RegimeDetector

    detector = RegimeDetector()

    # Classify single bar
    regime_label, confidence = detector.classify(macro_features_dict)

    # Classify full DataFrame
    df_with_regimes = detector.classify_batch(df)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# GMM v3.1 Required Features (19)
# ─────────────────────────────────────────────────────────────────────────────
REGIME_FEATURES = [
    'VIX_Z', 'DXY_Z', 'YC_SPREAD', 'YC_Z',
    'BTC.D_Z', 'USDT.D_Z',
    'RV_7', 'RV_20', 'RV_30', 'RV_60',
    'funding_Z', 'OI_CHANGE',
    'TOTAL_RET', 'TOTAL2_RET', 'TOTAL3_RET', 'ALT_ROTATION',
    'FOMC_D0', 'CPI_D0', 'NFP_D0'
]

# Event calendar (2022-2025)
FOMC_DATES = [
    '2022-01-26', '2022-03-16', '2022-05-04', '2022-06-15', '2022-07-27',
    '2022-09-21', '2022-11-02', '2022-12-14',
    '2023-02-01', '2023-03-22', '2023-05-03', '2023-06-14', '2023-07-26',
    '2023-09-20', '2023-11-01', '2023-12-13',
    '2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12', '2024-07-31',
    '2024-09-18', '2024-11-07', '2024-12-18',
    '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18', '2025-07-30'
]

CPI_DATES = [
    '2022-01-12', '2022-02-10', '2022-03-10', '2022-04-12', '2022-05-11', '2022-06-10',
    '2022-07-13', '2022-08-10', '2022-09-13', '2022-10-13', '2022-11-10', '2022-12-13',
    '2023-01-12', '2023-02-14', '2023-03-14', '2023-04-12', '2023-05-10', '2023-06-13',
    '2023-07-12', '2023-08-10', '2023-09-13', '2023-10-12', '2023-11-14', '2023-12-12',
    '2024-01-11', '2024-02-13', '2024-03-12', '2024-04-10', '2024-05-15', '2024-06-12',
    '2024-07-11', '2024-08-14', '2024-09-11', '2024-10-10', '2024-11-13', '2024-12-11',
    '2025-01-15', '2025-02-12', '2025-03-12', '2025-04-10', '2025-05-13', '2025-06-11'
]

NFP_DATES = [
    '2022-01-07', '2022-02-04', '2022-03-04', '2022-04-01', '2022-05-06', '2022-06-03',
    '2022-07-08', '2022-08-05', '2022-09-02', '2022-10-07', '2022-11-04', '2022-12-02',
    '2023-01-06', '2023-02-03', '2023-03-10', '2023-04-07', '2023-05-05', '2023-06-02',
    '2023-07-07', '2023-08-04', '2023-09-01', '2023-10-06', '2023-11-03', '2023-12-08',
    '2024-01-05', '2024-02-02', '2024-03-08', '2024-04-05', '2024-05-03', '2024-06-07',
    '2024-07-05', '2024-08-02', '2024-09-06', '2024-10-04', '2024-11-01', '2024-12-06',
    '2025-01-10', '2025-02-07', '2025-03-07', '2025-04-04', '2025-05-02', '2025-06-06'
]


class RegimeDetector:
    """
    Regime classifier using GMM v3.1.

    Loads trained GMM model, computes required features from raw macro data,
    and classifies market regime.
    """

    def __init__(self, model_path='models/regime_gmm_v3.1_fixed.pkl'):
        """Load trained GMM v3.1 model."""
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"GMM model not found: {model_path}\n"
                f"Run: python3 bin/train_regime_gmm_v3.1_fixed.py first"
            )

        with open(model_path, 'rb') as f:
            self.model_data = pickle.load(f)

        self.gmm = self.model_data['gmm']
        self.scaler = self.model_data['scaler']
        self.label_map = self.model_data['label_map']
        self.features = self.model_data['features']

        # Validate feature list matches expected
        if self.features != REGIME_FEATURES:
            print("⚠️  Warning: Model features differ from expected REGIME_FEATURES")
            print(f"   Model: {self.features}")
            print(f"   Expected: {REGIME_FEATURES}")

        # Convert event dates to datetime
        self._fomc_dates = pd.to_datetime(FOMC_DATES)
        self._cpi_dates = pd.to_datetime(CPI_DATES)
        self._nfp_dates = pd.to_datetime(NFP_DATES)

    @staticmethod
    def compute_z_score(series: pd.Series, window: int = 252) -> pd.Series:
        """Compute rolling z-score."""
        mean = series.rolling(window=window, min_periods=50).mean()
        std = series.rolling(window=window, min_periods=50).std()
        return (series - mean) / std.replace(0, np.nan)

    @staticmethod
    def compute_realized_volatility(returns: pd.Series, window: int) -> pd.Series:
        """Compute annualized realized volatility."""
        return returns.rolling(window=window).std() * np.sqrt(252)

    def is_event_day(self, timestamp: pd.Timestamp, event_dates: pd.DatetimeIndex) -> bool:
        """Check if timestamp is within 24h of event."""
        # Handle timezone mismatch - convert to naive for comparison
        ts_naive = timestamp.replace(tzinfo=None) if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo else timestamp

        for event_dt in event_dates:
            # Event dates are naive, compare with naive timestamp
            if abs((ts_naive - event_dt).total_seconds()) < 86400:  # 24 hours
                return True
        return False

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all 19 GMM features from raw macro data.

        Expects df to have columns: VIX, DXY, YIELD_2Y, YIELD_10Y, BTC.D, USDT.D,
                                     TOTAL, TOTAL2, funding, oi, rv_20d, rv_60d

        Returns:
            DataFrame with all 19 REGIME_FEATURES columns added
        """
        df = df.copy()

        # ─── 1. Realized Volatility (7, 20, 30, 60-day) ───
        if 'TOTAL' in df.columns:
            df['TOTAL_ret'] = df['TOTAL'].pct_change().fillna(0)
            df['RV_7'] = self.compute_realized_volatility(df['TOTAL_ret'], window=7*24)
            df['RV_30'] = self.compute_realized_volatility(df['TOTAL_ret'], window=30*24)

        # Rename existing rv_20d, rv_60d
        if 'rv_20d' in df.columns:
            df['RV_20'] = df['rv_20d']
            df['RV_60'] = df['rv_60d']

        # ─── 2. Open Interest Change ───
        if 'oi' in df.columns:
            df['OI_CHANGE'] = df['oi'].pct_change(periods=24).fillna(0) * 100

        # ─── 3. Yield Curve Features ───
        if 'YIELD_10Y' in df.columns and 'YIELD_2Y' in df.columns:
            df['YC_SPREAD'] = df['YIELD_10Y'] - df['YIELD_2Y']
            df['YC_Z'] = self.compute_z_score(df['YC_SPREAD'], window=252*24)

        # ─── 4. Breadth Returns & Alt Rotation ───
        if 'TOTAL' in df.columns:
            df['TOTAL_RET'] = df['TOTAL'].pct_change(periods=24).fillna(0) * 100
        if 'TOTAL2' in df.columns:
            df['TOTAL2_RET'] = df['TOTAL2'].pct_change(periods=24).fillna(0) * 100
        if 'TOTAL3' in df.columns:
            df['TOTAL3_RET'] = df['TOTAL3'].pct_change(periods=24).fillna(0) * 100

        # Alt rotation - if TOTAL3 not available, use TOTAL2 as proxy
        if 'TOTAL3_RET' in df.columns and 'TOTAL_RET' in df.columns:
            df['ALT_ROTATION'] = df['TOTAL3_RET'] - df['TOTAL_RET']
        elif 'TOTAL2_RET' in df.columns and 'TOTAL_RET' in df.columns:
            # Fallback: Use TOTAL2 (altcoins) as proxy for small caps
            df['ALT_ROTATION'] = df['TOTAL2_RET'] - df['TOTAL_RET']
            df['TOTAL3_RET'] = df['TOTAL2_RET']  # Use TOTAL2 as TOTAL3 proxy
        else:
            # Last resort: fill with zeros
            df['ALT_ROTATION'] = 0.0
            df['TOTAL3_RET'] = 0.0

        # ─── 5. Event Flags ───
        if 'timestamp' in df.columns:
            df['FOMC_D0'] = df['timestamp'].apply(
                lambda ts: int(self.is_event_day(ts, self._fomc_dates))
            )
            df['CPI_D0'] = df['timestamp'].apply(
                lambda ts: int(self.is_event_day(ts, self._cpi_dates))
            )
            df['NFP_D0'] = df['timestamp'].apply(
                lambda ts: int(self.is_event_day(ts, self._nfp_dates))
            )
        else:
            # If no timestamp, set all event flags to 0
            df['FOMC_D0'] = 0
            df['CPI_D0'] = 0
            df['NFP_D0'] = 0

        # ─── 6. Z-Scores for Continuous Features ───
        continuous_features = {
            'VIX': 'VIX_Z',
            'DXY': 'DXY_Z',
            'BTC.D': 'BTC.D_Z',
            'USDT.D': 'USDT.D_Z',
            'funding': 'funding_Z'
        }

        for raw_feat, z_feat in continuous_features.items():
            if raw_feat in df.columns:
                df[z_feat] = self.compute_z_score(df[raw_feat], window=252*24)

        return df

    def classify(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Classify regime from pre-computed features.

        Args:
            features: Dict with all 19 REGIME_FEATURES keys

        Returns:
            (regime_label, confidence): e.g., ('risk_on', 0.87)
        """
        # Extract features in correct order
        X = np.array([features[feat] for feat in self.features]).reshape(1, -1)

        # Check for NaNs
        if np.isnan(X).any():
            return ('neutral', 0.0)  # Default to neutral with 0 confidence if missing data

        # Scale and predict
        X_scaled = self.scaler.transform(X)
        cluster_label = self.gmm.predict(X_scaled)[0]
        probabilities = self.gmm.predict_proba(X_scaled)[0]

        # Map cluster to regime
        regime_label = self.label_map[cluster_label]
        confidence = probabilities.max()

        return regime_label, confidence

    def classify_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify multiple bars at once.

        Args:
            df: DataFrame with raw macro features

        Returns:
            DataFrame with added columns: regime_label, regime_confidence
        """
        # Engineer all required features
        df_features = self.engineer_features(df)

        # Extract feature matrix
        try:
            X = df_features[self.features].values
        except KeyError as e:
            missing_feats = set(self.features) - set(df_features.columns)
            raise ValueError(
                f"Missing required features: {missing_feats}\n"
                f"Available columns: {list(df_features.columns)}"
            ) from e

        # Handle missing data
        valid_mask = ~np.isnan(X).any(axis=1)

        # Initialize output columns
        df['regime_label'] = 'neutral'
        df['regime_confidence'] = 0.0

        if valid_mask.sum() == 0:
            print("⚠️  Warning: No valid samples (all NaN), returning neutral regime")
            return df

        # Classify valid samples
        X_valid = X[valid_mask]
        X_scaled = self.scaler.transform(X_valid)
        cluster_labels = self.gmm.predict(X_scaled)
        probabilities = self.gmm.predict_proba(X_scaled)

        # Map clusters to regimes
        regime_labels = [self.label_map[c] for c in cluster_labels]
        confidences = probabilities.max(axis=1)

        # Assign back to DataFrame
        df.loc[valid_mask, 'regime_label'] = regime_labels
        df.loc[valid_mask, 'regime_confidence'] = confidences

        return df


if __name__ == '__main__':
    """Quick test of regime detector."""
    print("\n" + "="*80)
    print("REGIME DETECTOR V10 - QUICK TEST")
    print("="*80)

    # Initialize detector
    detector = RegimeDetector()
    print("\n✅ Loaded GMM v3.1 model")
    print(f"   Features: {len(detector.features)}")
    print(f"   Label map: {detector.label_map}")

    # Test on dummy data
    dummy_features = {feat: 0.0 for feat in REGIME_FEATURES}
    dummy_features['VIX_Z'] = -0.5  # Low VIX (bullish)
    dummy_features['ALT_ROTATION'] = 1.0  # Alts outperforming (bullish)

    regime, conf = detector.classify(dummy_features)
    print("\n✅ Test classification:")
    print(f"   Regime: {regime}")
    print(f"   Confidence: {conf:.3f}")

    print("\n" + "="*80)
    print("✅ REGIME DETECTOR READY")
    print("="*80)
