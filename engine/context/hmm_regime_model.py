"""
HMM Regime Model V2 - Production Implementation

4-state Hidden Markov Model for crypto regime classification.
Uses 21-day rolling windows with crypto-native features.

Architecture:
- 4 states: risk_on, neutral, risk_off, crisis
- 21-day (504 hour) rolling window
- 15 crypto-native features (funding, OI, liquidations, macro)
- Viterbi decoding for batch mode
- Incremental updates for stream mode

This is the BRAINSTEM of the Bull Machine - regime awareness that filters reality.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from collections import deque
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)

# 15 regime features (Tier 1-4 from research report)
REGIME_FEATURES_V2 = [
    # Tier 1: Crypto-native (highest signal)
    'funding_Z',           # 30-day z-score of funding rate
    'OI_CHANGE',           # 24h open interest % change
    'RV_21',               # 21-day realized volatility (annualized)
    'LIQ_VOL_24h',         # 24h liquidation volume ($M)

    # Tier 2: Market structure
    'USDT.D',              # USDT dominance (%)
    'BTC.D',               # BTC dominance (%)
    'TOTAL_RET_21d',       # Total market cap 21d return (%)
    'ALT_ROTATION',        # TOTAL3 outperformance vs TOTAL

    # Tier 3: Macro
    'VIX_Z',               # VIX z-score (252d window)
    'DXY_Z',               # DXY z-score (252d window)
    'YC_SPREAD',           # 10Y - 2Y yield (bps)
    'M2_GROWTH_YOY',       # M2 money supply YoY growth (%)

    # Tier 4: Event flags
    'FOMC_D0',             # 1 if FOMC day, else 0
    'CPI_D0',              # 1 if CPI release day, else 0
    'NFP_D0'               # 1 if NFP day, else 0
]


class HMMRegimeModel:
    """
    4-state Hidden Markov Model for regime classification.

    Supports:
    - Batch mode: Classify full historical dataset (Viterbi algorithm)
    - Stream mode: Incremental updates for live trading
    - Feature parity: Identical results in both modes
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize HMM regime model.

        Args:
            model_path: Path to trained HMM model pickle file
        """
        self.model = None
        self.state_map = {}
        self.scaler = None
        self.feature_order = REGIME_FEATURES_V2

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        """
        Load trained HMM model from pickle.

        Args:
            model_path: Path to model file
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"HMM model not found: {model_path}")

        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        self.model = data['model']
        self.state_map = data['state_map']
        self.scaler = data.get('scaler')
        # Prioritize 'feature_order' key (used by simplified training script)
        self.feature_order = data.get('feature_order', data.get('features', REGIME_FEATURES_V2))

        logger.info(f"Loaded HMM model from {model_path}")
        logger.info(f"State mapping: {self.state_map}")
        logger.info(f"Features: {len(self.feature_order)}")

    def classify_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify all bars in dataset using Viterbi algorithm (backtesting).

        Args:
            df: DataFrame with macro/price features

        Returns:
            DataFrame with added columns:
            - regime_label: [risk_on, neutral, risk_off, crisis]
            - regime_confidence: [0.0-1.0]
            - regime_proba_risk_on: P(risk_on)
            - regime_proba_neutral: P(neutral)
            - regime_proba_risk_off: P(risk_off)
            - regime_proba_crisis: P(crisis)
        """
        if self.model is None:
            raise ValueError("No HMM model loaded. Call load_model() first.")

        logger.info(f"Classifying {len(df)} bars with HMM...")

        # Extract features
        X = self._extract_features(df)

        # Handle NaNs (fill with 0 for robustness)
        n_missing = X.isna().sum().sum()
        if n_missing > 0:
            logger.warning(f"Filling {n_missing} missing feature values with 0")
            X = X.fillna(0)

        # Scale if scaler available
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X.values)
        else:
            X_scaled = X.values

        # Viterbi decode (most likely state sequence)
        states = self.model.predict(X_scaled)
        probs = self.model.predict_proba(X_scaled)

        # Map states to regime labels
        df = df.copy()
        df['regime_label'] = [self.state_map[s] for s in states]
        df['regime_confidence'] = probs.max(axis=1)

        # Add individual probabilities (sum over states mapping to same regime)
        for regime in ['risk_on', 'neutral', 'risk_off', 'crisis']:
            regime_states = [s for s, r in self.state_map.items() if r == regime]
            df[f'regime_proba_{regime}'] = probs[:, regime_states].sum(axis=1)

        # Log regime distribution
        regime_dist = df['regime_label'].value_counts()
        logger.info("Regime distribution:")
        for regime, count in regime_dist.items():
            pct = count / len(df) * 100
            logger.info(f"  {regime:12s}: {count:6d} ({pct:5.1f}%)")

        return df

    def classify_stream(self, bar: dict) -> Tuple[str, float]:
        """
        Classify single bar (live trading mode).

        This is a simplified implementation. For production stream mode,
        use StreamHMMClassifier which maintains rolling buffer.

        Args:
            bar: Dictionary with all features

        Returns:
            (regime_label, confidence)
        """
        if self.model is None:
            raise ValueError("No HMM model loaded. Call load_model() first.")

        # Extract features
        x = np.array([bar.get(f, 0.0) for f in self.feature_order], dtype=float)

        # Scale if scaler available
        if self.scaler is not None:
            x = self.scaler.transform([x])[0]

        # Predict
        state = self.model.predict([x])[0]
        probs = self.model.predict_proba([x])[0]

        regime = self.state_map[state]
        confidence = probs[state]

        return regime, confidence

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract HMM features from DataFrame.

        Args:
            df: Raw DataFrame with price/macro columns

        Returns:
            DataFrame with engineered features
        """
        # Check which features already exist
        existing = [f for f in self.feature_order if f in df.columns]
        missing = [f for f in self.feature_order if f not in df.columns]

        if missing:
            logger.warning(f"Missing {len(missing)} features: {missing}")
            logger.warning("Will attempt to compute them on the fly")

        features = pd.DataFrame(index=df.index)

        for feat in self.feature_order:
            if feat in df.columns:
                # Feature already exists
                features[feat] = df[feat]
            elif feat == 'funding_Z':
                # Compute funding z-score
                if 'funding' in df.columns or 'funding_rate' in df.columns:
                    funding_col = 'funding' if 'funding' in df.columns else 'funding_rate'
                    features[feat] = self._rolling_zscore(df[funding_col], window=30*24)
                else:
                    features[feat] = 0.0
            elif feat == 'OI_CHANGE':
                # Compute 24h OI change
                if 'oi' in df.columns:
                    features[feat] = df['oi'].pct_change(24) * 100
                else:
                    features[feat] = 0.0
            elif feat == 'RV_21':
                # Compute 21-day realized vol
                if 'close' in df.columns:
                    returns = df['close'].pct_change()
                    features[feat] = self._realized_vol(returns, window=21*24)
                elif 'RV_20' in df.columns:
                    # Approximate with RV_20
                    features[feat] = df['RV_20'] * 100  # Convert to annualized %
                else:
                    features[feat] = 0.0
            elif feat == 'LIQ_VOL_24h':
                # 24h liquidation volume
                if 'liquidations' in df.columns:
                    features[feat] = df['liquidations'].rolling(24).sum() / 1e6
                else:
                    features[feat] = 0.0
            elif feat == 'TOTAL_RET_21d':
                # TOTAL market cap 21d returns
                if 'TOTAL' in df.columns:
                    features[feat] = df['TOTAL'].pct_change(21*24) * 100
                elif 'TOTAL_RET' in df.columns:
                    features[feat] = df['TOTAL_RET'] * 100
                else:
                    features[feat] = 0.0
            elif feat == 'ALT_ROTATION':
                # Altcoin rotation score
                if 'TOTAL3' in df.columns and 'TOTAL' in df.columns:
                    total3_ret = df['TOTAL3'].pct_change(21*24)
                    total_ret = df['TOTAL'].pct_change(21*24)
                    features[feat] = (total3_ret - total_ret) * 100
                else:
                    features[feat] = 0.0
            elif feat == 'VIX_Z':
                # VIX z-score
                if 'VIX' in df.columns:
                    features[feat] = self._rolling_zscore(df['VIX'], window=252*24)
                elif 'VIX_Z' in df.columns:
                    features[feat] = df['VIX_Z']
                else:
                    features[feat] = 0.0
            elif feat == 'DXY_Z':
                # DXY z-score
                if 'DXY' in df.columns:
                    features[feat] = self._rolling_zscore(df['DXY'], window=252*24)
                elif 'DXY_Z' in df.columns:
                    features[feat] = df['DXY_Z']
                else:
                    features[feat] = 0.0
            elif feat == 'YC_SPREAD':
                # Yield curve spread
                if 'YIELD_10Y' in df.columns and 'YIELD_2Y' in df.columns:
                    features[feat] = (df['YIELD_10Y'] - df['YIELD_2Y']) * 100
                elif 'YC_SPREAD' in df.columns:
                    features[feat] = df['YC_SPREAD']
                else:
                    features[feat] = 0.0
            elif feat == 'M2_GROWTH_YOY':
                # M2 YoY growth
                if 'M2' in df.columns:
                    features[feat] = df['M2'].pct_change(252*24) * 100
                else:
                    features[feat] = 0.0
            elif feat in ['FOMC_D0', 'CPI_D0', 'NFP_D0']:
                # Event flags (not implemented yet)
                features[feat] = 0.0
            else:
                # Unknown feature, set to 0
                logger.warning(f"Unknown feature {feat}, setting to 0")
                features[feat] = 0.0

        return features

    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int, min_periods: int = 50) -> pd.Series:
        """Compute rolling z-score."""
        mean = series.rolling(window, min_periods=min_periods).mean()
        std = series.rolling(window, min_periods=min_periods).std()
        return (series - mean) / std.replace(0, np.nan)

    @staticmethod
    def _realized_vol(returns: pd.Series, window: int) -> pd.Series:
        """Compute annualized realized volatility (%)."""
        return returns.rolling(window).std() * np.sqrt(252 * 24) * 100


class StreamHMMClassifier:
    """
    Incremental HMM classifier for live trading.

    Maintains 504-bar (21-day) rolling buffer and re-decodes
    on each new bar using Viterbi algorithm.
    """

    def __init__(self, model_path: str, window_size: int = 504):
        """
        Initialize stream classifier.

        Args:
            model_path: Path to trained HMM model
            window_size: Rolling window size (default: 504 = 21 days * 24h)
        """
        self.hmm = HMMRegimeModel(model_path)
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.current_regime = 'neutral'
        self.current_confidence = 0.0

    def update(self, bar: dict) -> Tuple[str, float]:
        """
        Update with new bar and return current regime.

        Args:
            bar: Dictionary with all features

        Returns:
            (regime_label, confidence)
        """
        # Add to buffer
        self.buffer.append(bar)

        if len(self.buffer) < self.window_size:
            # Warmup period - not enough data yet
            logger.debug(f"Warmup: {len(self.buffer)}/{self.window_size} bars")
            return 'neutral', 0.0

        # Convert buffer to DataFrame
        df = pd.DataFrame(list(self.buffer))

        # Classify using batch mode (Viterbi over rolling window)
        df_classified = self.hmm.classify_batch(df)

        # Extract current bar's regime (last row)
        self.current_regime = df_classified['regime_label'].iloc[-1]
        self.current_confidence = df_classified['regime_confidence'].iloc[-1]

        return self.current_regime, self.current_confidence

    def get_regime_history(self, n: int = 24) -> List[str]:
        """
        Get regime labels for last n bars in buffer.

        Args:
            n: Number of bars to return

        Returns:
            List of regime labels
        """
        if len(self.buffer) < self.window_size:
            return ['neutral'] * min(n, len(self.buffer))

        # Re-classify buffer
        df = pd.DataFrame(list(self.buffer))
        df_classified = self.hmm.classify_batch(df)

        return df_classified['regime_label'].iloc[-n:].tolist()


# Example usage and testing
if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python hmm_regime_model.py <model_path>")
        print("\nTest with:")
        print("  python hmm_regime_model.py models/hmm_regime_v2.pkl")
        sys.exit(1)

    model_path = sys.argv[1]

    try:
        # Load model
        hmm = HMMRegimeModel(model_path)
        print(f"\n✅ Loaded HMM model from {model_path}")
        print(f"   States: {list(hmm.state_map.values())}")
        print(f"   Features: {len(hmm.feature_order)}")

        # Test classification
        test_bar = {
            'funding_Z': -1.5,
            'OI_CHANGE': -10.0,
            'RV_21': 65.0,
            'LIQ_VOL_24h': 300.0,
            'USDT.D': 6.5,
            'BTC.D': 48.0,
            'TOTAL_RET_21d': -8.0,
            'ALT_ROTATION': -5.0,
            'VIX_Z': 1.2,
            'DXY_Z': 0.8,
            'YC_SPREAD': -50.0,
            'M2_GROWTH_YOY': 3.0,
            'FOMC_D0': 0.0,
            'CPI_D0': 0.0,
            'NFP_D0': 0.0
        }

        regime, confidence = hmm.classify_stream(test_bar)

        print("\n📊 Test Classification (crisis-like conditions):")
        print(f"   Regime: {regime}")
        print(f"   Confidence: {confidence:.1%}")

        print("\n✅ HMM model test passed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
