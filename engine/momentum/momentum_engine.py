"""
Momentum Engine - Guard Rails and Consistent Weights

Provides momentum fallback signals with proper bounds and price-normalized MACD.
Protects RSI from divide-by-zero and keeps momentum capped at ±0.05 total contribution.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

@dataclass
class MomentumSignal:
    """Momentum analysis signal"""
    timestamp: pd.Timestamp
    direction: str     # 'long', 'short', 'neutral'
    strength: float    # 0-1
    confidence: float  # 0-1

    # Component indicators
    rsi: float
    macd_normalized: float
    momentum_delta: float  # -0.05 to +0.05

    metadata: Dict[str, Any]

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate RSI with divide-by-zero protection.
    Returns 50.0 as neutral if calculation fails.
    """
    try:
        if len(df) < period + 1:
            return 50.0

        delta = df["close"].diff()
        up = delta.clip(lower=0).rolling(period).mean()
        down = (-delta.clip(upper=0)).rolling(period).mean()

        # Protect against divide-by-zero
        rs = up / down.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        result = rsi.iloc[-1]
        return float(np.nan_to_num(result, nan=50.0))

    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return 50.0

def calculate_macd_norm(df: pd.DataFrame, fast=12, slow=26, signal=9) -> float:
    """
    Calculate MACD normalized by price to keep cross-asset comparable.
    Prevents overweighting high-priced assets.
    """
    try:
        if len(df) < slow + signal:
            return 0.0

        c = df["close"]
        ema_f = c.ewm(span=fast, adjust=False).mean()
        ema_s = c.ewm(span=slow, adjust=False).mean()
        macd = ema_f - ema_s
        sig = macd.ewm(span=signal, adjust=False).mean()
        hist = macd - sig

        # Normalize by price to keep cross-asset comparable
        current_price = c.iloc[-1]
        normalized_hist = hist.iloc[-1] / max(1e-9, current_price)

        return float(normalized_hist)

    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        return 0.0

def momentum_delta(df: pd.DataFrame, cfg: Dict) -> float:
    """
    Calculate momentum delta with strict bounds.

    Returns delta in range [-0.05, +0.05] to prevent double-counting
    when used alongside main domain weights.
    """
    try:
        rsi = calculate_rsi(df, cfg.get("rsi_period", 14))
        macd_n = calculate_macd_norm(df)

        delta = 0.0

        # RSI signals
        rsi_overbought = cfg.get("rsi_overbought", 70)
        rsi_oversold = cfg.get("rsi_oversold", 30)

        if rsi > rsi_overbought:
            delta -= 0.025
        elif rsi < rsi_oversold:
            delta += 0.025

        # MACD signals
        if macd_n > 0:
            delta += 0.025
        elif macd_n < 0:
            delta -= 0.025

        # Strict bounds to prevent overflow
        return float(np.clip(delta, -0.05, 0.05))

    except Exception as e:
        logger.error(f"Error calculating momentum delta: {e}")
        return 0.0

class MomentumEngine:
    """
    Momentum Analysis Engine

    Provides momentum fallback signals with proper safeguards and normalization.
    Designed to supplement (not replace) primary domain signals.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.macd_fast = config.get('macd_fast', 12)
        self.macd_slow = config.get('macd_slow', 26)
        self.macd_signal = config.get('macd_signal', 9)

    def analyze(self, data: pd.DataFrame) -> Optional[MomentumSignal]:
        """
        Analyze momentum indicators for fallback signals.

        Args:
            data: OHLCV price data

        Returns:
            MomentumSignal with bounded delta, or None if insufficient data
        """
        try:
            if len(data) < max(self.macd_slow + self.macd_signal, self.rsi_period + 1):
                return None

            # Calculate indicators
            rsi = calculate_rsi(data, self.rsi_period)
            macd_norm = calculate_macd_norm(data, self.macd_fast, self.macd_slow, self.macd_signal)
            delta = momentum_delta(data, self.config)

            # Determine direction and strength
            if delta > 0.02:
                direction = 'long'
                strength = min(1.0, abs(delta) * 20)  # Scale to 0-1
                confidence = 0.6
            elif delta < -0.02:
                direction = 'short'
                strength = min(1.0, abs(delta) * 20)
                confidence = 0.6
            else:
                direction = 'neutral'
                strength = 0.0
                confidence = 0.3

            return MomentumSignal(
                timestamp=data.index[-1],
                direction=direction,
                strength=strength,
                confidence=confidence,
                rsi=rsi,
                macd_normalized=macd_norm,
                momentum_delta=delta,
                metadata={
                    'rsi_overbought_threshold': self.rsi_overbought,
                    'rsi_oversold_threshold': self.rsi_oversold,
                    'macd_parameters': {
                        'fast': self.macd_fast,
                        'slow': self.macd_slow,
                        'signal': self.macd_signal
                    }
                }
            )

        except Exception as e:
            logger.error(f"Error in momentum analysis: {e}")
            return None

    def get_delta_only(self, data: pd.DataFrame) -> float:
        """
        Get only the momentum delta for fusion integration.

        This is the recommended way to integrate momentum into the fusion engine
        to avoid double-counting with main domain weights.
        """
        return momentum_delta(data, self.config)