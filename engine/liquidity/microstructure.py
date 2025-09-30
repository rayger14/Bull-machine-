"""
Microstructure Analysis - Order Flow and Market Dynamics

Provides order flow analysis and microstructure pattern detection
for enhanced liquidity analysis.
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class OrderFlowType(Enum):
    """Order flow signal types"""
    BUYING_PRESSURE = "buying_pressure"
    SELLING_PRESSURE = "selling_pressure"
    ABSORPTION = "absorption"
    EXHAUSTION = "exhaustion"

@dataclass
class OrderFlowSignal:
    """Order flow signal data structure"""
    signal_type: OrderFlowType
    timestamp: pd.Timestamp
    strength: float  # 0-1
    confidence: float  # 0-1
    volume_profile: Dict[str, float]
    metadata: Dict[str, Any]

class MicrostructureAnalyzer:
    """
    Basic microstructure analysis for order flow patterns.

    Provides simplified order flow analysis using OHLCV data.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def analyze_order_flow(self, data: pd.DataFrame) -> List[OrderFlowSignal]:
        """
        Analyze order flow patterns from OHLCV data.

        Args:
            data: OHLCV DataFrame

        Returns:
            List of order flow signals
        """
        try:
            if len(data) < 10:
                return []

            signals = []

            # Simple buying/selling pressure analysis
            for i in range(5, len(data)):
                signal = self._analyze_bar_flow(data.iloc[i], data.iloc[i-5:i])
                if signal:
                    signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"Error in order flow analysis: {e}")
            return []

    def _analyze_bar_flow(self, current_bar: pd.Series, context: pd.DataFrame) -> Optional[OrderFlowSignal]:
        """Analyze single bar order flow"""
        try:
            # Calculate pressure indicators
            body_size = abs(current_bar['close'] - current_bar['open'])
            range_size = current_bar['high'] - current_bar['low']

            if range_size == 0:
                return None

            # Body position in range
            if current_bar['close'] > current_bar['open']:
                # Bullish bar
                upper_wick = current_bar['high'] - current_bar['close']
                lower_wick = current_bar['open'] - current_bar['low']
                body_position = (current_bar['close'] - current_bar['low']) / range_size
            else:
                # Bearish bar
                upper_wick = current_bar['high'] - current_bar['open']
                lower_wick = current_bar['close'] - current_bar['low']
                body_position = (current_bar['open'] - current_bar['low']) / range_size

            # Volume analysis
            avg_volume = context['volume'].mean() if len(context) > 0 else current_bar['volume']
            volume_ratio = current_bar['volume'] / avg_volume if avg_volume > 0 else 1.0

            # Determine signal type
            if body_position > 0.7 and volume_ratio > 1.2:
                signal_type = OrderFlowType.BUYING_PRESSURE
                strength = min(1.0, body_position * volume_ratio / 2)
            elif body_position < 0.3 and volume_ratio > 1.2:
                signal_type = OrderFlowType.SELLING_PRESSURE
                strength = min(1.0, (1 - body_position) * volume_ratio / 2)
            else:
                return None

            return OrderFlowSignal(
                signal_type=signal_type,
                timestamp=current_bar.name,
                strength=strength,
                confidence=min(0.8, strength),
                volume_profile={
                    'volume_ratio': volume_ratio,
                    'body_position': body_position
                },
                metadata={
                    'body_size': body_size,
                    'range_size': range_size,
                    'upper_wick': upper_wick,
                    'lower_wick': lower_wick
                }
            )

        except Exception as e:
            logger.error(f"Error analyzing bar flow: {e}")
            return None