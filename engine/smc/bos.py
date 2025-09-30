"""
Break of Structure (BOS) Detection

Identifies when price breaks previous market structure (higher highs/lower lows)
indicating potential trend changes or continuation patterns.
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class BOSType(Enum):
    """Break of structure types"""
    BULLISH = "bullish"    # Break above previous high
    BEARISH = "bearish"    # Break below previous low

class TrendState(Enum):
    """Market trend states"""
    UPTREND = "uptrend"      # Higher highs, higher lows
    DOWNTREND = "downtrend"  # Lower highs, lower lows
    SIDEWAYS = "sideways"    # No clear trend

@dataclass
class BreakOfStructure:
    """Break of structure data structure"""
    timestamp: pd.Timestamp
    bos_type: BOSType
    broken_level: float     # The high/low that was broken
    break_price: float      # Price that confirmed the break
    previous_trend: TrendState
    new_trend: TrendState
    strength: float         # 0-1 based on break distance and volume
    confidence: float       # 0-1 based on follow-through
    volume_ratio: float     # Volume during break vs average
    break_distance_pct: float  # How far beyond level
    follow_through: bool    # Did price continue in break direction
    metadata: Dict[str, Any]

@dataclass
class SwingPoint:
    """Swing high/low point"""
    timestamp: pd.Timestamp
    price: float
    point_type: str  # 'high' or 'low'
    strength: float  # Based on how clear the swing is

class BOSDetector:
    """
    Break of Structure Detection Engine

    Identifies breaks of previous market structure to determine
    trend changes and continuation patterns.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.swing_lookback = config.get('swing_lookback', 5)  # Bars to look back/forward
        self.min_break_pct = config.get('min_break_pct', 0.001)  # 0.1% minimum break
        self.min_volume_ratio = config.get('min_volume_ratio', 1.2)
        self.follow_through_bars = config.get('follow_through_bars', 3)

    def detect_bos(self, data: pd.DataFrame) -> List[BreakOfStructure]:
        """
        Detect breaks of structure in price data.

        Args:
            data: OHLCV DataFrame

        Returns:
            List of break of structure events
        """
        try:
            if len(data) < self.swing_lookback * 4:
                return []

            # First identify swing points
            swing_highs = self._find_swing_highs(data)
            swing_lows = self._find_swing_lows(data)

            # Then detect breaks of those levels
            bos_events = []

            # Check for breaks of swing highs (bullish BOS)
            for swing_high in swing_highs:
                bos = self._check_high_break(data, swing_high)
                if bos:
                    bos_events.append(bos)

            # Check for breaks of swing lows (bearish BOS)
            for swing_low in swing_lows:
                bos = self._check_low_break(data, swing_low)
                if bos:
                    bos_events.append(bos)

            # Sort by timestamp
            bos_events.sort(key=lambda x: x.timestamp)
            return bos_events

        except Exception as e:
            logger.error(f"Error detecting BOS: {e}")
            return []

    def _find_swing_highs(self, data: pd.DataFrame) -> List[SwingPoint]:
        """Find swing high points"""
        try:
            swing_highs = []

            for i in range(self.swing_lookback, len(data) - self.swing_lookback):
                current_high = data['high'].iloc[i]

                # Check if this is a swing high
                left_bars = data['high'].iloc[i - self.swing_lookback:i]
                right_bars = data['high'].iloc[i + 1:i + 1 + self.swing_lookback]

                is_swing_high = (
                    (left_bars < current_high).all() and
                    (right_bars < current_high).all()
                )

                if is_swing_high:
                    # Calculate strength based on how clear the swing is
                    max_left = left_bars.max() if len(left_bars) > 0 else current_high
                    max_right = right_bars.max() if len(right_bars) > 0 else current_high

                    strength = min(1.0, (current_high - max(max_left, max_right)) / current_high * 100)

                    swing_highs.append(SwingPoint(
                        timestamp=data.index[i],
                        price=current_high,
                        point_type='high',
                        strength=max(0.1, strength)  # Minimum strength
                    ))

            return swing_highs

        except Exception as e:
            logger.error(f"Error finding swing highs: {e}")
            return []

    def _find_swing_lows(self, data: pd.DataFrame) -> List[SwingPoint]:
        """Find swing low points"""
        try:
            swing_lows = []

            for i in range(self.swing_lookback, len(data) - self.swing_lookback):
                current_low = data['low'].iloc[i]

                # Check if this is a swing low
                left_bars = data['low'].iloc[i - self.swing_lookback:i]
                right_bars = data['low'].iloc[i + 1:i + 1 + self.swing_lookback]

                is_swing_low = (
                    (left_bars > current_low).all() and
                    (right_bars > current_low).all()
                )

                if is_swing_low:
                    # Calculate strength based on how clear the swing is
                    min_left = left_bars.min() if len(left_bars) > 0 else current_low
                    min_right = right_bars.min() if len(right_bars) > 0 else current_low

                    strength = min(1.0, (min(min_left, min_right) - current_low) / current_low * 100)

                    swing_lows.append(SwingPoint(
                        timestamp=data.index[i],
                        price=current_low,
                        point_type='low',
                        strength=max(0.1, strength)  # Minimum strength
                    ))

            return swing_lows

        except Exception as e:
            logger.error(f"Error finding swing lows: {e}")
            return []

    def _check_high_break(self, data: pd.DataFrame, swing_high: SwingPoint) -> Optional[BreakOfStructure]:
        """Check if swing high was broken (bullish BOS)"""
        try:
            # Find data after this swing point
            swing_index = data.index.get_loc(swing_high.timestamp)
            future_data = data.iloc[swing_index + 1:]

            if len(future_data) < self.follow_through_bars:
                return None

            # Look for price breaking above this swing high
            for i, (timestamp, bar) in enumerate(future_data.iterrows()):
                if bar['close'] > swing_high.price:
                    # Break confirmed by close above level
                    break_distance_pct = (bar['close'] - swing_high.price) / swing_high.price

                    if break_distance_pct >= self.min_break_pct:
                        # Check volume
                        lookback_volume = data['volume'].iloc[max(0, swing_index-10):swing_index].mean()
                        volume_ratio = bar['volume'] / lookback_volume if lookback_volume > 0 else 1

                        if volume_ratio >= self.min_volume_ratio:
                            # Check follow-through
                            follow_through = self._check_follow_through(
                                future_data.iloc[i:], BOSType.BULLISH, swing_high.price
                            )

                            # Determine trend states
                            previous_trend = self._analyze_trend_before(data, swing_index)
                            new_trend = TrendState.UPTREND if follow_through else TrendState.SIDEWAYS

                            strength = min(1.0, break_distance_pct / 0.02)  # Scale to 2% max
                            confidence = min(1.0, (volume_ratio + swing_high.strength) / 2)

                            return BreakOfStructure(
                                timestamp=timestamp,
                                bos_type=BOSType.BULLISH,
                                broken_level=swing_high.price,
                                break_price=bar['close'],
                                previous_trend=previous_trend,
                                new_trend=new_trend,
                                strength=strength,
                                confidence=confidence,
                                volume_ratio=volume_ratio,
                                break_distance_pct=break_distance_pct,
                                follow_through=follow_through,
                                metadata={
                                    'swing_timestamp': swing_high.timestamp,
                                    'swing_strength': swing_high.strength
                                }
                            )

            return None

        except Exception as e:
            logger.error(f"Error checking high break: {e}")
            return None

    def _check_low_break(self, data: pd.DataFrame, swing_low: SwingPoint) -> Optional[BreakOfStructure]:
        """Check if swing low was broken (bearish BOS)"""
        try:
            # Find data after this swing point
            swing_index = data.index.get_loc(swing_low.timestamp)
            future_data = data.iloc[swing_index + 1:]

            if len(future_data) < self.follow_through_bars:
                return None

            # Look for price breaking below this swing low
            for i, (timestamp, bar) in enumerate(future_data.iterrows()):
                if bar['close'] < swing_low.price:
                    # Break confirmed by close below level
                    break_distance_pct = (swing_low.price - bar['close']) / swing_low.price

                    if break_distance_pct >= self.min_break_pct:
                        # Check volume
                        lookback_volume = data['volume'].iloc[max(0, swing_index-10):swing_index].mean()
                        volume_ratio = bar['volume'] / lookback_volume if lookback_volume > 0 else 1

                        if volume_ratio >= self.min_volume_ratio:
                            # Check follow-through
                            follow_through = self._check_follow_through(
                                future_data.iloc[i:], BOSType.BEARISH, swing_low.price
                            )

                            # Determine trend states
                            previous_trend = self._analyze_trend_before(data, swing_index)
                            new_trend = TrendState.DOWNTREND if follow_through else TrendState.SIDEWAYS

                            strength = min(1.0, break_distance_pct / 0.02)  # Scale to 2% max
                            confidence = min(1.0, (volume_ratio + swing_low.strength) / 2)

                            return BreakOfStructure(
                                timestamp=timestamp,
                                bos_type=BOSType.BEARISH,
                                broken_level=swing_low.price,
                                break_price=bar['close'],
                                previous_trend=previous_trend,
                                new_trend=new_trend,
                                strength=strength,
                                confidence=confidence,
                                volume_ratio=volume_ratio,
                                break_distance_pct=break_distance_pct,
                                follow_through=follow_through,
                                metadata={
                                    'swing_timestamp': swing_low.timestamp,
                                    'swing_strength': swing_low.strength
                                }
                            )

            return None

        except Exception as e:
            logger.error(f"Error checking low break: {e}")
            return None

    def _check_follow_through(self, future_data: pd.DataFrame, bos_type: BOSType, break_level: float) -> bool:
        """Check if there was follow-through after the break"""
        try:
            if len(future_data) < self.follow_through_bars:
                return False

            follow_data = future_data.iloc[:self.follow_through_bars]

            if bos_type == BOSType.BULLISH:
                # For bullish BOS, expect price to stay above break level
                return (follow_data['low'] > break_level * 0.999).any()
            else:
                # For bearish BOS, expect price to stay below break level
                return (follow_data['high'] < break_level * 1.001).any()

        except Exception as e:
            logger.error(f"Error checking follow-through: {e}")
            return False

    def _analyze_trend_before(self, data: pd.DataFrame, index: int) -> TrendState:
        """Analyze trend state before the break"""
        try:
            lookback_data = data.iloc[max(0, index - 20):index]
            if len(lookback_data) < 10:
                return TrendState.SIDEWAYS

            # Simple trend analysis based on price direction
            start_price = lookback_data['close'].iloc[0]
            end_price = lookback_data['close'].iloc[-1]

            change_pct = (end_price - start_price) / start_price

            if change_pct > 0.02:  # 2% up
                return TrendState.UPTREND
            elif change_pct < -0.02:  # 2% down
                return TrendState.DOWNTREND
            else:
                return TrendState.SIDEWAYS

        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return TrendState.SIDEWAYS