"""
Wick Magnets Detection - Institutional Liquidity Targets

Detects wick formations that act as liquidity magnets for institutional
order flow and price target calculations.
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class MagnetStrength(Enum):
    """Wick magnet strength classification"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    INSTITUTIONAL = "institutional"

@dataclass
class WickMagnet:
    """Wick magnet data structure"""
    price: float
    strength: MagnetStrength
    timestamp: pd.Timestamp
    direction: str  # 'up' or 'down'
    wick_ratio: float
    volume_confirmation: float
    probability_reach: float
    timeframe: str
    metadata: Dict[str, Any]

class WickMagnetDetector:
    """
    Wick Magnet Detection System for institutional liquidity analysis.

    Identifies significant wick formations that often act as future price targets
    based on institutional order flow patterns.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.wick_config = config.get('wick_magnets', {})

        # Detection thresholds
        self.min_wick_ratio = self.wick_config.get('min_wick_ratio', 1.5)
        self.min_volume_ratio = self.wick_config.get('min_volume_ratio', 1.2)
        self.institutional_wick_ratio = self.wick_config.get('institutional_ratio', 3.0)

        # Magnet validation
        self.magnet_decay_hours = self.wick_config.get('magnet_decay_hours', 168)  # 7 days
        self.min_distance_pct = self.wick_config.get('min_distance_pct', 0.005)  # 0.5%

        self.active_magnets = []

    def detect_wick_magnets(self, data: Dict[str, pd.DataFrame]) -> List[WickMagnet]:
        """
        Detect wick magnets across multiple timeframes.

        Args:
            data: Multi-timeframe OHLCV data

        Returns:
            List of active wick magnets
        """
        try:
            all_magnets = []

            # Analyze each timeframe
            for timeframe, df in data.items():
                if df is None or len(df) < 20:
                    continue

                tf_magnets = self._detect_timeframe_magnets(df, timeframe)
                all_magnets.extend(tf_magnets)

            # Filter and rank magnets
            filtered_magnets = self._filter_magnets(all_magnets)
            self.active_magnets = filtered_magnets

            return filtered_magnets

        except Exception as e:
            logger.error(f"Error detecting wick magnets: {e}")
            return []

    def _detect_timeframe_magnets(self, df: pd.DataFrame, timeframe: str) -> List[WickMagnet]:
        """Detect wick magnets in specific timeframe"""
        try:
            magnets = []
            lookback = min(100, len(df))
            recent_data = df.tail(lookback)

            for i in range(2, len(recent_data) - 1):
                current_bar = recent_data.iloc[i]
                timestamp = recent_data.index[i]

                # Analyze upper wicks
                upper_magnet = self._analyze_upper_wick(current_bar, recent_data, i, timestamp, timeframe)
                if upper_magnet:
                    magnets.append(upper_magnet)

                # Analyze lower wicks
                lower_magnet = self._analyze_lower_wick(current_bar, recent_data, i, timestamp, timeframe)
                if lower_magnet:
                    magnets.append(lower_magnet)

            return magnets

        except Exception as e:
            logger.error(f"Error detecting {timeframe} magnets: {e}")
            return []

    def _analyze_upper_wick(self, bar: pd.Series, data: pd.DataFrame, index: int,
                           timestamp: pd.Timestamp, timeframe: str) -> Optional[WickMagnet]:
        """Analyze upper wick for magnet formation"""
        try:
            # Calculate wick metrics
            body_top = max(bar['open'], bar['close'])
            body_bottom = min(bar['open'], bar['close'])
            body_size = body_top - body_bottom
            upper_wick = bar['high'] - body_top

            if body_size == 0 or upper_wick <= 0:
                return None

            wick_ratio = upper_wick / body_size

            # Check minimum wick ratio
            if wick_ratio < self.min_wick_ratio:
                return None

            # Volume confirmation
            volume_confirmation = self._check_volume_confirmation(data, index)

            # Calculate magnet strength
            strength = self._calculate_magnet_strength(wick_ratio, volume_confirmation)

            # Calculate probability of price reaching the wick high
            probability_reach = self._calculate_reach_probability(
                bar['high'], data, index, 'up', timeframe
            )

            return WickMagnet(
                price=bar['high'],
                strength=strength,
                timestamp=timestamp,
                direction='up',
                wick_ratio=wick_ratio,
                volume_confirmation=volume_confirmation,
                probability_reach=probability_reach,
                timeframe=timeframe,
                metadata={
                    'body_size': body_size,
                    'upper_wick': upper_wick,
                    'volume': bar['volume'],
                    'rejection_type': self._classify_rejection_type(bar, 'upper')
                }
            )

        except Exception as e:
            logger.error(f"Error analyzing upper wick: {e}")
            return None

    def _analyze_lower_wick(self, bar: pd.Series, data: pd.DataFrame, index: int,
                           timestamp: pd.Timestamp, timeframe: str) -> Optional[WickMagnet]:
        """Analyze lower wick for magnet formation"""
        try:
            # Calculate wick metrics
            body_top = max(bar['open'], bar['close'])
            body_bottom = min(bar['open'], bar['close'])
            body_size = body_top - body_bottom
            lower_wick = body_bottom - bar['low']

            if body_size == 0 or lower_wick <= 0:
                return None

            wick_ratio = lower_wick / body_size

            # Check minimum wick ratio
            if wick_ratio < self.min_wick_ratio:
                return None

            # Volume confirmation
            volume_confirmation = self._check_volume_confirmation(data, index)

            # Calculate magnet strength
            strength = self._calculate_magnet_strength(wick_ratio, volume_confirmation)

            # Calculate probability of price reaching the wick low
            probability_reach = self._calculate_reach_probability(
                bar['low'], data, index, 'down', timeframe
            )

            return WickMagnet(
                price=bar['low'],
                strength=strength,
                timestamp=timestamp,
                direction='down',
                wick_ratio=wick_ratio,
                volume_confirmation=volume_confirmation,
                probability_reach=probability_reach,
                timeframe=timeframe,
                metadata={
                    'body_size': body_size,
                    'lower_wick': lower_wick,
                    'volume': bar['volume'],
                    'rejection_type': self._classify_rejection_type(bar, 'lower')
                }
            )

        except Exception as e:
            logger.error(f"Error analyzing lower wick: {e}")
            return None

    def _check_volume_confirmation(self, data: pd.DataFrame, index: int) -> float:
        """Check volume confirmation for wick formation"""
        try:
            current_volume = data.iloc[index]['volume']

            # Compare to recent average
            lookback = min(20, index)
            if lookback < 5:
                return 0.5  # Default for insufficient data

            avg_volume = data.iloc[index-lookback:index]['volume'].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

            # Normalize volume confirmation
            if volume_ratio >= self.min_volume_ratio:
                return min(1.0, volume_ratio / 3.0)  # Cap at 3x volume
            else:
                return volume_ratio / self.min_volume_ratio * 0.5

        except Exception:
            return 0.5

    def _calculate_magnet_strength(self, wick_ratio: float, volume_confirmation: float) -> MagnetStrength:
        """Calculate overall magnet strength"""
        try:
            # Combine wick ratio and volume
            combined_score = (wick_ratio * 0.7 + volume_confirmation * 3.0 * 0.3)

            if combined_score >= 4.0:
                return MagnetStrength.INSTITUTIONAL
            elif combined_score >= 2.5:
                return MagnetStrength.STRONG
            elif combined_score >= 1.8:
                return MagnetStrength.MODERATE
            else:
                return MagnetStrength.WEAK

        except Exception:
            return MagnetStrength.WEAK

    def _calculate_reach_probability(self, target_price: float, data: pd.DataFrame,
                                   index: int, direction: str, timeframe: str) -> float:
        """Calculate probability of price reaching the target"""
        try:
            # Historical analysis of similar setups
            current_price = data.iloc[index]['close']
            distance_pct = abs(target_price - current_price) / current_price

            # Base probability decreases with distance
            base_prob = max(0.1, 0.8 - (distance_pct / 0.05))  # 5% distance = 0% base prob

            # Timeframe adjustments
            tf_multipliers = {
                '1H': 1.0,
                '4H': 0.9,
                '1D': 0.8,
                '1W': 0.6
            }
            tf_mult = tf_multipliers.get(timeframe, 1.0)

            # Direction bias (slight preference for upward moves in crypto)
            direction_mult = 1.1 if direction == 'up' else 0.95

            final_probability = base_prob * tf_mult * direction_mult
            return min(0.95, max(0.05, final_probability))

        except Exception:
            return 0.5

    def _classify_rejection_type(self, bar: pd.Series, wick_type: str) -> str:
        """Classify the type of rejection"""
        try:
            body_size = abs(bar['close'] - bar['open'])

            if wick_type == 'upper':
                wick_size = bar['high'] - max(bar['open'], bar['close'])
            else:
                wick_size = min(bar['open'], bar['close']) - bar['low']

            # Classify based on body vs wick characteristics
            if body_size < wick_size * 0.3:
                return 'doji_rejection'
            elif bar['close'] > bar['open'] and wick_type == 'upper':
                return 'bullish_rejection'
            elif bar['close'] < bar['open'] and wick_type == 'lower':
                return 'bearish_rejection'
            else:
                return 'mixed_rejection'

        except Exception:
            return 'unknown'

    def _filter_magnets(self, magnets: List[WickMagnet]) -> List[WickMagnet]:
        """Filter and deduplicate wick magnets"""
        try:
            if not magnets:
                return []

            # Remove expired magnets
            current_time = pd.Timestamp.now()
            active_magnets = [
                m for m in magnets
                if (current_time - m.timestamp).total_seconds() / 3600 <= self.magnet_decay_hours
            ]

            # Remove overlapping magnets (same price level)
            filtered_magnets = []
            used_levels = set()

            # Sort by strength and probability
            sorted_magnets = sorted(
                active_magnets,
                key=lambda x: (x.strength.value, x.probability_reach),
                reverse=True
            )

            for magnet in sorted_magnets:
                # Check if this price level is already represented
                level_key = f"{magnet.price:.6f}_{magnet.direction}"

                # Check for nearby levels
                too_close = False
                for existing_level in used_levels:
                    existing_price = float(existing_level.split('_')[0])
                    if abs(magnet.price - existing_price) / magnet.price < self.min_distance_pct:
                        too_close = True
                        break

                if not too_close:
                    filtered_magnets.append(magnet)
                    used_levels.add(level_key)

            return filtered_magnets

        except Exception as e:
            logger.error(f"Error filtering magnets: {e}")
            return magnets

    def get_nearest_magnets(self, current_price: float, direction: Optional[str] = None,
                           max_distance_pct: float = 0.05) -> List[WickMagnet]:
        """Get nearest wick magnets within specified distance"""
        try:
            nearby_magnets = []

            for magnet in self.active_magnets:
                distance_pct = abs(magnet.price - current_price) / current_price

                if distance_pct <= max_distance_pct:
                    # Filter by direction if specified
                    if direction is None or magnet.direction == direction:
                        nearby_magnets.append(magnet)

            # Sort by distance
            nearby_magnets.sort(key=lambda x: abs(x.price - current_price))
            return nearby_magnets

        except Exception as e:
            logger.error(f"Error getting nearest magnets: {e}")
            return []

    def get_magnet_targets(self, entry_price: float, direction: str,
                          max_targets: int = 3) -> List[Dict[str, Any]]:
        """Get wick magnet targets for position management"""
        try:
            targets = []

            # Get relevant magnets in the trade direction
            relevant_magnets = [
                m for m in self.active_magnets
                if m.direction == direction and
                ((direction == 'up' and m.price > entry_price) or
                 (direction == 'down' and m.price < entry_price))
            ]

            # Sort by distance and strength
            relevant_magnets.sort(
                key=lambda x: (abs(x.price - entry_price), -x.probability_reach)
            )

            for i, magnet in enumerate(relevant_magnets[:max_targets]):
                distance_pct = abs(magnet.price - entry_price) / entry_price

                targets.append({
                    'price': magnet.price,
                    'distance_pct': distance_pct,
                    'strength': magnet.strength.value,
                    'probability': magnet.probability_reach,
                    'timeframe': magnet.timeframe,
                    'target_number': i + 1
                })

            return targets

        except Exception as e:
            logger.error(f"Error getting magnet targets: {e}")
            return []