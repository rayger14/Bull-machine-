"""
HOB/pHOB Detection and Quality Assessment

Implements Hands-on-Back (HOB) and potential HOB (pHOB) detection
with quality scoring and institutional validation.
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

def volume_zscore(df: pd.DataFrame, bars: int) -> pd.Series:
    """
    Robust TF-aware volume z-score calculation.
    Guards against zero std, missing volume, and TF drift.
    """
    if "volume" not in df.columns:
        return pd.Series(0.0, index=df.index)  # gracefully degrade

    v = df["volume"].astype(float)
    mu = v.rolling(bars, min_periods=max(10, bars//3)).mean()
    sd = v.rolling(bars, min_periods=max(10, bars//3)).std().replace(0, np.nan)
    z = (v - mu) / sd

    # winsorize to avoid single-bar explosions
    return z.clip(-3.0, 5.0).fillna(0.0)

class HOBType(Enum):
    """HOB pattern types"""
    BULLISH_HOB = "bullish_hob"
    BEARISH_HOB = "bearish_hob"
    POTENTIAL_HOB = "potential_hob"
    FAILED_HOB = "failed_hob"

class HOBQuality(Enum):
    """HOB quality assessment"""
    INSTITUTIONAL = "institutional"  # High-quality institutional setup
    RETAIL = "retail"               # Lower-quality retail setup
    INVALID = "invalid"             # Failed validation

@dataclass
class LiquidityLevel:
    """Liquidity level identification"""
    price: float
    strength: float
    volume: float
    timestamp: pd.Timestamp
    level_type: str  # 'support', 'resistance', 'pivot'
    touches: int
    age_hours: float

@dataclass
class HOBSignal:
    """HOB detection signal"""
    hob_type: HOBType
    quality: HOBQuality
    timestamp: pd.Timestamp
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: Optional[float]
    confidence: float
    strength: float
    liquidity_levels: List[LiquidityLevel]
    metadata: Dict[str, Any]

class HOBDetector:
    """
    Advanced HOB/pHOB detection with institutional validation.

    Detects Hands-on-Back patterns using:
    - Demand/supply zone validation
    - Volume profile analysis
    - Institutional vs retail classification
    - Multi-timeframe confirmation
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hob_config = config.get('hob_detection', {})

        # Detection parameters
        self.min_reaction_pips = self.hob_config.get('min_reaction_pips', 50)
        self.max_consolidation_bars = self.hob_config.get('max_consolidation_bars', 24)
        self.volume_threshold = self.hob_config.get('volume_threshold', 1.5)
        self.institutional_threshold = self.hob_config.get('institutional_threshold', 0.7)

        # Quality assessment
        self.quality_weights = self.hob_config.get('quality_weights', {
            'volume_surge': 0.3,
            'level_strength': 0.25,
            'reaction_speed': 0.2,
            'wick_presence': 0.15,
            'confluence': 0.1
        })

        # State tracking
        self.active_levels = {}
        self.hob_history = []

    def detect_hob(self, data: pd.DataFrame) -> Optional[HOBSignal]:
        """Simple wrapper for single timeframe analysis"""
        try:
            timeframe = '4H'  # Default timeframe
            results = self.detect_hob_patterns({timeframe: data}, timeframe)
            return results[0] if results else None
        except:
            return None

    def detect_hob_patterns(self, data: Dict[str, pd.DataFrame], current_timeframe: str = '1H') -> List[HOBSignal]:
        """
        Detect HOB patterns across timeframes.

        Args:
            data: Multi-timeframe OHLCV data
            current_timeframe: Primary timeframe for detection

        Returns:
            List of HOB signals with quality assessment
        """
        try:
            if current_timeframe not in data:
                logger.warning(f"No data for timeframe {current_timeframe}")
                return []

            df = data[current_timeframe]
            if len(df) < 50:  # Need sufficient data
                return []

            signals = []

            # 1. Identify potential demand/supply zones
            liquidity_levels = self._identify_liquidity_levels(df)

            # 2. Scan for HOB patterns
            for level in liquidity_levels:
                hob_signal = self._analyze_hob_at_level(df, level, data)
                if hob_signal:
                    signals.append(hob_signal)

            # 3. Filter and rank by quality
            signals = self._filter_hob_signals(signals)
            signals = sorted(signals, key=lambda x: (x.quality.value, x.confidence), reverse=True)

            return signals

        except Exception as e:
            logger.error(f"Error in HOB detection: {e}")
            return []

    def _identify_liquidity_levels(self, df: pd.DataFrame) -> List[LiquidityLevel]:
        """Identify significant liquidity levels"""
        try:
            levels = []
            current_time = df.index[-1]

            # Find swing highs and lows
            swing_highs = self._find_swing_points(df['high'], window=5, find_peaks=True)
            swing_lows = self._find_swing_points(df['low'], window=5, find_peaks=False)

            # Process swing highs as resistance levels
            for idx, price in swing_highs.items():
                if pd.isna(price):
                    continue

                strength = self._calculate_level_strength(df, price, 'resistance')
                volume = df.loc[idx, 'volume'] if idx in df.index else 0
                touches = self._count_level_touches(df, price, tolerance=0.001)
                age_hours = (current_time - idx).total_seconds() / 3600

                if strength > 0.3:  # Minimum strength threshold
                    levels.append(LiquidityLevel(
                        price=price,
                        strength=strength,
                        volume=volume,
                        timestamp=idx,
                        level_type='resistance',
                        touches=touches,
                        age_hours=age_hours
                    ))

            # Process swing lows as support levels
            for idx, price in swing_lows.items():
                if pd.isna(price):
                    continue

                strength = self._calculate_level_strength(df, price, 'support')
                volume = df.loc[idx, 'volume'] if idx in df.index else 0
                touches = self._count_level_touches(df, price, tolerance=0.001)
                age_hours = (current_time - idx).total_seconds() / 3600

                if strength > 0.3:
                    levels.append(LiquidityLevel(
                        price=price,
                        strength=strength,
                        volume=volume,
                        timestamp=idx,
                        level_type='support',
                        touches=touches,
                        age_hours=age_hours
                    ))

            return levels

        except Exception as e:
            logger.error(f"Error identifying liquidity levels: {e}")
            return []

    def _analyze_hob_at_level(self, df: pd.DataFrame, level: LiquidityLevel,
                             multi_tf_data: Dict[str, pd.DataFrame]) -> Optional[HOBSignal]:
        """Analyze potential HOB pattern at liquidity level"""
        try:
            current_time = df.index[-1]
            current_price = df['close'].iloc[-1]

            # Check if price is near the level
            distance_pct = abs(current_price - level.price) / level.price
            if distance_pct > 0.02:  # 2% tolerance
                return None

            # Determine HOB type based on level and price action
            if level.level_type == 'support' and current_price <= level.price * 1.01:
                hob_type = HOBType.BULLISH_HOB
                direction = 'long'
            elif level.level_type == 'resistance' and current_price >= level.price * 0.99:
                hob_type = HOBType.BEARISH_HOB
                direction = 'short'
            else:
                return None

            # Analyze recent price action for HOB characteristics
            recent_bars = 20
            recent_data = df.tail(recent_bars)

            # 1. Check for consolidation before the level
            consolidation_score = self._analyze_consolidation(recent_data, level.price)
            if consolidation_score < 0.3:
                return None

            # 2. Check for volume characteristics
            volume_score = self._analyze_volume_profile(recent_data, level)

            # 3. Check for reaction speed and strength
            reaction_score = self._analyze_reaction_strength(recent_data, level, direction)

            # 4. Check for wick formation
            wick_score = self._analyze_wick_formation(recent_data, level, direction)

            # 5. Multi-timeframe confluence
            confluence_score = self._analyze_mtf_confluence(multi_tf_data, level, direction)

            # Calculate base quality score (no volume boost here)
            quality_score = (
                consolidation_score * self.quality_weights.get('level_strength', 0.25) +
                volume_score * self.quality_weights.get('volume_surge', 0.3) +
                reaction_score * self.quality_weights.get('reaction_speed', 0.2) +
                wick_score * self.quality_weights.get('wick_presence', 0.15) +
                confluence_score * self.quality_weights.get('confluence', 0.1)
            )

            # Determine quality classification
            if quality_score >= self.institutional_threshold:
                quality = HOBQuality.INSTITUTIONAL
            elif quality_score >= 0.5:
                quality = HOBQuality.RETAIL
            else:
                quality = HOBQuality.INVALID

            # Only proceed with valid quality
            if quality == HOBQuality.INVALID:
                return None

            # Calculate entry, stop, and targets
            entry_price, stop_loss, target_1, target_2 = self._calculate_hob_levels(
                current_price, level, direction, recent_data
            )

            # Calculate confidence based on multiple factors
            confidence = min(0.95, quality_score * 1.2)
            strength = level.strength * quality_score

            return HOBSignal(
                hob_type=hob_type,
                quality=quality,
                timestamp=current_time,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                confidence=confidence,
                strength=strength,
                liquidity_levels=[level],
                metadata={
                    'consolidation_score': consolidation_score,
                    'volume_score': volume_score,
                    'reaction_score': reaction_score,
                    'wick_score': wick_score,
                    'confluence_score': confluence_score,
                    'quality_score': quality_score,
                    'level_touches': level.touches,
                    'level_age_hours': level.age_hours
                }
            )

        except Exception as e:
            logger.error(f"Error analyzing HOB at level: {e}")
            return None

    def _find_swing_points(self, series: pd.Series, window: int = 5, find_peaks: bool = True) -> pd.Series:
        """Find swing highs/lows using rolling windows"""
        try:
            if find_peaks:
                # Find peaks (swing highs)
                condition = (series == series.rolling(window=window, center=True).max())
            else:
                # Find troughs (swing lows)
                condition = (series == series.rolling(window=window, center=True).min())

            swings = series.where(condition)
            return swings.dropna()

        except Exception as e:
            logger.error(f"Error finding swing points: {e}")
            return pd.Series()

    def _calculate_level_strength(self, df: pd.DataFrame, price: float, level_type: str) -> float:
        """Calculate strength of a liquidity level"""
        try:
            tolerance = 0.001  # 0.1%

            # Count touches within tolerance
            if level_type == 'support':
                touches = ((df['low'] <= price * (1 + tolerance)) &
                          (df['low'] >= price * (1 - tolerance))).sum()
            else:  # resistance
                touches = ((df['high'] >= price * (1 - tolerance)) &
                          (df['high'] <= price * (1 + tolerance))).sum()

            # Calculate volume at level
            level_mask = ((df['high'] >= price * (1 - tolerance)) &
                         (df['low'] <= price * (1 + tolerance)))
            avg_volume = df.loc[level_mask, 'volume'].mean() if level_mask.any() else 0
            overall_avg_volume = df['volume'].mean()

            # Strength components
            touch_strength = min(1.0, touches / 5.0)  # Normalize to 5 touches max
            volume_strength = min(1.0, avg_volume / overall_avg_volume) if overall_avg_volume > 0 else 0

            return (touch_strength * 0.6 + volume_strength * 0.4)

        except Exception as e:
            logger.error(f"Error calculating level strength: {e}")
            return 0.0

    def _count_level_touches(self, df: pd.DataFrame, price: float, tolerance: float = 0.001) -> int:
        """Count how many times price touched a level"""
        try:
            touches = ((df['high'] >= price * (1 - tolerance)) &
                      (df['low'] <= price * (1 + tolerance))).sum()
            return int(touches)
        except Exception:
            return 0

    def _analyze_consolidation(self, recent_data: pd.DataFrame, level_price: float) -> float:
        """Analyze consolidation pattern before HOB"""
        try:
            # Calculate range tightness
            high_range = recent_data['high'].max()
            low_range = recent_data['low'].min()
            range_pct = (high_range - low_range) / level_price

            # Tighter ranges indicate better consolidation
            consolidation_score = max(0.0, 1.0 - (range_pct / 0.05))  # 5% max range

            return min(1.0, consolidation_score)

        except Exception as e:
            logger.error(f"Error analyzing consolidation: {e}")
            return 0.0

    def _analyze_volume_profile(self, recent_data: pd.DataFrame, level: LiquidityLevel) -> float:
        """Analyze volume characteristics for HOB validation"""
        try:
            recent_volume = recent_data['volume'].tail(5).mean()
            avg_volume = recent_data['volume'].mean()

            # Look for volume surge at the level
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0

            # Score based on volume surge
            if volume_ratio >= self.volume_threshold:
                return min(1.0, volume_ratio / 3.0)  # Cap at 3x volume
            else:
                return volume_ratio / self.volume_threshold * 0.5

        except Exception as e:
            logger.error(f"Error analyzing volume profile: {e}")
            return 0.0

    def _analyze_reaction_strength(self, recent_data: pd.DataFrame, level: LiquidityLevel, direction: str) -> float:
        """Analyze strength and speed of reaction from level"""
        try:
            if len(recent_data) < 5:
                return 0.0

            entry_price = level.price
            current_price = recent_data['close'].iloc[-1]

            if direction == 'long':
                reaction_pips = (current_price - entry_price) / entry_price * 10000
            else:  # short
                reaction_pips = (entry_price - current_price) / entry_price * 10000

            # Score based on reaction strength
            if reaction_pips >= self.min_reaction_pips:
                return min(1.0, reaction_pips / (self.min_reaction_pips * 2))
            else:
                return max(0.0, reaction_pips / self.min_reaction_pips)

        except Exception as e:
            logger.error(f"Error analyzing reaction strength: {e}")
            return 0.0

    def calculate_hob_volume_delta(self, data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        Calculate HOB volume delta for fusion engine.
        Only applies when price touches/engages unmitigated HOB context.

        Returns:
            Dict with delta value and telemetry metadata
        """
        try:
            # Check if any HOB is relevant (unmitigated and within ATR distance)
            hob_relevance_config = self.hob_config.get('hob_relevance', {})
            max_atr_dist = hob_relevance_config.get('max_atr_dist', 0.25)
            max_bars_unmitigated = hob_relevance_config.get('max_bars_unmitigated', 200)

            # Calculate ATR for distance check
            if len(data) < 14:
                return {"delta": 0.0, "reason": "insufficient_data", "hob_relevant": False}

            atr = self._calculate_atr(data, 14)
            max_distance = current_price * max_atr_dist

            # Check for recent unmitigated levels near current price
            hob_relevant = False
            nearest_level = None

            # Get current active levels (simplified - would need full implementation)
            active_levels = self._get_active_levels(data, max_bars_unmitigated)

            for level in active_levels:
                distance = abs(current_price - level['price'])
                if distance <= max_distance:
                    hob_relevant = True
                    nearest_level = level
                    break

            if not hob_relevant:
                return {"delta": 0.0, "reason": "no_relevant_hob", "hob_relevant": False}

            # Check volume z-score on current bar
            vol_bars = self.hob_config.get('hob_quality_factors', {}).get('volume_window_bars', 84)
            vol_z_min = self.hob_config.get('hob_quality_factors', {}).get('volume_z_min', 1.5)
            z = volume_zscore(data, vol_bars).iloc[-1] if len(data) > vol_bars//3 else 0.0

            if z >= vol_z_min:
                hob_volume_boost = self.hob_config.get('hob_volume_conf_boost', 0.05)
                delta = min(0.05, hob_volume_boost)  # Cap at 0.05

                return {
                    "delta": delta,
                    "reason": "hob_volume_boost",
                    "hob_relevant": True,
                    "volume_z": z,
                    "level_price": nearest_level['price'] if nearest_level else None,
                    "distance_atr": distance / atr if nearest_level else None
                }
            else:
                return {
                    "delta": 0.0,
                    "reason": "volume_too_low",
                    "hob_relevant": True,
                    "volume_z": z,
                    "volume_z_required": vol_z_min
                }

        except Exception as e:
            logger.error(f"Error calculating HOB volume delta: {e}")
            return {"delta": 0.0, "reason": "error", "hob_relevant": False}

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high = data['high']
            low = data['low']
            close = data['close'].shift(1)

            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.rolling(period).mean().iloc[-1]
        except:
            return data['close'].iloc[-1] * 0.02  # 2% fallback

    def _get_active_levels(self, data: pd.DataFrame, max_bars: int) -> List[Dict]:
        """Get active HOB levels (simplified implementation)"""
        # This would normally track all active HOB levels
        # For now, return empty list - full implementation would maintain level state
        return []

    def _analyze_wick_formation(self, recent_data: pd.DataFrame, level: LiquidityLevel, direction: str) -> float:
        """Analyze wick formation at the level with volume confirmation"""
        try:
            latest_bar = recent_data.iloc[-1]

            if direction == 'long':
                # Look for long lower wick at support
                body_size = abs(latest_bar['close'] - latest_bar['open'])
                lower_wick = latest_bar['open'] - latest_bar['low'] if latest_bar['close'] > latest_bar['open'] else latest_bar['close'] - latest_bar['low']
                wick_ratio = lower_wick / body_size if body_size > 0 else 0
            else:  # short
                # Look for long upper wick at resistance
                body_size = abs(latest_bar['close'] - latest_bar['open'])
                upper_wick = latest_bar['high'] - latest_bar['open'] if latest_bar['close'] < latest_bar['open'] else latest_bar['high'] - latest_bar['close']
                wick_ratio = upper_wick / body_size if body_size > 0 else 0

            # Base wick score
            base_wick_score = min(1.0, wick_ratio / 2.0)  # Ratio of 2:1 gets max score

            # Volume confirmation - reuse same volume_zscore definition for consistency
            vol_bars = self.hob_config.get('hob_quality_factors', {}).get('volume_window_bars', 84)
            z = volume_zscore(recent_data, vol_bars).iloc[-1] if len(recent_data) > vol_bars//3 else 0.0

            # Boost wick score if accompanied by volume spike
            volume_multiplier = 1.0 + (0.3 * min(1.0, max(0.0, z / 2.0)))  # Up to 30% boost for z>=2

            return min(1.0, base_wick_score * volume_multiplier)

        except Exception as e:
            logger.error(f"Error analyzing wick formation: {e}")
            return 0.0

    def _analyze_mtf_confluence(self, multi_tf_data: Dict[str, pd.DataFrame],
                               level: LiquidityLevel, direction: str) -> float:
        """Analyze multi-timeframe confluence"""
        try:
            confluence_score = 0.0
            timeframes = ['4H', '1D']  # Higher timeframes for confluence

            for tf in timeframes:
                if tf not in multi_tf_data:
                    continue

                tf_data = multi_tf_data[tf]
                if len(tf_data) < 20:
                    continue

                # Check if level aligns with higher timeframe structure
                tf_levels = self._identify_liquidity_levels(tf_data)

                for tf_level in tf_levels:
                    distance = abs(tf_level.price - level.price) / level.price
                    if distance < 0.005:  # 0.5% tolerance
                        confluence_score += 0.5
                        break

            return min(1.0, confluence_score)

        except Exception as e:
            logger.error(f"Error analyzing MTF confluence: {e}")
            return 0.0

    def _calculate_hob_levels(self, current_price: float, level: LiquidityLevel,
                             direction: str, recent_data: pd.DataFrame) -> Tuple[float, float, float, Optional[float]]:
        """Calculate entry, stop loss, and target levels for HOB"""
        try:
            atr = self._calculate_atr(recent_data, 14)

            if direction == 'long':
                entry_price = level.price * 1.001  # Slightly above support
                stop_loss = level.price - (atr * 1.5)
                target_1 = entry_price + (atr * 2.0)
                target_2 = entry_price + (atr * 3.5)
            else:  # short
                entry_price = level.price * 0.999  # Slightly below resistance
                stop_loss = level.price + (atr * 1.5)
                target_1 = entry_price - (atr * 2.0)
                target_2 = entry_price - (atr * 3.5)

            return entry_price, stop_loss, target_1, target_2

        except Exception as e:
            logger.error(f"Error calculating HOB levels: {e}")
            return current_price, current_price, current_price, None

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high_low = data['high'] - data['low']
            high_close = (data['high'] - data['close'].shift(1)).abs()
            low_close = (data['low'] - data['close'].shift(1)).abs()

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(period).mean().iloc[-1]

            return atr if not pd.isna(atr) else data['high'].iloc[-1] - data['low'].iloc[-1]

        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return 0.001

    def _filter_hob_signals(self, signals: List[HOBSignal]) -> List[HOBSignal]:
        """Filter HOB signals based on quality and confluence"""
        try:
            # Remove invalid quality signals
            valid_signals = [s for s in signals if s.quality != HOBQuality.INVALID]

            # Remove overlapping signals (same level, different quality)
            filtered_signals = []
            used_levels = set()

            for signal in sorted(valid_signals, key=lambda x: x.confidence, reverse=True):
                level_key = f"{signal.liquidity_levels[0].price:.6f}"
                if level_key not in used_levels:
                    filtered_signals.append(signal)
                    used_levels.add(level_key)

            return filtered_signals

        except Exception as e:
            logger.error(f"Error filtering HOB signals: {e}")
            return signals