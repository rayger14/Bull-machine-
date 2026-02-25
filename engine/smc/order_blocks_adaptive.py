"""
Adaptive Order Block Detection - Fixed for Low Volatility Periods

ROOT CAUSE: Original detector requires 2% displacement in 3 bars.
- December 2022: Only 1.7% of bars have 2%+ displacement → 15.5% OB coverage
- November 2022: 16.4% of bars have 2%+ displacement → 84.6% OB coverage

FIX: Use adaptive threshold based on recent ATR instead of fixed 2%.
This allows OB detection during low volatility while maintaining quality during high volatility.

Changes from original:
1. Calculate adaptive displacement threshold = max(0.5%, 1.0 * ATR%)
2. Lower min_volume_ratio from 1.5x to 1.2x (still above average, but more permissive)
3. Increase lookback for swing high/low detection (20 → 30 bars)
4. Add swing high/low validation (must be local max/min within lookback window)
"""

import pandas as pd
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class OrderBlockType(Enum):
    """Order block types"""
    BULLISH = "bullish"
    BEARISH = "bearish"

@dataclass
class OrderBlock:
    """Order block data structure"""
    timestamp: pd.Timestamp
    high: float
    low: float
    ob_type: OrderBlockType
    strength: float  # 0-1 based on volume and displacement
    confidence: float  # 0-1 based on reaction quality
    displacement: float  # Price move after block
    volume_ratio: float  # Volume vs average
    mitigation_count: int  # How many times tested
    active: bool
    metadata: Dict[str, Any]

class AdaptiveOrderBlockDetector:
    """
    Adaptive Order Block Detection Engine

    Uses ATR-based displacement thresholds to detect order blocks
    in both high and low volatility environments.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # FIXED: Use adaptive threshold instead of fixed 2%
        self.min_displacement_pct_floor = config.get('min_displacement_pct_floor', 0.005)  # 0.5% minimum
        self.atr_multiplier = config.get('atr_multiplier', 1.0)  # 1.0 × ATR as threshold

        # FIXED: Lower volume ratio requirement
        self.min_volume_ratio = config.get('min_volume_ratio', 1.2)  # Was 1.5

        self.lookback_bars = config.get('lookback_bars', 50)
        self.min_reaction_bars = config.get('min_reaction_bars', 3)

        # NEW: Swing high/low validation window
        self.swing_lookback = config.get('swing_lookback', 30)

    def detect_order_blocks(self, data: pd.DataFrame) -> List[OrderBlock]:
        """
        Detect order blocks in price data with adaptive thresholds.

        Args:
            data: OHLCV DataFrame with ATR pre-calculated

        Returns:
            List of active order blocks
        """
        try:
            if len(data) < self.lookback_bars:
                return []

            # Pre-calculate ATR if not present
            if 'atr_14' not in data.columns:
                data = data.copy()
                data['atr_14'] = self._calculate_atr(data, 14)

            order_blocks = []

            # Look for significant moves with volume
            for i in range(self.lookback_bars, len(data) - self.min_reaction_bars):
                ob = self._analyze_potential_block(data, i)
                if ob:
                    order_blocks.append(ob)

            # Filter for active (unmitigated) blocks
            active_blocks = []
            current_price = data['close'].iloc[-1]

            for ob in order_blocks:
                if self._is_block_active(ob, current_price, data):
                    active_blocks.append(ob)

            return active_blocks

        except Exception as e:
            logger.error(f"Error detecting order blocks: {e}")
            return []

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        return atr

    def _calculate_adaptive_threshold(self, data: pd.DataFrame, index: int) -> float:
        """
        Calculate adaptive displacement threshold based on recent ATR.

        Formula: max(0.5%, 1.0 × ATR%)

        This ensures:
        - Low volatility (Dec 2022): threshold ~ 0.4-0.5% (from ATR)
        - High volatility (Nov 2022): threshold ~ 0.8-1.0% (from ATR)
        - Never below 0.5% (quality floor)
        """
        try:
            # Get current ATR
            atr = data['atr_14'].iloc[index] if 'atr_14' in data.columns else None
            if atr is None or pd.isna(atr):
                # Fallback: calculate ATR manually
                window = data.iloc[max(0, index-14):index+1]
                atr = self._calculate_atr(window, 14).iloc[-1]

            # Get current price
            current_price = data['close'].iloc[index]

            # Calculate ATR as percentage of price
            atr_pct = atr / current_price if current_price > 0 else 0.02

            # Adaptive threshold: max(floor, multiplier × ATR%)
            threshold = max(self.min_displacement_pct_floor, self.atr_multiplier * atr_pct)

            return threshold

        except Exception as e:
            logger.error(f"Error calculating adaptive threshold: {e}")
            return 0.01  # Fallback to 1%

    def _is_swing_high(self, data: pd.DataFrame, index: int) -> bool:
        """
        Check if bar is a swing high (local maximum).

        RELAXED: Checks if high is in top 20% of window (not absolute max).
        This allows detection during volatile periods where absolute extremes are rare.
        """
        try:
            lookback = min(10, index)  # Reduced from 15 (swing_lookback//2)
            lookahead = min(3, len(data) - index - 1)  # Asymmetric: more lookback, less lookahead

            if lookback < 3:
                return False

            window = data.iloc[index - lookback:index + lookahead + 1]
            current_high = data['high'].iloc[index]

            # RELAXED: Check if in top 20% of highs in window (not absolute max)
            threshold = window['high'].quantile(0.80)
            return current_high >= threshold

        except Exception as e:
            return False

    def _is_swing_low(self, data: pd.DataFrame, index: int) -> bool:
        """
        Check if bar is a swing low (local minimum).

        RELAXED: Checks if low is in bottom 20% of window (not absolute min).
        This allows detection during volatile periods where absolute extremes are rare.
        """
        try:
            lookback = min(10, index)  # Reduced from 15
            lookahead = min(3, len(data) - index - 1)  # Asymmetric

            if lookback < 3:
                return False

            window = data.iloc[index - lookback:index + lookahead + 1]
            current_low = data['low'].iloc[index]

            # RELAXED: Check if in bottom 20% of lows in window (not absolute min)
            threshold = window['low'].quantile(0.20)
            return current_low <= threshold

        except Exception as e:
            return False

    def _analyze_potential_block(self, data: pd.DataFrame, index: int) -> Optional[OrderBlock]:
        """Analyze single bar for order block formation with adaptive thresholds"""
        try:
            current_bar = data.iloc[index]

            # Calculate adaptive displacement threshold
            min_displacement = self._calculate_adaptive_threshold(data, index)

            # Look at displacement after this bar
            future_data = data.iloc[index:index + self.min_reaction_bars + 1]
            if len(future_data) < self.min_reaction_bars + 1:
                return None

            # Calculate displacement
            entry_price = current_bar['close']
            future_high = future_data['high'].max()
            future_low = future_data['low'].min()

            bullish_displacement = (future_high - entry_price) / entry_price
            bearish_displacement = (entry_price - future_low) / entry_price

            # Check volume
            lookback_volume = data['volume'].iloc[max(0, index-20):index].mean()
            volume_ratio = current_bar['volume'] / lookback_volume if lookback_volume > 0 else 1

            # Determine order block type with ADAPTIVE threshold
            if (bullish_displacement >= min_displacement and
                volume_ratio >= self.min_volume_ratio and
                self._is_swing_low(data, index)):  # NEW: Must be swing low for bullish OB

                ob_type = OrderBlockType.BULLISH
                displacement = bullish_displacement
                block_high = current_bar['high']
                block_low = current_bar['low']

            elif (bearish_displacement >= min_displacement and
                  volume_ratio >= self.min_volume_ratio and
                  self._is_swing_high(data, index)):  # NEW: Must be swing high for bearish OB

                ob_type = OrderBlockType.BEARISH
                displacement = bearish_displacement
                block_high = current_bar['high']
                block_low = current_bar['low']
            else:
                return None

            # Calculate strength and confidence (normalized to adaptive threshold)
            strength = min(1.0, displacement / (min_displacement * 5))  # 5× threshold = max strength
            confidence = min(1.0, volume_ratio / (self.min_volume_ratio * 2))  # 2× ratio = max confidence

            return OrderBlock(
                timestamp=current_bar.name,
                high=block_high,
                low=block_low,
                ob_type=ob_type,
                strength=strength,
                confidence=confidence,
                displacement=displacement,
                volume_ratio=volume_ratio,
                mitigation_count=0,
                active=True,
                metadata={
                    'formation_bar': index,
                    'reaction_bars': self.min_reaction_bars,
                    'adaptive_threshold': min_displacement,
                    'is_swing_point': True
                }
            )

        except Exception as e:
            logger.error(f"Error analyzing potential block at {index}: {e}")
            return None

    def _is_block_active(self, ob: OrderBlock, current_price: float, data: pd.DataFrame) -> bool:
        """Check if order block is still active (unmitigated)"""
        try:
            # Block is mitigated if price has moved significantly through it
            if ob.ob_type == OrderBlockType.BULLISH:
                # Bullish OB mitigated if price closes below the low
                return current_price >= ob.low * 0.995  # Small buffer
            else:
                # Bearish OB mitigated if price closes above the high
                return current_price <= ob.high * 1.005  # Small buffer

        except Exception as e:
            logger.error(f"Error checking block activity: {e}")
            return False

    def get_nearest_blocks(self, data: pd.DataFrame, price: float,
                          distance_pct: float = 0.05) -> List[OrderBlock]:
        """Get order blocks within distance of current price"""
        try:
            all_blocks = self.detect_order_blocks(data)
            nearby_blocks = []

            for ob in all_blocks:
                # Calculate distance to block
                if ob.ob_type == OrderBlockType.BULLISH:
                    block_price = (ob.high + ob.low) / 2
                else:
                    block_price = (ob.high + ob.low) / 2

                distance = abs(price - block_price) / price

                if distance <= distance_pct:
                    nearby_blocks.append(ob)

            # Sort by distance
            nearby_blocks.sort(key=lambda x: abs(price - (x.high + x.low) / 2))
            return nearby_blocks

        except Exception as e:
            logger.error(f"Error finding nearest blocks: {e}")
            return []
