"""
Order Block Detection

Identifies institutional order blocks - areas where large orders were placed
and price moved away with significance, leaving imbalances to be filled.
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
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

class OrderBlockDetector:
    """
    Order Block Detection Engine

    Identifies unmitigated institutional order blocks with high probability
    of providing support/resistance.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_displacement_pct = config.get('min_displacement_pct', 0.02)  # 2%
        self.min_volume_ratio = config.get('min_volume_ratio', 1.5)
        self.lookback_bars = config.get('lookback_bars', 50)
        self.min_reaction_bars = config.get('min_reaction_bars', 3)

    def detect_order_blocks(self, data: pd.DataFrame) -> List[OrderBlock]:
        """
        Detect order blocks in price data.

        Args:
            data: OHLCV DataFrame

        Returns:
            List of active order blocks
        """
        try:
            if len(data) < self.lookback_bars:
                return []

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

    def _analyze_potential_block(self, data: pd.DataFrame, index: int) -> Optional[OrderBlock]:
        """Analyze single bar for order block formation"""
        try:
            current_bar = data.iloc[index]

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

            # Determine order block type
            if (bullish_displacement >= self.min_displacement_pct and
                volume_ratio >= self.min_volume_ratio):

                ob_type = OrderBlockType.BULLISH
                displacement = bullish_displacement
                block_high = current_bar['high']
                block_low = current_bar['low']

            elif (bearish_displacement >= self.min_displacement_pct and
                  volume_ratio >= self.min_volume_ratio):

                ob_type = OrderBlockType.BEARISH
                displacement = bearish_displacement
                block_high = current_bar['high']
                block_low = current_bar['low']
            else:
                return None

            # Calculate strength and confidence
            strength = min(1.0, displacement / 0.05)  # Scale to max 5% move
            confidence = min(1.0, volume_ratio / 3.0)  # Scale to max 3x volume

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
                    'reaction_bars': self.min_reaction_bars
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