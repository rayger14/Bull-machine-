"""
Fair Value Gap (FVG) Detection

Identifies price gaps that represent institutional order imbalances,
creating zones that price tends to revisit for "fair value" fills.
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class FVGType(Enum):
    """Fair value gap types"""
    BULLISH = "bullish"  # Gap to be filled from below
    BEARISH = "bearish"  # Gap to be filled from above

@dataclass
class FairValueGap:
    """Fair value gap data structure"""
    timestamp: pd.Timestamp
    high: float  # Top of gap
    low: float   # Bottom of gap
    fvg_type: FVGType
    strength: float  # 0-1 based on gap size and volume
    confidence: float  # 0-1 based on formation quality
    gap_size_pct: float  # Gap size as % of price
    volume_ratio: float  # Volume during gap formation
    filled_percentage: float  # How much of gap has been filled (0-1)
    active: bool
    metadata: Dict[str, Any]

class FVGDetector:
    """
    Fair Value Gap Detection Engine

    Identifies institutional imbalance zones where price gapped due to
    large orders and is likely to return for efficient pricing.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_gap_pct = config.get('min_gap_pct', 0.003)  # 0.3%
        self.min_volume_ratio = config.get('min_volume_ratio', 1.2)
        self.lookback_bars = config.get('lookback_bars', 100)
        self.fill_threshold = config.get('fill_threshold', 0.7)  # 70% fill = inactive

    def detect_fvgs(self, data: pd.DataFrame) -> List[FairValueGap]:
        """
        Detect fair value gaps in price data.

        Args:
            data: OHLCV DataFrame

        Returns:
            List of active fair value gaps
        """
        try:
            if len(data) < 3:
                return []

            fvgs = []

            # Look for 3-bar FVG patterns
            for i in range(2, len(data)):
                fvg = self._analyze_fvg_pattern(data, i)
                if fvg:
                    fvgs.append(fvg)

            # Filter for active (unfilled) gaps
            active_fvgs = []
            current_price = data['close'].iloc[-1]

            for fvg in fvgs:
                if self._is_fvg_active(fvg, data, current_price):
                    active_fvgs.append(fvg)

            return active_fvgs

        except Exception as e:
            logger.error(f"Error detecting FVGs: {e}")
            return []

    def _analyze_fvg_pattern(self, data: pd.DataFrame, index: int) -> Optional[FairValueGap]:
        """
        Analyze 3-bar pattern for FVG formation.

        FVG occurs when:
        - Bar 1: Setup
        - Bar 2: Gap bar (creates imbalance)
        - Bar 3: Continuation (confirms direction)
        """
        try:
            if index < 2:
                return None

            bar1 = data.iloc[index - 2]  # Setup bar
            bar2 = data.iloc[index - 1]  # Gap bar
            bar3 = data.iloc[index]      # Confirmation bar

            # Check for bullish FVG
            # Bullish: bar1.high < bar3.low (gap between them)
            if bar1['high'] < bar3['low']:
                gap_high = bar3['low']
                gap_low = bar1['high']
                gap_size = gap_high - gap_low
                gap_pct = gap_size / bar2['close']

                if gap_pct >= self.min_gap_pct:
                    fvg_type = FVGType.BULLISH

                    # Check volume confirmation
                    avg_volume = data['volume'].iloc[max(0, index-10):index-1].mean()
                    volume_ratio = bar2['volume'] / avg_volume if avg_volume > 0 else 1

                    if volume_ratio >= self.min_volume_ratio:
                        strength = min(1.0, gap_pct / 0.02)  # Scale to max 2%
                        confidence = min(1.0, volume_ratio / 2.0)

                        return FairValueGap(
                            timestamp=bar2.name,
                            high=gap_high,
                            low=gap_low,
                            fvg_type=fvg_type,
                            strength=strength,
                            confidence=confidence,
                            gap_size_pct=gap_pct,
                            volume_ratio=volume_ratio,
                            filled_percentage=0.0,
                            active=True,
                            metadata={
                                'formation_bars': [index-2, index-1, index],
                                'bar1_high': bar1['high'],
                                'bar3_low': bar3['low']
                            }
                        )

            # Check for bearish FVG
            # Bearish: bar1.low > bar3.high (gap between them)
            elif bar1['low'] > bar3['high']:
                gap_high = bar1['low']
                gap_low = bar3['high']
                gap_size = gap_high - gap_low
                gap_pct = gap_size / bar2['close']

                if gap_pct >= self.min_gap_pct:
                    fvg_type = FVGType.BEARISH

                    # Check volume confirmation
                    avg_volume = data['volume'].iloc[max(0, index-10):index-1].mean()
                    volume_ratio = bar2['volume'] / avg_volume if avg_volume > 0 else 1

                    if volume_ratio >= self.min_volume_ratio:
                        strength = min(1.0, gap_pct / 0.02)  # Scale to max 2%
                        confidence = min(1.0, volume_ratio / 2.0)

                        return FairValueGap(
                            timestamp=bar2.name,
                            high=gap_high,
                            low=gap_low,
                            fvg_type=fvg_type,
                            strength=strength,
                            confidence=confidence,
                            gap_size_pct=gap_pct,
                            volume_ratio=volume_ratio,
                            filled_percentage=0.0,
                            active=True,
                            metadata={
                                'formation_bars': [index-2, index-1, index],
                                'bar1_low': bar1['low'],
                                'bar3_high': bar3['high']
                            }
                        )

            return None

        except Exception as e:
            logger.error(f"Error analyzing FVG pattern at {index}: {e}")
            return None

    def _is_fvg_active(self, fvg: FairValueGap, data: pd.DataFrame, current_price: float) -> bool:
        """Check if FVG is still active (not significantly filled)"""
        try:
            # Calculate how much of the gap has been filled
            gap_size = fvg.high - fvg.low

            if fvg.fvg_type == FVGType.BULLISH:
                # For bullish FVG, check how much price has retraced into gap
                if current_price <= fvg.low:
                    # Price is below gap - not filled
                    filled_pct = 0.0
                elif current_price >= fvg.high:
                    # Price is above gap - fully filled
                    filled_pct = 1.0
                else:
                    # Price is in gap - partially filled
                    filled_pct = (current_price - fvg.low) / gap_size

            else:  # BEARISH
                # For bearish FVG, check how much price has rallied into gap
                if current_price >= fvg.high:
                    # Price is above gap - not filled
                    filled_pct = 0.0
                elif current_price <= fvg.low:
                    # Price is below gap - fully filled
                    filled_pct = 1.0
                else:
                    # Price is in gap - partially filled
                    filled_pct = (fvg.high - current_price) / gap_size

            # Update filled percentage
            fvg.filled_percentage = filled_pct

            # Gap is inactive if filled beyond threshold
            return filled_pct < self.fill_threshold

        except Exception as e:
            logger.error(f"Error checking FVG activity: {e}")
            return False

    def get_nearest_fvgs(self, data: pd.DataFrame, price: float,
                        distance_pct: float = 0.03) -> List[FairValueGap]:
        """Get FVGs within distance of current price"""
        try:
            all_fvgs = self.detect_fvgs(data)
            nearby_fvgs = []

            for fvg in all_fvgs:
                # Calculate distance to gap center
                gap_center = (fvg.high + fvg.low) / 2
                distance = abs(price - gap_center) / price

                if distance <= distance_pct:
                    nearby_fvgs.append(fvg)

            # Sort by distance
            nearby_fvgs.sort(key=lambda x: abs(price - (x.high + x.low) / 2))
            return nearby_fvgs

        except Exception as e:
            logger.error(f"Error finding nearest FVGs: {e}")
            return []