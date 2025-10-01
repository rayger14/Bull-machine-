"""
Liquidity Sweep Detection

Identifies when price sweeps above/below previous highs/lows to trigger
stop losses and retail orders, then reverses - classic institutional behavior.
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class SweepType(Enum):
    """Liquidity sweep types"""
    SELL_SIDE = "sell_side"  # Sweep below lows (triggers sell stops)
    BUY_SIDE = "buy_side"    # Sweep above highs (triggers buy stops)

@dataclass
class LiquiditySweep:
    """Liquidity sweep data structure"""
    timestamp: pd.Timestamp
    sweep_type: SweepType
    trigger_level: float  # The high/low that was swept
    sweep_price: float    # How far price swept beyond level
    reversal_confirmation: bool  # Has reversal been confirmed
    strength: float       # 0-1 based on sweep distance and volume
    confidence: float     # 0-1 based on reversal quality
    wick_ratio: float     # Wick size vs body size
    volume_ratio: float   # Volume during sweep vs average
    pip_distance: float   # How many pips beyond the level
    metadata: Dict[str, Any]

class LiquiditySweepDetector:
    """
    Liquidity Sweep Detection Engine

    Identifies when institutional players sweep liquidity pools
    (stop losses clustered beyond previous highs/lows) then reverse.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_pip_sweep = config.get('min_pip_sweep', 10)  # Minimum pips beyond level
        self.min_wick_ratio = config.get('min_wick_ratio', 0.6)  # Wick should be 60% of range
        self.lookback_bars = config.get('lookback_bars', 50)
        self.min_volume_ratio = config.get('min_volume_ratio', 1.3)
        self.reversal_bars = config.get('reversal_bars', 3)

    def detect_sweeps(self, data: pd.DataFrame) -> List[LiquiditySweep]:
        """
        Detect liquidity sweeps in price data.

        Args:
            data: OHLCV DataFrame

        Returns:
            List of confirmed liquidity sweeps
        """
        try:
            if len(data) < self.lookback_bars:
                return []

            sweeps = []

            # Look for potential sweeps
            for i in range(self.lookback_bars, len(data) - self.reversal_bars):
                sweep = self._analyze_potential_sweep(data, i)
                if sweep:
                    sweeps.append(sweep)

            return sweeps

        except Exception as e:
            logger.error(f"Error detecting liquidity sweeps: {e}")
            return []

    def _analyze_potential_sweep(self, data: pd.DataFrame, index: int) -> Optional[LiquiditySweep]:
        """Analyze single bar for liquidity sweep"""
        try:
            current_bar = data.iloc[index]

            # Look back for recent highs/lows to sweep
            lookback_data = data.iloc[max(0, index - self.lookback_bars):index]
            if len(lookback_data) < 10:
                return None

            # Find recent significant highs and lows
            recent_high = lookback_data['high'].max()
            recent_low = lookback_data['low'].min()

            # Check for sell-side sweep (below recent low)
            if current_bar['low'] < recent_low:
                sweep_distance = recent_low - current_bar['low']
                pip_distance = sweep_distance  # Simplified pip calculation

                if pip_distance >= self.min_pip_sweep / 100000:  # Convert to price units
                    # Check for reversal pattern (long wick below)
                    bar_range = current_bar['high'] - current_bar['low']
                    lower_wick = min(current_bar['open'], current_bar['close']) - current_bar['low']
                    wick_ratio = lower_wick / bar_range if bar_range > 0 else 0

                    if wick_ratio >= self.min_wick_ratio:
                        # Check volume confirmation
                        avg_volume = lookback_data['volume'].mean()
                        volume_ratio = current_bar['volume'] / avg_volume if avg_volume > 0 else 1

                        if volume_ratio >= self.min_volume_ratio:
                            # Check for reversal confirmation in next bars
                            reversal_confirmed = self._check_reversal_confirmation(
                                data, index, SweepType.SELL_SIDE, current_bar['low']
                            )

                            strength = min(1.0, pip_distance * 100000 / 50)  # Scale to 50 pips max
                            confidence = min(1.0, (wick_ratio + volume_ratio - 1) / 2)

                            return LiquiditySweep(
                                timestamp=current_bar.name,
                                sweep_type=SweepType.SELL_SIDE,
                                trigger_level=recent_low,
                                sweep_price=current_bar['low'],
                                reversal_confirmation=reversal_confirmed,
                                strength=strength,
                                confidence=confidence,
                                wick_ratio=wick_ratio,
                                volume_ratio=volume_ratio,
                                pip_distance=pip_distance * 100000,
                                metadata={
                                    'lookback_low': recent_low,
                                    'sweep_distance': sweep_distance,
                                    'bar_range': bar_range
                                }
                            )

            # Check for buy-side sweep (above recent high)
            elif current_bar['high'] > recent_high:
                sweep_distance = current_bar['high'] - recent_high
                pip_distance = sweep_distance  # Simplified pip calculation

                if pip_distance >= self.min_pip_sweep / 100000:
                    # Check for reversal pattern (long wick above)
                    bar_range = current_bar['high'] - current_bar['low']
                    upper_wick = current_bar['high'] - max(current_bar['open'], current_bar['close'])
                    wick_ratio = upper_wick / bar_range if bar_range > 0 else 0

                    if wick_ratio >= self.min_wick_ratio:
                        # Check volume confirmation
                        avg_volume = lookback_data['volume'].mean()
                        volume_ratio = current_bar['volume'] / avg_volume if avg_volume > 0 else 1

                        if volume_ratio >= self.min_volume_ratio:
                            # Check for reversal confirmation in next bars
                            reversal_confirmed = self._check_reversal_confirmation(
                                data, index, SweepType.BUY_SIDE, current_bar['high']
                            )

                            strength = min(1.0, pip_distance * 100000 / 50)  # Scale to 50 pips max
                            confidence = min(1.0, (wick_ratio + volume_ratio - 1) / 2)

                            return LiquiditySweep(
                                timestamp=current_bar.name,
                                sweep_type=SweepType.BUY_SIDE,
                                trigger_level=recent_high,
                                sweep_price=current_bar['high'],
                                reversal_confirmation=reversal_confirmed,
                                strength=strength,
                                confidence=confidence,
                                wick_ratio=wick_ratio,
                                volume_ratio=volume_ratio,
                                pip_distance=pip_distance * 100000,
                                metadata={
                                    'lookback_high': recent_high,
                                    'sweep_distance': sweep_distance,
                                    'bar_range': bar_range
                                }
                            )

            return None

        except Exception as e:
            logger.error(f"Error analyzing potential sweep at {index}: {e}")
            return None

    def _check_reversal_confirmation(self, data: pd.DataFrame, sweep_index: int,
                                   sweep_type: SweepType, sweep_price: float) -> bool:
        """Check if price reversed after the sweep"""
        try:
            # Look at next few bars for reversal
            future_data = data.iloc[sweep_index + 1:sweep_index + 1 + self.reversal_bars]
            if len(future_data) < self.reversal_bars:
                return False

            if sweep_type == SweepType.SELL_SIDE:
                # After sell-side sweep, expect price to move back up
                reversal_high = future_data['high'].max()
                # Reversal confirmed if price moves back above the trigger level
                return reversal_high > sweep_price * 1.001  # Small buffer

            else:  # BUY_SIDE
                # After buy-side sweep, expect price to move back down
                reversal_low = future_data['low'].min()
                # Reversal confirmed if price moves back below the trigger level
                return reversal_low < sweep_price * 0.999  # Small buffer

        except Exception as e:
            logger.error(f"Error checking reversal confirmation: {e}")
            return False

    def get_recent_sweeps(self, data: pd.DataFrame, bars_back: int = 20) -> List[LiquiditySweep]:
        """Get liquidity sweeps from recent bars"""
        try:
            all_sweeps = self.detect_sweeps(data)

            # Filter for recent sweeps
            recent_sweeps = []
            cutoff_time = data.index[-bars_back] if len(data) > bars_back else data.index[0]

            for sweep in all_sweeps:
                if sweep.timestamp >= cutoff_time:
                    recent_sweeps.append(sweep)

            return recent_sweeps

        except Exception as e:
            logger.error(f"Error getting recent sweeps: {e}")
            return []