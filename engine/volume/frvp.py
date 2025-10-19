"""
FRVP (Fixed Range Volume Profile)

Analyzes volume distribution across price levels to identify:
- POC (Point of Control): Highest volume node
- HVN (High Volume Nodes): Liquidity clusters
- LVN (Low Volume Nodes): Liquidity gaps
- VA (Value Area): 70% of volume concentration

Author: Bull Machine v2.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class FRVPProfile:
    """
    Fixed Range Volume Profile result.

    Attributes:
        poc: Point of Control (price with highest volume)
        va_high: Value Area High (top of 70% volume zone)
        va_low: Value Area Low (bottom of 70% volume zone)
        hvn_levels: List of High Volume Nodes
        lvn_levels: List of Low Volume Nodes
        volume_distribution: Dict mapping price levels to volume
        current_position: 'above_va' | 'in_va' | 'below_va'
    """
    poc: float
    va_high: float
    va_low: float
    hvn_levels: List[float]
    lvn_levels: List[float]
    volume_distribution: Dict[float, float]
    current_position: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for feature store."""
        return {
            'frvp_poc': self.poc,
            'frvp_va_high': self.va_high,
            'frvp_va_low': self.va_low,
            'frvp_hvn_count': len(self.hvn_levels),
            'frvp_lvn_count': len(self.lvn_levels),
            'frvp_current_position': self.current_position,
            # Distance metrics
            'frvp_distance_to_poc': 0.0,  # Filled by caller
            'frvp_distance_to_va': 0.0,   # Filled by caller
        }


def build_volume_profile(df: pd.DataFrame, price_bins: int = 50) -> Tuple[Dict[float, float], float, float]:
    """
    Build volume profile by aggregating volume at price levels.

    Args:
        df: OHLCV DataFrame
        price_bins: Number of price bins to divide range into

    Returns:
        (volume_distribution, price_min, price_max)
    """
    price_min = df['low'].min()
    price_max = df['high'].max()

    # Create price bins
    price_range = price_max - price_min
    bin_size = price_range / price_bins

    # Initialize volume distribution
    volume_dist = {}

    # For each bar, distribute volume across touched price levels
    for idx, row in df.iterrows():
        bar_low = row['low']
        bar_high = row['high']
        bar_volume = row['volume']

        # Find bins this bar touched
        low_bin = int((bar_low - price_min) / bin_size)
        high_bin = int((bar_high - price_min) / bin_size)

        # Distribute volume evenly across touched bins
        bins_touched = max(1, high_bin - low_bin + 1)
        volume_per_bin = bar_volume / bins_touched

        for bin_idx in range(low_bin, high_bin + 1):
            if bin_idx < 0 or bin_idx >= price_bins:
                continue
            price_level = price_min + (bin_idx + 0.5) * bin_size
            volume_dist[price_level] = volume_dist.get(price_level, 0) + volume_per_bin

    return volume_dist, price_min, price_max


def find_poc(volume_dist: Dict[float, float]) -> float:
    """Find Point of Control (price with highest volume)."""
    if not volume_dist:
        return 0.0

    poc_price = max(volume_dist.items(), key=lambda x: x[1])[0]
    return float(poc_price)


def find_value_area(volume_dist: Dict[float, float], poc: float,
                    target_volume_pct: float = 0.70) -> Tuple[float, float]:
    """
    Find Value Area (70% of volume around POC).

    Args:
        volume_dist: Volume distribution
        poc: Point of Control
        target_volume_pct: Target volume percentage (default 0.70 = 70%)

    Returns:
        (va_high, va_low)
    """
    if not volume_dist:
        return 0.0, 0.0

    # Sort prices
    sorted_prices = sorted(volume_dist.keys())
    total_volume = sum(volume_dist.values())
    target_volume = total_volume * target_volume_pct

    # Find POC index
    poc_idx = min(range(len(sorted_prices)),
                  key=lambda i: abs(sorted_prices[i] - poc))

    # Expand outward from POC until we capture target_volume_pct
    va_low_idx = poc_idx
    va_high_idx = poc_idx
    captured_volume = volume_dist[sorted_prices[poc_idx]]

    while captured_volume < target_volume:
        # Check which direction has more volume
        volume_below = 0
        volume_above = 0

        if va_low_idx > 0:
            volume_below = volume_dist[sorted_prices[va_low_idx - 1]]

        if va_high_idx < len(sorted_prices) - 1:
            volume_above = volume_dist[sorted_prices[va_high_idx + 1]]

        # Expand toward higher volume
        if volume_below > volume_above and va_low_idx > 0:
            va_low_idx -= 1
            captured_volume += volume_below
        elif va_high_idx < len(sorted_prices) - 1:
            va_high_idx += 1
            captured_volume += volume_above
        else:
            break

    va_high = sorted_prices[va_high_idx]
    va_low = sorted_prices[va_low_idx]

    return float(va_high), float(va_low)


def find_hvn_lvn(volume_dist: Dict[float, float], threshold: float = 1.5) -> Tuple[List[float], List[float]]:
    """
    Find High Volume Nodes (HVN) and Low Volume Nodes (LVN).

    Args:
        volume_dist: Volume distribution
        threshold: HVN threshold multiplier (1.5 = 150% of mean volume)

    Returns:
        (hvn_levels, lvn_levels)
    """
    if not volume_dist:
        return [], []

    mean_volume = np.mean(list(volume_dist.values()))
    sorted_prices = sorted(volume_dist.keys())

    hvn_levels = []
    lvn_levels = []

    for price in sorted_prices:
        volume = volume_dist[price]

        # HVN: Volume > threshold * mean
        if volume > mean_volume * threshold:
            hvn_levels.append(float(price))

        # LVN: Volume < mean / threshold
        elif volume < mean_volume / threshold:
            lvn_levels.append(float(price))

    return hvn_levels, lvn_levels


def calculate_frvp(df: pd.DataFrame, lookback: int = 100,
                   price_bins: int = 50, config: Optional[Dict] = None) -> FRVPProfile:
    """
    Calculate Fixed Range Volume Profile.

    Args:
        df: OHLCV DataFrame
        lookback: Number of bars to include in profile
        price_bins: Number of price bins
        config: Optional configuration

    Returns:
        FRVPProfile with POC, VA, HVN, LVN

    Example:
        >>> frvp = calculate_frvp(df_4h, lookback=100)
        >>> if frvp.current_position == 'below_va':
        >>>     # Price below value area (cheap zone)
        >>>     fusion_score += 0.05
    """
    config = config or {}

    if len(df) < lookback:
        lookback = len(df)

    # Use recent bars for profile
    df_range = df.iloc[-lookback:].copy()

    if len(df_range) < 10:
        return FRVPProfile(
            poc=0.0,
            va_high=0.0,
            va_low=0.0,
            hvn_levels=[],
            lvn_levels=[],
            volume_distribution={},
            current_position='in_va'
        )

    # Build volume profile
    volume_dist, price_min, price_max = build_volume_profile(df_range, price_bins=price_bins)

    # Find POC
    poc = find_poc(volume_dist)

    # Find Value Area
    va_high, va_low = find_value_area(volume_dist, poc, target_volume_pct=0.70)

    # Find HVN/LVN
    hvn_threshold = config.get('frvp_hvn_threshold', 1.5)
    hvn_levels, lvn_levels = find_hvn_lvn(volume_dist, threshold=hvn_threshold)

    # Determine current position
    current_price = df['close'].iloc[-1]

    if current_price > va_high:
        current_position = 'above_va'  # Expensive zone
    elif current_price < va_low:
        current_position = 'below_va'  # Cheap zone
    else:
        current_position = 'in_va'  # Fair value zone

    return FRVPProfile(
        poc=poc,
        va_high=va_high,
        va_low=va_low,
        hvn_levels=hvn_levels,
        lvn_levels=lvn_levels,
        volume_distribution=volume_dist,
        current_position=current_position
    )


def apply_frvp_fusion_adjustment(fusion_score: float, frvp: FRVPProfile,
                                  current_price: float, direction: str,
                                  config: Optional[Dict] = None) -> tuple:
    """
    Apply FRVP fusion adjustment.

    Args:
        fusion_score: Current fusion score
        frvp: FRVP profile from calculate_frvp()
        current_price: Current price
        direction: Intended trade direction ('long' or 'short')
        config: Optional config

    Returns:
        (adjusted_score: float, adjustment: float, reasons: list)

    Logic:
        - Long from below VA: +0.05 (buying cheap)
        - Short from above VA: +0.05 (selling expensive)
        - Near POC: +0.03 (high liquidity, good fill)
        - Near LVN: -0.05 (low liquidity gap, risky)
    """
    config = config or {}
    adjustment = 0.0
    reasons = []

    # Distance to POC
    if frvp.poc > 0:
        distance_to_poc_pct = abs(current_price - frvp.poc) / frvp.poc

        # Near POC bonus (within 1%)
        if distance_to_poc_pct < 0.01:
            adjustment += 0.03
            reasons.append(f"Near POC ({frvp.poc:.2f}) - high liquidity")

    # Value Area position
    if frvp.current_position == 'below_va' and direction == 'long':
        # Buying cheap
        adjustment += 0.05
        reasons.append(f"Below VA ({frvp.va_low:.2f}) - buying cheap")
    elif frvp.current_position == 'above_va' and direction == 'short':
        # Selling expensive
        adjustment += 0.05
        reasons.append(f"Above VA ({frvp.va_high:.2f}) - selling expensive")

    # LVN proximity penalty
    if frvp.lvn_levels:
        nearest_lvn = min(frvp.lvn_levels, key=lambda x: abs(x - current_price))
        distance_to_lvn_pct = abs(current_price - nearest_lvn) / current_price

        if distance_to_lvn_pct < 0.01:
            # Very close to LVN (liquidity gap)
            adjustment -= 0.05
            reasons.append(f"Near LVN ({nearest_lvn:.2f}) - liquidity gap risk")

    # HVN proximity bonus
    if frvp.hvn_levels:
        nearest_hvn = min(frvp.hvn_levels, key=lambda x: abs(x - current_price))
        distance_to_hvn_pct = abs(current_price - nearest_hvn) / current_price

        if distance_to_hvn_pct < 0.01:
            # Very close to HVN (liquidity cluster)
            adjustment += 0.03
            reasons.append(f"Near HVN ({nearest_hvn:.2f}) - liquidity cluster")

    # Apply adjustment
    adjusted_score = max(0.0, min(fusion_score + adjustment, 1.0))

    return adjusted_score, adjustment, reasons
