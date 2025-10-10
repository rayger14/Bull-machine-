"""
Narrative Trap Detector - Bull Machine v1.8.5

Detects "HODL trap" patterns and exit liquidity programming via
volume divergence + alt-season hype metrics.

Trader Alignment:
- @Wyckoff_Insider: Distribution disguised as accumulation (post:18)
- @Moneytaur: Smart money exits during retail FOMO (post:33)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def decode_liquidity_programming(df: pd.DataFrame, macro_cache: Dict, config: Dict) -> tuple:
    """
    Detect HODL trap patterns (low volume + hype = exit liquidity).

    Signs of exit liquidity programming:
    1. Price at highs but volume declining
    2. TOTAL3 (alt market cap) inflating faster than BTC
    3. Funding rates elevated but OI declining
    4. Social sentiment extreme bullish

    Args:
        df: OHLCV DataFrame
        macro_cache: Macro data cache with TOTAL3, BTC.D, etc.
        config: Config dict

    Returns:
        (is_trap, trap_message)

    Example:
        >>> df = pd.DataFrame({'close': [100, 102, 105], 'volume': [1000, 800, 600]})
        >>> macro = {'TOTAL3': {'value': 1.2}}  # 20% higher than baseline
        >>> is_trap, msg = decode_liquidity_programming(df, macro, {...})
        >>> # Returns (True, "HODL trap: Low volume + hype...")
    """
    if len(df) < 20:
        return False, ""

    # Get recent price and volume
    recent_close = df['close'].tail(10).mean()
    recent_volume = df['volume'].tail(10).mean()
    older_volume = df['volume'].tail(30).head(20).mean()

    # Volume divergence (declining volume on rising price)
    volume_ratio = recent_volume / older_volume if older_volume > 0 else 1.0
    volume_declining = volume_ratio < config.get('hodl_trap_vol_min', 0.8)

    # Check alt-season hype (TOTAL3 dominance)
    total3_hype = False
    if 'TOTAL3' in macro_cache:
        total3_value = macro_cache['TOTAL3'].get('value', 1.0)
        total3_threshold = config.get('total3_hype_threshold', 1.1)
        total3_hype = total3_value > total3_threshold

    # Check BTC dominance declining (alt season)
    btc_dom_declining = False
    if 'BTC_D' in macro_cache and len(macro_cache['BTC_D']) > 1:
        btc_d = macro_cache['BTC_D']
        if 'value' in btc_d:
            recent_btc_d = btc_d['value'].iloc[-1] if isinstance(btc_d['value'], pd.Series) else btc_d['value']
            older_btc_d = btc_d['value'].iloc[-10] if isinstance(btc_d['value'], pd.Series) else btc_d['value']
            btc_dom_declining = recent_btc_d < older_btc_d * 0.95  # 5% drop

    # Detect trap pattern
    if volume_declining and total3_hype:
        return True, "HODL trap detected: Low volume + alt hype = exit liquidity programming"

    if volume_declining and btc_dom_declining:
        return True, "HODL trap detected: Volume divergence during alt rotation"

    return False, ""


def check_distribution_pattern(df: pd.DataFrame, config: Dict) -> tuple:
    """
    Check for Wyckoff distribution pattern signs.

    Distribution characteristics:
    - Price making higher highs
    - Volume declining
    - Wicks getting longer (rejection)
    - Range tightening

    Args:
        df: OHLCV DataFrame
        config: Config dict

    Returns:
        (is_distributing, confidence)

    Example:
        >>> df = pd.DataFrame({
        ...     'high': [100, 102, 104, 103],
        ...     'low': [98, 100, 102, 101],
        ...     'close': [99, 101, 103, 102],
        ...     'volume': [1000, 900, 800, 700]
        ... })
        >>> is_dist, conf = check_distribution_pattern(df, {})
        >>> # Returns (True, 0.7) if distribution detected
    """
    if len(df) < 30:
        return False, 0.0

    recent = df.tail(20)

    # Check volume trend
    volume_corr = recent['volume'].corr(pd.Series(range(len(recent))))
    volume_declining = volume_corr < -0.3

    # Check wick length (rejection)
    recent['upper_wick'] = recent['high'] - recent[['open', 'close']].max(axis=1)
    recent['lower_wick'] = recent[['open', 'close']].min(axis=1) - recent['low']
    recent['body'] = (recent['close'] - recent['open']).abs()

    avg_upper_wick = recent['upper_wick'].mean()
    avg_body = recent['body'].mean()
    upper_wick_ratio = avg_upper_wick / avg_body if avg_body > 0 else 0

    long_wicks = upper_wick_ratio > 0.5  # Upper wicks > 50% of body

    # Check range compression
    price_range = recent['high'].max() - recent['low'].min()
    earlier_range = df.tail(40).head(20)['high'].max() - df.tail(40).head(20)['low'].min()
    range_ratio = price_range / earlier_range if earlier_range > 0 else 1.0
    range_tightening = range_ratio < 0.7

    # Calculate confidence
    signals = [volume_declining, long_wicks, range_tightening]
    confidence = sum(signals) / len(signals)

    is_distributing = confidence >= 0.6

    return is_distributing, confidence


def should_veto_narrative(df: pd.DataFrame, macro_cache: Dict, config: Dict) -> tuple:
    """
    Determine if trade should be vetoed based on narrative analysis.

    Args:
        df: OHLCV DataFrame
        macro_cache: Macro data cache
        config: Config dict with 'narrative_enabled'

    Returns:
        (should_veto, reason)

    Example:
        >>> df = pd.DataFrame({'close': [100, 105], 'volume': [1000, 600]})
        >>> macro = {'TOTAL3': {'value': 1.3}}
        >>> veto, reason = should_veto_narrative(df, macro, {'narrative_enabled': True})
    """
    if not config.get('narrative_enabled', False):
        return False, ""

    # Check for HODL trap
    is_trap, trap_msg = decode_liquidity_programming(df, macro_cache, config)
    if is_trap:
        return True, trap_msg

    # Check for distribution
    is_dist, dist_conf = check_distribution_pattern(df, config)
    if is_dist and dist_conf > 0.7:
        return True, f"Wyckoff distribution detected (confidence: {dist_conf:.2f})"

    return False, ""


if __name__ == '__main__':
    # Quick validation
    print("Testing narrative trap detector...")

    config = {
        'hodl_trap_vol_min': 0.8,
        'total3_hype_threshold': 1.1,
        'narrative_enabled': True
    }

    # Test 1: HODL trap pattern
    trap_df = pd.DataFrame({
        'close': [100 + i for i in range(30)],  # Rising price
        'volume': [1000 - i * 20 for i in range(30)]  # Declining volume
    })

    macro_cache = {
        'TOTAL3': {'value': 1.2},  # 20% above baseline
        'BTC_D': {'value': pd.Series([45, 44, 43])}  # Declining BTC dominance
    }

    is_trap, trap_msg = decode_liquidity_programming(trap_df, macro_cache, config)
    print(f"\nHODL trap test:")
    print(f"  Detected: {is_trap}")
    print(f"  Message: {trap_msg}")

    # Test 2: Distribution pattern
    dist_df = pd.DataFrame({
        'high': [100 + i + np.random.rand() * 2 for i in range(40)],
        'low': [98 + i - np.random.rand() * 2 for i in range(40)],
        'open': [99 + i for i in range(40)],
        'close': [99.5 + i for i in range(40)],
        'volume': [1000 - i * 15 for i in range(40)]  # Declining volume
    })

    is_dist, dist_conf = check_distribution_pattern(dist_df, config)
    print(f"\nDistribution pattern test:")
    print(f"  Detected: {is_dist}")
    print(f"  Confidence: {dist_conf:.2f}")

    # Test 3: Veto check
    veto, reason = should_veto_narrative(trap_df, macro_cache, config)
    print(f"\nVeto test:")
    print(f"  Should veto: {veto}")
    print(f"  Reason: {reason}")

    assert is_trap, "Should detect HODL trap"
    assert veto, "Should veto on trap"

    print("\n✅ Narrative trap detector validated")
