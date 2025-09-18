"""Bull Machine v1.3 - Timeframe Conversion Utilities"""

def tf_to_pandas_freq(tf_str: str) -> str:
    """
    Convert timeframe string to pandas frequency string.

    Args:
        tf_str: Timeframe like '1H', '4H', '1D', '15m'

    Returns:
        Pandas frequency string
    """
    # Direct mapping for common timeframes
    mappings = {
        '1m': '1min',
        '3m': '3min',
        '5m': '5min',
        '15m': '15min',
        '30m': '30min',
        '1H': '1H',
        '1h': '1H',
        '2H': '2H',
        '2h': '2H',
        '4H': '4H',
        '4h': '4H',
        '6H': '6H',
        '6h': '6H',
        '12H': '12H',
        '12h': '12H',
        '1D': '1D',
        '1d': '1D',
        '3D': '3D',
        '3d': '3D',
        '1W': '1W',
        '1w': '1W',
    }

    return mappings.get(tf_str, tf_str)

def parse_tf(tf_str: str) -> tuple:
    """
    Parse timeframe string into (value, unit).

    Args:
        tf_str: Timeframe like '4H', '15m', '1D'

    Returns:
        Tuple of (value, unit) e.g. (4, 'H'), (15, 'm')
    """
    import re

    match = re.match(r'(\d+)([mHhDdWw])', tf_str)
    if match:
        value = int(match.group(1))
        unit = match.group(2).upper()
        return value, unit

    # Default
    return 1, 'H'

def tf_order(tf_str: str) -> int:
    """
    Get ordering value for timeframe (higher = longer timeframe).

    Args:
        tf_str: Timeframe string

    Returns:
        Integer for sorting (higher values = longer timeframes)
    """
    value, unit = parse_tf(tf_str)

    # Convert to minutes for comparison
    unit_minutes = {
        'M': 1,
        'H': 60,
        'D': 1440,
        'W': 10080
    }

    minutes = value * unit_minutes.get(unit, 60)
    return minutes

def tf_multiplier(tf_from: str, tf_to: str) -> int:
    """
    Calculate multiplier between timeframes.

    Args:
        tf_from: Source timeframe
        tf_to: Target timeframe

    Returns:
        Integer multiplier (e.g. '1H' to '4H' = 4)
    """
    from_minutes = tf_order(tf_from)
    to_minutes = tf_order(tf_to)

    if from_minutes == 0:
        return 1

    return to_minutes // from_minutes