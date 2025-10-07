"""
Macro Data Loader for Bull Machine v1.7.3 Live Feeds
Loads all macro series from CSVs (mock feed) or future API calls
"""
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Global macro series list (traders' complete context)
UNIVERSAL_MACRO_SERIES = [
    'DXY',       # Dollar Index (Moneytaur: liquidity drain)
    'WTI',       # Oil/Energy (ZeroIKA: inflation/stagflation)
    'US10Y',     # 10Y Treasury (yield curve)
    'US2Y',      # 2Y Treasury (inversion detection)
    'VIX',       # Volatility Index (Wyckoff: risk regime)
    'GOLD',      # Gold (Wyckoff: flight to safety)
    'MOVE',      # Bond Volatility (Wyckoff: credit stress)
    'EURUSD'     # EUR/USD (ZeroIKA: USD weakness)
]

CRYPTO_MACRO_SERIES = [
    'USDT.D',    # USDT Dominance (ZeroIKA: stablecoin coil)
    'USDC.D',    # USDC Dominance (Wyckoff: alt bleed)
    'TOTAL',     # Total Market Cap (flow strength)
    'TOTAL2',    # Alt Market Cap (alt flows)
    'TOTAL3',    # Alt Market Cap ex ETH/BTC (alt momentum)
    'BTC.D',     # Bitcoin Dominance (Wyckoff: crypto dominance)
    'FUNDING',   # Funding Rates (Moneytaur: leverage stress)
    'OI'         # Open Interest Premium (ZeroIKA: leverage stress)
]

STOCK_MACRO_SERIES = [
    'SPY_QQQ',   # SPY/QQQ ratio (large-cap tech dominance)
    'SPY_IWM',   # SPY/IWM ratio (large vs small-cap)
    'SPY_OI'     # SPY options/futures OI premium
]

# Complete series list
MACRO_SERIES = UNIVERSAL_MACRO_SERIES + CRYPTO_MACRO_SERIES + STOCK_MACRO_SERIES

def load_macro_data(data_dir: str = "data",
                   symbols: Optional[List[str]] = None,
                   asset_type: str = "crypto") -> Dict[str, pd.DataFrame]:
    """
    Load macro series from CSV files based on asset type.

    Args:
        data_dir: Directory containing macro CSV files
        symbols: List of symbols to load (default: auto-selected by asset_type)
        asset_type: "crypto", "stock", or "all" to determine which series to load

    Returns:
        Dict mapping symbol to DataFrame
    """
    if symbols is None:
        if asset_type == "crypto":
            symbols = UNIVERSAL_MACRO_SERIES + CRYPTO_MACRO_SERIES
        elif asset_type == "stock":
            symbols = UNIVERSAL_MACRO_SERIES + STOCK_MACRO_SERIES
        elif asset_type == "all":
            symbols = MACRO_SERIES
        else:
            symbols = UNIVERSAL_MACRO_SERIES  # Default to universal only

    macro_data = {}
    data_path = Path(data_dir)

    for symbol in symbols:
        file_patterns = [
            f"{symbol}_1D.csv",           # Standard format
            f"{symbol}.csv",              # Simple format
            f"TVC_{symbol}_1D.csv",       # TradingView format
            f"CRYPTOCAP_{symbol}_1D.csv", # Crypto format
            f"EASYMARKETS_{symbol}_1D.csv" # Broker format
        ]

        df = None
        for pattern in file_patterns:
            file_path = data_path / pattern
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)

                    # Standardize timestamp column
                    if 'time' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                    elif 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    else:
                        logger.warning(f"No timestamp column found in {file_path}")
                        continue

                    # Standardize value column
                    if 'close' in df.columns:
                        df['value'] = df['close']
                    elif 'Close' in df.columns:
                        df['value'] = df['Close']
                    elif 'value' in df.columns:
                        pass  # Already has value column
                    else:
                        logger.warning(f"No value column found in {file_path}")
                        continue

                    # Sort by timestamp
                    df = df.sort_values('timestamp').reset_index(drop=True)

                    logger.info(f"Loaded {symbol}: {len(df)} bars from {file_path}")
                    break

                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    continue

        if df is not None:
            macro_data[symbol] = df[['timestamp', 'value']].copy()
        else:
            logger.warning(f"Could not load {symbol} - creating empty DataFrame")
            macro_data[symbol] = pd.DataFrame(columns=['timestamp', 'value'])

    return macro_data

def fetch_macro_snapshot(macro_data: Dict[str, pd.DataFrame],
                        timestamp: pd.Timestamp,
                        stale_tolerance_minutes: int = 60) -> Dict[str, Dict]:
    """
    Get macro snapshot at specific timestamp with staleness check.

    Args:
        macro_data: Dict of macro DataFrames
        timestamp: Target timestamp
        stale_tolerance_minutes: Max allowed staleness

    Returns:
        Dict mapping symbol to {'value': float, 'timestamp': pd.Timestamp, 'stale': bool}
    """
    snapshot = {}

    for symbol, df in macro_data.items():
        if df.empty:
            snapshot[symbol] = {
                'value': None,
                'timestamp': timestamp,
                'stale': True,
                'status': 'missing'
            }
            continue

        # Find most recent value <= target timestamp (right-edge enforcement)
        recent_data = df[df['timestamp'] <= timestamp]

        if recent_data.empty:
            snapshot[symbol] = {
                'value': None,
                'timestamp': timestamp,
                'stale': True,
                'status': 'future_only'
            }
            continue

        latest = recent_data.iloc[-1]
        age_minutes = (timestamp - latest['timestamp']).total_seconds() / 60
        is_stale = age_minutes > stale_tolerance_minutes

        snapshot[symbol] = {
            'value': latest['value'],
            'timestamp': latest['timestamp'],
            'stale': is_stale,
            'age_minutes': age_minutes,
            'status': 'stale' if is_stale else 'fresh'
        }

    return snapshot

def get_macro_health_status(snapshot: Dict[str, Dict]) -> Dict[str, any]:
    """
    Assess macro data health for monitoring.

    Returns:
        Health metrics dict
    """
    total_series = len(snapshot)
    fresh_series = sum(1 for s in snapshot.values() if not s['stale'] and s['value'] is not None)
    missing_series = sum(1 for s in snapshot.values() if s['value'] is None)

    return {
        'total_series': total_series,
        'fresh_series': fresh_series,
        'missing_series': missing_series,
        'stale_series': total_series - fresh_series - missing_series,
        'health_pct': (fresh_series / total_series * 100) if total_series > 0 else 0,
        'critical_missing': [k for k, v in snapshot.items()
                           if k in ['DXY', 'VIX', 'US10Y'] and v['value'] is None]
    }