"""
TradingView Data Loader

Loads real market data from TradingView CSV exports with proper symbol mapping.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
import logging

logger = logging.getLogger(__name__)

def load_tradingview_data(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    """
    Load OHLCV data from TradingView CSV exports.

    Args:
        symbol: Asset symbol (e.g., 'ETH', 'BTC', 'DXY')
        timeframe: Timeframe (e.g., '1H', '4H', '1D')
        start: Start date string
        end: End date string

    Returns:
        DataFrame with OHLCV data
    """
    try:
        chart_logs_dir = Path("chart_logs")

        # Map timeframes to TradingView export conventions
        tf_map = {
            '1H': '60',
            '4H': '240',
            '12H': '720',
            '1D': '1D',
            '1W': '1W',
            '1M': '1M'
        }
        tv_timeframe = tf_map.get(timeframe, timeframe)

        # Map symbols to TradingView file prefixes
        symbol_map = {
            'DXY': 'TVC_DXY',
            'US2Y': 'TVC_US02',
            'US10Y': 'TVC_US10',
            'GOLD': 'FX_XAUUSD',
            'WTI': 'EASYMARKETS_OILUSD',
            'BTC.D': 'CRYPTOCAP_BTC.D',
            'USDT.D': 'CRYPTOCAP_USDT.D',
            'ETH.D': 'CRYPTOCAP_ETH.D',
            'TOTAL': 'CRYPTOCAP_TOTAL',
            'TOTAL3': 'CRYPTOCAP_TOTAL3',
            'BTC': 'COINBASE_BTCUSD',
            'ETH': 'COINBASE_ETHUSD',
            'SOL': 'COINBASE_SOLUSD',
            'XRP': 'COINBASE_XRPUSD',
            'ETHBTC': 'COINBASE_ETHBTC'
        }

        tv_symbol = symbol_map.get(symbol, symbol)

        # Use glob to find files with TradingView hash patterns
        pattern = str(chart_logs_dir / f"{tv_symbol}, {tv_timeframe}_*.csv")
        matching_files = glob.glob(pattern)

        df = None
        if matching_files:
            file_path = matching_files[0]  # Take first match
            logger.info(f"Loading {symbol} {timeframe} from {file_path}")
            df = pd.read_csv(file_path)
        else:
            logger.warning(f"No TradingView file found for {symbol} {timeframe}")
            logger.warning(f"Pattern attempted: {pattern}")
            return pd.DataFrame()

        if df is not None and len(df) > 0:
            # Handle TradingView CSV format with Unix timestamps
            if 'time' in df.columns:
                # Convert Unix timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('timestamp', inplace=True)
                df.drop('time', axis=1, inplace=True)
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            else:
                # Assume first column is timestamp
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], unit='s')
                df.set_index(df.columns[0], inplace=True)

            # TradingView exports use lowercase column names
            # Keep only core OHLCV columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            available_cols = {}

            for col in required_cols:
                if col in df.columns:
                    available_cols[col] = df[col]
                elif col.title() in df.columns:
                    available_cols[col] = df[col.title()]
                elif col.upper() in df.columns:
                    available_cols[col] = df[col.upper()]
                else:
                    # Default values for missing columns
                    if col == 'volume':
                        available_cols[col] = 1000000  # Default volume for indices
                    else:
                        available_cols[col] = df.get('close', 100)  # Use close as fallback

            # Create clean DataFrame with only OHLCV
            df = pd.DataFrame(available_cols, index=df.index)

            # Filter by date range
            if not df.empty:
                start_date = pd.to_datetime(start)
                end_date = pd.to_datetime(end)
                df = df[(df.index >= start_date) & (df.index <= end_date)]

            logger.info(f"Loaded {len(df)} bars for {symbol} {timeframe}")
            return df

    except Exception as e:
        logger.error(f"Error loading {symbol} {timeframe}: {e}")
        import traceback
        traceback.print_exc()

    return pd.DataFrame()


def create_synthetic_fallback(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    """Create synthetic data as fallback when real data unavailable"""

    logger.warning(f"Creating synthetic data for {symbol} {timeframe}")

    # Create date range based on timeframe
    freq_map = {'1H': '1H', '4H': '4H', '1D': '1D', '1W': '1W'}
    freq = freq_map.get(timeframe, '1H')

    dates = pd.date_range(start, end, freq=freq)

    # Realistic base prices
    base_prices = {
        'ETH': 2000, 'BTC': 40000, 'SOL': 50, 'XRP': 0.5,
        'DXY': 100, 'WTI': 75, 'GOLD': 1800,
        'US2Y': 4.5, 'US10Y': 4.2, 'VIX': 18, 'MOVE': 120,
        'BTC.D': 55, 'USDT.D': 7, 'ETH.D': 15,
        'TOTAL': 2000000000000, 'TOTAL3': 500000000000
    }

    base_price = base_prices.get(symbol, 100)

    # Add some realistic volatility
    np.random.seed(42)  # Reproducible
    returns = np.random.normal(0, 0.02, len(dates))  # 2% daily vol

    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    # Create OHLCV
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': [1000000 * (1 + np.random.normal(0, 0.5)) for _ in prices]
    }, index=dates)

    # Ensure high >= max(open, close) and low <= min(open, close)
    df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
    df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
    df['volume'] = np.abs(df['volume'])

    return df