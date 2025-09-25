"""
Bull Machine v1.5.0 - Data I/O with Live Data Stub
Handles both historical CSV data and live data feeds (when enabled).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import os
from datetime import datetime, timedelta


def load_live_data(symbol: str, timeframe: str, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load live market data (stub implementation).

    When live_data feature flag is enabled, this would connect to
    exchanges like Alpaca, CCXT, or other data providers.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSD', 'ETHUSD')
        timeframe: Timeframe string (e.g., '1H', '4H', '1D')
        config: Configuration with feature flags

    Returns:
        DataFrame with OHLCV data

    Note:
        This is a stub implementation. In production, this would:
        - Connect to WebSocket feeds
        - Handle authentication
        - Manage rate limits
        - Provide real-time data
    """
    if not config.get("features", {}).get("live_data", False):
        # Fallback to CSV when live data is disabled
        return load_csv_data(symbol, timeframe, config)

    # Live data stub - would be replaced with actual implementation
    print(f"⚠️  Live data requested for {symbol} {timeframe} but stub implementation active")
    print("   In production, this would connect to:")
    print("   - Alpaca Markets API")
    print("   - CCXT exchange connectors")
    print("   - WebSocket real-time feeds")

    # For now, return empty DataFrame or fallback to CSV
    try:
        return load_csv_data(symbol, timeframe, config)
    except FileNotFoundError:
        # Generate minimal dummy data if no CSV available
        return _generate_dummy_data(symbol, timeframe)


def load_csv_data(symbol: str, timeframe: str, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load historical data from CSV files.

    CSV path conventions:
    - data/{symbol}_{timeframe}.csv
    - chartlogs/{symbol}_{timeframe}.csv

    Args:
        symbol: Trading symbol
        timeframe: Timeframe string
        config: Configuration

    Returns:
        DataFrame with OHLCV data
    """
    # Try multiple path conventions
    possible_paths = [
        f"data/{symbol}_{timeframe}.csv",
        f"chartlogs/{symbol}_{timeframe}.csv",
        f"data/{symbol.upper()}_{timeframe}.csv",
        f"data/{symbol.lower()}_{timeframe}.csv"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return _standardize_dataframe(df)

    raise FileNotFoundError(f"No CSV data found for {symbol} {timeframe}")


def _standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize DataFrame columns and format.

    Args:
        df: Raw DataFrame from CSV

    Returns:
        Standardized DataFrame with required columns
    """
    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Handle timestamp
    if 'timestamp' not in df.columns:
        if 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
        else:
            # Generate timestamps if none exist
            df['timestamp'] = pd.date_range(
                start='2020-01-01',
                periods=len(df),
                freq='H'
            )

    # Handle volume
    if 'volume' not in df.columns:
        # Try alternative volume column names
        volume_candidates = ['Volume', 'vol', 'BUY+SELL V', 'Total Buy Volume']
        for candidate in volume_candidates:
            if candidate in df.columns:
                df['volume'] = df[candidate]
                break
        else:
            # Default volume if none available
            df['volume'] = 100000

    # Select and order columns
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()


def _generate_dummy_data(symbol: str, timeframe: str, bars: int = 200) -> pd.DataFrame:
    """
    Generate dummy OHLCV data for testing when no real data is available.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        bars: Number of bars to generate

    Returns:
        DataFrame with dummy OHLCV data
    """
    # Base price depending on symbol
    base_prices = {
        'BTCUSD': 45000,
        'ETHUSD': 3000,
        'SPY': 450
    }

    base_price = base_prices.get(symbol, 100)

    # Generate random walk
    np.random.seed(42)  # Reproducible dummy data
    returns = np.random.normal(0, 0.02, bars)  # 2% daily volatility
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC from prices
    data = []
    for i in range(bars):
        price = prices[i]
        volatility = price * 0.01  # 1% intraday volatility

        open_price = price * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, price) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, price) * (1 - abs(np.random.normal(0, 0.01)))
        close_price = price

        volume = np.random.lognormal(12, 0.5)  # Realistic volume distribution

        data.append({
            'timestamp': datetime.now() - timedelta(hours=bars-i),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': round(volume)
        })

    return pd.DataFrame(data)


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data quality and completeness.

    Args:
        df: DataFrame to validate

    Returns:
        Dict containing validation results
    """
    results = {
        "valid": True,
        "issues": [],
        "stats": {}
    }

    # Check for required columns
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        results["valid"] = False
        results["issues"].append(f"Missing columns: {missing_cols}")

    if not results["valid"]:
        return results

    # Check for NaN values
    nan_counts = df[required_cols].isnull().sum()
    if nan_counts.sum() > 0:
        results["issues"].append(f"NaN values found: {nan_counts.to_dict()}")

    # Check OHLC logic
    invalid_ohlc = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    ).sum()

    if invalid_ohlc > 0:
        results["issues"].append(f"Invalid OHLC relationships: {invalid_ohlc} bars")

    # Check for negative prices
    negative_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()
    if negative_prices > 0:
        results["issues"].append(f"Negative or zero prices: {negative_prices} bars")

    # Data statistics
    results["stats"] = {
        "total_bars": len(df),
        "date_range": {
            "start": df['timestamp'].min().isoformat() if 'timestamp' in df.columns else None,
            "end": df['timestamp'].max().isoformat() if 'timestamp' in df.columns else None
        },
        "price_range": {
            "min": float(df[['open', 'high', 'low', 'close']].min().min()),
            "max": float(df[['open', 'high', 'low', 'close']].max().max())
        },
        "avg_volume": float(df['volume'].mean()) if 'volume' in df.columns else 0
    }

    return results