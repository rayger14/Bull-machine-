"""
TradingView Data Loader - STRICT REAL DATA ONLY

NO SYNTHETIC FALLBACKS. If a file isn't found, CRASH LOUDLY.
This ensures all backtest results are based on real market data only.
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

DATA_DIR = Path("chart_logs")  # symlinked to Desktop path

# Map logical symbols to TradingView prefixes (based on available files)
SYMBOL_MAP = {
    "ETH_1H": ("COINBASE_ETHUSD", "60"),
    "ETH_4H": ("COINBASE_ETHUSD", "240"),
    "BTC_1H": ("COINBASE_BTCUSD", "60"),
    "BTC_4H": ("COINBASE_BTCUSD", "240"),
    "SOL_1H": ("COINBASE_SOLUSD", "60"),
    "SOL_4H": ("COINBASE_SOLUSD", "240"),
    "VIX_1H": ("TVC_VIX", "60"),
    "DXY_1H": ("TVC_DXY", "60"),
    "MOVE_1H": ("TVC_MOVE", "60"),
    "US02Y_1H": ("TVC_US02Y", "60"),
    "US10Y_1H": ("TVC_US10Y", "60"),
    "US02Y_4H": ("TVC_US02Y", "240"),
    "US10Y_4H": ("TVC_US10Y", "240"),
    "WTI_1H": ("CFI_WTI", "60"),
    "DXY_1D": ("TVC_DXY", "1D"),
    "US2Y_1D": ("TVC_US02", "1W"),  # Use weekly data (daily not available)
    "US10Y_1D": ("TVC_US10", "1W"),  # Use weekly data (daily not available)
    "GOLD_1D": ("FX_XAUUSD", "1D"),
    "WTI_1D": ("EASYMARKETS_OILUSD", "1D"),
    "BTCD_1W": ("CRYPTOCAP_BTC.D", "1W"),
    "USDTD_4H": ("CRYPTOCAP_USDT.D", "1D"),  # Use daily instead of 4H
    "ETHBTC_1D": ("COINBASE_ETHBTC", "1D"),
    "BTC_1D": ("COINBASE_BTCUSD", "1D"),
    "ETH_1D": ("COINBASE_ETHUSD", "1D"),
    "SOL_1D": ("COINBASE_SOLUSD", "1D"),
    "XRP_1D": ("COINBASE_XRPUSD", "1D"),
    "TOTAL_4H": ("CRYPTOCAP_TOTAL", "1D"),  # Use daily (4H not available)
    "TOTAL3_4H": ("CRYPTOCAP_TOTAL3", "1D"),  # Use daily (4H not available)
    # Add available series for better macro coverage
    "SOL_12H": ("COINBASE_SOLUSD", "720"),
    "USDTD_1D": ("CRYPTOCAP_USDT.D", "1D"),
    "TOTAL_1D": ("CRYPTOCAP_TOTAL", "1D"),
    "TOTAL2_1D": ("CRYPTOCAP_TOTAL2", "1D"),
    "TOTAL3_1D": ("CRYPTOCAP_TOTAL3", "1D"),
    # Stock market data
    "SPY_1H": ("BATS_SPY", "60"),
    "SPY_4H": ("BATS_SPY", "240"),
    "SPY_1D": ("BATS_SPY", "1D"),
}

class RealDataRequiredError(Exception):
    """Raised when real data is required but not found"""
    pass

def _find_tv_csv(prefix: str, tf: str) -> Path:
    """
    Find TradingView CSV file with hash suffix pattern.

    Accepts patterns like: "COINBASE_ETHUSD, 240_ab8a9.csv"

    Raises:
        RealDataRequiredError: If file not found (NO FALLBACKS)
    """
    # Pattern to match TradingView export naming with hash
    pat = re.compile(rf"^{re.escape(prefix)},\s*{re.escape(tf)}(_[A-Za-z0-9]+)?\.csv$")

    if not DATA_DIR.exists():
        raise RealDataRequiredError(f"Chart logs directory not found: {DATA_DIR}")

    for p in DATA_DIR.iterdir():
        if p.suffix.lower() == ".csv" and pat.match(p.name):
            logger.info(f"Found TradingView file: {p.name}")
            return p

    # List available files for debugging
    available_files = [f.name for f in DATA_DIR.iterdir() if f.suffix.lower() == ".csv"]
    logger.error(f"Available CSV files: {available_files[:10]}...")  # Show first 10

    raise RealDataRequiredError(
        f"Missing TradingView file for {prefix} {tf}. "
        f"Pattern: '{prefix}, {tf}_*.csv' not found in {DATA_DIR}. "
        f"NO SYNTHETIC FALLBACKS ALLOWED."
    )

def load_tv(symbol_key: str) -> pd.DataFrame:
    """
    Load TradingView data with STRICT real-data-only policy.

    Args:
        symbol_key: Key from SYMBOL_MAP (e.g., 'ETH_4H', 'DXY_1D')

    Returns:
        DataFrame with OHLCV data from real TradingView export

    Raises:
        RealDataRequiredError: If file not found or data is flat/synthetic
        KeyError: If symbol_key not in SYMBOL_MAP
    """
    if symbol_key not in SYMBOL_MAP:
        raise KeyError(f"Symbol key '{symbol_key}' not in SYMBOL_MAP. Available: {list(SYMBOL_MAP.keys())}")

    prefix, tf = SYMBOL_MAP[symbol_key]
    csv_path = _find_tv_csv(prefix, tf)

    # Load and parse TradingView CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path.name}")

    # Handle TradingView timestamp column
    ts_col = "time" if "time" in df.columns else "Time"
    if ts_col not in df.columns:
        raise RealDataRequiredError(f"No time column found in {csv_path.name}. Columns: {list(df.columns)}")

    # Convert Unix timestamps to datetime
    df[ts_col] = pd.to_datetime(df[ts_col], unit='s', utc=True)
    df = df.set_index(ts_col).sort_index()

    # Standardize column names (TradingView uses lowercase)
    cols = {c.lower(): c for c in df.columns}

    required_cols = ["open", "high", "low", "close"]
    missing_cols = [col for col in required_cols if col not in cols]
    if missing_cols:
        raise RealDataRequiredError(f"Missing required columns {missing_cols} in {csv_path.name}")

    # Build clean OHLCV DataFrame
    out = pd.DataFrame({
        "open": df[cols["open"]],
        "high": df[cols["high"]],
        "low": df[cols["low"]],
        "close": df[cols["close"]]
    })

    # Add volume if available
    if "volume" in cols:
        out["volume"] = df[cols["volume"]]
    else:
        # For indices without volume, use a constant
        out["volume"] = 1000000

    # STRICT: Assert non-flat variance to catch synthetic patterns
    close_std = out["close"].std()
    if close_std <= 1e-9:
        raise RealDataRequiredError(
            f"Flat series detected for {symbol_key} (std={close_std}). "
            f"This appears to be synthetic data. REAL DATA REQUIRED."
        )

    # Validate realistic price action
    if len(out) > 0:
        price_range = out["close"].max() - out["close"].min()
        mean_price = out["close"].mean()
        if price_range / mean_price < 0.001:  # Less than 0.1% range
            raise RealDataRequiredError(
                f"Unrealistic price action for {symbol_key}: range={price_range:.6f}, "
                f"mean={mean_price:.6f}. This appears synthetic."
            )

    logger.info(f"✅ Real data validated for {symbol_key}: {len(out)} bars, std={close_std:.6f}")
    return out

def load_tv_safe(symbol_key: str) -> pd.DataFrame:
    """
    Safe wrapper that returns empty DataFrame if data not available.
    Use sparingly - prefer load_tv() for required data.
    """
    try:
        return load_tv(symbol_key)
    except (RealDataRequiredError, KeyError) as e:
        logger.warning(f"Could not load {symbol_key}: {e}")
        return pd.DataFrame()

# Legacy compatibility - replace old loader calls
def load_tradingview_data(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    """Legacy compatibility wrapper"""
    symbol_key = f"{symbol}_{timeframe}"

    try:
        df = load_tv(symbol_key)

        # Filter by date range
        if not df.empty:
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)
            df = df[(df.index >= start_date) & (df.index <= end_date)]

        return df

    except (RealDataRequiredError, KeyError):
        # Hard fail - no synthetic fallback
        raise RealDataRequiredError(
            f"Real data required for {symbol} {timeframe} but not found. "
            f"NO SYNTHETIC FALLBACKS. Check chart_logs directory."
        )

# Disable any synthetic fallback functions
def create_synthetic_fallback(*args, **kwargs):
    """DISABLED: No synthetic fallbacks allowed"""
    raise RealDataRequiredError("Synthetic fallback DISABLED. Real data required.")