#!/usr/bin/env python3
"""
Build Macro Feature Dataset for Regime Classifier Training

Collects macro features from TradingView exports and creates a unified time-series
dataset suitable for training the regime classifier.

Usage:
    python3 bin/build_macro_dataset.py --start 2022-01-01 --end 2025-10-14 --output data/macro/macro_history.parquet
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from engine.io.tradingview_loader import load_tv, RealDataRequiredError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def resample_to_1h(df: pd.DataFrame, method: str = 'last') -> pd.DataFrame:
    """
    Resample DataFrame to 1H frequency

    Args:
        df: DataFrame with datetime index
        method: Resampling method ('last', 'mean', etc.')

    Returns:
        Resampled DataFrame
    """
    if method == 'last':
        return df.resample('1h').last().ffill()
    elif method == 'mean':
        return df.resample('1h').mean().ffill()
    else:
        raise ValueError(f"Unknown resample method: {method}")


def calculate_funding_oi(df_btc: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate funding rate and OI metrics from price data

    Note: This is a simplified calculation. In production, you'd fetch
    real funding rate data from exchange APIs.

    Args:
        df_btc: BTC price DataFrame

    Returns:
        DataFrame with funding and oi columns
    """
    # Placeholder: Calculate proxy metrics from price volatility
    # In production, fetch real funding rates from Binance/Bybit

    df = pd.DataFrame(index=df_btc.index)

    # Proxy funding rate from short-term momentum
    returns_8h = df_btc['close'].pct_change(8).fillna(0)
    df['funding'] = returns_8h.clip(-0.05, 0.05)  # Cap at ±5%

    # Proxy OI from volume (not accurate, but better than nothing)
    df['oi'] = (df_btc['volume'] / df_btc['volume'].rolling(168).mean() - 1.0).fillna(0)

    return df


def calculate_realized_volatility(df: pd.DataFrame, windows: list = [20, 60]) -> pd.DataFrame:
    """
    Calculate realized volatility

    Args:
        df: Price DataFrame with 'close' column
        windows: List of lookback windows (in hours)

    Returns:
        DataFrame with rv_Xh columns
    """
    result = pd.DataFrame(index=df.index)

    returns = np.log(df['close'] / df['close'].shift(1))

    for window in windows:
        rv = returns.rolling(window).std() * np.sqrt(24)  # Annualized
        result[f'rv_{window}d'] = rv.fillna(0.02)  # Default to 2% vol

    return result


def build_macro_dataset(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Build unified macro feature dataset

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with all macro features
    """
    logger.info(f"Building macro dataset from {start_date} to {end_date}")

    # Define data sources (only using what's available in chart_logs)
    sources = {
        'TOTAL': ('TOTAL_1D', 'close'),
        'TOTAL3': ('TOTAL3_1D', 'close'),
        # Add more if TradingView exports are available:
        # 'VIX': Not in current chart_logs
        # 'DXY': Not in current chart_logs
        # 'YIELD_2Y': Not in current chart_logs
        # 'YIELD_10Y': Use TVC_US10Y if available
        # 'BTC.D': Not in current chart_logs
        # 'USDT.D': CRYPTOCAP_USDT.D, 60 available but needs adjustment
    }

    # Try to load optional sources
    optional_sources = [
        ('USDT.D', 'USDTD_4H', 'close'),
        # ('SPY', 'SPY_1D', 'close'),  # Can add if needed
    ]

    for feature, symbol_key, column in optional_sources:
        try:
            df_test = load_tv(symbol_key)
            sources[feature] = (symbol_key, column)
            logger.info(f"  Optional feature {feature} is available")
        except (RealDataRequiredError, KeyError):
            logger.info(f"  Optional feature {feature} not available, will use defaults")
            pass

    # Load BTC data for funding/OI calculation
    logger.info("Loading BTC data...")
    try:
        df_btc = load_tv('BTC_1H')
        # Index is already set by load_tv()
    except RealDataRequiredError as e:
        logger.error(f"Failed to load BTC data: {e}")
        return pd.DataFrame()

    # Filter date range
    df_btc = df_btc[(df_btc.index >= start_date) & (df_btc.index <= end_date)]

    # Initialize result DataFrame with 1H frequency (tz-aware to match TradingView data)
    date_range = pd.date_range(start=start_date, end=end_date, freq='1h', tz='UTC')
    df_macro = pd.DataFrame(index=date_range)
    df_macro.index.name = 'timestamp'

    # Load each macro source
    for feature, source_info in sources.items():
        if source_info is None:
            continue

        symbol_key, column = source_info
        logger.info(f"Loading {feature} from {symbol_key}...")

        try:
            df_source = load_tv(symbol_key)
            # Index is already set by load_tv()
            df_source = df_source[(df_source.index >= start_date) & (df_source.index <= end_date)]

            # Resample to 1H and forward-fill
            series_1h = resample_to_1h(df_source[[column]], method='last')[column]

            # Merge into main DataFrame
            df_macro[feature] = series_1h

        except RealDataRequiredError as e:
            logger.warning(f"Skipping {feature}: {e}")
            df_macro[feature] = np.nan
        except Exception as e:
            logger.error(f"Error loading {feature}: {e}")
            df_macro[feature] = np.nan

    # Calculate derived features and add defaults for missing data
    logger.info("Calculating derived features and adding defaults...")

    # TOTAL2 = TOTAL - TOTAL3
    if 'TOTAL' in df_macro.columns and 'TOTAL3' in df_macro.columns:
        df_macro['TOTAL2'] = df_macro['TOTAL'] - df_macro['TOTAL3']
        logger.info("  TOTAL2: Calculated from TOTAL - TOTAL3")
    else:
        df_macro['TOTAL2'] = np.nan

    # Add default values for features not available in chart_logs
    # These defaults represent "neutral" market conditions
    defaults = {
        'VIX': 20.0,          # Neutral volatility
        'DXY': 102.0,         # Mid-range dollar index
        'MOVE': 100.0,        # Neutral bond volatility
        'YIELD_2Y': 4.0,      # Current range
        'YIELD_10Y': 4.2,     # Current range
        'BTC.D': 55.0,        # Mid-range dominance
        'USDT.D': 6.5,        # Mid-range USDT dominance
    }

    for feature, default_val in defaults.items():
        if feature not in df_macro.columns or df_macro[feature].isna().all():
            df_macro[feature] = default_val
            logger.info(f"  {feature}: Using default value {default_val} (not in chart_logs)")

    # Funding and OI from BTC price
    logger.info("  Funding & OI: Calculating from BTC price data")
    df_funding_oi = calculate_funding_oi(df_btc)
    df_macro = df_macro.join(df_funding_oi, how='left')

    # Realized volatility
    logger.info("  Realized Volatility: Calculating from BTC returns")
    df_rv = calculate_realized_volatility(df_btc, windows=[20, 60])
    df_macro = df_macro.join(df_rv, how='left')

    # Forward-fill missing values (macro data updates slowly)
    logger.info("Forward-filling missing values...")
    df_macro = df_macro.ffill()

    # Drop rows with too many NaNs (beginning of series)
    initial_rows = len(df_macro)
    df_macro = df_macro.dropna(thresh=len(df_macro.columns) * 0.5)  # Require 50% non-NaN
    final_rows = len(df_macro)
    logger.info(f"Dropped {initial_rows - final_rows} rows with excessive NaNs")

    # Summary
    logger.info(f"\nDataset Summary:")
    logger.info(f"  Date range: {df_macro.index[0]} to {df_macro.index[-1]}")
    logger.info(f"  Total rows: {len(df_macro)}")
    logger.info(f"  Columns: {list(df_macro.columns)}")

    # NaN counts
    nan_counts = df_macro.isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"\nRemaining NaN values:")
        for col, count in nan_counts[nan_counts > 0].items():
            pct = count / len(df_macro) * 100
            logger.warning(f"  {col}: {count} ({pct:.1f}%)")

    return df_macro


def main():
    parser = argparse.ArgumentParser(description="Build macro feature dataset")
    parser.add_argument('--start', default='2022-01-01', help="Start date (YYYY-MM-DD)")
    parser.add_argument('--end', default='2025-10-14', help="End date (YYYY-MM-DD)")
    parser.add_argument('--output', default='data/macro/macro_history.parquet', help="Output path")

    args = parser.parse_args()

    print("="*70)
    print("🔧 Bull Machine v1.9 - Macro Dataset Builder")
    print("="*70)

    # Build dataset
    df_macro = build_macro_dataset(args.start, args.end)

    if df_macro.empty:
        logger.error("Failed to build macro dataset")
        return 1

    # Save to parquet
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_macro.reset_index().to_parquet(output_path, index=False)
    logger.info(f"\n✅ Macro dataset saved to {output_path}")

    # Also save as CSV for inspection
    csv_path = output_path.with_suffix('.csv')
    df_macro.to_csv(csv_path)
    logger.info(f"✅ CSV version saved to {csv_path}")

    print("\n" + "="*70)
    print("✅ Macro dataset build complete!")
    print(f"   Rows: {len(df_macro)}")
    print(f"   Features: {len(df_macro.columns)}")
    print(f"   Output: {output_path}")
    print("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
