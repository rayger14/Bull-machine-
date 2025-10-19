"""
Process TOTAL/TOTAL2/TOTAL3 Market Cap Data from TradingView

Loads real crypto market cap data from TradingView exports and prepares it
for integration with macro feature store for regime classification.

REAL DATA ONLY - uses existing TradingView CSV files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.io.tradingview_loader import load_tv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_total_marketcap_data() -> pd.DataFrame:
    """
    Load TOTAL, TOTAL2, TOTAL3 market cap data from TradingView exports

    Returns:
        DataFrame with columns: timestamp, TOTAL, TOTAL2, TOTAL3, BTC.D
    """
    logger.info("=" * 70)
    logger.info("Loading Crypto Market Cap Data (TOTAL/TOTAL2/TOTAL3)")
    logger.info("=" * 70)

    # Load from TradingView data (daily resolution)
    total_df = load_tv("TOTAL_1D")
    total2_df = load_tv("TOTAL2_1D")
    total3_df = load_tv("TOTAL3_1D")

    logger.info(f"  TOTAL:  {len(total_df)} bars")
    logger.info(f"  TOTAL2: {len(total2_df)} bars")
    logger.info(f"  TOTAL3: {len(total3_df)} bars")

    # Use close prices for market cap values
    df = pd.DataFrame({
        'TOTAL': total_df['close'],
        'TOTAL2': total2_df['close'],
        'TOTAL3': total3_df['close']
    })

    # Calculate BTC.D (BTC Dominance) from TOTAL and TOTAL2
    # TOTAL = BTC + TOTAL2
    # BTC.D = (TOTAL - TOTAL2) / TOTAL * 100
    df['BTC_mcap'] = df['TOTAL'] - df['TOTAL2']
    df['BTC.D'] = (df['BTC_mcap'] / df['TOTAL']) * 100

    # Calculate ETH.D from TOTAL2 and TOTAL3
    # TOTAL2 = ETH + TOTAL3
    # ETH.D = (TOTAL2 - TOTAL3) / TOTAL * 100
    df['ETH_mcap'] = df['TOTAL2'] - df['TOTAL3']
    df['ETH.D'] = (df['ETH_mcap'] / df['TOTAL']) * 100

    # Convert to billions for readability
    df['TOTAL_billions'] = df['TOTAL'] / 1e9
    df['TOTAL2_billions'] = df['TOTAL2'] / 1e9
    df['TOTAL3_billions'] = df['TOTAL3'] / 1e9

    logger.info(f"\n✅ Market cap data loaded: {len(df)} daily records")
    logger.info(f"   Date range: {df.index.min()} to {df.index.max()}")

    # Show latest values
    logger.info(f"\n   Latest snapshot ({df.index[-1].date()}):")
    logger.info(f"   TOTAL:  ${df['TOTAL_billions'].iloc[-1]:.1f}B")
    logger.info(f"   TOTAL2: ${df['TOTAL2_billions'].iloc[-1]:.1f}B (ex-BTC)")
    logger.info(f"   TOTAL3: ${df['TOTAL3_billions'].iloc[-1]:.1f}B (ex-BTC, ex-ETH)")
    logger.info(f"   BTC.D:  {df['BTC.D'].iloc[-1]:.2f}%")
    logger.info(f"   ETH.D:  {df['ETH.D'].iloc[-1]:.2f}%")

    # Validate data quality
    assert df['BTC.D'].min() > 0, "BTC.D should be positive"
    assert df['BTC.D'].max() < 100, "BTC.D should be less than 100%"
    assert df['TOTAL'].min() > df['TOTAL2'].min(), "TOTAL should be greater than TOTAL2"
    assert df['TOTAL2'].min() > df['TOTAL3'].min(), "TOTAL2 should be greater than TOTAL3"

    logger.info("\n   Data quality checks passed ✅")

    return df


def upsample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Upsample daily data to hourly using forward fill

    Args:
        df: Daily DataFrame with index

    Returns:
        Hourly DataFrame
    """
    logger.info("\nUpsampling to hourly resolution...")

    # Resample to hourly and forward fill
    df_hourly = df.resample('1H').ffill()

    logger.info(f"  ✅ Upsampled to {len(df_hourly)} hourly records")

    return df_hourly


def save_to_parquet(df: pd.DataFrame, output_path: str):
    """Save processed market cap data to parquet"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Reset index to save timestamp as column
    df_out = df.reset_index()
    df_out = df_out.rename(columns={'index': 'timestamp'})

    df_out.to_parquet(output_path, index=False)

    logger.info(f"\n💾 Saved to: {output_path}")
    logger.info(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
    logger.info(f"   Columns: {list(df_out.columns)}")


def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute regime classification features from market cap data

    Features:
    - TOTAL2/TOTAL ratio (altcoin strength)
    - TOTAL2 20-day momentum
    - TOTAL3/TOTAL2 ratio (small-cap strength)
    - BTC.D 20-day change

    Args:
        df: DataFrame with TOTAL, TOTAL2, TOTAL3, BTC.D

    Returns:
        DataFrame with regime features added
    """
    logger.info("\nComputing regime classification features...")

    # TOTAL2/TOTAL ratio (altcoin market cap as % of total)
    df['altcoin_ratio'] = df['TOTAL2'] / df['TOTAL']

    # TOTAL2 momentum (20-day rate of change)
    df['total2_momentum_20d'] = df['TOTAL2'].pct_change(20)

    # TOTAL3/TOTAL2 ratio (small-cap strength)
    df['smallcap_ratio'] = df['TOTAL3'] / df['TOTAL2']

    # BTC.D change (20-day)
    df['btcd_change_20d'] = df['BTC.D'].diff(20)

    # TOTAL/TOTAL2 divergence (when BTC.D rises but TOTAL2 grows faster)
    # Positive = risk-on altcoin season, Negative = BTC flight-to-quality
    df['total_total2_divergence'] = df['total2_momentum_20d'] - (df['btcd_change_20d'] / 100)

    logger.info("  Features computed:")
    logger.info(f"    - altcoin_ratio: {df['altcoin_ratio'].iloc[-1]:.3f}")
    logger.info(f"    - total2_momentum_20d: {df['total2_momentum_20d'].iloc[-1]:.2%}")
    logger.info(f"    - smallcap_ratio: {df['smallcap_ratio'].iloc[-1]:.3f}")
    logger.info(f"    - btcd_change_20d: {df['btcd_change_20d'].iloc[-1]:.2f}pp")
    logger.info(f"    - total_total2_divergence: {df['total_total2_divergence'].iloc[-1]:.3f}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process TOTAL/TOTAL2/TOTAL3 market cap data")
    parser.add_argument('--output', type=str, default='data/macro/crypto_marketcap.parquet',
                        help='Output parquet path')
    parser.add_argument('--hourly', action='store_true',
                        help='Upsample to hourly resolution')

    args = parser.parse_args()

    # Load market cap data
    df = load_total_marketcap_data()

    # Compute regime features
    df = compute_regime_features(df)

    # Upsample to hourly if requested
    if args.hourly:
        df = upsample_to_hourly(df)

    # Save to parquet
    save_to_parquet(df, args.output)

    # Show summary stats
    print("\n" + "=" * 70)
    print("MARKET CAP SUMMARY STATISTICS")
    print("=" * 70)
    print(df[['TOTAL_billions', 'TOTAL2_billions', 'TOTAL3_billions',
              'BTC.D', 'ETH.D', 'altcoin_ratio', 'total2_momentum_20d']].describe())

    print("\n" + "=" * 70)
    print("✅ Processing complete!")
    print("=" * 70)
