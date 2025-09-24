#!/usr/bin/env python3
"""
Create multi-timeframe BTC data for v1.4.2 backtest
Generate realistic 1H, 4H data from daily OHLCV
"""

import pandas as pd
import numpy as np
from pathlib import Path


def create_intraday_from_daily(daily_df, timeframe_hours=1):
    """Create realistic intraday data from daily OHLCV."""
    np.random.seed(42)  # For reproducible results

    intraday_data = []
    bars_per_day = 24 // timeframe_hours

    for i, row in daily_df.iterrows():
        daily_open = row["open"]
        daily_high = row["high"]
        daily_low = row["low"]
        daily_close = row["close"]
        daily_volume = row["volume"]

        # Generate intraday progression
        intraday_prices = []
        current_price = daily_open

        # Create realistic price path through the day
        target_range = daily_high - daily_low
        volume_per_bar = daily_volume / bars_per_day

        for bar in range(bars_per_day):
            # Calculate progression through day
            day_progress = bar / (bars_per_day - 1) if bars_per_day > 1 else 0

            # Target price based on daily close
            target_price = daily_open + (daily_close - daily_open) * day_progress

            # Add some realistic noise
            noise = np.random.normal(0, target_range * 0.02)  # 2% of daily range
            target_price += noise

            # Ensure we stay within daily range (mostly)
            if target_price > daily_high * 1.001:  # Allow tiny overshoot
                target_price = daily_high - np.random.uniform(0, target_range * 0.1)
            if target_price < daily_low * 0.999:  # Allow tiny undershoot
                target_price = daily_low + np.random.uniform(0, target_range * 0.1)

            # Create OHLC for this bar
            bar_range = target_range * np.random.uniform(0.05, 0.15)  # 5-15% of daily range

            bar_open = current_price
            bar_close = target_price
            bar_high = max(bar_open, bar_close) + bar_range * np.random.uniform(0.2, 0.8)
            bar_low = min(bar_open, bar_close) - bar_range * np.random.uniform(0.2, 0.8)

            # Ensure high/low make sense
            bar_high = max(bar_high, bar_open, bar_close)
            bar_low = min(bar_low, bar_open, bar_close)

            # Constrain to daily range
            bar_high = min(bar_high, daily_high)
            bar_low = max(bar_low, daily_low)

            # Volume with some variation
            bar_volume = volume_per_bar * np.random.uniform(0.3, 2.0)  # Vary volume

            timestamp = pd.Timestamp(i) + pd.Timedelta(hours=bar * timeframe_hours)

            intraday_data.append(
                {
                    "timestamp": timestamp,
                    "open": round(bar_open, 2),
                    "high": round(bar_high, 2),
                    "low": round(bar_low, 2),
                    "close": round(bar_close, 2),
                    "volume": round(bar_volume, 0),
                }
            )

            current_price = bar_close

    return pd.DataFrame(intraday_data)


def enhance_volume_patterns(df):
    """Add realistic volume patterns and clustering."""
    df = df.copy()

    # Add volume clustering around key levels
    for i in range(len(df)):
        price_change_pct = abs(df.iloc[i]["close"] - df.iloc[i]["open"]) / df.iloc[i]["open"]

        # Higher volume on bigger moves
        if price_change_pct > 0.02:  # 2%+ moves
            df.iloc[i, df.columns.get_loc("volume")] *= np.random.uniform(1.5, 3.0)
        elif price_change_pct > 0.01:  # 1%+ moves
            df.iloc[i, df.columns.get_loc("volume")] *= np.random.uniform(1.2, 2.0)

        # Add some random volume spikes
        if np.random.random() < 0.05:  # 5% chance
            df.iloc[i, df.columns.get_loc("volume")] *= np.random.uniform(2.0, 4.0)

    return df


def main():
    """Create multi-timeframe BTC data."""
    print("📊 Creating Multi-Timeframe BTC Data for v1.4.2 Backtest")
    print("=" * 60)

    # Load daily data
    daily_df = pd.read_csv("btc_daily_clean.csv")
    daily_df["timestamp"] = pd.to_datetime(daily_df["timestamp"], unit="s")
    daily_df = daily_df.set_index("timestamp")

    print(f"Loaded {len(daily_df)} days of BTC data")
    print(f"Date range: {daily_df.index[0]} to {daily_df.index[-1]}")

    # Create data directory
    data_dir = Path("data/btc_multiframe")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate different timeframes
    timeframes = {
        "1H": 1,
        "4H": 4,
        "1D": 24,  # Just copy daily for consistency
    }

    for tf_name, hours in timeframes.items():
        print(f"\n🔄 Generating {tf_name} data...")

        if tf_name == "1D":
            # Use original daily data with enhancements
            tf_df = daily_df.reset_index()
            tf_df = enhance_volume_patterns(tf_df)
        else:
            # Generate intraday data
            tf_df = create_intraday_from_daily(daily_df, hours)
            tf_df = enhance_volume_patterns(tf_df)

        # Save to file
        filename = f"BTC_USD_{tf_name}.csv"
        filepath = data_dir / filename
        tf_df.to_csv(filepath, index=False)

        print(f"✅ Created {filename}: {len(tf_df)} bars")
        print(f"   Volume range: {tf_df['volume'].min():.0f} - {tf_df['volume'].max():.0f}")
        print(f"   Price range: ${tf_df['low'].min():.0f} - ${tf_df['high'].max():.0f}")

    print(f"\n📁 All data files saved to: {data_dir}")
    print("🚀 Ready for v1.4.2 multi-timeframe backtest!")


if __name__ == "__main__":
    main()
