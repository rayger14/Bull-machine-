#!/usr/bin/env python3
"""
Verify and align downloaded data to strict hourly grid
Detects gaps and ensures clean UTC index
"""
import pandas as pd
import sys
import argparse

def verify_alignment(path: str, interval: str = "1H"):
    """
    Check data alignment and gaps

    Args:
        path: CSV file path
        interval: Expected interval (1H, 4H, 1D)
    """
    print(f"📊 Verifying: {path}")
    print("=" * 70)

    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")

    print(f"Raw data:")
    print(f"  Bars: {len(df):,}")
    print(f"  Start: {df.index.min()}")
    print(f"  End: {df.index.max()}")
    print(f"  Duration: {(df.index.max() - df.index.min()).days} days")

    # Resample to strict interval to spot gaps
    strict = df.resample(interval).agg({
        "open":"first",
        "high":"max",
        "low":"min",
        "close":"last",
        "volume":"sum"
    })

    missing = strict[strict["open"].isna()]

    print(f"\nAlignment check (strict {interval}):")
    print(f"  Expected bars: {len(strict):,}")
    print(f"  Missing candles: {len(missing):,}")

    if len(missing) > 0:
        print(f"\n⚠️  Gap detected! Missing {len(missing)} candles:")
        print(missing.head(10))

        # Show gap periods
        if len(missing) > 0:
            gap_start = missing.index[0]
            gap_end = missing.index[-1]
            print(f"\n  First gap: {gap_start}")
            print(f"  Last gap: {gap_end}")
    else:
        print("\n✅ Perfect alignment - no gaps detected!")

    # Check timezone
    if df.index.tz is None:
        print("\n⚠️  WARNING: No timezone info (assuming UTC)")
    else:
        print(f"\n✅ Timezone: {df.index.tz}")

    # Basic stats
    print(f"\nPrice stats:")
    print(f"  Open: ${df['open'].iloc[0]:,.2f}")
    print(f"  Close: ${df['close'].iloc[-1]:,.2f}")
    print(f"  Min: ${df['low'].min():,.2f}")
    print(f"  Max: ${df['high'].max():,.2f}")
    print(f"  Avg Volume: {df['volume'].mean():,.0f}")

def main():
    ap = argparse.ArgumentParser(description="Verify and align historical data")
    ap.add_argument("path", help="CSV file to verify")
    ap.add_argument("--interval", default="1H", help="Expected interval (1H, 4H, 1D)")
    args = ap.parse_args()

    verify_alignment(args.path, args.interval)

if __name__ == "__main__":
    main()
