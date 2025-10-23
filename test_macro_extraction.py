#!/usr/bin/env python3
"""
Debug macro extraction to see why it's returning constant values
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from engine.context.loader import load_macro_data
from engine.exits.macro_echo import analyze_macro_echo

# Load macro data
macro_data = load_macro_data(asset_type='crypto')

print("=" * 80)
print("Macro Data Loaded:")
print("=" * 80)
for symbol in ['DXY', 'US10Y', 'WTI', 'VIX']:
    if symbol in macro_data and not macro_data[symbol].empty:
        df = macro_data[symbol]
        print(f"\n{symbol}: {len(df)} rows")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Sample (last 5): {df.tail(5)[['timestamp', 'value']].to_dict('records')}")
    else:
        print(f"\n{symbol}: NOT FOUND")

# Test extraction function
print("\n" + "=" * 80)
print("Testing extraction for Sept 5, 2024")
print("=" * 80)

test_date = pd.Timestamp('2024-09-05', tz='UTC')
lookback_start = test_date - pd.Timedelta(days=7)

def extract_macro_series(symbol: str, lookback_start_ts, end_ts) -> pd.Series:
    """Extract macro series for lookback window."""
    if symbol not in macro_data or macro_data[symbol].empty:
        defaults = {'DXY': 100.0, 'US10Y': 4.0, 'WTI': 75.0, 'VIX': 18.0}
        print(f"  {symbol}: Using default (not in macro_data)")
        return pd.Series([defaults.get(symbol, 50.0)])

    df = macro_data[symbol]

    # Convert timestamps to tz-naive for comparison
    lookback_naive = lookback_start_ts.replace(tzinfo=None) if hasattr(lookback_start_ts, 'tzinfo') else lookback_start_ts
    end_naive = end_ts.replace(tzinfo=None) if hasattr(end_ts, 'tzinfo') else end_ts

    print(f"\n  {symbol}:")
    print(f"    Lookback window: {lookback_naive} to {end_naive}")

    # Filter to lookback window
    window = df[(df['timestamp'] >= lookback_naive) & (df['timestamp'] <= end_naive)]

    print(f"    Window size: {len(window)} rows")

    if window.empty:
        # Fallback to most recent value
        recent = df[df['timestamp'] <= end_naive]
        print(f"    Empty window, checking recent: {len(recent)} rows <= {end_naive}")
        if not recent.empty:
            print(f"    Using most recent: {recent.iloc[-1]['value']} at {recent.iloc[-1]['timestamp']}")
            return pd.Series([recent.iloc[-1]['value']])
        defaults = {'DXY': 100.0, 'US10Y': 4.0, 'WTI': 75.0, 'VIX': 18.0}
        print(f"    No recent data, using default: {defaults.get(symbol, 50.0)}")
        return pd.Series([defaults.get(symbol, 50.0)])

    print(f"    Window values: {window['value'].tolist()}")
    return window['value'].reset_index(drop=True)

# Extract series
dxy_series = extract_macro_series('DXY', lookback_start, test_date)
yields_series = extract_macro_series('US10Y', lookback_start, test_date)
oil_series = extract_macro_series('WTI', lookback_start, test_date)
vix_series = extract_macro_series('VIX', lookback_start, test_date)

print("\n" + "=" * 80)
print("Extracted Series:")
print("=" * 80)
print(f"DXY: {len(dxy_series)} values = {dxy_series.tolist()}")
print(f"US10Y: {len(yields_series)} values = {yields_series.tolist()}")
print(f"WTI: {len(oil_series)} values = {oil_series.tolist()}")
print(f"VIX: {len(vix_series)} values = {vix_series.tolist()}")

# Call analyze_macro_echo
print("\n" + "=" * 80)
print("Calling analyze_macro_echo():")
print("=" * 80)

macro_echo = analyze_macro_echo({
    'DXY': dxy_series,
    'YIELDS_10Y': yields_series,
    'OIL': oil_series,
    'VIX': vix_series
}, lookback=7, config={})

print(f"\nResult:")
print(f"  Regime: {macro_echo.regime}")
print(f"  DXY Trend: {macro_echo.dxy_trend}")
print(f"  Yields Trend: {macro_echo.yields_trend}")
print(f"  Oil Trend: {macro_echo.oil_trend}")
print(f"  VIX Level: {macro_echo.vix_level}")
print(f"  Correlation Score: {macro_echo.correlation_score}")
