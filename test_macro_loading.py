#!/usr/bin/env python3
"""
Test macro data loading and analyze_macro_echo integration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from engine.context.loader import load_macro_data
from engine.exits.macro_echo import analyze_macro_echo

def test_macro_loading():
    """Test if macro data loads correctly and has varying values."""

    print("=" * 80)
    print("Macro Data Loading Test")
    print("=" * 80)

    # Load macro data (crypto asset type)
    macro_data = load_macro_data(asset_type='crypto')

    print(f"\nLoaded {len(macro_data)} macro series")

    # Check key series
    for symbol in ['DXY', 'VIX', 'US10Y', 'WTI']:
        if symbol in macro_data and not macro_data[symbol].empty:
            df = macro_data[symbol]
            print(f"\n{symbol}:")
            print(f"  Rows: {len(df)}")
            print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"  Value range: [{df['value'].min():.2f}, {df['value'].max():.2f}]")
            print(f"  Sample values (last 5): {df['value'].tail(5).tolist()}")
        else:
            print(f"\n{symbol}: NOT FOUND or EMPTY")

    # Test analyze_macro_echo with Q3 2024 data
    print("\n" + "=" * 80)
    print("Testing analyze_macro_echo() with Q3 2024 window")
    print("=" * 80)

    test_date = pd.Timestamp('2024-08-15', tz='UTC')
    lookback_start = test_date - pd.Timedelta(days=7)

    def extract_series(symbol: str) -> pd.Series:
        """Extract 7-day series ending at test_date."""
        if symbol not in macro_data or macro_data[symbol].empty:
            defaults = {'DXY': 100.0, 'US10Y': 4.0, 'WTI': 75.0, 'VIX': 18.0}
            return pd.Series([defaults.get(symbol, 50.0)])

        df = macro_data[symbol]
        window = df[(df['timestamp'] >= lookback_start) & (df['timestamp'] <= test_date)]

        if window.empty:
            recent = df[df['timestamp'] <= test_date]
            if not recent.empty:
                return pd.Series([recent.iloc[-1]['value']])
            defaults = {'DXY': 100.0, 'US10Y': 4.0, 'WTI': 75.0, 'VIX': 18.0}
            return pd.Series([defaults.get(symbol, 50.0)])

        return window['value'].reset_index(drop=True)

    # Extract series
    dxy_series = extract_series('DXY')
    yields_series = extract_series('US10Y')
    oil_series = extract_series('WTI')
    vix_series = extract_series('VIX')

    print(f"\nExtracted series for {test_date.date()}:")
    print(f"  DXY: {len(dxy_series)} values, range [{dxy_series.min():.2f}, {dxy_series.max():.2f}]")
    print(f"  US10Y: {len(yields_series)} values, range [{yields_series.min():.2f}, {yields_series.max():.2f}]")
    print(f"  WTI: {len(oil_series)} values, range [{oil_series.min():.2f}, {oil_series.max():.2f}]")
    print(f"  VIX: {len(vix_series)} values, range [{vix_series.min():.2f}, {vix_series.max():.2f}]")

    # Call analyze_macro_echo
    macro_echo = analyze_macro_echo({
        'DXY': dxy_series,
        'YIELDS_10Y': yields_series,
        'OIL': oil_series,
        'VIX': vix_series
    }, lookback=7, config={})

    print(f"\n{'=' * 80}")
    print(f"Macro Echo Result for {test_date.date()}")
    print(f"{'=' * 80}")
    print(f"  Regime: {macro_echo.regime}")
    print(f"  DXY Trend: {macro_echo.dxy_trend}")
    print(f"  Yields Trend: {macro_echo.yields_trend}")
    print(f"  Oil Trend: {macro_echo.oil_trend}")
    print(f"  VIX Level: {macro_echo.vix_level}")
    print(f"  Correlation Score: {macro_echo.correlation_score:.3f}")
    print(f"  Exit Recommended: {macro_echo.exit_recommended}")

    # Test with multiple dates to see variation
    print("\n" + "=" * 80)
    print("Testing variation across Q3 2024")
    print("=" * 80)

    test_dates = pd.date_range('2024-07-01', '2024-09-30', freq='7D', tz='UTC')
    regimes = []
    trends = []
    scores = []

    for date in test_dates:
        lookback_start = date - pd.Timedelta(days=7)

        def extract_for_date(symbol: str) -> pd.Series:
            if symbol not in macro_data or macro_data[symbol].empty:
                defaults = {'DXY': 100.0, 'US10Y': 4.0, 'WTI': 75.0, 'VIX': 18.0}
                return pd.Series([defaults.get(symbol, 50.0)])

            df = macro_data[symbol]
            window = df[(df['timestamp'] >= lookback_start) & (df['timestamp'] <= date)]

            if window.empty:
                recent = df[df['timestamp'] <= date]
                if not recent.empty:
                    return pd.Series([recent.iloc[-1]['value']])
                defaults = {'DXY': 100.0, 'US10Y': 4.0, 'WTI': 75.0, 'VIX': 18.0}
                return pd.Series([defaults.get(symbol, 50.0)])

            return window['value'].reset_index(drop=True)

        result = analyze_macro_echo({
            'DXY': extract_for_date('DXY'),
            'YIELDS_10Y': extract_for_date('US10Y'),
            'OIL': extract_for_date('WTI'),
            'VIX': extract_for_date('VIX')
        }, lookback=7)

        regimes.append(result.regime)
        trends.append(result.dxy_trend)
        scores.append(result.correlation_score)

    print(f"\nResults across {len(test_dates)} weekly samples:")
    print(f"  Unique regimes: {set(regimes)}")
    print(f"  Unique DXY trends: {set(trends)}")
    print(f"  Correlation scores: min={min(scores):.3f}, max={max(scores):.3f}, mean={sum(scores)/len(scores):.3f}")

    if len(set(regimes)) == 1 and len(set(trends)) == 1:
        print("\n❌ WARNING: Macro features are CONSTANT across Q3 2024")
        print("   → All regimes:", regimes[0])
        print("   → All DXY trends:", trends[0])
    else:
        print("\n✅ Macro features are VARYING correctly")

if __name__ == '__main__':
    test_macro_loading()
