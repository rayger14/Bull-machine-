#!/usr/bin/env python3
"""
Create mock macro CSV files for testing extended macro context
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def create_mock_vix(start_date: str, end_date: str) -> pd.DataFrame:
    """Create mock VIX data with realistic volatility patterns"""
    dates = pd.date_range(start_date, end_date, freq='1H')

    # Base VIX around 18-22 with occasional spikes
    base_vix = 19.0
    vix_values = []

    for i, date in enumerate(dates):
        # Add some cyclical volatility
        cycle = np.sin(i / 24) * 2  # Daily cycle
        weekly_cycle = np.sin(i / (24 * 7)) * 3  # Weekly cycle

        # Random spikes (5% chance of spike above 25)
        if np.random.random() < 0.05:
            spike = np.random.uniform(5, 15)  # VIX spike
        else:
            spike = 0

        noise = np.random.normal(0, 1)

        value = max(10, base_vix + cycle + weekly_cycle + spike + noise)
        vix_values.append(value)

    return pd.DataFrame({
        'time': [int(d.timestamp()) for d in dates],
        'value': vix_values
    })

def create_mock_gold(start_date: str, end_date: str) -> pd.DataFrame:
    """Create mock GOLD data with flight-to-safety patterns"""
    dates = pd.date_range(start_date, end_date, freq='1H')

    base_gold = 2500.0
    gold_values = []

    for i, date in enumerate(dates):
        # Gradual uptrend with volatility
        trend = i * 0.01  # Slow uptrend
        volatility = np.random.normal(0, 10)

        # Flight-to-safety spikes (correlated with VIX spikes)
        if np.random.random() < 0.03:
            safety_premium = np.random.uniform(20, 100)
        else:
            safety_premium = 0

        value = max(2000, base_gold + trend + volatility + safety_premium)
        gold_values.append(value)

    return pd.DataFrame({
        'time': [int(d.timestamp()) for d in dates],
        'value': gold_values
    })

def create_mock_move(start_date: str, end_date: str) -> pd.DataFrame:
    """Create mock MOVE (bond volatility) data"""
    dates = pd.date_range(start_date, end_date, freq='1H')

    base_move = 75.0
    move_values = []

    for i, date in enumerate(dates):
        # Lower volatility normally, spikes during stress
        noise = np.random.normal(0, 3)

        # Credit stress spikes (3% chance)
        if np.random.random() < 0.03:
            stress_spike = np.random.uniform(10, 30)
        else:
            stress_spike = 0

        value = max(50, base_move + noise + stress_spike)
        move_values.append(value)

    return pd.DataFrame({
        'time': [int(d.timestamp()) for d in dates],
        'value': move_values
    })

def create_mock_usdc_d(start_date: str, end_date: str) -> pd.DataFrame:
    """Create mock USDC.D data"""
    dates = pd.date_range(start_date, end_date, freq='1H')

    base_usdc_d = 0.025  # 2.5% dominance
    values = []

    for i, date in enumerate(dates):
        # Cyclical with risk-off spikes
        cycle = np.sin(i / (24 * 7)) * 0.005  # Weekly cycle
        noise = np.random.normal(0, 0.002)

        # Risk-off spike (alt bleed)
        if np.random.random() < 0.02:
            bleed_spike = np.random.uniform(0.01, 0.03)
        else:
            bleed_spike = 0

        value = max(0.01, base_usdc_d + cycle + noise + bleed_spike)
        values.append(value)

    return pd.DataFrame({
        'time': [int(d.timestamp()) for d in dates],
        'value': values
    })

def create_mock_btc_d(start_date: str, end_date: str) -> pd.DataFrame:
    """Create mock BTC.D data"""
    dates = pd.date_range(start_date, end_date, freq='1H')

    base_btc_d = 0.58  # 58% dominance
    values = []

    for i, date in enumerate(dates):
        # Trending with alt season cycles
        trend = np.sin(i / (24 * 30)) * 0.05  # Monthly cycle
        noise = np.random.normal(0, 0.01)

        value = max(0.45, min(0.70, base_btc_d + trend + noise))
        values.append(value)

    return pd.DataFrame({
        'time': [int(d.timestamp()) for d in dates],
        'value': values
    })

def create_mock_eurusd(start_date: str, end_date: str) -> pd.DataFrame:
    """Create mock EUR/USD data"""
    dates = pd.date_range(start_date, end_date, freq='1H')

    base_eurusd = 1.10
    values = []

    for i, date in enumerate(dates):
        # FX volatility
        noise = np.random.normal(0, 0.005)
        trend = np.sin(i / (24 * 14)) * 0.02  # Bi-weekly cycle

        value = max(1.05, min(1.15, base_eurusd + trend + noise))
        values.append(value)

    return pd.DataFrame({
        'time': [int(d.timestamp()) for d in dates],
        'value': values
    })

def create_mock_funding(start_date: str, end_date: str) -> pd.DataFrame:
    """Create mock funding rates data"""
    dates = pd.date_range(start_date, end_date, freq='1H')

    base_funding = 0.008  # 0.8% annualized
    values = []

    for i, date in enumerate(dates):
        noise = np.random.normal(0, 0.002)

        # Leverage stress spikes
        if np.random.random() < 0.02:
            stress = np.random.uniform(0.005, 0.015)
        else:
            stress = 0

        value = max(-0.01, min(0.03, base_funding + noise + stress))
        values.append(value)

    return pd.DataFrame({
        'time': [int(d.timestamp()) for d in dates],
        'value': values
    })

def create_mock_oi(start_date: str, end_date: str) -> pd.DataFrame:
    """Create mock Open Interest premium data"""
    dates = pd.date_range(start_date, end_date, freq='1H')

    base_oi = 0.012  # 1.2% premium
    values = []

    for i, date in enumerate(dates):
        noise = np.random.normal(0, 0.003)

        # Crisis spikes (OI premium surge)
        if np.random.random() < 0.01:
            crisis = np.random.uniform(0.01, 0.02)
        else:
            crisis = 0

        value = max(0.005, min(0.04, base_oi + noise + crisis))
        values.append(value)

    return pd.DataFrame({
        'time': [int(d.timestamp()) for d in dates],
        'value': values
    })

def main():
    print("🔧 Creating Mock Macro Data for Bull Machine v1.7.3")
    print("=" * 55)

    # Date range for mock data (align with crypto data)
    start_date = "2025-05-01"
    end_date = "2025-09-30"

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Create mock series
    mock_series = {
        'VIX': create_mock_vix,
        'GOLD': create_mock_gold,
        'MOVE': create_mock_move,
        'USDC.D': create_mock_usdc_d,
        'BTC.D': create_mock_btc_d,
        'EURUSD': create_mock_eurusd,
        'FUNDING': create_mock_funding,
        'OI': create_mock_oi
    }

    for symbol, create_func in mock_series.items():
        print(f"Creating {symbol}...")
        df = create_func(start_date, end_date)

        # Add standard columns for compatibility
        df['open'] = df['value']
        df['high'] = df['value'] * 1.001
        df['low'] = df['value'] * 0.999
        df['close'] = df['value']
        df['volume'] = 1000000  # Mock volume

        # Save to data directory
        output_file = data_dir / f"{symbol}_1D.csv"
        df.to_csv(output_file, index=False)
        print(f"   ✓ Saved {len(df)} bars to {output_file}")

    print(f"\n✅ Mock macro data created successfully!")
    print(f"📁 Files saved to {data_dir}/")
    print(f"📊 Date range: {start_date} to {end_date}")
    print(f"\n🧪 Ready for macro engine testing!")

if __name__ == "__main__":
    main()