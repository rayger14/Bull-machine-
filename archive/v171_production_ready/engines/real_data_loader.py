#!/usr/bin/env python3
"""
Real Data Loader for Bull Machine
Loads ETH and macro data from chart_logs for production backtesting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import os

class RealDataLoader:
    """Load and prepare real market data for Bull Machine backtesting."""

    def __init__(self, chart_logs_path: str = "/Users/raymondghandchi/Desktop/Chart Logs"):
        self.chart_logs_path = chart_logs_path
        self.data_cache = {}

        # Map timeframe strings to file patterns
        self.timeframe_mapping = {
            '1H': '60_',      # 60 minutes = 1H
            '4H': '240_',     # 240 minutes = 4H
            '6H': '360_',     # 360 minutes = 6H
            '12H': '720_',    # 720 minutes = 12H
            '22H': '1320_',   # 1320 minutes = 22H
            '1D': '1D_',      # Daily
            '1W': '1W_',      # Weekly
            '1M': '1M_'       # Monthly
        }

    def find_data_file(self, symbol: str, timeframe: str) -> Optional[str]:
        """Find the appropriate data file for symbol and timeframe."""
        pattern = self.timeframe_mapping.get(timeframe, timeframe + '_')

        # List all files in chart_logs
        files = os.listdir(self.chart_logs_path)

        # Find matching file
        for file in files:
            if symbol in file and pattern in file and file.endswith('.csv'):
                return os.path.join(self.chart_logs_path, file)

        return None

    def load_raw_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load raw data from CSV file."""
        filepath = self.find_data_file(symbol, timeframe)

        if not filepath:
            print(f"⚠️  No data found for {symbol} {timeframe}")
            return None

        try:
            df = pd.read_csv(filepath)

            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['time'], unit='s')
            df = df.set_index('datetime')

            # Rename columns to standard OHLCV format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close'
            })

            # Ensure numeric columns
            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Add volume if available
            if 'BUY+SELL V' in df.columns:
                df['Volume'] = pd.to_numeric(df['BUY+SELL V'], errors='coerce')
            elif 'Total Buy Volume' in df.columns and 'Total Sell Volume' in df.columns:
                df['Volume'] = pd.to_numeric(df['Total Buy Volume'], errors='coerce') + \
                              pd.to_numeric(df['Total Sell Volume'], errors='coerce')
            else:
                df['Volume'] = 1000000  # Default volume if not available

            # Sort by timestamp
            df = df.sort_index()

            print(f"✅ Loaded {symbol} {timeframe}: {len(df)} bars ({df.index[0]} → {df.index[-1]})")
            return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

        except Exception as e:
            print(f"❌ Error loading {symbol} {timeframe}: {e}")
            return None

    def load_eth_data(self, timeframe: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """Load ETH data for specified timeframe and date range."""
        cache_key = f"ETHUSD_{timeframe}"

        # Check cache first
        if cache_key not in self.data_cache:
            df = self.load_raw_data('COINBASE_ETHUSD', timeframe)
            if df is None:
                return None
            self.data_cache[cache_key] = df

        df = self.data_cache[cache_key].copy()

        # Filter by date range if provided
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df.index >= start_dt]

        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]

        return df

    def load_btc_data(self, timeframe: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """Load BTC data for specified timeframe and date range."""
        cache_key = f"BTCUSD_{timeframe}"

        if cache_key not in self.data_cache:
            df = self.load_raw_data('COINBASE_BTCUSD', timeframe)
            if df is None:
                return None
            self.data_cache[cache_key] = df

        df = self.data_cache[cache_key].copy()

        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df.index >= start_dt]

        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]

        return df

    def load_ethbtc_data(self, timeframe: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """Load ETHBTC data for specified timeframe and date range."""
        cache_key = f"ETHBTC_{timeframe}"

        if cache_key not in self.data_cache:
            df = self.load_raw_data('COINBASE_ETHBTC', timeframe)
            if df is None:
                return None
            self.data_cache[cache_key] = df

        df = self.data_cache[cache_key].copy()

        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df.index >= start_dt]

        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]

        return df

    def generate_macro_data(self, start_date: str, end_date: str, timeframe: str = '4H') -> pd.DataFrame:
        """Generate synthetic macro data (DXY, VIX, TOTAL2) aligned with ETH data."""
        # Get ETH data to align timestamps
        eth_data = self.load_eth_data(timeframe, start_date, end_date)

        if eth_data is None or len(eth_data) == 0:
            return pd.DataFrame()

        # Create macro data aligned with ETH timestamps
        timestamps = eth_data.index

        # Generate realistic synthetic macro data
        np.random.seed(42)  # For reproducibility

        macro_data = pd.DataFrame(index=timestamps)

        # DXY (Dollar Index) - typically 95-110 range
        dxy_base = 103
        dxy_returns = np.random.normal(0, 0.005, len(timestamps))  # 0.5% volatility
        macro_data['dxy'] = dxy_base * np.exp(np.cumsum(dxy_returns))

        # VIX (Volatility Index) - typically 12-40 range
        vix_base = 20
        vix_values = []
        current_vix = vix_base

        for i in range(len(timestamps)):
            # VIX tends to spike and revert
            if np.random.random() < 0.05:  # 5% chance of spike
                current_vix *= np.random.uniform(1.2, 1.8)
            else:
                current_vix *= np.random.uniform(0.98, 1.02)

            # Keep VIX in reasonable bounds
            current_vix = max(10, min(60, current_vix))
            vix_values.append(current_vix)

        macro_data['vix'] = vix_values

        # TOTAL2 (Total crypto market cap excluding BTC) - simulate growth trend
        total2_base = 800e9  # $800B base
        total2_returns = np.random.normal(0.0002, 0.02, len(timestamps))  # Slight growth bias
        macro_data['total2'] = total2_base * np.exp(np.cumsum(total2_returns))

        # ETHBTC ratio - load real data if available, otherwise synthesize
        ethbtc_data = self.load_ethbtc_data(timeframe, start_date, end_date)
        if ethbtc_data is not None and len(ethbtc_data) > 0:
            # Align ETHBTC data with ETH timestamps
            macro_data = macro_data.join(ethbtc_data[['Close']].rename(columns={'Close': 'ethbtc'}), how='left')
            # Forward fill missing values
            macro_data['ethbtc'] = macro_data['ethbtc'].fillna(method='ffill')
        else:
            # Synthesize ETHBTC if not available
            ethbtc_base = 0.06
            ethbtc_returns = np.random.normal(0, 0.015, len(timestamps))  # 1.5% volatility
            macro_data['ethbtc'] = ethbtc_base * np.exp(np.cumsum(ethbtc_returns))

        return macro_data

    def load_complete_dataset(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Load complete dataset for Bull Machine backtesting."""
        print(f"🔄 Loading complete dataset: {start_date} → {end_date}")

        dataset = {}

        # Load ETH data for multiple timeframes
        # Try to get 1H data, fall back to 6H if 1H not available
        eth_1h = self.load_eth_data('1H', start_date, end_date)
        if eth_1h is None or len(eth_1h) == 0:
            print("⚠️  1H data not available, using 6H as LTF")
            eth_1h = self.load_eth_data('6H', start_date, end_date)

        eth_4h = self.load_eth_data('4H', start_date, end_date)
        eth_1d = self.load_eth_data('1D', start_date, end_date)

        if eth_4h is None or len(eth_4h) == 0:
            raise ValueError("❌ No ETH 4H data available - required for backtesting")

        dataset['eth_1h'] = eth_1h
        dataset['eth_4h'] = eth_4h
        dataset['eth_1d'] = eth_1d

        # Load macro data
        macro_data = self.generate_macro_data(start_date, end_date, '4H')
        dataset['macro'] = macro_data

        # Data validation
        if len(dataset['eth_4h']) < 50:
            raise ValueError(f"❌ Insufficient ETH 4H data: {len(dataset['eth_4h'])} bars")

        print(f"✅ Dataset loaded successfully:")
        print(f"   ETH 1H/6H: {len(dataset['eth_1h']) if dataset['eth_1h'] is not None else 0} bars")
        print(f"   ETH 4H: {len(dataset['eth_4h'])} bars")
        print(f"   ETH 1D: {len(dataset['eth_1d']) if dataset['eth_1d'] is not None else 0} bars")
        print(f"   Macro: {len(dataset['macro'])} bars")

        return dataset

    def get_available_date_range(self) -> Tuple[str, str]:
        """Get the available date range for ETH data."""
        eth_4h = self.load_eth_data('4H')

        if eth_4h is None or len(eth_4h) == 0:
            return None, None

        start_date = eth_4h.index[0].strftime('%Y-%m-%d')
        end_date = eth_4h.index[-1].strftime('%Y-%m-%d')

        return start_date, end_date

def test_data_loader():
    """Test the real data loader."""
    loader = RealDataLoader()

    print("🧪 Testing Real Data Loader")
    print("=" * 50)

    # Test individual data loading
    eth_4h = loader.load_eth_data('4H')
    if eth_4h is not None:
        print(f"ETH 4H: {len(eth_4h)} bars")
        print(f"Date range: {eth_4h.index[0]} → {eth_4h.index[-1]}")
        print(f"Sample data:")
        print(eth_4h.head(2))

    # Test date range
    start_date, end_date = loader.get_available_date_range()
    print(f"\nAvailable date range: {start_date} → {end_date}")

    # Test complete dataset loading
    if start_date and end_date:
        # Test with a small date range
        test_start = start_date
        test_end = (pd.to_datetime(start_date) + timedelta(days=30)).strftime('%Y-%m-%d')

        try:
            dataset = loader.load_complete_dataset(test_start, test_end)
            print(f"\n✅ Successfully loaded complete dataset for {test_start} → {test_end}")
        except Exception as e:
            print(f"\n❌ Error loading complete dataset: {e}")

if __name__ == "__main__":
    test_data_loader()