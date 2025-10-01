#!/usr/bin/env python3
"""
Live Data Adapters for Bull Machine v1.7.3
Handles CSV replay streaming, OHLCV updates, and MTF alignment with right-edge enforcement
"""

import sys
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple

import pandas as pd

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from bull_machine_config import DATA_DIR


class LiveDataAdapter:
    """Production-grade data adapter for live feeds with right-edge enforcement."""

    def __init__(self):
        self.timeframe_mapping = {
            '1H': '60_',      # 60 minutes = 1H
            '4H': '240_',     # 240 minutes = 4H
            '1D': '1D_',      # Daily
        }
        self.drift_tolerance_minutes = 5  # Max allowed time drift

    def find_data_file(self, asset: str, timeframe: str) -> Optional[Path]:
        """Find data file using production path resolver."""
        pattern = self.timeframe_mapping.get(timeframe, timeframe + '_')
        symbol = f"COINBASE_{asset}USD"

        # Use production data directory
        data_files = list(DATA_DIR.glob("*.csv"))

        for file_path in data_files:
            if symbol in file_path.name and pattern in file_path.name:
                return file_path

        return None

    def stream_csv(self, asset: str, timeframe: str, start_date: Optional[str] = None,
                   end_date: Optional[str] = None, speed: float = 1.0) -> Generator[Dict, None, None]:
        """
        Stream CSV data as OHLCV ticks in timestamp order.

        Args:
            asset: ETH, BTC, SOL
            timeframe: 1H, 4H, 1D
            start_date: YYYY-MM-DD format
            end_date: YYYY-MM-DD format
            speed: Replay speed multiplier (1.0 = normal)

        Yields:
            Dict with timestamp, Open, High, Low, Close, Volume
        """
        file_path = self.find_data_file(asset, timeframe)
        if not file_path:
            raise FileNotFoundError(f"No data file found for {asset} {timeframe}")

        # Load and prepare data
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('datetime').sort_index()

        # Rename to standard OHLCV
        df = df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'
        })

        # Add volume if available
        if 'BUY+SELL V' in df.columns:
            df['Volume'] = pd.to_numeric(df['BUY+SELL V'], errors='coerce')
        else:
            df['Volume'] = 0

        # Filter date range
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        # Stream ticks
        for timestamp, row in df.iterrows():
            yield {
                'timestamp': timestamp,
                'Open': float(row['Open']),
                'High': float(row['High']),
                'Low': float(row['Low']),
                'Close': float(row['Close']),
                'Volume': float(row.get('Volume', 0))
            }

    def update_ohlcv(self, df: pd.DataFrame, tick: Dict, max_bars: int = 1000) -> pd.DataFrame:
        """
        Safely append new tick to DataFrame with right-edge enforcement.

        Args:
            df: Existing OHLCV DataFrame
            tick: New tick dict
            max_bars: Maximum bars to keep in memory

        Returns:
            Updated DataFrame with no future leak
        """
        # Create new row
        new_row = pd.DataFrame({
            'Open': [tick['Open']],
            'High': [tick['High']],
            'Low': [tick['Low']],
            'Close': [tick['Close']],
            'Volume': [tick['Volume']]
        }, index=[tick['timestamp']])

        # Right-edge enforcement: ensure no future data
        if len(df) > 0:
            last_time = df.index[-1]
            new_time = tick['timestamp']

            # Only allow forward progression or small drift
            time_diff = (new_time - last_time).total_seconds() / 60  # minutes
            if time_diff < -self.drift_tolerance_minutes:
                print(f"⚠️  Time regression detected: {time_diff:.1f}m, skipping tick")
                return df

        # Append new data
        updated_df = pd.concat([df, new_row])

        # Keep only recent bars
        if len(updated_df) > max_bars:
            updated_df = updated_df.tail(max_bars)

        return updated_df

    def align_mtf(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame,
                  df_1d: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Right-edge align multi-timeframe data with tolerance checking.

        Args:
            df_1h: 1H timeframe data
            df_4h: 4H timeframe data
            df_1d: 1D timeframe data

        Returns:
            Tuple of aligned DataFrames
        """
        if len(df_1h) == 0:
            return df_1h, df_4h, df_1d

        # Get latest 1H timestamp as reference
        latest_1h = df_1h.index[-1]

        # Align 4H data (allow up to 4H + tolerance)
        if len(df_4h) > 0:
            latest_4h = df_4h.index[-1]
            h4_diff_hours = (latest_1h - latest_4h).total_seconds() / 3600

            if h4_diff_hours > 4.1:  # 4H + 6min tolerance
                print(f"⚠️  4H data lagging by {h4_diff_hours:.1f}h")

        # Align 1D data (allow up to 24H + tolerance)
        if len(df_1d) > 0:
            latest_1d = df_1d.index[-1]
            d1_diff_hours = (latest_1h - latest_1d).total_seconds() / 3600

            if d1_diff_hours > 24.1:  # 24H + 6min tolerance
                print(f"⚠️  1D data lagging by {d1_diff_hours:.1f}h")

        return df_1h, df_4h, df_1d

    def create_websocket_adapter(self) -> 'WebSocketAdapter':
        """Create placeholder WebSocket adapter for shadow mode."""
        return WebSocketAdapter()


class WebSocketAdapter:
    """Placeholder WebSocket adapter for v1.7.3 (no real network)."""

    def __init__(self):
        self.connected = False

    def connect(self, symbol: str) -> bool:
        """Placeholder connection."""
        print(f"📡 WebSocket adapter (placeholder) for {symbol}")
        self.connected = True
        return True

    def get_tick(self) -> Optional[Dict]:
        """Placeholder tick generation."""
        # In real implementation, this would parse WebSocket events
        return None

    def disconnect(self):
        """Placeholder disconnect."""
        self.connected = False


if __name__ == "__main__":
    # Test adapter
    adapter = LiveDataAdapter()

    # Test ETH 1H streaming
    try:
        stream = adapter.stream_csv("ETH", "1H", "2025-06-01", "2025-06-02")
        for i, tick in enumerate(stream):
            print(f"Tick {i}: {tick['timestamp']} Close: {tick['Close']}")
            if i >= 5:  # Only show first few ticks
                break
    except Exception as e:
        print(f"Test failed: {e}")
