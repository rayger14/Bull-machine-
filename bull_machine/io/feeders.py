import os
import pandas as pd
import logging
from typing import List
from ..core.types import Series, Bar

def load_csv_to_series(file_path: str, symbol: str = "UNKNOWN", timeframe: str = "1h") -> Series:
    """Load CSV and convert to Series of Bars. Required columns: open,high,low,close."""
    try:
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")
        file_size = os.path.getsize(file_path)
        if file_size > 100 * 1024 * 1024:
            raise ValueError(f"File too large: {file_size} bytes (limit 100MB)")
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower().str.strip()
        required_cols = ['open','high','low','close']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if len(df) < 10:
            raise ValueError(f"Insufficient data: only {len(df)} rows")
        # basic data quality checks
        invalid = df[(df['high'] < df['low']) | (df['high'] < df['open']) | (df['high'] < df['close']) |
                     (df['low'] > df['open']) | (df['low'] > df['close'])]
        if len(invalid) > 0:
            logging.warning(f"Dropping {len(invalid)} invalid OHLC rows")
            df = df.drop(invalid.index)
        # timestamp handling
        ts_col = next((c for c in ['timestamp','datetime','date','time'] if c in df.columns), None)
        if ts_col:
            df[ts_col] = pd.to_datetime(df[ts_col])
            df['ts'] = (df[ts_col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        else:
            df['ts'] = range(len(df))
        for c in required_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        if 'volume' not in df.columns:
            df['volume'] = 0.0
        else:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0.0)
        df = df.dropna(subset=required_cols)
        bars = [Bar(ts=int(r['ts']), open=float(r['open']), high=float(r['high']), low=float(r['low']), close=float(r['close']), volume=float(r['volume'])) for _, r in df.iterrows()]
        logging.info(f"Loaded {len(bars)} bars for {symbol}")
        return Series(bars=bars, timeframe=timeframe, symbol=symbol)
    except Exception as e:
        raise Exception(f"Failed to load CSV data: {e}")
