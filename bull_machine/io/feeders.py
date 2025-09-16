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
        # timestamp handling with unit autodetection for numeric epoch values
        ts_col = next((c for c in ['timestamp','datetime','date','time'] if c in df.columns), None)
        if ts_col:
            # Try numeric epoch detection first
            if pd.api.types.is_numeric_dtype(df[ts_col]):
                sample = int(df[ts_col].dropna().iloc[-1]) if len(df[ts_col].dropna()) > 0 else 0
                # Heuristic thresholds (common scales)
                # ns: >= 1e18, us: >=1e15, ms: >=1e12, s: >=1e9
                unit = None
                if abs(sample) >= 1e18:
                    unit = 'ns'
                elif abs(sample) >= 1e15:
                    unit = 'us'
                elif abs(sample) >= 1e12:
                    unit = 'ms'
                elif abs(sample) >= 1e9:
                    unit = 's'
                # fallback: if unit still None but values look small (<1e6) treat as seconds indices
                try:
                    if unit:
                        df[ts_col] = pd.to_datetime(df[ts_col], unit=unit, utc=True)
                    else:
                        # maybe tiny ints (like 1,2,3) — treat as seconds since epoch if > 1000, else as index
                        if sample > 1000:
                            df[ts_col] = pd.to_datetime(df[ts_col], unit='s', utc=True)
                        else:
                            # not an epoch — fallback to sequential index timestamps
                            df['ts'] = range(len(df))
                    if 'ts' not in df.columns:
                        df['ts'] = (df[ts_col] - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta('1s')
                except Exception:
                    # fallback to string parse
                    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors='coerce')
                    df['ts'] = (df[ts_col] - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta('1s')
            else:
                # string-like timestamps
                df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors='coerce')
                df['ts'] = (df[ts_col] - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta('1s')
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
