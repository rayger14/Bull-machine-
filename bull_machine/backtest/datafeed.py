import os
from typing import Dict

import pandas as pd

TF_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1H": "1h",
    "2H": "2h",
    "4H": "4h",
    "8H": "8h",
    "12H": "12h",
    "1D": "1D",
    "1W": "1W",
    "1M": "1ME",
}


class DataFeed:
    def __init__(self, sources: Dict[str, str], tz: str = "UTC"):
        self.sources = sources
        self.frames: Dict[str, pd.DataFrame] = {}
        self.tz = tz
        loaded = 0

        for sym, path in sources.items():
            # Fail-loud path checks
            if not os.path.isabs(path):
                raise ValueError(f"[DATA] Path must be absolute for {sym}: {path}")
            if not os.path.exists(path):
                raise FileNotFoundError(f"[DATA] Missing file for {sym}: {path}")

            print(f"[DATA] Loading {sym} from {path}")
            df = pd.read_csv(path)
            initial_rows = len(df)

            # Normalize columns
            df.columns = [c.lower().strip() for c in df.columns]

            # Timestamp handling with better error context
            if "timestamp" in df.columns:
                try:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                except Exception:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
                df = df.set_index("timestamp")
            elif "time" in df.columns:
                try:
                    df["time"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
                except Exception:
                    df["time"] = pd.to_datetime(df["time"])
                df = df.set_index("time")
            else:
                print(f"[DATA] WARNING: {sym} has no timestamp/time column, using synthetic dates")
                df.index = pd.date_range("2018-01-01", periods=len(df), freq="1h")

            # Required columns check
            for col in ["open", "high", "low", "close"]:
                if col not in df.columns:
                    raise ValueError(f"[DATA] {sym}: missing required column '{col}' in {path}")

            if "volume" not in df.columns:
                df["volume"] = 0.0

            df = df[["open", "high", "low", "close", "volume"]].sort_index()

            # Validate non-empty
            if len(df) == 0:
                raise ValueError(f"[DATA] {sym}: loaded 0 rows from {path}")

            df = df.tz_localize(self.tz, nonexistent="shift_forward", ambiguous="NaT")
            self.frames[sym] = df
            loaded += 1

            # Log summary
            print(
                f"[DATA] {sym}: loaded {len(df)}/{initial_rows} rows, "
                f"first={df.index[0].strftime('%Y-%m-%d %H:%M')}, "
                f"last={df.index[-1].strftime('%Y-%m-%d %H:%M')}"
            )

        if loaded == 0:
            raise RuntimeError("[DATA] Loaded 0 symbols. Check config.data.sources paths and schema.")

        print(f"[DATA] Successfully loaded {loaded} symbol(s): {list(self.frames.keys())}")

    def resample(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        freq = TF_MAP.get(tf, "1H")
        ohlc = (
            df.resample(freq, label="right", closed="right")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )
        return ohlc

    def window(self, symbol: str, tf: str, end_idx: int, lookback: int = 250) -> pd.DataFrame:
        base = self.frames[symbol]
        df_tf = self.resample(base, tf)
        if end_idx <= 0:
            end_idx = len(df_tf)
        start = max(0, end_idx - lookback)
        return df_tf.iloc[start:end_idx]
