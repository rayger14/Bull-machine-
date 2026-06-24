#!/usr/bin/env python3
"""Download BTCUSDT 1h klines from data.binance.vision (geo-unblocked CDN).

Monthly zips for full months, daily zips for the current partial month.
Output: data/cache/binance_vision/klines/BTCUSDT_1h.parquet
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "data/cache/binance_vision/klines"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://data.binance.vision/data/spot"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "n_trades", "taker_base", "taker_quote", "ignore"]


def fetch_zip(url: str) -> pd.DataFrame | None:
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = z.namelist()[0]
        df = pd.read_csv(z.open(name), header=None, names=COLS)
    # Binance switched open_time to microseconds in 2025 archives
    unit = "us" if df["open_time"].iloc[0] > 1e14 else "ms"
    df["timestamp"] = pd.to_datetime(df["open_time"], unit=unit, utc=True)
    return df.set_index("timestamp")[["open", "high", "low", "close", "volume"]].astype(float)


def main():
    start = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp("2024-05-01")
    end = pd.Timestamp.utcnow().tz_localize(None)

    frames = []
    # Monthly archives
    months = pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")
    for m in months:
        url = f"{BASE}/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-{m}.zip"
        df = fetch_zip(url)
        if df is not None:
            frames.append(df)
            print(f"  monthly {m}: {len(df)} bars")
        else:
            # Partial month → daily archives
            days = pd.date_range(m.start_time, min(m.end_time, end), freq="D")
            got = 0
            for d in days:
                ds = d.strftime("%Y-%m-%d")
                ddf = fetch_zip(f"{BASE}/daily/klines/BTCUSDT/1h/BTCUSDT-1h-{ds}.zip")
                if ddf is not None:
                    frames.append(ddf)
                    got += 1
            print(f"  daily fallback {m}: {got} days")

    full = pd.concat(frames).sort_index()
    full = full[~full.index.duplicated(keep="last")]
    out = OUT_DIR / "BTCUSDT_1h.parquet"
    full.to_parquet(out)
    print(f"\nSaved {len(full)} bars {full.index[0]} → {full.index[-1]} to {out}")


if __name__ == "__main__":
    main()
