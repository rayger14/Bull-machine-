#!/usr/bin/env python3
import argparse, time, requests
from datetime import datetime, timezone
from dateutil import parser as dtp
import pandas as pd
from tqdm import tqdm

BASE = "https://api.binance.com"  # public REST (spot)
ENDPT = "/api/v3/klines"          # 1000 bars per call (max)

INTERVAL_MAP = {
    "1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m",
    "1h":"1h","2h":"2h","4h":"4h","6h":"6h","8h":"8h","12h":"12h",
    "1d":"1d"
}

def to_ms(dt_str):
    # parse ISO-like or yyyy-mm-dd into UTC ms
    dt = dtp.parse(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp()*1000)

def fetch(symbol, interval, start_ms, end_ms, rate_sleep=0.25):
    url = BASE + ENDPT
    params = dict(symbol=symbol, interval=interval, limit=1000, startTime=start_ms, endTime=end_ms)
    all_rows = []
    last_open = None
    with tqdm(total=0, unit="bars", desc=f"{symbol} {interval}") as pbar:
        while True:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            chunk = r.json()
            if not chunk:
                break
            # Avoid infinite loops if API echoes same bar
            if last_open is not None and chunk[0][0] == last_open:
                break
            last_open = chunk[0][0]
            all_rows.extend(chunk)
            pbar.total += len(chunk); pbar.update(0)
            # next page
            next_start = chunk[-1][0] + 1
            if next_start >= end_ms:
                break
            params["startTime"] = next_start
            time.sleep(rate_sleep)
    # Convert to dataframe
    cols = ["open_time","open","high","low","close","volume",
            "close_time","qav","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(all_rows, columns=cols)
    if df.empty:
        return df
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    num_cols = ["open","high","low","close","volume","qav","taker_base","taker_quote"]
    df[num_cols] = df[num_cols].astype(float)
    df["trades"] = df["trades"].astype(int)
    df = df[["open_time","open","high","low","close","volume","trades","close_time"]].rename(
        columns={"open_time":"timestamp"}
    ).set_index("timestamp").sort_index()
    return df

def main():
    ap = argparse.ArgumentParser(description="Download Binance klines (public, no key).")
    ap.add_argument("--symbol", required=True, help="e.g., BTCUSDT, ETHUSDT")
    ap.add_argument("--interval", default="1h", choices=INTERVAL_MAP.keys())
    ap.add_argument("--start", required=True, help="e.g., 2024-01-01 or 2024-01-01T00:00:00Z")
    ap.add_argument("--end", required=True, help="e.g., 2025-10-13")
    ap.add_argument("--out", required=True, help="CSV path")
    args = ap.parse_args()

    start_ms = to_ms(args.start)
    end_ms = to_ms(args.end)
    df = fetch(args.symbol, INTERVAL_MAP[args.interval], start_ms, end_ms)
    if df.empty:
        print("No data returned.")
        return
    df.to_csv(args.out)
    # optional parquet for speed
    df.to_parquet(args.out.replace(".csv",".parquet"))
    print(f"Saved {len(df):,} rows → {args.out}")

if __name__ == "__main__":
    main()
