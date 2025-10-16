#!/usr/bin/env python3
import argparse, time, requests
from datetime import datetime, timezone
from dateutil import parser as dtp
import pandas as pd
from tqdm import tqdm

BASE = "https://api.bybit.com"
ENDPT = "/v5/market/kline"   # public
# category: "linear" (USDT perps), "inverse", "spot"
# interval: "60" for 1h (Bybit uses minutes as string)

def to_ms(dt_str):
    dt = dtp.parse(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp()*1000)

def fetch(symbol, start_ms, end_ms, category="linear", interval="60", limit=1000, rate_sleep=0.2):
    url = BASE + ENDPT
    all_rows = []
    cursor = None
    with tqdm(total=0, unit="bars", desc=f"{symbol} {interval}m") as pbar:
        while True:
            params = {
                "category": category,
                "symbol": symbol,
                "interval": interval,
                "start": start_ms,
                "end": end_ms,
                "limit": limit
            }
            if cursor:
                params["cursor"] = cursor
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            if data.get("retCode") != 0:
                raise RuntimeError(f"Bybit error: {data}")
            result = data["result"]
            klines = result.get("list", [])
            if not klines:
                break
            # Each kline: [start, open, high, low, close, volume, turnover]
            # Bybit returns newest-first; reverse to chronological
            klines = list(reversed(klines))
            all_rows.extend(klines)
            pbar.total += len(klines); pbar.update(0)
            cursor = result.get("nextPageCursor")
            if not cursor:
                break
            time.sleep(rate_sleep)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows, columns=["start","open","high","low","close","volume","turnover"])
    df["start"] = pd.to_datetime(df["start"].astype(int), unit="ms", utc=True)
    num_cols = ["open","high","low","close","volume","turnover"]
    df[num_cols] = df[num_cols].astype(float)
    df = df.rename(columns={"start":"timestamp"}).set_index("timestamp").sort_index()
    return df

def main():
    ap = argparse.ArgumentParser(description="Download Bybit v5 klines (public, no key).")
    ap.add_argument("--symbol", required=True, help="e.g., BTCUSDT, ETHUSDT")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--category", default="linear", choices=["linear","inverse","spot"])
    ap.add_argument("--interval", default="60", help="minutes: 60=1h, 240=4h, 1440=1d")
    args = ap.parse_args()

    df = fetch(args.symbol, to_ms(args.start), to_ms(args.end), category=args.category, interval=args.interval)
    if df.empty:
        print("No data returned.")
        return
    df.to_csv(args.out)
    df.to_parquet(args.out.replace(".csv",".parquet"))
    print(f"Saved {len(df):,} rows → {args.out}")

if __name__ == "__main__":
    main()
