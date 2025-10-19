#!/usr/bin/env python3
import argparse, pandas as pd, yfinance as yf

TICKERS = {
    "VIX":"^VIX", "DXY":"DX-Y.NYB", "MOVE":"^MOVE", "US10Y":"^TNX",
    "GOLD":"GC=F", "OIL":"CL=F", "SPY":"SPY", "QQQ":"QQQ"
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--outdir", default="data/raw/macro")
    args = ap.parse_args()

    for name, tic in TICKERS.items():
        df = yf.download(tic, start=args.start, end=args.end, interval="1d", auto_adjust=False, progress=False)
        if df.empty:
            print(f"Empty: {name} ({tic})"); continue
        df = df.rename(columns=str.lower)[["open","high","low","close","volume"]]
        df.index = pd.to_datetime(df.index, utc=True)
        path = f"{args.outdir}/{name}_1d_{args.start}_{args.end}.csv"
        df.to_csv(path)
        print(f"Saved {name} → {path}")

if __name__ == "__main__":
    main()
