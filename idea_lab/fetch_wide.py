"""
WIDE-BASKET FETCHER (STUDY ONLY) -- daily OHLCV for the 16 NEW markets + SPX retry,
via yfinance (worked for GOLD/NDX in add.59). Cryptos + GOLD + NDX are already cached
(one_strategy/, xasset/) and are reused verbatim to REPRODUCE the existing 13 exactly.

Writes wide_basket/<KEY>.parquet in the idea_lab load schema: [ts, open, high, low,
close, volume]. yfinance auto_adjust=False so OHLC are raw (matches GOLD/NDX cache).
"""
from __future__ import annotations
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import yfinance as yf

WB = "/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/" \
     "da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad/wide_basket"
os.makedirs(WB, exist_ok=True)

# key -> yahoo ticker.  (KEY is the internal display symbol used by the harness.)
NEW = {
    "GOLD":     "GC=F",      # add.68: XA cache deleted in disk cleanup -> refetch
    "NDX":      "^NDX",      # add.68: XA cache deleted in disk cleanup -> refetch
    "SPX":      "^GSPC",     # retry (raw-endpoint gated before; yfinance path)
    "DJI":      "^DJI",
    "RUT":      "^RUT",
    "N225":     "^N225",
    "DAX":      "^GDAXI",
    "FTSE":     "^FTSE",
    "SILVER":   "SI=F",
    "COPPER":   "HG=F",
    "OIL":      "CL=F",
    "PLATINUM": "PL=F",
    "EURUSD":   "EURUSD=X",
    "GBPUSD":   "GBPUSD=X",
    "USDJPY":   "USDJPY=X",
    "AUDUSD":   "AUDUSD=X",
    "AAPL":     "AAPL",
    "MSFT":     "MSFT",
    "NVDA":     "NVDA",
}


def fetch_one(key, ticker):
    for attempt in range(4):
        try:
            df = yf.download(ticker, start="1990-01-01", end="2026-08-14",
                             interval="1d", auto_adjust=False, progress=False,
                             threads=False)
            if df is None or len(df) == 0:
                raise RuntimeError("empty")
            # yfinance may return MultiIndex columns (ticker level) -> flatten
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns=str.lower)
            df = df.reset_index().rename(columns={"date": "ts", "Date": "ts",
                                                  "index": "ts"})
            cols = {c.lower(): c for c in df.columns}
            df.columns = [c.lower() for c in df.columns]
            if "volume" not in df.columns:
                df["volume"] = 0.0
            out = df[["ts", "open", "high", "low", "close", "volume"]].copy()
            out["ts"] = pd.to_datetime(out["ts"])
            for c in ("open", "high", "low", "close", "volume"):
                out[c] = pd.to_numeric(out[c], errors="coerce")
            out = out.dropna(subset=["open", "high", "low", "close"]).sort_values("ts")
            out = out.drop_duplicates(subset=["ts"]).reset_index(drop=True)
            path = os.path.join(WB, f"{key}.parquet")
            out.to_parquet(path)
            yrs = (out["ts"].iloc[-1] - out["ts"].iloc[0]).days / 365.25
            print(f"  OK  {key:<9}{ticker:<10} n={len(out):>6}  "
                  f"{out['ts'].iloc[0].date()} -> {out['ts'].iloc[-1].date()}  "
                  f"{yrs:.1f}yr  zeroVol%={100*(out['volume']==0).mean():.1f}",
                  flush=True)
            return True
        except Exception as e:
            print(f"  retry {attempt} {key} ({ticker}): {e}", flush=True)
            time.sleep(3 + attempt * 3)
    print(f"  FAIL {key} ({ticker}) after retries", flush=True)
    return False


def main():
    print(f"WIDE-BASKET FETCH -> {WB}")
    ok = 0
    for key, tk in NEW.items():
        if fetch_one(key, tk):
            ok += 1
        time.sleep(1.0)
    print(f"\nfetched {ok}/{len(NEW)} new markets")


if __name__ == "__main__":
    main()
