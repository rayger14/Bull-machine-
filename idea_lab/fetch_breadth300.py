"""
BREADTH-300 FETCHER (add.64; STUDY ONLY; nothing ships)
=======================================================
Fetch ADJUSTED daily OHLCV (auto_adjust=True) for ALL current S&P 500 constituents
via yfinance. auto_adjust=True removes split/dividend discontinuities (multiplicative,
scale-invariant -> door signals unchanged, causally safe). See breadth300_PREREGISTRATION.txt.

Writes B3/<TICKER>.parquet in the idea_lab load schema [ts, open, high, low, close, volume].
Batched download with per-ticker retry.
"""
from __future__ import annotations
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import yfinance as yf

SCROOT = ("/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/"
          "da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad")
B3 = os.path.join(SCROOT, "breadth300")
os.makedirs(B3, exist_ok=True)

RAW = json.load(open(os.path.join(SCROOT, "sp500_raw.json")))
SYMBOLS = RAW["symbols"]


def _normalize(df, ticker):
    if df is None or len(df) == 0:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        # single-ticker slice
        try:
            df = df.xs(ticker, axis=1, level=1)
        except Exception:
            df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df = df.reset_index().rename(columns={"date": "ts", "Date": "ts", "index": "ts"})
    df.columns = [c.lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    need = ["ts", "open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in need):
        return None
    out = df[need].copy()
    out["ts"] = pd.to_datetime(out["ts"])
    for c in ("open", "high", "low", "close", "volume"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"]).sort_values("ts")
    out = out.drop_duplicates(subset=["ts"]).reset_index(drop=True)
    if len(out) < 100:
        return None
    return out


def fetch_batch(tickers):
    """Batch download; return dict ticker->df for those that parsed."""
    got = {}
    try:
        data = yf.download(tickers, start="1990-01-01", end="2026-08-14",
                           interval="1d", auto_adjust=True, progress=False,
                           threads=True, group_by="ticker")
    except Exception as e:
        print(f"  batch err: {e}", flush=True)
        return got
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                sub = data[t].copy()
            else:
                sub = data.copy()
            out = _normalize(sub, t)
            if out is not None:
                got[t] = out
        except Exception:
            pass
    return got


def fetch_one(ticker):
    for attempt in range(3):
        try:
            df = yf.download(ticker, start="1990-01-01", end="2026-08-14",
                             interval="1d", auto_adjust=True, progress=False,
                             threads=False)
            out = _normalize(df, ticker)
            if out is not None:
                return out
        except Exception as e:
            time.sleep(2 + attempt * 2)
    return None


def save(ticker, out):
    path = os.path.join(B3, f"{ticker}.parquet")
    out.to_parquet(path)
    yrs = (out["ts"].iloc[-1] - out["ts"].iloc[0]).days / 365.25
    return yrs


def main():
    print(f"BREADTH-300 FETCH -> {B3}  ({len(SYMBOLS)} symbols, auto_adjust=True)", flush=True)
    done = set(f[:-8] for f in os.listdir(B3) if f.endswith(".parquet"))
    todo = [s for s in SYMBOLS if s not in done]
    print(f"  already cached: {len(done)}  todo: {len(todo)}", flush=True)

    BATCH = 40
    ok = len(done)
    for bi in range(0, len(todo), BATCH):
        chunk = todo[bi:bi + BATCH]
        got = fetch_batch(chunk)
        # per-ticker retry for misses
        miss = [t for t in chunk if t not in got]
        for t in miss:
            o = fetch_one(t)
            if o is not None:
                got[t] = o
        for t, out in got.items():
            yrs = save(t, out)
            ok += 1
        still = [t for t in chunk if t not in got]
        print(f"  batch {bi//BATCH+1}: got {len(got)}/{len(chunk)}  "
              f"cumulative {ok}  miss={still[:8]}", flush=True)
        time.sleep(1.0)

    print(f"\nDONE fetched/cached {ok}/{len(SYMBOLS)}", flush=True)


if __name__ == "__main__":
    main()
