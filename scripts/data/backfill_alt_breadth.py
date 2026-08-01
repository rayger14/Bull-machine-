#!/usr/bin/env python3
"""Alt-breadth backfill — resurrecting the Moneytaur / Wyckoff-Insider
cross-market insight (dormant since birth in trap_detector.py/macro_pulse.py:
TOTAL3-vs-BTC divergence as the tell for traps vs genuine moves).

Modern implementation: hourly closes for the major alts with 2018+ coverage
(ETH, BNB, XRP) -> per-hour alt-basket return, so any BTC flush can be
classified LOCAL (alts holding = engineered stop-hunt) vs GLOBAL (alts
flushing too = systemic risk-off, the falling knife).

PRE-REGISTERED SPLIT (defined before results, 2026-08-01):
  alt_4h = equal-weight mean of ETH/BNB/XRP 4-hour returns at entry bar
  LOCAL flush  = alt_4h > -1.0%   (market holding while BTC flushes)
  GLOBAL flush = alt_4h <= -1.0%  (everything bleeding together)
  Bar: LOCAL-class PF >= GLOBAL-class PF in BOTH train and holdout on the
  clean wick_trap populations; n>=30/cell for claims.
"""
from __future__ import annotations
import io, sys, tempfile, time, urllib.request, zipfile
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/rebuild/alt_breadth"
URL = "https://data.binance.vision/data/spot/monthly/klines/{sym}/1h/{sym}-1h-{ym}.zip"
SYMS = ["ETHUSDT", "BNBUSDT", "XRPUSDT"]

def months(start="2018-01", end="2026-06"):
    cur, last = pd.Period(start, "M"), pd.Period(end, "M")
    while cur <= last:
        yield str(cur); cur += 1

def fetch(sym, ym):
    with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
        with urllib.request.urlopen(URL.format(sym=sym, ym=ym), timeout=60) as r:
            tmp.write(r.read())
        tmp.flush()
        with zipfile.ZipFile(tmp.name) as zf:
            with zf.open(zf.namelist()[0]) as f:
                df = pd.read_csv(f, header=None, usecols=[0, 4], names=["ts", "close"])
    hdr = df["ts"].astype(str).str.contains("open", case=False, na=False)
    df = df[~hdr]
    ts = df["ts"].astype("int64")
    ts = ts.where(ts < 10**14, ts // 1000)
    df["ts"] = pd.to_datetime(ts, unit="ms")
    df["close"] = df["close"].astype(float)
    return df.set_index("ts")["close"]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for sym in SYMS:
        out = OUT / f"{sym}.parquet"
        if out.exists():
            continue
        parts = []
        for ym in months():
            for attempt in range(3):
                try:
                    parts.append(fetch(sym, ym)); break
                except Exception as e:
                    if "404" in str(e): break  # pre-listing months
                    time.sleep(10)
        s = pd.concat(parts).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        s.to_frame().to_parquet(out)
        print(f"{sym}: {len(s):,} hours ({s.index[0]} -> {s.index[-1]})", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
