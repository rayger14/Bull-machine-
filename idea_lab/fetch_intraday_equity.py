"""Fetch ~730d of 1H bars for a crypto-INDEPENDENT second family (equities/metals).
yfinance intraday cap = 730d of 1h. Cache to scratch/fractal_mtf_eq/<SYM>_1H.parquet."""
import os, sys, time
import pandas as pd, yfinance as yf

OUT = "/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad/fractal_mtf_eq"
os.makedirs(OUT, exist_ok=True)
# metals + single-stocks + broad-market ETFs (independent of crypto macro factor)
SYMS = {"GLD": "GLD", "SLV": "SLV", "SPY": "SPY", "QQQ": "QQQ",
        "AAPL": "AAPL", "MSFT": "MSFT", "NVDA": "NVDA", "GC=F": "GOLD"}

for tk, name in SYMS.items():
    try:
        df = yf.download(tk, period="730d", interval="1h", auto_adjust=False,
                         progress=False, threads=False)
        if df is None or len(df) == 0:
            print(f"{name}: EMPTY"); continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        df.to_parquet(f"{OUT}/{name}_1H.parquet")
        print(f"{name}: {len(df)} bars {df.index[0].date()}->{df.index[-1].date()}")
        time.sleep(1)
    except Exception as e:
        print(f"{name}: ERROR {e}")
