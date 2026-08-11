"""
FRESH-ASSET FETCHER v2 (STUDY ONLY) -- daily OHLCV via Coinbase Exchange public
candles (https://api.exchange.coinbase.com/products/<P>/candles?granularity=86400).

WHY COINBASE, NOT YAHOO (documented substitution): the Yahoo v8 chart JSON endpoint
returned persistent HTTP 429 (Too Many Requests) from this environment across
multi-minute exponential backoff on both query1/query2 hosts; Stooq gates behind a
JS proof-of-work challenge; Binance returns HTTP 451 (geo-block). Coinbase Exchange
public candles respond cleanly. The un-mined FRESH ASSETS are unchanged; only the
vendor differs. Coinbase is also the venue Part-3 deployment targets (INTX perps),
so its spot daily series is the most decision-relevant reference.

Caches to <scratch>/one_strategy/<SYM>.parquet in the idea_lab load schema:
columns [timestamp, open, high, low, close, volume].
Coinbase raw candle = [time, low, high, open, close, volume], newest-first, <=300/call.
"""
from __future__ import annotations
import os, json, time, random, urllib.request
import datetime as dt
import pandas as pd

# Cache dir for fetched parquets (study scratchpad). Override with ONE_STRATEGY_DATA.
SC = os.environ.get(
    "ONE_STRATEGY_DATA",
    "/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/"
    "da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad/one_strategy")
os.makedirs(SC, exist_ok=True)

# fresh crypto universe on Coinbase (BNB dropped -> not listed on Coinbase; BTC = ref)
PRODUCTS = ["ETH-USD", "SOL-USD", "LTC-USD", "XRP-USD", "ADA-USD",
            "DOGE-USD", "DOT-USD", "AVAX-USD", "LINK-USD", "BTC-USD"]

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")
DAY = 86400
START = dt.datetime(2015, 1, 1)


def _get(url, tries=6):
    for a in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            return json.loads(urllib.request.urlopen(req, timeout=30).read())
        except Exception as e:
            wait = (a + 1) * 4 + random.random() * 3
            print(f"    retry {a}: {e}; sleep {wait:.0f}s", flush=True)
            time.sleep(wait)
    return None


def fetch_product(product) -> pd.DataFrame | None:
    rows = {}
    win = 300 * DAY
    t0 = int(START.timestamp())
    t_end = int(time.time())
    cur = t0
    while cur < t_end:
        s = dt.datetime.utcfromtimestamp(cur).isoformat()
        e = dt.datetime.utcfromtimestamp(min(cur + win, t_end)).isoformat()
        url = (f"https://api.exchange.coinbase.com/products/{product}/candles"
               f"?granularity={DAY}&start={s}&end={e}")
        j = _get(url)
        if j:
            for c in j:            # [time, low, high, open, close, volume]
                rows[int(c[0])] = c
        cur += win
        time.sleep(0.4 + random.random() * 0.3)
    if not rows:
        return None
    ts = sorted(rows)
    df = pd.DataFrame({
        "timestamp": [dt.datetime.utcfromtimestamp(t) for t in ts],
        "low":  [rows[t][1] for t in ts],
        "high": [rows[t][2] for t in ts],
        "open": [rows[t][3] for t in ts],
        "close":[rows[t][4] for t in ts],
        "volume":[rows[t][5] for t in ts],
    })
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    return df.reset_index(drop=True)


def main():
    for p in PRODUCTS:
        out = os.path.join(SC, f"{p}.parquet")
        if os.path.exists(out):
            df = pd.read_parquet(out)
            print(f"cached  {p:9s} bars={len(df):5d} "
                  f"{df['timestamp'].min().date()} -> {df['timestamp'].max().date()}",
                  flush=True)
            continue
        print(f"fetching {p} ...", flush=True)
        df = fetch_product(p)
        if df is None or len(df) == 0:
            print(f"FAILED  {p}", flush=True)
            continue
        # sanity: drop duplicate/zero rows, enforce monotonic daily
        df = df[df["close"] > 0].reset_index(drop=True)
        df.to_parquet(out)
        yrs = (df["timestamp"].max() - df["timestamp"].min()).days / 365.25
        gaps = (df["timestamp"].diff().dt.days.fillna(1) > 3).sum()
        print(f"OK      {p:9s} bars={len(df):5d} "
              f"{df['timestamp'].min().date()} -> {df['timestamp'].max().date()} "
              f"({yrs:.1f} yr, {gaps} multi-day gaps)", flush=True)


if __name__ == "__main__":
    main()
