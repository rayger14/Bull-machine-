#!/usr/bin/env python3
"""Real dominance/rotation backfill — Wyckoff Insider's stablecoin-rotation
dimension from ACTUAL market-cap histories (CoinGecko daily, 2018+).

Derives (daily):
  usdt_d_real   = USDT mcap / total-approx
  stables_d     = (USDT+USDC) mcap / total-approx
  total3_approx = total-approx - BTC - ETH mcap
  btc_d_real    = BTC / total-approx
where total-approx = sum of top-coin mcaps fetched (stated approximation).

PRE-REGISTERED test (before results): on wick_trap clean populations,
within the GLOBAL-flush class, split by stables_d RISING over the prior
3 days (capital hiding = drain confirmed) vs falling/flat. Wyckoff-Insider
prediction: GLOBAL + stables-rotation-rising = the true exodus (worst);
GLOBAL without rotation = less reliable drain signal.
Output: results/rebuild/dominance_daily.parquet
"""
from __future__ import annotations
import json, time, urllib.request
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/rebuild/dominance_daily.parquet"
COINS = ["bitcoin", "ethereum", "tether", "usd-coin", "binancecoin", "ripple",
         "solana", "cardano", "dogecoin", "tron", "litecoin", "polkadot",
         "chainlink", "bitcoin-cash", "stellar"]

def mcap_series(coin: str) -> pd.Series:
    url = (f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
           f"?vs_currency=usd&days=max&interval=daily")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    pts = data.get("market_caps") or []
    s = pd.Series({pd.Timestamp(int(t), unit="ms").normalize(): v for t, v in pts})
    return s[~s.index.duplicated(keep="last")].sort_index()

def main():
    cache = REPO / "results/rebuild/dominance_cache"
    cache.mkdir(parents=True, exist_ok=True)
    series = {}
    for c in COINS:
        cf = cache / f"{c}.parquet"
        if cf.exists():
            series[c] = pd.read_parquet(cf)["mcap"]
            continue
        for attempt in range(6):
            try:
                series[c] = mcap_series(c)
                series[c].rename("mcap").to_frame().to_parquet(cf)
                print(f"{c}: {len(series[c])} days "
                      f"({series[c].index[0].date()} -> {series[c].index[-1].date()})", flush=True)
                break
            except Exception as e:
                print(f"{c} attempt {attempt+1}: {e}", flush=True)
                time.sleep(45)
        time.sleep(10)  # free-tier rate limit
    missing = [c for c in ("bitcoin", "ethereum", "tether") if c not in series]
    if missing:
        raise SystemExit(f"required coins missing after retries: {missing}")
    df = pd.DataFrame(series)
    total = df.sum(axis=1)
    out = pd.DataFrame({
        "total_approx": total,
        "usdt_d_real": df.get("tether") / total,
        "stables_d": (df.get("tether").fillna(0) + (df["usd-coin"].fillna(0) if "usd-coin" in df else 0.0)) / total,
        "btc_d_real": df.get("bitcoin") / total,
        "total3_approx": total - df.get("bitcoin").fillna(0) - df.get("ethereum").fillna(0),
    })
    out.to_parquet(OUT)
    print(f"WROTE {OUT}: {len(out)} days", flush=True)

if __name__ == "__main__":
    main()
